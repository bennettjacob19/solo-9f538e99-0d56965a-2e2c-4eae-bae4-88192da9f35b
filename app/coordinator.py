"""Coordinator: FastAPI web app that orchestrates the two training workers.

Responsibilities
----------------
* Validate the uploaded CSV and deterministically split it into two
  mutually-exclusive shards, one per independent worker process.
* Drive round-by-round synchronous training: ask both workers for their shard
  gradients, combine them into the *full-data average gradient*, ask both
  workers to apply the identical momentum step and durably persist it, and
  only then atomically confirm the round locally.
* Persist all progress with atomic writes so a container restart resumes from
  the last confirmed round (redo of an unconfirmed round is allowed; skipping
  rounds or double-applying a confirmed update is impossible).
* Fence every round message with ``task_id`` + ``gen`` so late messages from a
  previous training group can never advance a new group.
* Keep a downloadable weight artifact that only ever reflects confirmed
  rounds; a failed save leaves the previous complete artifact in place.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import lrmath
from storage import (atomic_write_json, delete_files, exists,
                     if_exists_read_json, read_json)

DATA_DIR = os.environ.get("COORD_DATA", "/data")
JOB_PATH = os.path.join(DATA_DIR, "job.json")
STATUS_PATH = os.path.join(DATA_DIR, "status.json")
WEIGHTS_DIR = os.path.join(DATA_DIR, "weights")
LATEST_WEIGHTS = os.path.join(WEIGHTS_DIR, "weights_latest.json")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

WORKERS = {
    "worker1": os.environ.get("WORKER1_URL", "http://worker1:8000"),
    "worker2": os.environ.get("WORKER2_URL", "http://worker2:8000"),
}
HTTP_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_ROWS, MIN_ROWS = 200, 2
N_FEATURES = 2


class Ctx:
    """In-memory runtime state, always rebuildable from durable files."""

    def __init__(self) -> None:
        self.job: Optional[Dict[str, Any]] = None
        self.status: Optional[Dict[str, Any]] = None
        self.task: Optional[asyncio.Task] = None
        self.client: Optional[httpx.AsyncClient] = None
        self.worker_live: Dict[str, bool] = {k: False for k in WORKERS}


CTX = Ctx()


# ============================================================ small utilities
def now_ms() -> int:
    return int(time.time() * 1000)


def httpc() -> httpx.AsyncClient:
    assert CTX.client is not None
    return CTX.client


def save_status(status: Dict[str, Any]) -> Dict[str, Any]:
    atomic_write_json(STATUS_PATH, status)
    CTX.status = status
    return status


def require_job() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if CTX.job is None or CTX.status is None:
        raise HTTPException(404, "no training task yet; upload a CSV first")
    return CTX.job, CTX.status


def split_indices(n: int) -> Tuple[List[int], List[int]]:
    """Deterministic, mutually-exclusive contiguous shard assignment."""
    cut = (n + 1) // 2  # worker1 gets the extra row when n is odd
    return list(range(0, cut)), list(range(cut, n))


def fresh_status_for_job(job: Dict[str, Any], status_str: str, event: str,
                         withdrawal: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    """Zero-initialised status document for a (possibly revised) job."""
    x = np.asarray(job["x"], dtype=np.float64)
    y = np.asarray(job["y"], dtype=np.float64)
    zero = lrmath.to_list(lrmath.zero_buffers())
    doc: Dict[str, Any] = {
        "task_id": job["task_id"], "gen": 0,
        "status": status_str, "confirmed_round": 0,
        "total_rounds": int(job["rounds"]),
        "w": zero, "v": zero,
        "loss_history": [{"round": 0,
                          "loss": lrmath.bce_loss(lrmath.augment(x), y, zero),
                          "at": now_ms()}],
        "running_round": None, "last_event": event,
        "last_error": None, "created_at": now_ms(),
    }
    if withdrawal is not None:
        doc["withdrawal"] = withdrawal
    return doc


def purge_weights() -> None:
    """Irrevocably delete every published weight artifact.

    Called the moment a record withdrawal is accepted: the old model must
    become undownloadable at once and unrecoverable even after a restart.
    """
    delete_files(WEIGHTS_DIR,
                 lambda name: name.startswith("weights_")
                 and name.endswith(".json"))


def weights_downloadable(st: Optional[Dict[str, Any]]) -> bool:
    """Download gate: a withdrawal revokes the old model immediately, and a
    revised task only publishes weights once retraining has completed."""
    if not exists(LATEST_WEIGHTS):
        return False
    if st:
        rec = st.get("withdrawal")
        if rec and rec.get("state") == "pending":
            return False
        if rec and st.get("status") != "completed":
            return False
    return True


# ============================================================ CSV validation
def parse_csv(raw: bytes) -> Tuple[np.ndarray, np.ndarray]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, f"CSV must be UTF-8 encoded: {exc}")
    reader = list(csv.reader(io.StringIO(text)))
    reader = [r for r in reader if any(c.strip() for c in r)]  # drop blanks
    if len(reader) < MIN_ROWS:
        raise HTTPException(400, "CSV needs 2-200 data rows (plus an optional "
                                 "header row)")

    header, data_rows = reader[0], reader[1:]
    # Detect an optional header row: if the first row has a non-numeric cell,
    # treat it as the header; otherwise every row is data (no header).
    def _row_is_numeric(row: List[str]) -> bool:
        if len(row) != N_FEATURES + 1:
            return False
        try:
            float(row[0]), float(row[1])
        except ValueError:
            return False
        return row[2].strip() in ("0", "1")

    first_is_header = not _row_is_numeric(header)
    rows = data_rows if first_is_header else reader

    if not (MIN_ROWS <= len(rows) <= MAX_ROWS):
        raise HTTPException(400, f"row count must be in [{MIN_ROWS},{MAX_ROWS}],"
                                 f" got {len(rows)}")
    xs: List[List[float]] = []
    ys: List[float] = []
    for i, row in enumerate(rows):
        if len(row) != N_FEATURES + 1:
            raise HTTPException(400, f"row {i+1}: expected 3 columns "
                                     f"(x1,x2,label), got {len(row)}")
        try:
            x1, x2 = float(row[0]), float(row[1])
        except ValueError:
            raise HTTPException(400, f"row {i+1}: non-numeric feature value")
        label = row[2].strip()
        if label not in ("0", "1"):
            raise HTTPException(400, f"row {i+1}: label must be 0 or 1")
        if not (np.isfinite(x1) and np.isfinite(x2)):
            raise HTTPException(400, f"row {i+1}: feature must be finite")
        xs.append([x1, x2])
        ys.append(float(label))
    if len(set(ys)) < 2:
        raise HTTPException(400, "labels must contain both classes 0 and 1")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


# ============================================================ weight artifacts
def write_weights_artifact(round_: int, w: List[float], v: List[float],
                           loss: float) -> None:
    """Publish weights for a confirmed round.

    Writes a new immutable file and atomically swaps the 'latest' pointer.
    If anything fails, the previous complete artifact stays untouched because
    the swap is the very last step and itself atomic.
    """
    payload = {
        "round": round_, "confirmed_round": round_, "loss": loss,
        "weights": {"bias": w[0], "w1": w[1], "w2": w[2]},
        "momentum": {"v_bias": v[0], "v1": v[1], "v2": v[2]},
        "saved_at": now_ms(),
    }
    named = os.path.join(WEIGHTS_DIR, f"weights_round_{round_:04d}.json")
    atomic_write_json(named, payload)
    atomic_write_json(LATEST_WEIGHTS, payload)


# ============================================================ worker messaging
async def worker_call(worker_id: str, path: str, payload: Dict[str, Any],
                      tolerate: bool = False) -> Dict[str, Any]:
    url = WORKERS[worker_id] + path
    try:
        r = await httpc().post(url, json=payload, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise HTTPException(r.status_code,
                                f"{worker_id} {path} -> {r.status_code}: "
                                f"{r.text[:200]}")
        CTX.worker_live[worker_id] = True
        return r.json()
    except (httpx.HTTPError, HTTPException) as exc:
        CTX.worker_live[worker_id] = False
        if tolerate:
            return {"_error": str(exc)}
        raise


async def worker_health(worker_id: str) -> Dict[str, Any]:
    try:
        r = await httpc().get(WORKERS[worker_id] + "/health",
                              timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        CTX.worker_live[worker_id] = True
        return r.json()
    except Exception as exc:  # noqa: BLE001 - surfaced as status field
        CTX.worker_live[worker_id] = False
        return {"worker_id": worker_id, "alive": False, "error": str(exc)}


# ============================================================ shard distribution
def shard_rows(job: Dict[str, Any], worker_id: str) -> List[Dict[str, Any]]:
    """This worker's mutually-exclusive slice of the job's dataset."""
    return [{"x": [float(job["x"][i][0]), float(job["x"][i][1])],
             "y": float(job["y"][i])} for i in job["shards"][worker_id]]


async def push_shards(job: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Install the job's shards on both workers (best effort per worker)."""
    return dict(zip(WORKERS, await asyncio.gather(*(
        worker_call(wid, "/load",
                    {"task_id": job["task_id"], "worker_id": wid,
                     "rows": shard_rows(job, wid)}, tolerate=True)
        for wid in WORKERS))))


def shards_ok(job: Dict[str, Any], results: Dict[str, Dict[str, Any]]) -> bool:
    """Both workers acknowledged the exact shard of the exact task."""
    for wid in WORKERS:
        res = results.get(wid, {})
        if "_error" in res:
            return False
        if res.get("task_id") != job["task_id"]:
            return False
        if int(res.get("n", -1)) != len(job["shards"][wid]):
            return False
    return True


# ============================================================ training loop
async def run_training() -> None:
    """Round machine. A failure at any point pauses the whole group; progress
    never advances past the last durably confirmed round."""
    job, status = require_job()
    st = dict(status)
    st["status"] = "running"
    st["last_event"] = "training loop started"
    st["running_round"] = None
    save_status(st)

    x = np.asarray(job["x"], dtype=np.float64)
    y = np.asarray(job["y"], dtype=np.float64)
    x_aug = lrmath.augment(x)
    total_rounds = int(job["rounds"])
    try:
        while st["confirmed_round"] < total_rounds:
            rnd = st["confirmed_round"] + 1
            st["status"] = "running"
            st["running_round"] = rnd
            st["last_event"] = f"round {rnd}: requesting shard gradients"
            save_status(st)

            gen = int(st["gen"])
            payload = {"task_id": job["task_id"], "gen": gen, "round": rnd}
            g1, g2 = await asyncio.gather(
                worker_call("worker1", "/gradient", payload),
                worker_call("worker2", "/gradient", payload),
            )
            # Defensive: both gradients must be computed at the same
            # confirmed round within the same fenced generation.
            for g in (g1, g2):
                if g.get("gen") != gen or g.get("from_round") != rnd - 1:
                    raise RuntimeError(f"fence/round mismatch in gradient: {g}")

            n1, n2 = int(g1["n"]), int(g2["n"])
            if n1 + n2 != len(y):
                raise RuntimeError("shard sizes do not cover the full dataset")
            # Concatenate residuals in canonical dataset row order (worker1's
            # shard precedes worker2's), then take ONE full-data gradient:
            # identical summation order as uninterrupted single-process
            # training, therefore bit-identical even after fault recovery.
            residuals = np.asarray(
                list(g1["residuals"]) + list(g2["residuals"]),
                dtype=np.float64)
            if residuals.shape[0] != len(y):
                raise RuntimeError("residual count mismatch")
            full_grad = lrmath.full_average_gradient(x_aug, residuals)
            st["last_event"] = (f"round {rnd}: rebuilt full-data average "
                                f"gradient over {n1+n2} samples")
            save_status(st)

            commit_payload = {"task_id": job["task_id"], "gen": gen,
                              "round": rnd,
                              "grad": lrmath.to_list(full_grad)}
            c1, c2 = await asyncio.gather(
                worker_call("worker1", "/commit", commit_payload),
                worker_call("worker2", "/commit", commit_payload),
            )
            for c in (c1, c2):
                if c.get("gen") != gen or c.get("round") != rnd:
                    raise RuntimeError(f"fence/round mismatch in commit: {c}")
            # Both workers persisted identical w,v for this round. Verify the
            # coordinator's own math matches what the workers applied.
            w_exp, v_exp = lrmath.momentum_update(
                st["w"], st["v"], full_grad)
            if not (np.allclose(c1["w"], c2["w"], atol=0.0, rtol=0.0)
                    and np.allclose(c1["v"], c2["v"], atol=0.0, rtol=0.0)
                    and np.array_equal(np.asarray(c1["w"]), w_exp)):
                raise RuntimeError("worker state divergence detected")

            loss = lrmath.bce_loss(x_aug, y, w_exp)
            st["confirmed_round"] = rnd
            st["w"] = lrmath.to_list(w_exp)
            st["v"] = lrmath.to_list(v_exp)
            st["loss_history"].append({"round": rnd, "loss": loss,
                                       "at": now_ms()})
            st["running_round"] = None
            st["last_event"] = f"round {rnd}: confirmed (both sides persisted)"
            save_status(st)                     # durable confirmation first
            write_weights_artifact(rnd, st["w"], st["v"], loss)  # then export

        st["status"] = "completed"
        st["running_round"] = None
        st["last_event"] = f"all {total_rounds} rounds confirmed"
        save_status(st)
    except Exception as exc:  # noqa: BLE001 - any worker failure pauses group
        st["status"] = "paused"
        st["running_round"] = None
        st["last_error"] = f"{type(exc).__name__}: {exc}"
        st["last_event"] = ("group paused: round "
                            f"{st['confirmed_round'] + 1} not confirmed; "
                            "restart the dead worker and press Resume")
        save_status(st)


def ensure_not_busy() -> None:
    if CTX.status and CTX.status["status"] == "running":
        raise HTTPException(409, "a round is currently in progress")


# ============================================================ API: schemas
class DelayIn(BaseModel):
    seconds: float = Field(..., ge=0, le=120)


class WithdrawIn(BaseModel):
    # 1-based row number within the CSV's data rows (header excluded).
    row: int = Field(..., ge=1)


# ============================================================ application
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    CTX.client = httpx.AsyncClient()
    CTX.job = if_exists_read_json(JOB_PATH)
    CTX.status = if_exists_read_json(STATUS_PATH)

    # Crash between writing job.json and status.json (task creation or a
    # record withdrawal): the durable job is authoritative. Rebuild a fresh
    # zero-state status for it; a withdrawal additionally stays pending.
    if CTX.job and (CTX.status is None
                    or CTX.status.get("task_id") != CTX.job["task_id"]):
        job = CTX.job
        withdrawn = job.get("withdrawn_row")
        if withdrawn is not None:
            withdrawal = {"row": int(withdrawn), "state": "pending",
                          "old_task_id": job.get("previous_task_id"),
                          "new_task_id": job["task_id"],
                          "requested_at": now_ms()}
            save_status(fresh_status_for_job(
                job, "withdrawing",
                f"recovered after restart: withdrawal of record {withdrawn} "
                f"is still pending", withdrawal))
        else:
            save_status(fresh_status_for_job(
                job, "paused",
                "recovered after restart during task creation; "
                "reloading shards"))
            try:
                results = await push_shards(job)
                if shards_ok(job, results):
                    st = dict(CTX.status)
                    st["status"] = "idle"
                    st["last_event"] = ("shards reloaded after restart; "
                                        "press Start")
                    save_status(st)
            except Exception:  # noqa: BLE001 - stays paused, user may reset
                pass

    if CTX.status is not None and CTX.status["status"] == "running":
        # Coordinator died mid-round: nothing after the last confirmed round
        # may count. Mark paused; the next resume re-prepares both workers.
        CTX.status["status"] = "paused"
        CTX.status["running_round"] = None
        CTX.status["last_event"] = (
            "coordinator restarted while a round was in flight; "
            "unconfirmed round will be redone")
        save_status(CTX.status)

    # A pending withdrawal survives restarts: the old model stays revoked
    # (re-purge in case the crash happened before the deletion), and the
    # shard redistribution is retried best-effort. If a worker is still
    # down the withdrawal simply remains pending; the web page can retry
    # the same row at any time without re-uploading the CSV.
    if CTX.status and CTX.status.get("withdrawal", {}).get("state") == "pending":
        purge_weights()
        if CTX.status["status"] != "withdrawing":
            CTX.status["status"] = "withdrawing"
            save_status(CTX.status)
        try:
            await finalize_withdrawal_push()
        except Exception:  # noqa: BLE001 - stays pending; page can retry
            pass
    yield
    if CTX.task:
        CTX.task.cancel()
    await CTX.client.aclose()


app = FastAPI(title="Fault-tolerant logistic regression trainer",
              lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    with open(os.path.join(WEB_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/status")
async def status() -> JSONResponse:
    health = await asyncio.gather(*(worker_health(w) for w in WORKERS))
    out: Dict[str, Any] = {"workers": {h["worker_id"]: h for h in health}}
    if CTX.status:
        out.update(CTX.status)
    out["worker_live"] = CTX.worker_live
    out["n_rows"] = int(CTX.job["n"]) if CTX.job else None
    out["weights_available"] = weights_downloadable(CTX.status)
    return JSONResponse(out)


@app.post("/api/job")
async def create_job(file: UploadFile = File(...),
                     rounds: int = 10) -> JSONResponse:
    if not (1 <= rounds <= 100):
        raise HTTPException(400, "rounds must be in [1,100]")
    if CTX.status and CTX.status["status"] not in ("completed",):
        raise HTTPException(409, "a task already exists; reset it first")
    raw = await file.read()
    x, y = parse_csv(raw)
    n = len(y)
    idx1, idx2 = split_indices(n)

    task_id = uuid.uuid4().hex
    job = {
        "task_id": task_id, "rounds": int(rounds), "n": n,
        "x": [[float(a), float(b)] for a, b in x],
        "y": [float(v) for v in y],
        "shards": {"worker1": idx1, "worker2": idx2},
        "created_at": now_ms(),
        "hyper": {"lr": lrmath.LEARNING_RATE,
                  "momentum": lrmath.MOMENTUM},
    }
    # Durable first, before touching any worker.
    atomic_write_json(JOB_PATH, job)
    CTX.job = job  # keep in-memory context in sync with the durable file
    status_doc = fresh_status_for_job(job, "idle", "task created")
    save_status(status_doc)
    zero = status_doc["w"]
    write_weights_artifact(0, zero, zero, status_doc["loss_history"][0]["loss"])

    # Push the mutually-exclusive shards to the two independent processes.
    results = await push_shards(job)
    if not shards_ok(job, results):
        st = dict(CTX.status)
        st["status"] = "paused"
        errs = {w: r.get("_error") for w, r in results.items()
                if "_error" in r}
        st["last_error"] = f"shard load failed: {errs}"
        st["last_event"] = "task created but a worker is unreachable"
        save_status(st)
        return JSONResponse(st, status_code=202)
    st = dict(CTX.status)
    st["last_event"] = "shards loaded on both workers; press Start"
    save_status(st)
    return JSONResponse(st)


async def _prepare_workers(gen: int) -> None:
    """Install a fresh fencing generation and align both workers to the
    coordinator's last confirmed round (durable rollback/catch-up)."""
    job, st = require_job()
    payload = {"task_id": job["task_id"], "gen": int(gen),
               "round": int(st["confirmed_round"]),
               "w": st["w"], "v": st["v"]}
    p1, p2 = await asyncio.gather(
        worker_call("worker1", "/prepare", payload),
        worker_call("worker2", "/prepare", payload),
    )
    for p in (p1, p2):
        if p.get("gen") != gen or p.get("round") != st["confirmed_round"]:
            raise RuntimeError(f"prepare fence/round mismatch: {p}")


@app.post("/api/start")
async def start() -> JSONResponse:
    job, st = require_job()
    ensure_not_busy()
    if st["status"] not in ("idle", "paused"):
        raise HTTPException(409, f"cannot start from status {st['status']}")
    new_gen = int(st["gen"]) + 1  # every (re)start opens a new training group
    st = dict(st)
    st["gen"] = new_gen
    st["status"] = "preparing"
    st["last_error"] = None
    st["last_event"] = f"opening training generation {new_gen}"
    save_status(st)
    await _prepare_workers(new_gen)
    CTX.task = asyncio.create_task(run_training())
    return JSONResponse(save_status({**st, "last_event":
                                     f"generation {new_gen} prepared; running"}))


@app.post("/api/resume")
async def resume() -> JSONResponse:
    # Resume == guarded restart: same entry point, new generation, roll both
    # workers to the confirmed round, then continue. No round is skipped.
    return await start()


@app.post("/api/pause")
async def pause() -> JSONResponse:
    job, st = require_job()
    if st["status"] != "running":
        raise HTTPException(409, "not running")
    if CTX.task:
        CTX.task.cancel()
        CTX.task = None
    st = dict(st)
    st["status"] = "paused"
    st["running_round"] = None
    st["last_event"] = "paused between rounds; resume redoes nothing confirmed"
    save_status(st)
    return JSONResponse(st)


# ============================================================ record withdrawal
async def finalize_withdrawal_push() -> JSONResponse:
    """(Re)distribute the revised shards and, only when BOTH workers hold the
    revised shard of the new task, mark the withdrawal applied.

    Safe to call repeatedly (accept, web-page retry, restart recovery): the
    durable status never leaves ``withdrawing`` until both sides acknowledge,
    so a dead worker or a coordinator restart simply keeps it pending.
    """
    job, st = require_job()
    results = await push_shards(job)
    st = dict(CTX.status)
    if shards_ok(job, results):
        rec = dict(st["withdrawal"])
        rec["state"] = "applied"
        rec["applied_at"] = now_ms()
        st["status"] = "idle"
        st["withdrawal"] = rec
        st["last_error"] = None
        st["last_event"] = (f"record {rec['row']} withdrawn; revised shards "
                            f"loaded on both workers; training restarts from "
                            f"zero (weights download unlocks at completion)")
        save_status(st)
        return JSONResponse(st)
    st["status"] = "withdrawing"
    errs = {w: r.get("_error") for w, r in results.items() if "_error" in r}
    st["last_error"] = (f"withdrawal pending, a worker is unreachable: {errs}")
    st["last_event"] = ("withdrawal stays pending; old weights remain revoked; "
                        "retry the same row to continue")
    save_status(st)
    return JSONResponse(st, status_code=202)


@app.post("/api/withdraw")
async def withdraw(body: WithdrawIn) -> JSONResponse:
    """Withdraw one voided record (by original CSV data-row number, 1-based).

    Accepted only while the task is paused or completed. Acceptance is a
    one-way door: the old weights are revoked immediately and irrevocably,
    the revised dataset is re-sharded to both workers under a NEW task id
    (old-generation traffic is fenced), and training restarts from zero.
    """
    job, st = require_job()
    if st["status"] in ("running", "preparing"):
        raise HTTPException(409, "pause or finish training before withdrawing "
                                 "a record")
    # Make sure no training loop is still settling before mutating state.
    if CTX.task and not CTX.task.done():
        CTX.task.cancel()
        await asyncio.gather(CTX.task, return_exceptions=True)
    CTX.task = None

    row = int(body.row)
    rec = st.get("withdrawal")
    if rec and rec.get("state") == "pending":
        if int(rec["row"]) != row:
            raise HTTPException(409, f"withdrawal of record {rec['row']} is "
                                     f"already pending; retry that row or "
                                     f"reset the task")
        # Idempotent retry of the same withdrawal: the revised job is already
        # durable, only the shard redistribution needs to finish. No CSV
        # re-upload is needed - the data lives in job.json.
        return await finalize_withdrawal_push()
    if (rec and rec.get("state") == "applied"
            and int(rec.get("row", -1)) == row
            and st["status"] == "idle" and st["confirmed_round"] == 0):
        # Same withdrawal already fully applied; a duplicate retry is a no-op.
        return JSONResponse(st)
    if st["status"] not in ("paused", "completed"):
        raise HTTPException(409, f"cannot withdraw from status "
                                 f"'{st['status']}'; pause or complete first")

    n = int(job["n"])
    if row > n:
        raise HTTPException(400, f"row must be in [1,{n}] (data rows of the "
                                 f"current dataset)")
    # Validate the revised dataset BEFORE changing any state: withdrawing
    # must leave at least two rows and both label classes, otherwise the
    # request is rejected and the original task stays fully usable.
    keep = [i for i in range(n) if i != row - 1]
    kept_labels = {float(job["y"][i]) for i in keep}
    if len(keep) < MIN_ROWS:
        raise HTTPException(400, "withdrawal would leave fewer than 2 data "
                                 "rows; rejected, task unchanged")
    if len(kept_labels) < 2:
        raise HTTPException(400, "withdrawal would leave a single label "
                                 "class; rejected, task unchanged")

    # Build the revised job under a new task id. The new id is what fences
    # off every in-flight or late message of the old training group.
    new_task_id = uuid.uuid4().hex
    idx1, idx2 = split_indices(len(keep))
    revised = {
        "task_id": new_task_id, "rounds": int(job["rounds"]), "n": len(keep),
        "x": [job["x"][i] for i in keep],
        "y": [job["y"][i] for i in keep],
        "shards": {"worker1": idx1, "worker2": idx2},
        "created_at": now_ms(),
        "hyper": job.get("hyper", {"lr": lrmath.LEARNING_RATE,
                                   "momentum": lrmath.MOMENTUM}),
        "withdrawn_row": row,
        "previous_task_id": job["task_id"],
    }
    withdrawal = {"row": row, "state": "pending",
                  "old_task_id": job["task_id"], "new_task_id": new_task_id,
                  "requested_at": now_ms()}
    # Durable order: revised job, then the pending-withdrawal status, then the
    # weight revocation. A crash anywhere is recovered at startup (lifespan
    # reconciles a job/status task-id mismatch and re-purges the artifacts).
    atomic_write_json(JOB_PATH, revised)
    CTX.job = revised
    save_status(fresh_status_for_job(
        revised, "withdrawing",
        f"record {row} withdrawn; redistributing revised shards",
        withdrawal))
    purge_weights()  # old model is now undownloadable and unrecoverable
    return await finalize_withdrawal_push()


@app.post("/api/reset")
async def reset() -> JSONResponse:
    ensure_not_busy()
    if CTX.task and not CTX.task.done():
        CTX.task.cancel()
        await asyncio.gather(CTX.task, return_exceptions=True)
    CTX.task = None
    for p in (JOB_PATH, STATUS_PATH):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
    purge_weights()  # a cleared task must not keep serving old artifacts
    CTX.job = None
    CTX.status = None
    return JSONResponse({"ok": True, "event": "task cleared"})


@app.post("/api/workers/{worker_id}/delay")
async def set_worker_delay(worker_id: str, body: DelayIn) -> JSONResponse:
    if worker_id not in WORKERS:
        raise HTTPException(404, "unknown worker")
    res = await worker_call(worker_id, "/delay",
                            {"seconds": body.seconds}, tolerate=True)
    return JSONResponse({"worker_id": worker_id, "response": res})


@app.post("/api/workers/delay")
async def set_both_delay(body: DelayIn) -> JSONResponse:
    results = await asyncio.gather(*(
        worker_call(w, "/delay", {"seconds": body.seconds}, tolerate=True)
        for w in WORKERS))
    return JSONResponse(dict(zip(WORKERS, results)))


@app.post("/api/workers/{worker_id}/kill")
async def kill_worker(worker_id: str) -> JSONResponse:
    if worker_id not in WORKERS:
        raise HTTPException(404, "unknown worker")
    # Fire-and-forget SIGKILL. The in-flight round (if any) fails and the group
    # pauses; the entrypoint supervisor respawns the process fresh.
    asyncio.create_task(
        httpc().post(WORKERS[worker_id] + "/kill", timeout=2.0))
    CTX.worker_live[worker_id] = False
    return JSONResponse({"worker_id": worker_id,
                         "event": "SIGKILL sent; group will pause"})


@app.get("/api/weights")
async def download_weights() -> FileResponse:
    st = CTX.status
    rec = st.get("withdrawal") if st else None
    if rec and rec.get("state") == "pending":
        raise HTTPException(409, f"weights revoked: withdrawal of record "
                                 f"{rec.get('row')} is still pending")
    if rec and st["status"] != "completed":
        raise HTTPException(409, "record withdrawn: weights become "
                                 "downloadable once retraining completes")
    if not exists(LATEST_WEIGHTS):
        raise HTTPException(404, "no confirmed weights yet")
    doc = read_json(LATEST_WEIGHTS)
    # The artifact only ever contains a fully confirmed round.
    if CTX.status and doc["confirmed_round"] > CTX.status["confirmed_round"]:
        raise HTTPException(409, "weight artifact ahead of confirmed state")
    return FileResponse(
        LATEST_WEIGHTS, media_type="application/json",
        filename=f"weights_round_{doc['confirmed_round']:04d}.json")


@app.get("/api/reference")
async def reference() -> JSONResponse:
    """Single-process full-data reference trajectory (for verification)."""
    job, _ = require_job()
    x = np.asarray(job["x"], dtype=np.float64)
    y = np.asarray(job["y"], dtype=np.float64)
    ws, vs, losses = lrmath.reference_train(x, y, int(job["rounds"]))
    return JSONResponse({
        "rounds": int(job["rounds"]),
        "weights": [lrmath.to_list(w) for w in ws],
        "momentum": [lrmath.to_list(v) for v in vs],
        "loss": losses,
    })
