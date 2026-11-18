"""End-to-end fault-injection tests against real uvicorn processes.

Spins up one coordinator + two independent worker processes (each with its own
temporary data directory, exactly like the Docker topology) and verifies:

1. clean distributed training equals the single-process reference trajectory
   bit-for-bit;
2. killing a worker mid-training pauses the group, and resume reproduces the
   exact uninterrupted trajectory (no skipped / double-applied round);
3. redoing an unconfirmed round after coordinator restart stays exact;
4. stale-generation messages are fenced (409) and cannot advance a new group;
5. downloaded weights only ever reflect confirmed rounds.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
ROUNDS = 30


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Service:
    def __init__(self, name, module, port, env, health_path="/health"):
        self.name, self.module, self.port, self.env = name, module, port, env
        self.health_path = health_path
        self.proc: subprocess.Popen | None = None

    def start(self):
        import site
        env = dict(os.environ)
        env.update(self.env)
        # Use -P (cwd excluded from sys.path, guarding against planted modules)
        # rather than -I, which would also drop PYTHONPATH and user site
        # packages. The app dir and locally installed deps are supplied here.
        extra = [str(APP), site.getusersitepackages()]
        env["PYTHONPATH"] = os.pathsep.join(extra) + os.pathsep + env.get(
            "PYTHONPATH", "")
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-m", "uvicorn",
             f"{self.module}:app", "--host", "127.0.0.1",
             "--port", str(self.port)],
            cwd=APP, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self._wait_up()

    def _wait_up(self, timeout=20):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise RuntimeError(f"{self.name} exited early:\n{out}")
            try:
                r = httpx.get(
                    f"http://127.0.0.1:{self.port}{self.health_path}",
                    timeout=1.0)
                if r.status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.15)
        raise TimeoutError(f"{self.name} did not come up")

    def hard_kill(self):
        assert self.proc
        self.proc.send_signal(signal.SIGKILL)
        self.proc.wait()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


@pytest.fixture()
def cluster(tmp_path):
    p1, p2, pc = _free_port(), _free_port(), _free_port()
    d1, d2, dc = (tmp_path / "w1"), (tmp_path / "w2"), (tmp_path / "coord")
    for d in (d1, d2, dc):
        d.mkdir()
    w1 = Service("worker1", "worker", p1,
                 {"WORKER_ID": "worker1", "WORKER_DATA": str(d1)})
    w2 = Service("worker2", "worker", p2,
                 {"WORKER_ID": "worker2", "WORKER_DATA": str(d2)})
    coord = Service("coordinator", "coordinator", pc, {
        "COORD_DATA": str(dc),
        "WORKER1_URL": f"http://127.0.0.1:{p1}",
        "WORKER2_URL": f"http://127.0.0.1:{p2}",
    }, health_path="/api/status")
    w1.start(); w2.start(); coord.start()
    yield {
        "coord": f"http://127.0.0.1:{pc}",
        "coord_svc": coord, "w1": w1, "w2": w2,
        "paths": {"w1": d1, "w2": d2, "coord": dc},
    }
    for s in (w1, w2, coord):
        s.stop()


def _dataset_csv(tmp_path: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    z = 1.2 * x[:, 0] - 0.6 * x[:, 1]
    y = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    p = tmp_path / "data.csv"
    lines = ["x1,x2,label"]
    for (a, b), c in zip(x, y):
        lines.append(f"{a:.10f},{b:.10f},{c}")
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _wait_status(base, predicate, timeout=40):
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        last = httpx.get(base + "/api/status", timeout=5).json()
        if predicate(last):
            return last
        time.sleep(0.25)
    raise AssertionError(f"condition not met; last={json.dumps(last)[:600]}")


def _create_and_start(base, csv_path, rounds=ROUNDS):
    with open(csv_path, "rb") as f:
        r = httpx.post(base + f"/api/job?rounds={rounds}",
                       files={"file": ("data.csv", f, "text/csv")},
                       timeout=15)
    r.raise_for_status()
    httpx.post(base + "/api/start", timeout=10).raise_for_status()


# --------------------------------------------------------------------- tests
def test_clean_run_matches_reference(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    st = _wait_status(base, lambda d: d["status"] == "completed")
    assert st["confirmed_round"] == ROUNDS

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    # Exact, bit-for-bit equality with uninterrupted single-process training.
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert np.allclose(st["w"], ref["weights"][-1], atol=0.0, rtol=0.0)
    assert np.allclose(st["v"], ref["momentum"][-1], atol=0.0, rtol=0.0)

    # Weights download corresponds to the confirmed round only.
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == ROUNDS
    assert wdoc["weights"]["w1"] == st["w"][1]


def test_worker_kill_and_resume_is_numerically_exact(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path))
    # Let some rounds confirm, then SIGKILL worker1 mid-flight.
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 8)
    confirmed_at_kill = st["confirmed_round"]
    cluster["w1"].hard_kill()

    st = _wait_status(base, lambda d: d["status"] == "paused")
    assert st["confirmed_round"] == confirmed_at_kill  # nothing half-counted

    # The killed worker's durable state never exceeds the confirmed round by
    # more than one (a commit may have landed just before the kill).
    state1 = json.loads((cluster["paths"]["w1"] / "state.json").read_text())
    assert state1["round"] <= confirmed_at_kill + 1

    # Restart the worker process (fresh generation-neutral process).
    cluster["w1"].start()
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed")

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k], f"round {k} differs"
    np.testing.assert_array_equal(np.asarray(st["w"]),
                                  np.asarray(ref["weights"][-1]))
    # Exactly one loss entry per round: no duplicate application.
    rounds_seen = [p["round"] for p in st["loss_history"]]
    assert rounds_seen == list(range(ROUNDS + 1))


def test_coordinator_restart_mid_round_redoes_without_skip(cluster, tmp_path):
    base = cluster["coord"]
    # Add delay so a round is almost certainly in flight at kill time.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    cluster["coord_svc"].hard_kill()
    time.sleep(0.8)  # let at least one delayed in-flight round drift
    cluster["coord_svc"].start()

    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["status"] == "paused"
    assert st["confirmed_round"] == confirmed
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)

    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(13))


def test_stale_generation_messages_are_fenced(cluster, tmp_path):
    base = cluster["coord"]
    # Slow each round down so the pause reliably lands while training runs.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.5}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    old_gen = st["gen"]
    task_id = st["task_id"]

    # Pause and resume -> a strictly newer generation is installed.
    deadline = time.time() + 15
    while True:
        r = httpx.post(base + "/api/pause", timeout=5)
        if r.status_code == 200:
            break
        if httpx.get(base + "/api/status").json()["status"] == "completed":
            raise AssertionError("training completed before pause took effect")
        assert time.time() < deadline
        time.sleep(0.3)
    _wait_status(base, lambda d: d["status"] == "paused")
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st2 = _wait_status(base, lambda d: d["gen"] > old_gen)
    new_gen = st2["gen"]

    # A late message from the old training group must be rejected (409) and
    # must not mutate any state. Reach the worker directly on its own port.
    w1_port = cluster["w1"].port
    r = httpx.post(f"http://127.0.0.1:{w1_port}/gradient",
                   json={"task_id": task_id, "gen": old_gen,
                         "round": st2["confirmed_round"] + 1},
                   timeout=5)
    assert r.status_code == 409

    # Unknown task id is also rejected.
    r2 = httpx.post(f"http://127.0.0.1:{w1_port}/commit",
                    json={"task_id": "deadbeef", "gen": new_gen,
                          "round": 1, "grad": [0, 0, 0]}, timeout=5)
    assert r2.status_code in (409, 410)

    final = _wait_status(base, lambda d: d["status"] == "completed")
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(final["loss_history"]):
        assert point["loss"] == ref["loss"][k]


def test_full_cluster_reboot_keeps_durable_state(cluster, tmp_path):
    base = cluster["coord"]
    # Slow rounds so the kill reliably lands mid-training, not after completion.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.4}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=15)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 4)
    confirmed = st["confirmed_round"]
    # Kill every process, then bring the whole cluster back up.
    cluster["coord_svc"].hard_kill()
    cluster["w1"].hard_kill()
    cluster["w2"].hard_kill()
    for s in (cluster["w1"], cluster["w2"], cluster["coord_svc"]):
        s.start()
    st = httpx.get(base + "/api/status", timeout=10).json()
    # More rounds may have confirmed between observation and the kill; the
    # durable invariant that matters is that progress is never lost and the
    # reboot resumes from a confirmed round >= the one we observed.
    assert st["confirmed_round"] >= confirmed
    assert st["status"] in ("paused", "idle", "completed")
    if st["status"] != "completed":
        httpx.post(base + "/api/resume", timeout=15).raise_for_status()
        st = _wait_status(base, lambda d: d["status"] == "completed",
                          timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert [p["round"] for p in st["loss_history"]] == list(range(16))


def test_csv_validation_rules(cluster, tmp_path):
    base = cluster["coord"]
    good = _dataset_csv(tmp_path, n=20)
    # Below the 2-row minimum: header + a single data row -> 400.
    p = tmp_path / "small.csv"
    p.write_text("x1,x2,label\n1,2,0\n")
    with open(p, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("small.csv", f, "text/csv")})
    assert r.status_code == 400
    # The minimum allowed size (2 rows, both classes) is accepted.
    pmin = tmp_path / "min.csv"
    pmin.write_text("1,2,0\n3,4,1\n")  # no header, exactly 2 rows
    with open(pmin, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=3",
                       files={"file": ("min.csv", f, "text/csv")})
    assert r.status_code in (200, 202)
    httpx.post(base + "/api/reset", timeout=5)
    # Bad label.
    p2 = tmp_path / "badlabel.csv"
    lines = good.read_text().splitlines()
    head, body = lines[0], lines[1:]
    body[0] = body[0][:-1] + "9"
    p2.write_text("\n".join([head] + body))
    with open(p2, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("b.csv", f, "text/csv")})
    assert r.status_code == 400
    # Rounds out of range.
    with open(good, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=101",
                       files={"file": ("g.csv", f, "text/csv")})
    assert r.status_code == 400
    # Over the 200-row maximum.
    big = tmp_path / "big.csv"
    rows = ["x1,x2,label"] + [f"{i},1,{i % 2}" for i in range(201)]
    big.write_text("\n".join(rows))
    with open(big, "rb") as f:
        r = httpx.post(base + "/api/job?rounds=5",
                       files={"file": ("big.csv", f, "text/csv")})
    assert r.status_code == 400


def test_download_weights_only_confirmed_and_save_keeps_last(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=12)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 5)
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == st["confirmed_round"]
    # On-disk immutable artifact plus latest pointer exist.
    cdir = cluster["paths"]["coord"] / "weights"
    assert (cdir / f"weights_round_{st['confirmed_round']:04d}.json").exists()
    assert (cdir / "weights_latest.json").exists()
    final = _wait_status(base, lambda d: d["status"] == "completed")
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == final["confirmed_round"] == 12


def test_worker_commit_dedup_and_prepare_rollback(cluster, tmp_path):
    """Direct worker protocol: a retried commit applies once, and /prepare
    rolls an unconfirmed advance back to the last confirmed round."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=8)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    task_id, gen = st["task_id"], st["gen"]
    w1_port = cluster["w1"].port
    url = f"http://127.0.0.1:{w1_port}"

    h = httpx.get(url + "/health", timeout=5).json()
    at = h["round"]
    # Gradient for the next round.
    g = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen, "round": at + 1},
                   timeout=5).json()
    grad = [0.01, -0.02, 0.03]
    c1 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c1["round"] == at + 1 and c1["dedup"] is False
    w_after, v_after = c1["w"], c1["v"]
    # Retried commit for the same round must NOT apply the step a second time.
    c2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": gen,
                          "round": at + 1, "grad": grad}, timeout=5).json()
    assert c2["dedup"] is True and c2["w"] == w_after and c2["v"] == v_after

    # A newer generation's /prepare rolls the worker back to the confirmed
    # round (discarding the unconfirmed local advance above).
    new_gen = gen + 1
    p = httpx.post(url + "/prepare",
                   json={"task_id": task_id, "gen": new_gen,
                         "round": at, "w": st["w"], "v": st["v"]},
                   timeout=5).json()
    assert p["gen"] == new_gen and p["round"] == at
    h2 = httpx.get(url + "/health", timeout=5).json()
    assert h2["round"] == at and h2["gen"] == new_gen

    # Old-generation traffic is now fenced.
    r = httpx.post(url + "/gradient",
                   json={"task_id": task_id, "gen": gen, "round": at + 1},
                   timeout=5)
    assert r.status_code == 409
    # No skipping rounds: committing round at+2 before at+1 is rejected.
    r2 = httpx.post(url + "/commit",
                    json={"task_id": task_id, "gen": new_gen,
                          "round": at + 2, "grad": grad}, timeout=5)
    assert r2.status_code == 409


def test_delay_endpoint_survives_and_rounds_still_exact(cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.2}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=6)
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]


# ----------------------------------------------------- record withdrawal tests
def _pause_group(base):
    """Pause a running group, tolerating the between-rounds race."""
    deadline = time.time() + 20
    while True:
        r = httpx.post(base + "/api/pause", timeout=5)
        if r.status_code == 200:
            return
        st = httpx.get(base + "/api/status", timeout=5).json()["status"]
        if st == "completed":
            raise AssertionError("training completed before pause took effect")
        assert time.time() < deadline, f"could not pause; status={st}"
        time.sleep(0.3)


def test_withdraw_rejected_while_running(cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.3}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    # The page only offers withdrawal while paused/completed; the API agrees.
    r = httpx.post(base + "/api/withdraw", json={"row": 1}, timeout=5)
    assert r.status_code == 409
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    assert st["confirmed_round"] == 10
    assert "withdrawal" not in st


def test_withdraw_validation_rejects_and_task_stays_usable(cluster, tmp_path):
    base = cluster["coord"]
    p = tmp_path / "four.csv"
    p.write_text("x1,x2,label\n0,0,0\n1,1,1\n2,2,1\n3,3,0\n")
    _create_and_start(base, p, rounds=6)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    old_task = st["task_id"]
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200

    # Out-of-range row numbers are rejected (1-based data rows).
    assert httpx.post(base + "/api/withdraw", json={"row": 5},
                      timeout=5).status_code == 400
    assert httpx.post(base + "/api/withdraw", json={"row": 0},
                      timeout=5).status_code in (400, 422)
    # The original task is untouched and fully usable.
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["task_id"] == old_task and st["status"] == "completed"
    assert st["confirmed_round"] == 6
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200

    # Withdrawal that would leave a single label class -> rejected.
    (tmp_path / "three.csv").write_text("x1,x2,label\n0,0,0\n1,1,1\n2,2,1\n")
    httpx.post(base + "/api/reset", timeout=5)
    with open(tmp_path / "three.csv", "rb") as f:
        httpx.post(base + "/api/job?rounds=4",
                   files={"file": ("t.csv", f, "text/csv")},
                   timeout=15).raise_for_status()
    httpx.post(base + "/api/start", timeout=10)
    _wait_status(base, lambda d: d["status"] == "completed")
    r = httpx.post(base + "/api/withdraw", json={"row": 1}, timeout=5)
    assert r.status_code == 400 and "single label" in r.text
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["status"] == "completed" and st["confirmed_round"] == 4
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200

    # Withdrawal that would leave fewer than two rows -> rejected.
    (tmp_path / "two.csv").write_text("x1,x2,label\n0,0,0\n1,1,1\n")
    httpx.post(base + "/api/reset", timeout=5)
    with open(tmp_path / "two.csv", "rb") as f:
        httpx.post(base + "/api/job?rounds=2",
                   files={"file": ("t2.csv", f, "text/csv")},
                   timeout=15).raise_for_status()
    httpx.post(base + "/api/start", timeout=10)
    _wait_status(base, lambda d: d["status"] == "completed")
    r = httpx.post(base + "/api/withdraw", json={"row": 1}, timeout=5)
    assert r.status_code == 400 and "fewer than 2" in r.text
    st = httpx.get(base + "/api/status", timeout=5).json()
    assert st["status"] == "completed" and st["confirmed_round"] == 2
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200


def test_withdraw_completed_task_retrains_from_zero(cluster, tmp_path):
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path, n=12), rounds=8)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    old_task = st["task_id"]
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200

    # Withdraw data row 3 (1-based): accepted, task rebuilt from zero.
    r = httpx.post(base + "/api/withdraw", json={"row": 3}, timeout=10)
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["status"] == "idle"
    assert st["withdrawal"]["state"] == "applied"
    assert st["withdrawal"]["row"] == 3
    assert st["task_id"] != old_task
    assert st["confirmed_round"] == 0 and st["gen"] == 0
    assert [p["round"] for p in st["loss_history"]] == [0]
    assert st["total_rounds"] == 8

    # Old weights are revoked immediately and irrevocably (files gone).
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 409
    assert not list((cluster["paths"]["coord"] / "weights")
                    .glob("weights_*.json"))

    # Both workers hold only the revised shard of the NEW task.
    for w in ("w1", "w2"):
        h = httpx.get(f"http://127.0.0.1:{cluster[w].port}/health",
                      timeout=5).json()
        assert h["task_id"] == st["task_id"] and h["round"] == 0
    # Old training messages cannot advance the new task.
    r = httpx.post(f"http://127.0.0.1:{cluster['w1'].port}/gradient",
                   json={"task_id": old_task, "gen": 9, "round": 1}, timeout=5)
    assert r.status_code == 410

    # Retrain from zero; the download stays locked until completion.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.3}, timeout=5)
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    _pause_group(base)
    st = _wait_status(base, lambda d: d["status"] == "paused")
    assert 0 < st["confirmed_round"] < 8
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 409
    assert httpx.get(base + "/api/status",
                     timeout=5).json()["weights_available"] is False

    httpx.post(base + "/api/workers/delay", json={"seconds": 0}, timeout=5)
    httpx.post(base + "/api/resume", timeout=15).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)

    # Bit-exact against the single-process reference over the revised data.
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    assert [p["round"] for p in st["loss_history"]] == list(range(9))
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    np.testing.assert_array_equal(np.asarray(st["w"]),
                                  np.asarray(ref["weights"][-1]))
    wdoc = httpx.get(base + "/api/weights", timeout=5).json()
    assert wdoc["confirmed_round"] == 8
    assert wdoc["weights"]["w1"] == st["w"][1]


def test_withdraw_pending_when_worker_down_then_retry(cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.3}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=10)
    st = _wait_status(base, lambda d: d["confirmed_round"] >= 3)
    old_task = st["task_id"]
    _pause_group(base)
    _wait_status(base, lambda d: d["status"] == "paused")

    # One side is unreachable: the withdrawal is accepted but stays pending.
    cluster["w2"].hard_kill()
    r = httpx.post(base + "/api/withdraw", json={"row": 2}, timeout=10)
    assert r.status_code == 202, r.text
    st = r.json()
    assert st["status"] == "withdrawing"
    assert st["withdrawal"]["state"] == "pending"
    assert st["withdrawal"]["row"] == 2
    new_task = st["task_id"]
    assert new_task != old_task

    # While pending: old weights revoked, group cannot start, no old model.
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 409
    assert httpx.get(base + "/api/status",
                     timeout=5).json()["weights_available"] is False
    assert httpx.post(base + "/api/start", timeout=5).status_code == 409
    assert not list((cluster["paths"]["coord"] / "weights")
                    .glob("weights_*.json"))

    # A different row conflicts; the same row is an idempotent retry.
    assert httpx.post(base + "/api/withdraw", json={"row": 5},
                      timeout=5).status_code == 409
    assert httpx.post(base + "/api/withdraw", json={"row": 2},
                      timeout=10).status_code == 202

    # worker1 already holds the revised shard; the dead worker2 never got it,
    # so no mixture of old/new shards can be trained on.
    h1 = httpx.get(f"http://127.0.0.1:{cluster['w1'].port}/health",
                   timeout=5).json()
    assert h1["task_id"] == new_task and h1["round"] == 0

    # Revive worker2 and retry the same withdrawal: it completes.
    cluster["w2"].start()
    r = httpx.post(base + "/api/withdraw", json={"row": 2}, timeout=10)
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["status"] == "idle" and st["withdrawal"]["state"] == "applied"
    h2 = httpx.get(f"http://127.0.0.1:{cluster['w2'].port}/health",
                   timeout=5).json()
    assert h2["task_id"] == new_task and h2["round"] == 0

    # Old-task traffic is fenced on both workers.
    for w in ("w1", "w2"):
        rr = httpx.post(f"http://127.0.0.1:{cluster[w].port}/gradient",
                        json={"task_id": old_task, "gen": 9, "round": 1},
                        timeout=5)
        assert rr.status_code == 410

    # Train the revised task to completion: exact vs the revised reference.
    httpx.post(base + "/api/workers/delay", json={"seconds": 0}, timeout=5)
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    assert [p["round"] for p in st["loss_history"]] == list(range(11))
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200


def test_withdraw_pending_survives_coordinator_restart(cluster, tmp_path):
    base = cluster["coord"]
    httpx.post(base + "/api/workers/delay", json={"seconds": 0.3}, timeout=5)
    _create_and_start(base, _dataset_csv(tmp_path), rounds=8)
    _wait_status(base, lambda d: d["confirmed_round"] >= 2)
    _pause_group(base)
    _wait_status(base, lambda d: d["status"] == "paused")

    cluster["w2"].hard_kill()
    r = httpx.post(base + "/api/withdraw", json={"row": 4}, timeout=10)
    assert r.status_code == 202
    assert r.json()["withdrawal"]["state"] == "pending"

    # Coordinator restarts while the withdrawal is pending.
    cluster["coord_svc"].hard_kill()
    cluster["coord_svc"].start()

    st = httpx.get(base + "/api/status", timeout=10).json()
    assert st["status"] == "withdrawing"
    assert st["withdrawal"]["state"] == "pending"
    assert st["withdrawal"]["row"] == 4
    assert st["weights_available"] is False
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 409
    assert httpx.post(base + "/api/start", timeout=5).status_code == 409
    # The old model cannot be restored: artifacts are gone from disk.
    assert not list((cluster["paths"]["coord"] / "weights")
                    .glob("weights_*.json"))

    # Revive the missing side and retry the same withdrawal (no CSV needed).
    cluster["w2"].start()
    r = httpx.post(base + "/api/withdraw", json={"row": 4}, timeout=10)
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["status"] == "idle" and st["withdrawal"]["state"] == "applied"

    httpx.post(base + "/api/workers/delay", json={"seconds": 0}, timeout=5)
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200


def test_withdraw_crash_between_job_and_status_write_recovers(cluster, tmp_path):
    """Crash window inside acceptance: job.json already revised (new task id,
    withdrawn row recorded) while status.json is still the old task and the
    old weight artifacts still exist. Startup must reconcile to a pending
    withdrawal, purge the old model, and finish the redistribution."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=6)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    old_task = st["task_id"]
    cdir = cluster["paths"]["coord"]

    cluster["coord_svc"].hard_kill()
    job = json.loads((cdir / "job.json").read_text())
    revised = {**job, "task_id": "taskB", "withdrawn_row": 2,
               "previous_task_id": old_task,
               "x": job["x"][:1] + job["x"][2:],
               "y": job["y"][:1] + job["y"][2:]}
    revised["n"] = len(revised["y"])
    cut = (revised["n"] + 1) // 2
    revised["shards"] = {"worker1": list(range(cut)),
                         "worker2": list(range(cut, revised["n"]))}
    (cdir / "job.json").write_text(json.dumps(revised))
    # status.json untouched (old task); old weights still on disk.
    assert (cdir / "weights" / "weights_latest.json").exists()
    cluster["coord_svc"].start()

    st = httpx.get(base + "/api/status", timeout=10).json()
    assert st["task_id"] == "taskB"
    assert st["withdrawal"]["row"] == 2
    # Both workers were reachable, so recovery finished the redistribution.
    assert st["status"] == "idle"
    assert st["withdrawal"]["state"] == "applied"
    # The old model was purged during recovery and cannot be downloaded.
    assert not list((cdir / "weights").glob("weights_*.json"))
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 409
    for w in ("w1", "w2"):
        h = httpx.get(f"http://127.0.0.1:{cluster[w].port}/health",
                      timeout=5).json()
        assert h["task_id"] == "taskB" and h["round"] == 0

    # The recovered task trains from zero, exactly like a clean withdrawal.
    httpx.post(base + "/api/start", timeout=10).raise_for_status()
    st = _wait_status(base, lambda d: d["status"] == "completed", timeout=60)
    ref = httpx.get(base + "/api/reference", timeout=10).json()
    for k, point in enumerate(st["loss_history"]):
        assert point["loss"] == ref["loss"][k]
    assert httpx.get(base + "/api/weights", timeout=5).status_code == 200


def test_worker_interrupted_load_never_mixes_shard_and_model(cluster, tmp_path):
    """A worker killed between writing a new shard and its matching state
    (e.g. mid-withdrawal) restarts with the new shard at round 0 - never the
    new shard combined with the old task's model."""
    base = cluster["coord"]
    _create_and_start(base, _dataset_csv(tmp_path), rounds=6)
    st = _wait_status(base, lambda d: d["status"] == "completed")
    old_task = st["task_id"]

    # Simulate an interrupted /load on worker1: shard.json already belongs to
    # the revised task B while state.json is still task A at round 6.
    w1dir = cluster["paths"]["w1"]
    shard = json.loads((w1dir / "shard.json").read_text())
    state = json.loads((w1dir / "state.json").read_text())
    assert state["task_id"] == old_task and state["round"] == 6
    cluster["w1"].hard_kill()
    shard["task_id"] = "taskB"
    shard["rows"] = shard["rows"][:-1]  # revised (smaller) shard
    (w1dir / "shard.json").write_text(json.dumps(shard))
    cluster["w1"].start()

    # Reconciled to the new shard's task, training state reset to zero, and
    # the reconciliation itself is durable.
    h = httpx.get(f"http://127.0.0.1:{cluster['w1'].port}/health",
                  timeout=5).json()
    assert h["task_id"] == "taskB" and h["round"] == 0
    disk = json.loads((w1dir / "state.json").read_text())
    assert disk["task_id"] == "taskB" and disk["round"] == 0
    # Old-task traffic is fenced.
    r = httpx.post(f"http://127.0.0.1:{cluster['w1'].port}/gradient",
                   json={"task_id": old_task, "gen": 3, "round": 7}, timeout=5)
    assert r.status_code == 410
