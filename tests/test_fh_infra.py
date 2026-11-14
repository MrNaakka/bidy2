"""Decision logic of infra/verda/fh.py, with every SkyPilot and network call replaced."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra" / "verda"))
import fh


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    for name, fname in [("ENV_PATH", ".env"), ("STATE_PATH", "state.json"), ("LOCK_PATH", "lock"),
                        ("TUNNEL_PIDFILE", "tunnel.pid"), ("TUNNEL_LOG", "tunnel.log"), ("STAGE_DIR", "stage")]:
        monkeypatch.setattr(fh, name, tmp_path / fname)
    monkeypatch.setattr(fh.time, "sleep", lambda s: None)
    calls = SimpleNamespace(launch_serve=0, launch_train=0, down=[], tunnel=0)
    monkeypatch.setattr(fh, "launch_serve", lambda state: (setattr(calls, "launch_serve", calls.launch_serve + 1),
                                                           state["serve"].__setitem__("launched_at", fh.time.time())))
    monkeypatch.setattr(fh, "launch_train", lambda run: setattr(calls, "launch_train", calls.launch_train + 1))
    monkeypatch.setattr(fh, "down", lambda name: calls.down.append(name))
    monkeypatch.setattr(fh, "start_tunnel", lambda: setattr(calls, "tunnel", calls.tunnel + 1))
    monkeypatch.setattr(fh, "tunnel_pid", lambda: 1)
    monkeypatch.setattr(fh, "stop_tunnel", lambda: None)
    monkeypatch.setattr(fh, "checkpoint_and_stop", lambda: calls.down.append("checkpoint"))
    return calls


def cloud(monkeypatch, status, job, healthy=True, cost=0.0):
    monkeypatch.setattr(fh, "cluster_status", lambda name: status)
    monkeypatch.setattr(fh, "last_job_status", lambda name: job)
    monkeypatch.setattr(fh, "healthy", lambda names: healthy)
    monkeypatch.setattr(fh, "cluster_cost_since", lambda name, since: cost)


def args(**kw):
    base = {"run": "r1", "data": None, "epochs": 1, "lr": 1e-4, "grad_accum": 8, "lora_r": 16,
            "save_every": 10, "trainer_arg": None, "max_usd": 45, "on_demand": False}
    return SimpleNamespace(**{**base, **kw})


def test_serve_healthy_box_is_left_alone(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "RUNNING")
    assert fh.serve_tick({"serve": {"launched_at": 0}}, 0)
    assert sandbox.launch_serve == 0
    assert "QWEN_BASE_URL=http://10.0.2.2:18000/v1" in fh.ENV_PATH.read_text()


@pytest.mark.parametrize("status,job", [(None, None), ("INIT", None), ("UP", "FAILED"), ("UP", "SUCCEEDED")])
def test_serve_relaunches_a_gone_or_dead_box(sandbox, monkeypatch, status, job):
    cloud(monkeypatch, status, job)
    fh.serve_tick({"serve": {"launched_at": 0}}, 0)
    assert sandbox.launch_serve == 1


def test_serve_waits_for_a_booting_box_then_replaces_it_after_the_grace(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "RUNNING", healthy=False)
    state = {"serve": {"launched_at": fh.time.time()}}
    assert not fh.serve_tick(state, 0)
    assert sandbox.launch_serve == 0
    state["serve"]["launched_at"] -= fh.SERVE_GRACE_S + 1
    fh.serve_tick(state, 0)
    assert sandbox.launch_serve == 1


def test_trainer_args_reject_unsafe_or_unloadable_values():
    with pytest.raises(SystemExit, match="flag not allowed"):
        fh.trainer_args("r1", "/mnt/fh/data/d", args(trainer_arg=["--out=/tmp/x"]))
    with pytest.raises(SystemExit, match="plain"):
        fh.trainer_args("r1", "/mnt/fh/data/d", args(trainer_arg=["x;rm"]))
    with pytest.raises(SystemExit, match="1..16"):
        fh.trainer_args("r1", "/mnt/fh/data/d", args(trainer_arg=["--lora-r=32"]))
    assert "--seed=3" in fh.trainer_args("r1", "/mnt/fh/data/d", args(trainer_arg=["--seed=3"]))


def test_run_definition_tracks_data_content(tmp_path):
    data = tmp_path / "sft"
    data.mkdir()
    (data / "sft.jsonl").write_text("a\n")
    first = fh.run_definition(args(data=str(data)))
    assert json.loads(first["manifest"])["trainer_args"] == first["trainer_args"]
    (data / "sft.jsonl").write_text("b\n")
    second = fh.run_definition(args(data=str(data)))
    assert second["data_sha"] != first["data_sha"] and second["manifest"] != first["manifest"]
    with pytest.raises(SystemExit, match="changed since run"):
        fh.stage(first)


def train_state(**kw):
    return {"train": {"run": "r1", "started_at": 0, "max_usd": 45, "failures": 0, **kw}}


def test_train_budget_stops_the_run(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "RUNNING", cost=50)
    state = train_state()
    assert fh.train_tick(state).startswith("budget")
    assert state["train"]["failed"] and sandbox.down == ["checkpoint", "fh-train"] and sandbox.launch_train == 0


def test_train_relaunches_after_spot_loss_without_counting_a_failure(sandbox, monkeypatch):
    cloud(monkeypatch, None, None)
    state = train_state()
    assert fh.train_tick(state) == "launched"
    assert state["train"]["failures"] == 0


def test_train_gives_up_after_repeated_failures(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "FAILED")
    monkeypatch.setattr(fh, "ssh_out", lambda c, cmd, timeout=60: SimpleNamespace(returncode=1))
    state = train_state(failures=fh.TRAIN_MAX_FAILURES - 2)
    assert fh.train_tick(state) == "launched"
    assert fh.train_tick(state).startswith("failed")
    assert sandbox.launch_train == 1


def test_train_manifest_mismatch_fails_at_once(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "FAILED")
    monkeypatch.setattr(fh, "ssh_out", lambda c, cmd, timeout=60: SimpleNamespace(returncode=0))
    state = train_state()
    assert fh.train_tick(state) == "failed: manifest mismatch"
    assert sandbox.launch_train == 0


def test_train_success_fetches_and_deletes_the_box(sandbox, monkeypatch, tmp_path):
    cloud(monkeypatch, "UP", "SUCCEEDED")
    monkeypatch.setattr(fh, "fetch_run", lambda cluster, run: fh.ROOT / "train" / "out" / run)
    state = train_state()
    assert fh.train_tick(state).startswith("done")
    assert state["train"]["done"] and state["train"]["adapter"] == "/mnt/fh/adapters/r1"
    assert sandbox.down == ["fh-train"]


def test_inactive_run_keeps_the_box_deleted(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "RUNNING")
    assert fh.train_tick(train_state(stopped=True)) == "inactive"
    assert sandbox.down == ["fh-train"]


def test_harbor_entry_declares_image_input():
    entry = fh.harbor_agent_entry("http://h/v1", fh.LORA_NAME, "1.19.5")
    model = entry["kwargs"]["opencode_config"]["provider"]["verda-qwen"]["models"][fh.LORA_NAME]
    assert entry["model_name"] == "verda-qwen/qwen3.8-27b-fh" and entry["kwargs"]["version"] == "1.19.5"
    assert "image" in model["modalities"]["input"]


def test_train_success_deletes_the_box_even_when_the_fetch_fails(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "SUCCEEDED")

    def broken_fetch(cluster, run):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fh, "fetch_run", broken_fetch)
    state = train_state()
    assert "train-fetch" in fh.train_tick(state)
    assert state["train"]["done"] and sandbox.down == ["fh-train"]


def test_run_names_cannot_escape_the_stage_dir(tmp_path):
    for bad in ("..", ".", ".hidden", "a/b"):
        with pytest.raises(SystemExit, match="--run"):
            fh.run_definition(args(run=bad, data=str(tmp_path)))


def test_serve_gives_up_after_repeated_failures(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "RUNNING", healthy=False)
    state = {"serve": {"launched_at": 0, "failed_relaunches": fh.SERVE_MAX_FAILED_RELAUNCHES}}
    assert not fh.serve_tick(state, 0)
    assert "serve" not in state and sandbox.down == ["fh-serve"] and sandbox.launch_serve == 0


def test_serve_survives_a_tunnel_that_will_not_start(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "RUNNING", healthy=True)
    monkeypatch.setattr(fh, "tunnel_pid", lambda: None)

    def busy_port():
        raise RuntimeError("ssh tunnel exited with 255")

    monkeypatch.setattr(fh, "start_tunnel", busy_port)
    state = {"serve": {"launched_at": 0}}
    assert not fh.serve_tick(state, 0)
    assert sandbox.launch_serve == 1


def test_serve_counts_a_crashing_job_toward_the_limit(sandbox, monkeypatch):
    cloud(monkeypatch, "UP", "FAILED")
    state = {"serve": {"launched_at": fh.time.time(), "failed_relaunches": fh.SERVE_MAX_FAILED_RELAUNCHES}}
    assert not fh.serve_tick(state, 0)
    assert "serve" not in state and sandbox.launch_serve == 0


def failing_launch(*a, **k):
    raise fh.subprocess.CalledProcessError(1, "sky launch")


def test_train_counts_a_failed_launch_that_left_a_box(sandbox, monkeypatch):
    cloud(monkeypatch, "INIT", None)
    monkeypatch.setattr(fh, "launch_train", failing_launch)
    state = train_state(failures=fh.TRAIN_MAX_FAILURES - 1)
    assert fh.train_tick(state).startswith("failed")
    assert sandbox.down == ["fh-train"]


def test_train_does_not_count_a_launch_that_found_no_capacity(sandbox, monkeypatch):
    cloud(monkeypatch, None, None)
    monkeypatch.setattr(fh, "launch_train", failing_launch)
    state = train_state()
    with pytest.raises(fh.subprocess.CalledProcessError):
        fh.train_tick(state)
    assert state["train"]["failures"] == 0


def test_serve_ensure_deletes_a_box_nobody_wants(sandbox, monkeypatch):
    cloud(monkeypatch, "INIT", None)
    fh.cmd_serve_ensure(SimpleNamespace(wait_min=0))
    assert sandbox.down == ["fh-serve"]
