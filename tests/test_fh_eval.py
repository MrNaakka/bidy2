"""Parallel evals in infra/verda/fh.py (eval-up/status/fetch/wait/down), every SkyPilot and ssh call replaced."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra" / "verda"))
import fh
from test_fh_infra import sandbox  # noqa: F401  (autouse fixture: no cloud, no tunnel, records down())


class FakeVerda:
    def __init__(self, instances, volumes):
        self.instances, self.volumes, self.deleted = instances, volumes, []

    def instances_get(self):
        return self.instances

    def volumes_get(self):
        return self.volumes

    def instance_action(self, iid, action, volume_ids=None):
        self.deleted.append((iid, action, volume_ids))

    def volume_delete(self, vid):
        self.deleted.append((vid, "volume"))


@pytest.fixture
def verda(monkeypatch):
    fake = FakeVerda([], [])
    monkeypatch.setattr(fh, "verda_client", lambda: fake)
    monkeypatch.setattr(fh, "sky_user_hash", lambda: "f5c0b31c")
    return fake


@pytest.fixture
def evalbox(tmp_path, monkeypatch, verda):
    for name, fname in [("EVAL_STATE_PATH", "evals.json"), ("EVAL_LOCK_PATH", "evals.lock"), ("RUNS_DIR", "runs")]:
        monkeypatch.setattr(fh, name, tmp_path / fname)
    fh.ENV_PATH.write_text("QWEN_API_KEY=k\n")
    tasks = tmp_path / "tasks" / "dev"
    for t in ("a-1", "b-2"):
        (tasks / t / "tests").mkdir(parents=True)
        (tasks / t / "task.toml").write_text(f"secret {t}\n")
    lora = tmp_path / "out" / "r1" / "final"
    lora.mkdir(parents=True)
    (lora / "adapter_config.json").write_text(json.dumps({"r": 16}))
    (lora / "adapter_model.safetensors").write_bytes(b"w")
    (lora / "README.md").write_text("not shipped")
    monkeypatch.setattr(fh, "git_commit", lambda: "abc")
    return SimpleNamespace(tasks=tasks, lora=lora.parent, tmp=tmp_path)


def eval_args(evalbox, **kw):
    base = {"tasks": str(evalbox.tasks), "split": None, "label": None, "lora": None, "runs": 3, "only": None,
            "n": 10, "instance_type": None, "disk_size": None, "on_demand": False, "model_repo": None,
            "model_rev": None, "max_usd": 60, "max_run_hours": 4, "new": False, "resume": False}
    return SimpleNamespace(**{**base, **kw})


def test_definition_pins_the_yaml_instance_type_and_detects_lora(evalbox):
    d = fh.eval_definition(eval_args(evalbox))
    assert d["fleet"] == "dev-qwen" and d["instance_type"] == "1H100.80S.30V" and d["use_spot"]
    assert d["served_name"] == fh.SERVED_NAME and d["n_tasks"] == 2 and d["model_rev"].startswith("3756181")
    d = fh.eval_definition(eval_args(evalbox, lora=str(evalbox.lora), on_demand=True))
    assert d["fleet"] == "dev-qwen-fh" and d["served_name"] == fh.LORA_NAME and not d["use_spot"]
    assert d["lora_dir"].endswith("final") and d["lora_sha"]


def test_definition_rejects_unloadable_adapters_and_bad_names(evalbox):
    (evalbox.lora / "final" / "adapter_config.json").write_text(json.dumps({"r": 32}))
    with pytest.raises(SystemExit, match="rank"):
        fh.eval_definition(eval_args(evalbox, lora=str(evalbox.lora)))
    with pytest.raises(SystemExit, match="label"):
        fh.eval_definition(eval_args(evalbox, label="Bad_Label"))


def test_stage_holds_tasks_privately_and_targets_the_relay(evalbox):
    d = fh.eval_definition(eval_args(evalbox, lora=str(evalbox.lora)))
    stage = fh.stage_eval(d)
    assert (stage / "tasks" / "a-1" / "task.toml").read_text() == "secret a-1\n"
    assert all((p.stat().st_mode & 0o077) == 0 for p in [stage, stage / "tasks", stage / "tasks" / "a-1"])
    assert sorted(p.name for p in (stage / "lora").iterdir()) == sorted(fh.LORA_FILES)
    agent = json.loads((stage / "cfg.json").read_text())["agents"][0]
    assert agent["base_url"] == "http://172.17.0.1:18000/v1" and agent["model_name"] == "verda-qwen/qwen3.8-27b-fh"
    assert agent["kwargs"]["version"] == fh.OPENCODE_VERSION
    assert json.loads((stage / "meta.json").read_text())["tasks_sha"] == d["tasks_sha"]
    assert (stage / ".skyignore").exists() and (stage / "verify_weights.py").exists()
    assert (stage / "eval_job.sh").read_text() == fh.EVAL_JOB.read_text()


def test_launch_args_are_identical_across_boxes_but_the_job(evalbox):
    d = fh.eval_definition(eval_args(evalbox, on_demand=True))
    a1, a2 = (fh.eval_launch_args(d, Path("/s"), i) for i in (1, 2))
    assert "--no-use-spot" in a1 and a1[a1.index("--instance-type") + 1] == "1H100.80S.30V"
    assert "JOB=fh-dev-qwen-run1" in a1 and "JOB=fh-dev-qwen-run2" in a2

    def strip(a):
        return [x for x in a if not x.startswith(("JOB=", "fh-ev-"))]

    assert strip(a1) == strip(a2) and "N_CONCURRENT=10" in a1 and "LORA=0" in a1
    assert "QWEN_API_KEY" in a1 and not any(x.endswith("=k") for x in a1)   # key only via --secret from the env


def fake_launch(rcs):
    def launch(defn, stage, runs):
        return {i: rcs.get(i, 0) for i in runs}
    return launch


def test_up_records_failed_launches_and_refuses_an_incomparable_relaunch(evalbox, sandbox, monkeypatch):
    monkeypatch.setattr(fh, "cluster_status", lambda name: None)
    monkeypatch.setattr(fh, "launch_eval_boxes", fake_launch({2: 1}))
    fh.cmd_eval_up(eval_args(evalbox))
    boxes = fh.load_evals()["dev-qwen"]["boxes"]
    assert [boxes[i]["state"] for i in "123"] == ["running", "launch_failed", "running"]
    fh.cmd_eval_up(eval_args(evalbox, only=[2]))
    assert fh.load_evals()["dev-qwen"]["boxes"]["2"]["launches"] == 2
    with pytest.raises(SystemExit, match="not be comparable"):
        fh.cmd_eval_up(eval_args(evalbox, only=[2], n=4))
    (evalbox.tasks / "a-1" / "task.toml").write_text("rebuilt\n")
    with pytest.raises(SystemExit, match="tasks_sha"):
        fh.cmd_eval_up(eval_args(evalbox, only=[2]))


def test_up_refuses_to_relaunch_a_running_box(evalbox, sandbox, monkeypatch):
    monkeypatch.setattr(fh, "cluster_status", lambda name: None)
    monkeypatch.setattr(fh, "launch_eval_boxes", fake_launch({}))
    fh.cmd_eval_up(eval_args(evalbox))
    monkeypatch.setattr(fh, "cluster_status", lambda name: "UP")
    with pytest.raises(SystemExit, match="is running"):
        fh.cmd_eval_up(eval_args(evalbox, only=[1]))
    assert sandbox.down == []


def fleet_state(evalbox, states, max_usd=60):
    d = fh.eval_definition(eval_args(evalbox))
    boxes = {str(i): {"cluster": f"fh-ev-dev-qwen-{i}", "job": f"fh-dev-qwen-run{i}", "state": s, "launched_at": 0}
             for i, s in enumerate(states, 1)}
    return {"meta": {**d, "created_at": 0}, "boxes": boxes, "max_usd": max_usd}


def test_tick_fetches_and_deletes_finished_boxes_and_reports_lost_ones(evalbox, sandbox, monkeypatch):
    fleet = fleet_state(evalbox, ["running", "running", "running"])
    status = {"fh-ev-dev-qwen-1": "UP", "fh-ev-dev-qwen-2": None, "fh-ev-dev-qwen-3": "UP"}
    jobs = {"fh-ev-dev-qwen-1": "SUCCEEDED", "fh-ev-dev-qwen-3": "RUNNING"}
    monkeypatch.setattr(fh, "cluster_status", lambda n: status[n])
    monkeypatch.setattr(fh, "last_job_status", lambda n: jobs[n])
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 1.0)
    fetched = []
    monkeypatch.setattr(fh, "fetch_eval", lambda f, b, j: fetched.append((b["job"], j)))
    notes = fh.eval_tick(fleet)
    assert fetched == [("fh-dev-qwen-run1", "SUCCEEDED")] and sandbox.down == ["fh-ev-dev-qwen-1"]
    assert [b["state"] for b in fleet["boxes"].values()] == ["done", "lost", "running"]
    assert any("--only 2" in n for n in notes)


def test_tick_cost_cap_fetches_partial_results_and_deletes_everything(evalbox, sandbox, monkeypatch):
    fleet = fleet_state(evalbox, ["running", "running"], max_usd=5)
    monkeypatch.setattr(fh, "cluster_status", lambda n: "UP" if n.endswith("1") else "INIT")
    monkeypatch.setattr(fh, "last_job_status", lambda n: "RUNNING")
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 3.0)
    fetched = []
    monkeypatch.setattr(fh, "fetch_eval", lambda f, b, j: fetched.append(b["job"]))
    fh.eval_tick(fleet)
    assert fetched == ["fh-dev-qwen-run1"] and sorted(sandbox.down) == ["fh-ev-dev-qwen-1", "fh-ev-dev-qwen-2"]
    assert {b["state"] for b in fleet["boxes"].values()} == {"budget"}


def test_tick_retries_a_failed_fetch_before_deleting(evalbox, sandbox, monkeypatch):
    fleet = fleet_state(evalbox, ["running"])
    monkeypatch.setattr(fh, "cluster_status", lambda n: "UP")
    monkeypatch.setattr(fh, "last_job_status", lambda n: "FAILED")
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 0.0)

    def broken(f, b, j):
        raise fh.subprocess.CalledProcessError(23, "rsync")

    monkeypatch.setattr(fh, "fetch_eval", broken)
    fh.eval_tick(fleet)
    assert fleet["boxes"]["1"]["state"] == "running" and sandbox.down == []


def test_fetch_writes_comparability_metadata(evalbox, sandbox, monkeypatch):
    fleet = fleet_state(evalbox, ["running"])
    box = fleet["boxes"]["1"]
    monkeypatch.setattr(fh, "ssh_base", lambda c: ["ssh"])
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 2.5)
    calls = []

    def fake_rsync(cmd, **k):
        calls.append(cmd)
        if cmd[-1].endswith("fh-status/"):
            st = Path(cmd[-1])
            st.mkdir(parents=True)
            (st / "box.json").write_text('{"gpus": ["NVIDIA H100"], "vllm_cmd": ["vllm", "serve"]}')
            (st / "timings.txt").write_text("job_start 100\nvllm_healthy 400\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fh.subprocess, "run", fake_rsync)
    dest = fh.fetch_eval(fleet, box, "SUCCEEDED")
    meta = json.loads((dest / "fh-eval-meta.json").read_text())
    assert meta["complete"] and meta["instance_type"] == "1H100.80S.30V" and meta["n_concurrent"] == 10
    assert meta["box"]["gpus"] == ["NVIDIA H100"] and meta["timings_rel_s"]["vllm_healthy"] == 300
    assert meta["cost_so_far"] == 2.5 and meta["run"] == "fh-dev-qwen-run1"
    assert any(c.startswith("--rsync-path=mkdir -p fh/runs/fh-dev-qwen-run1") for c in calls[0])


def test_tick_gives_up_fetching_after_a_few_tries(evalbox, sandbox, monkeypatch):
    fleet = fleet_state(evalbox, ["running"])
    monkeypatch.setattr(fh, "cluster_status", lambda n: "UP")
    monkeypatch.setattr(fh, "last_job_status", lambda n: "FAILED_SETUP")
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 0.0)

    def broken(f, b, j):
        raise fh.subprocess.CalledProcessError(23, "rsync")

    monkeypatch.setattr(fh, "fetch_eval", broken)
    for _ in range(fh.EVAL_FETCH_TRIES):
        fh.eval_tick(fleet)
    assert fleet["boxes"]["1"]["state"] == "fetch_failed" and sandbox.down == ["fh-ev-dev-qwen-1"]


def test_sweep_deletes_only_leaked_instances_of_clusters_skypilot_lost(verda, monkeypatch):
    inst = lambda iid, host, vol: SimpleNamespace(instance_id=iid, hostname=host, status="running", os_volume_id=vol)
    verda.instances = [inst("a", "fh-ev-dev-qwen-2-f5c0b31c-head", "va"),     # leaked by a timed-out create
                       inst("b", "fh-ev-dev-qwen-1-f5c0b31c-head", "vb"),     # SkyPilot still has this cluster
                       inst("c", "fhbot-serve-9768facc16ac", "vc"),           # the live endpoint
                       inst("d", "fh-ev-dev-qwen-22-f5c0b31c-head", "vd")]    # another cluster's name prefix
    verda.volumes = [SimpleNamespace(id="v1", name="fh-ev-dev-qwen-2-f5c0b31c-head", status="detached"),
                     SimpleNamespace(id="q", name="fh-qwen", status="detached"),
                     SimpleNamespace(id="v2", name="fh-ev-dev-qwen-2-x-head", status="attached")]
    verda.instances.append(inst("e", "fh-ev-dev-qwen-2-0teammate-head", "ve"))   # same name, someone else's
    monkeypatch.setattr(fh, "sky_user_hash", lambda: "f5c0b31c")
    monkeypatch.setattr(fh, "cluster_status", lambda n: "UP" if n == "fh-ev-dev-qwen-1" else None)
    notes = fh.sweep_leaks(["fh-ev-dev-qwen-1", "fh-ev-dev-qwen-2", "fh-serve"])
    assert verda.deleted == [("a", "delete", ["va"]), ("v1", "volume")] and len(notes) == 2


def test_up_leaves_unknown_clusters_alone_and_new_needs_a_stopped_fleet(evalbox, sandbox, monkeypatch):
    monkeypatch.setattr(fh, "cluster_status", lambda name: "UP")
    with pytest.raises(SystemExit, match="not a box of fleet"):
        fh.cmd_eval_up(eval_args(evalbox))
    assert sandbox.down == []
    monkeypatch.setattr(fh, "cluster_status", lambda name: None)
    monkeypatch.setattr(fh, "launch_eval_boxes", fake_launch({}))
    fh.cmd_eval_up(eval_args(evalbox))
    with pytest.raises(SystemExit, match="eval-down first"):
        fh.cmd_eval_up(eval_args(evalbox, n=4, new=True, only=[1]))


def test_wait_keeps_sweeping_after_a_failed_launch_for_a_while(evalbox, sandbox, monkeypatch):
    fleet = fleet_state(evalbox, ["launch_failed"])
    fleet["boxes"]["1"]["launched_at"] = fh.time.time()
    assert fh.sweeping(fleet["boxes"]["1"])
    fleet["boxes"]["1"]["launched_at"] -= fh.LEAK_WINDOW_S + 1
    assert not fh.sweeping(fleet["boxes"]["1"])


def test_new_fleet_keeps_sweeping_the_old_fleets_failed_launches(evalbox, sandbox, monkeypatch):
    monkeypatch.setattr(fh, "cluster_status", lambda name: None)
    monkeypatch.setattr(fh, "launch_eval_boxes", fake_launch({1: 1, 2: 1, 3: 1}))
    fh.cmd_eval_up(eval_args(evalbox))
    monkeypatch.setattr(fh, "launch_eval_boxes", fake_launch({}))
    fh.cmd_eval_up(eval_args(evalbox, n=4, new=True, only=[1]))
    fleet = fh.load_evals()["dev-qwen"]
    assert list(fleet["boxes"]) == ["1"] and fleet["meta"]["n_concurrent"] == 4
    assert [o["cluster"] for o in fleet["orphans"]] == ["fh-ev-dev-qwen-1", "fh-ev-dev-qwen-2", "fh-ev-dev-qwen-3"]
    swept = []
    monkeypatch.setattr(fh, "sweep_leaks", lambda clusters: swept.extend(clusters) or [])
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 0.0)
    monkeypatch.setattr(fh, "last_job_status", lambda n: "RUNNING")
    fh.eval_tick(fleet)
    assert "fh-ev-dev-qwen-2" in swept and "fh-ev-dev-qwen-1" in swept


def test_tick_leaves_a_box_alone_while_eval_up_is_launching_on_it(evalbox, sandbox, monkeypatch):
    fleet = fleet_state(evalbox, ["launching"])   # e.g. --resume: the box still shows its old, finished job
    fleet["boxes"]["1"]["launched_at"] = fh.time.time()
    monkeypatch.setattr(fh, "cluster_status", lambda n: "UP")
    monkeypatch.setattr(fh, "last_job_status", lambda n: "SUCCEEDED")
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 0.0)
    monkeypatch.setattr(fh, "fetch_eval", lambda f, b, j: pytest.fail("fetched a box that is being launched"))
    fh.eval_tick(fleet)
    assert fleet["boxes"]["1"]["state"] == "launching" and sandbox.down == []


def test_probe_parsing():
    p = fh.parse_probe("trials=3\ndone=2\nphase=harbor_start\nvllm:num_requests_running=7\n"
                       "vllm:kv_cache_usage_perc=0.41\n")
    assert p["done"] == 2 and p["num_requests_running"] == 7 and p["kv_cache_usage_perc"] == 0.41
    assert p["phase"] == "harbor_start"
