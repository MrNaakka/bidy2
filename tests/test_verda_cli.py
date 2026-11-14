"""Failure-path tests for infra/verda/verda_cli.py (all cloud calls mocked)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from verda.exceptions import APIException

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra" / "verda"))
import verda_cli as vc  # noqa: E402


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    for name, fname in [("ENV_PATH", ".env"), ("STATE_PATH", "state.json"), ("LEDGER_PATH", "ledger.jsonl"),
                        ("LOCK_PATH", "lock"), ("TUNNEL_PIDFILE", "tunnel.pid"), ("TUNNEL_LOG", "tunnel.log"),
                        ("KNOWN_HOSTS", "known_hosts")]:
        monkeypatch.setattr(vc, name, tmp_path / fname)
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    monkeypatch.setattr(vc.time, "sleep", lambda s: None)
    (tmp_path / ".env").write_text("QWEN_API_KEY=k\nQWEN_BASE_URL=http://10.0.2.2:18000/v1\n")
    return tmp_path


def ledger(tmp_path) -> list[dict]:
    p = tmp_path / "ledger.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def inst(id="i1", status="running", hostname="fh-qwen-serve-1", ip="1.2.3.4"):
    return SimpleNamespace(id=id, status=status, hostname=hostname, ip=ip, os_volume_id=None,
                           instance_type="1A6000.10V", is_spot=True, price_per_hour=0.4, location="FIN-01")


def fake_create(inst_id):
    """Stand-in for create_instance that tracks the box in state like the real one."""
    def create(c, state, **k):
        state[k["role"]] = {"id": inst_id, "hostname": k["hostname"], "created": 0, **(k.get("record") or {})}
        vc.save_state(state)
        return inst(inst_id, hostname=k["hostname"])
    return create


NOT_FOUND = APIException("not_found", "Instance not found")
TRANSIENT = APIException("internal_error", "upstream 502")


# ---- 1. delete_instance -----------------------------------------------------------
def test_delete_persistent_failure_raises_and_writes_no_ledger(sandbox):
    c = mock.Mock()
    c.instances.get_by_id.return_value = inst()
    c.instances.action.side_effect = TRANSIENT
    with pytest.raises(vc.DeleteFailed):
        vc.delete_instance(c, "i1", attempts=3)
    assert c.instances.action.call_count == 3
    assert not any(r["event"] == "delete" for r in ledger(sandbox))


def test_delete_network_error_on_lookup_is_not_success(sandbox):
    c = mock.Mock()
    c.instances.get_by_id.side_effect = ConnectionError("reset")
    with pytest.raises(vc.DeleteFailed):
        vc.delete_instance(c, "i1", attempts=2)
    assert ledger(sandbox) == []


def test_delete_retries_transient_then_confirms(sandbox):
    c = mock.Mock()
    # attempt 1: lookup ok, action fails; attempt 2: lookup ok, action ok, then confirmation 404
    c.instances.get_by_id.side_effect = [inst(), inst(), inst(status="deleting"), NOT_FOUND]
    c.instances.action.side_effect = [TRANSIENT, None]
    vc.delete_instance(c, "i1", reason="t")
    assert c.instances.action.call_count == 2
    assert [r["event"] for r in ledger(sandbox)] == ["delete"]


def test_delete_already_gone_is_success(sandbox):
    c = mock.Mock()
    c.instances.get_by_id.side_effect = NOT_FOUND
    vc.delete_instance(c, "i1")
    c.instances.action.assert_not_called()
    assert ledger(sandbox)[0]["id"] == "i1"


def test_delete_discontinued_status_counts_as_gone(sandbox):
    c = mock.Mock()
    c.instances.get_by_id.return_value = inst(status="discontinued")
    vc.delete_instance(c, "i1")
    c.instances.action.assert_not_called()
    assert len(ledger(sandbox)) == 1


def test_delete_that_never_takes_effect_fails(sandbox, monkeypatch):
    c = mock.Mock()
    c.instances.get_by_id.return_value = inst()  # still running after every delete request
    clock = iter(range(0, 10**6, 100))
    monkeypatch.setattr(vc.time, "time", lambda: next(clock))
    with pytest.raises(vc.DeleteFailed):
        vc.delete_instance(c, "i1", attempts=2, confirm_s=150)
    assert ledger(sandbox) == []


def test_serve_down_keeps_state_when_delete_fails(sandbox, monkeypatch):
    vc.save_state({"serve": {"id": "i1"}})
    c = mock.Mock()
    c.instances.get.return_value = [inst()]
    c.instances.get_by_id.return_value = inst()
    c.instances.action.side_effect = TRANSIENT
    monkeypatch.setattr(vc, "client", lambda: c)
    stop = mock.Mock()
    monkeypatch.setattr(vc, "stop_tunnel", stop)
    with pytest.raises(vc.DeleteFailed):
        vc.cmd_serve_down(SimpleNamespace())
    assert vc.load_state()["serve"]["id"] == "i1"
    assert vc.load_env()["QWEN_BASE_URL"] == "http://10.0.2.2:18000/v1"
    stop.assert_not_called()
    assert ledger(sandbox) == []


def test_serve_down_success_clears_state(sandbox, monkeypatch):
    vc.save_state({"serve": {"id": "i1"}})
    c = mock.Mock()
    c.instances.get.return_value = [inst()]
    c.instances.get_by_id.side_effect = [inst(), NOT_FOUND]
    c.startup_scripts.get.return_value = []
    monkeypatch.setattr(vc, "client", lambda: c)
    monkeypatch.setattr(vc, "stop_tunnel", mock.Mock())
    vc.cmd_serve_down(SimpleNamespace())
    assert "serve" not in vc.load_state()
    assert vc.load_env()["QWEN_BASE_URL"] == ""
    vc.stop_tunnel.assert_called_once()


def test_main_turns_delete_failure_into_nonzero_exit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["verda_cli.py", "delete", "i1"])
    vc.save_state({"initial_balance": "x"})
    monkeypatch.setattr(vc, "cmd_delete", mock.Mock(side_effect=vc.DeleteFailed("boom")))
    monkeypatch.setattr(vc, "client", mock.Mock())
    monkeypatch.setattr(vc, "reconcile", mock.Mock())
    with pytest.raises(SystemExit) as ei:
        vc.main()
    assert "boom" in str(ei.value)


# ---- 2. startup failures clean up -------------------------------------------------
def _download_args(**kw):
    return SimpleNamespace(repos=list(vc.MODELS), type="CPU.8V.32G", on_demand=True, keep=False,
                           timeout_min=1, **kw)


def _vol_state():
    return {"volume": {"id": "v1", "location": "FIN-01", "mount": "/mnt/fh", "size_gb": 300},
            "weights": {r: {"path": f"/mnt/fh/models/{r.split('/')[1]}"} for r in vc.MODELS}}


@pytest.fixture
def cloud(monkeypatch):
    c = mock.Mock()
    c.instances.get.return_value = []
    c.startup_scripts.get.return_value = []
    monkeypatch.setattr(vc, "client", lambda: c)
    monkeypatch.setattr(vc, "get_volume", lambda c, s: True)
    monkeypatch.setattr(vc, "render", lambda *a, **k: "script")
    deleted = []
    monkeypatch.setattr(vc, "delete_instance", lambda c, i, reason="": deleted.append(i))
    c.deleted = deleted
    return c


def test_download_boot_timeout_deletes_instance(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    monkeypatch.setattr(vc, "create_instance", fake_create("dl1"))
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=TimeoutError("not running")))
    with pytest.raises(TimeoutError):
        vc.cmd_download(_download_args())
    assert cloud.deleted == ["dl1"]


def test_download_interrupted_create_cleans_up_by_hostname(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    hostnames = []

    def create(*a, **k):
        hostnames.append(k["hostname"])
        raise KeyboardInterrupt  # order placed, we never got the id back

    monkeypatch.setattr(vc, "create_instance", create)
    cloud.instances.get.side_effect = lambda: [inst("dl2", hostname=hostnames[0])] if hostnames else []
    with pytest.raises(KeyboardInterrupt):
        vc.cmd_download(_download_args())
    assert cloud.deleted == ["dl2"]


def test_download_keep_skips_cleanup(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    monkeypatch.setattr(vc, "create_instance", fake_create("dl1"))
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=TimeoutError()))
    with pytest.raises(TimeoutError):
        vc.cmd_download(SimpleNamespace(**{**vars(_download_args()), "keep": True}))
    assert cloud.deleted == []


def _serve_args(**kw):
    d = dict(type=None, variant=None, on_demand=False, mtp=False, timeout_min=0.01, keep=False, no_wait=False,
             lora=None)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture
def serve_cloud(cloud, monkeypatch, sandbox):
    vc.save_state(_vol_state())
    monkeypatch.setattr(vc, "pick_candidate", lambda c, a: [(0.4, "1A6000.10V", "fp8", [], True)])
    monkeypatch.setattr(vc, "create_instance", fake_create("g1"))
    return cloud


def test_serve_up_health_timeout_deletes_new_instance(serve_cloud, monkeypatch):
    monkeypatch.setattr(vc, "wait_running", lambda c, i: inst("g1"))
    monkeypatch.setattr(vc, "ensure_tunnel", lambda i, ip: True)
    monkeypatch.setattr(vc, "health", lambda u, k, **kw: False)
    monkeypatch.setattr(vc, "ssh", mock.Mock(side_effect=OSError))
    with pytest.raises(SystemExit):
        vc.cmd_serve_up(_serve_args())
    assert serve_cloud.deleted == ["g1"]
    assert "serve" not in vc.load_state()


def test_serve_up_instance_error_deletes_new_instance(serve_cloud, monkeypatch):
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=RuntimeError("entered error")))
    with pytest.raises(RuntimeError):
        vc.cmd_serve_up(_serve_args())
    assert serve_cloud.deleted == ["g1"]


def test_serve_up_keep_preserves_instance(serve_cloud, monkeypatch):
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=TimeoutError()))
    with pytest.raises(TimeoutError):
        vc.cmd_serve_up(_serve_args(keep=True))
    assert serve_cloud.deleted == []
    assert vc.load_state()["serve"]["id"] == "g1"


def test_serve_up_no_wait_returns_without_cleanup(serve_cloud, monkeypatch):
    wr = mock.Mock()
    monkeypatch.setattr(vc, "wait_running", wr)
    vc.cmd_serve_up(_serve_args(no_wait=True))
    wr.assert_not_called()
    assert serve_cloud.deleted == []
    assert vc.load_state()["serve"]["id"] == "g1"


def test_serve_up_does_not_delete_preexisting_instance(serve_cloud, monkeypatch):
    serve_cloud.instances.get.return_value = [inst("old")]
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=TimeoutError()))
    with pytest.raises(TimeoutError):
        vc.cmd_serve_up(_serve_args())
    assert serve_cloud.deleted == []


# ---- 3. tunnel ------------------------------------------------------------------
def test_serve_script_binds_localhost_only():
    text = (vc.REMOTE / "serve.sh").read_text()
    assert "--host 127.0.0.1" in text and "0.0.0.0" not in text


def test_tunnel_command_options():
    cmd = vc.tunnel_cmd("1.2.3.4", "inst-1")
    joined = " ".join(cmd)
    assert cmd[0] == "ssh" and "-N" in cmd and "ControlPath=none" in joined and cmd[1:3] == ["-F", "none"]
    for opt in ("ServerAliveInterval=15", "ExitOnForwardFailure=yes", "StrictHostKeyChecking=accept-new",
                f"UserKnownHostsFile={vc.KNOWN_HOSTS}", "HostKeyAlias=inst-1"):
        assert opt in joined
    assert f"{vc.TUNNEL_BIND}:{vc.TUNNEL_PORT}:127.0.0.1:{vc.PORT}" in cmd
    assert cmd[-1] == "root@1.2.3.4"


def test_container_url_is_instance_independent():
    assert vc.container_base_url() == f"http://{vc.TUNNEL_CONTAINER_HOST}:{vc.TUNNEL_PORT}/v1"
    assert "1.2.3.4" not in vc.container_base_url()


def test_ensure_tunnel_restarts_dead_and_repoints_stale(monkeypatch):
    start = mock.Mock(return_value=True)
    monkeypatch.setattr(vc, "start_tunnel", start)
    # dead tunnel (pidfile present, process gone)
    vc.TUNNEL_PIDFILE.write_text(json.dumps({"pid": 999999999, "inst_id": "a", "ip": "1.1.1.1"}))
    assert vc.ensure_tunnel("a", "1.1.1.1")
    start.assert_called_once_with("a", "1.1.1.1")
    # live tunnel pointing at the old box
    start.reset_mock()
    monkeypatch.setattr(vc, "tunnel_info", lambda: {"pid": 1, "inst_id": "a", "ip": "1.1.1.1"})
    assert vc.ensure_tunnel("b", "2.2.2.2")
    start.assert_called_once_with("b", "2.2.2.2")
    # live tunnel already correct -> no restart
    start.reset_mock()
    assert vc.ensure_tunnel("a", "1.1.1.1")
    start.assert_not_called()


def test_tunnel_info_ignores_unrelated_pid(sandbox):
    import os
    vc.TUNNEL_PIDFILE.write_text(json.dumps({"pid": os.getpid(), "inst_id": "a", "ip": "x"}))
    assert vc.tunnel_info() is None  # alive, but it's pytest, not our ssh


def test_serve_ensure_after_preemption_reprovisions_and_keeps_base_url(sandbox, cloud, monkeypatch):
    st = _vol_state()
    st["serve"] = {"id": "old", "created": 0}
    vc.save_state(st)
    up = mock.Mock()
    monkeypatch.setattr(vc, "cmd_serve_up", up)
    vc.cmd_serve_ensure(_serve_args(wait=False))  # instance list is empty: preempted
    up.assert_called_once()
    assert "serve" not in vc.load_state()
    assert any(r["id"] == "old" and r["event"] == "delete" for r in ledger(sandbox))
    assert vc.load_env()["QWEN_BASE_URL"] == "http://10.0.2.2:18000/v1"


def test_serve_ensure_healthy_new_box_repoints_tunnel_same_url(sandbox, cloud, monkeypatch):
    st = _vol_state()
    st["serve"] = {"id": "new", "created": 0}
    vc.save_state(st)
    cloud.instances.get.return_value = [inst("new", ip="5.6.7.8")]
    monkeypatch.setattr(vc, "tunnel_info", lambda: {"pid": 1, "inst_id": "old", "ip": "1.2.3.4"})
    start = mock.Mock(return_value=True)
    monkeypatch.setattr(vc, "start_tunnel", start)
    monkeypatch.setattr(vc, "health", lambda u, k, **kw: u == vc.local_base_url())
    before = vc.load_env()["QWEN_BASE_URL"]
    vc.cmd_serve_ensure(_serve_args(wait=False))
    start.assert_called_once_with("new", "5.6.7.8")
    assert vc.load_env()["QWEN_BASE_URL"] == before == vc.container_base_url()
    assert cloud.deleted == []


def test_serve_up_preempted_while_booting_deletes_and_clears_state(serve_cloud, monkeypatch):
    monkeypatch.setattr(vc, "wait_running", lambda c, i: inst("g1"))
    monkeypatch.setattr(vc, "ensure_tunnel", lambda i, ip: False)  # sshd unreachable
    monkeypatch.setattr(vc, "ssh", mock.Mock(side_effect=OSError))
    serve_cloud.instances.get_by_id.return_value = inst("g1", status="discontinued")
    with pytest.raises(RuntimeError, match="discontinued"):
        vc.cmd_serve_up(_serve_args(timeout_min=40))
    assert serve_cloud.deleted == ["g1"]
    assert "serve" not in vc.load_state()


def test_cleanup_ignores_second_ctrl_c(serve_cloud, monkeypatch):
    import signal
    seen = []
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=KeyboardInterrupt))
    monkeypatch.setattr(vc, "delete_instance",
                        lambda c, i, reason="": seen.append(signal.getsignal(signal.SIGINT)))
    with pytest.raises(KeyboardInterrupt):
        vc.cmd_serve_up(_serve_args())
    assert seen == [signal.SIG_IGN]
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN


# ---- r3 (1): API key never on the vLLM command line --------------------------------
def test_vllm_key_only_in_environment_file():
    text = (vc.REMOTE / "serve.sh").read_text()
    exec_line = next(l for l in text.splitlines() if l.startswith("ExecStart="))
    assert "--api-key" not in exec_line and "API_KEY" not in exec_line
    assert "EnvironmentFile=/etc/fh-vllm.env" in text
    assert "VLLM_API_KEY=__API_KEY__" in text  # vLLM's auth middleware reads envs.VLLM_API_KEY
    assert "install -m 600 /dev/null /etc/fh-vllm.env" in text


def test_rendered_serve_script_keeps_key_out_of_execstart():
    vol = {"mount": "/mnt/fh", "size_gb": 300}
    text = vc.render("serve.sh", {"__VLLM_VERSION__": "0.33.0", "__MODEL_DIR__": "/m", "__SERVED_NAME__": "q",
                                  "__PORT__": "8000", "__VLLM_ARGS__": "--x", "__API_KEY__": "fhq-SECRET"}, vol)
    exec_line = next(l for l in text.splitlines() if l.startswith("ExecStart="))
    assert "fhq-SECRET" not in exec_line and text.count("fhq-SECRET") == 1


# ---- r3 (2): per-attempt download status ---------------------------------------------
def test_parse_download_status_ignores_other_attempts():
    assert vc.parse_download_status("done abc123 2026-01-01", "abc123") == "done"
    assert vc.parse_download_status("failed abc123 x", "abc123") == "failed"
    assert vc.parse_download_status("done OLD 2026-01-01", "abc123") == ""
    assert vc.parse_download_status("done 2026-01-01T00:00:00+00:00", "abc123") == ""  # legacy format
    assert vc.parse_download_status("", "abc123") == ""


def test_download_script_writes_attempt_scoped_status_before_installing_anything():
    text = (vc.REMOTE / "download.sh").read_text()
    assert 'STATUS="$MNT/logs/download-$ATTEMPT.status"' in text
    assert text.index('echo "running $ATTEMPT') < text.index("install_uv")
    assert 'echo "done $ATTEMPT' in text


def test_download_does_not_accept_stale_done(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state() | {"weights": {}})
    monkeypatch.setattr(vc, "create_instance", fake_create("dl1"))
    monkeypatch.setattr(vc, "wait_running", lambda c, i: inst("dl1"))
    monkeypatch.setattr(vc.secrets, "token_hex", lambda n: "NEWATTEMPT")
    seen_cmds = []

    def fake_ssh(ip, cmd, **k):
        seen_cmds.append(cmd)
        return SimpleNamespace(stdout="done OLDATTEMPT 2026-01-01\n55G\n[fh] installing uv\n")

    monkeypatch.setattr(vc, "ssh", fake_ssh)
    clock = iter(range(0, 10**6, 30))
    monkeypatch.setattr(vc.time, "time", lambda: next(clock))
    with pytest.raises(SystemExit, match="timeout"):
        vc.cmd_download(_download_args())
    assert vc.load_state()["weights"] == {}  # stale 'done' did not mark weights available
    assert cloud.deleted == ["dl1"]
    assert "download-NEWATTEMPT.status" in seen_cmds[0]


def test_download_accepts_own_done(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state() | {"weights": {}})
    monkeypatch.setattr(vc, "create_instance", fake_create("dl1"))
    monkeypatch.setattr(vc, "wait_running", lambda c, i: inst("dl1"))
    monkeypatch.setattr(vc.secrets, "token_hex", lambda n: "NEWATTEMPT")
    monkeypatch.setattr(vc, "ssh", lambda ip, cmd, **k: SimpleNamespace(stdout="done NEWATTEMPT 2026\n"))
    vc.cmd_download(_download_args())
    st = vc.load_state()
    assert set(st["weights"]) == set(vc.MODELS) and "download" not in st


def test_download_replaces_box_without_recorded_attempt(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    cloud.instances.get.return_value = [inst("stray", hostname="fh-qwen-dl-1")]
    monkeypatch.setattr(vc, "create_instance", fake_create("dl1"))
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=TimeoutError))
    with pytest.raises(TimeoutError):
        vc.cmd_download(_download_args())
    assert cloud.deleted[0] == "stray" and "dl1" in cloud.deleted


# ---- r3 (3): creation intent + reconcile ---------------------------------------------
def _real_create_cloud(monkeypatch):
    c = mock.Mock()
    c.startup_scripts.get.return_value = []
    c.startup_scripts.create.return_value = SimpleNamespace(id="sc")
    monkeypatch.setattr(vc, "ensure_ssh_key", lambda c, s: "key")
    monkeypatch.setattr(vc, "wait_volume_free", lambda c, v: None)
    monkeypatch.setattr(vc, "eur_prices", lambda c: {})
    return c


def test_create_intent_is_persisted_before_the_api_call(sandbox, monkeypatch):
    c = _real_create_cloud(monkeypatch)
    snapshot = {}

    def create(**kw):
        snapshot.update(vc.load_state())  # what is on disk at the moment the order goes out
        raise ConnectionError("response lost")

    c.instances.create.side_effect = create
    c.instances.get.side_effect = ConnectionError("still down")  # cleanup lookup fails too
    st = _vol_state()
    with pytest.raises(ConnectionError):
        vc.create_instance(c, st, role="download", itype="CPU.8V.32G", hostname="fh-qwen-dl-9",
                           script_name="x", script="s", spot=False)
    rec = snapshot["pending_creates"]["fh-qwen-dl-9"]
    assert rec["role"] == "download" and rec["type"] == "CPU.8V.32G" and rec["pid"] == vc.os.getpid()
    assert "fh-qwen-dl-9" in vc.load_state()["pending_creates"]  # survives the failure
    # the in-process cleanup can't reach the API either: intent must stay recorded
    vc.delete_by_hostnames(c, ["fh-qwen-dl-9"], reason="t")
    assert "fh-qwen-dl-9" in vc.load_state()["pending_creates"]


def test_create_clean_rejection_forgets_intent(sandbox, monkeypatch):
    c = _real_create_cloud(monkeypatch)
    c.instances.create.side_effect = APIException("not_enough_resources", "no capacity")
    c.instances.get.return_value = []
    with pytest.raises(APIException):
        vc.create_instance(c, _vol_state(), role="serve", itype="1A6000.10V", hostname="h1",
                           script_name="x", script="s", spot=True)
    assert "h1" not in vc.load_state().get("pending_creates", {})


def test_create_success_tracks_role_and_clears_intent(sandbox, monkeypatch):
    c = _real_create_cloud(monkeypatch)
    c.instances.create.return_value = inst("new1", hostname="h1")
    st = _vol_state()
    vc.create_instance(c, st, role="serve", itype="1A6000.10V", hostname="h1",
                       script_name="x", script="s", spot=True)
    disk = vc.load_state()
    assert disk["serve"]["id"] == "new1" and not disk["pending_creates"]
    assert ledger(sandbox)[0]["id"] == "new1"


def _dead_owner():
    return {"pid": 999999999, "pid_start": "1"}


@pytest.fixture
def recon(monkeypatch):
    c = mock.Mock()
    deleted = []
    monkeypatch.setattr(vc, "delete_instance", lambda c, i, reason="": deleted.append(i))
    c.deleted = deleted
    return c


def test_reconcile_deletes_orphaned_download_create(sandbox, recon):
    vc.save_state({"pending_creates": {"fh-qwen-dl-9": {"role": "download", "ts": 0, **_dead_owner()}}})
    recon.instances.get.return_value = [inst("cpu1", hostname="fh-qwen-dl-9")]
    out = vc.reconcile(recon)
    assert recon.deleted == ["cpu1"] and out["deleted"] == ["cpu1"]
    assert "pending_creates" not in vc.load_state()
    assert any(r["event"] == "create" and r["id"] == "cpu1" for r in ledger(sandbox))  # cost is accounted


def test_reconcile_adopts_orphaned_serve_create_when_none_tracked(sandbox, recon):
    vc.save_state({"pending_creates": {"fh-qwen-serve-9": {"role": "serve", "ts": 5, **_dead_owner()}}})
    recon.instances.get.return_value = [inst("g9", hostname="fh-qwen-serve-9")]
    out = vc.reconcile(recon)
    assert out["adopted"] == ["g9"] and recon.deleted == []
    assert vc.load_state()["serve"]["id"] == "g9"


def test_reconcile_deletes_second_serve_box(sandbox, recon):
    vc.save_state({"serve": {"id": "g1"},
                   "pending_creates": {"fh-qwen-serve-9": {"role": "serve", "ts": 5, **_dead_owner()}}})
    recon.instances.get.return_value = [inst("g1"), inst("g9", hostname="fh-qwen-serve-9")]
    vc.reconcile(recon)
    assert recon.deleted == ["g9"] and vc.load_state()["serve"]["id"] == "g1"


def test_reconcile_leaves_in_flight_creates_alone(sandbox, recon):
    vc.save_state({"pending_creates": {"h": {"role": "download", "ts": 0, **vc.proc_identity()}}})
    recon.instances.get.return_value = [inst("x", hostname="h")]
    out = vc.reconcile(recon)
    assert recon.deleted == [] and out["unknown"] == []
    assert "h" in vc.load_state()["pending_creates"]


def test_reconcile_drops_stale_intent_that_never_materialized(sandbox, recon):
    vc.save_state({"pending_creates": {"old": {"role": "serve", "ts": 0, **_dead_owner()},
                                       "recent": {"role": "serve", "ts": vc.time.time(), **_dead_owner()}}})
    recon.instances.get.return_value = []
    out = vc.reconcile(recon)
    assert out["dropped"] == ["old"] and list(vc.load_state()["pending_creates"]) == ["recent"]


def test_reconcile_deletes_orphaned_download_box_but_not_kept_one(sandbox, recon):
    vc.save_state({"download": {"id": "d1", "attempt": "a", **_dead_owner()}})
    recon.instances.get.return_value = [inst("d1", hostname="fh-qwen-dl-1")]
    vc.reconcile(recon)
    assert recon.deleted == ["d1"] and "download" not in vc.load_state()
    recon.deleted.clear()
    vc.save_state({"download": {"id": "d1", "attempt": "a", "keep": True, **_dead_owner()}})
    vc.reconcile(recon)
    assert recon.deleted == []


def test_reconcile_flags_unknown_instances_without_deleting(sandbox, recon, capsys):
    vc.save_state({"serve": {"id": "g1"}})
    recon.instances.get.return_value = [inst("g1"), inst("zz", hostname="someone-elses-box")]
    out = vc.reconcile(recon)
    assert out["unknown"] == ["zz"] and recon.deleted == []
    assert "not tracked by state.json" in capsys.readouterr().out


def test_reconcile_survives_api_outage(sandbox, recon):
    vc.save_state({"pending_creates": {"h": {"role": "serve", "ts": 0, **_dead_owner()}}})
    recon.instances.get.side_effect = ConnectionError
    vc.reconcile(recon)
    assert "h" in vc.load_state()["pending_creates"]


def test_main_runs_reconcile_before_cloud_commands(sandbox, monkeypatch):
    vc.save_state({"initial_balance": "x"})
    order = []
    monkeypatch.setattr(vc, "client", mock.Mock())
    monkeypatch.setattr(vc, "reconcile", lambda c: order.append("reconcile"))
    monkeypatch.setattr(vc, "cmd_instances", lambda a: order.append("cmd"))
    monkeypatch.setattr(sys, "argv", ["verda_cli.py", "instances"])
    vc.main()
    assert order == ["reconcile", "cmd"]


def test_serve_ensure_reconciles_under_its_lock(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    calls = []
    monkeypatch.setattr(vc, "reconcile", lambda c: calls.append(vc.LOCK_PATH.read_text()))
    monkeypatch.setattr(vc, "cmd_serve_up", lambda a: None)
    vc.cmd_serve_ensure(_serve_args(wait=False))
    assert calls and json.loads(calls[0])["pid"] == vc.os.getpid()


# ---- r3 (4): finite HTTP timeouts and a lock that survives a hung holder --------------
def test_verda_sdk_auth_and_api_calls_get_a_timeout(monkeypatch):
    import requests
    from verda import VerdaClient

    seen = []

    def send(self, request, **kw):
        seen.append((request.url, kw.get("timeout")))
        raise requests.ConnectionError("blocked in test")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    vc.install_http_timeouts()
    with pytest.raises(requests.ConnectionError):
        VerdaClient("id", "secret")  # authenticates in the constructor (requests.post)
    assert seen and all(t == vc.HTTP_TIMEOUT for _, t in seen)
    seen.clear()
    with pytest.raises(requests.ConnectionError):
        requests.request("GET", "https://api.verda.com/v1/instances")  # what the SDK's HTTPClient uses
    with pytest.raises(requests.ConnectionError):
        requests.get("https://x.invalid", timeout=3)  # explicit timeouts are kept
    assert [t for _, t in seen] == [vc.HTTP_TIMEOUT, 3]


def _spawn_lock_holder(path, deadline):
    import subprocess as sp
    code = ("import fcntl,json,os,sys,time\n"
            f"f=open({str(path)!r},'a+'); fcntl.flock(f, fcntl.LOCK_EX)\n"
            "st=open(f'/proc/{os.getpid()}/stat').read().rsplit(')',1)[1].split()[19]\n"
            f"f.seek(0); f.truncate(); f.write(json.dumps({{'pid':os.getpid(),'pid_start':st,'deadline':{deadline}}})); f.flush()\n"
            "print('locked', flush=True); time.sleep(120)\n")
    p = sp.Popen([sys.executable, "-c", code], stdout=sp.PIPE, text=True)
    assert p.stdout.readline().strip() == "locked"
    return p


def test_lock_takes_over_from_hung_holder(sandbox, monkeypatch):
    p = _spawn_lock_holder(vc.LOCK_PATH, deadline=0)  # deadline long past
    try:
        with vc.file_lock(vc.LOCK_PATH, ttl_s=60):
            assert json.loads(vc.LOCK_PATH.read_text())["pid"] == vc.os.getpid()
        assert p.wait(timeout=20) is not None  # the hung holder was terminated
    finally:
        if p.poll() is None:
            p.kill()


def test_lock_respects_holder_within_deadline(sandbox):
    p = _spawn_lock_holder(vc.LOCK_PATH, deadline=vc.time.time() + 3600)
    try:
        with pytest.raises(SystemExit) as ei:
            with vc.file_lock(vc.LOCK_PATH, ttl_s=60):
                pass
        assert ei.value.code == 0 and p.poll() is None
    finally:
        p.kill()


def test_lock_released_when_holder_dies(sandbox):
    p = _spawn_lock_holder(vc.LOCK_PATH, deadline=vc.time.time() + 3600)
    p.kill()
    p.wait()
    with vc.file_lock(vc.LOCK_PATH, ttl_s=60):
        pass


def test_deadline_interrupts_a_stuck_call():
    t0 = vc.time.time()
    with pytest.raises(vc.Deadline):
        with vc.deadline(1):
            while vc.time.time() - t0 < 10:
                pass
    assert vc.time.time() - t0 < 5


def test_reconcile_clears_serve_record_of_vanished_instance(sandbox, recon):
    vc.save_state({"serve": {"id": "gone1", "hostname": "fh-qwen-serve-1"}, "pending_creates": {}})
    recon.instances.get.return_value = []
    out = vc.reconcile(recon)
    st = vc.load_state()
    assert "serve" not in st and "pending_creates" not in st and out["dropped"] == ["gone1"]
    assert [r["event"] for r in ledger(sandbox)] == ["delete"]
    vc.save_state({"serve": {"id": "gone1"}})  # already closed in the ledger: no duplicate record
    vc.reconcile(recon)
    assert len(ledger(sandbox)) == 1


def test_reconcile_keeps_serve_record_of_live_instance(sandbox, recon):
    vc.save_state({"serve": {"id": "g1"}})
    recon.instances.get.return_value = [inst("g1")]
    vc.reconcile(recon)
    assert vc.load_state()["serve"]["id"] == "g1"


# ---- harbor-agent JobConfig fragment and LoRA serving ----------------------------------
def _harbor_cfg(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["verda_cli.py", "harbor-agent", *argv])
    vc.save_state({})  # no initial_balance: a cloud command would call client() here
    monkeypatch.setattr(vc, "client", mock.Mock(side_effect=AssertionError("harbor-agent must stay local")))
    vc.main()
    return json.loads(capsys.readouterr().out)


def test_harbor_agent_prints_jobconfig_fragment_locally(sandbox, monkeypatch, capsys):
    cfg = _harbor_cfg(monkeypatch, capsys)
    assert list(cfg) == ["agents"] and len(cfg["agents"]) == 1
    a = cfg["agents"][0]
    assert a["name"] == "opencode" and a["model_name"] == "verda-qwen/qwen3.8-27b"
    assert a["base_url"] == "http://10.0.2.2:18000/v1" and a["model_api_key_name"] == "QWEN_API_KEY"
    assert "version" not in a["kwargs"]
    model = a["kwargs"]["opencode_config"]["provider"]["verda-qwen"]["models"]["qwen3.8-27b"]
    assert model["modalities"]["input"] == ["text", "image"] and model["attachment"] is True


def test_harbor_agent_lora_name_and_opencode_version(sandbox, monkeypatch, capsys):
    cfg = _harbor_cfg(monkeypatch, capsys, "--served-name", "qwen3.8-27b-fh", "--opencode-version", "1.19.5")
    a = cfg["agents"][0]
    assert a["model_name"] == "verda-qwen/qwen3.8-27b-fh" and a["kwargs"]["version"] == "1.19.5"
    models = a["kwargs"]["opencode_config"]["provider"]["verda-qwen"]["models"]
    base = json.loads((vc.HERE / "opencode-qwen.json").read_text())["provider"]["verda-qwen"]["models"]["qwen3.8-27b"]
    assert list(models) == ["qwen3.8-27b-fh"]
    assert {k: v for k, v in models["qwen3.8-27b-fh"].items() if k != "name"} == \
        {k: v for k, v in base.items() if k != "name"}  # same capabilities as the base model


def test_harbor_agent_rejects_unknown_served_name(sandbox, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["verda_cli.py", "harbor-agent", "--served-name", "other"])
    with pytest.raises(SystemExit):
        vc.main()


def test_vllm_args_lora():
    a = vc.vllm_args("bf16", [], False, "/mnt/fh/lora/fh-v1")
    i = a.index("--lora-modules")
    assert a[i + 1] == "qwen3.8-27b-fh=/mnt/fh/lora/fh-v1"
    assert "--enable-lora" in a and a[a.index("--max-lora-rank") + 1] == "16"
    assert "--enable-lora" not in vc.vllm_args("bf16", [], False)
    with pytest.raises(SystemExit):
        vc.vllm_args("bf16", [], False, "/mnt/x; rm -rf /")


def test_health_requires_lora_model_when_served(monkeypatch):
    resp = SimpleNamespace(status_code=200, json=lambda: {"data": [{"id": "qwen3.8-27b"}]})
    monkeypatch.setattr(vc.httpx, "get", lambda *a, **k: resp)
    assert vc.health("u", "k")
    assert not vc.health("u", "k", names=vc.served_names("/mnt/fh/lora"))


def test_serve_up_lora_is_rendered_and_remembered(serve_cloud, monkeypatch):
    rendered = {}
    monkeypatch.setattr(vc, "render", lambda f, subs, vol: rendered.update(subs) or "script")
    vc.cmd_serve_up(_serve_args(no_wait=True, lora="/mnt/fh/lora/fh-v1"))
    assert "--lora-modules qwen3.8-27b-fh=/mnt/fh/lora/fh-v1" in rendered["__VLLM_ARGS__"]
    st = vc.load_state()
    assert st["serve"]["lora"] == "/mnt/fh/lora/fh-v1" and st["serve_lora"] == "/mnt/fh/lora/fh-v1"


def test_serve_ensure_reprovisions_with_remembered_lora(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state() | {"serve_lora": "/mnt/fh/lora/fh-v1"})
    up = mock.Mock()
    monkeypatch.setattr(vc, "cmd_serve_up", up)
    vc.cmd_serve_ensure(_serve_args(wait=False))
    assert up.call_args[0][0].lora == "/mnt/fh/lora/fh-v1"
