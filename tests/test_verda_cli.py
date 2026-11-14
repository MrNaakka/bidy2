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
                        ("KNOWN_HOSTS", "known_hosts"), ("STATE_LOCK", ".state.lock")]:
        monkeypatch.setattr(vc, name, tmp_path / fname)
    monkeypatch.setattr(vc, "ROOT", tmp_path)
    monkeypatch.setattr(vc.time, "sleep", lambda s: None)
    (tmp_path / ".env").write_text("QWEN_API_KEY=k\nQWEN_BASE_URL=http://10.0.2.2:18000/v1\n")
    return tmp_path


def ledger(tmp_path) -> list[dict]:
    p = tmp_path / "ledger.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def inst(id="i1", status="running", hostname="fhbot-serve-c4ca4238a0b9", ip="1.2.3.4"):
    return SimpleNamespace(id=id, status=status, hostname=hostname, ip=ip, os_volume_id=None,
                           instance_type="1A6000.10V", is_spot=True, price_per_hour=0.4, location="FIN-01")


def fake_create(inst_id):
    """Stand-in for create_instance that tracks the box in state like the real one."""
    def create(c, state, **k):
        rec = {"id": inst_id, "hostname": k["hostname"], "created": 0, **(k.get("record") or {})}
        vc.update_state(lambda st: st.__setitem__(k["role"], rec))
        state[k["role"]] = rec
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
    gone = []
    c.instances.get.side_effect = lambda: [] if gone else [inst()]
    c.instances.action.side_effect = lambda *a, **k: gone.append(1)
    c.instances.get_by_id.side_effect = lambda i: (_ for _ in ()).throw(NOT_FOUND) if gone else inst()
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
    monkeypatch.setattr(vc, "_serve_up", up)
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
    monkeypatch.setattr(vc, "ssh", lambda ip, cmd, **k: SimpleNamespace(
        stdout="".join(f"{r} {rev}\n" for r, rev in vc.MODELS.items()) if ".verified" in cmd
        else "done NEWATTEMPT 2026\n"))
    vc.cmd_download(_download_args())
    st = vc.load_state()
    assert set(st["weights"]) == set(vc.MODELS) and "download" not in st


def test_download_replaces_box_without_recorded_attempt(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    cloud.instances.get.return_value = [inst("stray", hostname="fhbot-dl-c4ca4238a0b9")]
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
    vc.save_state({"pending_creates": {"fhbot-dl-45c48cce2e2d": {"role": "download", "ts": 0, **_dead_owner()}}})
    recon.instances.get.return_value = [inst("cpu1", hostname="fhbot-dl-45c48cce2e2d")]
    out = vc.reconcile(recon)
    assert recon.deleted == ["cpu1"] and out["deleted"] == ["cpu1"]
    assert "pending_creates" not in vc.load_state()
    assert any(r["event"] == "create" and r["id"] == "cpu1" for r in ledger(sandbox))  # cost is accounted


def test_reconcile_adopts_orphaned_serve_create_when_none_tracked(sandbox, recon):
    vc.save_state({"pending_creates": {"fhbot-serve-45c48cce2e2d": {"role": "serve", "ts": 5, **_dead_owner()}}})
    recon.instances.get.return_value = [inst("g9", hostname="fhbot-serve-45c48cce2e2d")]
    out = vc.reconcile(recon)
    assert out["adopted"] == ["g9"] and recon.deleted == []
    assert vc.load_state()["serve"]["id"] == "g9"


def test_reconcile_deletes_second_serve_box(sandbox, recon):
    vc.save_state({"serve": {"id": "g1"},
                   "pending_creates": {"fhbot-serve-45c48cce2e2d": {"role": "serve", "ts": 5, **_dead_owner()}}})
    recon.instances.get.return_value = [inst("g1"), inst("g9", hostname="fhbot-serve-45c48cce2e2d")]
    vc.reconcile(recon)
    assert recon.deleted == ["g9"] and vc.load_state()["serve"]["id"] == "g1"


def test_reconcile_leaves_in_flight_creates_alone(sandbox, recon):
    vc.save_state({"pending_creates": {"fhbot-dl-2510c39011c5": {"role": "download", "ts": 0, **vc.proc_identity()}}})
    recon.instances.get.return_value = [inst("x", hostname="fhbot-dl-2510c39011c5")]
    out = vc.reconcile(recon)
    assert recon.deleted == [] and out["unknown"] == []
    assert "fhbot-dl-2510c39011c5" in vc.load_state()["pending_creates"]


def test_reconcile_deletes_orphaned_download_box_but_not_kept_one(sandbox, recon):
    vc.save_state({"download": {"id": "d1", "attempt": "a", **_dead_owner()}})
    recon.instances.get.return_value = [inst("d1", hostname="fhbot-dl-c4ca4238a0b9")]
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
    assert "not ours" in capsys.readouterr().out


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
    monkeypatch.setattr(vc, "_serve_up", lambda c, a: None)
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
    vc.save_state({"serve": {"id": "gone1", "hostname": "fhbot-serve-c4ca4238a0b9"}, "pending_creates": {}})
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
    monkeypatch.setattr(vc, "_serve_up", up)
    vc.cmd_serve_ensure(_serve_args(wait=False))
    assert up.call_args[0][1].lora == "/mnt/fh/lora/fh-v1"


# ---- r5 (1): unmatched intents are never cleared by an empty listing -----------------
def test_cleanup_clears_intent_only_after_confirmed_delete(sandbox, monkeypatch):
    vc.save_state({"pending_creates": {"h": {"role": "download", "ts": 0}}})
    c = mock.Mock()
    c.instances.get.return_value = [inst("x", hostname="h")]
    monkeypatch.setattr(vc, "delete_instance", mock.Mock(side_effect=vc.DeleteFailed("api down")))
    with pytest.raises(vc.DeleteFailed):
        vc.delete_by_hostnames(c, ["h"], reason="t")
    assert "h" in vc.load_state()["pending_creates"]
    monkeypatch.setattr(vc, "delete_instance", mock.Mock())
    vc.delete_by_hostnames(c, ["h"], reason="t")
    assert "pending_creates" not in vc.load_state()


def test_late_instance_for_kept_intent_is_cleaned_by_reconcile(sandbox, recon):
    vc.save_state({"pending_creates": {"fhbot-dl-2510c39011c5": {"role": "download", "ts": vc.time.time(), **_dead_owner()}}})
    recon.instances.get.return_value = []
    vc.reconcile(recon)  # nothing yet: kept
    recon.instances.get.return_value = [inst("late", hostname="fhbot-dl-2510c39011c5")]  # the order went through after all
    out = vc.reconcile(recon)
    assert recon.deleted == ["late"] and out["unknown"] == [] and "pending_creates" not in vc.load_state()


# ---- r5 (2): serialized state transactions, no stale snapshots -------------------------
def test_reconcile_does_not_erase_intent_written_during_its_listing(sandbox, recon):
    vc.save_state({"serve": {"id": "g1", "created": 0}})

    def listing():  # another command records an intent while reconcile waits on the API
        vc.update_state(lambda st: st.setdefault("pending_creates", {}).__setitem__(
            "fhbot-dl-8f14e45fceea", {"role": "download", "ts": vc.time.time(), **vc.proc_identity()}))
        return [inst("g1")]

    recon.instances.get.side_effect = listing
    vc.reconcile(recon)
    assert "fhbot-dl-8f14e45fceea" in vc.load_state()["pending_creates"]


def test_reconcile_keeps_record_created_after_its_listing(sandbox, recon):
    vc.save_state({})

    def listing():  # serve-up records a new box after the listing request went out
        vc.update_state(lambda st: st.__setitem__("serve", {"id": "new", "created": vc.time.time() + 1}))
        return []

    recon.instances.get.side_effect = listing
    vc.reconcile(recon)
    assert vc.load_state()["serve"]["id"] == "new"


def test_commands_never_save_full_snapshots():
    src = Path(vc.__file__).read_text()
    assert src.count("save_state(") == 1  # only the definition; all writers use update_state


def test_nested_state_transaction_is_refused(sandbox):
    with pytest.raises(RuntimeError, match="nested"):
        vc.update_state(lambda st: vc.update_state(lambda s2: None))


def test_state_lock_serializes_processes(sandbox):
    import subprocess as sp
    vc.save_state({"a": 1})
    code = ("import fcntl,json,sys,time\n"
            f"f=open({str(vc.STATE_LOCK)!r},'a'); fcntl.flock(f, fcntl.LOCK_EX)\n"
            f"p={str(vc.STATE_PATH)!r}; s=json.load(open(p)); print('locked', flush=True); time.sleep(1.0)\n"
            "s['other']=True; json.dump(s, open(p,'w'))\n")
    proc = sp.Popen([sys.executable, "-c", code], stdout=sp.PIPE, text=True)
    assert proc.stdout.readline().strip() == "locked"
    import time as _t
    t0 = _t.monotonic()
    vc.update_state(lambda st: st.__setitem__("mine", True))  # must wait for the other writer
    assert _t.monotonic() - t0 > 0.5
    proc.wait(timeout=10)
    st = vc.load_state()
    assert st["other"] is True and st["mine"] is True and st["a"] == 1


# ---- r5 (3): download attempts are tied to their requested repos + revisions -----------
FP8 = "Qwen/Qwen3.8-27B-FP8"
BF16 = "Qwen/Qwen3.8-27B"


def test_parse_verified_only_matching_request():
    req = {FP8: vc.MODELS[FP8]}
    text = f"{FP8} {vc.MODELS[FP8]}\n{BF16} {vc.MODELS[BF16]}\n{FP8} deadbeef\n"
    assert vc.parse_verified(text, req) == {FP8: vc.MODELS[FP8]}


def test_kept_box_for_other_request_is_refused_not_reused(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state() | {"download": {"id": "dlk", "attempt": "OLD", "keep": True,
                                               "request": {FP8: vc.MODELS[FP8]}, **_dead_owner()}})
    cloud.instances.get.return_value = [inst("dlk", hostname="fhbot-dl-c4ca4238a0b9")]
    create = mock.Mock()
    monkeypatch.setattr(vc, "create_instance", create)
    with pytest.raises(SystemExit, match="different request"):
        vc.cmd_download(_download_args())  # requests both checkpoints
    create.assert_not_called()
    assert cloud.deleted == []  # the kept box is left alone
    assert vc.load_state()["weights"] == _vol_state()["weights"]


def test_matching_kept_box_is_reused(sandbox, cloud, monkeypatch):
    req = {FP8: vc.MODELS[FP8]}
    vc.save_state(_vol_state() | {"weights": {}, "download": {"id": "dlk", "attempt": "OLD", "keep": True,
                                                              "request": req, **_dead_owner()}})
    cloud.instances.get.return_value = [inst("dlk", hostname="fhbot-dl-c4ca4238a0b9")]
    monkeypatch.setattr(vc, "create_instance", mock.Mock(side_effect=AssertionError("must reuse")))
    monkeypatch.setattr(vc, "wait_running", lambda c, i: inst("dlk"))
    monkeypatch.setattr(vc, "ssh", lambda ip, cmd, **k: SimpleNamespace(
        stdout=f"{FP8} {vc.MODELS[FP8]}\n" if ".verified" in cmd else "done OLD 2026\n"))
    vc.cmd_download(SimpleNamespace(**{**vars(_download_args()), "repos": [FP8]}))
    assert vc.load_state()["weights"] == {FP8: {"revision": vc.MODELS[FP8], "path": "/mnt/fh/models/Qwen3.8-27B-FP8"}}


def test_download_records_only_verified_and_merges(sandbox, cloud, monkeypatch):
    prev = {FP8: {"revision": vc.MODELS[FP8], "path": "/mnt/fh/models/Qwen3.8-27B-FP8"}}
    vc.save_state(_vol_state() | {"weights": prev})
    monkeypatch.setattr(vc, "create_instance", fake_create("dl1"))
    monkeypatch.setattr(vc, "wait_running", lambda c, i: inst("dl1"))
    monkeypatch.setattr(vc.secrets, "token_hex", lambda n: "A1")
    seen = {}

    def fake_ssh(ip, cmd, **k):
        if ".verified" in cmd:
            return SimpleNamespace(stdout="")  # 'done' but nothing verified: don't trust it
        return SimpleNamespace(stdout="done A1 2026\n")

    monkeypatch.setattr(vc, "ssh", fake_ssh)
    monkeypatch.setattr(vc, "render", lambda f, subs, vol: seen.update(subs) or "s")
    with pytest.raises(SystemExit, match="verified only"):
        vc.cmd_download(SimpleNamespace(**{**vars(_download_args()), "repos": [BF16]}))
    assert seen["__MODELS__"] == f"{BF16} {vc.MODELS[BF16]}"
    assert vc.load_state()["weights"] == prev  # BF16 not recorded, FP8 kept


def test_download_record_carries_request(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    rec = {}
    monkeypatch.setattr(vc, "create_instance", lambda c, st, **k: rec.update(k["record"]) or inst("d"))
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=TimeoutError))
    with pytest.raises(TimeoutError):
        vc.cmd_download(SimpleNamespace(**{**vars(_download_args()), "repos": [FP8]}))
    assert rec["request"] == {FP8: vc.MODELS[FP8]} and rec["attempt"]


def test_download_script_records_only_verified_checkpoints():
    text = (vc.REMOTE / "download.sh").read_text()
    assert 'VERIFIED="$MNT/logs/download-$ATTEMPT.verified"' in text
    loop = text[text.index("while read -r repo rev"):text.index("__MODELS__")]
    assert loop.index('rm -f "$MNT/models/$name.REVISION"') < loop.index("hf download")
    assert loop.index('>> "$VERIFIED"') > loop.index("verify_weights.py")


# ---- r5 (6): serving config persisted before create, carried into adoption ----------
def test_serve_up_remembers_lora_before_create(serve_cloud, monkeypatch):
    at_call = {}

    def create(c, state, **k):
        at_call.update(vc.load_state())
        at_call["record"] = k["record"]
        raise KeyboardInterrupt

    monkeypatch.setattr(vc, "create_instance", create)
    with pytest.raises(KeyboardInterrupt):
        vc.cmd_serve_up(_serve_args(lora="/mnt/fh/lora/fh-v1"))
    assert at_call["serve_lora"] == "/mnt/fh/lora/fh-v1" and at_call["record"]["lora"] == "/mnt/fh/lora/fh-v1"


def test_adopted_orphan_keeps_lora_for_recovery(sandbox, recon, cloud, monkeypatch):
    cfg = {"variant": "bf16", "lora": "/mnt/fh/lora/fh-v1", "vllm_args": ["--enable-lora"]}
    vc.save_state(_vol_state() | {"pending_creates": {
        "fhbot-serve-45c48cce2e2d": {"role": "serve", "ts": 5, "record": cfg, **_dead_owner()}}})
    recon.instances.get.return_value = [inst("g9", hostname="fhbot-serve-45c48cce2e2d")]
    vc.reconcile(recon)
    st = vc.load_state()
    assert st["serve"]["lora"] == cfg["lora"] and st["serve"]["vllm_args"] == cfg["vllm_args"]
    assert st["serve_lora"] == cfg["lora"]
    # the adopted box is then preempted: serve-ensure re-provisions with the adapter
    up = mock.Mock()
    monkeypatch.setattr(vc, "_serve_up", up)
    monkeypatch.setattr(vc, "reconcile", lambda c: None)
    vc.cmd_serve_ensure(_serve_args(wait=False))
    assert up.call_args[0][1].lora == cfg["lora"]


# ---- r6: unique hostnames, direct create POST, fh- sweep, one lifecycle lock -------------
def _post_cloud(monkeypatch, post):
    c = _real_create_cloud(monkeypatch)
    c._http_client.post.side_effect = post
    c.instances.create.side_effect = AssertionError("must not use the SDK's polling create")
    return c


def test_hostnames_are_unique_and_prefixed():
    names = {vc.new_hostname("download") for _ in range(1000)}
    assert len(names) == 1000 and all(vc.OWN_HOSTNAME_RE.match(n) and n.startswith("fhbot-dl-") for n in names)
    assert vc._role_of(vc.new_hostname("serve")) == "serve"


def test_r6_1_fallback_attempts_keep_independent_intents(sandbox, monkeypatch):
    # attempt 1: response lost (ambiguous); attempt 2 in the same second: rejected by the API
    calls = []

    def post(url, json):
        calls.append(json["hostname"])
        if len(calls) == 1:
            raise ConnectionError("response lost")
        raise APIException("invalid_request", "Not enough resources")

    c = _post_cloud(monkeypatch, post)
    h1, h2 = vc.new_hostname("download"), vc.new_hostname("download")
    with pytest.raises(ConnectionError):
        vc.create_instance(c, _vol_state(), role="download", itype="CPU.8V.32G", hostname=h1,
                           script_name="x", script="s", spot=True)
    with pytest.raises(APIException):
        vc.create_instance(c, _vol_state(), role="download", itype="CPU.8V.32G", hostname=h2,
                           script_name="x", script="s", spot=False)
    assert list(vc.load_state()["pending_creates"]) == [h1]  # only the rejected one is gone


def test_r6_1_late_instance_without_any_intent_is_swept(sandbox, recon):
    vc.save_state({})  # tracking was lost entirely
    recon.instances.get.return_value = [inst("late", hostname="fhbot-dl-0123456789ab")]
    out = vc.reconcile(recon)
    assert recon.deleted == ["late"] and out["unknown"] == []


def test_r6_2_id_is_persisted_from_the_post_before_any_polling(sandbox, monkeypatch):
    c = _post_cloud(monkeypatch, lambda url, json: SimpleNamespace(text="inst-123"))
    c.instances.get_by_id.side_effect = APIException("unauthorized_request", "credentials revoked")
    cfg = {"variant": "bf16", "lora": "/mnt/fh/lora/a"}
    got = vc.create_instance(c, _vol_state(), role="serve", itype="1A100.22V", hostname="fhbot-serve-900150983cd2",
                             script_name="x", script="s", spot=True, record=cfg)
    st = vc.load_state()
    assert got.id == "inst-123" and st["serve"]["id"] == "inst-123" and st["serve"]["lora"] == "/mnt/fh/lora/a"
    assert "pending_creates" not in st and ledger(sandbox)[0]["id"] == "inst-123"
    c.instances.get_by_id.assert_not_called()
    with pytest.raises(APIException):  # a later polling error ...
        vc.wait_running(c, got.id)
    assert vc.load_state()["serve"]["id"] == "inst-123"  # ... does not untrack the accepted instance


def test_r6_2_intent_with_config_is_saved_before_the_post(sandbox, monkeypatch):
    seen = {}

    def post(url, json):
        seen.update(vc.load_state())
        raise ConnectionError("lost")

    c = _post_cloud(monkeypatch, post)
    cfg = {"variant": "bf16", "vllm_args": ["--enable-lora"], "lora": "/mnt/fh/lora/a"}
    with pytest.raises(ConnectionError):
        vc.create_instance(c, _vol_state(), role="serve", itype="1A100.22V", hostname="fhbot-serve-900150983cd2",
                           script_name="x", script="s", spot=True, record=cfg)
    rec = seen["pending_creates"]["fhbot-serve-900150983cd2"]
    assert rec["record"] == cfg and rec["pid"] == vc.os.getpid()
    assert "fhbot-serve-900150983cd2" in vc.load_state()["pending_creates"]  # unknown outcome: kept


def test_r6_2_post_rejection_drops_intent(sandbox, monkeypatch):
    c = _post_cloud(monkeypatch, mock.Mock(side_effect=APIException("server_error", "boom")))
    with pytest.raises(APIException):
        vc.create_instance(c, _vol_state(), role="serve", itype="1A100.22V", hostname="fhbot-serve-900150983cd2",
                           script_name="x", script="s", spot=True)
    assert "pending_creates" not in vc.load_state()


def test_r6_3_serve_ensure_leaves_box_of_live_creator_alone(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state() | {"pending_creates": {
        "fhbot-serve-d0dbe915091d": {"role": "serve", "ts": vc.time.time(), "record": {}, **vc.proc_identity()}}})
    cloud.instances.get.return_value = [inst("ordered1", status="ordered", hostname="fhbot-serve-d0dbe915091d", ip=None)]
    up = mock.Mock()
    monkeypatch.setattr(vc, "_serve_up", up)
    vc.cmd_serve_ensure(_serve_args(wait=False))
    assert cloud.deleted == [] and up.call_count == 0


def test_r6_3_lifecycle_commands_are_mutually_exclusive(sandbox, monkeypatch):
    holder = _spawn_lock_holder(vc.LOCK_PATH, deadline=vc.time.time() + 3600)
    monkeypatch.setattr(vc, "client", mock.Mock(side_effect=AssertionError("must not run")))
    try:
        for fn, args, code in ((vc.cmd_serve_up, _serve_args(), 1),
                               (vc.cmd_serve_down, SimpleNamespace(), 1),
                               (vc.cmd_serve_ensure, _serve_args(wait=False), 0)):
            with pytest.raises(SystemExit) as ei:
                fn(args)
            assert ei.value.code == code
    finally:
        holder.kill()


def test_intent_lifecycle_rules(sandbox, recon):
    now = vc.time.time()
    vc.save_state({"serve": {"id": "g1", "hostname": "fhbot-serve-9c3d7e347f8b", "created": 0},
                   "pending_creates": {
                       "fhbot-serve-9c3d7e347f8b": {"role": "serve", "ts": now, **_dead_owner()},   # tracked -> drop
                       "fhbot-dl-149603e6c035": {"role": "download", "ts": now - 2 * vc.PENDING_TTL_S, **_dead_owner()},  # never appeared -> drop
                       "fhbot-dl-8154f2ab3669": {"role": "download", "ts": now - 60, **_dead_owner()},  # may still appear -> keep
                       "fhbot-dl-7bd3d5f5b3cd": {"role": "download", "ts": 0, **vc.proc_identity()}}})  # creator alive -> keep
    recon.instances.get.return_value = [inst("g1", hostname="fhbot-serve-9c3d7e347f8b")]
    vc.reconcile(recon)
    assert sorted(vc.load_state()["pending_creates"]) == ["fhbot-dl-7bd3d5f5b3cd", "fhbot-dl-8154f2ab3669"]
    assert recon.deleted == []


def test_r7_1_sweep_only_exact_generated_names(sandbox, recon):
    vc.save_state({"serve": {"id": "g1", "created": 0}})
    stray = "fhbot-serve-0123456789ab"
    recon.instances.get.return_value = [
        inst("g1"), inst("stray", hostname=stray), inst("stray-dl", hostname="fhbot-dl-abcdefabcdef"),
        inst("manual", hostname="fh-training"), inst("legacy", hostname="fh-qwen-serve-1794618791"),
        inst("old-r6", hostname="fh-serve-0123456789ab"), inst("near", hostname="fhbot-serve-0123456789AB"),
        inst("longer", hostname="fhbot-dl-0123456789abc"), inst("foreign", hostname="teammate-box")]
    out = vc.reconcile(recon)
    assert sorted(recon.deleted) == ["stray", "stray-dl"]
    assert sorted(out["unknown"]) == ["foreign", "legacy", "longer", "manual", "near", "old-r6"]


def test_r7_1_untracked_manual_fh_box_survives_serve_ensure(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state())
    cloud.instances.get.return_value = [inst("manual", hostname="fh-training")]
    monkeypatch.setattr(vc, "_serve_up", mock.Mock())
    vc.cmd_serve_ensure(_serve_args(wait=False))
    assert cloud.deleted == []


def test_r7_2_serve_down_deletes_tracked_legacy_instance(sandbox, monkeypatch):
    vc.save_state({"serve": {"id": "old1", "hostname": "fh-qwen-serve-1794618791", "created": 0}})
    c = mock.Mock()
    legacy = inst("old1", hostname="fh-qwen-serve-1794618791")
    gone = []
    c.instances.get.side_effect = lambda: [] if gone else [legacy, inst("manual", hostname="fh-serve-manual")]
    c.startup_scripts.get.return_value = []
    monkeypatch.setattr(vc, "client", lambda: c)
    monkeypatch.setattr(vc, "stop_tunnel", mock.Mock())
    deleted = []
    monkeypatch.setattr(vc, "delete_instance", lambda c, i, reason="": deleted.append(i) or gone.append(i))
    vc.cmd_serve_down(SimpleNamespace())
    assert deleted == ["old1"]  # tracked by id despite the legacy name; the untracked fh- box is left
    assert "serve" not in vc.load_state()


def test_r7_2_role_instances_select_by_tracked_id(sandbox):
    vc.save_state({"serve": {"id": "old1"}, "download": {"id": "dl-old"}})
    c = mock.Mock()
    c.instances.get.return_value = [inst("old1", hostname="fh-qwen-serve-1"), inst("dl-old", hostname="fh-dl-x"),
                                    inst("new", hostname="fhbot-serve-0123456789ab"),
                                    inst("other", hostname="fh-serve-y")]
    assert [i.id for i in vc.role_instances(c, "serve")] == ["old1", "new"]
    assert [i.id for i in vc.role_instances(c, "download")] == ["dl-old"]


def test_r7_2_serve_ensure_handles_tracked_legacy_box(sandbox, cloud, monkeypatch):
    st = _vol_state()
    st["serve"] = {"id": "old1", "hostname": "fh-qwen-serve-1", "created": vc.time.time()}
    vc.save_state(st)
    cloud.instances.get.return_value = [inst("old1", hostname="fh-qwen-serve-1", ip="9.9.9.9")]
    monkeypatch.setattr(vc, "ensure_tunnel", lambda i, ip: True)
    monkeypatch.setattr(vc, "health", lambda u, k, **kw: True)
    up = mock.Mock()
    monkeypatch.setattr(vc, "_serve_up", up)
    vc.cmd_serve_ensure(_serve_args(wait=False))
    assert up.call_count == 0 and cloud.deleted == []  # recognised as the healthy serving box


# ---- r8: shutdown cancels pending serve creates; late arrivals are deleted ----------------
def test_r8_late_serve_box_after_serve_down_is_deleted_not_adopted(sandbox, monkeypatch, capsys):
    hn = "fhbot-serve-0123456789ab"
    # a serve-up lost its create response and died; cleanup and serve-down list nothing
    vc.save_state({"pending_creates": {hn: {"role": "serve", "ts": vc.time.time(),
                                            "record": {"lora": "/x"}, **_dead_owner()}}})
    c = mock.Mock()
    late = []
    c.instances.get.side_effect = lambda: list(late)
    c.startup_scripts.get.return_value = []
    monkeypatch.setattr(vc, "client", lambda: c)
    monkeypatch.setattr(vc, "stop_tunnel", mock.Mock())
    vc.cmd_serve_down(SimpleNamespace())
    st = vc.load_state()
    assert st["serve_desired"] is False and st["pending_creates"][hn]["cancelled"] is True
    assert "1 creates unresolved" in capsys.readouterr().out
    # the GPU shows up later; any command's reconcile must delete it, not adopt it
    late.append(inst("gpu", hostname=hn))
    deleted = []
    monkeypatch.setattr(vc, "delete_instance", lambda c, i, reason="": deleted.append(i))
    out = vc.reconcile(c)
    assert deleted == ["gpu"] and out["adopted"] == [] and "serve" not in vc.load_state()


def test_r8_orphan_without_intent_not_adopted_while_serving_off(sandbox, recon):
    vc.save_state({"serve_desired": False})
    recon.instances.get.return_value = [inst("gpu", hostname="fhbot-serve-abcdefabcdef")]
    out = vc.reconcile(recon)
    assert recon.deleted == ["gpu"] and out["adopted"] == []


def test_r8_serve_ensure_idle_while_serving_off(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state() | {"serve_desired": False})
    up = mock.Mock()
    monkeypatch.setattr(vc, "_serve_up", up)
    vc.cmd_serve_ensure(_serve_args(wait=False))
    up.assert_not_called()


def test_r8_serve_up_switches_serving_back_on(sandbox, cloud, monkeypatch):
    vc.save_state(_vol_state() | {"serve_desired": False})
    seen = []
    monkeypatch.setattr(vc, "_serve_up", lambda c, a: seen.append(vc.load_state()["serve_desired"]))
    vc.cmd_serve_up(_serve_args(no_wait=True))
    assert seen == [True]


# ---- LoRA training on Verda ------------------------------------------------------------------
def _train_state(**extra):
    st = _vol_state() | {"serve_desired": False}
    st["weights"]["Qwen/Qwen3.8-27B"] = {"path": "/mnt/fh/models/Qwen3.8-27B"}
    return st | extra


def _run(**kw):
    r = {"run": "r1", "trainer_args": ["--model", "/m", "--out", "/mnt/fh/runs/r1", "--epochs", "1.0"],
         "data_dir": "/d", "data_name": "sft-v1", "data_sha": "D", "code_sha": "C", "req_hash": "R",
         "tokens": 12.9e6, "max_eur": 40.0, "desired": True, "restarts": 0}
    r.update(kw)
    return r


def test_train_plan_prefers_spot_then_cheapest_expected_cost():
    prices = {"1H100.80S.30V": {"spot_price": "2.341", "price_per_hour": "4.683"},
              "1A100.22V": {"spot_price": "1.071", "price_per_hour": "2.142"}}
    opts = vc.train_plan(prices, {"1H100.80S.30V", "1A100.22V"}, {"1A100.22V"}, 12.9e6)
    assert [(o["type"], o["spot"]) for o in opts] == [("1H100.80S.30V", True), ("1A100.22V", True),
                                                       ("1A100.22V", False)]
    h100 = opts[0]
    assert h100["hours"] == round(vc.TRAIN_SETUP_H + 12.9e6 / 1500 / 3600, 2)
    assert h100["expected_eur"] == round(2.341 * (vc.TRAIN_SETUP_H + 12.9e6 / 1500 / 3600) * vc.TRAIN_SPOT_OVERHEAD, 2)
    # no spot capacity at all: on-demand fallback
    assert [o["spot"] for o in vc.train_plan(prices, set(), {"1A100.22V"}, 12.9e6)] == [False]


def test_train_hostnames_are_swept_pattern():
    hn = vc.new_hostname("train")
    assert hn.startswith("fhbot-train-") and vc._role_of(hn) == "train"


def test_train_script_renders_run_and_venv():
    vol = {"mount": "/mnt/fh", "size_gb": 300}
    text = vc.render("train.sh", {"__RUN__": "r1", "__REQ_HASH__": "abc123abc123", "__REQS__": "torch==2.15.0",
                                  "__TRAIN_ARGS__": "--epochs 1.0 --out /mnt/fh/runs/r1"}, vol)
    assert 'VENV="$MNT/envs/train-abc123abc123"' in text and "torch==2.15.0" in text
    exec_line = next(l for l in text.splitlines() if l.startswith("ExecStart="))
    assert "sft_lora.py --epochs 1.0 --out /mnt/fh/runs/r1" in exec_line
    assert "systemctl enable" not in text and "systemctl start" not in text  # the CLI starts it after upload
    import re as _re
    assert not _re.search(r"__[A-Z_]+__", text)  # every placeholder substituted


def test_trainer_args_rejects_shell_metacharacters():
    a = SimpleNamespace(epochs=1.0, lr=1e-4, grad_accum=8, lora_r=16, save_every=10, trainer_arg=["--keep=3"])
    args = vc.trainer_args("r1", "/m", "sft-v1", a)
    assert args[args.index("--data") + 1] == "/mnt/fh/data/sft-v1" and args[-1] == "--keep=3"
    a.trainer_arg = ["--x;rm -rf /"]
    with pytest.raises(SystemExit):
        vc.trainer_args("r1", "/m", "sft-v1", a)


def test_parse_train_status_and_eta():
    losses = "\n".join("loss=" + json.dumps({"step": s, "loss": 1.0 / s, "seconds": 60.0}) for s in range(1, 11))
    st = vc.parse_train_status("ready=1\nservice=active\nexit=0\ndone=0\nfatal=0\ntotal_steps=110\n"
                               'eval={"step": 10, "eval_loss": 0.5}\n' + losses)
    assert st["ready"] and st["service"] == "active" and st["step"] == 10 and st["total_steps"] == 110
    assert st["eta_h"] == round(100 * 60 / 3600, 2) and st["eval"]["eval_loss"] == 0.5
    empty = vc.parse_train_status("ready=0\nservice=inactive\ndone=0\nfatal=0\ntotal_steps=\n")
    assert empty["step"] == 0 and empty["eta_h"] is None


def test_sft_lora_writes_done_marker_after_final_adapter():
    src = (Path(__file__).resolve().parents[1] / "train" / "sft_lora.py").read_text()
    assert src.index("model.save_pretrained(final)") < src.index('os.replace(tmp, a.out / "DONE")')


@pytest.fixture
def train_cloud(cloud, monkeypatch, sandbox):
    monkeypatch.setattr(vc, "data_sha", lambda d: "D")
    monkeypatch.setattr(vc, "train_code_sha", lambda: "C")
    monkeypatch.setattr(vc, "train_req_hash", lambda: "R")
    monkeypatch.setattr(vc, "epoch_tokens", lambda d: 12_900_000)
    return cloud


def _up_args(**kw):
    d = dict(run="r1", data_dir="/tmp/sft-v1", epochs=1.0, lr=1e-4, grad_accum=8, lora_r=16, save_every=10,
             trainer_arg=None, type=None, on_demand=False, max_eur=40.0, timeout_min=1, no_wait=True, keep=False)
    d.update(kw)
    return SimpleNamespace(**d)


def test_train_up_refused_while_serving_is_on_or_tracked(train_cloud, monkeypatch):
    prov = mock.Mock()
    monkeypatch.setattr(vc, "_train_provision", prov)
    vc.save_state(_train_state(serve_desired=True))
    with pytest.raises(SystemExit, match="serve-down first"):
        vc.cmd_train_up(_up_args())
    st = _train_state()
    st.pop("serve_desired")  # default is on
    vc.save_state(st)
    with pytest.raises(SystemExit, match="serve-down first"):
        vc.cmd_train_up(_up_args())
    vc.save_state(_train_state(serve={"id": "g1", "created": vc.time.time() + 100}))
    train_cloud.instances.get.return_value = [inst("g1")]
    with pytest.raises(SystemExit, match="still tracked"):
        vc.cmd_train_up(_up_args())
    prov.assert_not_called()


def test_train_up_persists_run_and_provisions(train_cloud, monkeypatch):
    vc.save_state(_train_state())
    seen = {}
    monkeypatch.setattr(vc, "_train_provision", lambda c, run, **k: seen.update(run) or "t1")
    vc.cmd_train_up(_up_args(epochs=2.0, lr=5e-5))
    run = vc.load_state()["train_run"]
    assert run["desired"] and run["run"] == "r1" and run["tokens"] == 2 * 12_900_000
    a = run["trainer_args"]
    assert a[a.index("--epochs") + 1] == "2.0" and a[a.index("--lr") + 1] == "5e-05"
    assert a[a.index("--out") + 1] == "/mnt/fh/runs/r1" and a[a.index("--model") + 1] == "/mnt/fh/models/Qwen3.8-27B"
    assert seen["trainer_args"] == a


def test_train_up_same_run_different_args_refused(train_cloud, monkeypatch):
    vc.save_state(_train_state())
    monkeypatch.setattr(vc, "_train_provision", lambda c, run, **k: "t1")
    vc.cmd_train_up(_up_args())
    vc.update_state(lambda s: s["train_run"].update(desired=False))
    with pytest.raises(SystemExit, match="different"):
        vc.cmd_train_up(_up_args(lr=3e-4))


def test_train_budget_guard(sandbox, monkeypatch):
    vc.save_state(_train_state())
    c = mock.Mock()
    monkeypatch.setattr(vc, "eur_prices", lambda c: {"1H100.80S.30V": {"spot_price": "2.341", "price_per_hour": "4.683"}})
    monkeypatch.setattr(vc, "_availability", lambda c, loc: ({"1H100.80S.30V"}, set()))
    monkeypatch.setattr(vc, "ledger_total_eur", lambda: 35.0)
    create = mock.Mock()
    monkeypatch.setattr(vc, "create_instance", create)
    with pytest.raises(SystemExit, match="budget"):
        vc._train_provision(c, _run(max_eur=40.0))  # 35 + ~7.8 > 40
    create.assert_not_called()
    monkeypatch.setattr(vc, "ledger_total_eur", lambda: 5.0)
    monkeypatch.setattr(vc, "render", lambda f, subs, vol: subs)
    create.return_value = SimpleNamespace(id="t1")
    assert vc._train_provision(c, _run(max_eur=40.0)) == "t1"
    k = create.call_args.kwargs
    assert k["role"] == "train" and k["hostname"].startswith("fhbot-train-") and k["spot"] is True
    assert k["record"]["run"] == "r1" and "--epochs 1.0" in k["script"]["__TRAIN_ARGS__"]


def _tick_env(monkeypatch, status_text, *, box_extra=None, run_extra=None):
    st = _train_state(train_run=_run(**(run_extra or {})),
                      train={"id": "t1", "hostname": "fhbot-train-0123456789ab", "created": vc.time.time(),
                             **(box_extra or {})})
    vc.save_state(st)
    c = mock.Mock()
    c.instances.get_by_id.return_value = inst("t1", hostname="fhbot-train-0123456789ab", ip="5.5.5.5")
    calls = {"ssh": [], "deleted": [], "upload": 0, "fetch": 0}

    def fake_ssh(ip, cmd, **k):
        calls["ssh"].append(cmd)
        if "ready=" in cmd:
            return SimpleNamespace(stdout=status_text, stderr="", returncode=0)
        return SimpleNamespace(stdout="active", stderr="", returncode=0)

    monkeypatch.setattr(vc, "ssh", fake_ssh)
    monkeypatch.setattr(vc, "train_upload", lambda ip, alias, run: calls.__setitem__("upload", calls["upload"] + 1))
    monkeypatch.setattr(vc, "train_fetch_from", lambda ip, alias, run: calls.__setitem__("fetch", 1) or vc.ROOT / "train/out/r1")
    monkeypatch.setattr(vc, "delete_instance", lambda c, i, reason="": calls["deleted"].append(i))
    return c, calls


READY_IDLE = "ready=1\nservice=inactive\nexit=\ndone=0\nfatal=0\ntotal_steps=\n"


def test_tick_waits_for_box_setup(sandbox, monkeypatch):
    c, calls = _tick_env(monkeypatch, "ready=0\nservice=inactive\ndone=0\nfatal=0\n")
    assert vc.train_tick(c) == "setup" and calls["upload"] == 0


def test_tick_uploads_and_starts_on_fresh_box(sandbox, monkeypatch):
    c, calls = _tick_env(monkeypatch, READY_IDLE)
    assert vc.train_tick(c) == "running"
    assert calls["upload"] == 1 and any("systemctl start fh-train" in x for x in calls["ssh"])
    assert vc.load_state()["train"]["started"] is True and vc.load_state()["train_run"]["restarts"] == 0


def test_tick_reports_running(sandbox, monkeypatch):
    c, calls = _tick_env(monkeypatch, "ready=1\nservice=active\ndone=0\nfatal=0\ntotal_steps=100\n"
                         + 'loss={"step": 25, "loss": 0.7, "seconds": 50}\n', box_extra={"started": True})
    assert vc.train_tick(c) == "running" and calls["upload"] == 0
    run = vc.load_state()["train_run"]
    assert run["step"] == 25 and run["progress"] == 0.25


def test_tick_restarts_crashed_trainer_then_gives_up(sandbox, monkeypatch):
    c, calls = _tick_env(monkeypatch, READY_IDLE.replace("service=inactive", "service=failed").replace("exit=", "exit=1"),
                         box_extra={"started": True})
    assert vc.train_tick(c) == "restarted" and vc.load_state()["train_run"]["restarts"] == 1
    vc.update_state(lambda s: s["train_run"].update(restarts=vc.TRAIN_MAX_RESTARTS))
    assert vc.train_tick(c) == "failed"
    st = vc.load_state()
    assert calls["deleted"] == ["t1"] and "train" not in st and st["train_run"]["desired"] is False


def test_tick_done_publishes_fetches_deletes_and_leaves_serving_flag(sandbox, monkeypatch):
    c, calls = _tick_env(monkeypatch, "ready=1\nservice=inactive\ndone=1\nfatal=0\ntotal_steps=100\n",
                         box_extra={"started": True})
    assert vc.train_tick(c) == "done"
    st = vc.load_state()
    assert calls["deleted"] == ["t1"] and calls["fetch"] == 1 and "train" not in st
    assert any("/mnt/fh/adapters/r1" in x for x in calls["ssh"])
    run = st["train_run"]
    assert run["done"] and not run["desired"] and run["adapter"] == "/mnt/fh/adapters/r1"
    assert st["serve_desired"] is False  # untouched
    assert vc.train_tick(c) == "idle"


def test_preempted_training_box_is_reprovisioned_with_same_run(sandbox, monkeypatch):
    run = _run(trainer_args=["--epochs", "2.0", "--lr", "5e-05", "--out", "/mnt/fh/runs/r1"], progress=0.5)
    vc.save_state(_train_state(train_run=run, train={"id": "t-old", "hostname": "fhbot-train-aaaaaaaaaaaa",
                                                     "created": 0}))
    c = mock.Mock()
    c.instances.get.return_value = []  # preempted
    vc.reconcile(c)
    assert "train" not in vc.load_state()
    seen = {}
    monkeypatch.setattr(vc, "_train_provision", lambda c, r, **k: seen.update(r) or "t-new")
    assert vc.train_tick(c) == "provisioned"
    assert seen["trainer_args"] == run["trainer_args"] and seen["run"] == "r1" and seen["progress"] == 0.5


def test_reprovision_budget_uses_remaining_fraction(sandbox, monkeypatch):
    vc.save_state(_train_state())
    monkeypatch.setattr(vc, "eur_prices", lambda c: {"1H100.80S.30V": {"spot_price": "2.341", "price_per_hour": "4.683"}})
    monkeypatch.setattr(vc, "_availability", lambda c, loc: ({"1H100.80S.30V"}, set()))
    monkeypatch.setattr(vc, "ledger_total_eur", lambda: 36.0)
    monkeypatch.setattr(vc, "render", lambda f, subs, vol: "s")
    monkeypatch.setattr(vc, "create_instance", lambda *a, **k: SimpleNamespace(id="t9"))
    assert vc._train_provision(mock.Mock(), _run(progress=0.9)) == "t9"  # ~€1.6 left to spend fits


def test_reconcile_adopts_train_orphan_only_for_desired_run(sandbox, recon):
    hn = "fhbot-train-0123456789ab"
    intent = {"role": "train", "ts": 5, "record": {"run": "r1"}, **_dead_owner()}
    vc.save_state(_train_state(train_run=_run(), pending_creates={hn: intent}))
    recon.instances.get.return_value = [inst("t1", hostname=hn)]
    out = vc.reconcile(recon)
    assert out["adopted"] == ["t1"] and vc.load_state()["train"]["run"] == "r1"
    vc.save_state(_train_state(train_run=_run(desired=False), pending_creates={hn: intent}))
    recon.deleted.clear()
    vc.reconcile(recon)
    assert recon.deleted == ["t1"]


def test_train_down_cancels_and_deletes(sandbox, monkeypatch):
    vc.save_state(_train_state(train_run=_run(), train={"id": "t1", "hostname": "fhbot-train-0123456789ab",
                                                       "created": 0},
                               pending_creates={"fhbot-train-bbbbbbbbbbbb": {"role": "train", "ts": vc.time.time(),
                                                                            **_dead_owner()}}))
    c = mock.Mock()
    gone = []
    c.instances.get.side_effect = lambda: [] if gone else [inst("t1", hostname="fhbot-train-0123456789ab")]
    monkeypatch.setattr(vc, "client", lambda: c)
    monkeypatch.setattr(vc, "delete_instance", lambda c, i, reason="": gone.append(i))
    vc.cmd_train_down(SimpleNamespace())
    st = vc.load_state()
    assert gone == ["t1"] and st["train_run"]["desired"] is False and "train" not in st
    assert st["pending_creates"]["fhbot-train-bbbbbbbbbbbb"]["cancelled"] is True


def test_serving_refused_while_training(sandbox, cloud, monkeypatch):
    vc.save_state(_train_state(train_run=_run(), serve_desired=True))
    monkeypatch.setattr(vc, "_serve_up", mock.Mock())
    with pytest.raises(SystemExit, match="training run is active"):
        vc.cmd_serve_up(_serve_args(no_wait=True))
    vc.cmd_serve_ensure(_serve_args(wait=False))
    vc._serve_up.assert_not_called()


def test_train_upload_only_when_needed_and_refuses_changed_local(sandbox, monkeypatch):
    run = _run()
    uploads = []
    monkeypatch.setattr(vc, "_upload_dir", lambda ip, a, src, members, dest, sha: uploads.append(dest))
    monkeypatch.setattr(vc, "ssh", lambda ip, cmd, **k: SimpleNamespace(stdout="D\nC\n", returncode=0))
    vc.train_upload("ip", "t1", run)
    assert uploads == []
    monkeypatch.setattr(vc, "ssh", lambda ip, cmd, **k: SimpleNamespace(stdout="\n\n", returncode=0))
    monkeypatch.setattr(vc, "data_sha", lambda d: "D")
    monkeypatch.setattr(vc, "train_code_sha", lambda: "C")
    vc.train_upload("ip", "t1", run)
    assert uploads == ["/mnt/fh/data/sft-v1", "/mnt/fh/runs/r1/code"]
    monkeypatch.setattr(vc, "data_sha", lambda d: "CHANGED")
    with pytest.raises(SystemExit, match="changed since train-up"):
        vc.train_upload("ip", "t1", run)
