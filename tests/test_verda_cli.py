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
                           instance_type="1A6000.10V", is_spot=True, price_per_hour=0.4)


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
    monkeypatch.setattr(vc, "create_instance", lambda *a, **k: inst("dl1", hostname=k["hostname"]))
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
    monkeypatch.setattr(vc, "create_instance", lambda *a, **k: inst("dl1", hostname=k["hostname"]))
    monkeypatch.setattr(vc, "wait_running", mock.Mock(side_effect=TimeoutError()))
    with pytest.raises(TimeoutError):
        vc.cmd_download(SimpleNamespace(**{**vars(_download_args()), "keep": True}))
    assert cloud.deleted == []


def _serve_args(**kw):
    d = dict(type=None, variant=None, on_demand=False, mtp=False, timeout_min=0.01, keep=False, no_wait=False)
    d.update(kw)
    return SimpleNamespace(**d)


@pytest.fixture
def serve_cloud(cloud, monkeypatch, sandbox):
    vc.save_state(_vol_state())
    monkeypatch.setattr(vc, "pick_candidate", lambda c, a: [(0.4, "1A6000.10V", "fp8", [], True)])
    monkeypatch.setattr(vc, "create_instance", lambda *a, **k: inst("g1", hostname=k["hostname"]))
    return cloud


def test_serve_up_health_timeout_deletes_new_instance(serve_cloud, monkeypatch):
    monkeypatch.setattr(vc, "wait_running", lambda c, i: inst("g1"))
    monkeypatch.setattr(vc, "ensure_tunnel", lambda i, ip: True)
    monkeypatch.setattr(vc, "health", lambda u, k: False)
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
    monkeypatch.setattr(vc, "health", lambda u, k: u == vc.local_base_url())
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
