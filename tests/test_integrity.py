"""Reward integrity: secret seed derivation, sidecar evidence dump, separate verifier, task layout."""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
import tomllib
from pathlib import Path

import pytest

from framehunt import splits
from framehunt.evidence import dump
from framehunt.server import FrameHuntEnv, FrameHuntError, build_mcp
from framehunt.verify import reward_from_result, verify

ROOT = Path(__file__).resolve().parent.parent
SALT = "a" * 64


# ------------------------------------------------------------- seed derivation
def test_derive_seed_deterministic_and_keyed():
    s = splits.derive_seed("eval", "logstream-7001", SALT)
    assert s == splits.derive_seed("eval", "logstream-7001", SALT)
    assert s == splits.derive_seed("eval", "logstream-7001", SALT.encode())
    assert 0 <= s < 2**64
    assert s != splits.derive_seed("eval", "logstream-7001", "b" * 64)
    assert s != splits.derive_seed("dev", "logstream-7001", SALT)
    assert s != splits.derive_seed("eval", "logstream-7002", SALT)
    # The public seed is not the scene seed.
    assert s != 7001


def test_derive_seed_known_value():
    import hashlib
    import hmac
    mac = hmac.new(SALT.encode(), b"eval:dialogs-7101", hashlib.sha256).digest()
    assert splits.derive_seed("eval", "dialogs-7101", SALT) == int.from_bytes(mac[:8], "big")


def test_derive_seed_rejects_empty_salt():
    with pytest.raises(ValueError):
        splits.derive_seed("eval", "x", "")


def test_derived_seeds_distinct_per_split():
    for name, items in splits.SPLITS.items():
        seeds = {splits.derive_seed(name, i, SALT) for i, _, _ in items}
        assert len(seeds) == len(items)


def test_load_salt_env_then_file_then_create(tmp_path):
    assert splits.load_salt("eval", secrets_dir=tmp_path, environ={}) is None
    assert not (tmp_path / "eval.salt").exists()
    created = splits.load_salt("eval", create=True, secrets_dir=tmp_path, environ={})
    assert len(created) == 64 and int(created, 16) >= 0
    p = tmp_path / "eval.salt"
    assert p.read_text().strip() == created
    assert (p.stat().st_mode & 0o777) == 0o600
    # file is stable across calls; env var wins over the file
    assert splits.load_salt("eval", create=True, secrets_dir=tmp_path, environ={}) == created
    assert splits.load_salt("eval", secrets_dir=tmp_path, environ={"FH_SALT_EVAL": " envsalt\n"}) == "envsalt"
    assert splits.load_salt("dev", secrets_dir=tmp_path, environ={}) is None
    (tmp_path / "dev.salt").write_text("\n")
    with pytest.raises(ValueError):
        splits.load_salt("dev", secrets_dir=tmp_path, environ={})


def test_secrets_dir_is_gitignored():
    assert "secrets/" in (ROOT / ".gitignore").read_text().split()


# --------------------------------------------------------------------- verifier
def test_reward_from_result_strict():
    assert reward_from_result({"submitted": True, "correct": True}) == 1
    for bad in [None, [], "x", {}, {"submitted": True, "correct": False}, {"submitted": False, "correct": True},
                {"submitted": True, "correct": "true"}, {"submitted": 1, "correct": 1},
                {"reward": 1.0}, {"submitted": True, "correct": True}.items()]:
        assert reward_from_result(bad) == 0, bad


def _write_evidence(d: Path, result, log="") -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(result if isinstance(result, str) else json.dumps(result))
    (d / "calls.jsonl").write_text(log)
    return d


def test_verify_correct_and_copies_evidence(tmp_path):
    ev = _write_evidence(tmp_path / "ev", {"submitted": True, "answer": "K", "correct": True, "reward": 1.0},
                         '{"i": 0}\n')
    out = tmp_path / "out"
    assert verify(ev, out) == 1
    assert (out / "reward.txt").read_text().strip() == "1"
    assert json.loads((out / "result.json").read_text())["correct"] is True
    assert (out / "calls.jsonl").read_text() == '{"i": 0}\n'


@pytest.mark.parametrize("result", [
    {"submitted": False, "answer": None, "correct": False, "reward": 0.0},
    {"submitted": True, "answer": "X", "correct": False, "reward": 1.0},
    "not json", "", "[1]", '{"submitted": true, "correct": "true"}',
])
def test_verify_zero_cases(tmp_path, result):
    ev = _write_evidence(tmp_path / "ev", result)
    out = tmp_path / "out"
    assert verify(ev, out) == 0
    assert (out / "reward.txt").read_text().strip() == "0"


def test_verify_missing_evidence_scores_zero_and_clears_stale_rewards(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    for name in ("reward.txt", "reward.json", "result.json"):
        (out / name).write_text("1")
    assert verify(tmp_path / "nope", out) == 0
    assert (out / "reward.txt").read_text().strip() == "0"
    assert not (out / "reward.json").exists()
    assert not (out / "result.json").exists()


def test_verify_ignores_symlinked_result(tmp_path):
    real = _write_evidence(tmp_path / "real", {"submitted": True, "correct": True})
    ev = tmp_path / "ev"
    ev.mkdir()
    (ev / "result.json").symlink_to(real / "result.json")
    assert verify(ev, tmp_path / "out") == 0


# ------------------------------------------------- evidence dump from a live server
class _Meta:
    width, height, fps, n_frames = 320, 180, 10, 100
    duration_s = 10.0


class _Scene:
    kind, seed, meta = "stub", 1, _Meta()
    question, answer, answer_format = "q?", "K7Q2ZP", "6 chars"

    def render(self, frame):
        from PIL import Image
        return Image.new("RGB", (320, 180))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live():
    import uvicorn

    servers = []

    def start(token="tok-ev"):
        env = FrameHuntEnv(_Scene(), budget_px=10_000, verify_token=token)
        port = _free_port()
        app = build_mcp(env, host="127.0.0.1", port=port).streamable_http_app()
        srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        th = threading.Thread(target=srv.run, daemon=True)
        th.start()
        for _ in range(200):
            if srv.started:
                break
            time.sleep(0.05)
        servers.append((srv, th))
        return env, f"http://127.0.0.1:{port}"

    yield start
    for srv, th in servers:
        srv.should_exit = True
        th.join(5)


def test_evidence_dump_then_verify(live, tmp_path):
    env, base = live()
    ev = tmp_path / "ev"
    assert dump(ev, base=base, token="tok-ev")["submitted"] is False
    assert verify(ev, tmp_path / "o0") == 0  # no submission

    env.view(1.0, out_width=32)
    env.submit("k7q2zp")
    with pytest.raises(FrameHuntError):
        env.submit("WRONG")  # refused; the first submission counts
    ev1 = tmp_path / "ev1"
    res = dump(ev1, base=base, token="tok-ev")
    assert res["correct"] is True
    assert len((ev1 / "calls.jsonl").read_text().splitlines()) == 3
    assert verify(ev1, tmp_path / "o1") == 1


def test_evidence_dump_wrong_first_submission(live, tmp_path):
    env, base = live()
    env.submit("WRONG")
    with pytest.raises(FrameHuntError):
        env.submit("K7Q2ZP")
    ev = tmp_path / "ev"
    dump(ev, base=base, token="tok-ev")
    assert verify(ev, tmp_path / "o") == 0


def test_evidence_dump_bad_token_writes_nothing(live, tmp_path):
    _env, base = live()
    ev = tmp_path / "ev"
    with pytest.raises(Exception):
        dump(ev, base=base, token="wrong")
    assert not (ev / "result.json").exists()
    assert verify(ev, tmp_path / "o") == 0
    with pytest.raises(RuntimeError):
        dump(ev, base=base, token="")


def test_evidence_cli_uses_env_token(live, tmp_path, monkeypatch):
    import framehunt.evidence as evmod

    _env, base = live()
    monkeypatch.setenv("VERIFY_TOKEN", "tok-ev")
    assert evmod.main([str(tmp_path / "ev"), base]) == 0
    assert json.loads((tmp_path / "ev" / "result.json").read_text())["submitted"] is False
    monkeypatch.setenv("VERIFY_TOKEN", "nope")
    assert evmod.main([str(tmp_path / "ev2"), base]) == 1
    assert not (tmp_path / "ev2" / "result.json").exists()


# --------------------------------------------------------------- task layout
def _load_build_tasks():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_tasks
    finally:
        sys.path.pop(0)
    return build_tasks


def test_built_task_uses_separate_verifier_and_secret_seed(tmp_path):
    bt = _load_build_tasks()
    from framehunt.scenes import make_scene

    item_id, kind, public_seed = splits.EVAL[0]
    d = bt.build_item(tmp_path, "eval", item_id, kind, public_seed, 500_000, salt=SALT)
    secret = splits.derive_seed("eval", item_id, SALT)

    cfg = tomllib.loads((d / "task.toml").read_text())
    assert "seed" not in cfg["metadata"]
    assert cfg["metadata"]["item_id"] == item_id
    v = cfg["verifier"]
    assert v["environment_mode"] == "separate"
    assert v["collect"] == [{"service": "fh-mcp", "command": "python -m framehunt.evidence /fh-evidence",
                             "timeout_sec": 60.0}]
    assert {"source": "/fh-evidence", "service": "fh-mcp"} in cfg["artifacts"]

    compose = (d / "environment" / "docker-compose.yaml").read_text()
    assert f'FH_SEED: "{secret}"' in compose

    # The verifier image is self-contained and trusts only the collected evidence.
    assert (d / "tests" / "Dockerfile").is_file()
    assert (d / "tests" / "verify.py").read_text() == (ROOT / "framehunt" / "verify.py").read_text()
    test_sh = (d / "tests" / "test.sh").read_text()
    assert "/fh-evidence" in test_sh and "kill" not in test_sh and "curl" not in test_sh

    # Neither the secret seed, the token nor the answer appears anywhere the agent
    # (or a solution) could read: instruction.md, task.toml, tests/, solution/.
    scene = make_scene(kind, secret)
    token = compose.split('VERIFY_TOKEN: "')[1].split('"')[0]
    for p in [d / "instruction.md", d / "task.toml", *(d / "tests").rglob("*"), *(d / "solution").rglob("*")]:
        if p.is_file():
            text = p.read_text(errors="replace")
            assert str(secret) not in text, p
            assert token not in text, p
            assert scene.answer not in text, p
    # The instruction describes the secret-seeded scene, not the public-seed one.
    assert scene.question in (d / "instruction.md").read_text()
    assert make_scene(kind, public_seed).answer != scene.answer


def test_build_main_uses_env_salt(tmp_path, monkeypatch):
    bt = _load_build_tasks()
    monkeypatch.setenv("FH_SALT_EVAL", SALT)
    item_id = splits.EVAL[10][0]
    assert bt.main(["--split", "eval", "--out", str(tmp_path), "--only", item_id]) == 0
    compose = (tmp_path / item_id / "environment" / "docker-compose.yaml").read_text()
    assert f'FH_SEED: "{splits.derive_seed("eval", item_id, SALT)}"' in compose


def test_sidecar_copy_has_no_secrets(tmp_path):
    bt = _load_build_tasks()
    item_id, kind, public_seed = splits.EVAL[20]
    d = bt.build_item(tmp_path, "eval", item_id, kind, public_seed, 500_000, salt=SALT)
    names = {p.name for p in (d / "environment").rglob("*")}
    assert not any(n.endswith(".salt") for n in names)
    assert "evidence.py" in names  # the collect hook runs this inside the sidecar
