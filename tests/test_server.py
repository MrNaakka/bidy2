"""Unit and in-process HTTP/MCP tests for framehunt.server, using a stub scene."""

from __future__ import annotations

import asyncio
import io
import json
import math
import socket
import threading
import time
from dataclasses import dataclass

import pytest
from PIL import Image

from framehunt.server import FrameHuntEnv, FrameHuntError, build_mcp

W, H, FPS, N = 1920, 1080, 10, 600  # 60 s


@dataclass(frozen=True)
class Meta:
    width: int = W
    height: int = H
    fps: int = FPS
    n_frames: int = N

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


class StubScene:
    """Frame f is filled with color (f % 256, f // 256, 7); a white 10x10 square at (100, 200)."""

    kind = "stub"
    seed = 1
    meta = Meta()
    question = "What is the code?"
    answer = "K7Q2ZP"
    answer_format = "6 characters, A-Z and 0-9"

    def __init__(self):
        self.calls = []
        self.fail = False

    def render(self, frame: int) -> Image.Image:
        assert 0 <= frame < N
        self.calls.append(frame)
        if self.fail:
            raise RuntimeError("render boom")
        img = Image.new("RGB", (W, H), (frame % 256, frame // 256, 7))
        img.paste((255, 255, 255), (100, 200, 110, 210))
        return img

    def debug_info(self):
        return {"secret": self.answer}


def norm(s: str) -> str:
    return "".join(s.split()).upper()


def make_env(budget=500_000, token="tok-123", **kw):
    return FrameHuntEnv(StubScene(), budget_px=budget, verify_token=token, normalize=norm, **kw)


def png_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


def png_pixel(data: bytes, xy=(0, 0)):
    return Image.open(io.BytesIO(data)).convert("RGB").getpixel(xy)


# ----------------------------------------------------------------- budget math
def test_view_default_cost_and_budget():
    env = make_env()
    png, text = env.view(12.3)
    assert png_size(png) == (320, 180)
    assert env.budget_left == 500_000 - 320 * 180
    assert env.pixels_used == 320 * 180
    assert "frame=123" in text and "cost=57600" in text and f"budget_left={env.budget_left}" in text
    assert png_pixel(png) == (123, 0, 7)


def test_view_native_crop_cost():
    env = make_env()
    png, _ = env.view(1.0, x=100, y=200, w=10, h=10, out_width=10)
    assert png_size(png) == (10, 10)
    assert png_pixel(png) == (255, 255, 255)
    assert env.pixels_used == 100


def test_snap_to_nearest_frame():
    env = make_env()
    for t, f in [(0, 0), (0.04, 0), (0.06, 1), (59.94, 599), (60.0, 599)]:
        _, text = env.view(t, out_width=8)
        assert f"frame={f} " in text, (t, text)


def test_out_width_capped_at_crop_width_no_upscale():
    env = make_env()
    png, _ = env.view(0, x=0, y=0, w=50, h=40, out_width=320)
    assert png_size(png) == (50, 40)
    assert env.pixels_used == 2000


def test_crop_clipped_to_frame():
    env = make_env()
    png, text = env.view(0, x=1900, y=1000, w=10**9, h=10**9, out_width=10**9)
    assert png_size(png) == (20, 80)
    assert "w=20, h=80" in text
    assert env.pixels_used == 1600


def test_contact_sheet_cost_and_layout():
    env = make_env()
    png, text = env.contact_sheet(0, 59.9, 16, out_width=160)
    assert png_size(png) == (4 * 160 + 3 * 2, 4 * 90 + 3 * 2)
    assert env.pixels_used == 16 * 160 * 90
    frames = [int(line.split("frame=")[1]) for line in text.splitlines()[1:]]
    assert frames[0] == 0 and frames[-1] == 599 and len(frames) == 16
    assert frames == sorted(frames)
    # tile 2 sits right of tile 1 after a 2-px separator
    assert png_pixel(png, (0, 0)) == (0, 0, 7)
    assert png_pixel(png, (162, 0)) == (frames[1] % 256, frames[1] // 256, 7)
    assert png_pixel(png, (160, 0)) == (128, 128, 128)


def test_contact_sheet_non_square_grid():
    env = make_env()
    png, _ = env.contact_sheet(0, 10, 5, out_width=100)
    # 5 tiles -> 3 cols x 2 rows; separators free
    assert png_size(png) == (3 * 100 + 2 * 2, 2 * 56 + 2)
    assert env.pixels_used == 5 * 100 * 56


# --------------------------------------------------------- refusals, no charge
BAD_VIEWS = [
    dict(t=-0.1), dict(t=float("nan")), dict(t=float("inf")), dict(t=-float("inf")), dict(t=60.01),
    dict(t=1e308), dict(t=10**400), dict(t="5"), dict(t=None), dict(t=True),
    dict(t=0, x=-1), dict(t=0, y=-5), dict(t=0, x=1920), dict(t=0, y=1080), dict(t=0, x=10**30),
    dict(t=0, w=0), dict(t=0, h=0), dict(t=0, w=-10), dict(t=0, h=-1), dict(t=0, x=1915),
    dict(t=0, out_width=7), dict(t=0, out_width=0), dict(t=0, out_width=-320), dict(t=0, out_width=1.5),
    dict(t=0, x=0.5), dict(t=0, w=float("nan")), dict(t=0, out_width=float("inf")),
]


@pytest.mark.parametrize("kw", BAD_VIEWS)
def test_bad_view_args_refused_without_charge(kw):
    env = make_env()
    with pytest.raises(FrameHuntError):
        env.view(**kw)
    assert env.pixels_used == 0
    assert env.scene.calls == []
    log = [json.loads(line) for line in env.log_jsonl().splitlines()]
    assert log[-1]["ok"] is False and log[-1]["cost"] == 0


BAD_SHEETS = [
    dict(t_start=0, t_end=10, n=1), dict(t_start=0, t_end=10, n=17), dict(t_start=0, t_end=10, n=10**12),
    dict(t_start=0, t_end=10, n=-3), dict(t_start=5, t_end=5, n=2), dict(t_start=10, t_end=5, n=2),
    dict(t_start=0, t_end=0.2, n=4), dict(t_start=-1, t_end=5, n=2), dict(t_start=0, t_end=61, n=2),
    dict(t_start=0, t_end=float("nan"), n=2), dict(t_start=0, t_end=10, n=4, out_width=4),
    dict(t_start=0, t_end=10, n=2.5),
]


@pytest.mark.parametrize("kw", BAD_SHEETS)
def test_bad_sheet_args_refused_without_charge(kw):
    env = make_env()
    with pytest.raises(FrameHuntError):
        env.contact_sheet(**kw)
    assert env.pixels_used == 0


def test_over_budget_refused_without_charge_and_without_render():
    env = make_env(budget=1000)
    with pytest.raises(FrameHuntError, match="nothing was charged"):
        env.view(0, out_width=320)  # 57600 > 1000
    assert env.budget_left == 1000
    assert env.scene.calls == []
    png, _ = env.view(0, x=0, y=0, w=40, h=25, out_width=40)  # exactly 1000
    assert env.budget_left == 0
    with pytest.raises(FrameHuntError):
        env.view(0, w=8, h=1, out_width=8)
    with pytest.raises(FrameHuntError):
        env.contact_sheet(0, 10, 2, w=8, h=8, out_width=8)
    assert env.budget_left == 0


def test_render_failure_charges_nothing_and_server_survives():
    env = make_env()
    env.scene.fail = True
    with pytest.raises(RuntimeError):
        env.view(3)
    assert env.pixels_used == 0
    env.scene.fail = False
    env.view(3)
    assert env.pixels_used == 320 * 180


def test_concurrent_views_cannot_overspend():
    cost = 320 * 180
    env = make_env(budget=10 * cost + cost // 2)
    ok, refused = [], []
    barrier = threading.Barrier(40)

    def worker(i):
        barrier.wait()
        try:
            env.view(i * 0.5)
            ok.append(i)
        except FrameHuntError:
            refused.append(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(ok) == 10 and len(refused) == 30
    assert env.budget_left == cost // 2
    log = [json.loads(line) for line in env.log_jsonl().splitlines()]
    assert sum(r["cost"] for r in log) == 10 * cost
    assert min(r["budget_left"] for r in log) >= 0


# ------------------------------------------------------------------- submit
def test_single_submit_and_no_feedback():
    env = make_env()
    msg = env.submit(" k7q 2zp ")
    assert "recorded" in msg.lower()
    assert "correct" not in msg.lower().replace("no correctness feedback", "")
    with pytest.raises(FrameHuntError, match="already submitted"):
        env.submit("WRONG1")
    r = env.result()
    assert r == {"submitted": True, "answer": " k7q 2zp ", "correct": True, "reward": 1.0,
                 "pixels_used": 0, "n_calls": 2}


def test_wrong_then_right_still_wrong():
    env = make_env()
    env.submit("AAAAAA")
    with pytest.raises(FrameHuntError):
        env.submit("K7Q2ZP")
    assert env.result()["correct"] is False and env.result()["reward"] == 0.0
    # the refused second submission message is identical whether or not it would be right
    env2 = make_env()
    env2.submit("AAAAAA")
    with pytest.raises(FrameHuntError) as e1:
        env2.submit("BBBBBB")
    with pytest.raises(FrameHuntError) as e2:
        env.submit("K7Q2ZP")
    assert str(e1.value) == str(e2.value)


def test_submit_message_identical_for_right_and_wrong():
    a, b = make_env(), make_env()
    assert a.submit("K7Q2ZP") == b.submit("ZZZZZZ")


@pytest.mark.parametrize("bad", ["", "   ", "x" * 257, None, 123, ["K7Q2ZP"]])
def test_invalid_submit_not_recorded(bad):
    env = make_env()
    with pytest.raises(FrameHuntError):
        env.submit(bad)
    assert env.result()["submitted"] is False
    env.submit("K7Q2ZP")
    assert env.result()["correct"] is True


def test_concurrent_submits_only_one_wins():
    env = make_env()
    barrier = threading.Barrier(20)
    wins = []

    def worker(i):
        barrier.wait()
        try:
            env.submit("K7Q2ZP" if i % 2 else f"WRONG{i}")
            wins.append(i)
        except FrameHuntError:
            pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(wins) == 1
    assert env.result()["correct"] == (wins[0] % 2 == 1)


def test_no_submit_reward_zero():
    env = make_env()
    env.view(1)
    r = env.result()
    assert r["submitted"] is False and r["reward"] == 0.0 and r["answer"] is None and r["correct"] is False


# --------------------------------------------------------------- token, info, log
def test_token_checks():
    env = make_env(token="s3cret")
    assert env.check_token("s3cret")
    for bad in [None, "", "s3cre", "s3cret ", "S3CRET", "s3cret\x00", 123, "\udcff"]:
        assert not env.check_token(bad)
    empty = make_env(token="")
    assert not empty.check_token("")
    assert not empty.check_token(None)


def test_info_contents_and_no_leak():
    env = make_env()
    text = env.info()
    for s in ["What is the code?", "6 characters", "width: 1920", "height: 1080", "fps: 10",
              "n_frames: 600", "duration_s: 60", "budget_total: 500000", "budget_left: 500000"]:
        assert s in text
    assert "K7Q2ZP" not in text
    assert env.pixels_used == 0


def test_log_contents(tmp_path):
    clock = iter(range(100, 200))
    path = tmp_path / "calls.jsonl"
    env = make_env(clock=lambda: float(next(clock)), log_path=str(path))
    env.info()
    env.view(1.0, out_width=100)
    with pytest.raises(FrameHuntError):
        env.view(float("nan"))
    env.contact_sheet(0, 5, 4, x=10, y=10, w=200, h=100, out_width=50)
    env.submit("abc")
    log = [json.loads(line) for line in env.log_jsonl().splitlines()]
    assert [r["tool"] for r in log] == ["info", "view", "view", "contact_sheet", "submit"]
    assert [r["i"] for r in log] == [0, 1, 2, 3, 4]
    assert [r["ts"] for r in log] == [100.0, 101.0, 102.0, 103.0, 104.0]
    assert log[1]["cost"] == 100 * 56 and log[1]["frames"] == [10] and log[1]["args"]["t"] == 1.0
    assert log[2]["ok"] is False and log[2]["args"]["t"] == "nan" and "error" in log[2]
    assert log[3]["cost"] == 4 * 50 * 25 and log[3]["frames"] == [0, 17, 33, 50]
    assert log[4]["args"] == {"answer": "abc"} and "correct" not in log[4]
    assert log[-1]["budget_left"] == env.budget_left
    assert path.read_text() == env.log_jsonl()
    assert env.result()["n_calls"] == 5


# --------------------------------------------------------------- HTTP routes
def test_http_routes():
    from starlette.testclient import TestClient

    env = make_env(token="tok-abc")
    app = build_mcp(env).streamable_http_app()
    with TestClient(app) as c:
        assert c.get("/_health").text == "ok"
        for url in ["/_result", "/_result?token=", "/_result?token=tok-ab", "/_log", "/_log?token=x",
                    "/_result?token=TOK-ABC", "/_result?token=tok-abc&token=x"]:
            r = c.get(url)
            assert r.status_code == 403, url
            assert "K7Q2ZP" not in r.text
        r = c.get("/_result?token=tok-abc")
        assert r.status_code == 200
        assert r.json() == {"submitted": False, "answer": None, "correct": False, "reward": 0.0,
                            "pixels_used": 0, "n_calls": 0}
        assert c.get("/_log?token=tok-abc").text == ""
        assert c.post("/_result?token=tok-abc").status_code == 405
        assert c.get("/debug").status_code == 404


def test_http_routes_disabled_without_token():
    from starlette.testclient import TestClient

    env = make_env(token="")
    with TestClient(build_mcp(env).streamable_http_app()) as c:
        assert c.get("/_result?token=").status_code == 403
        assert c.get("/_result").status_code == 403
        assert c.get("/_log").status_code == 403


# ------------------------------------------------- MCP over streamable HTTP
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server():
    import uvicorn

    env = make_env(budget=200_000, token="tok-live")
    port = _free_port()
    app = build_mcp(env, host="127.0.0.1", port=port).streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield env, f"http://127.0.0.1:{port}"
    server.should_exit = True
    th.join(5)


def test_mcp_end_to_end(live_server):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    env, base = live_server

    async def go():
        async with streamable_http_client(base + "/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                tools = {t.name: t for t in (await s.list_tools()).tools}
                assert set(tools) == {"info", "view", "contact_sheet", "submit"}
                assert (await s.list_resources()).resources == []

                res = await s.call_tool("info", {})
                assert not res.isError and "budget_left: 200000" in res.content[0].text

                res = await s.call_tool("view", {"t": 2.0, "out_width": 160})
                assert not res.isError
                assert res.content[0].type == "image" and res.content[0].mimeType == "image/png"
                assert "cost=14400" in res.content[1].text

                res = await s.call_tool("view", {"t": 0, "out_width": 1920})  # 2 Mpx > budget
                assert res.isError and "nothing was charged" in res.content[0].text

                res = await s.call_tool("view", {"t": "nan"})
                assert res.isError

                res = await s.call_tool("view", {"t": 1e309})
                assert res.isError

                res = await s.call_tool("contact_sheet", {"t_start": 0, "t_end": 30, "n": 4, "out_width": 80})
                assert not res.isError and res.content[0].type == "image"

                res = await s.call_tool("submit", {"answer": "K7Q2ZP"})
                assert not res.isError
                res = await s.call_tool("submit", {"answer": "OTHER"})
                assert res.isError

        # a new session shares the same budget and submission state
        async with streamable_http_client(base + "/mcp") as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize()
                res = await s.call_tool("info", {})
                assert f"budget_left: {200_000 - 14400 - 4 * 80 * 45}" in res.content[0].text
                res = await s.call_tool("submit", {"answer": "X"})
                assert res.isError

    asyncio.run(go())
    r = env.result()
    assert r["correct"] is True and r["pixels_used"] == 14400 + 4 * 80 * 45

    import httpx
    assert httpx.get(base + "/_result?token=tok-live").json()["reward"] == 1.0
    assert httpx.get(base + "/_result?token=nope").status_code == 403
    lines = httpx.get(base + "/_log?token=tok-live").text.splitlines()
    assert len(lines) == r["n_calls"]


# --------------------------------------------------------- real scenes, if present
def test_real_scene_smoke():
    scenes = pytest.importorskip("framehunt.scenes")
    for kind in scenes.KINDS:
        try:
            scene = scenes.make_scene(kind, 7001)
        except ImportError:
            continue
        env = FrameHuntEnv(scene, verify_token="t")
        png, text = env.view(scene.meta.duration_s / 2, out_width=240)
        assert png_size(png)[0] == 240
        assert scene.answer not in env.info()
        assert math.isclose(env.pixels_used, 240 * round(scene.meta.height * 240 / scene.meta.width), abs_tol=240)


def test_oracle_smoke_client(live_server):
    """scripts/mcp_smoke.py (the oracle plumbing check) works against the real HTTP app."""
    import subprocess
    import sys
    from pathlib import Path

    env, base = live_server
    script = Path(__file__).resolve().parent.parent / "scripts" / "mcp_smoke.py"
    p = subprocess.run([sys.executable, str(script), base + "/mcp"], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "SMOKE OK" in p.stdout
    r = env.result()
    assert r["submitted"] and r["answer"] == "PLUMBING-CHECK" and r["reward"] == 0.0
    assert r["pixels_used"] == 64 * 36
