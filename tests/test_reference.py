"""The reference policy solves every EVAL item through the real MCP server (in-process HTTP).

Slow-ish (renders real scenes): run alone with `uv run pytest -m reference -q`, or skip
with `-m "not reference"`. Set FH_REF_SPLIT=dev to run the DEV split instead.
"""

from __future__ import annotations

import io
import json
import os
import socket
import threading
import time

import pytest

from framehunt import splits
from framehunt.scenes import make_scene
from framehunt.server import FrameHuntEnv, build_mcp

pytestmark = pytest.mark.reference

SPLIT = os.environ.get("FH_REF_SPLIT", "eval")
ITEMS = splits.SPLITS[SPLIT]
MAX_PIXELS = 70_000   # reference needs <= ~59k; the 120k budget is set at about 2x this (docs/design.md)


def scene_seed(item_id: str, seed: int) -> int:
    """Scene seed as the task builder makes it: derive_seed (salted) if present, else the raw seed."""
    derive = getattr(splits, "derive_seed", None)
    return derive(SPLIT, item_id) if derive else seed


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def serve():
    import uvicorn

    servers = []

    def start(env: FrameHuntEnv) -> str:
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
        return f"http://127.0.0.1:{port}/mcp"

    yield start
    for srv, th in servers:
        srv.should_exit = True
        th.join(5)


@pytest.mark.parametrize("item_id,kind,seed", ITEMS, ids=[i for i, _, _ in ITEMS])
def test_reference_solves_item(serve, item_id, kind, seed):
    from framehunt.reference import run

    env = FrameHuntEnv(make_scene(kind, scene_seed(item_id, seed)), verify_token="t")
    url = serve(env)
    out = io.StringIO()
    res = run(url, out=out)
    r = env.result()
    s = res["summary"]
    print(json.dumps({"item": item_id, "correct": r["correct"], "pixels_used": r["pixels_used"],
                      "n_calls": r["n_calls"], "answer": s["answer"]}))
    assert r["submitted"]
    assert r["correct"], f"{item_id}: read {s['answer']!r}; trace:\n{out.getvalue()}"
    assert r["pixels_used"] == s["pixels_used"]
    assert r["pixels_used"] <= MAX_PIXELS
    # Only the agent's tools were used.
    assert {t["tool"] for t in res["trace"]} <= {"info", "view", "contact_sheet", "submit"}


# ------------------------------------------------------------ many seeds, in-process
class _InProcess:
    """The reference's MCP client, short-circuited to FrameHuntEnv (same tools, same bytes)."""

    def __init__(self, env: FrameHuntEnv):
        self.env = env

    def call(self, tool: str, args: dict):
        from PIL import Image

        fn = getattr(self.env, tool)
        if tool in ("view", "contact_sheet"):
            png, text = fn(**args)
            return Image.open(io.BytesIO(png)).convert("RGB"), text
        return None, fn(**args)


N_MANY = int(os.environ.get("FH_REF_N", "200"))
N_MANY_DOC = int(os.environ.get("FH_REF_N_DOC", "100"))   # scandoc pages are slower to render


@pytest.mark.parametrize("kind", ["toasts", "leaderboard", "scandoc"])
def test_reference_many_seeds(kind):
    """The reference solves every one of N_MANY random 64-bit seeds of the new kinds."""
    from framehunt.reference import _run

    fails, px = [], []
    n_many = N_MANY_DOC if kind == "scandoc" else N_MANY
    for i in range(n_many):
        seed = splits.derive_seed("refcheck", f"{kind}-{i}", "reference-many-seeds")
        env = FrameHuntEnv(make_scene(kind, seed), verify_token="t")
        _run(_InProcess(env), None, time.time())
        r = env.result()
        px.append(r["pixels_used"])
        if not r["correct"]:
            fails.append((i, seed, r["answer"]))
    print(json.dumps({"kind": kind, "n": n_many, "solved": n_many - len(fails), "max_px": max(px),
                      "mean_px": round(sum(px) / len(px))}))
    assert not fails, fails
    assert max(px) <= MAX_PIXELS
