"""Generate FrameHunt SFT conversations in OpenCode's exact request format.

    uv run python train/make_sft.py --out train/data/sft-v1 [--per-kind 300] [--workers 4]

For each TRAIN item (framehunt.splits.train_items, scene seed
derive_seed("train", item_id) with the train split's own secret salt) this runs a
teacher policy in-process against FrameHuntEnv (budget 120000, the server default)
and records every tool call with the exact text and PNG bytes the server returns.
The teacher is the reference policy's perception (template matching and OCR with the
bundled fonts, framehunt/reference.py) driving a varied coarse-to-fine plan:

  1. (often) info; then a low-res overview (width 120..240 px) to see the layout;
  2. a contact sheet of the trace (counter / badge / tile strip) over the whole video,
     every tile's value spelled out with its tile number and time (the Opus failure
     was misindexing a sheet), sometimes a second, narrower sheet;
  3. binary search with single cheap views down to the first frame of the event;
  4. ALWAYS a cheap verification look at the chosen frame (and the frame before)
     before spending pixels on reading;
  5. a cheap locating look (ERROR row / dialog card / KPI row), then one native crop
     of the answer text (with random padding, sometimes including the label);
  6. submit, then one short final message (OpenCode ends the run there).

Each assistant turn has a short reasoning text (what the last image showed, what to
do next and why) and exactly one tool call. Only trajectories whose submitted answer
the server marks correct are kept. Output:

    <out>/sft.jsonl        one conversation per line (format: train/fhdata.py)
    <out>/images/<id>/NN.png  the exact PNG bytes FrameHuntEnv returned
    <out>/tools.json       OpenCode's tool list (same for every row)
    <out>/rejected.jsonl   items where the teacher failed (and why)
    <out>/stats.json       counts, pixels, calls

Pixel parity with the MCP sidecar: the sidecar (python:3.12-slim + pillow 13.0.0)
has no libraqm, so Pillow lays out text with its BASIC engine. A host with libraqm
would render slightly different glyph positions (we measured up to 63 levels of
difference on captured frames), so this script forces BASIC layout before any font
is loaded. With that, frames are pixel-identical to what the sidecar returned in
the capture (checked by --check-capture). PNG bytes can still differ from the
sidecar's by zlib build (zlib-ng here); the decoded pixels, which are all the model
sees, are identical. Run inside python:3.12-slim (see train/README.md) for
byte-identical PNGs too.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import multiprocessing as mp
import random
import re
import shutil
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

import PIL.ImageFont as _IF

_IF.core.HAVE_RAQM = False   # match the sidecar's Pillow (no libraqm): BASIC text layout

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "train"))

import fhdata

from framehunt import reference as R
from framehunt.scenes import make_scene
from framehunt.server import (
    DEFAULT_BUDGET_PX,
    MAX_SHEET_N,
    MIN_OUT_WIDTH,
    MIN_SHEET_N,
    SHEET_SEPARATOR_PX,
    FrameHuntEnv,
    FrameHuntError,
)
from framehunt.splits import derive_seed, load_salt, train_items

W, H = 1920, 1080


def instruction_md(scene, budget: int) -> str:
    """instruction.md exactly as scripts/build_tasks.py writes it."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fh_build_tasks", ROOT / "scripts" / "build_tasks.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    m = scene.meta
    return bt.INSTRUCTION_MD.format(
        question=scene.question, answer_format=scene.answer_format, width=m.width, height=m.height, fps=m.fps,
        n_frames=m.n_frames, duration=m.duration_s, budget=budget, full=m.width * m.height,
        min_out=MIN_OUT_WIDTH, min_n=MIN_SHEET_N, max_n=MAX_SHEET_N, sep=SHEET_SEPARATOR_PX)


def tnum(frame: int, fps: int) -> float | int:
    """A frame's time as a model would write it (10 fps: one decimal)."""
    t = round(frame / fps, 3)
    return int(t) if t == int(t) else t


def ts(frame: int, fps: int) -> str:
    return f"{frame / fps:.1f}"


# ============================================================== recorder
class Episode:
    """Calls FrameHuntEnv like the MCP server does and records an OpenCode conversation."""

    def __init__(self, env: FrameHuntEnv, rng: random.Random, conv_id: str):
        self.env, self.rng, self.id = env, rng, conv_id
        self.fps = env.meta.fps
        self.n_frames = env.meta.n_frames
        self.last = self.n_frames - 1
        self.msgs: list[dict] = []
        self.images: list[bytes] = []
        self.note = ""          # what the last result showed; prefixed to the next reasoning
        self._n = 0

    @property
    def left(self) -> int:
        return self.env.budget_left

    def _call_id(self) -> str:
        self._n += 1
        return f"call_{self.rng.getrandbits(48):012x}"

    def call(self, tool: str, args: dict, plan: str) -> tuple[np.ndarray | None, str]:
        reasoning = (self.note + " " + plan).strip() if self.note else plan
        self.note = ""
        cid = self._call_id()
        self.msgs.append({"role": "assistant", "content": "", "reasoning": reasoning,
                          "tool_calls": [{"id": cid, "type": "function",
                                          "function": {"name": fhdata.TOOL_PREFIX + tool,
                                                       "arguments": fhdata.args_json(args)}}]})
        fn = getattr(self.env, tool)
        png = None
        if tool in ("view", "contact_sheet"):
            png, text = fn(**args)
        else:
            text = fn(**args)
        self.msgs.append({"role": "tool", "tool_call_id": cid, "content": text})
        arr = None
        if png is not None:
            k = len(self.images)
            self.images.append(png)
            self.msgs.append({"role": "user", "content": [
                {"type": "text", "text": fhdata.ATTACHED_MEDIA_TEXT},
                {"type": "image_url", "image_url": {"url": f"images/{self.id}/{k:02d}.png"}}]})
            arr = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.int32)
        return arr, text

    def view(self, frame: int, box: tuple[int, int, int, int] | None, out_width: int, plan: str) -> np.ndarray:
        args: dict = {"t": tnum(frame, self.fps)}
        if box is not None:
            x, y, w, h = box
            args.update({"x": x, "y": y, "w": w, "h": h})
        args["out_width"] = out_width
        arr, text = self.call("view", args, plan)
        got = int(re.search(r"frame=(\d+)", text).group(1))
        assert got == frame, (got, frame, text)
        return arr

    def sheet(self, f0: int, f1: int, n: int, box, out_width: int, plan: str) -> tuple[list[np.ndarray], list[int], str]:
        x, y, w, h = box
        args = {"t_start": tnum(f0, self.fps), "t_end": tnum(f1, self.fps), "n": n,
                "x": x, "y": y, "w": w, "h": h, "out_width": out_width}
        arr, text = self.call("contact_sheet", args, plan)
        frames = [int(v) for v in re.findall(r"tile \d+: t=[\d.]+s frame=(\d+)", text)]
        m = re.search(r"(\d+) columns x (\d+) rows.*tile=(\d+)x(\d+)", text)
        cols, rows, tw, th = map(int, m.groups())
        s = SHEET_SEPARATOR_PX
        tiles = []
        for i in range(n):
            r, c = divmod(i, cols)
            tiles.append(arr[r * (th + s):r * (th + s) + th, c * (tw + s):c * (tw + s) + tw])
        return tiles, frames, f"{cols} columns x {rows} rows"

    def submit(self, answer: str, plan: str) -> str:
        _, text = self.call("submit", {"answer": answer}, plan)
        return text

    def final(self, reasoning: str, content: str) -> None:
        self.msgs.append({"role": "assistant", "content": content, "reasoning": reasoning})


# ============================================================== generic trace search
def pick(rng, xs):
    return xs[rng.randrange(len(xs))]


def sheet_n_for(rng, per_tile: int, budget_cap: int, frames: int) -> int | None:
    opts = [n for n in (6, 8, 9, 10, 12, 16) if n * per_tile <= budget_cap and n <= frames]
    return pick(rng, opts) if opts else None


def search_event(ep: Episode, rng, box, ow: int, read, k_desc: str, val_desc, is_hit, trace_name: str,
                 sheet_cap: int) -> int:
    """Find the first frame f with is_hit(read(img)). Returns f (the event frame).

    read(img) -> value; val_desc(value) -> short text ("errors: 2"); is_hit(value) -> bool.
    Coarse contact sheet over the whole video, optional second sheet, then binary
    search with single views. Every observation is spelled out in the next reasoning.
    """
    fps = ep.fps
    tw, th = R.out_size(box[2], box[3], ow)
    per_tile = tw * th
    n = sheet_n_for(rng, per_tile, sheet_cap, ep.n_frames) or 6
    tiles, frames, grid = ep.sheet(
        0, ep.last, n, box, ow,
        pick(rng, [(f"Scan the {trace_name} over the whole video with one contact sheet of {n} frames "
                   f"({tw}x{th} px tiles, {n * per_tile} px) to bracket when it reaches {k_desc}."),
                   (f"A contact sheet of the {trace_name} across the full {ep.last / fps:.0f} s "
                   f"({n} tiles of {tw}x{th}, cost {n * per_tile}) will show roughly when it reaches {k_desc}."),
                   (f"Coarse pass: {n} evenly spaced frames of just the {trace_name}, tile width {tw} "
                   f"({n * per_tile} px total), to find the interval where it reaches {k_desc}.")]))
    vals = [read(t) for t in tiles]
    hits = [is_hit(v) for v in vals]
    j = hits.index(True)       # the last frame always hits (events end before 90 %)
    assert j > 0, "event before the first sheet tile"
    lo, hi = frames[j - 1], frames[j]     # lo: not hit, hi: hit
    listing = "; ".join(f"tile {i + 1} (t={ts(f, fps)}s) {val_desc(v)}" for i, (f, v) in enumerate(zip(frames, vals)))
    ep.note = (f"The sheet is {grid}, read left to right then top to bottom: {listing}. "
               f"So it first reaches {k_desc} between tile {j} (t={ts(lo, fps)}s) and tile {j + 1} (t={ts(hi, fps)}s).")

    # Optional second, narrower sheet inside the bracket.
    gap = hi - lo
    if gap > 40 and rng.random() < 0.45:
        n2 = pick(rng, [4, 5, 6, 8])
        n2 = min(n2, gap - 1)
        if n2 >= 2 and n2 * per_tile < ep.left - 40_000:
            f0, f1 = lo + 1, hi - 1
            tiles, frames, grid = ep.sheet(
                f0, f1, n2, box, ow,
                f"Narrow it down with a second sheet of {n2} frames between t={ts(f0, fps)}s and t={ts(f1, fps)}s.")
            vals = [read(t) for t in tiles]
            listing = "; ".join(f"tile {i + 1} (t={ts(f, fps)}s) {val_desc(v)}" for i, (f, v) in enumerate(zip(frames, vals)))
            hit_idx = [i for i, v in enumerate(vals) if is_hit(v)]
            if hit_idx:
                j = hit_idx[0]
                if j > 0:
                    lo, hi = frames[j - 1], frames[j]
                    where = f"between tile {j} (t={ts(lo, fps)}s) and tile {j + 1} (t={ts(hi, fps)}s)"
                else:
                    hi = frames[0]
                    where = f"before tile 1 (t={ts(hi, fps)}s), after t={ts(lo, fps)}s"
            else:
                lo = frames[-1]
                where = f"after the last tile (t={ts(lo, fps)}s), by t={ts(hi, fps)}s"
            ep.note = f"Second sheet ({grid}): {listing}. The change is {where}."

    # Binary search with single views.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        img = ep.view(mid, box, ow, pick(rng, [
            f"Bracket is t={ts(lo, fps)}s..{ts(hi, fps)}s ({hi - lo} frames); check the {trace_name} at the midpoint t={ts(mid, fps)}s.",
            f"Binary search: look at the {trace_name} at t={ts(mid, fps)}s (between {ts(lo, fps)} and {ts(hi, fps)}).",
            f"Halve the interval: {trace_name} at t={ts(mid, fps)}s."]))
        v = read(img)
        if is_hit(v):
            hi = mid
            ep.note = f"At t={ts(mid, fps)}s it shows {val_desc(v)}, so the change is at or before this frame."
        else:
            lo = mid
            ep.note = f"At t={ts(mid, fps)}s it shows {val_desc(v)}, so the change comes later."
    return hi


def verify_event(ep: Episode, rng, f: int, box, vow: int, read, val_desc, is_hit, trace_name: str) -> bool:
    """The verification look, at a higher resolution than the search used: the trace
    must show the event at f (and, in the 2-tile variant, not at f-1)."""
    fps = ep.fps
    if rng.random() < 0.4:
        tiles, _frames, _ = ep.sheet(f - 1, f, 2, box, vow, pick(rng, [
            (f"Before reading anything at native resolution, verify the event frame at a higher resolution ({vow} px wide): "
            f"a 2-tile sheet of the {trace_name} at t={ts(f - 1, fps)}s and t={ts(f, fps)}s should show the change between them."),
            f"Verify first, sharper than the search looks: {trace_name} at t={ts(f - 1, fps)}s (tile 1) and t={ts(f, fps)}s (tile 2), {vow} px wide."]))
        v0, v1 = read(tiles[0]), read(tiles[1])
        ok = (not is_hit(v0)) and is_hit(v1)
        ep.note = (f"Verified: tile 1 (t={ts(f - 1, fps)}s) shows {val_desc(v0)} and tile 2 (t={ts(f, fps)}s) shows "
                   f"{val_desc(v1)}, so t={ts(f, fps)}s (frame {f}) is the first frame of the event.")
    else:
        img = ep.view(f, box, vow, pick(rng, [
            (f"The search says the event starts at t={ts(f, fps)}s (frame {f}). Before spending pixels on the native crop, "
            f"confirm the {trace_name} there at a higher resolution ({vow} px wide), since the search looks were tiny."),
            f"Double-check the {trace_name} at t={ts(f, fps)}s, the frame I am about to read from, at {vow} px wide so the reading is unambiguous.",
            f"Verification look: {trace_name} at the chosen frame t={ts(f, fps)}s, sharper ({vow} px wide) than the search."]))
        v1 = read(img)
        ok = is_hit(v1)
        ep.note = (f"Verified: at t={ts(f, fps)}s the {trace_name} clearly shows {val_desc(v1)}, and t={ts(f - 1, fps)}s "
                   f"did not, so frame {f} is the first frame of the event.")
    return ok


def overview(ep: Episode, rng, frame: int, widths, why: str) -> np.ndarray:
    ow = pick(rng, widths)
    oh = R.out_size(W, H, ow)[1]
    return ep.view(frame, None, ow, pick(rng, [
        f"Start with one cheap low-res overview (t={ts(frame, ep.fps)}s, {ow}x{oh} = {ow * oh} px) to see the screen layout{why}.",
        f"First a {ow}-px-wide overview of the whole frame at t={ts(frame, ep.fps)}s ({ow * oh} px) to locate the screen elements{why}.",
        f"Low-res look at the full screen at t={ts(frame, ep.fps)}s, {ow} px wide, to get the layout{why}."]))


def maybe_info(ep: Episode, rng) -> None:
    if rng.random() < 0.6:
        ep.call("info", {}, pick(rng, [
            "The task is in the instruction; info is free, so confirm the question, video length and budget first.",
            "Check the free info tool for the question, frame count and budget before spending pixels.",
            "Read the free task info first."]))
        ep.note = pick(rng, ["Info confirms the question and the 120000 px budget.",
                             "Budget is 120000 px; the question matches the instruction.",
                             "Info matches the instruction."])


# ============================================================== kinds
def solve_logstream(ep: Episode, rng, q: dict) -> str:
    L = R.Log
    k, _fps = q["k"], ep.fps
    maybe_info(ep, rng)
    overview(ep, rng, pick(rng, [ep.last, rng.randrange(ep.n_frames // 3, ep.last)]), (120, 160, 200, 240),
             ", especially the 'errors: N' counter in the bottom status bar")
    box = L.COUNTER_BOX
    ep.note = (f"The overview shows a terminal log with a status bar at the bottom; the errors counter box is at the "
               f"bottom right, about x={box[0]}..{box[0] + box[2]}, y={box[1]}..{box[1] + box[3]} in native pixels.")
    ow = pick(rng, [91, 104, 122])
    tm = L.counter_templates(ow)

    def read(img):
        return R.nearest(img, tm)[0]

    f = search_event(ep, rng, box, ow, read, f"errors: {k}", lambda v: f"errors: {v}", lambda v: v >= k,
                     "errors counter", sheet_cap=18_000)
    vow = pick(rng, [182, 243])
    tv = L.counter_templates(vow)
    if not verify_event(ep, rng, f, box, vow, lambda img: R.nearest(img, tv)[0], lambda v: f"errors: {v}",
                        lambda v: v >= k, "errors counter"):
        raise RuntimeError("verification failed")
    ep.note += f" The {_ord(k)} ERROR line was printed on this frame, so it is the newest ERROR line on screen."

    # Locate the newest ERROR row from the red level column.
    n_rows = pick(rng, [16, 24, L.N_ROWS])
    div = pick(rng, [2, 2, 3])
    r0 = L.N_ROWS - n_rows
    sbox = (L.TEXT_X + L.LEVEL_COL * L.CELL_W, L.TEXT_Y + r0 * L.CELL_H, 5 * L.CELL_W, n_rows * L.CELL_H)
    strip = ep.view(f, sbox, sbox[2] // div, pick(rng, [
        f"Find which row holds that ERROR line: a narrow strip of the level column (the bottom {n_rows} rows) at 1/{div} scale; ERROR is red.",
        f"Locate the newest red ERROR level in the last {n_rows} log rows with a cheap 1/{div}-scale strip of the level column."]))
    rh = strip.shape[0] / n_rows
    red = [int((strip[int(i * rh):int((i + 1) * rh), :, 0]
                - strip[int(i * rh):int((i + 1) * rh), :, 1:].max(-1)).max()) for i in range(n_rows)]
    rows = [i for i, v in enumerate(red) if v > 70]
    if not rows:
        raise RuntimeError("no ERROR row in strip")
    row = r0 + rows[-1]
    y_row = L.TEXT_Y + row * L.CELL_H
    ep.note = (f"The lowest red ERROR entry in the strip is row {rows[-1] + 1} of {n_rows}, i.e. terminal row {row + 1} of "
               f"{L.N_ROWS}, at y={y_row}..{y_row + L.CELL_H}.")

    # Native crop of the request id, with random padding (sometimes including "req=").
    lead = pick(rng, [0, 0, 4])          # cells before the id ("req=")
    padl, padr = rng.randint(0, 6), rng.randint(0, 20)
    padt, padb = rng.randint(0, 4), rng.randint(0, 4)
    x0 = L.TEXT_X + (L.RID_COL - lead) * L.CELL_W - padl
    y0 = y_row - padt
    bw = (8 + lead) * L.CELL_W + padl + padr
    bh = L.CELL_H + padt + padb
    crop = ep.view(f, (x0, y0, bw, bh), bw, pick(rng, [
        f"Read the request id at native resolution: a {bw}x{bh} crop of that row{' including req=' if lead else ''} ({bw * bh} px).",
        f"Now one native-resolution crop ({bw}x{bh}, {bw * bh} px) of the request id on row {row + 1}."]))
    sub = crop[padt:padt + L.CELL_H, padl + lead * L.CELL_W:padl + (lead + 8) * L.CELL_W]
    rid, _margin = R.ocr_grid(sub, 8, L.CELL_W, L.CELL_H, L.Y_OFF, "mono", 16, L.FG, L.BG, "0123456789abcdef")
    ep.note = f"The crop reads req={rid}."
    return rid


def _ord(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def solve_dialogs(ep: Episode, rng, q: dict) -> str:
    D = R.Dialogs
    k, fps = q["k"], ep.fps
    maybe_info(ep, rng)
    overview(ep, rng, pick(rng, [ep.last, rng.randrange(ep.n_frames // 3, ep.last)]), (120, 160, 200, 240),
             ", especially the notification bell and its badge in the top bar")
    box = D.TRACE_BOX
    ep.note = (f"The overview shows a desktop app with a navy top bar; the bell with the red badge is at the top right, "
               f"about x={box[0]}..{box[0] + box[2]}, y={box[1]}..{box[1] + box[3]}.")
    ow = pick(rng, [30, 40, 48])
    tm = D.badge_templates(ow)

    def read(img):
        return R.nearest(img, tm)[0]

    f = search_event(ep, rng, box, ow, read, f"badge {k}", lambda v: f"badge {v}", lambda v: v >= k,
                     "bell badge", sheet_cap=24_000)
    vow = pick(rng, [60, 80])
    tv = D.badge_templates(vow)
    if not verify_event(ep, rng, f, box, vow, lambda img: R.nearest(img, tv)[0], lambda v: f"badge {v}",
                        lambda v: v >= k, "bell badge"):
        raise RuntimeError("verification failed")
    ep.note += f" The {_ord(k)} dialog opened on this frame."

    sc = pick(rng, [16, 16, 12])
    ov = ep.view(f, None, W // sc, pick(rng, [
        f"Locate the open dialog card with a 1/{sc}-scale look at t={ts(f, fps)}s ({W // sc} px wide); the rest of the screen is dimmed behind a modal.",
        f"Where is the dialog? One {W // sc}-px-wide look at the full frame at t={ts(f, fps)}s."]))
    top = 112 // sc + 1
    bright = ov[top:].min(-1) > 225
    ys, xs = np.nonzero(bright)
    if len(xs) < 20:
        raise RuntimeError("no dialog card visible")
    cx0 = int(xs.min()) * sc
    cy0 = (int(ys.min()) + top) * sc
    ep.note = (f"The white dialog card is near x={cx0}, y={cy0} (from the low-res look, +-{sc} px); its error code box "
               f"sits about {D.CODE_REL[0]} px right and {D.CODE_REL[1]} px below the card's top-left corner.")
    mx, my = 2 * sc + rng.randint(0, 8), 2 * sc + rng.randint(0, 8)
    bx0, by0 = D.CODE_REL[0] - 10, D.CODE_REL[1] - 6
    cbox = (max(0, cx0 + bx0 - mx), max(0, cy0 + by0 - my), 75 + 2 * mx + rng.randint(0, 20), 32 + 2 * my)
    crop = ep.view(f, cbox, cbox[2], pick(rng, [
        f"Read the code at native resolution: a {cbox[2]}x{cbox[3]} crop around the code box ({cbox[2] * cbox[3]} px).",
        f"One native crop ({cbox[2]}x{cbox[3]}, {cbox[2] * cbox[3]} px) of the code box, with margin for the low-res position estimate."]))
    mask = (np.abs(crop - np.asarray(D.CODE_BOX_RGB)).max(-1) <= 1)
    ys, xs = np.nonzero(mask.sum(1) >= 40)[0], np.nonzero(mask.sum(0) >= 12)[0]
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("code box not in crop")
    ox, oy = int(xs.min()) + 9, int(ys.min()) + 5
    best = None
    from PIL import ImageDraw
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            code, _margin = R.ocr_string(crop, (ox + dx, oy + dy), *D.CODE_FONT, D.INK, D.CODE_BOX_RGB,
                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 6, 6)
            im = Image.new("RGB", (crop.shape[1], crop.shape[0]), D.CODE_BOX_RGB)
            ImageDraw.Draw(im).text((ox + dx, oy + dy), code, font=R.font(*D.CODE_FONT), fill=D.INK)
            r = np.asarray(im, np.int32)
            y0, y1, x0, x1 = oy - 2, oy + 20, ox - 3, ox + 58
            err = float(np.abs(r[y0:y1, x0:x1] - crop[y0:y1, x0:x1]).sum())
            if best is None or err < best[0]:
                best = (err, code)
    code = best[1]
    ep.note = f"The crop shows the error code {code}."
    return code


def solve_dashboard(ep: Episode, rng, q: dict) -> str:
    Dh = R.Dash
    fps = ep.fps
    maybe_info(ep, rng)
    ov = overview(ep, rng, ep.last, (160, 192, 240),
                  "; on the last frame the named tile should already be red, which tells me which tile it is")
    sc = W / ov.shape[1]
    y0, y1 = int((Dh.TILE_Y + 136) / sc), int((Dh.TILE_Y + 162) / sc)
    red = [i for i in range(Dh.N_TILES)
           if Dh.is_red(ov[y0:y1, int((Dh.tile_x(i) + 20) / sc):int((Dh.tile_x(i) + Dh.TILE_W - 20) / sc)])]
    if len(red) != 1:
        raise RuntimeError(f"red tiles on last frame: {red}")
    ti = red[0]
    band = (Dh.tile_x(ti) + 10, Dh.TILE_Y + 136, Dh.TILE_W - 20, 26)
    ep.note = (f"The overview shows a row of {Dh.N_TILES} service tiles; on the last frame only tile {ti + 1} from the left "
               f"(x={Dh.tile_x(ti)}..{Dh.tile_x(ti) + Dh.TILE_W}) is red, so that is the \"{q['tile']}\" tile. Its lower "
               f"strip y={band[1]}..{band[1] + band[3]} has no text, so its colour alone is a cheap trace.")
    ow = pick(rng, [24, 36, 48])

    def read(img):
        return Dh.is_red(img)

    f = search_event(ep, rng, band, ow, read, "red", lambda v: "red" if v else "not red", lambda v: v,
                     f"tile {ti + 1} strip", sheet_cap=6_000)
    # Verify on the whole tile (status text visible) at about 1/4 scale.
    tile = (Dh.tile_x(ti), Dh.TILE_Y, Dh.TILE_W, Dh.TILE_H)
    vow = pick(rng, [64, 76])

    def read_tile(img):
        s_ = img.shape[1] / Dh.TILE_W
        return Dh.is_red(img[int(136 * s_):int(162 * s_), int(20 * s_):img.shape[1] - int(20 * s_)])

    if not verify_event(ep, rng, f, tile, vow, read_tile, lambda v: "red (DOWN)" if v else "not red",
                        lambda v: v, f"whole tile {ti + 1}"):
        raise RuntimeError("verification failed")

    # Which table row holds the KPI.
    div = pick(rng, [2, 2, 2.5])
    nb = (Dh.TABLE_X + 8, Dh.TABLE_Y + 44, 200, Dh.N_ROWS * Dh.ROW_H)
    now = int(nb[2] / div)
    names = ep.view(f, nb, now, pick(rng, [
        f"Find the row of \"{q['kpi']}\" in the Key metrics table: the label column at ~1/{div:g} scale ({now} px wide).",
        f"Cheap look at the Key metrics labels (1/{div:g} scale) to find which row is \"{q['kpi']}\"."]))
    f16 = R.font("sans", 16)
    from PIL import ImageDraw
    errs = []
    rh = names.shape[0] / Dh.N_ROWS
    for r in range(Dh.N_ROWS):
        im = Image.new("RGB", nb[2:], Dh.PANEL)
        d = ImageDraw.Draw(im)
        for rr in range(Dh.N_ROWS):
            if rr % 2:
                d.rectangle((0, rr * Dh.ROW_H, nb[2], rr * Dh.ROW_H + Dh.ROW_H - 1), fill=Dh.ALT)
        d.text((Dh.TABLE_X + 16 - nb[0], r * Dh.ROW_H + 11), q["kpi"], font=f16, fill=Dh.TEXT)
        t = R.shrink(im, now)
        ry0, ry1 = int(r * rh), int((r + 1) * rh)
        errs.append((float(np.abs(t[ry0:ry1] - names[ry0:ry1]).mean()), r))
    errs.sort()
    row = errs[0][1]
    bg = Dh.ALT if row % 2 else Dh.PANEL
    vy = Dh.TABLE_Y + 44 + row * Dh.ROW_H + 11
    ep.note = f"\"{q['kpi']}\" is row {row + 1} of the table; its value sits at about x={Dh.VALUE_X}, y={vy}."

    with_label = rng.random() < 0.3
    padt, padb = rng.randint(2, 6), rng.randint(2, 6)
    x0 = (Dh.TABLE_X + 8) if with_label else Dh.VALUE_X - rng.randint(2, 8)
    bw = (Dh.VALUE_X + 75 - x0) + rng.randint(0, 12)
    box = (x0, vy - padt, bw, 19 + padt + padb)
    crop = ep.view(f, box, bw, pick(rng, [
        f"Read the value at native resolution on frame {f}: a {box[2]}x{box[3]} crop{' including the row label' if with_label else ''} ({box[2] * box[3]} px).",
        f"Native crop ({box[2]}x{box[3]}, {box[2] * box[3]} px) of row {row + 1}{' with its label' if with_label else ''} at t={ts(f, fps)}s."]))
    val, _margin = R.ocr_string(crop, (Dh.VALUE_X - x0, padt), "mono", 14, Dh.TEXT, bg, "0123456789", 1, 6)
    ep.note = f"The crop reads {q['kpi']} = {val}."
    return val


SOLVERS = {"logstream": solve_logstream, "dialogs": solve_dialogs, "dashboard": solve_dashboard}


# ============================================================== one item
def run_item(item: tuple, split: str, salt: str, budget: int, gen_seed: int, out: Path, base_date: date) -> dict:
    item_id, kind, _public = item
    scene = make_scene(kind, derive_seed(split, item_id, salt))
    env = FrameHuntEnv(scene, budget_px=budget)
    rng = random.Random(f"{gen_seed}:{item_id}")
    ep = Episode(env, rng, item_id)
    q = R.parse_question(scene.question)
    t0 = time.time()
    try:
        answer = SOLVERS[kind](ep, rng, q)
        ep.submit(answer, pick(rng, [f"Submit {answer}.", "That is the answer; submit it.", f"Submitting {answer}."]))
        ep.note = ""
        res = env.result()
        ep.final("The answer is recorded.", pick(rng, [f"Submitted {answer}.", f"Answer submitted: {answer}",
                                                      f"{answer}"]))
    except (RuntimeError, AssertionError, FrameHuntError, ValueError, IndexError) as e:
        return {"ok": False, "item_id": item_id, "kind": kind, "error": f"{type(e).__name__}: {e}",
                "pixels_used": env.pixels_used}
    if not res["correct"]:
        return {"ok": False, "item_id": item_id, "kind": kind, "error": "wrong answer",
                "submitted": res["answer"], "pixels_used": res["pixels_used"]}
    d = base_date + timedelta(days=rng.randint(-20, 20))
    date_str = d.strftime("%a %b %d %Y")
    msgs = [{"role": "system", "content": fhdata.system_prompt(date_str)},
            fhdata.user_turn(instruction_md(scene, budget))] + ep.msgs
    img_dir = out / "images" / item_id
    img_dir.mkdir(parents=True, exist_ok=True)
    for i, png in enumerate(ep.images):
        (img_dir / f"{i:02d}.png").write_bytes(png)
    tools_used = [m["tool_calls"][0]["function"]["name"] for m in ep.msgs if m.get("tool_calls")]
    return {"ok": True, "row": {
        "id": item_id, "item_id": item_id, "kind": kind, "split": split, "answer": res["answer"],
        "correct": True, "pixels_used": res["pixels_used"], "budget": budget, "n_calls": len(tools_used),
        "n_images": len(ep.images), "tools_used": tools_used, "date": date_str,
        "gen_seconds": round(time.time() - t0, 2), "messages": msgs}}


def _worker(args):
    try:
        return run_item(*args)
    except Exception as e:  # never kill the pool; report
        return {"ok": False, "item_id": args[0][0], "kind": args[0][1],
                "error": f"crash {type(e).__name__}: {e}", "tb": traceback.format_exc()[-2000:]}


# ============================================================== capture parity check
def check_capture(budget: int) -> None:
    """Rebuild the captured eval conversation's user turn and images in-process and compare.

    Uses the eval salt only to recompute that one item (nothing is printed or written).
    """
    req = json.loads((fhdata.CAPTURE / "requests" / "req_06.json").read_text())["body"]
    msgs = req["messages"]
    salt = load_salt("eval")
    if salt is None:
        print("check-capture: no eval salt here; skipped")
        return
    scene = make_scene("logstream", derive_seed("eval", "logstream-7001", salt))
    env = FrameHuntEnv(scene, budget_px=budget)
    assert msgs[0]["content"] == fhdata.system_prompt(), "system prompt differs from capture"
    assert msgs[1] == fhdata.user_turn(instruction_md(scene, budget)), "user turn differs from capture"
    assert req["tools"] == fhdata.tools(), "tools differ from capture"
    calls = [m["tool_calls"][0]["function"] for m in msgs if m.get("tool_calls")]
    results = iter(m for m in msgs if m["role"] == "tool")
    img_urls = iter(p["image_url"]["url"] for m in msgs if isinstance(m.get("content"), list)
                    for p in m["content"] if p.get("type") == "image_url")
    n_px = n_bytes = 0
    for c in calls:
        tool = c["name"].removeprefix(fhdata.TOOL_PREFIX)
        args = json.loads(c["arguments"])
        r = getattr(env, tool)(**args)
        text = r[1] if isinstance(r, tuple) else r
        want = next(results)["content"]
        assert text == want, f"{tool} text differs:\n{text}\n---\n{want}"
        if isinstance(r, tuple):
            got = base64.b64decode(next(img_urls).split(",", 1)[1])
            a = np.asarray(Image.open(io.BytesIO(got)).convert("RGB"))
            b = np.asarray(Image.open(io.BytesIO(r[0])).convert("RGB"))
            assert a.shape == b.shape and (a == b).all(), f"{tool} pixels differ from the sidecar's"
            n_px += 1
            n_bytes += got == r[0]
    print(f"check-capture: system prompt, user turn, tools, {len(calls)} tool texts and {n_px} images "
          f"pixel-identical to the captured OpenCode run ({n_bytes}/{n_px} PNGs byte-identical)")


# ============================================================== main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "train" / "data" / "sft-v1")
    ap.add_argument("--per-kind", type=int, default=300)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET_PX)
    ap.add_argument("--seed", type=int, default=0, help="teacher randomness (plan variations, dates)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--date", default="2026-11-14", help="centre of the system prompt dates (+-20 days)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--check-capture", action="store_true", help="only run the capture parity check")
    a = ap.parse_args(argv)
    if a.check_capture:
        check_capture(a.budget)
        return 0
    split = "train"
    salt = load_salt(split)
    if salt is None:
        salt = load_salt(split, create=True)
        print("generated a new secret salt for split 'train' (secrets/train.salt); keep it to regenerate the same items")
    items = list(train_items(a.per_kind))
    if a.only:
        items = [it for it in items if it[0] in set(a.only)]
    out = a.out
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    shutil.copy(fhdata.TOOLS_FILE, out / "tools.json")
    base_date = date.fromisoformat(a.date)
    jobs = [(it, split, salt, a.budget, a.seed, out, base_date) for it in items]
    t0 = time.time()
    rows, rejected = [], []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(_worker, jobs, chunksize=4)):
            (rows.append(r["row"]) if r["ok"] else rejected.append(r))
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} done, kept {len(rows)}, {time.time() - t0:.0f}s", flush=True)
    with open(out / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(out / "rejected.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    stats = {"items": len(items), "kept": len(rows), "rejected": len(rejected), "seconds": round(time.time() - t0, 1),
             "budget": a.budget, "gen_seed": a.seed, "by_kind": {}}
    for kind in ("logstream", "dialogs", "dashboard"):
        kr = [r for r in rows if r["kind"] == kind]
        px = sorted(r["pixels_used"] for r in kr)
        calls = [r["n_calls"] for r in kr]
        stats["by_kind"][kind] = {
            "items": sum(1 for it in items if it[1] == kind), "kept": len(kr),
            "pixels_mean": round(sum(px) / len(px)) if px else None, "pixels_max": px[-1] if px else None,
            "calls_mean": round(sum(calls) / len(calls), 1) if calls else None,
            "calls_min": min(calls) if calls else None, "calls_max": max(calls) if calls else None,
            "with_contact_sheet": sum(1 for r in kr if fhdata.TOOL_PREFIX + "contact_sheet" in r["tools_used"]),
            "with_info": sum(1 for r in kr if fhdata.TOOL_PREFIX + "info" in r["tools_used"])}
    from collections import Counter
    stats["reject_reasons"] = dict(Counter(r["error"].split(":")[0] for r in rejected))
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
