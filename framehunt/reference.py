"""FrameHunt reference policy: scripted coarse-to-fine search over the agent's MCP tools.

    python -m framehunt.reference --url http://fh-mcp:8000/mcp [--trace trace.json]

It talks to the server exactly like an agent does (MCP streamable HTTP, tools
``info``, ``view``, ``submit``) and sees exactly what an agent sees. It never uses the
seed, the scene generator or any hidden route. What it brings is knowledge an agent
could pick up from one look at the screen: where the trace (counter, badge, tile) and
the answer text sit on a 1920x1080 screen of each kind, and which bundled font and size
the text uses.

Plan (same for every kind):
  1. ``info`` (free): question, video length, budget. The question names the kind.
  2. One low-res overview of the last frame (1/8 scale): confirms the layout and, on the
     dashboard, shows which service tile ended up red.
  3. Binary search for the event frame over the lasting, monotonic trace, reading only
     a small crop of the trace at <= 1/8 scale per probe.
  4. If needed, one cheap low-res look on the event frame to locate the answer (the
     dialog card, the ERROR row, the KPI row, the top toast's colour, the endpoint's
     row in the re-sorting table).
  5. One native-resolution crop of the answer text, read by deterministic template
     matching against the bundled fonts, then ``submit``.

Every tool call is printed to stdout as one JSON line (and optionally saved with
``--trace``). Dependencies: pillow, numpy, mcp.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import re
import sys
import time
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import numpy as np
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path(os.environ.get("FH_FONT_DIR") or Path(__file__).resolve().parent / "assets" / "fonts")
FONT_FILES = {"sans": "DejaVuSans.ttf", "sans-bold": "DejaVuSans-Bold.ttf",
              "mono": "DejaVuSansMono.ttf", "mono-bold": "DejaVuSansMono-Bold.ttf"}
RESAMPLE = Image.Resampling.LANCZOS  # what the server uses to downscale
W, H = 1920, 1080

_fonts: dict = {}


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    key = (name, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(str(FONT_DIR / FONT_FILES[name]), size)
    return _fonts[key]


# ============================================================== MCP client
class MCPClient:
    """Synchronous wrapper over the mcp library's async ClientSession."""

    def __init__(self, url: str):
        with ExitStack() as stack:
            self._portal = stack.enter_context(start_blocking_portal())
            read, write, _ = stack.enter_context(self._portal.wrap_async_context_manager(streamable_http_client(url)))
            self._session = stack.enter_context(self._portal.wrap_async_context_manager(ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120))))
            self._portal.call(self._session.initialize)
            self._stack = stack.pop_all()

    def close(self):
        self._stack.close()

    def call(self, tool: str, args: dict) -> tuple[Image.Image | None, str]:
        res = self._portal.call(self._session.call_tool, tool, args)
        text = "\n".join(c.text for c in res.content if c.type == "text")
        if res.isError:
            raise RuntimeError(f"{tool}({args}) refused: {text}")
        img = next((Image.open(io.BytesIO(base64.b64decode(c.data))).convert("RGB")
                    for c in res.content if c.type == "image"), None)
        return img, text


# ============================================================== session + trace
class Session:
    def __init__(self, client: MCPClient, out=sys.stdout):
        self.c = client
        self.out = out
        self.trace: list[dict] = []
        self.fps = 10
        self.n_frames = 0

    def _log(self, rec: dict) -> None:
        rec = {"step": len(self.trace), **rec}
        self.trace.append(rec)
        if self.out is not None:
            print(json.dumps(rec), file=self.out, flush=True)

    def info(self) -> dict:
        _, text = self.c.call("info", {})
        d = {}
        for line in text.splitlines():
            if ":" in line and not line.startswith(" "):
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
        self.fps, self.n_frames = int(d["fps"]), int(d["n_frames"])
        self._log({"tool": "info", "args": {}, "purpose": "read question and budget", "cost": 0,
                   "budget_left": int(d["budget_left"])})
        return d

    def view(self, frame: int, box: tuple[int, int, int, int], out_width: int, purpose: str) -> np.ndarray:
        x, y, w, h = box
        args = {"t": round(frame / self.fps, 4), "x": x, "y": y, "w": w, "h": h, "out_width": out_width}
        img, text = self.c.call("view", args)
        got = int(re.search(r"(?:frame|page)=(\d+)", text).group(1))
        if got != frame:
            raise RuntimeError(f"asked for frame {frame}, got {got}: {text}")
        self._log({"tool": "view", "args": args, "purpose": purpose, "frame": frame,
                   "cost": int(re.search(r"cost=(\d+)", text).group(1)),
                   "budget_left": int(re.search(r"budget_left=(\d+)", text).group(1)),
                   "output": list(img.size)})
        return np.asarray(img, np.int32)

    def submit(self, answer: str, note: dict) -> str:
        _, text = self.c.call("submit", {"answer": answer})
        self._log({"tool": "submit", "args": {"answer": answer}, "cost": 0, **note})
        return text

    @property
    def pixels_used(self) -> int:
        return sum(r.get("cost", 0) for r in self.trace)


# ============================================================== image helpers
def out_size(w: int, h: int, out_width: int) -> tuple[int, int]:
    """Same rule as the server: cap at crop width, keep aspect, round half up."""
    ow = min(out_width, w)
    return ow, max(1, math.floor(h * ow / w + 0.5))


def shrink(img: Image.Image, out_width: int) -> np.ndarray:
    size = out_size(img.width, img.height, out_width)
    if size != img.size:
        img = img.resize(size, RESAMPLE)
    return np.asarray(img, np.int32)


def nearest(obs: np.ndarray, templates: dict) -> tuple[object, float, float]:
    """Key of the template closest to obs (mean abs error), its error and the runner-up's."""
    errs = sorted((float(np.abs(obs - t).mean()), k) for k, t in templates.items())
    return errs[0][1], errs[0][0], errs[1][0] if len(errs) > 1 else math.inf


def first_true(lo: int, hi: int, pred) -> int:
    """Smallest f in [lo, hi] with pred(f) true, given pred(hi) is true and pred is monotone."""
    while lo < hi:
        mid = (lo + hi) // 2
        if pred(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def ocr_string(obs: np.ndarray, origin: tuple[float, float], font_name: str, size: int, fg, bg,
               alphabet: str, min_len: int, max_len: int) -> tuple[str, float]:
    """Read a string drawn with ImageDraw.text at `origin` in a crop, left to right.

    For each position, render prefix + candidate exactly like the scene does (same font,
    size, colours, origin) and keep the candidate whose rendering best matches the crop
    up to the end of that character. Returns the text and the worst per-position margin
    (runner-up error minus best error; larger is more certain).
    """
    f = font(font_name, size)
    hgt, wid = obs.shape[:2]
    ox, oy = origin
    y0, y1 = max(0, int(oy)), min(hgt, int(oy) + size + 6)
    adv = f.getlength("0")
    text, margin = "", math.inf

    def render(s: str) -> np.ndarray:
        im = Image.new("RGB", (wid, hgt), bg)
        ImageDraw.Draw(im).text((ox, oy), s, font=f, fill=fg)
        return np.asarray(im, np.int32)

    for i in range(max_len):
        x1 = min(wid, int(ox + adv * (i + 1)) + 2)
        x0 = max(0, int(ox + adv * i) - 2)
        cands = list(alphabet) + ([""] if i >= min_len else [])
        errs = sorted((float(np.abs(render(text + c)[y0:y1, x0:x1] - obs[y0:y1, x0:x1]).sum()), c)
                      for c in cands)
        margin = min(margin, errs[1][0] - errs[0][0])
        if errs[0][1] == "":
            break
        text += errs[0][1]
    return text, margin


def ocr_grid(obs: np.ndarray, n: int, cell_w: int, cell_h: int, y_off: int, font_name: str, size: int,
             fg, bg, alphabet: str) -> tuple[str, float]:
    """Read n characters drawn one per fixed terminal cell (glyph blended over bg)."""
    f = font(font_name, size)
    fgv, bgv = np.asarray(fg, np.int32), np.asarray(bg, np.int32)
    tmpl = {}
    for c in alphabet:
        m = Image.new("L", (cell_w, cell_h), 0)
        ImageDraw.Draw(m).text((0, y_off), c, font=f, fill=255)
        a = np.asarray(m, np.int32)[..., None]
        tmpl[c] = ((fgv - bgv) * a + 127) // 255 + bgv   # same blend as the terminal renderer
    text, margin = "", math.inf
    for j in range(n):
        cell = obs[:cell_h, j * cell_w:(j + 1) * cell_w]
        c, e0, e1 = nearest(cell, tmpl)
        text += c
        margin = min(margin, e1 - e0)
    return text, margin


# ============================================================== question parsing
def parse_question(q: str) -> dict:
    if m := re.search(r"on the (\d+)(?:st|nd|rd|th) ERROR line", q):
        return {"kind": "logstream", "k": int(m.group(1))}
    if m := re.search(r"in the (\d+)(?:st|nd|rd|th) error dialog", q):
        return {"kind": "dialogs", "k": int(m.group(1))}
    if m := re.search(r'metric "([^"]+)" in the "Key metrics" table.*?"([^"]+)" service tile', q):
        return {"kind": "dashboard", "kpi": m.group(1), "tile": m.group(2)}
    if m := re.search(r"code was shown on the (\d+)(?:st|nd|rd|th) Error toast", q):
        return {"kind": "toasts", "k": int(m.group(1))}
    if m := re.search(r'the endpoint "([^"]+)" show in that table.*?"alerts fired" counter.*?reached (\d+)', q):
        return {"kind": "leaderboard", "endpoint": m.group(1), "k": int(m.group(2))}
    if m := re.search(r'the note titled "([^"]+)" has a table "([^"]+)".*?for "([^"]+)" in the column "([^"]+)" '
                      r'at 31 December (\d{4})', q):
        return {"kind": "scandoc", "title": m.group(1), "table": m.group(2), "row": m.group(3), "col": m.group(4),
                "year": int(m.group(5))}
    raise ValueError(f"unrecognised question: {q!r}")


def overview_kind(ov: np.ndarray) -> str:
    """Guess the screen kind from the 1/8 overview (sanity check against the question)."""
    top = ov[:4].reshape(-1, 3).mean(0)
    body = ov[20:].reshape(-1, 3).mean(0)
    if top[2] > top[0] + 12 and top[2] > 45:   # navy top bar: the Ledgerly app
        stripes = ov[16:90, 182:185].reshape(-1, 3)
        colourful = (stripes.max(-1) - stripes.min(-1) > 80).mean()
        return "toasts" if colourful > 0.2 else "dialogs"
    counter = ov[10:20, 152:236].reshape(-1, 3)
    if (np.abs(counter[:, 0] - counter[:, 1]) > 60).mean() > 0.6 and ov[10:20, 4:140].mean() < 60:
        return "leaderboard"        # big alerts counter top right, dark table on the left
    tiles = ov[12:32, 2:238].reshape(-1, 3)
    if (np.abs(tiles[:, 0] - tiles[:, 1]) > 60).mean() > 0.3:
        return "dashboard"          # a row of coloured status tiles
    return "logstream" if body.mean() < 80 else "unknown"


# ============================================================== logstream
class Log:
    COUNTER_BOX = (1180, 970, 730, 100)       # bottom-right "errors: N" box
    COUNTER_FONT = ("mono-bold", 80)
    LEVEL_COL, RID_COL, N_ROWS = 25, 47, 45  # terminal columns / visible rows
    CELL_W, CELL_H, Y_OFF = 10, 20, 1
    TEXT_X, TEXT_Y = 8, 44
    FG, BG = (204, 208, 214), (16, 18, 22)

    @classmethod
    def counter_templates(cls, out_width: int) -> dict:
        _x, _y, w, h = cls.COUNTER_BOX
        f = font(*cls.COUNTER_FONT)
        out = {}
        for n in range(21):
            im = Image.new("RGB", (w, h))
            d = ImageDraw.Draw(im)
            d.rectangle((0, 0, w - 1, h - 1), fill=(170, 20, 20) if n else (25, 95, 45))
            txt = f"errors: {n}"
            d.text((w - 24 - f.getlength(txt), -4), txt, font=f, fill=(255, 255, 255))
            out[n] = shrink(im, out_width)
        return out

    @classmethod
    def solve(cls, s: Session, q: dict) -> tuple[str, dict]:
        k, last = q["k"], s.n_frames - 1
        ow = cls.COUNTER_BOX[2] // 8
        tm = cls.counter_templates(ow)

        def errors_at(f: int) -> int:
            n, _e0, _e1 = nearest(s.view(f, cls.COUNTER_BOX, ow, f"read error counter (want >= {k})"), tm)
            return n

        f = first_true(0, last, lambda f: errors_at(f) >= k)
        # The K-th ERROR line arrived on frame f, so it is the newest ERROR line on screen.
        # Find its row from the colour of the level column (red = ERROR) at half scale.
        for n_rows in (16, cls.N_ROWS):
            r0 = cls.N_ROWS - n_rows
            box = (cls.TEXT_X + cls.LEVEL_COL * cls.CELL_W, cls.TEXT_Y + r0 * cls.CELL_H, 5 * cls.CELL_W,
                   n_rows * cls.CELL_H)
            strip = s.view(f, box, box[2] // 2, "find the newest ERROR row (red level column)")
            rh = strip.shape[0] / n_rows
            red = [int((strip[int(i * rh):int((i + 1) * rh), :, 0]
                        - strip[int(i * rh):int((i + 1) * rh), :, 1:].max(-1)).max()) for i in range(n_rows)]
            rows = [i for i, v in enumerate(red) if v > 70]
            if rows:
                row = r0 + rows[-1]
                break
        else:
            raise RuntimeError("no ERROR row visible")
        box = (cls.TEXT_X + cls.RID_COL * cls.CELL_W, cls.TEXT_Y + row * cls.CELL_H, 8 * cls.CELL_W, cls.CELL_H)
        crop = s.view(f, box, box[2], "read the request id at native resolution")
        rid, margin = ocr_grid(crop, 8, cls.CELL_W, cls.CELL_H, cls.Y_OFF, "mono", 16, cls.FG, cls.BG,
                               "0123456789abcdef")
        return rid.upper(), {"event_frame": f, "row": row, "ocr_margin": round(margin, 2)}


# ============================================================== dialogs
class Dialogs:
    TRACE_BOX = (1560, 8, 240, 96)            # bell + red badge, top bar
    BADGE_FONT = ("sans-bold", 76)
    CODE_FONT = ("mono-bold", 15)
    CODE_BOX_RGB = (241, 243, 245)
    INK = (33, 37, 41)
    NAVY = (32, 39, 56)
    CODE_REL = (142, 184)                     # code text origin relative to the dialog card
    DLG_W, DLG_H = 700, 300

    @classmethod
    def badge_templates(cls, out_width: int, n_max: int = 20) -> dict:
        _x, _y, w, h = cls.TRACE_BOX
        f = font(*cls.BADGE_FONT)
        bell = (230, 234, 242)
        out = {}
        for n in range(n_max + 1):
            im = Image.new("RGB", (w, h), cls.NAVY)
            d = ImageDraw.Draw(im)
            bx, by = 6, 18
            d.pieslice((bx, by, bx + 56, by + 60), 180, 360, fill=bell)
            d.rectangle((bx, by + 29, bx + 56, by + 50), fill=bell)
            d.polygon([(bx - 8, by + 56), (bx + 64, by + 56), (bx + 56, by + 46), (bx, by + 46)], fill=bell)
            d.ellipse((bx + 20, by + 54, bx + 36, by + 68), fill=bell)
            txt = str(n)
            tw = f.getlength(txt)
            pw = max(84, int(tw) + 36)
            d.rounded_rectangle((72, 2, 72 + pw, 94), 46, fill=(214, 40, 40) if n else (90, 98, 115))
            d.text((72 + (pw - tw) / 2, -1), txt, font=f, fill=(255, 255, 255))
            out[n] = shrink(im, out_width)
        return out

    @classmethod
    def solve(cls, s: Session, q: dict) -> tuple[str, dict]:
        k, last = q["k"], s.n_frames - 1
        ow = cls.TRACE_BOX[2] // 8
        tm = cls.badge_templates(ow)

        def badge_at(f: int) -> int:
            return nearest(s.view(f, cls.TRACE_BOX, ow, f"read bell badge (want >= {k})"), tm)[0]

        f = first_true(0, last, lambda f: badge_at(f) >= k)
        # Dialog K opened on frame f. Locate the white card at 1/16 scale: everything
        # behind a modal is dimmed, so the brightest blob below the top bar is the card.
        sc = 16
        ov = s.view(f, (0, 0, W, H), W // sc, "locate the open dialog card")
        bright = ov[112 // sc + 1:].min(-1) > 225
        ys, xs = np.nonzero(bright)
        if len(xs) < 20:
            raise RuntimeError("no dialog card visible")
        cx0 = int(xs.min()) * sc
        cy0 = (int(ys.min()) + 112 // sc + 1) * sc
        # Native crop around the expected code box, with margin for the 1/16 estimate.
        mx, my = 2 * sc, 2 * sc
        bx0, by0 = cls.CODE_REL[0] - 10, cls.CODE_REL[1] - 6
        box = (max(0, cx0 + bx0 - mx), max(0, cy0 + by0 - my), 75 + 2 * mx, 32 + 2 * my)
        crop = s.view(f, box, box[2], "read the error code at native resolution")
        mask = (np.abs(crop - np.asarray(cls.CODE_BOX_RGB)).max(-1) <= 1)
        # The box interior is a solid run of this grey; ignore stray anti-aliased pixels
        # elsewhere by requiring long runs per row/column.
        ys = np.nonzero(mask.sum(1) >= 40)[0]
        xs = np.nonzero(mask.sum(0) >= 12)[0]
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("code box not found in crop")
        ox, oy = int(xs.min()) + 9, int(ys.min()) + 5      # interior starts 1 px inside the outline
        best = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                code, margin = ocr_string(crop, (ox + dx, oy + dy), *cls.CODE_FONT, cls.INK, cls.CODE_BOX_RGB,
                                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", 6, 6)
                im = Image.new("RGB", (crop.shape[1], crop.shape[0]), cls.CODE_BOX_RGB)
                ImageDraw.Draw(im).text((ox + dx, oy + dy), code, font=font(*cls.CODE_FONT), fill=cls.INK)
                r = np.asarray(im, np.int32)
                y0, y1, x0, x1 = oy - 2, oy + 20, ox - 3, ox + 58
                err = float(np.abs(r[y0:y1, x0:x1] - crop[y0:y1, x0:x1]).sum())
                if best is None or err < best[0]:
                    best = (err, code, margin, (dx, dy))
        _, code, margin, d = best
        return code, {"event_frame": f, "card_xy_est": [int(cx0), int(cy0)], "origin_fix": d,
                      "ocr_margin": round(margin, 2)}


# ============================================================== dashboard
class Dash:
    N_TILES, TILE_Y, TILE_H, TILE_GAP = 6, 92, 170, 12
    TILE_W = (W - 32 - (N_TILES - 1) * TILE_GAP) // N_TILES
    TABLE_X, TABLE_Y, ROW_H, N_ROWS = 16, 290, 40, 8
    VALUE_X = TABLE_X + 380
    TEXT, PANEL, ALT = (204, 204, 220), (24, 27, 31), (29, 32, 37)

    @classmethod
    def tile_x(cls, i: int) -> int:
        return 16 + i * (cls.TILE_W + cls.TILE_GAP)

    @staticmethod
    def is_red(px: np.ndarray) -> bool:
        r, g, b = px.reshape(-1, 3).mean(0)
        return r > 150 and g < 90 and b < 110

    @classmethod
    def solve(cls, s: Session, q: dict, overview: np.ndarray) -> tuple[str, dict]:
        last = s.n_frames - 1
        # The named tile turns red and stays red; no other tile ever goes red. The 1/8
        # overview of the last frame shows which tile it is.
        sc = W / overview.shape[1]
        y0, y1 = int((cls.TILE_Y + 136) / sc), int((cls.TILE_Y + 162) / sc)
        red_tiles = [i for i in range(cls.N_TILES)
                     if cls.is_red(overview[y0:y1, int((cls.tile_x(i) + 20) / sc):int((cls.tile_x(i) + cls.TILE_W - 20) / sc)])]
        if len(red_tiles) != 1:
            raise RuntimeError(f"expected exactly one red tile on the last frame, found {red_tiles}")
        ti = red_tiles[0]
        band = (cls.tile_x(ti) + 10, cls.TILE_Y + 136, cls.TILE_W - 20, 26)   # text-free strip of the tile

        f = first_true(0, last, lambda f: cls.is_red(
            s.view(f, band, 24, f"is tile {ti} ({q['tile']}) red?")))

        # Which table row holds the KPI: match the rendered KPI name against each row label
        # at half scale (labels are static, so any frame works; use f).
        nb = (cls.TABLE_X + 8, cls.TABLE_Y + 44, 200, cls.N_ROWS * cls.ROW_H)
        names = s.view(f, nb, nb[2] // 2, f"find the row of \"{q['kpi']}\" in Key metrics")
        f16 = font("sans", 16)
        errs = []
        for r in range(cls.N_ROWS):
            im = Image.new("RGB", nb[2:], cls.PANEL)
            d = ImageDraw.Draw(im)
            for rr in range(cls.N_ROWS):
                if rr % 2:
                    d.rectangle((0, rr * cls.ROW_H, nb[2], rr * cls.ROW_H + cls.ROW_H - 1), fill=cls.ALT)
            d.text((cls.TABLE_X + 16 - nb[0], r * cls.ROW_H + 11), q["kpi"], font=f16, fill=cls.TEXT)
            t = shrink(im, nb[2] // 2)
            ry0, ry1 = int(r * cls.ROW_H / 2), int((r + 1) * cls.ROW_H / 2)
            errs.append((float(np.abs(t[ry0:ry1] - names[ry0:ry1]).mean()), r))
        errs.sort()
        row = errs[0][1]
        say(f"\"{q['title']}\" is row {row + 1} of the contents.")
        bg = cls.ALT if row % 2 else cls.PANEL
        vy = cls.TABLE_Y + 44 + row * cls.ROW_H + 11
        box = (cls.VALUE_X - 4, vy - 4, 79, 27)
        crop = s.view(f, box, box[2], "read the KPI value at native resolution")
        val, margin = ocr_string(crop, (4, 4), "mono", 14, cls.TEXT, bg, "0123456789", 1, 6)
        return val, {"event_frame": f, "tile_index": ti, "kpi_row": row,
                     "row_match_margin": round(errs[1][0] - errs[0][0], 2), "ocr_margin": round(margin, 2)}


# ============================================================== toasts
class Toasts:
    TRACE_BOX = Dialogs.TRACE_BOX             # same bell + badge as the dialogs app
    SLOT0 = (1456, 128, 440, 108)             # newest toast (top of the stack)
    STRIPE_W = 24
    CODE_BOX = (1456 + 100, 128 + 68, 74, 31)
    CODE_TEXT = (10, 6)
    CODE_FONT = ("mono-bold", 15)
    CODE_BOX_RGB = (241, 243, 245)
    INK = (33, 37, 41)
    ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"   # the stated alphabet (no I, O)

    @staticmethod
    def is_red(px: np.ndarray) -> bool:
        r, g, b = px.reshape(-1, 3).mean(0)
        return r > 150 and g < 100 and b < 100

    @classmethod
    def solve(cls, s: Session, q: dict) -> tuple[str, dict]:
        k, last = q["k"], s.n_frames - 1
        ow = cls.TRACE_BOX[2] // 8
        tm = Dialogs.badge_templates(ow, 99)

        def badge_at(f: int) -> int:
            return nearest(s.view(f, cls.TRACE_BOX, ow, f"read bell badge (want >= {k})"), tm)[0]

        f = first_true(0, last, lambda f: badge_at(f) >= k)
        # Error toast K arrived on frame f, so on f it is the newest toast: the top slot.
        # Check the top toast's colour stripe at 1/8 (red = Error) before reading.
        top = s.view(f, cls.SLOT0, cls.SLOT0[2] // 8, "check the top toast is red (Error) at 1/8")
        stripe = top[2:-2, 0:cls.STRIPE_W // 8]
        if not cls.is_red(stripe):
            raise RuntimeError("top toast is not an Error toast")
        mx, my = 4, 4
        x, y, w, h = cls.CODE_BOX
        box = (x - mx, y - my, w + 2 * mx, h + 2 * my)
        crop = s.view(f, box, box[2], "read the toast code at native resolution")
        code, margin = ocr_string(crop, (mx + cls.CODE_TEXT[0], my + cls.CODE_TEXT[1]), *cls.CODE_FONT,
                                  cls.INK, cls.CODE_BOX_RGB, cls.ALPHABET, 6, 6)
        return code, {"event_frame": f, "ocr_margin": round(margin, 2)}


# ============================================================== leaderboard
class Board:
    COUNTER_BOX = (1184, 68, 720, 104)        # "alerts fired: N", top right
    COUNTER_FONT = ("mono-bold", 64)
    DIGITS_W = 200                            # right part of the counter: just the number
    PANEL_X, ROW_Y0, ROW_H, N_ROWS = 16, 156, 46, 10
    NAME_DX, NAME_DY, VALUE_DX, VALUE_DY = 70, 12, 820, 16
    NAME_FONT = ("sans-bold", 18)
    VALUE_FONT = ("mono", 13)
    TEXT, PANEL, ALT = (204, 204, 220), (24, 27, 31), (29, 32, 37)
    NAMES_W = 340

    @classmethod
    def digits_box(cls) -> tuple[int, int, int, int]:
        x, y, w, h = cls.COUNTER_BOX
        return (x + w - cls.DIGITS_W, y, cls.DIGITS_W, h)

    @classmethod
    def counter_templates(cls, out_width: int) -> dict:
        _x, _y, w, h = cls.COUNTER_BOX
        f = font(*cls.COUNTER_FONT)
        out = {}
        for n in range(31):
            im = Image.new("RGB", (w, h), (170, 20, 20) if n else (25, 95, 45))
            txt = f"alerts fired: {n}"
            ImageDraw.Draw(im).text((w - 24 - f.getlength(txt), 12), txt, font=f, fill=(255, 255, 255))
            out[n] = shrink(im.crop((w - cls.DIGITS_W, 0, w, h)), out_width)
        return out

    @classmethod
    def find_row(cls, names: np.ndarray, endpoint: str, out_width: int) -> tuple[int, float]:
        """Row of `endpoint` in a low-res crop of the name column (template match per row)."""
        nb_x = cls.PANEL_X + cls.NAME_DX - 6
        size = (cls.NAMES_W, cls.N_ROWS * cls.ROW_H)
        f = font(*cls.NAME_FONT)
        rh = names.shape[0] / cls.N_ROWS
        errs = []
        for r in range(cls.N_ROWS):
            im = Image.new("RGB", size, cls.PANEL)
            d = ImageDraw.Draw(im)
            for rr in range(1, cls.N_ROWS, 2):
                d.rectangle((0, rr * cls.ROW_H, size[0], (rr + 1) * cls.ROW_H - 1), fill=cls.ALT)
            d.text((cls.PANEL_X + cls.NAME_DX - nb_x, r * cls.ROW_H + cls.NAME_DY), endpoint, font=f, fill=cls.TEXT)
            t = shrink(im, out_width)
            ry0, ry1 = int(r * rh), int((r + 1) * rh)
            errs.append((float(np.abs(t[ry0:ry1] - names[ry0:ry1]).mean()), r))
        errs.sort()
        return errs[0][1], errs[1][0] - errs[0][0]

    @classmethod
    def solve(cls, s: Session, q: dict) -> tuple[str, dict]:
        k, last = q["k"], s.n_frames - 1
        db = cls.digits_box()
        ow = db[2] // 8
        tm = cls.counter_templates(ow)

        def alerts_at(f: int) -> int:
            return nearest(s.view(f, db, ow, f"read alerts counter (want >= {k})"), tm)[0]

        f = first_true(0, last, lambda f: alerts_at(f) >= k)
        # The table re-sorts every half second: find the endpoint's row on this same frame,
        # from the name column at 1/4, then read its value there at native resolution.
        nb = (cls.PANEL_X + cls.NAME_DX - 6, cls.ROW_Y0, cls.NAMES_W, cls.N_ROWS * cls.ROW_H)
        now = nb[2] // 4
        names = s.view(f, nb, now, f"find the row of \"{q['endpoint']}\" at 1/4")
        row, row_margin = cls.find_row(names, q["endpoint"], now)
        bg = cls.ALT if row % 2 else cls.PANEL
        vx, vy = cls.PANEL_X + cls.VALUE_DX, cls.ROW_Y0 + row * cls.ROW_H + cls.VALUE_DY
        box = (vx - 4, vy - 4, 48, 26)
        crop = s.view(f, box, box[2], "read the errors/min value at native resolution")
        val, margin = ocr_string(crop, (4, 4), *cls.VALUE_FONT, cls.TEXT, bg, "0123456789", 3, 4)
        return val, {"event_frame": f, "row": row, "row_match_margin": round(row_margin, 2),
                     "ocr_margin": round(margin, 2)}


# ============================================================== scanned document
def fmt_amount_text(v: str) -> str:
    return v if len(v) <= 3 else v[:-3] + " " + v[-3:]


class Doc:
    """Scanned annual report (kind ``scandoc``): page index = frame index.

    Plan: find the Contents page (header title at 1/4), find the note's row in the
    contents by its title (title column at 1/4), read its printed page number (1/4),
    jump to that PDF index and correct by the printed page number found there until the
    two agree (the printed numbering is shifted by the front matter and by unnumbered
    inserted pages), then step through the note's pages reading only the table date
    (1/4) until the asked year, find the row (row labels at 1/4) and the column (column
    headers at 1/4), and read the amount from one native crop.

    Every page is a scan: shifted by up to 4 px and skewed by up to 0.12 degrees. Each
    low-res read therefore searches a small grid of shifts; the native read searches
    every 1-px shift around the estimate.
    """
    PW, PH = 1240, 1754
    ML, MR = 100, 1140
    CUE = ("sans-bold", 36)
    TOC_FONT = ("sans", 36)
    HEADER_Y, RULE_Y = 48, 104
    TOC_Y0, TOC_PITCH, TOC_MAX_ROWS, TOC_TITLE_X = 180, 60, 22, 290
    DATE_Y, COLHEAD_Y, ROW_Y0, ROW_PITCH, MAX_ROWS = 205, 300, 380, 64, 9
    COL_R = (820, 1140)
    AMOUNT_FONT, CELL, CELL_H, AMOUNT_DY = ("mono", 12), 8, 18, 12
    INK = 28
    SOFTEN = ImageFilter.Kernel((3, 3), (1, 2, 1, 2, 4, 2, 1, 2, 1), 16)
    SCALE = 4                                  # cue reads at 1/4
    LEFT_BOX = (80, 30, 440, 90)               # header title (+ left end of the rule)
    FOLIO_BOX = (900, 30, 260, 90)             # printed page number (+ right end of the rule)
    TITLES_W = 280                             # first ~14 characters of each contents title
    LABELS_W = 230

    # ---------------------------------------------------------------- imaging
    @staticmethod
    def norm(obs: np.ndarray) -> np.ndarray:
        """Grey levels mapped to 0 (ink) .. 1 (paper), per crop."""
        g = obs.astype(np.float32).mean(-1)
        paper = max(float(np.percentile(g, 90)), 120.0)
        return np.clip((g - 24.0) / (paper - 24.0), 0.0, 1.0)

    @classmethod
    def template(cls, box, items, shift=(0, 0), out_width: int | None = None) -> np.ndarray:
        """Render text items [(x, y, text, font, size, align)] and rectangles [("rect", x0, y0, x1, y1)]
        in page coordinates, shifted by `shift`, inside `box`; soften; downscale like the server."""
        bx, by, bw, bh = box
        dx, dy = shift
        im = Image.new("L", (bw, bh), 255)
        d = ImageDraw.Draw(im)
        for it in items:
            if it[0] == "rect":
                _, x0, y0, x1, y1 = it
                d.rectangle((x0 - bx + dx, y0 - by + dy, x1 - bx + dx, y1 - by + dy), fill=cls.INK)
                continue
            if it[0] == "dot":
                _, x, y = it
                d.ellipse((x - bx + dx, y - by + dy, x + 4 - bx + dx, y + 4 - by + dy), fill=120)
                continue
            x, y, text, fname, size, align = it
            f = font(fname, size)
            if align == "right":
                x -= f.getlength(text)
            d.text((x - bx + dx, y - by + dy), text, font=f, fill=cls.INK)
        im = im.filter(cls.SOFTEN)
        if out_width is not None and out_width < bw:
            im = im.resize(out_size(bw, bh, out_width), RESAMPLE)
        a = np.asarray(im, np.float32)
        return np.clip((a - cls.INK) / (255.0 - cls.INK), 0.0, 1.0)

    @staticmethod
    def err(a: np.ndarray, b: np.ndarray) -> float:
        h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        return float(np.abs(a[:h, :w] - b[:h, :w]).mean())

    SHIFTS = [(dx, dy) for dx in (-4, -2, 0, 2, 4) for dy in (-4, -2, 0, 2, 4)]

    @classmethod
    def best_shift(cls, obs: np.ndarray, box, items, out_width, shifts=None):
        """(error, shift) of the best-matching shift of a known layout."""
        return min((cls.err(obs, cls.template(box, items, sh, out_width)), sh) for sh in (shifts or cls.SHIFTS))

    @staticmethod
    def near(sh, r=1):
        return [(sh[0] + a, sh[1] + b) for a in range(-r, r + 1) for b in range(-r, r + 1)]

    # ---------------------------------------------------------------- reads
    @classmethod
    def read_header_left(cls, s: Session, page: int, titles: tuple[str, ...], purpose: str):
        """Which of `titles` (or no title) is at the top left of `page`? Returns (title or None, shift)."""
        box, ow = cls.LEFT_BOX, cls.LEFT_BOX[2] // cls.SCALE
        obs = cls.norm(s.view(page, box, ow, purpose))
        rule = ("rect", cls.ML, cls.RULE_Y, cls.MR, cls.RULE_Y + 2)
        best = min(cls.best_shift(obs, box, [rule, (cls.ML, cls.HEADER_Y, t, *cls.CUE, "left")], ow) + (t,)
                   for t in titles)
        e0 = min(cls.err(obs, cls.template(box, its, (0, 0), ow)) for its in ([], [rule]))
        return (best[2] if best[0] < 0.75 * e0 else None), best[1]

    @classmethod
    def read_folio(cls, s: Session, page: int, purpose: str, max_folio: int = 520):
        """Printed (arabic) page number at the top right, or None (no number: an inserted page)."""
        box, ow = cls.FOLIO_BOX, cls.FOLIO_BOX[2] // cls.SCALE
        obs = cls.norm(s.view(page, box, ow, purpose))
        rule = ("rect", cls.ML, cls.RULE_Y, cls.MR, cls.RULE_Y + 2)
        e_rule, sh = cls.best_shift(obs, box, [rule], ow)
        e_none = cls.err(obs, cls.template(box, [], (0, 0), ow))
        if e_rule >= e_none:           # no header rule: unnumbered page
            return None, sh
        def items(n):
            return [rule, (cls.MR, cls.HEADER_Y, str(n), *cls.CUE, "right")]

        top = sorted((cls.err(obs, cls.template(box, items(n), sh, ow)), n) for n in range(1, max_folio + 1))[:5]
        _, sh = cls.best_shift(obs, box, items(top[0][1]), ow, cls.near(sh))
        _, n = min((cls.err(obs, cls.template(box, items(m), sh, ow)), m) for _, m in top)
        return n, sh

    @classmethod
    def toc_row_items(cls, folio: str, title_end: float, k: int):
        y = cls.TOC_Y0 + k * cls.TOC_PITCH
        items = [(cls.MR, y, folio, *cls.CUE, "right")]
        x1 = cls.MR - font(*cls.CUE).getlength(folio) - 16
        x = title_end + 16
        x += (-x) % 18
        while x < x1:
            items.append(("dot", x, y + 30))
            x += 18
        return items

    @classmethod
    def solve(cls, s: Session, q: dict) -> tuple[str, dict]:
        # `say` (optional, used by the SFT teacher) receives what each look showed.
        say = getattr(s, "say", None) or (lambda msg: None)
        # 1. the Contents page: within the first few pages, header "Contents" at the top left
        toc_page, toc_shift = None, (0, 0)
        for p in range(1, 4):           # cover, then optional notice and blank pages
            t, sh = cls.read_header_left(s, p, ("Contents", "Important notice"),
                                         f"Is PDF page {p} the Contents page? Its header title (top left) at 1/4 scale.")
            say(f"PDF page {p}: the header reads \"{t}\"." if t else f"PDF page {p} has no header title.")
            if t == "Contents":
                toc_page, toc_shift = p, sh
                break
        if toc_page is None:
            raise RuntimeError("Contents page not found")
        # 2. the note's row: its title in the title column at 1/4
        nrows = cls.TOC_MAX_ROWS
        box = (cls.TOC_TITLE_X - 10, cls.TOC_Y0 - 10, cls.TITLES_W, nrows * cls.TOC_PITCH + 10)
        ow = box[2] // cls.SCALE
        obs = cls.norm(s.view(toc_page, box, ow, f"find \"{q['title']}\" in the contents (title column at 1/4)"))
        rh = cls.TOC_PITCH / cls.SCALE
        errs = []
        for k in range(nrows):
            y0, y1 = int(k * rh), int((k + 1) * rh) + 2
            e = min(cls.err(obs[y0:y1], cls.template(box, [(cls.TOC_TITLE_X, cls.TOC_Y0 + k * cls.TOC_PITCH, q["title"], *cls.TOC_FONT, "left")], sh, ow)[y0:y1])
                    for sh in cls.near(toc_shift, 2)[::2])
            errs.append((e, k))
        errs.sort()
        row = errs[0][1]
        # 3. its printed page number (right column, 1/4)
        yr = cls.TOC_Y0 + row * cls.TOC_PITCH
        fbox = (cls.MR - 150, yr - 10, 170, 60)
        fow = fbox[2] // cls.SCALE
        fobs = cls.norm(s.view(toc_page, fbox, fow, "read the note's printed page number in the contents (1/4)"))
        title_end = cls.TOC_TITLE_X + font(*cls.TOC_FONT).getlength(q["title"])
        printed = min((cls.err(fobs, cls.template(fbox, cls.toc_row_items(str(n), title_end, row), sh, fow)), n)
                      for n in range(1, 520) for sh in cls.near(toc_shift, 2)[::4])[1]
        say(f"The contents give printed page {printed} for this note. Printed numbers are not PDF indexes, "
            "so I jump and correct by the number printed on the page I land on.")
        # 4. jump: printed page numbers are not PDF indexes; correct by the number found
        idx, hops = printed, []
        for _ in range(10):
            idx = min(max(idx, 0), s.n_frames - 1)
            folio, sh = cls.read_folio(s, idx, f"printed page number at PDF page {idx} (want {printed})")
            hops.append((idx, folio))
            say(f"PDF page {idx} " + ("has no printed page number (an unnumbered inserted sheet)." if folio is None
                                      else f"is printed page {folio}" + ("." if folio == printed else
                                                                         f"; I want {printed}, so move {printed - folio:+d}.")))
            if folio == printed:
                break
            idx += 1 if folio is None else printed - folio
        else:
            raise RuntimeError(f"could not reach printed page {printed}: {hops}")
        note_page = idx
        # 5. the table for the asked year: read only the date line of the following pages
        years = (q["year"], q["year"] + 1, q["year"] - 1)
        dbox = (cls.ML - 10, cls.DATE_Y - 10, 440, 60)
        dow = dbox[2] // cls.SCALE
        table_page, seen = None, []
        for p in range(note_page, min(note_page + 8, s.n_frames)):
            dobs = cls.norm(s.view(p, dbox, dow, f"table date on PDF page {p} (want 31 December {q['year']})"))
            res = {y: cls.best_shift(dobs, dbox, [(cls.ML, cls.DATE_Y, f"31 December {y}", *cls.CUE, "left")], dow)
                   for y in years}
            e_blank = cls.err(dobs, cls.template(dbox, [], (0, 0), dow))
            ybest = min(res, key=lambda y: res[y][0])
            seen.append((p, ybest, round(res[ybest][0], 3), round(e_blank, 3)))
            say(f"PDF page {p}: " + (f"a table dated 31 December {ybest}." if res[ybest][0] < 0.6 * e_blank
                                     else "no table date (a text page)."))
            if ybest == q["year"] and res[ybest][0] < 0.6 * e_blank:
                table_page, tshift = p, res[ybest][1]
                break
        if table_page is None:
            raise RuntimeError(f"table for {q['year']} not found: {seen}")
        # 6. the row (labels at 1/4) and the column (headers at 1/4)
        lbox = (cls.ML - 10, cls.ROW_Y0 - 10, cls.LABELS_W, cls.MAX_ROWS * cls.ROW_PITCH)
        low = lbox[2] // cls.SCALE
        lobs = cls.norm(s.view(table_page, lbox, low, f"find the row \"{q['row']}\" (row labels at 1/4)"))
        rh = cls.ROW_PITCH / cls.SCALE
        rerrs = []
        for k in range(cls.MAX_ROWS - 1):
            y0, y1 = int(k * rh), int((k + 1) * rh)
            e = min(cls.err(lobs[y0:y1], cls.template(lbox, [(cls.ML, cls.ROW_Y0 + k * cls.ROW_PITCH, q["row"], "sans", 36, "left")], sh, low)[y0:y1])
                    for sh in cls.near(tshift, 2)[::2])
            rerrs.append((e, k))
        rerrs.sort()
        trow = rerrs[0][1]
        say(f"\"{q['row']}\" is row {trow + 1} of the table.")
        cbox = (500, cls.COLHEAD_Y - 10, 660, 60)
        cow = cbox[2] // cls.SCALE
        cobs = cls.norm(s.view(table_page, cbox, cow, f"which column is \"{q['col']}\"? (headers at 1/4)"))
        cerrs = sorted((min(cls.err(cobs, cls.template(cbox, [(cls.COL_R[c], cls.COLHEAD_Y, q["col"], *cls.CUE, "right")], sh, cow))
                            for sh in cls.near(tshift, 2)[::2]), c) for c in (0, 1))
        col = cerrs[0][1]
        say(f"\"{q['col']}\" is the {('left', 'right')[col]} amount column.")
        # 7. one native crop of the amount
        ry = cls.ROW_Y0 + trow * cls.ROW_PITCH + cls.AMOUNT_DY
        n_cells = 7
        m = 8
        abox = (cls.COL_R[col] - n_cells * cls.CELL - m, ry - m, n_cells * cls.CELL + 2 * m, cls.CELL_H + 6 + 2 * m)
        aobs = cls.norm(s.view(table_page, abox, abox[2], "read the amount at native resolution"))
        val, margin = cls.ocr_amount(aobs, abox, cls.COL_R[col], ry)
        say(f"The native crop reads {fmt_amount_text(val)}.")
        return val, {"toc_page": toc_page, "toc_row": row, "printed_page": printed, "hops": hops,
                     "note_page": note_page, "table_page": table_page, "row": trow, "col": col,
                     "ocr_margin": round(margin, 3)}

    _cells: dict = {}

    @classmethod
    def cell_templates(cls) -> dict:
        if not cls._cells:
            f = font(*cls.AMOUNT_FONT)
            for ch in "0123456789 ":
                im = Image.new("L", (cls.CELL + 6, cls.CELL_H + 6), 255)
                ImageDraw.Draw(im).text((3, 3), ch.strip(), font=f, fill=cls.INK)
                a = np.asarray(im.filter(cls.SOFTEN), np.float32)
                cls._cells[ch] = np.clip((a - cls.INK) / (255.0 - cls.INK), 0, 1)[3:3 + cls.CELL_H, 3:3 + cls.CELL]
        return cls._cells

    @classmethod
    def ocr_amount(cls, obs: np.ndarray, box, right: int, top: int) -> tuple[str, float]:
        """Read 'ddd ddd' right-aligned at `right` (one glyph per CELL-px cell), searching 1-px shifts."""
        tm = cls.cell_templates()
        best = None
        for dy in range(-6, 7):
            for dx in range(-6, 7):
                y0 = top - box[1] + dy
                if y0 < 0 or y0 + cls.CELL_H > obs.shape[0]:
                    continue
                if right - 7 * cls.CELL - box[0] + dx < 0 or right - box[0] + dx > obs.shape[1]:
                    continue
                total, picks = 0.0, []
                for k in range(1, 8):          # cells from the right
                    x0 = right - k * cls.CELL - box[0] + dx
                    cell = obs[y0:y0 + cls.CELL_H, x0:x0 + cls.CELL]
                    errs = sorted((float(np.abs(cell - t).mean()), c) for c, t in tm.items())
                    if k == 4:                 # the thousands separator is always blank
                        total += float(np.abs(cell - tm[" "]).mean())
                        picks.append((" ", math.inf))
                        continue
                    if k <= 3:                 # always digits
                        errs = [e for e in errs if e[1] != " "]
                    total += errs[0][0]
                    picks.append((errs[0][1], errs[1][0] - errs[0][0]))
                text, margin = "", math.inf
                for k, (c, mg) in enumerate(picks, start=1):
                    if k == 4:
                        continue
                    if c == " ":
                        break
                    text = c + text
                    margin = min(margin, mg)
                if best is None or total < best[0]:
                    best = (total, text, margin)
        return best[1], best[2]


# ============================================================== main
def run(url: str, out=sys.stdout) -> dict:
    t0 = time.time()
    c = MCPClient(url)
    try:
        return _run(c, out, t0)
    finally:
        c.close()


def _run(c: MCPClient, out, t0: float) -> dict:
    s = Session(c, out=out)
    info = s.info()
    q = parse_question(info["question"])
    if q["kind"] == "scandoc":
        answer, note = Doc.solve(s, q)
        note = {"kind": q["kind"], **note}
        return _finish(s, info, q, answer, note, out, t0)
    ov = s.view(s.n_frames - 1, (0, 0, W, H), W // 8, "overview of the last frame at 1/8")
    seen = overview_kind(ov)
    if q["kind"] == "logstream":
        answer, note = Log.solve(s, q)
    elif q["kind"] == "dialogs":
        answer, note = Dialogs.solve(s, q)
    elif q["kind"] == "toasts":
        answer, note = Toasts.solve(s, q)
    elif q["kind"] == "leaderboard":
        answer, note = Board.solve(s, q)
    else:
        answer, note = Dash.solve(s, q, ov)
    note = {"kind": q["kind"], "overview_kind": seen, **note}
    return _finish(s, info, q, answer, note, out, t0)


def _finish(s: Session, info: dict, q: dict, answer: str, note: dict, out, t0: float) -> dict:
    s.submit(answer, note)
    summary = {"summary": True, "kind": q["kind"], "answer": answer, "pixels_used": s.pixels_used,
               "budget_total": int(info["budget_total"]), "n_calls": len(s.trace),
               "seconds": round(time.time() - t0, 2), **note}
    if out is not None:
        print(json.dumps(summary), file=out, flush=True)
    return {"summary": summary, "trace": s.trace}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FrameHunt reference policy (MCP tools only).")
    ap.add_argument("--url", default=os.environ.get("FH_MCP_URL", "http://fh-mcp:8000/mcp"))
    ap.add_argument("--trace", type=Path, default=None, help="also write the full trace as JSON here")
    a = ap.parse_args(argv)
    res = run(a.url)
    if a.trace:
        a.trace.parent.mkdir(parents=True, exist_ok=True)
        a.trace.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
