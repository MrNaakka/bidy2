"""`logstream`: a terminal tailing a service log.

Question: request id on the K-th ERROR line. Trace: the big `errors: N` counter in
the status bar (counts ERROR lines printed so far). Distractors: WARN lines with ids;
every line carries a request id, and all ids in a scene are unique.
"""
from __future__ import annotations

import bisect
import datetime as dt
from functools import cached_property

import numpy as np
from PIL import Image, ImageDraw
from stretchable.style import FlexDirection

from .base import GlyphAtlas, Scene, SceneMeta, draw_text, layout, node, pick_event_frames

W, H = 1920, 1080
TITLE_H = 36
STATUS_H = 120
LINE_FONT = ("mono", 16)          # answer text: glyphs ~12 px tall native, ~3 px at 1/4
LINE_H = 20                       # terminal cell 10 x 20 px
CELL_W = 10
GLYPH_Y_OFF = 1
N_COLS = 128
COUNTER_FONT = ("mono-bold", 80)  # trace: digits ~58 px native, ~7 px at 1/8

L = layout(node(
    node(key="title", size=(W, TITLE_H)),
    node(node(key="text", flex_grow=1), key="term", flex_grow=1, padding=(8, 0, 0, 8)),
    node(node(key="counter", size=(730, STATUS_H - 20)),
         key="status", size=(W, STATUS_H), padding=(10, 10, 10, 1180)),
    size=(W, H), flex_direction=FlexDirection.COLUMN))
N_VISIBLE = L["text"].h // LINE_H   # 45 rows
COUNTER_BOX = L["counter"]

BG = (16, 18, 22)
FG = (204, 208, 214)
DIM = (120, 128, 140)
LEVELS = ["INFO", "DEBUG", "WARN", "ERROR"]
LEVEL_COL = {"INFO": (110, 190, 255), "DEBUG": (150, 150, 160),
             "WARN": (240, 200, 60), "ERROR": (255, 85, 85)}
PALETTE = np.array([FG, DIM] + [LEVEL_COL[lv] for lv in LEVELS], np.uint8)
C_FG, C_DIM = 0, 1
LEVEL_IDX = {lv: 2 + k for k, lv in enumerate(LEVELS)}
HEX = "0123456789abcdef"

COMPONENTS = ["http", "orders", "payments", "db.pool", "cache", "auth", "inventory",
              "kafka", "scheduler", "grpc"]
INFO_MSGS = [
    "GET /api/v2/orders/{n} 200 {ms}ms", "POST /api/v2/cart 201 {ms}ms",
    "order {n} state PENDING -> CONFIRMED", "cache hit ratio={p}% keys={n}",
    "published event OrderCreated offset={n}", "user session refreshed uid={n}",
    "GET /healthz 200 {ms}ms", "batch flushed rows={n} took={ms}ms",
    "reserved stock sku={n} qty={q}", "PUT /api/v2/orders/{n}/address 200 {ms}ms",
]
DEBUG_MSGS = [
    "pool stats active={q} idle={q} waiting=0", "acquired lock order:{n}",
    "span finished name=checkout dur={ms}ms", "retry budget remaining={q}",
    "decoded payload bytes={n}", "gc pause={q}ms heap={n}KB",
]
WARN_MSGS = [
    "slow query took {ms}ms (threshold 500ms)", "retrying upstream call attempt {q}/3",
    "payment provider latency high p99={ms}ms", "cache miss storm keys={n}",
    "deprecated header X-Client-Ver used", "connection pool near limit {q}/32",
    "request failed, will retry: upstream timeout",
]
ERROR_MSGS = [
    "payment gateway timeout after {ms}ms", "failed to commit order {n}: deadlock detected",
    "upstream inventory returned 503", "unhandled exception in checkout handler",
    "kafka produce failed: NOT_LEADER_FOR_PARTITION", "db connection reset by peer",
]


def _ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class LogStreamScene(Scene):
    kind = "logstream"

    def __init__(self, seed: int):
        super().__init__(seed)
        r = self.rng("setup")
        n_frames = r.randint(1800, 4800)
        self.meta = SceneMeta(W, H, self.FPS, n_frames)
        self.service = r.choice(["orders-svc", "checkout-api", "billing-svc", "cart-svc"])
        self.host = f"api-prod-{r.randint(1, 24):02d}"
        self.base_time = dt.datetime(2026, r.randint(1, 12), r.randint(1, 28),
                                     r.randint(0, 23), r.randint(0, 59), r.randint(0, 59))
        self._build_timeline(n_frames)
        self._build_lines()

    # --- timeline ----------------------------------------------------------------
    def _build_timeline(self, n_frames: int) -> None:
        fps = self.FPS
        for attempt in range(100):
            rr = self.rng(f"stream/{attempt}")
            rate = rr.uniform(3.0, 5.0)            # mean lines per second
            arrivals = [0] * N_VISIBLE              # screen is full at frame 0
            t = 0.0
            while True:
                if rr.random() < 0.25:
                    t += rr.uniform(0.0, 0.08)      # burst
                else:
                    t += rr.expovariate(rate * 1.25)
                f = int(t * fps)
                if f >= n_frames:
                    break
                arrivals.append(max(f, 1))
            n_err = rr.randint(3, 9)
            err_frames = pick_event_frames(rr, n_err, int(0.1 * n_frames) + 1,
                                           int(0.9 * n_frames) - 1, 3 * fps)
            err_lines: list[int] = []
            ok = True
            for ef in err_frames:
                i = bisect.bisect_left(arrivals, ef)
                if i >= len(arrivals) or (err_lines and arrivals[i] <= arrivals[err_lines[-1]]):
                    ok = False
                    break
                err_lines.append(i)
            for i in err_lines if ok else []:
                off = arrivals[i + N_VISIBLE] if i + N_VISIBLE < len(arrivals) else n_frames
                if off - arrivals[i] < 6 * fps + 1:
                    ok = False                      # ERROR line must stay >= 6 s
                if not (0.1 * n_frames <= arrivals[i] <= 0.9 * n_frames):
                    ok = False
            if ok:
                break
        else:  # pragma: no cover
            raise RuntimeError("could not build log stream")
        self.rate = rate
        self.arrivals = arrivals
        self.error_lines = err_lines
        self.error_frames = [arrivals[i] for i in err_lines]

    def _build_lines(self) -> None:
        rc = self.rng("content")
        err_set = set(self.error_lines)
        ids: set[str] = set()
        self.lines: list[tuple[str, str, str, str]] = []   # (level, component, rid, msg)
        n = len(self.arrivals)
        for i in range(n):
            level = "ERROR" if i in err_set else rc.choices(["INFO", "DEBUG", "WARN"], [60, 25, 15])[0]
            while True:
                rid = "".join(rc.choice(HEX) for _ in range(8))
                if rid not in ids:
                    ids.add(rid)
                    break
            msgs = {"INFO": INFO_MSGS, "DEBUG": DEBUG_MSGS, "WARN": WARN_MSGS,
                    "ERROR": ERROR_MSGS}[level]
            msg = rc.choice(msgs).format(n=rc.randint(1000, 99999), ms=rc.randint(2, 4000),
                                         p=rc.randint(50, 99), q=rc.randint(1, 9))
            self.lines.append((level, rc.choice(COMPONENTS), rid, msg))

        self.k = rc.randint(1, len(self.error_lines))
        self.answer = self.lines[self.error_lines[self.k - 1]][2].upper()
        self.question = (
            f"The terminal tails the log of {self.service}. What is the request id "
            f"(the 8 hex characters after 'req=') on the {_ordinal(self.k)} ERROR line "
            f"printed in the log? The 'errors: N' counter in the bottom status bar counts "
            f"ERROR lines printed so far."
        )
        self.answer_format = "8 characters, hex digits 0-9 and a-f (case-insensitive)"

        self._rid_col = len("2026-01-01T00:00:00.000Z ") + len("ERROR ") + len("[scheduler] req=")
        self._atlas = GlyphAtlas.get(*LINE_FONT, CELL_W, LINE_H, GLYPH_Y_OFF)

    @cached_property
    def _grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Every line as fixed-width char codes + per-char colour index (built lazily)."""
        n = len(self.lines)
        codes = np.full((n, N_COLS), 32, np.uint8)
        cidx = np.zeros((n, N_COLS), np.uint8)
        jit = self.rng("ms")
        prev_ms = -1
        for i, (level, comp, rid, msg) in enumerate(self.lines):
            ms = max(prev_ms + 1, self.arrivals[i] * 100 + jit.randint(0, 99))
            prev_ms = ms                               # monotonic timestamps
            t = self.base_time + dt.timedelta(milliseconds=ms - 4000)
            ts = t.strftime("%Y-%m-%dT%H:%M:%S") + f".{t.microsecond // 1000:03d}Z"
            segs = [(ts + " ", C_DIM), (f"{level:<5} ", LEVEL_IDX[level]),
                    (f"[{comp:<9}] req=", C_DIM), (rid, C_FG), (" " + msg, C_FG)]
            col = 0
            for k, (text, c) in enumerate(segs):
                if k == 3:
                    assert col == self._rid_col
                text = text[:N_COLS - col]
                codes[i, col:col + len(text)] = np.frombuffer(text.encode("ascii"), np.uint8)
                cidx[i, col:col + len(text)] = c
                col += len(text)
        codes.setflags(write=False)
        cidx.setflags(write=False)
        return codes, cidx

    # --- rendering ---------------------------------------------------------------
    @cached_property
    def _background(self) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        title_bar, status = L["title"], L["status"]
        d.rectangle(title_bar.xyxy, fill=(44, 46, 52))
        for k, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse((14 + k * 24, 11, 28 + k * 24, 25), fill=c)
        title = f"deploy@{self.host}: ~ - tail -F /var/log/{self.service}/{self.service}.log"
        draw_text(d, (title_bar.center_x, 9), title, "sans", 15, (200, 200, 205), "center")
        d.rectangle(status.edges, fill=(30, 33, 40))
        d.line((status.x, status.y, status.right, status.y), fill=(70, 74, 84), width=2)
        draw_text(d, status.at(20, 18), f"[{self.service}] 0:tail*  1:bash  2:htop",
                  "mono", 18, (150, 200, 150))
        return im

    def errors_at(self, frame: int) -> int:
        return bisect.bisect_right(self.error_frames, frame)

    def visible_lines(self, frame: int) -> range:
        last = bisect.bisect_right(self.arrivals, frame)    # lines [0, last) printed
        return range(max(0, last - N_VISIBLE), last)

    def _row_xy(self, row: int, col: int) -> tuple[int, int]:
        return L["text"].at(col * CELL_W, row * LINE_H)

    def render(self, frame: int) -> Image.Image:
        frame = self._check_frame(frame)
        im = self._background.copy()
        vis = self.visible_lines(frame)
        codes, cidx = self._grid
        block = self._atlas.compose(codes[vis.start:vis.stop], cidx[vis.start:vis.stop],
                                    PALETTE, BG)
        im.paste(Image.fromarray(block), self._row_xy(0, 0))
        d = ImageDraw.Draw(im)
        now = self.base_time + dt.timedelta(seconds=frame / self.FPS)
        draw_text(d, L["status"].at(20, 62),
                  f"{self.host}  lines: {vis.stop:6d}  {now.strftime('%H:%M:%S')}",
                  "mono", 18, (170, 175, 185))
        # The trace: big, high-contrast error counter (bottom right).
        n = self.errors_at(frame)
        d.rectangle(COUNTER_BOX.xyxy, fill=(170, 20, 20) if n else (25, 95, 45))
        draw_text(d, (COUNTER_BOX.right - 24, COUNTER_BOX.y - 4), f"errors: {n}", *COUNTER_FONT,
                  (255, 255, 255), "right")
        return im

    def debug_info(self) -> dict:
        errs = []
        for k, i in enumerate(self.error_lines, start=1):
            a = self.arrivals[i]
            off = self.arrivals[i + N_VISIBLE] if i + N_VISIBLE < len(self.arrivals) else self.meta.n_frames
            row = i - self.visible_lines(a).start
            errs.append({"k": k, "line_index": i, "request_id": self.lines[i][2],
                         "appear_frame": a, "scroll_off_frame": off,
                         "rid_xy_at_appear": self._row_xy(row, self._rid_col)})
        key = errs[self.k - 1]
        return {
            "kind": self.kind, "seed": self.seed, "k": self.k, "n_errors": len(errs),
            "lines_per_s": self.rate, "n_lines": len(self.lines), "visible_rows": N_VISIBLE,
            "errors": errs,
            "event_frame": key["appear_frame"],
            "answer_text": {"frame": key["appear_frame"], "xy": key["rid_xy_at_appear"],
                            "text": key["request_id"], "font": LINE_FONT[0],
                            "size": LINE_FONT[1], "fill": FG, "bg": BG,
                            "grid_cell": (CELL_W, LINE_H, GLYPH_Y_OFF)},
            "trace": {"box": COUNTER_BOX, "font_px": COUNTER_FONT[1],
                      "change_frames": list(self.error_frames),
                      "value_at_event": self.k},
            "answer_font_px": LINE_FONT[1],
        }
