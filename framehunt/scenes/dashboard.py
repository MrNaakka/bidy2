"""`dashboard`: a Grafana-like monitoring dashboard.

Question: what KPI X (in the "Key metrics" table) read on the first frame where service
tile S is red. Trace: tile S turns red (DOWN) and stays red. Distractors: other tiles
blink amber (DEGRADED, never red), S itself may go amber before it goes red, KPI values
change every 2 s, and big stat panels show other changing numbers.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
from PIL import Image, ImageDraw

from .base import Scene, SceneMeta, draw_text, text_width

W, H = 1920, 1080
NAV_H = 52
TICK = 20                       # KPI update period in frames (2 s)
N_TILES = 6
TILE_Y, TILE_H = 92, 170
TILE_GAP = 12
TILE_W = (W - 32 - (N_TILES - 1) * TILE_GAP) // N_TILES
TILE_NAME_FONT = ("sans-bold", 30)
TILE_STATUS_FONT = ("sans-bold", 44)
VALUE_FONT = ("mono", 14)       # answer text: digits ~10 px native, <3 px at 1/4
TABLE = (16, 290, 640, 410)     # x, y, w, h of the Key metrics panel
ROW_H = 40
VALUE_X = TABLE[0] + 380

BG = (17, 18, 23)
PANEL = (24, 27, 31)
BORDER = (44, 48, 56)
TEXT = (204, 204, 220)
MUTED = (140, 145, 160)
GREEN = (55, 135, 45)
AMBER = (224, 155, 0)
RED = (212, 34, 50)
SERIES = [(115, 191, 105), (87, 148, 242), (255, 152, 48), (184, 119, 217)]

SERVICES = ["checkout", "payments", "search", "auth", "inventory", "shipping", "catalog",
            "pricing", "notifications", "recommender", "billing", "gateway", "accounts",
            "media", "reviews", "loyalty"]
KPIS = [("Requests / s", "req/s"), ("p95 latency", "ms"), ("Active sessions", ""),
        ("Queue depth", "msgs"), ("Orders / min", ""), ("DB connections", ""),
        ("Cache entries", "k"), ("Payments / min", ""), ("Open carts", ""),
        ("Kafka lag", "msgs"), ("Worker threads", ""), ("Bytes out", "KB/s")]


def _walk(rng: np.random.Generator, n: int, lo: int, hi: int) -> np.ndarray:
    """Integer random walk reflected into [lo, hi] (reflection avoids piling up at bounds)."""
    span = hi - lo
    steps = rng.normal(0, span * 0.04, n)
    x = np.empty(n, np.int64)
    v = float(rng.uniform(lo + span / 4, hi - span / 4))
    for i in range(n):
        v += steps[i]
        if v > hi:
            v = 2 * hi - v
        if v < lo:
            v = 2 * lo - v
        x[i] = int(round(min(max(v, lo), hi)))
    return x


class DashboardScene(Scene):
    kind = "dashboard"

    def __init__(self, seed: int):
        super().__init__(seed)
        r = self.rng("setup")
        fps = self.FPS
        n_frames = r.randint(1800, 4800)
        self.meta = SceneMeta(W, H, fps, n_frames)
        self.base_time = dt.datetime(2026, r.randint(1, 12), r.randint(1, 28),
                                     r.randint(0, 23), r.randint(0, 59), r.randint(0, 59))
        self.tiles = r.sample(SERVICES, N_TILES)
        self.kpis = r.sample(KPIS, 8)
        nrng = np.random.default_rng(self.np_seed("values"))

        # --- event: tile S goes red at f_red ------------------------------------------
        rt = self.rng("timeline")
        self.s_idx = rt.randrange(N_TILES)
        self.x_idx = rt.randrange(len(self.kpis))
        self.f_red = rt.randint(int(0.1 * n_frames) + 40, int(0.9 * n_frames) - 40)

        # Tile states per frame: 0 green, 1 amber, 2 red. Stored as change intervals.
        self.amber: list[list[tuple[int, int]]] = [[] for _ in range(N_TILES)]
        if rt.random() < 0.6:   # S degrades first
            d = rt.randint(3 * fps, 20 * fps)
            self.amber[self.s_idx].append((self.f_red - d, self.f_red))
        others = [i for i in range(N_TILES) if i != self.s_idx]
        for i in rt.sample(others, rt.randint(1, 3)):
            for _ in range(rt.randint(1, 3)):
                a = rt.randint(0, n_frames - 30 * fps)
                self.amber[i].append((a, a + rt.randint(3 * fps, 15 * fps)))

        # --- KPI table: each KPI ticks every 2 s with its own phase ----------------
        n_ticks = n_frames // TICK + 3
        self.kpi_phase = [rt.randrange(TICK) for _ in self.kpis]
        self.kpi_vals = []
        for k, _ in enumerate(self.kpis):
            digits = rt.randint(3, 5)
            lo, hi = 10 ** (digits - 1), 10 ** digits - 1
            self.kpi_vals.append(_walk(nrng, n_ticks, lo, hi))
        # Target KPI holds the answer for 40 frames [s, s+39] with f_red at offset 10..29.
        o = rt.randint(10, 29)
        s = self.f_red - o
        self.kpi_phase[self.x_idx] = s % TICK
        j = self._tick_index(self.x_idx, s)
        vals = self.kpi_vals[self.x_idx]
        vals[j + 1] = vals[j]
        lo, hi = int(vals.min()), int(vals.max())
        for jj in (j - 1, j + 2):     # neighbours must differ, so the window is exact
            if jj >= 0 and jj < len(vals) and vals[jj] == vals[j]:
                vals[jj] = vals[j] + 1 if vals[j] < 10 ** len(str(vals[j])) - 1 else vals[j] - 1
        self.kpi_window = (s, s + 2 * TICK - 1)
        self.answer = str(int(vals[j]))

        # --- charts & stat panels (global 2 s tick) ----------------------------------
        hist = 120
        self.g_phase = rt.randrange(TICK)
        self.series = [nrng.normal(0, 1, n_ticks + hist).cumsum() for _ in range(5)]
        self.stats = [
            ("CPU usage", _walk(nrng, n_ticks, 20, 95), "%"),
            ("Pods running", _walk(nrng, n_ticks, 80, 240), ""),
            ("Memory", _walk(nrng, n_ticks, 30, 90), "%"),
            ("Nodes ready", _walk(nrng, n_ticks, 8, 40), ""),
        ]
        self.hist = hist

        S_name, X_name = self.tiles[self.s_idx], self.kpis[self.x_idx][0]
        self.question = (
            f"On this monitoring dashboard, what value did the metric \"{X_name}\" in the "
            f"\"Key metrics\" table show on the first frame where the \"{S_name}\" service "
            f"tile is red (status DOWN)?"
        )
        self.answer_format = "integer, digits only (3 to 5 digits), no unit"

    # --- helpers ------------------------------------------------------------------
    def _tick_index(self, k: int, frame: int) -> int:
        p = self.kpi_phase[k]
        return 0 if frame < p else (frame - p) // TICK + 1

    def kpi_value(self, k: int, frame: int) -> int:
        return int(self.kpi_vals[k][self._tick_index(k, frame)])

    def tile_state(self, i: int, frame: int) -> int:
        if i == self.s_idx and frame >= self.f_red:
            return 2
        return 1 if any(a <= frame < b for a, b in self.amber[i]) else 0

    def tile_box(self, i: int) -> tuple[int, int, int, int]:
        return 16 + i * (TILE_W + TILE_GAP), TILE_Y, TILE_W, TILE_H

    def _value_xy(self, k: int) -> tuple[int, int]:
        return VALUE_X, TABLE[1] + 44 + k * ROW_H + 11

    # --- rendering ----------------------------------------------------------------
    def _panel(self, d, box, title):
        x, y, w, h = box
        d.rounded_rectangle((x, y, x + w - 1, y + h - 1), 4, fill=PANEL, outline=BORDER)
        draw_text(d, (x + 12, y + 9), title, "sans-bold", 15, TEXT)

    def _chart_boxes(self):
        return [((672, 290, 608, 410), "Request rate", (0, 1)),
                ((1296, 290, 608, 410), "Latency p95 / p99", (2, 3)),
                ((16, 716, 1050, 348), "Errors / s by service", (4,))]

    def _stat_boxes(self):
        return [(1082 + (i % 2) * 416, 716 + (i // 2) * 180, 404, 168) for i in range(4)]

    def _background(self) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, W, NAV_H - 1), fill=PANEL)
        d.line((0, NAV_H - 1, W, NAV_H - 1), fill=BORDER)
        d.ellipse((16, 12, 44, 40), fill=(245, 120, 30))
        d.ellipse((24, 20, 36, 32), fill=PANEL)
        draw_text(d, (60, 15), "Home  ›  Dashboards  ›  Production overview", "sans", 17, TEXT)
        d.rounded_rectangle((1500, 10, 1700, 42), 4, outline=BORDER, fill=BG)
        draw_text(d, (1516, 16), "Last 15 minutes", "sans", 16, TEXT)
        d.rounded_rectangle((1712, 10, 1800, 42), 4, outline=BORDER, fill=BG)
        draw_text(d, (1726, 16), "⟳  2s", "sans", 16, TEXT)
        draw_text(d, (16, 64), "Service status", "sans-bold", 17, TEXT)
        # Key metrics table (static parts).
        self._panel(d, TABLE, "Key metrics")
        x, y, w, h = TABLE
        d.line((x + 1, y + 40, x + w - 2, y + 40), fill=BORDER)
        for k, (name, unit) in enumerate(self.kpis):
            ry = y + 44 + k * ROW_H
            if k % 2:
                d.rectangle((x + 1, ry, x + w - 2, ry + ROW_H - 1), fill=(29, 32, 37))
            draw_text(d, (x + 16, ry + 11), name, "sans", 16, TEXT)
            draw_text(d, (VALUE_X + 80, ry + 11), unit, "sans", 15, MUTED)
        for box, title, _ in self._chart_boxes():
            self._panel(d, box, title)
            bx, by, bw, bh = box
            for g in range(5):
                gy = by + 44 + g * (bh - 80) // 4
                d.line((bx + 50, gy, bx + bw - 14, gy), fill=(36, 39, 46))
        for i, box in enumerate(self._stat_boxes()):
            self._panel(d, box, self.stats[i][0])
        return im

    def _row_bgs(self):
        x, y, w, h = TABLE
        return [(29, 32, 37) if k % 2 else PANEL for k in range(len(self.kpis))]

    def render(self, frame: int) -> Image.Image:
        frame = self._check_frame(frame)
        im = self.static("bg", self._background).copy()
        d = ImageDraw.Draw(im)
        now = self.base_time + dt.timedelta(seconds=frame / self.FPS)
        draw_text(d, (1820, 16), now.strftime("%H:%M:%S"), "sans", 16, MUTED)
        # Service tiles (the trace).
        for i, name in enumerate(self.tiles):
            st = self.tile_state(i, frame)
            x, y, w, h = self.tile_box(i)
            d.rounded_rectangle((x, y, x + w - 1, y + h - 1), 6, fill=(GREEN, AMBER, RED)[st])
            tw = text_width(name, *TILE_NAME_FONT)
            draw_text(d, (x + (w - tw) / 2, y + 22), name, *TILE_NAME_FONT, (255, 255, 255))
            label = ("UP", "DEGRADED", "DOWN")[st]
            tw = text_width(label, *TILE_STATUS_FONT)
            draw_text(d, (x + (w - tw) / 2, y + 82), label, *TILE_STATUS_FONT, (255, 255, 255))
        # KPI values (the answer lives here).
        bgs = self._row_bgs()
        for k in range(len(self.kpis)):
            vx, vy = self._value_xy(k)
            d.rectangle((vx - 4, vy - 4, vx + 74, vy + 22), fill=bgs[k])
            draw_text(d, (vx, vy), str(self.kpi_value(k, frame)), *VALUE_FONT, TEXT)
        # Charts: last `hist` global ticks.
        gt = 0 if frame < self.g_phase else (frame - self.g_phase) // TICK + 1
        for box, _, idxs in self._chart_boxes():
            bx, by, bw, bh = box
            x0, x1, y0, y1 = bx + 50, bx + bw - 14, by + 44, by + bh - 36
            for c, si in enumerate(idxs):
                seg = self.series[si][gt:gt + self.hist]
                lo, hi = float(seg.min()), float(seg.max())
                sc = (y1 - y0) / (hi - lo + 1e-9)
                xs = np.linspace(x0, x1, len(seg))
                ys = y1 - (seg - lo) * sc
                d.line(list(zip(xs.tolist(), ys.tolist())), fill=SERIES[(si + c) % 4], width=2)
            for g in range(3):
                t = now - dt.timedelta(seconds=2 * self.hist * (1 - g / 2))
                draw_text(d, (x0 + g * (x1 - x0) / 2 - 22, y1 + 10), t.strftime("%H:%M"),
                          "sans", 13, MUTED)
        # Big stat panels (distractor numbers).
        for i, box in enumerate(self._stat_boxes()):
            name, vals, unit = self.stats[i]
            txt = f"{int(vals[gt])}{unit}"
            tw = text_width(txt, "sans-bold", 64)
            bx, by, bw, bh = box
            draw_text(d, (bx + (bw - tw) / 2, by + 50), txt, "sans-bold", 64, SERIES[i % 4])
        return im

    def debug_info(self) -> dict:
        x0, x1 = self.kpi_window
        return {
            "kind": self.kind, "seed": self.seed,
            "tile": self.tiles[self.s_idx], "tile_index": self.s_idx,
            "kpi": self.kpis[self.x_idx][0], "kpi_index": self.x_idx,
            "first_red_frame": self.f_red,
            "event_frame": self.f_red,
            "kpi_window": (x0, x1),
            "amber_intervals": {self.tiles[i]: list(v) for i, v in enumerate(self.amber) if v},
            "answer_text": {"frame": self.f_red, "xy": self._value_xy(self.x_idx),
                            "text": self.answer, "font": VALUE_FONT[0], "size": VALUE_FONT[1],
                            "fill": TEXT, "bg": self._row_bgs()[self.x_idx]},
            "trace": {"box": self.tile_box(self.s_idx), "font_px": TILE_STATUS_FONT[1],
                      "change_frames": [self.f_red], "red_rgb": RED},
            "tile_boxes": [self.tile_box(i) for i in range(N_TILES)],
            "answer_font_px": VALUE_FONT[1],
        }
