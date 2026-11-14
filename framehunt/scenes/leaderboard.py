"""`leaderboard`: an API error tracker with a live, re-sorted endpoint table.

The "Endpoints by errors / min" table lists 10 endpoints (all always visible), sorted by
their current errors/min, and refreshes every 0.5 s (5 frames): values change and the
table re-sorts, so an endpoint's row moves. Endpoint names are sans-bold 18 (readable
at 1/2 scale); the values are small monospace digits, readable only at native
resolution.

Question: the errors/min value endpoint E showed on the first frame where the big
`alerts fired: N` counter (top right; lasting, monotonic, legible at 1/8) reached K.
Every alert fires on a table refresh, so the asked value is shown for exactly the 5
frames [f, f+4]. Construction guarantees: E's value AND E's row differ at f-1 and at
f+5, so reading one frame early, or half a second late, or re-using a row found on
another frame, gives a well-formed wrong number.
"""
from __future__ import annotations

import bisect
import datetime as dt
import math
from functools import cached_property

import numpy as np
from PIL import Image, ImageDraw

from .base import Box, Scene, SceneMeta, draw_text, pick_event_frames

W, H = 1920, 1080
NAV_H = 52
TICK = 5                        # table refresh period in frames (0.5 s)
N_ROWS = 10
PANEL_BOX = Box(16, 68, 1152, 572)
HEAD_Y = PANEL_BOX.y + 48
ROW_Y0, ROW_H = HEAD_Y + 40, 46
RANK_DX, NAME_DX, SERVICE_DX, VALUE_DX = 20, 70, 560, 820
NAME_FONT = ("sans-bold", 18)    # endpoint names: cap ~13 px native, ~6.5 px at 1/2
VALUE_FONT = ("mono", 13)        # answer text: digits ~10 px native, ~2.4 px at 1/4
TEXT_DY_NAME, TEXT_DY_VALUE = 12, 16
COUNTER_BOX = Box(1184, 68, 720, 104)
COUNTER_FONT = ("mono-bold", 64)  # trace: digits ~47 px native, ~6 px at 1/8
VMIN, VMAX = 100, 9999

BG = (17, 18, 23)
PANEL = (24, 27, 31)
BORDER = (44, 48, 56)
TEXT = (204, 204, 220)
MUTED = (140, 145, 160)
ROW_ALT = (29, 32, 37)
GRID = (36, 39, 46)
GREEN = (25, 95, 45)
RED = (170, 20, 20)
WHITE = (255, 255, 255)
SERIES = [(115, 191, 105), (87, 148, 242), (255, 152, 48), (184, 119, 217)]

ENDPOINTS = [
    ("GET /orders", "orders"), ("POST /checkout", "checkout"), ("GET /search", "search"),
    ("POST /login", "auth"), ("GET /cart", "cart"), ("PUT /profile", "accounts"),
    ("GET /catalog", "catalog"), ("POST /payments", "payments"), ("GET /inventory", "inventory"),
    ("DELETE /session", "auth"), ("GET /recommendations", "recommender"),
    ("POST /reviews", "reviews"), ("GET /shipping/quote", "shipping"),
    ("POST /refunds", "payments"), ("GET /invoices", "billing"), ("PATCH /address", "accounts"),
    ("GET /wishlist", "wishlist"), ("POST /coupons/apply", "pricing"),
    ("GET /media/thumb", "media"), ("POST /webhooks/bank", "payments"),
]
STATS = [("Requests / s", 900, 4000, ""), ("Error budget left", 10, 99, "%"),
         ("p99 latency", 180, 2400, " ms")]


def _ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def row_box(row: int) -> Box:
    return Box(PANEL_BOX.x + 1, ROW_Y0 + row * ROW_H, PANEL_BOX.w - 2, ROW_H)


def value_xy(row: int) -> tuple[int, int]:
    return PANEL_BOX.x + VALUE_DX, ROW_Y0 + row * ROW_H + TEXT_DY_VALUE


def row_bg(row: int):
    return ROW_ALT if row % 2 else PANEL


class LeaderboardScene(Scene):
    kind = "leaderboard"

    def __init__(self, seed: int):
        super().__init__(seed)
        r = self.rng("setup")
        fps = self.FPS
        n_frames = r.randint(1800, 4800)
        self.meta = SceneMeta(W, H, fps, n_frames)
        self.base_time = dt.datetime(2026, r.randint(1, 12), r.randint(1, 28),
                                     r.randint(0, 23), r.randint(0, 59), r.randint(0, 59))
        self.endpoints = r.sample(ENDPOINTS, N_ROWS)
        self.phase = r.randrange(TICK)
        self.n_ticks = self._tick(n_frames - 1) + 2
        self._build(n_frames)

    # --- ticks -----------------------------------------------------------------------
    def _tick(self, frame: int) -> int:
        """Table refresh index shown on `frame` (refreshes at frames phase + j*TICK)."""
        return 0 if frame < self.phase else (frame - self.phase) // TICK + 1

    def tick_start(self, j: int) -> int:
        return 0 if j == 0 else self.phase + (j - 1) * TICK

    # --- values, event, alerts ------------------------------------------------------------
    def _build(self, n_frames: int) -> None:
        T = self.n_ticks
        for attempt in range(200):
            rng = np.random.default_rng(self.np_seed(f"values/{attempt}"))
            rt = self.rng(f"timeline/{attempt}")
            # Log-space walks with mean reversion: bases spread over 250..2500 errors/min,
            # steps large enough that the table re-sorts often (an endpoint changes row
            # on roughly a third of the refreshes).
            base = rng.uniform(math.log(250), math.log(2500), N_ROWS)
            x = np.empty((T, N_ROWS))
            x[0] = base + rng.normal(0, 0.15, N_ROWS)
            steps = rng.normal(0, 0.11, (T, N_ROWS))
            for t in range(1, T):
                x[t] = x[t - 1] + 0.08 * (base - x[t - 1]) + steps[t]
            vals = np.clip(np.rint(np.exp(x)), VMIN, VMAX).astype(np.int64)
            # Rows: sorted by value (desc), ties by name.
            names = [e[0] for e in self.endpoints]
            order = np.array([sorted(range(N_ROWS), key=lambda i: (-int(v[i]), names[i]))
                              for v in vals])                     # order[t][row] = endpoint
            rank = np.argsort(order, axis=1)                       # rank[t][endpoint] = row
            e = rt.randrange(N_ROWS)
            lo_t = self._tick(int(0.1 * n_frames) + TICK) + 1
            hi_t = self._tick(int(0.9 * n_frames) - 2 * TICK)
            cands = [j for j in range(max(lo_t, 2), hi_t)
                     if rank[j, e] != rank[j - 1, e] and rank[j, e] != rank[j + 1, e]
                     and vals[j, e] != vals[j - 1, e] and vals[j, e] != vals[j + 1, e]
                     and int((vals[j] == vals[j, e]).sum()) == 1]     # no other row shows it
            if len(cands) >= 3:
                break
        else:  # pragma: no cover
            raise RuntimeError("could not build leaderboard")
        self.vals, self.order, self.rank = vals, order, rank
        self.e_idx = e
        self.j_event = rt.choice(cands)
        self.f_event = self.tick_start(self.j_event)

        # Alerts fire on table refreshes, >= 3 s apart, inside 5-95 % of the video.
        n_alerts = rt.randint(3, 12)
        t_lo = self._tick(int(0.05 * n_frames)) + 1
        t_hi = self._tick(int(0.95 * n_frames))
        gap = 30 // TICK                                   # alerts >= 3 s apart
        n_before = rt.randint(0, min(n_alerts - 1, max(0, (self.j_event - gap - t_lo) // gap + 1)))
        n_after = min(n_alerts - 1 - n_before, max(0, (t_hi - self.j_event - gap) // gap + 1))
        before = pick_event_frames(rt, n_before, t_lo, self.j_event - gap, gap) if n_before else []
        after = pick_event_frames(rt, n_after, self.j_event + gap, t_hi, gap) if n_after else []
        self.alert_ticks = before + [self.j_event] + after
        self.alert_frames = [self.tick_start(j) for j in self.alert_ticks]
        self.k = n_before + 1
        self.answer = str(int(vals[self.j_event, e]))

        # Charts and stat panels (cosmetic, same refresh).
        hist = 90
        self.hist = hist
        self.series = [rng.normal(0, 1, T + hist).cumsum() for _ in range(3)]
        self.stats = []
        for _name, slo, shi, _unit in STATS:
            w = np.empty(T)
            v = rng.uniform(slo, shi)
            for t in range(T):
                v = min(max(v + rng.normal(0, (shi - slo) * 0.03), slo), shi)
                w[t] = v
            self.stats.append(np.rint(w).astype(np.int64))

        E = self.endpoints[e][0]
        self.question = (
            f"This API error tracker shows the table \"Endpoints by errors / min\", which "
            f"refreshes and re-sorts every half second, so rows move. What errors/min value did "
            f"the endpoint \"{E}\" show in that table on the first frame where the "
            f"\"alerts fired\" counter (top right) reached {self.k}?"
        )
        self.answer_format = "integer, digits only (3 to 4 digits)"

    # --- state -------------------------------------------------------------------------
    def alerts_at(self, frame: int) -> int:
        return bisect.bisect_right(self.alert_frames, frame)

    def row_of(self, endpoint: int, frame: int) -> int:
        return int(self.rank[self._tick(frame), endpoint])

    def value_of(self, endpoint: int, frame: int) -> int:
        return int(self.vals[self._tick(frame), endpoint])

    def table_at(self, frame: int) -> list[tuple[str, int]]:
        """(endpoint name, value) per row, top to bottom."""
        t = self._tick(frame)
        return [(self.endpoints[i][0], int(self.vals[t, i])) for i in self.order[t]]

    # --- rendering -----------------------------------------------------------------------
    @staticmethod
    def _panel(d, box: Box, title):
        d.rounded_rectangle(box.xyxy, 4, fill=PANEL, outline=BORDER)
        draw_text(d, box.at(12, 10), title, "sans-bold", 17, TEXT)

    @cached_property
    def _background(self) -> Image.Image:
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, W, NAV_H - 1), fill=PANEL)
        d.line((0, NAV_H - 1, W, NAV_H - 1), fill=BORDER)
        d.ellipse((16, 12, 44, 40), fill=(245, 120, 30))
        d.ellipse((24, 20, 36, 32), fill=PANEL)
        draw_text(d, (60, 15), "Home  ›  Dashboards  ›  API error tracker", "sans", 17, TEXT)
        d.rounded_rectangle((1500, 10, 1700, 42), 4, outline=BORDER, fill=BG)
        draw_text(d, (1516, 16), "Last 60 seconds", "sans", 16, TEXT)
        d.rounded_rectangle((1712, 10, 1800, 42), 4, outline=BORDER, fill=BG)
        draw_text(d, (1726, 16), "⟳  0.5s", "sans", 16, TEXT)
        # Table (static parts).
        self._panel(d, PANEL_BOX, "Endpoints by errors / min  ·  sorted, refreshes every 0.5 s")
        d.line((PANEL_BOX.x + 1, HEAD_Y, PANEL_BOX.right - 2, HEAD_Y), fill=BORDER)
        for text, dx in (("#", RANK_DX), ("Endpoint", NAME_DX), ("Service", SERVICE_DX),
                         ("Errors / min", VALUE_DX)):
            draw_text(d, (PANEL_BOX.x + dx, HEAD_Y + 12), text, "sans-bold", 15, MUTED)
        for row in range(N_ROWS):
            b = row_box(row)
            if row % 2:
                d.rectangle((b.x, b.y, b.right - 1, b.bottom - 1), fill=ROW_ALT)
            draw_text(d, (PANEL_BOX.x + RANK_DX, b.y + 12), str(row + 1), "sans", 17, MUTED)
        # Right column: alert rules and stat panels; bottom: chart.
        rules = Box(1184, 188, 720, 452)
        self._panel(d, rules, "Alert rules")
        for i, (txt, sev) in enumerate([
                ("gateway 5xx spike (3 min window)", "critical"),
                ("error budget burn rate > 10x", "critical"),
                ("p99 latency > 2 s", "warning"),
                ("5xx ratio > 5 % (gateway)", "warning"),
                ("checkout success < 97 %", "critical"),
                ("payment provider timeouts > 50/min", "warning")]):
            y = rules.y + 56 + i * 62
            col = (212, 34, 50) if sev == "critical" else (224, 155, 0)
            d.rounded_rectangle((rules.x + 16, y, rules.x + 110, y + 28), 6, fill=col)
            draw_text(d, (rules.x + 24, y + 5), sev, "sans-bold", 14, WHITE)
            draw_text(d, (rules.x + 126, y + 4), txt, "sans", 17, TEXT)
        chart = self._chart_box
        self._panel(d, chart, "Errors / min, all endpoints")
        for g in range(5):
            gy = chart.y + 50 + g * (chart.h - 90) // 4
            d.line((chart.x + 50, gy, chart.right - 14, gy), fill=GRID)
        for (name, *_), box in zip(STATS, self._stat_boxes):
            self._panel(d, box, name)
        return im

    _chart_box = Box(16, 656, 1152, 408)
    _stat_boxes = [Box(1184, 656, 352, 200), Box(1552, 656, 352, 200), Box(1184, 872, 720, 192)]

    def render(self, frame: int) -> Image.Image:
        frame = self._check_frame(frame)
        im = self._background.copy()
        d = ImageDraw.Draw(im)
        t = self._tick(frame)
        now = self.base_time + dt.timedelta(seconds=frame / self.FPS)
        draw_text(d, (1820, 16), now.strftime("%H:%M:%S"), "sans", 16, MUTED)
        # Table rows (names move with their values).
        for row, i in enumerate(self.order[t]):
            b = row_box(row)
            name, service = self.endpoints[i]
            draw_text(d, (PANEL_BOX.x + NAME_DX, b.y + TEXT_DY_NAME), name, *NAME_FONT, TEXT)
            draw_text(d, (PANEL_BOX.x + SERVICE_DX, b.y + 14), service, "sans", 16, MUTED)
            draw_text(d, value_xy(row), str(int(self.vals[t, i])), *VALUE_FONT, TEXT)
        # The trace: alerts fired so far.
        n = self.alerts_at(frame)
        c = COUNTER_BOX
        d.rectangle(c.xyxy, fill=RED if n else GREEN)
        draw_text(d, (c.right - 24, c.y + 12), f"alerts fired: {n}", *COUNTER_FONT, WHITE, "right")
        # Chart: last `hist` refreshes.
        cb = self._chart_box
        x0, y0, x1, y1 = cb.x + 50, cb.y + 50, cb.right - 14, cb.bottom - 40
        for si, s in enumerate(self.series):
            seg = s[t:t + self.hist]
            lo, hi = float(seg.min()), float(seg.max())
            sc = (y1 - y0) / (hi - lo + 1e-9)
            xs = np.linspace(x0, x1, len(seg))
            ys = y1 - (seg - lo) * sc
            d.line(list(zip(xs.tolist(), ys.tolist())), fill=SERIES[si], width=2)
        for (_n, _lo, _hi, unit), vals, box in zip(STATS, self.stats, self._stat_boxes):
            draw_text(d, (box.center_x, box.y + 64), f"{int(vals[t])}{unit}", "sans-bold", 56,
                      SERIES[3], "center")
        return im

    def debug_info(self) -> dict:
        f = self.f_event
        e = self.e_idx
        row = self.row_of(e, f)
        return {
            "kind": self.kind, "seed": self.seed, "k": self.k,
            "endpoint": self.endpoints[e][0], "endpoint_index": e,
            "event_frame": f, "value_window": (f, f + TICK - 1),
            "row_at_event": row, "row_before": self.row_of(e, f - 1),
            "row_after": self.row_of(e, f + TICK),
            "value_before": self.value_of(e, f - 1), "value_after": self.value_of(e, f + TICK),
            "alert_frames": list(self.alert_frames),
            "answer_text": {"frame": f, "xy": value_xy(row), "text": self.answer,
                            "font": VALUE_FONT[0], "size": VALUE_FONT[1], "fill": TEXT,
                            "bg": row_bg(row)},
            "trace": {"box": COUNTER_BOX, "font_px": COUNTER_FONT[1],
                      "change_frames": list(self.alert_frames), "value_at_event": self.k},
            "row_boxes": [row_box(r) for r in range(N_ROWS)],
            "answer_font_px": VALUE_FONT[1],
        }
