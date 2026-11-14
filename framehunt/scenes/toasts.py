"""`toasts`: the Ledgerly desktop app with a stack of non-modal toast notifications.

Toasts (Info blue, Warning amber, Error red) pop up in a column at the top right. The
newest toast takes the top slot and pushes the others down; at most 5 are visible, and
each one lives 4-6 s (or until 5 newer toasts have pushed it out). Toasts come in
bursts (an incident: fast, Error-heavy) separated by calm stretches (slow, mostly Info),
so 2-4 red Error toasts with the same code prefix are often visible together. Every
toast carries a 6-character code, and all codes in a scene share a 2-character prefix.

Question: the code on the K-th Error toast. Trace: the notification bell's badge counts
Error toasts shown so far (lasting, monotonic, legible at 1/8).

Construction guarantees (what makes being slightly off a confident wrong answer):
  * K >= 2, and at frame f-1 (just before the K-th Error toast appears) the (K-1)-th
    Error toast is still on screen, red and with the same code prefix, as the newest
    red toast, so reading one frame early gives a convincing wrong code;
  * the next toast arrives 0.2-0.6 s after f, so the target leaves the top slot: a
    few frames late, "the top toast" is a different toast (often the next Error);
  * the (K+1)-th Error arrives within 1.5 s of f, so soon after f the target is one of
    several red look-alikes lower in the stack, no longer the newest red toast.
The target stays on screen for at least 1 s (usually its whole 4-6 s life), only in a
different slot.
"""
from __future__ import annotations

import bisect
import datetime as dt
from functools import cached_property

from PIL import Image, ImageDraw

from .base import Box, Scene, SceneMeta, draw_text
from .dialogs import (
    BADGE_FONT,
    BLUE,
    CODE_ALPHABET,
    CODE_FONT,
    CODE_FORMAT,
    INK,
    LINE,
    MUTED,
    RED,
    TOP_H,
    TRACE_BOX,
    WHITE,
    DialogsScene,
    _ordinal,
    draw_app_background,
    draw_bell,
)

W, H = 1920, 1080
TOAST_W, TOAST_H = 440, 108
TOAST_X = W - 24 - TOAST_W                  # 1456
SLOT_Y0, SLOT_PITCH = TOP_H + 16, TOAST_H + 12
N_SLOTS = 5
STRIPE_W = 24                               # level colour stripe: 3 px wide at 1/8
CODE_BOX = (100, 68, 74, 31)                # relative to the toast's top-left
CODE_TEXT = (10, 6)                         # code text origin inside CODE_BOX
CODE_BOX_RGB = (241, 243, 245)
MAX_ERRORS = 99                             # badge stays 1-2 digits

LEVELS = ("info", "warning", "error")
BURST_WEIGHTS = (25, 20, 55)                # info, warning, error
CALM_WEIGHTS = (65, 27, 8)
LEVEL_RGB = {"info": BLUE, "warning": (230, 160, 0), "error": RED}
LEVEL_TINT = {"info": (234, 242, 255), "warning": (255, 247, 224), "error": (253, 234, 234)}
LEVEL_WORD = {"info": "Info", "warning": "Warning", "error": "Error"}
LEVEL_ICON = {"info": "i", "warning": "!", "error": "×"}
TEXTS = {
    "error": [("Sync failed", "Could not post {inv} to the general ledger."),
              ("Export failed", "Batch {inv} was rejected by the bank."),
              ("Save failed", "Lost connection while saving {inv}."),
              ("Validation failed", "{inv} failed tax validation on the server."),
              ("Ledger rejected entry", "Journal entry for {inv} was rejected.")],
    "warning": [("Slow sync", "Ledger sync for {inv} is running slow."),
                ("Payment pending", "{inv}: bank confirmation still pending."),
                ("Due soon", "{inv} is due in 3 days."),
                ("Rate limit", "Bank API at 80 % of its rate limit.")],
    "info": [("Invoice saved", "{inv} was saved."),
             ("Payment received", "Payment for {inv} was received."),
             ("Export done", "Batch {inv} was exported."),
             ("Reminder sent", "A payment reminder for {inv} was emailed.")],
}


def slot_box(slot: int) -> Box:
    return Box(TOAST_X, SLOT_Y0 + slot * SLOT_PITCH, TOAST_W, TOAST_H)


def code_xy(slot: int) -> tuple[int, int]:
    b = slot_box(slot)
    return b.x + CODE_BOX[0] + CODE_TEXT[0], b.y + CODE_BOX[1] + CODE_TEXT[1]


class ToastsScene(Scene):
    kind = "toasts"

    def __init__(self, seed: int):
        super().__init__(seed)
        r = self.rng("setup")
        n_frames = r.randint(1800, 4800)
        fps = self.FPS
        self.meta = SceneMeta(W, H, fps, n_frames)
        self.base_time = dt.datetime(2026, r.randint(1, 12), r.randint(1, 28),
                                     r.randint(8, 17), r.randint(0, 59), r.randint(0, 59))
        self.user = r.choice(["M. Virtanen", "A. Korhonen", "J. Nieminen", "S. Laine",
                              "K. Mäkinen", "E. Heikkinen"])
        self._rows = DialogsScene._table_rows(self.rng("table"))[:19]
        self._build_timeline(n_frames)
        self.question = (
            f"Toast notifications pop up in the top-right corner of this app: blue Info, amber "
            f"Warning and red Error toasts. The newest toast appears at the top of the stack and "
            f"pushes the older ones down. What code was shown on the {_ordinal(self.k)} Error "
            f"toast? The red badge on the notification bell in the top bar counts the Error "
            f"toasts shown so far."
        )
        self.answer_format = CODE_FORMAT

    # --- timeline ----------------------------------------------------------------
    def _build_timeline(self, n_frames: int) -> None:
        for attempt in range(200):
            rr = self.rng(f"timeline/{attempt}")
            arrivals, levels, lives = [], [], []
            f = rr.randint(10, 30)
            burst = rr.random() < 0.5
            while f < n_frames:
                arrivals.append(f)
                # Bursts are fast and Error-heavy (an incident); calm stretches are slow
                # and mostly Info, so several red look-alikes are on screen in a burst.
                levels.append(rr.choices(LEVELS, BURST_WEIGHTS if burst else CALM_WEIGHTS)[0])
                lives.append(rr.randint(40, 60))            # 4-6 s
                if rr.random() < (0.2 if burst else 0.07):     # bursts ~5 toasts, calm ~15
                    burst = not burst
                f += rr.randint(2, 9) if burst else rr.randint(10, 30)   # 0.2-3 s apart
            n = len(arrivals)
            # A toast leaves when its life ends or when 5 newer toasts have arrived.
            ends = [min(a + life, arrivals[i + N_SLOTS] if i + N_SLOTS < n else a + life)
                    for i, (a, life) in enumerate(zip(arrivals, lives))]
            errs = [i for i, lv in enumerate(levels) if lv == "error"]
            if not 3 <= len(errs) <= MAX_ERRORS:
                continue
            cands = []
            for k in range(2, len(errs)):
                i, j, nx = errs[k - 1], errs[k - 2], errs[k]
                fk = arrivals[i]
                if not (0.1 * n_frames <= fk <= 0.9 * n_frames):
                    continue
                if not arrivals[j] <= fk - 1 < ends[j]:     # (K-1)-th Error still visible
                    continue
                if not 2 <= arrivals[i + 1] - fk <= 6:      # next toast 0.2-0.6 s later
                    continue
                if not arrivals[nx] - fk <= 15:              # (K+1)-th Error within 1.5 s
                    continue
                if ends[i] < fk + 10:                         # target on screen >= 1 s
                    continue
                cands.append(k)
            if cands:
                break
        else:  # pragma: no cover
            raise RuntimeError("could not build toast timeline")
        self.k = rr.choice(cands)
        prefix = "".join(rr.choice(CODE_ALPHABET) for _ in range(2))
        codes: list[str] = []
        seen: set[str] = set()
        while len(codes) < n:
            c = prefix + "".join(rr.choice(CODE_ALPHABET) for _ in range(4))
            if c not in seen:
                seen.add(c)
                codes.append(c)
        self.toasts = []
        for i in range(n):
            title, msg = rr.choice(TEXTS[levels[i]])
            self.toasts.append({"arrive": arrivals[i], "end": ends[i], "level": levels[i],
                                "code": codes[i], "title": title,
                                "msg": msg.format(inv=f"INV-{rr.randint(10000, 99999)}")})
        self.arrivals = arrivals
        self.error_idx = errs
        self.error_frames = [arrivals[i] for i in errs]
        self.target = errs[self.k - 1]
        self.answer = codes[self.target]

    # --- state ---------------------------------------------------------------------
    def badge_at(self, frame: int) -> int:
        return bisect.bisect_right(self.error_frames, frame)

    def visible_at(self, frame: int) -> list[int]:
        """Indices of the visible toasts, newest first (slot 0 = top)."""
        last = bisect.bisect_right(self.arrivals, frame)
        out = []
        for i in range(last - 1, max(-1, last - 40), -1):
            if self.toasts[i]["end"] > frame:
                out.append(i)
                if len(out) == N_SLOTS:
                    break
        return out

    # --- rendering -------------------------------------------------------------------
    @cached_property
    def _background(self) -> Image.Image:
        return draw_app_background(self.user, self._rows)

    def _draw_toast(self, d: ImageDraw.ImageDraw, i: int, slot: int) -> None:
        t = self.toasts[i]
        b = slot_box(slot)
        lv = t["level"]
        col = LEVEL_RGB[lv]
        d.rounded_rectangle((b.x + 3, b.y + 4, b.right + 2, b.bottom + 3), 8, fill=(214, 218, 224))
        d.rounded_rectangle(b.xyxy, 8, fill=LEVEL_TINT[lv], outline=(200, 205, 210))
        d.rectangle((b.x + 1, b.y + 1, b.x + STRIPE_W, b.bottom - 2), fill=col)
        d.ellipse((b.x + 40, b.y + 12, b.x + 64, b.y + 36), fill=col)
        draw_text(d, (b.x + 52, b.y + 13), LEVEL_ICON[lv], "sans-bold", 16, WHITE, "center")
        draw_text(d, (b.x + 76, b.y + 12), f"{LEVEL_WORD[lv]}: {t['title']}", "sans-bold", 18, INK)
        draw_text(d, (b.right - 30, b.y + 6), "×", "sans", 22, MUTED)
        draw_text(d, (b.x + 40, b.y + 42), t["msg"], "sans", 15, MUTED)
        draw_text(d, (b.x + 40, b.y + 74), "Code", "sans", 15, INK)
        cx, cy, cw, ch = CODE_BOX
        d.rounded_rectangle((b.x + cx, b.y + cy, b.x + cx + cw - 1, b.y + cy + ch - 1), 6,
                            fill=CODE_BOX_RGB, outline=LINE)
        draw_text(d, (b.x + cx + CODE_TEXT[0], b.y + cy + CODE_TEXT[1]), t["code"], *CODE_FONT, INK)

    def render(self, frame: int) -> Image.Image:
        frame = self._check_frame(frame)
        im = self._background.copy()
        d = ImageDraw.Draw(im)
        now = self.base_time + dt.timedelta(seconds=frame / self.FPS)
        draw_text(d, (1290, 44), now.strftime("%a %d %b  %H:%M:%S"), "sans", 19, (200, 206, 220))
        draw_bell(d, self.badge_at(frame))
        for slot, i in enumerate(self.visible_at(frame)):
            self._draw_toast(d, i, slot)
        return im

    def debug_info(self) -> dict:
        f = self.toasts[self.target]["arrive"]
        nxt = self.toasts[self.target + 1]["arrive"]
        errs = [{"k": k, "index": i, "arrive_frame": self.toasts[i]["arrive"],
                 "end_frame": self.toasts[i]["end"], "code": self.toasts[i]["code"]}
                for k, i in enumerate(self.error_idx, start=1)]
        return {
            "kind": self.kind, "seed": self.seed, "k": self.k, "n_toasts": len(self.toasts),
            "n_errors": len(errs), "errors": errs, "target_index": self.target,
            "event_frame": f,
            "target_end_frame": self.toasts[self.target]["end"],
            "top_slot_until": nxt,                   # first frame the target is no longer in slot 0
            "prev_error_code": errs[self.k - 2]["code"],
            "answer_text": {"frame": f, "xy": code_xy(0), "text": self.answer,
                            "font": CODE_FONT[0], "size": CODE_FONT[1], "fill": INK,
                            "bg": CODE_BOX_RGB},
            "trace": {"box": TRACE_BOX, "font_px": BADGE_FONT[1],
                      "change_frames": list(self.error_frames), "value_at_event": self.k},
            "slot_boxes": [slot_box(s) for s in range(N_SLOTS)],
            "answer_font_px": CODE_FONT[1],
        }
