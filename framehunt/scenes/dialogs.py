"""`dialogs`: a desktop accounting app where modal error dialogs pop up and close.

Question: error code shown in the K-th error dialog. Trace: the notification bell's
badge in the top bar counts dialogs shown so far. Distractors: all codes in a scene
share the same 2-character prefix, so they only differ in fine detail.
"""
from __future__ import annotations

import datetime as dt

from PIL import Image, ImageDraw, ImageFilter

from .base import Scene, SceneMeta, draw_text, pick_event_frames, text_width

W, H = 1920, 1080
TOP_H = 112
SIDE_W = 280
STATUS_H = 30
CODE_FONT = ("mono-bold", 15)        # answer text: cap height ~11 px native, <3 px at 1/4
BADGE_FONT = ("sans-bold", 76)       # trace: digits ~55 px native, ~7 px at 1/8
TRACE_BOX = (1560, 8, 1800 - 1560, 96)   # bell + badge region (x, y, w, h)
DLG_W, DLG_H = 700, 300
SHADOW = 18
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"   # no I/O (vs 1/0)

NAVY = (32, 39, 56)
WHITE = (255, 255, 255)
INK = (33, 37, 41)
MUTED = (108, 117, 125)
LINE = (222, 226, 230)
BLUE = (13, 110, 253)
RED = (214, 40, 40)

CUSTOMERS = ["Northwind Traders", "Contoso Ltd", "Fabrikam Inc", "Tailspin Toys",
             "Wide World Importers", "Adventure Works", "Litware Inc", "Proseware",
             "Woodgrove Bank", "Coho Winery", "Alpine Ski House", "Blue Yonder Air",
             "Fourth Coffee", "Margie's Travel", "Lucerne Publishing", "Trey Research"]
DIALOG_TEXTS = [
    ("Sync error", "Could not post invoice {inv} to the general ledger."),
    ("Export failed", "Payment batch {inv} could not be exported to the bank."),
    ("Ledger rejected entry", "The ledger service rejected the journal entry for {inv}."),
    ("Connection error", "Lost connection while saving {inv}. Changes were not saved."),
    ("Validation failed", "Invoice {inv} failed tax validation on the server."),
]


def _ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


class DialogsScene(Scene):
    kind = "dialogs"

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

        # Timeline: dialogs open for 3-6 s, at least 4 s apart, inside 10-90 %.
        rt = self.rng("timeline")
        n = rt.randint(3, 8)
        max_dur = 6 * fps
        opens = pick_event_frames(rt, n, int(0.1 * n_frames) + 1,
                                  int(0.9 * n_frames) - max_dur, max_dur + 4 * fps)
        prefix = "".join(rt.choice(CODE_ALPHABET) for _ in range(2))
        codes: list[str] = []
        while len(codes) < n:
            c = prefix + "".join(rt.choice(CODE_ALPHABET) for _ in range(4))
            if c not in codes:
                codes.append(c)
        self.dialogs = []
        for i, (f0, code) in enumerate(zip(opens, codes)):
            title, msg = rt.choice(DIALOG_TEXTS)
            self.dialogs.append({
                "open": f0, "close": f0 + rt.randint(3 * fps, max_dur), "code": code,
                "title": title, "msg": msg.format(inv=f"INV-{rt.randint(10000, 99999)}"),
                "x": (SIDE_W + W - DLG_W) // 2 + rt.randint(-160, 160) - SIDE_W // 2,
                "y": (TOP_H + H - DLG_H) // 2 + rt.randint(-120, 120),
            })
        self.k = rt.randint(1, n)
        self.answer = codes[self.k - 1]
        self.question = (
            f"Error dialogs pop up in this app from time to time. What error code was shown "
            f"in the {_ordinal(self.k)} error dialog? The red badge on the notification bell "
            f"in the top bar counts the error dialogs shown so far."
        )
        self.answer_format = "6 characters, A-Z and 0-9"
        self._rows = self._table_rows(self.rng("table"))

    # --- static layers -------------------------------------------------------------
    @staticmethod
    def _table_rows(r):
        rows = []
        for _ in range(22):
            status = r.choices(["Paid", "Pending", "Overdue", "Draft"], [5, 3, 2, 1])[0]
            rows.append((f"INV-{r.randint(10000, 99999)}", r.choice(CUSTOMERS),
                         f"{r.randint(1, 28):02d}.{r.randint(1, 12):02d}.2026",
                         f"{r.randint(120, 48000):,}.{r.randint(0, 99):02d} EUR".replace(",", " "),
                         status))
        return rows

    def _background(self) -> Image.Image:
        im = Image.new("RGB", (W, H), WHITE)
        d = ImageDraw.Draw(im)
        # Top bar.
        d.rectangle((0, 0, W, TOP_H - 1), fill=NAVY)
        d.rounded_rectangle((28, 30, 80, 82), 10, fill=(46, 204, 113))
        draw_text(d, (41, 33), "L", "sans-bold", 36, WHITE)
        draw_text(d, (98, 36), "Ledgerly", "sans-bold", 32, WHITE)
        d.rounded_rectangle((520, 34, 1180, 78), 22, fill=(52, 60, 82))
        draw_text(d, (552, 45), "Search invoices, customers, payments...", "sans", 18,
                  (160, 168, 185))
        d.ellipse((1830, 30, 1882, 82), fill=(90, 110, 160))
        initials = "".join(p[0] for p in self.user.replace(".", "").split())
        draw_text(d, (1838, 42), initials[:2], "sans-bold", 22, WHITE)
        # Sidebar.
        d.rectangle((0, TOP_H, SIDE_W, H), fill=(245, 247, 250))
        d.line((SIDE_W, TOP_H, SIDE_W, H), fill=LINE)
        for i, item in enumerate(["Dashboard", "Invoices", "Customers", "Payments",
                                  "Ledger", "Reports", "Settings"]):
            y = TOP_H + 30 + i * 52
            if item == "Invoices":
                d.rounded_rectangle((14, y - 8, SIDE_W - 14, y + 34), 8, fill=(225, 236, 255))
            draw_text(d, (40, y), item, "sans-bold" if item == "Invoices" else "sans", 19,
                      BLUE if item == "Invoices" else INK)
        # Main content: invoice table.
        x0 = SIDE_W + 40
        draw_text(d, (x0, TOP_H + 26), "Invoices", "sans-bold", 30, INK)
        draw_text(d, (x0, TOP_H + 70), "All invoices · Q4 2026", "sans", 17, MUTED)
        d.rounded_rectangle((W - 260, TOP_H + 28, W - 40, TOP_H + 72), 8, fill=BLUE)
        draw_text(d, (W - 232, TOP_H + 39), "+ New invoice", "sans-bold", 18, WHITE)
        cols = [("Invoice", 0), ("Customer", 200), ("Due date", 560), ("Amount", 780),
                ("Status", 1060), ("Owner", 1260)]
        ty = TOP_H + 120
        d.rectangle((x0, ty, W - 40, ty + 44), fill=(248, 249, 250))
        for name, cx in cols:
            draw_text(d, (x0 + 16 + cx, ty + 12), name, "sans-bold", 16, MUTED)
        for i, (inv, cust, due, amt, status) in enumerate(self._rows[:19]):
            y = ty + 44 + i * 40
            d.line((x0, y, W - 40, y), fill=LINE)
            draw_text(d, (x0 + 16, y + 10), inv, "sans", 16, BLUE)
            draw_text(d, (x0 + 216, y + 10), cust, "sans", 16, INK)
            draw_text(d, (x0 + 576, y + 10), due, "sans", 16, INK)
            draw_text(d, (x0 + 796, y + 10), amt, "sans", 16, INK)
            fill, ink = {"Paid": ((220, 245, 228), (25, 135, 84)),
                         "Pending": ((255, 243, 205), (150, 110, 0)),
                         "Overdue": ((255, 229, 208), (190, 90, 20)),
                         "Draft": ((233, 236, 239), MUTED)}[status]
            d.rounded_rectangle((x0 + 1076, y + 7, x0 + 1076 + 96, y + 33), 12, fill=fill)
            draw_text(d, (x0 + 1092, y + 10), status, "sans", 15, ink)
            draw_text(d, (x0 + 1276, y + 10), self.user, "sans", 16, MUTED)
        # Status bar.
        d.rectangle((0, H - STATUS_H, W, H), fill=(233, 236, 239))
        draw_text(d, (16, H - STATUS_H + 6), "Connected to ledger.prod  ·  v4.12.3",
                  "sans", 14, MUTED)
        return im

    def _background_dim(self) -> Image.Image:
        im = self.static("bg", self._background).copy()
        body = im.crop((0, TOP_H, W, H))
        im.paste(Image.blend(body, Image.new("RGB", body.size, (0, 0, 0)), 0.35), (0, TOP_H))
        return im

    def _dialog_image(self, i: int) -> Image.Image:
        """RGBA dialog card with its drop shadow (size DLG + 2*SHADOW)."""
        dl = self.dialogs[i]
        S = SHADOW
        im = Image.new("RGBA", (DLG_W + 2 * S, DLG_H + 2 * S), (0, 0, 0, 0))
        sh = Image.new("L", im.size, 0)
        ImageDraw.Draw(sh).rounded_rectangle((S, S + 6, S + DLG_W, S + DLG_H + 6), 10, fill=110)
        im.putalpha(sh.filter(ImageFilter.GaussianBlur(9)))
        d = ImageDraw.Draw(im)
        x0, y0 = S, S
        d.rounded_rectangle((x0, y0, x0 + DLG_W, y0 + DLG_H), 10, fill=WHITE + (255,),
                            outline=(200, 205, 210, 255))
        d.rectangle((x0 + 1, y0 + 56, x0 + DLG_W - 1, y0 + 57), fill=LINE + (255,))
        d.ellipse((x0 + 22, y0 + 14, x0 + 52, y0 + 44), fill=RED + (255,))
        draw_text(d, (x0 + 33, y0 + 15), "!", "sans-bold", 22, WHITE)
        draw_text(d, (x0 + 66, y0 + 16), dl["title"], "sans-bold", 21, INK)
        draw_text(d, (x0 + DLG_W - 40, y0 + 14), "×", "sans", 26, MUTED)
        draw_text(d, (x0 + 28, y0 + 80), dl["msg"], "sans", 17, INK)
        draw_text(d, (x0 + 28, y0 + 110), "Please try again. If the problem persists, contact "
                  "support", "sans", 17, MUTED)
        draw_text(d, (x0 + 28, y0 + 134), "and quote the error code below.", "sans", 17, MUTED)
        draw_text(d, (x0 + 28, y0 + 182), "Error code:", "sans", 17, INK)
        cx, cy = self._code_xy_rel()
        d.rounded_rectangle((x0 + cx - 10, y0 + cy - 6, x0 + cx + 64, y0 + cy + 25), 6,
                            fill=(241, 243, 245, 255), outline=LINE + (255,))
        draw_text(d, (x0 + cx, y0 + cy), dl["code"], *CODE_FONT, INK)
        d.rounded_rectangle((x0 + DLG_W - 260, y0 + DLG_H - 62, x0 + DLG_W - 130,
                             y0 + DLG_H - 22), 8, outline=(180, 186, 192, 255), fill=WHITE + (255,))
        draw_text(d, (x0 + DLG_W - 240, y0 + DLG_H - 52), "Copy code", "sans", 17, INK)
        d.rounded_rectangle((x0 + DLG_W - 114, y0 + DLG_H - 62, x0 + DLG_W - 26,
                             y0 + DLG_H - 22), 8, fill=BLUE + (255,))
        draw_text(d, (x0 + DLG_W - 84, y0 + DLG_H - 52), "OK", "sans-bold", 17, WHITE)
        return im

    @staticmethod
    def _code_xy_rel() -> tuple[int, int]:
        return 142, 184

    def _code_xy(self, i: int) -> tuple[int, int]:
        dl = self.dialogs[i]
        cx, cy = self._code_xy_rel()
        return dl["x"] + cx, dl["y"] + cy

    # --- per frame ---------------------------------------------------------------
    def badge_at(self, frame: int) -> int:
        return sum(1 for dl in self.dialogs if dl["open"] <= frame)

    def open_dialog_at(self, frame: int):
        for i, dl in enumerate(self.dialogs):
            if dl["open"] <= frame < dl["close"]:
                return i
        return None

    def render(self, frame: int) -> Image.Image:
        frame = self._check_frame(frame)
        i = self.open_dialog_at(frame)
        if i is None:
            im = self.static("bg", self._background).copy()
        else:
            im = self.static("bg_dim", self._background_dim).copy()
            card = self._cache.get(("dialog", i), lambda: self._dialog_image(i))
            dl = self.dialogs[i]
            im.paste(card, (dl["x"] - SHADOW, dl["y"] - SHADOW), card)
        d = ImageDraw.Draw(im)
        now = self.base_time + dt.timedelta(seconds=frame / self.FPS)
        draw_text(d, (1290, 44), now.strftime("%a %d %b  %H:%M:%S"), "sans", 19,
                  (200, 206, 220))
        self._draw_bell(d, self.badge_at(frame))
        return im

    @staticmethod
    def _draw_bell(d: ImageDraw.ImageDraw, n: int) -> None:
        x, y, w, h = TRACE_BOX
        d.rectangle((x, y, x + w - 1, y + h - 1), fill=NAVY)
        # Bell icon.
        bx, by = x + 6, y + 18
        d.pieslice((bx, by, bx + 56, by + 60), 180, 360, fill=(230, 234, 242))
        d.rectangle((bx, by + 29, bx + 56, by + 50), fill=(230, 234, 242))
        d.polygon([(bx - 8, by + 56), (bx + 64, by + 56), (bx + 56, by + 46), (bx, by + 46)],
                  fill=(230, 234, 242))
        d.ellipse((bx + 20, by + 54, bx + 36, by + 68), fill=(230, 234, 242))
        # Badge pill with the count.
        txt = str(n)
        tw = text_width(txt, *BADGE_FONT)
        pw = max(84, int(tw) + 36)
        px0 = x + 72
        d.rounded_rectangle((px0, y + 2, px0 + pw, y + 94), 46,
                            fill=RED if n else (90, 98, 115))
        draw_text(d, (px0 + (pw - tw) / 2, y - 1), txt, *BADGE_FONT, WHITE)

    def debug_info(self) -> dict:
        dls = [{"k": i + 1, "open_frame": dl["open"], "close_frame": dl["close"],
                "code": dl["code"], "box": (dl["x"], dl["y"], DLG_W, DLG_H),
                "code_xy": self._code_xy(i)} for i, dl in enumerate(self.dialogs)]
        key = dls[self.k - 1]
        return {
            "kind": self.kind, "seed": self.seed, "k": self.k, "n_dialogs": len(dls),
            "dialogs": dls,
            "event_frame": key["open_frame"],
            "answer_text": {"frame": key["open_frame"], "xy": key["code_xy"],
                            "text": key["code"], "font": CODE_FONT[0], "size": CODE_FONT[1],
                            "fill": INK, "bg": (241, 243, 245)},
            "trace": {"box": TRACE_BOX, "font_px": BADGE_FONT[1],
                      "change_frames": [dl["open"] for dl in self.dialogs],
                      "value_at_event": self.k},
            "answer_font_px": CODE_FONT[1],
        }
