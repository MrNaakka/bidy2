"""`dialogs`: a desktop accounting app where modal error dialogs pop up and close.

Question: error code shown in the K-th error dialog. Trace: the notification bell's
badge in the top bar counts dialogs shown so far. Distractors: all codes in a scene
share the same 2-character prefix, so they only differ in fine detail.
"""
from __future__ import annotations

import datetime as dt
from functools import cached_property

from PIL import Image, ImageDraw, ImageFilter
from stretchable.style import FlexDirection, JustifyContent

from .base import Scene, SceneMeta, draw_text, layout, node, pick_event_frames, text_width

W, H = 1920, 1080
TOP_H = 112
SIDE_W = 280
STATUS_H = 30
CODE_FONT = ("mono-bold", 15)        # answer text: cap height ~11 px native, <3 px at 1/4
BADGE_FONT = ("sans-bold", 76)       # trace: digits ~55 px native, ~7 px at 1/8
TRACE_BOX = (1560, 8, 1800 - 1560, 96)   # bell + badge region (x, y, w, h)
DLG_W, DLG_H = 700, 300
SHADOW = 18
CODE_TEXT = (10, 6)
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"   # no I/O (vs 1/0)
CODE_FORMAT = "6 characters: uppercase letters A-Z except I and O, and digits 0-9"

NAVY = (32, 39, 56)
WHITE = (255, 255, 255)
INK = (33, 37, 41)
MUTED = (108, 117, 125)
LINE = (222, 226, 230)
BLUE = (13, 110, 253)
RED = (214, 40, 40)

NAV_ITEMS = ["Dashboard", "Invoices", "Customers", "Payments", "Ledger", "Reports", "Settings"]
N_ROWS = 19
TABLE_COLS = [("Invoice", 0), ("Customer", 200), ("Due date", 560), ("Amount", 780),
              ("Status", 1060), ("Owner", 1260)]
STATUS_COLS = {"Paid": ((220, 245, 228), (25, 135, 84)),
               "Pending": ((255, 243, 205), (150, 110, 0)),
               "Overdue": ((255, 229, 208), (190, 90, 20)),
               "Draft": ((233, 236, 239), MUTED)}

COL = FlexDirection.COLUMN

L = layout(node(
    node(size=(W, TOP_H)),
    node(
        node(*(node(key=f"nav{i}", size=(None, 42), margin=(0, 14, 0, 14))
               for i in range(len(NAV_ITEMS))),
             key="side", size=(SIDE_W, None), flex_direction=COL, gap=10, padding=(22, 0, 0, 0)),
        node(
            node(node(key="heading"), node(key="new", size=(220, 44), margin=(2, 0, 0, 0)),
                 size=(None, 94), justify_content=JustifyContent.SPACE_BETWEEN),
            node(key="thead", size=(None, 44)),
            *(node(key=f"row{i}", size=(None, 40)) for i in range(N_ROWS)),
            key="main", flex_grow=1, flex_direction=COL, padding=(26, 40, 0, 40)),
        flex_grow=1),
    node(key="status", size=(W, STATUS_H)),
    size=(W, H), flex_direction=COL))
NAV = [L[f"nav{i}"] for i in range(len(NAV_ITEMS))]
ROWS = [L[f"row{i}"] for i in range(N_ROWS)]

CARD = layout(node(
    node(
        node(node(key="code_label", size=(104, 31)), node(key="code", size=(74, 31))),
        node(flex_grow=1),
        node(node(key="copy", size=(130, 40)), node(key="ok", size=(88, 40)),
             justify_content=JustifyContent.END, gap=16),
        key="card", size=(DLG_W, DLG_H), flex_direction=COL, padding=(178, 26, 22, 28)),
    padding=SHADOW))

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


def draw_app_background(user: str, rows) -> Image.Image:
    """The Ledgerly app (top bar, sidebar, invoice table, status bar) without clock and bell."""
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
    initials = "".join(p[0] for p in user.replace(".", "").split())
    draw_text(d, (1838, 42), initials[:2], "sans-bold", 22, WHITE)
    # Sidebar.
    side = L["side"]
    d.rectangle(side.edges, fill=(245, 247, 250))
    d.line((side.right, side.y, side.right, side.bottom), fill=LINE)
    for item, box in zip(NAV_ITEMS, NAV):
        active = item == "Invoices"
        if active:
            d.rounded_rectangle(box.edges, 8, fill=(225, 236, 255))
        draw_text(d, box.at(26, 8), item, "sans-bold" if active else "sans", 19,
                  BLUE if active else INK)
    # Main content: invoice table.
    heading, new = L["heading"], L["new"]
    draw_text(d, heading.at(0, 0), "Invoices", "sans-bold", 30, INK)
    draw_text(d, heading.at(0, 44), "All invoices · Q4 2026", "sans", 17, MUTED)
    d.rounded_rectangle(new.edges, 8, fill=BLUE)
    draw_text(d, new.at(28, 11), "+ New invoice", "sans-bold", 18, WHITE)
    thead = L["thead"]
    d.rectangle(thead.edges, fill=(248, 249, 250))
    for name, cx in TABLE_COLS:
        draw_text(d, thead.at(16 + cx, 12), name, "sans-bold", 16, MUTED)
    for row, (inv, cust, due, amt, status) in zip(ROWS, rows):
        d.line((row.x, row.y, row.right, row.y), fill=LINE)
        cells = [(inv, BLUE), (cust, INK), (due, INK), (amt, INK)]
        for (text, ink), (_, cx) in zip(cells, TABLE_COLS):
            draw_text(d, row.at(16 + cx, 10), text, "sans", 16, ink)
        fill, ink = STATUS_COLS[status]
        pill_x = row.x + 16 + TABLE_COLS[4][1]
        d.rounded_rectangle((pill_x, row.y + 7, pill_x + 96, row.y + 33), 12, fill=fill)
        draw_text(d, (pill_x + 16, row.y + 10), status, "sans", 15, ink)
        draw_text(d, row.at(16 + TABLE_COLS[5][1], 10), user, "sans", 16, MUTED)
    # Status bar.
    status_bar = L["status"]
    d.rectangle(status_bar.edges, fill=(233, 236, 239))
    draw_text(d, status_bar.at(16, 6), "Connected to ledger.prod  ·  v4.12.3",
              "sans", 14, MUTED)
    return im


def draw_bell(d: ImageDraw.ImageDraw, n: int) -> None:
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
    draw_text(d, (px0 + pw / 2, y - 1), txt, *BADGE_FONT, WHITE, "center")


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
        self.answer_format = CODE_FORMAT
        self._rows = self._table_rows(self.rng("table"))[:N_ROWS]

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

    @cached_property
    def _background(self) -> Image.Image:
        return draw_app_background(self.user, self._rows)

    @cached_property
    def _background_dim(self) -> Image.Image:
        im = self._background.copy()
        body = im.crop((0, TOP_H, W, H))
        im.paste(Image.blend(body, Image.new("RGB", body.size, (0, 0, 0)), 0.35), (0, TOP_H))
        return im

    @cached_property
    def _cards(self):
        return [self._dialog_image(i) for i in range(len(self.dialogs))]

    def _dialog_image(self, i: int) -> Image.Image:
        """RGBA dialog card with its drop shadow (size DLG + 2*SHADOW)."""
        dl = self.dialogs[i]
        card = CARD["card"]
        im = Image.new("RGBA", (card.right + SHADOW, card.bottom + SHADOW), (0, 0, 0, 0))
        sh = Image.new("L", im.size, 0)
        ImageDraw.Draw(sh).rounded_rectangle((card.x, card.y + 6, card.right, card.bottom + 6),
                                             10, fill=110)
        im.putalpha(sh.filter(ImageFilter.GaussianBlur(9)))
        d = ImageDraw.Draw(im)
        d.rounded_rectangle(card.edges, 10, fill=WHITE + (255,), outline=(200, 205, 210, 255))
        d.rectangle((card.x + 1, card.y + 56, card.right - 1, card.y + 57), fill=LINE + (255,))
        d.ellipse((*card.at(22, 14), *card.at(52, 44)), fill=RED + (255,))
        draw_text(d, card.at(33, 15), "!", "sans-bold", 22, WHITE)
        draw_text(d, card.at(66, 16), dl["title"], "sans-bold", 21, INK)
        draw_text(d, (card.right - 40, card.y + 14), "×", "sans", 26, MUTED)
        draw_text(d, card.at(28, 80), dl["msg"], "sans", 17, INK)
        draw_text(d, card.at(28, 110), "Please try again. If the problem persists, contact "
                  "support", "sans", 17, MUTED)
        draw_text(d, card.at(28, 134), "and quote the error code below.", "sans", 17, MUTED)
        draw_text(d, CARD["code_label"].at(0, 4), "Error code:", "sans", 17, INK)
        code = CARD["code"]
        d.rounded_rectangle(code.edges, 6, fill=(241, 243, 245, 255), outline=LINE + (255,))
        draw_text(d, code.at(*CODE_TEXT), dl["code"], *CODE_FONT, INK)
        copy, ok = CARD["copy"], CARD["ok"]
        d.rounded_rectangle(copy.edges, 8, outline=(180, 186, 192, 255), fill=WHITE + (255,))
        draw_text(d, copy.at(20, 10), "Copy code", "sans", 17, INK)
        d.rounded_rectangle(ok.edges, 8, fill=BLUE + (255,))
        draw_text(d, ok.at(30, 10), "OK", "sans-bold", 17, WHITE)
        return im

    def _code_xy(self, i: int):
        dl = self.dialogs[i]
        cx, cy = CARD["code"].at(*CODE_TEXT)
        return dl["x"] + cx - SHADOW, dl["y"] + cy - SHADOW

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
            im = self._background.copy()
        else:
            im = self._background_dim.copy()
            card = self._cards[i]
            dl = self.dialogs[i]
            im.paste(card, (dl["x"] - SHADOW, dl["y"] - SHADOW), card)
        d = ImageDraw.Draw(im)
        now = self.base_time + dt.timedelta(seconds=frame / self.FPS)
        draw_text(d, (1290, 44), now.strftime("%a %d %b  %H:%M:%S"), "sans", 19,
                  (200, 206, 220))
        draw_bell(d, self.badge_at(frame))
        return im

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
