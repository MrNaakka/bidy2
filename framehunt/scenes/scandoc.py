"""`scandoc`: a long scanned annual report (a PDF without a text layer).

The "video" is a PDF: page index i is frame i (fps = 1, so t = i). Every page is a
1240 x 1754 scan (A4 at 150 dpi) of a printed annual report, 150-400 pages long:

  * front matter numbered in lower-case roman numerals: cover, optional notice and blank
    pages, the Contents page, and a few front sections (CEO letter, key figures, ...);
  * main matter numbered from 1: report sections, the consolidated financial statements,
    then the numbered notes to the financial statements, then trailing sections;
  * 2-4 unnumbered inserted pages (scanned letters / correction slips) in the main matter,
    which do not count in the printed numbering. At least one sits in the notes, before
    the target note, so an offset calibrated at the start of the main matter is wrong
    by the time you reach the notes.

Navigation cues (running header: section or note title at the top left, printed page
number at the top right; the Contents entries; table captions, the table date, column
headers and row labels) are set in 36 px type: legible at 1/4 scale. The table amounts
are 12 px monospace digits: legible only near native resolution.

Question: the amount for row R, column C, in the table of the note titled T, at
31 December Y. The note's table is printed twice, on adjacent pages: for year Y and for
the comparative year (same caption, same rows and columns, different amounts). Every
table in the notes uses the same set of country rows and one of a few column pairs, so
landing on the wrong note or the wrong year still gives a well-formed amount.

Determinism: the whole layout (pages, numbering, inserts, tables, amounts) is planned
in __init__ from the seed. render(i) draws page i from per-page RNG streams, then
applies a seeded "scan" degradation: shift (<= 4 px), skew (<= 0.12 degrees),
softening blur, sensor noise from a cached noise field, paper tint, and a red stamp on
the cover and the inserted pages.
"""
from __future__ import annotations

import math
from functools import cached_property

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .base import Scene, SceneMeta, draw_text, font, text_width

PW, PH = 1240, 1754
MARGIN_L, MARGIN_R = 100, 1140                  # text block (native px, before degradation)
CUE_FONT = ("sans-bold", 36)                    # header title, folio, captions, labels
TOC_FONT = ("sans", 36)                         # contents titles
BODY_FONT = ("sans", 22)
AMOUNT_FONT = ("mono", 12)
AMOUNT_CELL = 8                                 # amounts: one digit per 8-px cell
HEADER_Y = 48                                   # header text top
RULE_Y = 104
TOC_Y0, TOC_PITCH, TOC_MAX_ROWS = 180, 60, 22
TOC_TITLE_X = 290                               # titles; "Note n" labels at MARGIN_L
CAPTION_Y, DATE_Y = 150, 205
COLHEAD_Y = 300
ROW_Y0, ROW_PITCH = 380, 64
COL_R = (820, 1140)                             # right edges of the two amount columns
AMOUNT_DY = 12                                  # amount text top relative to the row top
INK, PAPER = 28, 255

# Degradation
MAX_SHIFT = 4
MAX_SKEW_DEG = 0.12
# Softening (a scanner's optics): a 3x3 binomial kernel, about a 0.6-px Gaussian.
SOFTEN = ImageFilter.Kernel((3, 3), (1, 2, 1, 2, 4, 2, 1, 2, 1), 16)
NOISE_SIGMA = 6.0
NOISE_PAD = 64

COUNTRIES = ("Finland", "Sweden", "Norway", "Denmark", "Estonia", "Latvia", "Lithuania",
             "Poland", "Germany", "Netherlands", "France", "Spain")
COLUMN_PAIRS = (("Carrying amount", "Fair value"), ("Gross amount", "Net amount"),
                ("Opening", "Closing"), ("Current", "Non-current"), ("Cost", "Book value"))
NOTE_POOL = (
    ("Borrowings", "Borrowings by country"),
    ("Trade receivables", "Receivables by country"),
    ("Inventories", "Inventories by country"),
    ("Property, plant and equipment", "Assets by country"),
    ("Intangible assets", "Intangibles by country"),
    ("Provisions", "Provisions by country"),
    ("Leases", "Lease liabilities by country"),
    ("Deferred taxes", "Deferred tax by country"),
    ("Trade payables", "Payables by country"),
    ("Cash and cash equivalents", "Cash by country"),
    ("Goodwill", "Goodwill by country"),
    ("Revenue", "Revenue by country"),
    ("Employee benefits", "Pension obligations"),
    ("Derivatives", "Derivatives by country"),
    ("Investments in associates", "Associates by country"),
    ("Other receivables", "Other receivables"),
)
FRONT_POOL = ("Letter from the CEO", "The year in brief", "Key figures", "Our strategy",
              "Our business", "Highlights of the year")
SECTION_POOL = ("Board of Directors' report", "Sustainability statement", "Corporate governance",
                "Remuneration report", "Risk management")
TRAILING = ("Auditor's report", "Information for shareholders")
COMPANIES = ("Northwind Holdings", "Kestrel Group", "Aurora Industries", "Baltic Timber",
             "Lumo Energy", "Harbor Logistics", "Vanta Foods", "Silva Materials")
WORDS = ("the group", "operations", "during the year", "revenue", "increased", "decreased",
         "market", "customers", "investments", "as reported", "in accordance with", "IFRS",
         "management", "the board", "risk", "strategy", "performance", "cash flow", "assets",
         "liabilities", "employees", "sustainability", "measured at", "fair value", "the period",
         "segment", "subsidiaries", "net sales", "costs", "financing", "were", "is", "and",
         "of", "for", "to", "in", "by", "on", "with", "a", "an", "our", "its", "this", "which",
         "recognised", "estimated", "the company", "carrying amount", "impairment", "tests")
AMOUNT_FORMAT = ("integer of 4 to 6 digits (the amount in EUR thousand as printed; the space used as a "
                 "thousands separator does not matter)")


def roman(n: int) -> str:
    out = ""
    for v, s in ((1000, "m"), (900, "cm"), (500, "d"), (400, "cd"), (100, "c"), (90, "xc"),
                 (50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while n >= v:
            out += s
            n -= v
    return out


def fmt_amount(v: int) -> str:
    s = str(v)
    return s if len(s) <= 3 else s[:-3] + " " + s[-3:]


class ScanDocScene(Scene):
    kind = "scandoc"
    medium = "document"
    WIDTH, HEIGHT, FPS = PW, PH, 1

    def __init__(self, seed: int):
        super().__init__(seed)
        r = self.rng("setup")
        self.company = r.choice(COMPANIES)
        self.year = r.randint(2016, 2025)
        self.countries = r.sample(COUNTRIES, 8)
        self._plan(r)
        self.meta = SceneMeta(PW, PH, 1, len(self.pages))
        t = self.target
        self.question = (
            f"This is a scanned PDF of the {self.company} Annual Report {self.year}. In the notes to the "
            f"financial statements, the note titled \"{t['title']}\" has a table \"{t['table_title']}\" "
            f"(amounts in EUR thousand). What amount does that table give for \"{t['row']}\" in the column "
            f"\"{t['col']}\" at 31 December {t['year']}?"
        )
        self.answer_format = AMOUNT_FORMAT
        self.answer = str(t["value"])

    # ------------------------------------------------------------------ planning
    def _plan(self, r) -> None:
        n_target = r.randint(152, 400)          # + 2-4 inserts - 4 below: 150-400 pages
        # --- front matter
        front: list[dict] = [{"type": "cover"}]
        if r.random() < 0.6:
            front.append({"type": "notice", "header": "Important notice"})
        if r.random() < 0.5:
            front.append({"type": "blank", "header": ""})
        front.append({"type": "contents", "header": "Contents"})
        front_titles = r.sample(FRONT_POOL, r.randint(2, 4))
        front_entries = []
        for ti in front_titles:
            front_entries.append((ti, len(front)))
            for j in range(r.randint(1, 4)):
                front.append({"type": "text", "header": ti, "first": j == 0})
        for i, p in enumerate(front):
            p["folio"] = roman(i + 1) if p["type"] != "cover" else None
        # --- notes
        n_notes = r.randint(8, 11)
        note_specs = [("Accounting policies", None)] + r.sample(NOTE_POOL, n_notes - 1)
        notes = []
        for n, (title, ttitle) in enumerate(note_specs, start=1):
            pages = []
            if ttitle is None:
                pages += ["text"] * r.randint(4, 8)
            else:
                pages += ["text"] * r.randint(1, 2)
                pages += ["table_cur", "table_prev"] if r.random() < 0.7 else ["table_prev", "table_cur"]
                pages += ["text"] * r.randint(0, 2)
            cols = r.choice(COLUMN_PAIRS)
            rows = r.sample(self.countries, r.randint(6, 8))
            notes.append({"n": n, "title": title, "table_title": ttitle, "pages": pages,
                          "cols": cols, "rows": rows})
        # --- main sections, sized so the document has about n_target pages
        sections = r.sample(SECTION_POOL, r.randint(3, 4))
        fin_len = r.randint(4, 8)
        trailing = [(TRAILING[0], r.randint(3, 5)), (TRAILING[1], r.randint(2, 3))]
        fixed = len(front) + fin_len + sum(len(nt["pages"]) for nt in notes) + sum(n for _, n in trailing) + 4
        rest = max(10 * len(sections), n_target - fixed)
        w = [r.random() + 0.3 for _ in sections]
        extra = rest - 10 * len(sections)
        sizes = [10 + int(extra * x / sum(w)) for x in w]
        sizes[-1] += rest - sum(sizes)
        # --- the main matter, numbered from 1 (inserts added afterwards)
        main: list[dict] = []
        entries_main = []

        def add_block(header: str, n_pages: int, kind: str = "text", note=None):
            entries_main.append((header, len(main), note))
            for j in range(n_pages):
                main.append({"type": kind, "header": header, "first": j == 0, "note": note})

        for title, size in zip(sections, sizes):
            add_block(title, size)
        add_block("Consolidated financial statements", fin_len)
        notes_start = len(main)
        for nt in notes:
            entries_main.append((f"Note {nt['n']}  {nt['title']}", len(main), nt["n"]))
            for j, kind in enumerate(nt["pages"]):
                main.append({"type": "note_" + kind if kind != "text" else "text",
                             "header": f"Note {nt['n']}  {nt['title']}", "first": j == 0, "note": nt["n"]})
        notes_end = len(main)
        for title, size in trailing:
            add_block(title, size)
        for i, p in enumerate(main):
            p["folio"] = str(i + 1)
        # --- the target: a note with a table, not the first one with a table
        cands = [nt for nt in notes if nt["table_title"] is not None and nt["n"] >= 3]
        tn = r.choice(cands)
        t_first = next(i for i, p in enumerate(main) if p.get("note") == tn["n"])
        t_last = max(i for i, p in enumerate(main) if p.get("note") == tn["n"])
        # --- unnumbered inserts: one in the report sections, one in the notes before the
        # target note, 0-1 more anywhere before it, 0-1 after it. Never inside the target note.
        ins = {r.randint(1, notes_start - 1), r.randint(notes_start + 1, t_first)}
        if r.random() < 0.5:
            ins.add(r.randint(1, t_first))
        if r.random() < 0.5:
            ins.add(r.randint(t_last + 1, len(main)))
        pages = list(front)
        for i, p in enumerate(main):
            if i in ins:
                pages.append({"type": "insert", "header": None, "folio": None})
            pages.append(p)
        if len(main) in ins:
            pages.append({"type": "insert", "header": None, "folio": None})
        self.pages = pages
        self.front_len = len(front)
        idx_of_main = [i for i, p in enumerate(pages) if p["type"] != "insert" and i >= len(front)]
        # --- contents (printed folios)
        toc = [("", ti, roman(fi + 1)) for ti, fi in front_entries]
        for header, mi, note in entries_main:
            if note is not None:
                nt = notes[note - 1]
                toc.append((f"Note {note}", nt["title"], str(mi + 1)))
            else:
                toc.append(("", header, str(mi + 1)))
        assert len(toc) <= TOC_MAX_ROWS, len(toc)
        self.toc = toc
        # --- tables: amounts, unique across the document
        used: set[int] = set()

        def amount() -> int:
            while True:
                nd = r.choice((4, 5, 5, 6))
                v = r.randint(10 ** (nd - 1), 10 ** nd - 1)
                if v not in used:
                    used.add(v)
                    return v

        def draw_values() -> None:
            for nt in notes:
                if nt["table_title"] is None:
                    continue
                nt["values"] = {}
                for y in (self.year, self.year - 1):
                    nt["values"][y] = [[amount() for _ in nt["cols"]] for _ in nt["rows"]]

        def printed() -> list[int]:
            # every amount printed in a table, including the Total row's column sums
            out = []
            for nt in notes:
                for vals in nt.get("values", {}).values():
                    out += [x for v in vals for x in v] + [sum(v[c] for v in vals) for c in range(2)]
            return out

        draw_values()
        # Totals are sums, so they can (rarely) equal another printed amount. Redraw all
        # values until every printed amount is unique. Scenes without a collision consume
        # the generator exactly as before, so they are unchanged.
        while len(set(p := printed())) != len(p):
            draw_values()
        self.notes = notes
        # --- page index of every page of the target table
        tpages = {}
        for pi, p in enumerate(pages):
            if p.get("note") == tn["n"] and p["type"].startswith("note_table"):
                tpages[self.year if p["type"] == "note_table_cur" else self.year - 1] = pi
        year = r.choice((self.year, self.year, self.year - 1))
        ri = r.randrange(len(tn["rows"]))
        ci = r.randrange(2)
        first_idx = idx_of_main[t_first]
        self.target = {
            "note": tn["n"], "title": tn["title"], "table_title": tn["table_title"], "row": tn["rows"][ri],
            "row_index": ri, "col": tn["cols"][ci], "col_index": ci, "year": year,
            "value": tn["values"][year][ri][ci], "page": tpages[year],
            "lookalike_page": tpages[self.year + self.year - 1 - year],
            "note_first_page": first_idx, "note_first_folio": t_first + 1,
            "inserts_before": sum(1 for p in pages[:first_idx] if p["type"] == "insert"),
        }
        self.inserts = [i for i, p in enumerate(pages) if p["type"] == "insert"]

    # ------------------------------------------------------------------ helpers
    def table_on_page(self, i: int) -> dict | None:
        """The table printed on page i (note, year, rows, cols, values), or None."""
        p = self.pages[i]
        if not p["type"].startswith("note_table"):
            return None
        nt = self.notes[p["note"] - 1]
        y = self.year if p["type"] == "note_table_cur" else self.year - 1
        return {"note": nt["n"], "year": y, "rows": nt["rows"], "cols": nt["cols"], "values": nt["values"][y]}

    def amount_xy(self, row: int, col: int, value: int) -> tuple[int, int]:
        """Top-left of the amount text before degradation."""
        s = fmt_amount(value)
        return COL_R[col] - len(s) * AMOUNT_CELL, ROW_Y0 + row * ROW_PITCH + AMOUNT_DY

    # ------------------------------------------------------------------ rendering
    @cached_property
    def _noise(self) -> np.ndarray:
        rng = np.random.default_rng(self.np_seed("noise"))
        n = rng.standard_normal((PH + NOISE_PAD, PW + NOISE_PAD), dtype=np.float32) * NOISE_SIGMA
        return Image.fromarray(np.clip(np.rint(n) + 128, 0, 255).astype(np.uint8), "L")   # biased by +128

    @cached_property
    def _stamp(self) -> Image.Image:
        r = self.rng("stamp")
        im = Image.new("L", (360, 140), 0)
        d = ImageDraw.Draw(im)
        d.rounded_rectangle((4, 4, 355, 135), 12, outline=255, width=6)
        draw_text(d, (180, 18), "RECEIVED", "sans-bold", 44, 255, "center")
        draw_text(d, (180, 78), f"{r.randint(1, 28):02d} MAR {self.year + 1}", "sans-bold", 34, 255, "center")
        return im.rotate(r.uniform(-14, -6), resample=Image.Resampling.BICUBIC, expand=True)

    def _wlen(self, f, wd: str) -> float:
        c = self.__dict__.setdefault("_wcache", {})
        v = c.get(wd)
        if v is None:
            v = c[wd] = f.getlength(wd + " ")
        return v

    def _body(self, d: ImageDraw.ImageDraw, r, y0: int, y1: int) -> None:
        f = font(*BODY_FONT)
        y = y0
        while y + 30 < y1:
            n_lines = r.randint(3, 8)
            for li in range(n_lines):
                if y + 30 >= y1:
                    break
                words, w = [], 0.0
                limit = (MARGIN_R - MARGIN_L) * (r.uniform(0.3, 0.8) if li == n_lines - 1 else 1.0)
                while True:
                    wd = r.choice(WORDS)
                    if r.random() < 0.04:
                        wd = f"EUR {r.randint(1, 999)}.{r.randint(0, 9)} million"
                    ww = self._wlen(f, wd)
                    if w + ww > limit:
                        break
                    words.append(wd)
                    w += ww
                line = " ".join(words)
                if li == 0:
                    line = line[:1].upper() + line[1:]
                d.text((MARGIN_L, y), line, font=f, fill=INK)
                y += 34
            y += 22

    def _header(self, d, p) -> None:
        if p.get("header"):
            draw_text(d, (MARGIN_L, HEADER_Y), p["header"], *CUE_FONT, INK)
        if p.get("folio"):
            draw_text(d, (MARGIN_R, HEADER_Y), p["folio"], *CUE_FONT, INK, "right")
        if p.get("header") is not None:
            d.rectangle((MARGIN_L, RULE_Y, MARGIN_R, RULE_Y + 2), fill=INK)

    def _draw_page(self, i: int) -> Image.Image:
        p = self.pages[i]
        r = self.rng(f"page/{i}")
        im = Image.new("L", (PW, PH), PAPER)
        d = ImageDraw.Draw(im)
        t = p["type"]
        self._header(d, p)
        if t == "cover":
            draw_text(d, (PW // 2, 520), self.company, "sans-bold", 72, INK, "center")
            draw_text(d, (PW // 2, 640), f"Annual Report {self.year}", "sans", 60, INK, "center")
            d.rectangle((300, 760, 940, 766), fill=INK)
        elif t == "notice":
            self._body(d, r, 160, 900)
        elif t == "blank":
            draw_text(d, (PW // 2, 860), "This page is intentionally left blank.", "sans", 26, INK, "center")
        elif t == "contents":
            ft = font(*TOC_FONT)
            for k, (label, title, folio) in enumerate(self.toc):
                y = TOC_Y0 + k * TOC_PITCH
                if label:
                    draw_text(d, (MARGIN_L, y), label, *CUE_FONT, INK)
                draw_text(d, (TOC_TITLE_X, y), title, *TOC_FONT, INK)
                x0 = TOC_TITLE_X + ft.getlength(title) + 16
                x1 = MARGIN_R - text_width(folio, *CUE_FONT) - 16
                x = x0 + (-x0) % 18
                while x < x1:
                    d.ellipse((x, y + 30, x + 4, y + 34), fill=120)
                    x += 18
                draw_text(d, (MARGIN_R, y), folio, *CUE_FONT, INK, "right")
        elif t in ("text",):
            y0 = 150
            if p.get("first"):
                draw_text(d, (MARGIN_L, 150), p["header"], "sans-bold", 44, INK)
                y0 = 240
            self._body(d, r, y0, 1680)
        elif t.startswith("note_table"):
            self._draw_table(d, r, p)
        elif t == "insert":
            draw_text(d, (MARGIN_L, 110), r.choice(("Correction notice", "Letter to shareholders",
                                                       "Errata slip", "Notice of meeting")),
                      "sans-bold", 40, INK)
            draw_text(d, (MARGIN_L, 170), f"{self.company} - inserted sheet", "sans", 26, INK)
            self._body(d, r, 260, 1300)
            d.line((MARGIN_L, 1480, 520, 1480), fill=INK, width=2)
            draw_text(d, (MARGIN_L, 1490), "Signed on behalf of the Board", "sans", 22, INK)
        return im

    def _draw_table(self, d, r, p) -> None:
        nt = self.notes[p["note"] - 1]
        y = self.year if p["type"] == "note_table_cur" else self.year - 1
        draw_text(d, (MARGIN_L, CAPTION_Y), f"Table {nt['n']}.1  {nt['table_title']}", *CUE_FONT, INK)
        draw_text(d, (MARGIN_L, DATE_Y), f"31 December {y}", *CUE_FONT, INK)
        draw_text(d, (MARGIN_R, DATE_Y + 10), "EUR thousand", "sans", 22, INK, "right")
        for c, name in enumerate(nt["cols"]):
            draw_text(d, (COL_R[c], COLHEAD_Y), name, *CUE_FONT, INK, "right")
        d.rectangle((MARGIN_L, COLHEAD_Y + 56, MARGIN_R, COLHEAD_Y + 58), fill=INK)
        fa = font(*AMOUNT_FONT)
        vals = nt["values"][y]
        rows = list(nt["rows"]) + ["Total"]
        tot = [sum(v[c] for v in vals) for c in range(2)]
        for k, label in enumerate(rows):
            ry = ROW_Y0 + k * ROW_PITCH
            draw_text(d, (MARGIN_L, ry), label, *(CUE_FONT if label == "Total" else ("sans", 36)), INK)
            for c in range(2):
                v = vals[k][c] if k < len(vals) else tot[c]
                s = fmt_amount(v)
                x0 = COL_R[c] - len(s) * AMOUNT_CELL
                for j, ch in enumerate(s):
                    if ch != " ":
                        d.text((x0 + j * AMOUNT_CELL, ry + AMOUNT_DY), ch, font=fa, fill=INK)
            d.line((MARGIN_L, ry + ROW_PITCH - 8, MARGIN_R, ry + ROW_PITCH - 8), fill=190, width=1)
        self._body(d, r, ROW_Y0 + len(rows) * ROW_PITCH + 40, 1680)

    def _degrade(self, i: int, im: Image.Image) -> Image.Image:
        r = self.rng(f"scan/{i}")
        dx, dy = r.randint(-MAX_SHIFT, MAX_SHIFT), r.randint(-MAX_SHIFT, MAX_SHIFT)
        a = math.radians(r.uniform(-MAX_SKEW_DEG, MAX_SKEW_DEG))
        ca, sa = math.cos(a), math.sin(a)
        cx, cy = PW / 2, PH / 2
        # output (x, y) samples input (ca*(x-cx-dx) + sa*(y-cy-dy) + cx, -sa*(x-cx-dx) + ca*(y-cy-dy) + cy)
        data = (ca, sa, cx - ca * (cx + dx) - sa * (cy + dy),
                -sa, ca, cy + sa * (cx + dx) - ca * (cy + dy))
        im = im.transform((PW, PH), Image.Transform.AFFINE, data, resample=Image.Resampling.BILINEAR,
                          fillcolor=PAPER)
        # speckles in the margins only (never over text)
        d = ImageDraw.Draw(im)
        for _ in range(r.randint(5, 40)):
            x = r.choice((r.randint(5, 80), r.randint(1160, PW - 6)))
            y = r.randint(5, PH - 6)
            s = r.randint(1, 3)
            d.ellipse((x, y, x + s, y + s), fill=r.randint(40, 140))
        im = im.filter(SOFTEN)
        ox, oy = r.randrange(NOISE_PAD), r.randrange(NOISE_PAD)
        v = ImageChops.add(im, self._noise.crop((ox, oy, ox + PW, oy + PH)), scale=1.0, offset=-128)
        # Paper tint: map grey levels to a warm paper colour with a per-page lookup table.
        paper = r.uniform(236, 250)
        tint = np.asarray([paper, paper - 3, paper - 11], np.float32)
        ink = np.asarray([24, 24, 30], np.float32)
        lut = np.rint(ink + (np.arange(256, dtype=np.float32)[:, None] / 255.0) * (tint - ink)).astype(np.uint8)
        out = Image.merge("RGB", [v.point(lut[:, c].tolist()) for c in range(3)])
        if self.pages[i]["type"] in ("cover", "insert"):
            st = self._stamp
            px = r.randint(680, 1240 - st.width - 20)
            py = r.randint(1380, 1700 - st.height)
            red = Image.new("RGB", st.size, (190, 30, 40))
            out.paste(red, (px, py), st.point(lambda v: v * 0.75))
        return out

    def render(self, frame: int) -> Image.Image:
        i = self._check_frame(frame)
        return self._degrade(i, self._draw_page(i))

    def debug_info(self) -> dict:
        t = self.target
        x, y = self.amount_xy(t["row_index"], t["col_index"], t["value"])
        return {
            "kind": self.kind, "seed": self.seed, "n_pages": len(self.pages), "front_len": self.front_len,
            "inserts": list(self.inserts), "toc": [list(e) for e in self.toc],
            "target": dict(t), "event_frame": t["page"],
            "folios": [p.get("folio") for p in self.pages],
            "answer_text": {"frame": t["page"], "xy": (x, y), "text": fmt_amount(t["value"]),
                            "font": AMOUNT_FONT[0], "size": AMOUNT_FONT[1], "cell": AMOUNT_CELL},
            "answer_font_px": AMOUNT_FONT[1], "cue_font_px": CUE_FONT[1],
        }
