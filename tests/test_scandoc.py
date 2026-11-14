"""The `scandoc` kind: a long scanned annual report (framehunt/scenes/scandoc.py).

Checks the document's structure (numbering, contents, inserts), the traps the task is
built on (printed page numbers are not PDF indexes; the comparative-year table is on
the facing page), legibility (cues at 1/4, amounts only near native), determinism,
the server's document mode, and that cheap shortcuts score ~0.
"""
from __future__ import annotations

import io
import random
import re
import time

import numpy as np
import pytest
from PIL import Image

from framehunt.answers import normalize_answer
from framehunt.reference import Doc
from framehunt.scenes import make_scene
from framehunt.scenes.scandoc import AMOUNT_CELL, AMOUNT_FONT, COL_R, PH, PW, fmt_amount, roman
from framehunt.server import FrameHuntEnv, FrameHuntError
from framehunt.splits import derive_seed

N = 200
SEEDS = [derive_seed("scandoc-test", f"scandoc-{i}", "scandoc-test-salt") for i in range(N)]


@pytest.fixture(scope="module")
def docs():
    return [make_scene("scandoc", s) for s in SEEDS]


def value_for(sc, page: int, row: str, col: str):
    """What a perfect reader gets for (row, col) on PDF page `page` (None: no such cell)."""
    tb = sc.table_on_page(page)
    if tb is None or row not in tb["rows"] or col not in tb["cols"]:
        return None
    return tb["values"][tb["rows"].index(row)][tb["cols"].index(col)]


# --- basics ----------------------------------------------------------------------------
def test_meta_question_answer(docs):
    lengths = []
    for sc in docs:
        m = sc.meta
        assert (m.width, m.height, m.fps) == (PW, PH, 1) and m.n_frames == len(sc.pages)
        assert 150 <= m.n_frames <= 400
        lengths.append(m.n_frames)
        assert re.fullmatch(r"[1-9][0-9]{3,5}", sc.answer)
        assert sc.answer == normalize_answer(fmt_amount(int(sc.answer)))
        assert sc.answer not in sc.question
        t = sc.target
        for s in (t["title"], t["table_title"], t["row"], t["col"], f"31 December {t['year']}"):
            assert s in sc.question
    assert min(lengths) < 175 and max(lengths) > 375
    assert len({sc.answer for sc in docs}) >= N - 3


def test_render_shape_and_range(docs):
    sc = docs[0]
    im = sc.render(0)
    assert im.mode == "RGB" and im.size == (PW, PH)
    with pytest.raises(IndexError):
        sc.render(sc.meta.n_frames)
    with pytest.raises(IndexError):
        sc.render(-1)


def test_determinism_independent_of_order():
    seed = SEEDS[3]
    a, b = make_scene("scandoc", seed), make_scene("scandoc", seed)
    t = a.target
    pages = [0, 1, t["page"], t["lookalike_page"], t["note_first_page"], a.meta.n_frames - 1]
    ra = {p: a.render(p).tobytes() for p in pages}
    for p in random.Random(0).sample(range(a.meta.n_frames), 8):     # other history for b
        b.render(p)
    for p in reversed(pages):
        assert b.render(p).tobytes() == ra[p], p
    assert a.debug_info() == b.debug_info()
    assert (a.question, a.answer, a.answer_format) == (b.question, b.answer, b.answer_format)


def test_render_speed():
    sc = make_scene("scandoc", SEEDS[0])
    sc.render(0)
    pages = random.Random(0).sample(range(sc.meta.n_frames), 10)
    t0 = time.perf_counter()
    for p in pages:
        sc.render(p)
    assert (time.perf_counter() - t0) / len(pages) < 0.5


# --- structure: numbering, contents, inserts --------------------------------------------
def test_numbering_and_contents(docs):
    for sc in docs:
        folios = [p.get("folio") for p in sc.pages]
        assert folios[0] is None                                       # cover
        assert folios[1:sc.front_len] == [roman(i + 1) for i in range(1, sc.front_len)]
        arabic = [f for f in folios[sc.front_len:] if f is not None]
        assert arabic == [str(i + 1) for i in range(len(arabic))]
        inserts = [i for i, p in enumerate(sc.pages) if p["type"] == "insert"]
        assert inserts == sc.inserts and 2 <= len(inserts) <= 4
        assert all(folios[i] is None and sc.pages[i]["header"] is None for i in inserts)
        # Every contents entry points at the printed page that starts that section.
        by_folio = {f: i for i, f in enumerate(folios) if f is not None}
        assert sum(1 for p in sc.pages if p["type"] == "contents") == 1
        for label, title, folio in sc.toc:
            p = sc.pages[by_folio[folio]]
            assert p.get("first") and (p["header"] == title or p["header"] == f"{label}  {title}")


def test_printed_page_numbers_are_not_pdf_indexes(docs):
    """The offset trap: a printed number is never the PDF index of that page, and the
    offset measured at the start of the main matter is wrong by the time you reach the
    target note (at least one unnumbered insert sits in the notes before it)."""
    for sc in docs:
        t = sc.target
        first, folio = t["note_first_page"], t["note_first_folio"]
        assert sc.pages[first]["folio"] == str(folio)
        assert first - folio >= sc.front_len                     # front matter + inserts
        assert t["inserts_before"] >= 2
        main_start = sc.front_len
        assert sc.pages[main_start]["folio"] == "1"
        offset_at_start = main_start - 1
        assert folio + offset_at_start != first
        notes_start = next(i for i, p in enumerate(sc.pages) if (p.get("header") or "").startswith("Note 1  "))
        assert any(notes_start < i < first for i in sc.inserts)
        # Pages that a naive jump lands on are not the target note.
        for guess in (folio, folio + sc.front_len - 1, folio + sc.front_len):
            assert sc.pages[guess].get("note") != t["note"] or guess >= first


def printed_amounts(sc) -> list[int]:
    """Every amount printed in any table, including the Total row's column sums."""
    out = []
    for nt in sc.notes:
        for vals in nt.get("values", {}).values():
            out += [v for r in vals for v in r] + [sum(r[c] for r in vals) for c in range(len(vals[0]))]
    return out


def test_totals_unique():
    """Seed 796 once printed a Total equal to a cell of another table; all printed amounts are unique."""
    sc = make_scene("scandoc", 796)
    vals = printed_amounts(sc)
    assert len(vals) == len(set(vals))


def test_lookalike_comparative_table(docs):
    """The comparative-year table of the same note is on the facing page: same caption,
    rows and columns, other date, different amounts. All amounts are unique, so any
    wrong cell is a well-formed wrong answer."""
    for sc in docs:
        t = sc.target
        a, b = t["page"], t["lookalike_page"]
        assert abs(a - b) == 1
        ta, tb = sc.table_on_page(a), sc.table_on_page(b)
        assert ta["note"] == tb["note"] == t["note"] and ta["rows"] == tb["rows"] and ta["cols"] == tb["cols"]
        assert {ta["year"], tb["year"]} == {sc.year, sc.year - 1} and ta["year"] == t["year"]
        assert sc.pages[a]["header"] == sc.pages[b]["header"]
        assert value_for(sc, a, t["row"], t["col"]) == int(sc.answer)
        assert value_for(sc, b, t["row"], t["col"]) != int(sc.answer)
        vals = printed_amounts(sc)
        assert len(vals) == len(set(vals))
    # Both orders occur, and both years are asked.
    assert {sc.target["page"] < sc.target["lookalike_page"] for sc in docs} == {True, False}
    assert {sc.target["year"] == sc.year for sc in docs} == {True, False}


# --- legibility ---------------------------------------------------------------------------
def _amount_box(sc, m: int = 8):
    t = sc.target
    ry = Doc.ROW_Y0 + t["row_index"] * Doc.ROW_PITCH + Doc.AMOUNT_DY
    right = COL_R[t["col_index"]]
    return (right - 7 * AMOUNT_CELL - m, ry - m, 7 * AMOUNT_CELL + 2 * m, Doc.CELL_H + 6 + 2 * m), right, ry


def test_answer_readable_native_not_quarter(docs):
    """A native crop of the amount reads exactly (bundled-font OCR, searching 1-px
    shifts); the same crop at 1/4 scale, blown back up, gives a wrong amount."""
    assert AMOUNT_FONT == ("mono", 12) and Doc.AMOUNT_FONT == AMOUNT_FONT
    accs, exact = [], 0
    for sc in docs[:40]:
        box, right, ry = _amount_box(sc)
        im = sc.render(sc.target["page"]).crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        got, _ = Doc.ocr_amount(Doc.norm(np.asarray(im, np.int32)), box, right, ry)
        assert got == sc.answer, (got, sc.answer)
        small = im.resize((im.width // 4, im.height // 4), Image.Resampling.LANCZOS).resize(im.size, Image.Resampling.BICUBIC)
        got4, _ = Doc.ocr_amount(Doc.norm(np.asarray(small, np.int32)), box, right, ry)
        n = len(sc.answer)
        accs.append(sum(a == b for a, b in zip(got4[-n:].rjust(n), sc.answer)) / n)
        exact += got4 == sc.answer
    # A matched filter that knows the font and searches 1-px shifts gets ~40 % of the
    # digits at 1/4 (chance: 10 %) and almost never the whole amount.
    assert np.mean(accs) <= 0.45 and exact <= 2, (np.mean(accs), exact)


def test_cues_read_at_quarter_scale(docs):
    """Printed page numbers (top right) read correctly from 1/4-scale header crops."""
    from framehunt.reference import Session

    class _C:
        def __init__(self, env):
            self.env = env

        def call(self, tool, args):
            png, text = getattr(self.env, tool)(**args)
            return Image.open(io.BytesIO(png)).convert("RGB"), text

    for sc in docs[:6]:
        env = FrameHuntEnv(sc, budget_px=10 ** 9)
        s = Session(_C(env), out=None)
        s.fps, s.n_frames = 1, sc.meta.n_frames
        t = sc.target
        for p in (t["note_first_page"], t["page"], sc.front_len, sc.inserts[0]):
            n, _ = Doc.read_folio(s, p, "test")
            want = sc.pages[p].get("folio")
            assert (str(n) if n is not None else None) == want, (p, n, want)


# --- server document mode -------------------------------------------------------------------
def test_server_document_mode():
    sc = make_scene("scandoc", SEEDS[1])
    env = FrameHuntEnv(sc, verify_token="t")
    info = env.info()
    assert f"n_pages: {sc.meta.n_frames}" in info and "PDF page index" in info and sc.answer not in info
    png, text = env.view(5, 0, 0, 400, 100, out_width=100)
    assert text.startswith("page=5 ") and "cost=2500" in text
    _, text = env.view(4.6, 0, 0, 100, 100, out_width=10)
    assert text.startswith("page=5 ")
    for bad in (-1, sc.meta.n_frames - 0.5, sc.meta.n_frames):
        with pytest.raises(FrameHuntError, match="not a page index"):
            env.view(bad, out_width=10)
    _, text = env.contact_sheet(0, 3, 4, 0, 0, 100, 100, out_width=10)
    assert "4 pages" in text and "tile 4: page=3" in text


# --- reward integrity: cheap strategies ------------------------------------------------------
def test_shortcuts_score_about_zero(docs):
    """Perfect readers that skip a step. Upper bounds on what each shortcut can earn."""
    hits = dict.fromkeys(("printed_as_index", "front_offset_only", "other_year", "last_table", "first_note_table"), 0)
    lazy = 0
    for sc in docs:
        t = sc.target
        want = int(sc.answer)
        n = sc.meta.n_frames
        # Use the printed number as the PDF index; read the first table from there on.
        for name, start in (("printed_as_index", t["note_first_folio"]),
                            ("front_offset_only", t["note_first_folio"] + sc.front_len - 1)):
            for i in range(start, min(start + 3, n)):
                v = value_for(sc, i, t["row"], t["col"])
                if v is not None:
                    hits[name] += v == want
                    break
        # Right note, wrong year (the facing page).
        hits["other_year"] += value_for(sc, t["lookalike_page"], t["row"], t["col"]) == want
        # Read the last table in the document / in the first note with a table.
        last_tab = max(i for i in range(n) if sc.table_on_page(i))
        hits["last_table"] += value_for(sc, last_tab, t["row"], t["col"]) == want
        first_tab = min(i for i in range(n) if sc.table_on_page(i))
        hits["first_note_table"] += value_for(sc, first_tab, t["row"], t["col"]) == want
        # Jump with the front-matter offset only, then take the first table found, without
        # checking headers or the date (skips verification; not cheap, but documented).
        for i in range(t["note_first_folio"] + sc.front_len - 1, n):
            if sc.table_on_page(i):
                lazy += value_for(sc, i, t["row"], t["col"]) == want
                break
    assert all(v <= 0.05 * N for v in hits.values()), hits
    assert hits["other_year"] == 0
    assert lazy <= 0.2 * N, lazy
