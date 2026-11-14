"""Tests for the FrameHunt scene generators (framehunt/scenes)."""
from __future__ import annotations

import collections
import itertools
import random
import re
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from framehunt.answers import normalize_answer
from framehunt.scenes import DOCUMENT_KINDS, KINDS, VIDEO_KINDS, Scene, SceneMeta, make_scene
from framehunt.scenes.base import FONT_DIR, font

SEEDS = [0, 1, 2, 7, 42, 1234]
FORMATS = {
    "logstream": re.compile(r"^[0-9A-F]{8}$"),
    "dialogs": re.compile(r"^[A-Z0-9]{6}$"),
    "dashboard": re.compile(r"^[1-9][0-9]{2,4}$"),
    "toasts": re.compile(r"^[A-HJ-NP-Z0-9]{6}$"),
    "leaderboard": re.compile(r"^[1-9][0-9]{2,3}$"),
}


def eighth(im: Image.Image) -> Image.Image:
    return im.resize((im.width // 8, im.height // 8), Image.Resampling.LANCZOS)


def differs(a: np.ndarray, b: np.ndarray, min_px: int = 8) -> bool:
    """At least `min_px` low-res pixels change strongly (a digit flips, a colour swaps)."""
    return int((np.abs(a - b).max(-1) > 40).sum()) >= min_px


def lowres_box(im: Image.Image, box, inset: int = 3) -> np.ndarray:
    """Trace box region of the 1/8-scale frame, inset to avoid resampling bleed."""
    x, y, w, h = box
    small = np.asarray(eighth(im), np.int16)
    return small[y // 8 + inset:(y + h) // 8 - inset, x // 8 + inset:(x + w) // 8 - inset]


@pytest.fixture(scope="module", params=[(k, s) for k in VIDEO_KINDS for s in SEEDS],
                ids=lambda p: f"{p[0]}-{p[1]}")
def scene(request) -> Scene:
    return make_scene(*request.param)


# --- basics ----------------------------------------------------------------------
def test_normalize_answer():
    assert normalize_answer("  k7q 2zp\n") == "K7Q2ZP"
    assert normalize_answer("\t3a9F 00c1 ") == "3A9F00C1"
    assert normalize_answer("") == ""


def test_kinds_and_unknown():
    assert KINDS == ("logstream", "dialogs", "dashboard", "toasts", "leaderboard", "scandoc")
    assert DOCUMENT_KINDS == ("scandoc",) and VIDEO_KINDS == KINDS[:5]
    with pytest.raises(ValueError):
        make_scene("nope", 1)


def test_fonts_bundled():
    names = {p.name for p in FONT_DIR.iterdir()}
    for f in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSansMono.ttf",
              "DejaVuSansMono-Bold.ttf", "LICENSE"):
        assert f in names
    assert str(FONT_DIR) in font("mono", 16).path


def test_meta_and_fields(scene):
    m = scene.meta
    assert isinstance(m, SceneMeta)
    assert (m.width, m.height, m.fps) == (1920, 1080, 10)
    assert 1800 <= m.n_frames <= 4800            # 3-8 minutes
    assert m.duration_s == m.n_frames / m.fps
    assert scene.answer == normalize_answer(scene.answer)
    assert FORMATS[scene.kind].match(scene.answer), scene.answer
    assert scene.question and scene.answer_format
    assert scene.answer not in scene.question
    im = scene.render(0)
    assert im.mode == "RGB" and im.size == (1920, 1080)
    with pytest.raises(IndexError):
        scene.render(m.n_frames)
    with pytest.raises(IndexError):
        scene.render(-1)


# --- determinism -----------------------------------------------------------------
def test_determinism_independent_of_order(scene):
    kind, seed = scene.kind, scene.seed
    ev = scene.debug_info()["event_frame"]
    n = scene.meta.n_frames
    frames = [0, ev - 1, ev, ev + 13, n // 2, n - 1, random.Random(seed).randrange(n)]
    a = make_scene(kind, seed)
    b = make_scene(kind, seed)
    ra = {f: a.render(f).tobytes() for f in frames}
    for f in range(0, n, max(1, n // 25)):       # warm b's caches with other frames
        b.render(f)
    rb = {f: b.render(f).tobytes() for f in reversed(frames)}
    for f in frames:
        assert ra[f] == rb[f], f"frame {f} differs"
    # The module-level fixture instance (other history) agrees too.
    assert scene.render(ev).tobytes() == ra[ev]
    assert a.debug_info() == b.debug_info()
    assert (a.question, a.answer, a.answer_format) == (b.question, b.answer, b.answer_format)


def test_render_returns_fresh_image(scene):
    im = scene.render(5)
    ImageDraw.Draw(im).rectangle((0, 0, 1919, 1079), fill=(1, 2, 3))
    assert scene.render(5).getpixel((960, 540)) != (1, 2, 3)


# --- answer randomness -------------------------------------------------------------
@pytest.mark.parametrize("kind", VIDEO_KINDS)   # scandoc: tests/test_scandoc.py
def test_answers_random_across_seeds(kind):
    answers, ks, events = [], [], []
    for seed in range(200):
        sc = make_scene(kind, seed)
        info = sc.debug_info()
        answers.append(sc.answer)
        ks.append(info.get("k"))
        events.append(info["event_frame"] / sc.meta.n_frames)
    counts = collections.Counter(answers)
    assert len(counts) >= 190, counts.most_common(5)
    assert counts.most_common(1)[0][1] <= 3, counts.most_common(5)
    assert len({round(e, 3) for e in events}) >= 150
    assert min(events) < 0.25 and max(events) > 0.75
    if kind == "leaderboard":
        assert len(set(ks)) >= 5
        assert len({make_scene(kind, s).debug_info()["endpoint"] for s in range(40)}) >= 10
    elif kind != "dashboard":
        assert len(set(ks)) >= 5
    else:
        assert len({make_scene(kind, s).debug_info()["tile"] for s in range(30)}) >= 6
        assert len({make_scene(kind, s).debug_info()["kpi"] for s in range(30)}) >= 6


# --- answer legibility ----------------------------------------------------------------
def _template(info: dict, crop: Image.Image, text: str) -> Image.Image:
    """Re-draw `text` independently with the bundled font on the answer's background."""
    t = Image.new("RGB", crop.size, tuple(info["bg"]))
    d = ImageDraw.Draw(t)
    f = font(info["font"], info["size"])
    if "grid_cell" in info:      # terminal: one glyph per fixed cell
        cw, _, yoff = info["grid_cell"]
        for i, ch in enumerate(text):
            d.text((4 + i * cw, 4 + yoff), ch, font=f, fill=tuple(info["fill"]))
    else:
        d.text((4, 4), text, font=f, fill=tuple(info["fill"]))
    return t


def test_answer_visible_at_key_frame(scene):
    info = scene.debug_info()
    at = info["answer_text"]
    assert normalize_answer(at["text"]) == scene.answer
    x, y = at["xy"]
    w = int(font(at["font"], at["size"]).getlength(at["text"])) + 8
    h = at["size"] + 10
    crop = scene.render(at["frame"]).crop((x - 4, y - 4, x - 4 + w, y - 4 + h))
    a = np.asarray(crop, np.int16)
    good = np.abs(a - np.asarray(_template(at, crop, at["text"]), np.int16)).mean()
    assert good < 2.0, good
    # A one-character change is clearly worse, so the crop really shows this answer.
    wrong = at["text"][:-1] + ("7" if at["text"][-1] != "7" else "2")
    bad = np.abs(a - np.asarray(_template(at, crop, wrong), np.int16)).mean()
    assert bad > max(3 * good, 1.0), (good, bad)


def test_font_sizes_trace_vs_answer(scene):
    info = scene.debug_info()
    # Answer glyphs: <= 18 px font -> under ~4.5 px at 1/4 scale (unreadable).
    assert 13 <= info["answer_font_px"] <= 18
    # Trace glyphs: >= 44 px font -> >= 5.5 px at 1/8 (digits/words large & high contrast).
    assert info["trace"]["font_px"] >= 44


# --- trace rule ------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["logstream", "dialogs", "toasts", "leaderboard"])
@pytest.mark.parametrize("seed", SEEDS[:4])
def test_counter_trace_at_eighth_scale(kind, seed):
    sc = make_scene(kind, seed)
    info = sc.debug_info()
    box = info["trace"]["box"]
    changes = info["trace"]["change_frames"]
    n = sc.meta.n_frames
    assert changes == sorted(changes) and len(set(changes)) == len(changes)
    # Reference region for each count value, then check monotonic consistency.
    regions = {}
    probe = sorted({0, n - 1} | {c for c in changes} | {c - 1 for c in changes}
                   | {c + 17 for c in changes if c + 17 < n} | set(range(0, n, n // 12)))
    for f in probe:
        cnt = sum(1 for c in changes if c <= f)
        reg = lowres_box(sc.render(f), box)
        if cnt in regions:
            assert np.abs(reg - regions[cnt]).max() <= 2, (f, cnt)
        else:
            regions[cnt] = reg
    vals = list(regions.values())
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            assert differs(vals[i], vals[j]), "different counts look the same"
    for c in changes:
        a, b = lowres_box(sc.render(c - 1), box), lowres_box(sc.render(c), box)
        assert differs(a, b)


def test_logstream_counter_values():
    for seed in SEEDS:
        sc = make_scene("logstream", seed)
        info = sc.debug_info()
        frames = [e["appear_frame"] for e in info["errors"]]
        assert frames == info["trace"]["change_frames"]
        for k, f in enumerate(frames, start=1):
            assert sc.errors_at(f - 1) == k - 1
            assert sc.errors_at(f) == k
            end = frames[k] if k < len(frames) else sc.meta.n_frames
            assert all(sc.errors_at(g) == k for g in range(f, end, 7))
        assert sc.errors_at(info["event_frame"]) == info["k"]


def test_dialogs_badge_values():
    for seed in SEEDS:
        sc = make_scene("dialogs", seed)
        info = sc.debug_info()
        for d in info["dialogs"]:
            assert sc.badge_at(d["open_frame"] - 1) == d["k"] - 1
            assert sc.badge_at(d["open_frame"]) == d["k"]
            assert sc.badge_at(d["close_frame"]) == d["k"]
        assert sc.badge_at(info["event_frame"]) == info["k"]


@pytest.mark.parametrize("seed", SEEDS)
def test_dashboard_red_tile_at_eighth_scale(seed):
    sc = make_scene("dashboard", seed)
    info = sc.debug_info()
    fr = info["first_red_frame"]
    n = sc.meta.n_frames

    def is_red(reg):
        r, g, b = reg.reshape(-1, 3).mean(0)
        return r > 150 and g < 90 and b < 100

    for f in sorted({0, fr - 1, fr, fr + 1, n - 1, fr // 2, (fr + n) // 2}):
        im = sc.render(f)
        for i, box in enumerate(info["tile_boxes"]):
            red = is_red(lowres_box(im, box))
            assert red == (i == info["tile_index"] and f >= fr), (f, i)


def test_dashboard_only_named_tile_ever_red():
    for seed in SEEDS:
        sc = make_scene("dashboard", seed)
        info = sc.debug_info()
        s = info["tile_index"]
        for f in range(sc.meta.n_frames):
            for i in range(len(info["tile_boxes"])):
                st = sc.tile_state(i, f)
                assert (st == 2) == (i == s and f >= info["first_red_frame"])


# --- timing & unambiguity -----------------------------------------------------------
def _in_10_90(f, n):
    return 0.1 * n <= f <= 0.9 * n


def test_logstream_timing_and_unique_ids():
    for seed in range(40):
        sc = make_scene("logstream", seed)
        info = sc.debug_info()
        n = sc.meta.n_frames
        for e in info["errors"]:
            assert e["scroll_off_frame"] - e["appear_frame"] >= 60   # visible >= 6 s
            assert _in_10_90(e["appear_frame"], n)
            i = e["line_index"]
            assert i in sc.visible_lines(e["appear_frame"])
            assert i in sc.visible_lines(e["scroll_off_frame"] - 1)
            if e["scroll_off_frame"] < n:
                assert i not in sc.visible_lines(e["scroll_off_frame"])
            assert i not in sc.visible_lines(e["appear_frame"] - 1)
        rids = [ln[2] for ln in sc.lines]
        assert len(set(rids)) == len(rids)                      # answer shown only once
        levels = collections.Counter(ln[0] for ln in sc.lines)
        assert levels["ERROR"] == info["n_errors"] and levels["WARN"] > 10
        assert sc.answer == info["errors"][info["k"] - 1]["request_id"].upper()


def test_dialogs_timing_and_codes():
    for seed in range(40):
        sc = make_scene("dialogs", seed)
        info = sc.debug_info()
        n = sc.meta.n_frames
        dls = info["dialogs"]
        codes = [d["code"] for d in dls]
        assert len(set(codes)) == len(codes)
        assert len({c[:2] for c in codes}) == 1                 # look-alike prefix
        assert sc.answer == codes[info["k"] - 1]
        for a, b in itertools.pairwise(dls):
            assert b["open_frame"] - a["close_frame"] >= 40       # >= 4 s apart
        for d in dls:
            assert 30 <= d["close_frame"] - d["open_frame"] <= 60
            assert _in_10_90(d["open_frame"], n) and d["close_frame"] <= 0.9 * n
            assert sc.open_dialog_at(d["open_frame"]) == d["k"] - 1
            assert sc.open_dialog_at(d["close_frame"] - 1) == d["k"] - 1
            assert sc.open_dialog_at(d["close_frame"]) is None
            assert sc.open_dialog_at(d["open_frame"] - 1) is None


def test_dashboard_kpi_window():
    for seed in range(40):
        sc = make_scene("dashboard", seed)
        info = sc.debug_info()
        n = sc.meta.n_frames
        fr = info["first_red_frame"]
        k = info["kpi_index"]
        w0, w1 = info["kpi_window"]
        assert _in_10_90(fr, n)
        assert fr - w0 >= 10 and w1 - fr >= 10                  # >= 1 s on both sides
        assert {sc.kpi_value(k, f) for f in range(w0, w1 + 1)} == {int(sc.answer)}
        assert sc.kpi_value(k, w0 - 1) != int(sc.answer)
        assert sc.kpi_value(k, w1 + 1) != int(sc.answer)
        assert sc.tile_state(info["tile_index"], fr - 1) != 2
        assert info["tile"] in sc.question and info["kpi"] in sc.question


# --- speed ----------------------------------------------------------------------------
@pytest.mark.parametrize("kind", VIDEO_KINDS)
def test_render_speed(kind):
    sc = make_scene(kind, 3)
    sc.render(0)                                   # build static layers once
    frames = random.Random(0).sample(range(sc.meta.n_frames), 20)
    t = time.perf_counter()
    for f in frames:
        sc.render(f)
    per = (time.perf_counter() - t) / len(frames)
    assert per < 0.1, f"{kind}: {per * 1000:.1f} ms/frame"


# --- toasts & leaderboard: the event window --------------------------------------------
# Being slightly off must give a confident wrong answer: at f-1 the target location
# shows a well-formed distractor, [f, f+9] shows the answer, and by f+15 the target
# has moved or changed.
NEW_SEEDS = range(60)


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_toasts_window(seed):
    sc = make_scene("toasts", seed)
    info = sc.debug_info()
    f, k, n = info["event_frame"], info["k"], sc.meta.n_frames
    tgt = info["target_index"]
    assert _in_10_90(f, n) and k >= 2
    assert sc.badge_at(f - 1) == k - 1 and sc.badge_at(f) == k
    codes = [t["code"] for t in sc.toasts]
    assert len(set(codes)) == len(codes) and len({c[:2] for c in codes}) == 1
    assert sc.answer == codes[tgt] and sc.toasts[tgt]["level"] == "error"
    # f-1: the top slot holds another toast; the (K-1)-th Error toast is on screen and is
    # the newest red one (the convincing distractor).
    vis = sc.visible_at(f - 1)
    assert vis and vis[0] != tgt and tgt not in vis
    prev = sc.error_idx[k - 2]
    assert prev in vis
    assert [i for i in vis if sc.toasts[i]["level"] == "error"][0] == prev
    # [f, f+9]: the answer is on screen; on f it is the top toast.
    assert sc.visible_at(f)[0] == tgt
    for g in range(f, f + 10):
        assert tgt in sc.visible_at(g)
    # The next toast arrives 0.2-0.6 s later, so by f+6 the target left the top slot.
    assert 2 <= info["top_slot_until"] - f <= 6
    assert sc.visible_at(f + 6)[0] != tgt
    assert sc.visible_at(f + 15)[0] != tgt
    # At most 5 toasts, newest first, nothing ever reappears.
    for g in range(f - 200, f + 200, 7):
        v = sc.visible_at(g)
        assert len(v) <= 5 and v == sorted(v, reverse=True)


@pytest.mark.parametrize("seed", NEW_SEEDS)
def test_leaderboard_window(seed):
    from framehunt.scenes.leaderboard import TICK

    sc = make_scene("leaderboard", seed)
    info = sc.debug_info()
    f, k, n = info["event_frame"], info["k"], sc.meta.n_frames
    e = info["endpoint_index"]
    assert _in_10_90(f, n)
    assert sc.alerts_at(f - 1) == k - 1 and sc.alerts_at(f) == k
    assert info["endpoint"] in sc.question and str(k) in sc.question
    row = sc.row_of(e, f)
    # [f, f+TICK-1]: same row, same value (= answer).
    for g in range(f, f + TICK):
        assert sc.row_of(e, g) == row and str(sc.value_of(e, g)) == sc.answer
        assert sc.table_at(g)[row] == (info["endpoint"], int(sc.answer))
    # f-1 and f+TICK: E sits in another row with another value; E's row at f shows
    # another endpoint with a well-formed value.
    for g in (f - 1, f + TICK):
        assert sc.row_of(e, g) != row and str(sc.value_of(e, g)) != sc.answer
        name, val = sc.table_at(g)[row]
        assert name != info["endpoint"] and 100 <= val <= 9999
    # No other row shows the same number on the event frame (a wrong row never scores).
    assert [v for _, v in sc.table_at(f)].count(int(sc.answer)) == 1
    # Every alert fires on a table refresh (so the asked value is shown for exactly 1 s).
    for a in info["alert_frames"]:
        assert sc._tick(a) != sc._tick(a - 1)
    assert all(b - a >= 30 for a, b in itertools.pairwise(info["alert_frames"]))


def test_leaderboard_rows_move():
    """Rows really move: an endpoint changes row on a good share of the refreshes."""
    rates = []
    for seed in range(20):
        sc = make_scene("leaderboard", seed)
        r = sc.rank
        rates.append(float((r[1:] != r[:-1]).mean()))
    assert 0.2 < np.mean(rates) < 0.6, np.mean(rates)


# --- toasts & leaderboard: legibility -----------------------------------------------------
def _answer_crop(sc, pad: int = 8):
    at = sc.debug_info()["answer_text"]
    x, y = at["xy"]
    w = int(font(at["font"], at["size"]).getlength(at["text"]))
    im = sc.render(at["frame"])
    return at, im.crop((x - pad, y - pad, x + w + pad, y + at["size"] + 10))


ALPHABETS = {"toasts": "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789", "leaderboard": "0123456789"}


@pytest.mark.parametrize("kind", ["toasts", "leaderboard"])
def test_answer_readable_native_not_quarter(kind):
    """Native crop OCRs exactly; the same crop seen at 1/4 scale (then blown back up)
    gets at most ~2 of 6 characters right on average.

    This models a reader that looks at the image. A matched filter with the bundled font
    and the exact sub-pixel origin can still decode 1/4 crops (on every kind); that only
    saves pixels, never replaces finding the event. See docs/design.md, hard rule 4.
    """
    from framehunt.reference import ocr_string

    accs = []
    for seed in SEEDS:
        sc = make_scene(kind, seed)
        at, crop = _answer_crop(sc)
        n = len(at["text"])
        a = np.asarray(crop, np.int32)
        args = (at["font"], at["size"], tuple(at["fill"]), tuple(at["bg"]), ALPHABETS[kind], n, n)
        assert ocr_string(a, (8, 8), *args)[0] == at["text"]
        small = crop.resize((crop.width // 4, crop.height // 4), Image.Resampling.LANCZOS)
        up = np.asarray(small.resize(crop.size, Image.Resampling.BICUBIC), np.int32)
        got = ocr_string(up, (8, 8), *args)[0]
        accs.append(sum(a_ == b_ for a_, b_ in zip(got, at["text"])) / n)
    assert np.mean(accs) <= 2 / 6 + 1e-9, accs


@pytest.mark.parametrize("seed", SEEDS)
def test_toast_level_stripe_at_eighth_scale(seed):
    """The level of every visible toast is identifiable from its colour stripe at 1/8."""
    from framehunt.scenes.toasts import LEVEL_RGB, STRIPE_W, slot_box

    sc = make_scene("toasts", seed)
    f = sc.debug_info()["event_frame"]
    for g in (f - 1, f, f + 7, f + 31):
        small = np.asarray(eighth(sc.render(g)), np.int32)
        for slot, i in enumerate(sc.visible_at(g)):
            b = slot_box(slot)
            px = small[b.y // 8 + 2:b.bottom // 8 - 2, b.x // 8 + 1:(b.x + STRIPE_W) // 8].reshape(-1, 3).mean(0)
            best = min(LEVEL_RGB, key=lambda lv: np.abs(px - np.asarray(LEVEL_RGB[lv])).sum())
            assert best == sc.toasts[i]["level"], (g, slot, px)
            assert np.abs(px - np.asarray(LEVEL_RGB[best])).sum() < 90


def test_toast_texts_fit():
    from framehunt.scenes.base import text_width
    from framehunt.scenes.toasts import LEVEL_WORD, TEXTS, TOAST_W

    for lv, items in TEXTS.items():
        for title, msg in items:
            assert text_width(msg.format(inv="INV-88888"), "sans", 15) <= TOAST_W - 56
            assert text_width(f"{LEVEL_WORD[lv]}: {title}", "sans-bold", 18) <= TOAST_W - 116


def test_toasts_lookalikes_around_the_event():
    """The (K+1)-th Error arrives within 1.5 s of f, so a second late the target is no
    longer the newest red toast, and on most items several red toasts are on screen."""
    many_red = 0
    for seed in range(100):
        sc = make_scene("toasts", seed)
        info = sc.debug_info()
        f, k, t = info["event_frame"], info["k"], info["target_index"]
        assert info["errors"][k]["arrive_frame"] - f <= 15            # (K+1)-th Error
        reds = [i for i in sc.visible_at(f + 15) if sc.toasts[i]["level"] == "error"]
        assert reds[0] != t
        many_red += len(reds) >= 2
        ends = [tt["end"] for tt in sc.toasts]
        assert ends == sorted(ends)                                     # first in, first out
        arr = [tt["arrive"] for tt in sc.toasts]
        for i, tt in enumerate(sc.toasts):                              # 4-6 s, or pushed out by
            d = tt["end"] - tt["arrive"]                                # the 5th newer toast
            assert d >= 10                                              # (5 arrivals >= 10 frames)
            evicted = i + 5 < len(arr) and tt["end"] == arr[i + 5]
            assert d <= 60 and (evicted or d >= 40)
    assert many_red >= 80, many_red
