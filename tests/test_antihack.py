"""Cheap strategies score ~0 on the toasts and leaderboard kinds.

Every strategy here reads the scene state directly (a perfect reader), so these are
upper bounds on what each shortcut can earn:
  * stop-early: a well-formed random guess of the stated format (the baseline agent);
  * do-nothing: never submits;
  * "smart cheap": reads the obvious spot on one frame without finding the event
    (newest Error toast / top row / the endpoint's value on the last frame);
  * off-by-a-little: finds the event frame but reads one frame early, or a second late.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

from framehunt.answers import normalize_answer
from framehunt.scenes import KINDS, make_scene
from framehunt.scenes.leaderboard import TICK
from framehunt.server import FrameHuntEnv
from framehunt.splits import derive_seed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "framehunt" / "baselines"))
from stop_early_client import guess  # noqa: E402

N = 200
SALT = "antihack-test-salt"


def seeds(kind: str):
    return [derive_seed("antihack", f"{kind}-{i}", SALT) for i in range(N)]


@pytest.fixture(scope="module")
def toasts():
    return [make_scene("toasts", s) for s in seeds("toasts")]


@pytest.fixture(scope="module")
def boards():
    return [make_scene("leaderboard", s) for s in seeds("leaderboard")]


@pytest.mark.parametrize("kind", KINDS)
def test_stop_early_guess_scores_zero(kind):
    hits = 0
    for i, s in enumerate(seeds(kind)[:N]):
        sc = make_scene(kind, s)
        g = guess(sc.answer_format, random.Random(f"stop-early/{kind}-{i}"))
        hits += normalize_answer(g) == sc.answer
    assert hits == 0


@pytest.mark.parametrize("kind", ["toasts", "leaderboard", "scandoc"])
def test_stop_early_guess_is_well_formed(kind):
    import re
    pat = {"toasts": r"^[A-HJ-NP-Z0-9]{6}$", "leaderboard": r"^[1-9][0-9]{2,3}$", "scandoc": r"^[1-9][0-9]{3,5}$"}[kind]
    for i in range(50):
        sc = make_scene(kind, i)
        assert re.match(pat, guess(sc.answer_format, random.Random(i)))


@pytest.mark.parametrize("kind", ["toasts", "leaderboard", "scandoc"])
def test_do_nothing_scores_zero(kind):
    env = FrameHuntEnv(make_scene(kind, 1), verify_token="t")
    env.info()
    r = env.result()
    assert not r["submitted"] and r["reward"] == 0.0


def test_toasts_smart_cheap(toasts):
    newest_err_last = off_early = late = 0
    for sc in toasts:
        f = sc.debug_info()["event_frame"]
        last = sc.meta.n_frames - 1
        errs = [i for i in sc.visible_at(last) if sc.toasts[i]["level"] == "error"]
        newest_err_last += bool(errs) and sc.toasts[errs[0]]["code"] == sc.answer
        # Right search, one frame early: the newest red toast is the (K-1)-th Error.
        e = [i for i in sc.visible_at(f - 1) if sc.toasts[i]["level"] == "error"]
        off_early += bool(e) and sc.toasts[e[0]]["code"] == sc.answer
        # Right search, 0.6 s late, "the top toast".
        late += sc.toasts[sc.visible_at(f + 6)[0]]["code"] == sc.answer
    assert newest_err_last <= 0.02 * N
    assert off_early == 0 and late == 0


def test_leaderboard_smart_cheap(boards):
    top_last = e_last = early = late = stale_row = 0
    for sc in boards:
        info = sc.debug_info()
        f, e = info["event_frame"], info["endpoint_index"]
        last = sc.meta.n_frames - 1
        top_last += str(sc.table_at(last)[0][1]) == sc.answer
        e_last += str(sc.value_of(e, last)) == sc.answer
        early += str(sc.value_of(e, f - 1)) == sc.answer
        late += str(sc.value_of(e, f + TICK)) == sc.answer
        # Row found one frame early, value read on the right frame from that stale row.
        stale_row += str(sc.table_at(f)[sc.row_of(e, f - 1)][1]) == sc.answer
    assert top_last <= 0.02 * N and e_last <= 0.02 * N
    assert early == 0 and late == 0 and stale_row == 0
