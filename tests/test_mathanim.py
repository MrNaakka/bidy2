"""mathanim: the answer checker and the scene's timing guarantees (no video needed).

The reference run on the rendered DEV items is in test_reference.py:
FH_REF_SPLIT=dev uv run --group render pytest -m reference -k mathanim
"""
from __future__ import annotations

import pytest

from framehunt.mathcheck import check
from framehunt.scenes import make_scene
from framehunt.scenes.mathanim import TRANSFORM_FRAMES
from framehunt.server import FrameHuntEnv

TRUTH = "(7*x**3 - 12*x**2 + 4)/(x**5 - 9)"


@pytest.mark.parametrize("answer", [
    TRUTH,
    "(4 - 12*x**2 + 7*x**3) / (-9 + x**5)",
    "(-7*x**3 + 12*x**2 - 4)/(9 - x**5)",
    "7*x**3/(x**5 - 9) - (12*x**2 - 4)/(x**5 - 9)",
    "(7.0*x**3 - 12*x**2 + 4)/(x**5 - 9)",
    "(7*x**4 - 12*x**3 + 4*x)/(x*(x**5 - 9))",  # removable singularity at 0
    "(7*x**4 - 19*x**3 + 12*x**2 + 4*x - 4)/(x**6 - x**5 - 9*x + 9)",  # times (x-1)/(x-1)
])
def test_equivalent_forms_pass(answer):
    assert check(answer, TRUTH)


@pytest.mark.parametrize("answer", [
    "(7*x**3 - 13*x**2 + 4)/(x**5 - 9)",      # coefficient
    "(7*x**3 + 12*x**2 + 4)/(x**5 - 9)",      # sign
    "(7*x**3 - 12*x**4 + 4)/(x**5 - 9)",      # exponent
    "(7*x**3 - 12*x**2 + 4)/(x**5 + 9)",      # denominator sign
    "f(x) = (7*x**3 - 12*x**2 + 4)/(x**5 - 9)",
    "__import__('os').system('x')",
    "9**9**9**9",                              # would not finish: killed by the timeout
    "(1, 2)",
    f"{TRUTH} + (x + 60)//121",               # floor term: zero on a bounded range only
    f"{TRUTH} + ((x + 1000001)**2)**(1/2) - (x + 1000001)",   # zero for x > -1000001 only
    f"{TRUTH} + 2**(1/2)/10**40",             # tiny irrational offset
    f"{TRUTH} + 1/0",
])
def test_wrong_and_hostile_answers_fail(answer):
    assert not check(answer, TRUTH)


def test_server_uses_the_equivalence_check():
    sc = make_scene("mathanim", 11)
    env = FrameHuntEnv(sc, verify_token="t")
    num, den = sc.answer.split(")/(")
    env.submit(f"2*({num[1:]})/(2*({den[:-1]}))")
    assert env.result()["correct"]


@pytest.mark.parametrize("seed", range(200))
def test_timeline_guarantees(seed):
    """The box appears on a settled frame of step k, with room on both sides, and the
    steps before and after are different functions (the one-step-off decoys)."""
    sc = make_scene("mathanim", seed)
    k, starts = sc.k, sc.starts
    assert starts[k] + 3 <= sc.box_frame <= starts[k + 1] - TRANSFORM_FRAMES - 6
    assert 0.1 * sc.meta.n_frames <= starts[k] <= 0.9 * sc.meta.n_frames
    assert starts[-1] + 10 <= sc.meta.n_frames
    before, after = sc.debug_info()["neighbours"]
    assert before != sc.answer != after
