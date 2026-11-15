"""`mathanim`: a ManimCE math animation where an equation f(x) = <fraction> changes one
constant at a time.

Question: the right-hand side on the first frame where the yellow box is visible.
Trace: the yellow box appears once around the equation and stays to the end, and a
big `step N` counter (top right) counts the changes. Decoy: the equations one step
before and after differ from the answer in one coefficient, sign or exponent, and the
box is still there after the next change.

The video is rendered at build time by `mathanim_render.py` (needs manim and LaTeX:
`uv run --group render`) into a cache keyed by the plan and the render code. The MCP
sidecar only decodes the shipped video (`FH_MEDIA_DIR`), so it needs neither.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from fractions import Fraction
from functools import cached_property
from pathlib import Path

from PIL import Image

from .base import Scene, SceneMeta

W, H = 1920, 1080
FPS = 10
TRANSFORM_FRAMES = 10                 # one TransformMatchingTex per step
FONT_SIZE = 16                        # MathTex: digits ~11 px, exponents ~8 px native
EQ_ANCHOR = (880, 540)                # pixel position of the "=" sign's centre
BOX = (770, 475, 420, 130)            # yellow box (x, y, w, h) in pixels, around the equation
COUNTER_RIGHT_TOP = (1860, 60)        # "step N", digits ~60 px native, ~7 px at 1/8
COEFS = range(2, 20)
EXPS = range(2, 8)
RENDER_SCRIPT = Path(__file__).with_name("mathanim_render.py")
REPO = Path(__file__).resolve().parents[2]
ANSWER_FORMAT = (
    "the right-hand side only, as a sympy expression in x using integers, + - * / ( ) and ** for "
    "powers, e.g. (5*x**4 + 3)/(x**2 - 8). Any mathematically equivalent form is accepted.")


def media_dir() -> Path:
    return Path(os.environ.get("FH_MEDIA_DIR") or REPO / "cache" / "mathanim")


# An expression is (numerator terms, denominator terms); a term is (signed coefficient,
# exponent). The denominator is always x^q + d (leading coefficient 1).
def tex_tokens(terms) -> list[str]:
    out = []
    for i, (c, e) in enumerate(terms):
        if i or c < 0:
            out.append("-" if c < 0 else "+")
        if abs(c) != 1 or e == 0:
            out.append(str(abs(c)))
        if e:
            out.append(f"x^{{{e}}}")
    return out


def sympy_str(terms) -> str:
    s = ""
    for i, (c, e) in enumerate(terms):
        sign = "-" if c < 0 else "+"
        mono = str(abs(c)) if e == 0 else (f"x**{e}" if abs(c) == 1 else f"{abs(c)}*x**{e}")
        s += (f" {sign} " if i else ("-" if c < 0 else "")) + mono
    return s


def expr_str(expr) -> str:
    num, den = expr
    return f"({sympy_str(num)})/({sympy_str(den)})"


class MathAnimScene(Scene):
    kind = "mathanim"

    def __init__(self, seed: int):
        super().__init__(seed)
        r = self.rng("setup")
        n_frames = r.randint(1800, 4800)
        self.meta = SceneMeta(W, H, FPS, n_frames)

        exps = sorted(r.sample(EXPS, 2), reverse=True)[:r.choice((1, 2))] + [0]
        num = [(r.choice(COEFS) * (r.choice((1, -1)) if i else 1), e) for i, e in enumerate(exps)]
        expr = (num, [(1, r.choice(EXPS)), (r.choice(COEFS) * r.choice((1, -1)), 0)])

        # Timeline: step 0 is on screen from frame 0; each later step is a transform of
        # TRANSFORM_FRAMES frames, then a settled hold. starts[i] is step i's first
        # settled frame; the transform runs over the TRANSFORM_FRAMES frames before it.
        self.exprs, self.starts = [expr], [0]
        f = r.randint(10, 30)
        while f + TRANSFORM_FRAMES + 10 < n_frames:
            expr = self._change(r, expr)
            self.exprs.append(expr)
            self.starts.append(f + TRANSFORM_FRAMES)
            f += TRANSFORM_FRAMES + r.randint(10, 30)

        # The box appears on a settled frame of step k, 3-10 frames after the change, and
        # 6-12 frames before the next transform starts.
        lo, hi = 0.1 * n_frames, 0.9 * n_frames
        ks = [i for i in range(1, len(self.exprs) - 1) if lo <= self.starts[i] <= hi]
        self.k = r.choice(ks)
        self.box_frame = self.starts[self.k] + r.randint(3, 10)
        nxt = self.box_frame + r.randint(6, 12)
        shift = nxt - (self.starts[self.k + 1] - TRANSFORM_FRAMES)
        self.starts[self.k + 1:] = [s + shift for s in self.starts[self.k + 1:]]
        while self.starts[-1] + 10 > n_frames:
            self.exprs.pop()
            self.starts.pop()

        self.answer = expr_str(self.exprs[self.k])
        self.question = (
            "The video is a math animation of an equation f(x) = (a fraction) that changes one "
            "constant at a time: a coefficient, a sign or an exponent. The 'step N' counter in the "
            "top right counts the changes so far. At some point a yellow box appears around the "
            "equation and stays until the end. What is the right-hand side of the equation on the "
            "first frame where the yellow box is visible?")
        self.answer_format = ANSWER_FORMAT

    @staticmethod
    def _change(r, expr):
        """A copy of expr with one coefficient, sign or exponent changed."""
        num, den = [list(t) for t in expr]
        side, terms = r.choice([("num", num), ("num", num), ("den", den)])
        i = r.randrange(len(terms))
        c, e = terms[i]
        what = r.choice(["coef", "sign", "exp"] if e else ["coef", "sign"])
        if i == 0 and what == "sign":         # keep the leading term positive
            what = "coef"
        if side == "den" and i == 0:          # and the denominator monic
            what = "exp"
        if what == "coef":
            c = (1 if c > 0 else -1) * r.choice([v for v in COEFS if v != abs(c)])
        elif what == "sign":
            c = -c
        else:                                 # keep numerator exponents strictly decreasing
            hi = terms[i - 1][1] if side == "num" and i else max(EXPS) + 1
            lo = terms[i + 1][1] if side == "num" and i + 1 < len(terms) else min(EXPS) - 1
            opts = [v for v in EXPS if lo < v < hi and v != e]
            if not opts:
                c = (1 if c > 0 else -1) * r.choice([v for v in COEFS if v != abs(c)])
            else:
                e = r.choice(opts)
        terms[i] = (c, e)
        return num, den

    # --- rendering -----------------------------------------------------------------
    def plan(self) -> dict:
        """Everything the renderer needs; the video is a pure function of this and its code."""
        return {"width": W, "height": H, "fps": FPS, "n_frames": self.meta.n_frames,
                "transform_frames": TRANSFORM_FRAMES, "font_size": FONT_SIZE, "eq_anchor": EQ_ANCHOR,
                "box": BOX, "box_frame": self.box_frame, "counter_right_top": COUNTER_RIGHT_TOP,
                "steps": [{"start": s, "num": tex_tokens(e[0]), "den": tex_tokens(e[1])}
                          for s, e in zip(self.starts, self.exprs)]}

    @cached_property
    def video_path(self) -> Path:
        key = json.dumps(self.plan(), sort_keys=True).encode() + RENDER_SCRIPT.read_bytes()
        return media_dir() / f"{hashlib.sha256(key).hexdigest()[:24]}.mp4"

    def ensure_video(self) -> Path:
        """Render the video with manim if it is not in the cache yet."""
        p = self.video_path
        if not p.is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.mp4")
            subprocess.run([sys.executable, str(RENDER_SCRIPT), str(tmp)],
                           input=json.dumps(self.plan()), text=True, check=True, stdout=subprocess.DEVNULL)
            tmp.rename(p)
        return p

    @cached_property
    def _video(self):
        import av
        c = av.open(str(self.ensure_video()))
        return c, c.streams.video[0]

    def render(self, frame: int) -> Image.Image:
        frame = self._check_frame(frame)
        c, s = self._video
        c.seek(int(Fraction(frame, FPS) / s.time_base), stream=s)
        for vf in c.decode(s):
            i = round(vf.pts * s.time_base * FPS)
            if i == frame:
                return vf.to_image()
            if i > frame:
                break
        raise RuntimeError(f"frame {frame} not found in {self.video_path}")

    def check_answer(self, answer: str) -> bool:
        from framehunt.mathcheck import check
        return check(answer, self.answer)

    def debug_info(self) -> dict:
        return {"kind": self.kind, "seed": self.seed, "k": self.k, "box_frame": self.box_frame,
                "n_steps": len(self.exprs), "step_starts": self.starts,
                "event_frame": self.box_frame,
                "settled_window": (self.starts[self.k], self.starts[self.k + 1] - TRANSFORM_FRAMES),
                "answer": self.answer,
                "neighbours": (expr_str(self.exprs[self.k - 1]), expr_str(self.exprs[self.k + 1]))}
