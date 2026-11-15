#!/usr/bin/env python
"""Cut the reference's glyph templates for `mathanim` from renders of the real pipeline.

    uv run --group render python scripts/make_mathanim_glyphs.py

Renders 60 random equations with framehunt/scenes/mathanim_render.py (same font size and
position as the scenes), segments each fraction with reference.MathAnim and labels the
components from the known tokens. Writes framehunt/assets/mathanim_glyphs.npz.
"""
from __future__ import annotations

import random
import sys
import tempfile
from pathlib import Path

import av
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framehunt.reference import MathAnim
from framehunt.scenes import make_scene
from framehunt.scenes.mathanim import COEFS, EXPS, tex_tokens
from framehunt.scenes.mathanim_render import render

N, PER_LABEL, SIZE = 60, 16, 24


def labels(tokens: list[str]) -> list[str]:
    out = []
    for t in tokens:
        out += ["x", f"^{t[3:-1]}"] if t.startswith("x^") else list(t)
    return out


def main() -> None:
    r = random.Random(0)
    steps = []
    for i in range(N):
        exps = sorted(r.sample(EXPS, 2), reverse=True)[:r.choice((1, 2))] + [0]
        num = [(r.choice(COEFS) * (r.choice((1, -1)) if j else 1), e) for j, e in enumerate(exps)]
        den = [(1, r.choice(EXPS)), (r.choice(COEFS) * r.choice((1, -1)), 0)]
        steps.append({"start": 2 * i, "num": tex_tokens(num), "den": tex_tokens(den)})
    plan = {**make_scene("mathanim", 0).plan(), "steps": steps, "n_frames": 2 * N, "transform_frames": 1,
            "box_frame": 2 * N}
    x, y, w, h = MathAnim.RHS
    found: dict[str, list[np.ndarray]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        render(plan, f"{tmp}/glyphs.mp4")
        with av.open(f"{tmp}/glyphs.mp4") as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    for i, step in enumerate(steps):
        ink = MathAnim.ink(frames[2 * i][y:y + h, x:x + w].astype(np.int32))
        for line, toks in zip(MathAnim.lines(ink), (step["num"], step["den"])):
            labs = labels(toks)
            if len(line) != len(labs):
                sys.exit(f"step {i}: {len(line)} components for tokens {toks}")
            for (y0, x0, y1, x1), lab in zip(line, labs):
                g = np.zeros((SIZE, SIZE), np.float16)
                g[:y1 - y0, :x1 - x0] = ink[y0:y1, x0:x1]
                if len(found.setdefault(lab, [])) < PER_LABEL:
                    found[lab].append(g)
    labs = sorted(found)
    print({k: len(found[k]) for k in labs})
    np.savez_compressed(ROOT / "framehunt" / "assets" / "mathanim_glyphs.npz",
                        labels=np.array([k for k in labs for _ in found[k]]),
                        glyphs=np.stack([g for k in labs for g in found[k]]))


if __name__ == "__main__":
    main()
