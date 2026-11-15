"""Render a `mathanim` plan (JSON on stdin) to a lossless MP4 with ManimCE.

    python framehunt/scenes/mathanim_render.py OUT.mp4 < plan.json

Build time only (needs manim and LaTeX). Frames are captured one by one from manim's
Camera so the timeline is exact: a settled frame is rendered once and repeated, a
transform frame is TransformMatchingTex at alpha = j / transform_frames.
"""
from __future__ import annotations

import json
import sys
import tempfile
from fractions import Fraction

import av
import numpy as np
from manim import BOLD, WHITE, YELLOW, Camera, MathTex, Rectangle, Text, TransformMatchingTex, config

BG = "#101318"


def render(plan: dict, out: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _render(plan, out, tmp)


def _render(plan: dict, out: str, tmp: str) -> None:
    w, h, fps, n, tf = plan["width"], plan["height"], plan["fps"], plan["n_frames"], plan["transform_frames"]
    config.pixel_width, config.pixel_height, config.frame_rate = w, h, fps
    config.background_color = BG
    config.media_dir = tmp            # LaTeX scratch files
    unit = w / config.frame_width                     # pixels per manim unit

    def point(x, y):
        return np.array([(x - w / 2) / unit, (h / 2 - y) / unit, 0.0])

    def equation(step):
        m = MathTex("f(x)", "=", r"\frac{", *step["num"], "}{", *step["den"], "}", font_size=plan["font_size"],
                    color=WHITE)
        return m.shift(point(*plan["eq_anchor"]) - m[1].get_center())

    def counter(i):
        t = Text(f"step {i}", font="DejaVu Sans", weight=BOLD, font_size=72, color="#c9ced6")
        right, top = plan["counter_right_top"]
        return t.shift(point(right, top) - t.get_corner([1, 1, 0]))

    bx, by, bw, bh = plan["box"]
    box = Rectangle(width=bw / unit, height=bh / unit, color=YELLOW, stroke_width=8,
                    fill_color=YELLOW, fill_opacity=0.14).move_to(point(bx + bw / 2, by + bh / 2))

    camera = Camera()
    container = av.open(out, "w")
    stream = container.add_stream("libx264rgb", rate=fps, options={"qp": "0", "preset": "veryfast", "g": "20"})
    stream.width, stream.height, stream.pix_fmt = w, h, "rgb24"
    stream.time_base = Fraction(1, fps)
    written = 0

    def emit(mobs, count=1):
        nonlocal written
        camera.reset()
        camera.capture_mobjects(mobs)
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(camera.get_image().convert("RGB")), format="rgb24")
        for _ in range(count):
            frame.pts = written
            container.mux(stream.encode(frame))
            written += 1

    def boxed(mobs, frame):
        return [box, *mobs] if frame >= plan["box_frame"] else mobs

    steps = plan["steps"]
    eqs = [equation(s) for s in steps]
    for i, step in enumerate(steps):
        start, end = step["start"], steps[i + 1]["start"] - tf if i + 1 < len(steps) else n
        still = [eqs[i], counter(i)]
        if start < plan["box_frame"] < end:
            emit(still, plan["box_frame"] - start)
            start = plan["box_frame"]
        emit(boxed(still, start), end - start)
        if i + 1 < len(steps):
            anim = TransformMatchingTex(eqs[i].copy(), eqs[i + 1])
            anim.begin()
            for j in range(tf):
                anim.interpolate(j / tf)
                emit(boxed([anim.mobject, still[1]], end))
            anim.finish()                     # leaves eqs[i + 1] in its final state
    container.mux(stream.encode())
    container.close()
    if written != n:
        raise SystemExit(f"wrote {written} frames, plan has {n}")


if __name__ == "__main__":
    render(json.load(sys.stdin), sys.argv[1])
