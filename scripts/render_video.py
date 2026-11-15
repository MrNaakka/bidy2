#!/usr/bin/env python
"""Render a scene to an MP4 file, frame by frame, at its native frame rate.

Use a non-EVAL seed: the video shows the answer, so an EVAL item would leak it.

Usage:
    uv run --with imageio-ffmpeg python scripts/render_video.py logstream 7 --start 85 --length 15 --out docs/media/logstream-seed7-15s.mp4
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import imageio_ffmpeg

from framehunt.scenes import KINDS, make_scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("seed", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed-up (keeps every frame)")
    ap.add_argument("--start", type=float, default=0.0, help="clip start, seconds into the scene")
    ap.add_argument("--length", type=float, default=None, help="clip length in scene seconds (default: to the end)")
    ap.add_argument("--crf", type=int, default=23)
    args = ap.parse_args()

    sc = make_scene(args.kind, args.seed)
    m = sc.meta
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{m.width}x{m.height}",
           "-r", str(m.fps * args.speed), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(args.crf),
           "-preset", "medium", "-movflags", "+faststart", str(args.out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    f0 = round(args.start * m.fps)
    f1 = m.n_frames if args.length is None else min(m.n_frames, f0 + round(args.length * m.fps))
    for f in range(f0, f1):
        p.stdin.write(sc.render(f).convert("RGB").tobytes())
    p.stdin.close()
    if p.wait():
        sys.exit("ffmpeg failed")
    print(f"{args.out}: {m.width}x{m.height}, frames {f0}-{f1 - 1} of {m.n_frames}, "
          f"{(f1 - f0) / (m.fps * args.speed):.1f} s at {m.fps * args.speed:g} fps")
    print(f"question: {sc.question}\nanswer: {sc.answer}")


if __name__ == "__main__":
    main()
