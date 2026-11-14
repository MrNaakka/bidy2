#!/usr/bin/env python
"""Eyeball a scene: writes a contact sheet, a few full-res frames and an answer crop.

Usage:
    uv run python scripts/preview_scene.py logstream 7 --out /tmp/preview
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from framehunt.scenes import KINDS, make_scene


def contact_sheet(scene, frames, tile_w=384, cols=4) -> Image.Image:
    tile_h = tile_w * scene.meta.height // scene.meta.width
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tile_w + 4) + 4, rows * (tile_h + 22) + 4), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for n, f in enumerate(frames):
        x = 4 + (n % cols) * (tile_w + 4)
        y = 4 + (n // cols) * (tile_h + 22)
        sheet.paste(scene.render(f).resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y))
        d.text((x + 2, y + tile_h + 3), f"f={f}  t={f / scene.meta.fps:.1f}s", fill=(230, 230, 230))
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=KINDS)
    ap.add_argument("seed", type=int)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=16, help="contact sheet tiles")
    args = ap.parse_args()

    sc = make_scene(args.kind, args.seed)
    info = sc.debug_info()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.kind}_{args.seed}"
    nf = sc.meta.n_frames
    frames = [round(i * (nf - 1) / (args.n - 1)) for i in range(args.n)]
    contact_sheet(sc, frames).save(out / f"{stem}_sheet.png")

    ev = info["event_frame"]
    for name, f in [("first", 0), ("before_event", ev - 1), ("event", ev), ("last", nf - 1)]:
        im = sc.render(f)
        im.save(out / f"{stem}_{name}_f{f}.png")
        if name == "event":
            im.resize((480, 270), Image.Resampling.LANCZOS).save(out / f"{stem}_event_quarter.png")
            im.resize((240, 135), Image.Resampling.LANCZOS).save(out / f"{stem}_event_eighth.png")
            x, y = info["answer_text"]["xy"]
            im.crop((max(0, x - 200), max(0, y - 60), x + 300, y + 80)).save(out / f"{stem}_answer_crop.png")
    meta = {"question": sc.question, "answer": sc.answer, "answer_format": sc.answer_format,
            "n_frames": nf, "duration_s": sc.meta.duration_s, "debug": info}
    (out / f"{stem}_info.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("question", "answer", "n_frames")}, indent=2))
    print(f"wrote previews to {out}")


if __name__ == "__main__":
    main()
