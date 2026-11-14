"""Merge SFT data dirs (teacher data from make_sft.py, rollouts from rollouts_to_sft.py) into one.

    uv run python train/merge_sft.py --out train/data/ei-1-mix train/data/sft-v1 train/data/ei-1

sft_lora.py and `fh train-up` read one data dir; this makes it. Row ids must be unique across the inputs
and every input's tools.json must be identical (all rows are rendered with the same tool list). Images are
hard-linked (copied across filesystems) under the same relative paths, so each row's messages are kept
byte for byte. Each row gets `data_source` (the input dir's name); stats.json counts rows per source and kind.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


def link_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        q = dst / p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
            continue
        if q.exists():
            raise SystemExit(f"{q}: two inputs write the same image path")
        q.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(p, q)
        except OSError:
            shutil.copy2(p, q)


def image_dirs(row: dict) -> set[str]:
    out = set()
    for m in row["messages"]:
        if isinstance(m.get("content"), list):
            for part in m["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:"):
                        continue
                    out.add(str(Path(url).parent))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="data dirs with sft.jsonl, tools.json, images/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    a = ap.parse_args(argv)
    if len({p.resolve() for p in a.inputs}) != len(a.inputs):
        ap.error("an input is given twice")
    out_final = a.out.resolve()
    for p in a.inputs:
        q = p.resolve()
        if q == out_final or out_final in q.parents or q in out_final.parents:
            ap.error(f"--out {a.out} overlaps the input {p}")
    if a.out.exists() and not a.force:
        ap.error(f"{a.out} exists (--force replaces it)")
    tools = None
    rows, ids, img_owner = [], set(), {}
    stats = {"inputs": [str(p) for p in a.inputs], "by_source": {}}
    for src in a.inputs:
        t = (src / "tools.json").read_bytes()
        if tools is None:
            tools = t
        elif t != tools:
            raise SystemExit(f"{src}/tools.json differs from {a.inputs[0]}/tools.json")
        with open(src / "sft.jsonl") as f:
            src_rows = [json.loads(l) for l in f if l.strip()]
        by_kind = collections.Counter()
        for r in src_rows:
            if r["id"] in ids:
                raise SystemExit(f"row id {r['id']} appears in more than one input")
            ids.add(r["id"])
            for d in image_dirs(r):
                if img_owner.setdefault(d, src) != src:
                    raise SystemExit(f"{d}: used by rows of {img_owner[d]} and {src}")
            rows.append({**r, "data_source": src.name})
            by_kind[r["kind"]] += 1
        stats["by_source"][src.name] = {"rows": len(src_rows), "by_kind": dict(by_kind)}
    # build next to --out and swap it in at the end (all checks above ran before anything was written)
    final_out = a.out
    final_out.parent.mkdir(parents=True, exist_ok=True)
    a.out = Path(tempfile.mkdtemp(prefix=f".{final_out.name}.", suffix=".tmp", dir=final_out.parent))
    (a.out / "tools.json").write_bytes(tools)
    (a.out / "images").mkdir()
    for src in a.inputs:
        if (src / "images").is_dir():
            link_tree(src / "images", a.out / "images")
    with open(a.out / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats["rows"] = len(rows)
    stats["by_kind"] = dict(collections.Counter(r["kind"] for r in rows))
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    if final_out.exists():
        shutil.rmtree(final_out)
    a.out.rename(final_out)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
