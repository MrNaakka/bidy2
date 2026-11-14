"""Token counts per SFT conversation with the real Qwen3.8 processor (no weights needed).

    train/.venv/bin/python train/token_stats.py --data train/data/sft-v1 \
        [--processor train/hf/Qwen3.8-27B] [--limit N] [--show 1]

Renders every row exactly as the trainer does (train/fhdata.py: Qwen's chat template
with OpenCode's 17 tools, images expanded by the image processor) and reports total,
trained (assistant) and image tokens. Writes <data>/token_stats.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fhdata

DEFAULT_PROCESSOR = Path(__file__).resolve().parent / "hf" / "Qwen3.8-27B"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--processor", default=str(DEFAULT_PROCESSOR))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show", type=int, default=0, help="print the rendered text of the first N rows")
    a = ap.parse_args(argv)
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(a.processor)
    rows = fhdata.read_jsonl(a.data / "sft.jsonl")[: a.limit]
    tools = json.loads((a.data / "tools.json").read_text())
    per = []
    for i, r in enumerate(rows):
        e = fhdata.encode(r, a.data, proc, tools)
        per.append({"id": r["id"], "kind": r["kind"], "tokens": e["n_tokens"], "train_tokens": e["n_train_tokens"],
                    "image_tokens": e["n_image_tokens"], "images": e["n_images"], "turns": e["n_assistant_turns"]})
        if i < a.show:
            msgs, _ = fhdata.to_template(r, a.data)
            print(proc.apply_chat_template(msgs, tools=tools, tokenize=False))
            ids = e["input_ids"][0].tolist()
            lab = e["labels"][0].tolist()
            print("---- trained segments ----")
            seg = []
            for t, l in zip(ids, lab):
                if l != fhdata.IGNORE:
                    seg.append(t)
                elif seg:
                    print(repr(proc.tokenizer.decode(seg)))
                    seg = []
    out = {"rows": len(per)}
    for key in ("tokens", "train_tokens", "image_tokens", "images", "turns"):
        xs = [p[key] for p in per]
        out[key] = {"mean": round(sum(xs) / len(xs), 1), "min": min(xs), "p50": pct(xs, 50), "p95": pct(xs, 95),
                    "max": max(xs), "total": sum(xs)}
    out["by_kind"] = {}
    for kind in ("logstream", "dialogs", "dashboard"):
        xs = [p["tokens"] for p in per if p["kind"] == kind]
        if xs:
            out["by_kind"][kind] = {"mean_tokens": round(sum(xs) / len(xs)), "max_tokens": max(xs)}
    # the shared prefix (system + tools + first user turn) for reference
    r0 = rows[0]
    prefix = proc.apply_chat_template(fhdata.to_template({"messages": r0["messages"][:2]}, a.data)[0], tools=tools,
                                      tokenize=False)
    out["prefix_tokens_system_tools_user"] = len(proc.tokenizer(prefix)["input_ids"])
    (a.data / "token_stats.json").write_text(json.dumps({"summary": out, "per_row": per}, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
