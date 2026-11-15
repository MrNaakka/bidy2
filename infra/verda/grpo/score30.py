"""Per run reward on the scored EVAL kinds (default: logstream, dialogs, scandoc; toasts excluded).

    python3 score30.py /root/fh/runs fh-eval-qwen-fh3 fh-eval-qwen-grpo [--kinds logstream dialogs scandoc]

A trial without verifier/reward.txt counts 0; a missing trial (item not run) is reported.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--kinds", nargs="+", default=["logstream", "dialogs", "scandoc"])
    a = ap.parse_args()
    for prefix in a.prefixes:
        means = []
        for run in sorted(a.root.glob(f"{prefix}-run[0-9]")):
            by = {k: [] for k in a.kinds}
            pending = 0
            for t in sorted(run.iterdir()):
                if "__" not in t.name or not (t / "config.json").is_file():
                    continue
                kind = t.name.split("__")[0].rsplit("-", 1)[0]
                if kind not in by:
                    continue
                rf = t / "verifier" / "reward.txt"
                pending += not (t / "result.json").is_file()
                by[kind].append(float(rf.read_text().strip() or 0) if rf.is_file() else 0.0)
            n = sum(len(v) for v in by.values())
            m = sum(sum(v) for v in by.values()) / max(n, 1)
            means.append(m)
            print(json.dumps({"job": run.name, "n": n, "pending": pending, "mean": round(m, 4),
                              **{k: f"{sum(v):g}/{len(v)}" for k, v in by.items()}}))
        if means:
            print(f"{prefix}: mean of {len(means)} runs {sum(means) / len(means):.4f}  runs {[round(x, 4) for x in means]}")


if __name__ == "__main__":
    main()
