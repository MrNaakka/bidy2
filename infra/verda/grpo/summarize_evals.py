"""Per run and per kind mean reward of Harbor eval jobs (verifier/reward.txt; missing = 0).

    python3 summarize_evals.py /root/fh/runs fh-eval-qwen-fh3 fh-eval-qwen-grpo
"""

import json
import sys
from pathlib import Path


def run_stats(d: Path) -> dict:
    by_kind: dict[str, list[float]] = {}
    for t in sorted(d.iterdir()):
        if not (t / "config.json").is_file() or "__" not in t.name:
            continue
        item = t.name.split("__")[0]
        rf = t / "verifier" / "reward.txt"
        r = float(rf.read_text().strip() or 0) if rf.is_file() else 0.0
        by_kind.setdefault(item.rsplit("-", 1)[0], []).append(r)
    allr = [x for v in by_kind.values() for x in v]
    return {"n": len(allr), "mean": round(sum(allr) / max(len(allr), 1), 4),
            **{k: f"{sum(v):g}/{len(v)}" for k, v in sorted(by_kind.items())}}


def main() -> None:
    root = Path(sys.argv[1])
    for prefix in sys.argv[2:]:
        runs = sorted(root.glob(f"{prefix}-run*"))
        means = []
        for r in runs:
            s = run_stats(r)
            means.append(s["mean"])
            print(json.dumps({"job": r.name, **s}))
        if means:
            print(f"{prefix}: mean over {len(means)} runs {sum(means) / len(means):.4f} (runs {means})")


if __name__ == "__main__":
    main()
