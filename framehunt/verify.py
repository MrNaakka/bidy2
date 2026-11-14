"""FrameHunt verifier: compute the reward from the sidecar's evidence. Stdlib only.

Runs in Harbor's *separate* verifier container (built from the task's tests/
directory), never in the agent's container:

    python3 /tests/verify.py [EVIDENCE_DIR] [OUT_DIR]     # defaults: /fh-evidence /logs/verifier

EVIDENCE_DIR holds ``result.json`` and ``calls.jsonl`` as dumped by the fh-mcp sidecar
(``framehunt.evidence``) and collected by Harbor from that sidecar. The reward is 1
only if the sidecar's result says the first submission was correct
(``submitted is True and correct is True``). Anything else, including a missing or
malformed result, scores 0. The evidence is copied into OUT_DIR for analysis.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

DEFAULT_EVIDENCE = "/fh-evidence"
DEFAULT_OUT = "/logs/verifier"


def reward_from_result(result: object) -> int:
    if not isinstance(result, dict):
        return 0
    return 1 if result.get("submitted") is True and result.get("correct") is True else 0


def verify(evidence_dir: str | Path = DEFAULT_EVIDENCE, out_dir: str | Path = DEFAULT_OUT) -> int:
    ev, out = Path(evidence_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("reward.txt", "reward.json", "result.json", "calls.jsonl"):
        (out / name).unlink(missing_ok=True)

    reward = 0
    try:
        src = ev / "result.json"
        if src.is_symlink() or not src.is_file():
            raise FileNotFoundError(f"{src} is missing")
        raw = src.read_bytes()
        result = json.loads(raw)
        reward = reward_from_result(result)
        (out / "result.json").write_bytes(raw)
        print(f"result: {raw.decode('utf-8', 'replace').strip()}")
    except Exception as e:  # noqa: BLE001 - any problem with the evidence scores 0
        print(f"ERROR: no usable result in {ev}: {e!r}")
    try:
        src = ev / "calls.jsonl"
        if src.is_file() and not src.is_symlink():
            shutil.copyfile(src, out / "calls.jsonl")
            print(f"calls logged: {sum(1 for _ in (out / 'calls.jsonl').open('rb'))}")
        else:
            print(f"ERROR: {src} is missing")
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: could not copy the call log: {e!r}")

    (out / "reward.txt").write_text(f"{reward}\n")
    print(f"reward={reward}")
    return reward


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    verify(*(argv[:2] or []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
