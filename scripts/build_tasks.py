"""Write one Harbor task directory per FrameHunt item.

    uv run python scripts/build_tasks.py --split eval [--out tasks/eval] [--budget 500000] [--only ID ...]

Layout per item (follows step0/probe):
    task.toml                      MCP server `framehunt` at http://fh-mcp:8000/mcp
    instruction.md                 question, answer format, video facts, budget, cost rules
    environment/Dockerfile         the agent's (main) container
    environment/docker-compose.yaml  the fh-mcp sidecar with FH_KIND, FH_SEED, FH_BUDGET_PX, VERIFY_TOKEN
    environment/mcp/               sidecar Dockerfile + a copy of framehunt/
    tests/test.sh                  reads /_result and /_log, writes /logs/verifier/{reward.txt,result.json,calls.jsonl}
    solution/solve.sh              oracle (currently a plumbing check, see TODO inside)

VERIFY_TOKEN is random per task and per build (secrets.token_hex). It appears only in
the compose file and tests/, which the agent's container never sees. Rebuilding gives
new tokens; that does not change any item (scenes depend only on kind and seed).
"""

from __future__ import annotations

import argparse
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framehunt.scenes import make_scene  # noqa: E402
from framehunt.server import DEFAULT_BUDGET_PX, MAX_SHEET_N, MIN_OUT_WIDTH, MIN_SHEET_N, SHEET_SEPARATOR_PX  # noqa: E402
from framehunt.splits import SPLITS  # noqa: E402

# Pinned to uv.lock (Python 3.12 resolution).
SIDECAR_PIP = ["mcp==1.30.0", "pillow==13.0.0", "numpy==2.5.4"]
MCP_HOST = "fh-mcp"
AGENT_TIMEOUT_SEC = 1800.0
VERIFIER_TIMEOUT_SEC = 120.0

TASK_TOML = """\
schema_version = "1.4"

[task]
name = "framehunt/{item_id}"
version = "0.1.0"
description = "FrameHunt {kind} item: answer a question about a screen recording under a pixel budget."

[metadata]
split = "{split}"
kind = "{kind}"
seed = {seed}
budget_px = {budget}

[verifier]
timeout_sec = {verifier_timeout}
environment_mode = "shared"

[agent]
timeout_sec = {agent_timeout}

[environment]
network_mode = "public"
build_timeout_sec = 900.0

[[environment.mcp_servers]]
name = "framehunt"
transport = "streamable-http"
url = "http://{host}:8000/mcp"
"""

INSTRUCTION_MD = """\
# FrameHunt

Answer a question about a screen recording. You can't watch the video. You can only
request individual frames from the MCP server `framehunt`. Frames can be cropped and
downscaled, and every image pixel the server returns is charged against a fixed pixel
budget.

## Question

{question}

Answer format: {answer_format}

## Video

- Size: {width} x {height} pixels
- Frame rate: {fps} fps
- Length: {n_frames} frames, {duration:g} seconds. Frame `i` is shown at time `i / {fps}` s.

## Pixel budget

You have {budget} pixels in total. For scale, one full frame at native resolution is
{full} pixels.

## Tools and costs

- `info()`: shows the question, answer format, video facts, `budget_left` and these
  rules. Free.
- `view(t, x=0, y=0, w=None, h=None, out_width=320)`: returns one frame as a PNG image
  and one line of text with the actual frame time, frame index, output size, cost and
  `budget_left`.
  - `t` is in seconds, from 0 to {duration:g}, and is snapped to the nearest frame.
  - `x, y, w, h` give a crop rectangle in native pixels. The default is the whole
    frame. The crop is clipped to the frame.
  - The crop is resized to `out_width` pixels wide, keeping its aspect ratio.
    `out_width` must be at least {min_out}. It is capped at the crop width, so images
    are never upscaled.
  - **Cost = output width x output height.**
- `contact_sheet(t_start, t_end, n, x=0, y=0, w=None, h=None, out_width=160)`: returns
  one grid image of `n` evenly spaced frames from `t_start` to `t_end` (both included),
  plus a text list of the frame times. Tiles are read left to right, then top to bottom.
  - `n` is from {min_n} to {max_n}. The same crop and `out_width` rules as `view`
    apply to each tile.
  - **Cost = n x tile output width x tile output height.** The {sep}-pixel separators
    between tiles are free.
- `submit(answer)`: records your answer. Free. **Only the first submission counts.**
  Later submissions are refused. No feedback on correctness is given.

A request that costs more than your remaining budget is refused and charges nothing.
Invalid requests are refused and charge nothing.

## Scoring

The score is 1 if your submitted answer matches the correct answer exactly, ignoring
case and whitespace. Otherwise it is 0. Not submitting scores 0. The budget is a hard
limit. Pixels you don't use don't change the score.
"""

MAIN_DOCKERFILE = """\
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates python3 \\
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
"""

SIDECAR_DOCKERFILE = """\
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir {pins}
WORKDIR /srv
COPY framehunt /srv/framehunt
EXPOSE 8000
CMD ["python", "-m", "framehunt.server"]
"""

COMPOSE = """\
services:
  main:
    depends_on:
      {host}:
        condition: service_healthy
  {host}:
    build:
      context: ./mcp
    environment:
      FH_KIND: "{kind}"
      FH_SEED: "{seed}"
      FH_BUDGET_PX: "{budget}"
      VERIFY_TOKEN: "{token}"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/_health', timeout=2)"]
      interval: 2s
      timeout: 5s
      retries: 60
"""

TEST_SH = """\
#!/bin/bash
# FrameHunt verifier. Reads the result and call log from the fh-mcp sidecar
# (token-protected routes) and writes the reward. Reward = 1 only if the first
# submitted answer matched exactly; anything else (no submit, server unreachable) is 0.
set -u
TOKEN="{token}"
BASE="http://{host}:8000"
OUT=/logs/verifier
mkdir -p "$OUT"

# Kill every process left over from the agent phase (except PID 1 and this script's
# own ancestors), so nothing can rewrite the reward files after we write them.
# Processes in PID 1's process group (the container's keep-alive, e.g. `sleep infinity`)
# are kept too: killing them would stop the container. Agent processes can't join that
# group, because it belongs to another session.
stat_of() {{  # prints "PPID PGRP"; the name field may contain spaces or ')'
  local line
  read -r line 2>/dev/null < "/proc/$1/stat" || return 1
  line=${{line##*) }}
  set -- $line
  echo "$2 $3"
}}
keep=" 1 $$ "
p=$$
while [ "$p" -gt 1 ] 2>/dev/null; do
  st=$(stat_of "$p") || break
  pp=${{st% *}}
  keep="$keep$pp "
  p=$pp
done
for d in /proc/[0-9]*; do
  pid=${{d#/proc/}}
  case "$keep" in *" $pid "*) continue ;; esac
  st=$(stat_of "$pid") || continue
  [ "${{st% *}}" = "$$" ] && continue
  [ "${{st#* }}" = "1" ] && continue
  kill -9 "$pid" 2>/dev/null
done
rm -f "$OUT/reward.json" "$OUT/reward.txt"

REWARD=0
if curl -sf --max-time 60 "$BASE/_result?token=$TOKEN" -o "$OUT/result.json"; then
  echo "result: $(cat "$OUT/result.json")"
  if grep -q '"reward":1.0' "$OUT/result.json" && grep -q '"correct":true' "$OUT/result.json"; then
    REWARD=1
  fi
else
  echo "ERROR: could not read $BASE/_result"
fi
curl -sf --max-time 60 "$BASE/_log?token=$TOKEN" -o "$OUT/calls.jsonl" \\
  && echo "calls logged: $(wc -l < "$OUT/calls.jsonl")" \\
  || echo "ERROR: could not read $BASE/_log"
echo "$REWARD" > "$OUT/reward.txt"
echo "reward=$REWARD"
"""

SOLVE_SH = """\
#!/bin/bash
# TODO(reference): replace with the scripted coarse-to-fine reference policy
# (`python -m framehunt.reference`), which must use only the agent's MCP tools.
# The agent's container doesn't ship the framehunt package, so that step will have to
# bring it in here.
#
# For now this is a plumbing check: a stdlib MCP client calls info, view and submit
# with a dummy answer. Reward 0 is expected.
set -e
python3 "$(dirname "$0")/mcp_smoke.py" "http://{host}:8000/mcp"
"""


def _ignore(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo"))]


def build_item(out: Path, split: str, item_id: str, kind: str, seed: int, budget: int) -> Path:
    scene = make_scene(kind, seed)
    m = scene.meta
    token = secrets.token_hex(24)
    d = out / item_id
    if d.exists():
        shutil.rmtree(d)
    (d / "environment" / "mcp").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution").mkdir()

    fmt = dict(item_id=item_id, kind=kind, seed=seed, split=split, budget=budget, token=token, host=MCP_HOST,
               agent_timeout=AGENT_TIMEOUT_SEC, verifier_timeout=VERIFIER_TIMEOUT_SEC)
    (d / "task.toml").write_text(TASK_TOML.format(**fmt))
    (d / "instruction.md").write_text(INSTRUCTION_MD.format(
        question=scene.question, answer_format=scene.answer_format, width=m.width, height=m.height, fps=m.fps,
        n_frames=m.n_frames, duration=m.duration_s, budget=budget, full=m.width * m.height,
        min_out=MIN_OUT_WIDTH, min_n=MIN_SHEET_N, max_n=MAX_SHEET_N, sep=SHEET_SEPARATOR_PX))
    (d / "environment" / "Dockerfile").write_text(MAIN_DOCKERFILE)
    (d / "environment" / "docker-compose.yaml").write_text(COMPOSE.format(**fmt))
    (d / "environment" / "mcp" / "Dockerfile").write_text(SIDECAR_DOCKERFILE.format(pins=" ".join(SIDECAR_PIP)))
    shutil.copytree(ROOT / "framehunt", d / "environment" / "mcp" / "framehunt", ignore=_ignore)
    test_sh = d / "tests" / "test.sh"
    test_sh.write_text(TEST_SH.format(**fmt))
    solve_sh = d / "solution" / "solve.sh"
    solve_sh.write_text(SOLVE_SH.format(**fmt))
    shutil.copy(ROOT / "scripts" / "mcp_smoke.py", d / "solution" / "mcp_smoke.py")
    for p in (test_sh, solve_sh):
        p.chmod(0o755)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--out", type=Path, default=None, help="default: tasks/<split>")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET_PX)
    ap.add_argument("--only", nargs="*", default=None, help="build only these item ids")
    a = ap.parse_args(argv)
    if a.budget <= 0:
        ap.error("--budget must be positive")
    out = a.out or ROOT / "tasks" / a.split
    out.mkdir(parents=True, exist_ok=True)
    items = SPLITS[a.split]
    if a.only:
        unknown = set(a.only) - {i for i, _, _ in items}
        if unknown:
            ap.error(f"unknown item ids for split {a.split}: {sorted(unknown)}")
        items = [it for it in items if it[0] in a.only]
    for item_id, kind, seed in items:
        d = build_item(out, a.split, item_id, kind, seed, a.budget)
        print(d.relative_to(ROOT) if d.is_relative_to(ROOT) else d)
    print(f"built {len(items)} task(s) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
