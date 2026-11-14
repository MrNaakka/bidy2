"""Write one Harbor task directory per FrameHunt item.

    uv run python scripts/build_tasks.py --split eval [--out tasks/eval] [--budget 120000] [--only ID ...]
    uv run python scripts/build_tasks.py --split eval --v1     # the first EVAL set, into tasks/eval-v1

Layout per item (follows step0/probe):
    task.toml                      MCP server `framehunt` at http://fh-mcp:8000/mcp
    instruction.md                 question, answer format, video facts, budget, cost rules
    environment/Dockerfile         the agent's (main) container
    environment/docker-compose.yaml  the fh-mcp sidecar with FH_KIND, FH_SEED, FH_BUDGET_PX, VERIFY_TOKEN
    environment/mcp/               sidecar Dockerfile + a copy of framehunt/
    tests/                         the separate verifier image: Dockerfile, test.sh, verify.py
    solution/                      oracle: solve.sh runs framehunt/reference.py (copied with the fonts)

Verification never runs in the agent's container. task.toml sets
`[verifier] environment_mode = "separate"`. After the agent phase Harbor stops the
agent's container, runs a `[[verifier.collect]]` hook in the fh-mcp sidecar
(`python -m framehunt.evidence /fh-evidence`, which reads the token routes of the
running server), collects /fh-evidence from the sidecar as an artifact, and runs
tests/test.sh in a fresh container built from tests/Dockerfile. That script
(framehunt/verify.py) writes reward 1 only if the sidecar's result says the first
submission was correct; a missing dump scores 0.

FH_SEED is the secret scene seed, derive_seed(split, item_id, salt) from
framehunt/splits.py. It appears only in the compose file, which the agent's container
never sees. The salt is read from FH_SALT_<SPLIT> or secrets/<split>.salt; if neither
exists, a random one is generated and saved to secrets/<split>.salt (git-ignored).
Keep that file to rebuild the same items.

VERIFY_TOKEN is random per task and per build (secrets.token_hex). It appears only in
the compose file (the collect hook reads it from the sidecar's own env). Rebuilding
gives new tokens; that does not change any item (scenes depend only on kind and the
derived seed).
"""

from __future__ import annotations

import argparse
import os
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from framehunt.scenes import DOCUMENT_KINDS, make_scene
from framehunt.server import (
    DEFAULT_BUDGET_PX,
    MAX_SHEET_N,
    MIN_OUT_WIDTH,
    MIN_SHEET_N,
    SHEET_SEPARATOR_PX,
)
from framehunt.splits import EVAL_V1, SPLITS, derive_seed, load_salt, salt_path

# Pinned to uv.lock (Python 3.12 resolution).
SIDECAR_PIP = ["mcp==1.30.0", "pillow==13.0.0", "numpy==2.5.4", "stretchable==1.1.8", "attrs==25.4.0"]
MAIN_PIP = ["pillow==13.0.0", "numpy==2.5.4", "mcp==1.30.0"]  # for the reference solution
MCP_HOST = "fh-mcp"
AGENT_TIMEOUT_SEC = 1800.0
VERIFIER_TIMEOUT_SEC = 120.0

TASK_TOML = """\
schema_version = "1.4"

[task]
name = "framehunt/{item_id}"
version = "0.1.0"
description = "FrameHunt {kind} item: answer a question about a screen recording or scanned document under a pixel budget."

[metadata]
split = "{split}"
item_id = "{item_id}"
kind = "{kind}"
budget_px = {budget}

# Verification runs in its own container (built from tests/Dockerfile), never in the
# agent's. Harbor stops the agent's container, then the collect hook dumps the
# sidecar's result and call log to /fh-evidence, which is collected from the sidecar
# and re-materialized at /fh-evidence in the verifier container.
[verifier]
timeout_sec = {verifier_timeout}
environment_mode = "separate"

[verifier.environment]
build_timeout_sec = 600.0
network_mode = "no-network"

[[verifier.collect]]
service = "{host}"
command = "python -m framehunt.evidence {evidence_dir}"
timeout_sec = 60.0

[[artifacts]]
source = "{evidence_dir}"
service = "{host}"

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

INSTRUCTION_DOC_MD = """\
# FrameHunt

Answer a question about a long scanned document. You can't open the file. You can only
request individual page images from the MCP server `framehunt`. Pages can be cropped
and downscaled, and every image pixel the server returns is charged against a fixed
pixel budget.

## Question

{question}

Answer format: {answer_format}

## Document

- A scanned PDF of {n_frames} pages: page images only, there is no text layer.
- Each page image is {width} x {height} pixels (A4 at 150 dpi).
- Pages are addressed by their 0-based PDF page index `t`, from 0 to {last}:
  `t=0` is the first page of the PDF (the cover), `t={last}` the last one. The MCP
  tools call it `t` because the same server also serves videos; here `t` is simply the
  page index (it is rounded to the nearest integer).
- The page numbers printed on the pages and in the document's contents are the
  document's own numbering. They are not PDF page indexes.

## Pixel budget

You have {budget} pixels in total. For scale, one full page at native resolution is
{full} pixels, so reading everything is far out of reach.

## Tools and costs

- `info()`: shows the question, answer format, document facts, `budget_left` and these
  rules. Free.
- `view(t, x=0, y=0, w=None, h=None, out_width=320)`: returns one page as a PNG image
  and one line of text with the PDF page index, the crop, output size, cost and
  `budget_left`.
  - `t` is the PDF page index, from 0 to {last}.
  - `x, y, w, h` give a crop rectangle in native pixels. The default is the whole
    page. The crop is clipped to the page.
  - The crop is resized to `out_width` pixels wide, keeping its aspect ratio.
    `out_width` must be at least {min_out}. It is capped at the crop width, so images
    are never upscaled.
  - **Cost = output width x output height.**
- `contact_sheet(t_start, t_end, n, x=0, y=0, w=None, h=None, out_width=160)`: returns
  one grid image of `n` evenly spaced pages from page index `t_start` to `t_end` (both
  included), plus a text list of their page indexes. Tiles are read left to right, then
  top to bottom.
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
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates python3 python3-pip \\
    && rm -rf /var/lib/apt/lists/*
# Pinned like the sidecar. The reference solution (oracle) needs them; agents may use them too.
RUN pip3 install --no-cache-dir --break-system-packages {pins}
# Pre-install what Harbor's OpenCode agent installs at setup (nvm + Node 22 + opencode-ai, the
# version eval.sh pins). Harbor still runs its install step, but it then finds everything in
# place instead of downloading it in every trial. This only speeds up setup; the agent's
# environment is the same. Agents without Node are unaffected.
ENV NVM_DIR=/root/.nvm
RUN curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | env -u NODE_VERSION bash \
    && . "$NVM_DIR/nvm.sh" && nvm install 22 && nvm alias default 22 \
    && npm i -g opencode-ai@1.19.5 && opencode --version \
    && ln -sf "$(command -v node)" /usr/local/bin/node && ln -sf "$(command -v npm)" /usr/local/bin/npm
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

EVIDENCE_DIR = "/fh-evidence"

# The separate verifier's image. Built from tests/ (the build context), so it holds
# only test.sh and verify.py (stdlib only), nothing from the agent's container.
TESTS_DOCKERFILE = """\
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1
COPY test.sh verify.py /tests/
RUN chmod 0755 /tests/test.sh
"""

TEST_SH = """\
#!/bin/sh
# FrameHunt verifier. Runs in Harbor's separate verifier container, never in the
# agent's. Reads the evidence the fh-mcp sidecar dumped ({evidence_dir}/result.json,
# calls.jsonl), writes /logs/verifier/reward.txt and copies the evidence there.
# Reward 1 only if the sidecar says the first submission was correct; else 0.
exec python3 /tests/verify.py {evidence_dir} /logs/verifier
"""

SOLVE_SH = """\
#!/bin/bash
# Reference run (oracle): the scripted coarse-to-fine policy in framehunt/reference.py.
# It uses only the agent's MCP tools (info, view, submit) over the network, exactly
# like an agent. Its tool-call trace goes to stdout (oracle.txt) and /logs/agent.
set -euo pipefail
cd "$(dirname "$0")"
python3 -c "import PIL, numpy, mcp" 2>/dev/null \\
  || pip3 install --no-cache-dir --break-system-packages {pins} \\
  || pip install --no-cache-dir {pins}
mkdir -p /logs/agent
exec python3 -m framehunt.reference --url "http://{host}:8000/mcp" --trace /logs/agent/reference_trace.json
"""


def write_solution(d: Path, fmt: dict) -> Path:
    """solution/: solve.sh plus a minimal framehunt package (reference.py + fonts only)."""
    sol = d / "solution"
    pkg = sol / "framehunt"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    shutil.copy(ROOT / "framehunt" / "reference.py", pkg / "reference.py")
    shutil.copytree(ROOT / "framehunt" / "assets" / "fonts", pkg / "assets" / "fonts")
    solve_sh = sol / "solve.sh"
    solve_sh.write_text(SOLVE_SH.format(pins=" ".join(MAIN_PIP), **fmt))
    return solve_sh


def _ignore(_dir, names):
    return [n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo"))]


def build_item(out: Path, split: str, item_id: str, kind: str, seed: int, budget: int,
               salt: str | None = None) -> Path:
    """`seed` is the item's public seed (part of its id); the scene uses the secret
    derive_seed(split, item_id, salt). salt=None loads (or creates) the split's salt."""
    del seed  # public; never used to render
    scene_seed = derive_seed(split, item_id, salt)
    scene = make_scene(kind, scene_seed)
    m = scene.meta
    token = secrets.token_hex(24)
    d = out / item_id
    if d.exists():
        shutil.rmtree(d)
    (d / "environment" / "mcp").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution").mkdir()

    fmt = {"item_id": item_id, "kind": kind, "seed": scene_seed, "evidence_dir": EVIDENCE_DIR, "split": split, "budget": budget, "token": token, "host": MCP_HOST,
               "agent_timeout": AGENT_TIMEOUT_SEC, "verifier_timeout": VERIFIER_TIMEOUT_SEC}
    (d / "task.toml").write_text(TASK_TOML.format(**fmt))
    template = INSTRUCTION_DOC_MD if kind in DOCUMENT_KINDS else INSTRUCTION_MD
    (d / "instruction.md").write_text(template.format(
        question=scene.question, answer_format=scene.answer_format, width=m.width, height=m.height, fps=m.fps,
        n_frames=m.n_frames, last=m.n_frames - 1, duration=m.duration_s, budget=budget, full=m.width * m.height,
        min_out=MIN_OUT_WIDTH, min_n=MIN_SHEET_N, max_n=MAX_SHEET_N, sep=SHEET_SEPARATOR_PX))
    (d / "environment" / "Dockerfile").write_text(MAIN_DOCKERFILE.format(pins=" ".join(MAIN_PIP)))
    (d / "environment" / "docker-compose.yaml").write_text(COMPOSE.format(**fmt))
    (d / "environment" / "mcp" / "Dockerfile").write_text(SIDECAR_DOCKERFILE.format(pins=" ".join(SIDECAR_PIP)))
    shutil.copytree(ROOT / "framehunt", d / "environment" / "mcp" / "framehunt", ignore=_ignore)
    (d / "tests" / "Dockerfile").write_text(TESTS_DOCKERFILE)
    shutil.copy(ROOT / "framehunt" / "verify.py", d / "tests" / "verify.py")
    test_sh = d / "tests" / "test.sh"
    test_sh.write_text(TEST_SH.format(**fmt))
    solve_sh = write_solution(d, fmt)
    for p in (test_sh, solve_sh):
        p.chmod(0o755)
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLITS))
    ap.add_argument("--out", type=Path, default=None, help="default: tasks/<split>")
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET_PX)
    ap.add_argument("--only", nargs="*", default=None,
                    help="build only these item ids (otherwise the full list is built and other task dirs are removed)")
    ap.add_argument("--v1", action="store_true",
                    help="with --split eval: build the first evaluation set (EVAL_V1: logstream, dialogs, "
                         "dashboard), whose results the README reports as v1. Default out: tasks/eval-v1")
    a = ap.parse_args(argv)
    if a.v1 and a.split != "eval":
        ap.error("--v1 needs --split eval")
    if a.budget <= 0:
        ap.error("--budget must be positive")
    out = a.out or ROOT / "tasks" / ("eval-v1" if a.v1 else a.split)
    out.mkdir(parents=True, exist_ok=True)
    items = EVAL_V1 if a.v1 else SPLITS[a.split]
    if a.only:
        unknown = set(a.only) - {i for i, _, _ in items}
        if unknown:
            ap.error(f"unknown item ids for split {a.split}: {sorted(unknown)}")
        items = [it for it in items if it[0] in a.only]
    salt = load_salt(a.split)
    v1_salt = salt_path("eval-v1")
    if a.v1 and not os.environ.get("FH_SALT_EVAL") and v1_salt.is_file():
        # EVAL v1 was run (and its logs published) under the salt that was in use then,
        # kept as secrets/eval-v1.salt when EVAL v2 got a fresh one.
        salt = v1_salt.read_text().strip()
    if salt is None:
        salt = load_salt(a.split, create=True)
        print(f"generated a new secret salt for split {a.split!r}: {salt_path(a.split)} "
              "(keep it to rebuild these exact items; never commit it)")
    if not a.only:
        # A full build owns the directory: drop generated tasks that are not in this item
        # list (e.g. EVAL v1 items left in tasks/eval), so a run never evaluates stale items.
        keep = {i for i, _, _ in items}
        for d in sorted(out.iterdir()):
            if d.is_dir() and d.name not in keep and (d / "task.toml").is_file():
                shutil.rmtree(d)
                print(f"removed stale task {d.name} (not in this item list)")
    for item_id, kind, seed in items:
        d = build_item(out, a.split, item_id, kind, seed, a.budget, salt=salt)
        print(d.relative_to(ROOT) if d.is_relative_to(ROOT) else d)
    print(f"built {len(items)} task(s) in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
