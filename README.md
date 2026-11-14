# FrameHunt

FrameHunt is an RL environment for **active video sampling under a pixel budget**. The
agent gets a question about a long synthetic screen recording (a scrolling service log,
a desktop app with pop-up error dialogs, or a monitoring dashboard). It can't watch the
video. Through an MCP server it can only ask for single frames, optionally cropped and
downscaled, and every returned pixel is charged against a budget of 500,000 pixels
(about a quarter of one full 1920x1080 frame).

Each question is about a brief event that leaves a lasting trace on screen: a counter,
a badge or a tile that stays red. The trace is readable at 1/8 scale. The answer (a
request id, an error code, a KPI value) is random and only readable at native
resolution. Good play: binary-search the cheap trace to find *when*, then spend pixels
on one sharp crop of *where*. Reward is 1 for an exact answer on the first submission,
else 0. Design details: [docs/design.md](docs/design.md).

## Run the evaluation

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Harbor (`uv tool install harbor`),
and `ANTHROPIC_API_KEY` in the environment or in `.env` for the agent runs.

```bash
scripts/eval.sh
```

This builds the 30 EVAL tasks and runs, through Harbor, in this order:

| Stage | Harbor job (`runs/<job>`) | What it is |
| --- | --- | --- |
| build | - | `scripts/build_tasks.py --split eval` writes `tasks/eval/<item>/` |
| reference | `fh-eval-reference` | Harbor's `oracle` agent runs `solution/solve.sh`, the scripted policy in `framehunt/reference.py`. It uses only the agent's MCP tools: one 1/8 overview, a binary search on the trace, one native crop, template-matching OCR with the bundled fonts. |
| do-nothing | `fh-eval-nop` | Harbor's `nop` agent. Never submits. |
| stop-early | `fh-eval-stop-early` | `framehunt/baselines/stop_early.py`: `info`, one low-res look, then a well-formed guess (seeded by item id). |
| agents | `fh-eval-opencode-run1..3` | OpenCode with `anthropic/claude-opus-5-5`, three runs. |
| summary | - | `scripts/summarize.py` writes `results/summary.md` and `results/summary.json` |

Run parts separately:

```bash
scripts/eval.sh --stage baselines   # build + reference + nop + stop-early, no API spend
scripts/eval.sh --stage agents      # only the 3 agent runs (uses API credits)
scripts/eval.sh --stage summarize   # rebuild the summary from runs/
```

Options: `-n 4` concurrent trials (default 4), `--runs 3`, `--model ...`.

## Results

- `results/summary.md`: per run mean reward, correct/submitted counts, pixels used,
  tool calls, API cost and wall time; a per-kind breakdown; and a per-item table.
- `results/summary.json`: the same data, per trial.
- `results/stage_times.txt`: wall time of each stage.
- `runs/<job>/<trial>/`: Harbor's raw output. `verifier/result.json` holds the
  server's record (submitted, answer, correct, pixels used); `verifier/calls.jsonl`
  logs every tool call; `agent/` holds the agent's logs (for the reference,
  `oracle.txt` and `reference_trace.json`, its tool-call trace).

## Tests

```bash
uv run pytest -m "not reference"     # unit and server tests
uv run pytest -m reference -s        # the reference policy on all 30 EVAL items, in-process, no Docker
```

## Layout

- `framehunt/scenes/`: scene generators (`logstream`, `dialogs`, `dashboard`)
- `framehunt/server.py`: the MCP server (the agent's tools and the budget)
- `framehunt/reference.py`: the reference policy
- `framehunt/baselines/`: the stop-early baseline agent
- `framehunt/splits.py`: the fixed EVAL and DEV item lists
- `scripts/build_tasks.py`, `scripts/eval.sh`, `scripts/summarize.py`
