# FrameHunt

FrameHunt is an RL environment for **active video sampling under a pixel budget**. The
agent gets a question about a long synthetic screen recording (a scrolling service log,
a desktop app with stacking toast notifications, or an API error tracker whose endpoint
table re-sorts every half second). It can't watch the video. Through an MCP server it can only ask for single frames, optionally cropped and
downscaled, and every returned pixel is charged against a hard budget of 120,000 pixels
per item (about 6% of one full 1920x1080 frame, which is 2,073,600 pixels). A request
that would exceed the budget is refused.

Each question is about a brief event that leaves a lasting trace on screen: a counter
or a badge. The trace is readable at 1/8 scale. The answer (a request id, a toast code,
an errors/min value) is random and only readable at native resolution. Being slightly
off is punished with a confident wrong answer: one frame early, a look-alike (the
previous ERROR line, the previous Error toast, the endpoint's previous value) sits where
the answer will be; a little late, the answer has moved or changed. Good play: binary-search the cheap trace to find *when*, then spend pixels
on one sharp crop of *where*. Reward is 1 for an exact answer on the first submission,
else 0. Design details: [docs/design.md](docs/design.md).

## Results

**EVAL v2** (current, `framehunt/splits.py` `EVAL`): 10 logstream, 10 toasts and 10
leaderboard items. Final v2 numbers come later. The tables below are **v1** results on
the first EVAL set (`EVAL_V1`: 10 each of logstream, dialogs and dashboard; rebuild it
with `scripts/build_tasks.py --split eval --v1`). The logstream items are the same in
both sets. Dialogs and dashboard were too easy for Opus (8–10/10), so v2 replaces them
with `toasts` and `leaderboard`, which carry logstream's failure mechanism into new
settings (see "Second slice" in [docs/design.md](docs/design.md)).

v1: 30 EVAL_V1 items (10 per kind), 120k-pixel budget, run through Harbor.

| Run | Mean reward | Correct |
| --- | --- | --- |
| Reference (scripted, same MCP tools) | 1.000 | 30/30 |
| Do-nothing baseline | 0.000 | 0/30 |
| Stop-early baseline | 0.000 | 0/30 |
| Opus 5.5 (OpenCode), run 1 | 0.700 | 21/30 |
| Opus 5.5 (OpenCode), run 2 | 0.700 | 21/30 |
| Opus 5.5 (OpenCode), run 3 | in progress | |

Per kind (v1):

| Kind | Opus run 1 | Opus run 2 | Reference |
| --- | --- | --- | --- |
| logstream | 3/10 | 3/10 | 10/10 |
| dialogs | 9/10 | 8/10 | 10/10 |
| dashboard | 9/10 | 10/10 | 10/10 |

Pixels used: the reference averages 53.4k and never goes above 58.6k. Opus averages
102k (run 1) and 106k (run 2) on items it gets right. Every one of its 18 failures ended
with 113.5k or more spent, and 17 of them with less than 2.3k left.

A full Harbor run log of Opus run 1, with its score, is in
[examples/harbor-runs/](examples/harbor-runs/).

## Why it's hard

(Failure analysis of the v1 runs. Toasts and leaderboard were built to test the same
decisions: verify the event frame before reading, and read the moving target on that
same frame.)

The agent has everything it needs. The instruction states the question, the video
facts, the budget and the cost of every call. The trace (counter, badge or red tile) is
visible on every frame and readable at 1/8 scale. Each contact sheet comes with a text
list mapping every tile to its frame. The reference policy (`framehunt/reference.py`)
is a script that uses only the same MCP tools. Beyond the tools, it knows only the screen
layout and fonts, which an agent can see in one cheap low-res look. It scores 30/30.

What the agent has to get right is a short chain of decisions: plan the budget, find
the exact event frame on the cheap trace, check it, decide where to crop, then spend on
one sharp crop. The figures below come from the 18 failures in Opus runs 1 and 2 (9 per
run). 14 of the 18 are logstream items, and 5 logstream items (7001, 7004, 7007, 7008,
7009) failed in both runs. We checked each failure against the item's ground truth and
read the agent's own final message.

1. **Reading the log before the line exists (9 of 14 logstream failures).** The agent
   searched the `errors: N` counter, but then read the log at a frame *before* the K-th
   ERROR line was printed. In 5 cases it was off by only 1 to 4 frames (0.1 to 0.4 s).
   At that frame the counter still shows K-1, so the line isn't on screen yet. In all 9
   cases the agent had already bought a contact sheet of the counter that included that
   exact frame, and the tile showed K-1. It misread the sheet: the tiles were small
   (8 to 18 px tall), and the agent read the wrong tile as the change. It never took one
   direct look at the counter at the frame it was about to read. That look costs a few
   hundred pixels. Instead it spent the rest of the budget scanning a log that didn't
   have the answer. Decision: *verify before spending.*
2. **Right moment, wrong row (5 of 14 logstream failures).** The agent read the log
   3 to 16 frames after the line appeared. New lines arrive 3 to 4 times per second, so
   by then the ERROR line was no longer the newest. The agent assumed the bottom line
   was the ERROR line, or never got the request id at a legible scale. In 3 of the 5 it
   submitted the id of an INFO or DEBUG line. In the other 2 it read part of the id and
   guessed the rest. The ERROR level label is red and visible at low resolution. Finding
   it first and then cropping that one row is cheap. Decision: *where to crop.*
3. **Budget spent on broad looks.** Before its first look at the log, the agent had
   already spent a median of 62k pixels on logstream failures (35k to 91k). That is
   more than the reference spends on the whole item. In 14 of 60 trials the first call
   was a full frame at the default `out_width=320`, which costs 57,600 pixels, 48% of
   the budget. Several agents say this in their final message ("My opening full-frame
   overview cost 57,600 of the 120,000 pixels, nearly half"). Decision: *budget
   planning, coarse-to-fine.*
4. **Other kinds (4 failures).** In 3 dialogs failures the agent read the code at 1/2
   to 1/3 scale and misread it, or its crops missed the code field inside the dialog.
   In 1 dashboard failure it found the right frame and read the neighbouring KPI row
   (`p95 latency` 931 instead of `Payments / min` 78090).

In 17 of the 18 failures, the agent's final message says it is guessing. It notices
the problem only after the budget is gone.

## Run the evaluation

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Harbor (`uv tool install harbor`),
and `ANTHROPIC_API_KEY` in the environment or in `.env` for the agent runs.

```bash
scripts/eval.sh
```

This builds the 30 EVAL (v2) tasks and runs, through Harbor, in this order:

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

### Other models

`--model <provider/model>` is passed to `harbor run -a opencode -m <provider/model>`
(OpenCode pinned to 1.19.5). The default is `anthropic/claude-opus-5-5`, for example:

```bash
scripts/eval.sh --stage agents --model anthropic/claude-opus-5-5
```

The script reads the API key from the environment or `.env` and hands Harbor only that
one key. Right now it forwards only `ANTHROPIC_API_KEY` for every non-Qwen model. To use
another provider, run Harbor directly with an env file that holds that provider's key
under the name OpenCode expects:

```bash
harbor run -p tasks/eval -a opencode -m <provider/model> --ak version=1.19.5 \
  --env-file <file with the provider key> --job-name fh-eval-<name>-run1 -o runs -n 4 -y
```

The agent container has public network access, so any hosted provider works. Jobs with
any non-Qwen model are named `fh-eval-opencode-runN`, so a new run replaces the earlier
directory.

### Harness details

- Each task is a Harbor task in `tasks/eval/<item>/`, built by `scripts/build_tasks.py`.
- The MCP server runs as a docker-compose sidecar service, `fh-mcp`, next to the agent's
  `main` container (`environment/docker-compose.yaml`). OpenCode reaches it at
  `http://fh-mcp:8000/mcp`. The agent's container has public network access.
- The agent timeout is 1800 s (`[agent] timeout_sec` in each `task.toml`).
- The verifier runs in a separate container (`[verifier] environment_mode =
  "separate"`, no network). After the agent finishes, a collect hook in the sidecar
  dumps the server's result and call log. Harbor copies them into the verifier
  container, which writes the 0/1 score to `/logs/verifier/reward.txt`. Nothing the
  agent's container writes is read.
- Tested with Harbor on local Docker.
- Item seeds are derived from a secret per-checkout salt (`secrets/eval.salt`, created
  on the first build). A fresh checkout gets new items of the same kinds and difficulty.
  See "Item secrecy" in [docs/design.md](docs/design.md).

## Output files

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
uv run pytest -m reference -s        # the reference policy on all 30 EVAL items (in-process HTTP, no Docker)
                                     # and on 200 random seeds per new kind
```

## Layout

- `framehunt/scenes/`: scene generators (`logstream`, `toasts`, `leaderboard` in EVAL v2;
  `dialogs`, `dashboard` in EVAL v1, DEV and TRAIN)
- `framehunt/server.py`: the MCP server (the agent's tools and the budget)
- `framehunt/reference.py`: the reference policy
- `framehunt/baselines/`: the stop-early baseline agent
- `framehunt/splits.py`: the fixed EVAL (v2), EVAL_V1 and DEV item lists
- `scripts/build_tasks.py`, `scripts/eval.sh`, `scripts/summarize.py`
