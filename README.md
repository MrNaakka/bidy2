# FrameHunt

An RL environment where every pixel the agent looks at costs budget. Our entry to
AfterQuery's "Break the Agent" challenge (Junction 2026).

The agent gets one question about a recording: an E2E test video, a bug-report screen
recording, a terminal recording of a failed deploy, or a long scanned report with no
text layer. **It can't watch the video.** Through MCP tools it requests single frames
(or pages), optionally cropped and downscaled. Every returned pixel counts against a hard
budget of **120,000 pixels per item**, about 6% of one 1920x1080 frame.

The answer is random and needs a sharp, near-native crop to read. A vision model can't
read it at 1/4 scale or below. *When* and *where* to look can be found cheaply, because
a lasting trace (a counter, a badge, page numbers) is legible at 1/8 or 1/4 scale. Good
play: search the trace for *when*, verify it, then spend on one sharp crop of *where*.
Being slightly off gives a confident wrong answer, because a look-alike sits one frame
early, on the facing page or in the next row.

**Reward:** 1 for an exact answer on the first submission, else 0.

| Tool | Does | Cost |
| --- | --- | --- |
| `info()` | question, answer format, media facts, budget left, cost rules | free |
| `view(t, x, y, w, h, out_width)` | one frame, optionally cropped and downscaled | output width x height |
| `contact_sheet(t_start, t_end, n, ...)` | grid of n evenly spaced frames, with a frame list | n x tile pixels |
| `submit(answer)` | first submission counts; no feedback | free |

## The three kinds (EVAL: 30 items, 10 each)

- **logstream**: a terminal log scrolls 3-5 lines per second; give the request id of the K-th ERROR line.
- **dialogs**: an error dialog pops up and closes in a desktop app; give its code.
- **scandoc**: a 150-400 page scanned annual report; find one amount in one note's table.

## Results

All runs go through Harbor on the same 30 EVAL items.

| Agent | Run 1 | Run 2 | Run 3 | Mean | Mean pixels / item |
| --- | --- | --- | --- | --- | --- |
| Reference (scripted, same MCP tools) | 1.000 | (deterministic) | | **1.000** | 53k (max 58k) |
| **Opus 5.5** (OpenCode 1.19.5) | 0.700 | 0.500 | 0.600 | **0.600** | 113k |
| Qwen3.8-27B base | 0.200 | 0.233 | 0.200 | **0.211** | 94k |
| Qwen3.8-27B + our SFT v3 (LoRA) | 0.533 | 0.467 | 0.567 | **0.522** | 70-84k |
| Qwen3.8-27B + SFT v3 + our GRPO (RL, sync it2) | 0.700 | 0.633 | 0.633 | **0.656** | 79-85k |
| Qwen3.8-27B + SFT v3 + our GRPO (RL, async run 2) | 0.733 | 0.600 | 0.633 | **0.656** | 67-83k |
| Stop-early baseline (one look, well-formed guess) | 0.000 | | | **0.000** | 33k |
| Do-nothing baseline | 0.000 | | | **0.000** | 0 |

- Every Opus run is below the reference, by 9 to 15 items out of 30. The three runs
  cost $12.92 in API fees (about $0.14 per item).
- Opus is weakest on logstream (0.43) and scandoc (0.50); on dialogs it scores 0.87.
- Opus spends almost the whole budget whether it is right or wrong (108.9k on correct
  items, 118.9k on wrong ones). The reference needs 53k.
- Training Qwen3.8-27B on our environment raised it from 0.21 (base) to 0.52 with SFT
  on teacher trajectories, then to 0.66 with GRPO on the environment's own reward. Two GRPO
  adapters reach 0.656: the sync loop after 2 iterations, and an async run that continued
  from it with 10 updates of 12 groups. Every run of either beats every SFT v3 run. GRPO's
  gain is on dialogs (12 correct of 30 after SFT; 20 sync, 19 async run 2) and scandoc
  (8; 13 sync, 17 async run 2). Logstream goes from 27 to 26 (sync) and 23 (async run 2).
- The GRPO models' mean is above Opus's 0.600, but the run ranges overlap and Opus's
  video items ran on different instances of the same items (an earlier salt). It
  reaches Opus's level; it does not clearly beat it.
  Details: [docs/training.md](docs/training.md).

![Overall results](docs/results/overall.png)

## Run the evaluation

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Harbor 0.25.0
(`uv tool install 'harbor==0.25.0'`), and `ANTHROPIC_API_KEY` in the environment or in
`.env`.

```bash
scripts/eval.sh
```

This builds the 30 tasks (with a new secret salt on a fresh checkout), then runs the
reference, the do-nothing and stop-early baselines and three Opus runs, and writes
`results/summary.md`. On local Docker it takes several hours. The three Opus runs cost about
$13 of API credits. On Modal it is much faster:

```bash
uv tool install --force 'harbor[modal]==0.25.0'
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
scripts/eval.sh --env modal -n 30
```

We ran this on a fresh clone with one Opus run (`--runs 1`), before the toasts items
were removed from EVAL, so it covered 40 items: 14 min and $5.72. Rescored on the 30
current items: reference 30/30, both baselines 0, Opus 0.600. Details, options, other models and
harness notes: [docs/running.md](docs/running.md).

## More

| Doc | What's in it |
| --- | --- |
| [docs/why-its-hard.md](docs/why-its-hard.md) | each kind's trap; analysis of all 36 Opus failures |
| [docs/results/tables.md](docs/results/tables.md) | full tables, per kind, pixels, EVAL v1 history; charts and `results.json` in [docs/results/](docs/results/) |
| [docs/scoring-and-integrity.md](docs/scoring-and-integrity.md) | the reward, the separate verifier, secret instances, baselines |
| [docs/training.md](docs/training.md) | training Qwen3.8-27B (tiebreaker): SFT, then GRPO; method, results, caveats |
| [docs/running.md](docs/running.md) | the one command in detail, Modal, other models, harness, reproducibility |
| [docs/design.md](docs/design.md) | full spec: item rules, generators, interfaces, budget choice |
| [examples/harbor-runs/](examples/harbor-runs/) | full Harbor run logs with scores |
| [docs/framing.md](docs/framing.md) | pitch framing for the demo |

## Repository map

| Path | What |
| --- | --- |
| `framehunt/scenes/` | scene generators: `logstream`, `dialogs`, `scandoc` (EVAL); `dashboard`, `toasts`, `leaderboard` (DEV/TRAIN); `mathanim` (DEV, a ManimCE animation rendered at build time) |
| `framehunt/server.py` | the MCP server: tools, budget, evidence routes |
| `framehunt/reference.py` | the scripted reference policy |
| `framehunt/baselines/` | the stop-early baseline agent |
| `framehunt/splits.py` | EVAL, EVAL_V1, DEV, TRAIN item lists; seed derivation |
| `framehunt/verify.py`, `evidence.py` | the verifier and the sidecar's evidence dump |
| `scripts/eval.sh` | the one command; also `build_tasks.py`, `summarize.py` |
| `results/` | our run summary (`summary.md`, `summary.json`, `stage_times.txt`) |
| `docs/results/` | charts, `tables.md`, `results.json`, `make_results.py` |
| `train/` | SFT data and LoRA training; GRPO losses and rollout conversion ([train/README.md](train/README.md)) |
| `infra/verda/` | Qwen serving, training and parallel evals on Verda GPUs; `grpo/`, `grpo_async/` RL loops |
| `tests/` | `uv run pytest -m "not reference"`; `uv run pytest -m reference -s` runs the reference on all 30 EVAL items and on random seeds |
