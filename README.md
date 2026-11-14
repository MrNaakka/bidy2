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

## The four kinds (EVAL: 40 items, 10 each)

- **logstream**: a terminal log scrolls 3-5 lines per second; give the request id of the K-th ERROR line.
- **dialogs**: an error dialog pops up and closes in a desktop app; give its code.
- **toasts** (control): same trap as logstream, but the check is cheap and sits right next to the answer.
- **scandoc**: a 150-400 page scanned annual report; find one amount in one note's table.

## Results

All runs go through Harbor on the same 40 EVAL items.

| Agent | Run 1 | Run 2 | Run 3 | Mean | Mean pixels / item |
| --- | --- | --- | --- | --- | --- |
| Reference (scripted, same MCP tools) | 1.000 | (deterministic) | | **1.000** | 50k (max 58k) |
| **Opus 5.5** (OpenCode 1.19.5) | 0.750 | 0.550 | 0.650 | **0.650** | 110k |
| Qwen3.8-27B base | 0.200 | 0.275 | 0.275 | **0.250** | 100k |
| Qwen3.8-27B + our SFT v3 (LoRA) | 0.650 | 0.550 | 0.650 | **0.617** | 62-74k |
| Stop-early baseline (one look, well-formed guess) | 0.000 | | | **0.000** | 33k |
| Do-nothing baseline | 0.000 | | | **0.000** | 0 |

- Every Opus run is below the reference, by 10 to 18 items out of 40. The three runs
  cost $16.45 in API fees (about $0.14 per item).
- Opus is weakest on logstream (0.43) and scandoc (0.50). On the toasts control it
  scores 0.80.
- Opus spends almost the whole budget whether it is right or wrong (105.6k on correct
  items, 116.8k on wrong ones). The reference needs 50k.
- Fine-tuning Qwen3.8-27B on our environment's trajectories raised it from 0.25 to
  0.62. On logstream it beats Opus, 0.90 vs 0.43.

![Overall results](docs/results/overall.png)

## Run the evaluation

Requirements: Docker, [uv](https://docs.astral.sh/uv/), Harbor 0.25.0
(`uv tool install 'harbor==0.25.0'`), and `ANTHROPIC_API_KEY` in the environment or in
`.env`.

```bash
scripts/eval.sh
```

This builds the 40 tasks (with a new secret salt on a fresh checkout), then runs the
reference, the do-nothing and stop-early baselines and three Opus runs, and writes
`results/summary.md`. On local Docker it takes several hours and about $17 of API
credits. On Modal it is much faster:

```bash
uv tool install --force 'harbor[modal]==0.25.0'
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
scripts/eval.sh --env modal -n 40
```

We ran this on a fresh clone with one Opus run (`--runs 1`). It took 14 min and $5.72.
Reference 40/40, both baselines 0, Opus 0.700. Details, options, other models and
harness notes: [docs/running.md](docs/running.md).

## More

| Doc | What's in it |
| --- | --- |
| [docs/why-its-hard.md](docs/why-its-hard.md) | each kind's trap; analysis of all 42 Opus failures |
| [docs/results/tables.md](docs/results/tables.md) | full tables, per kind, pixels, EVAL v1 history; charts and `results.json` in [docs/results/](docs/results/) |
| [docs/scoring-and-integrity.md](docs/scoring-and-integrity.md) | the reward, the separate verifier, secret instances, baselines |
| [docs/training.md](docs/training.md) | fine-tuning Qwen3.8-27B (tiebreaker): data, training, results |
| [docs/running.md](docs/running.md) | the one command in detail, Modal, other models, harness, reproducibility |
| [docs/design.md](docs/design.md) | full spec: item rules, generators, interfaces, budget choice |
| [examples/harbor-runs/](examples/harbor-runs/) | full Harbor run logs with scores |
| [docs/framing.md](docs/framing.md) | pitch framing for the demo |

## Repository map

| Path | What |
| --- | --- |
| `framehunt/scenes/` | scene generators: `logstream`, `dialogs`, `toasts`, `scandoc` (EVAL); `dashboard`, `leaderboard` (DEV/TRAIN) |
| `framehunt/server.py` | the MCP server: tools, budget, evidence routes |
| `framehunt/reference.py` | the scripted reference policy |
| `framehunt/baselines/` | the stop-early baseline agent |
| `framehunt/splits.py` | EVAL, EVAL_V1, DEV, TRAIN item lists; seed derivation |
| `framehunt/verify.py`, `evidence.py` | the verifier and the sidecar's evidence dump |
| `scripts/eval.sh` | the one command; also `build_tasks.py`, `summarize.py` |
| `results/` | our run summary (`summary.md`, `summary.json`, `stage_times.txt`) |
| `docs/results/` | charts, `tables.md`, `results.json`, `make_results.py` |
| `train/` | SFT data generation and LoRA training ([train/README.md](train/README.md)) |
| `infra/verda/` | Qwen serving, training and parallel evals on Verda GPUs |
| `tests/` | `uv run pytest -m "not reference"`; `uv run pytest -m reference -s` runs the reference on all 40 EVAL items and on random seeds |
