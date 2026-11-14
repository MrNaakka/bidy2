# FrameHunt results (EVAL v2)

- **Set:** 30 items, 10 each of logstream, dialogs and scandoc.
- **Budget:** 120,000 pixels per item.
- **Reward:** 1 only for an exact first submission, else 0.
- **Charts:** [overall.png](overall.png) (score per agent, dots = runs), [per_kind.png](per_kind.png), [pixels_and_submissions.png](pixels_and_submissions.png), [sft_v3_loss.png](sft_v3_loss.png).
- **Data:** [results.json](results.json), built from `results/summary.json` by `make_results.py`. Regenerate with `uv run --with matplotlib python docs/results/make_results.py`.

## Overall

| Agent | Runs | Mean | Correct per run | Submitted per run | Timeouts | Mean pixels per run |
| --- | --- | --- | --- | --- | --- | --- |
| Reference (scripted) | 1.000 | **1.000** | 30 (of 30) | 30 | 0 | 53k |
| Opus 5.5 (OpenCode) | 0.700 / 0.500 / 0.600 | **0.600** | 21 / 15 / 18 (of 30) | 30 / 30 / 30 | 0 | 112k / 114k / 113k |
| Qwen3.8-27B + SFT v3 | 0.533 / 0.467 / 0.567 | **0.522** | 16 / 14 / 17 (of 30) | 30 / 29 / 30 | 1 | 70k / 84k / 71k |
| Qwen3.8-27B base | 0.200 / 0.233 / 0.200 | **0.211** | 6 / 7 / 6 (of 30) | 13 / 16 / 14 | 15 | 96k / 91k / 96k |
| Stop-early baseline | 0.000 | **0.000** | 0 (of 30) | 30 | 0 | 33k |
| Do-nothing baseline | 0.000 | **0.000** | 0 (of 30) | 0 | 0 | 0k |

## Per kind (mean reward over runs; mean pixels)

| Agent | logstream | dialogs | scandoc |
| --- | --- | --- | --- |
| Reference (scripted) | 1.00 (51k) | 1.00 (58k) | 1.00 (51k) |
| Opus 5.5 (OpenCode) | 0.43 (116k) | 0.87 (106k) | 0.50 (117k) |
| Qwen3.8-27B + SFT v3 | 0.90 (64k) | 0.40 (90k) | 0.27 (69k) |
| Qwen3.8-27B base | 0.33 (110k) | 0.30 (117k) | 0.00 (55k) |
| Stop-early baseline | 0.00 (32k) | 0.00 (32k) | 0.00 (34k) |
| Do-nothing baseline | 0.00 (0k) | 0.00 (0k) | 0.00 (0k) |

## Pixels on correct vs wrong answers

| Agent | Mean px, correct | Mean px, wrong | Max px |
| --- | --- | --- | --- |
| Reference (scripted) | 53,225 | - | 58,224 |
| Opus 5.5 (OpenCode) | 108,926 | 118,927 | 120,000 |
| Qwen3.8-27B base | 115,328 | 88,735 | 120,000 |
| Qwen3.8-27B + SFT v3 | 65,331 | 85,100 | 120,000 |

Opus API cost: $4.38 + $4.64 + $3.90 = $12.92 for 3 x 30 trials.

## EVAL v1 (history)

The first EVAL set had 30 items (10 each of logstream, dialogs and `dashboard`) and ran under the first salt, now retired. Dashboard was too easy (Opus 29/30 over three runs), so EVAL v2 replaced it with the scandoc document kind. The Qwen base v1 runs had heavier serving contention than the v2 runs. Rebuild v1 with `scripts/build_tasks.py --split eval --v1`.

Correct answers per run, out of 10 per kind:

| Agent | Runs | logstream | dialogs | dashboard |
| --- | --- | --- | --- | --- |
| Reference (scripted) | 1.00 | 10 | 10 | 10 |
| Opus 5.5 (OpenCode) | 0.70 / 0.70 / 0.70 | 3/3/4 | 9/8/7 | 9/10/10 |
| Qwen3.8-27B base | 0.27 / 0.13 / 0.20 | 4/0/0 | 1/1/1 | 3/3/5 |
| Do-nothing baseline | 0.00 | 0 | 0 | 0 |
| Stop-early baseline | 0.00 | 0 | 0 | 0 |
