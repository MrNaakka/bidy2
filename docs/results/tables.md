# FrameHunt results (EVAL v2)

- **Set:** 40 items, 10 each of logstream, dialogs, toasts and scandoc.
- **Budget:** 120,000 pixels per item.
- **Reward:** 1 only for an exact first submission, else 0.
- **SFT v2** (marked *) was evaluated on the 30 non-scandoc items only.
- **Charts:** [overall.png](overall.png) (score per agent, dots = runs), [per_kind.png](per_kind.png), [pixels_and_submissions.png](pixels_and_submissions.png), [sft_v3_loss.png](sft_v3_loss.png).
- **Data:** [results.json](results.json), built from `results/summary.json` by `make_results.py`. Regenerate with `uv run --with matplotlib python docs/results/make_results.py`.

## Overall

| Agent | Runs | Mean | Correct per run | Submitted per run | Timeouts | Mean pixels per run |
| --- | --- | --- | --- | --- | --- | --- |
| Reference (scripted) | 1.000 | **1.000** | 40 (of 40) | 40 | 0 | 50k |
| Opus 5.5 (OpenCode) | 0.750 / 0.550 / 0.650 | **0.650** | 30 / 22 / 26 (of 40) | 40 / 40 / 40 | 0 | 108k / 110k / 110k |
| Qwen3.8-27B + SFT v3 | 0.650 / 0.550 / 0.650 | **0.617** | 26 / 22 / 26 (of 40) | 40 / 39 / 40 | 1 | 62k / 74k / 64k |
| Qwen3.8-27B + SFT v2* | 0.700 / 0.500 / 0.633 | **0.611** | 21 / 15 / 19 (of 30) | 30 / 30 / 30 | 0 | 56k / 63k / 62k |
| Qwen3.8-27B base | 0.200 / 0.275 / 0.275 | **0.250** | 8 / 11 / 11 (of 40) | 18 / 25 / 23 | 19 | 101k / 97k / 102k |
| Stop-early baseline | 0.000 | **0.000** | 0 (of 40) | 40 | 0 | 33k |
| Do-nothing baseline | 0.000 | **0.000** | 0 (of 40) | 0 | 0 | 0k |

## Per kind (mean reward over runs; mean pixels)

| Agent | logstream | dialogs | toasts | scandoc |
| --- | --- | --- | --- | --- |
| Reference (scripted) | 1.00 (51k) | 1.00 (58k) | 1.00 (40k) | 1.00 (51k) |
| Opus 5.5 (OpenCode) | 0.43 (116k) | 0.87 (106k) | 0.80 (99k) | 0.50 (117k) |
| Qwen3.8-27B + SFT v3 | 0.90 (64k) | 0.40 (90k) | 0.90 (43k) | 0.27 (69k) |
| Qwen3.8-27B + SFT v2* | 0.77 (65k) | 0.47 (74k) | 0.60 (41k) | - |
| Qwen3.8-27B base | 0.33 (110k) | 0.30 (117k) | 0.37 (116k) | 0.00 (55k) |
| Stop-early baseline | 0.00 (32k) | 0.00 (32k) | 0.00 (32k) | 0.00 (34k) |
| Do-nothing baseline | 0.00 (0k) | 0.00 (0k) | 0.00 (0k) | 0.00 (0k) |

## Pixels on correct vs wrong answers

| Agent | Mean px, correct | Mean px, wrong | Max px |
| --- | --- | --- | --- |
| Reference (scripted) | 50,036 | - | 58,224 |
| Opus 5.5 (OpenCode) | 105,622 | 116,804 | 120,000 |
| Qwen3.8-27B base | 115,764 | 94,515 | 120,000 |
| Qwen3.8-27B + SFT v3 | 55,591 | 84,842 | 120,000 |

Opus API cost: $5.53 + $5.82 + $5.09 = $16.45 for 3 x 40 trials.

## EVAL v1 (history)

The first EVAL set had 30 items (10 each of logstream, dialogs and `dashboard`) and ran under the first salt, now retired. Dashboard was too easy (Opus 29/30 over three runs), so EVAL v2 replaced it with the toasts control and the scandoc document kind. The Qwen base v1 runs had heavier serving contention than the v2 runs. Rebuild v1 with `scripts/build_tasks.py --split eval --v1`.

Correct answers per run, out of 10 per kind:

| Agent | Runs | logstream | dialogs | dashboard |
| --- | --- | --- | --- | --- |
| Reference (scripted) | 1.00 | 10 | 10 | 10 |
| Opus 5.5 (OpenCode) | 0.70 / 0.70 / 0.70 | 3/3/4 | 9/8/7 | 9/10/10 |
| Qwen3.8-27B base | 0.27 / 0.13 / 0.20 | 4/0/0 | 1/1/1 | 3/3/5 |
| Do-nothing baseline | 0.00 | 0 | 0 | 0 |
| Stop-early baseline | 0.00 | 0 | 0 | 0 |
