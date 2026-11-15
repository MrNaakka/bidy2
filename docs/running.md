# Running the evaluation

## Requirements

- Docker
- [uv](https://docs.astral.sh/uv/)
- Harbor 0.25.0: `uv tool install 'harbor==0.25.0'`
- `ANTHROPIC_API_KEY` in the environment or in `.env`, for the agent runs

## The one command

```bash
scripts/eval.sh
```

It builds the 30 EVAL tasks and runs these stages through Harbor. A fresh checkout
creates its own secret salt on the first build.

| Stage | Harbor job (`runs/<job>`) | What it is |
| --- | --- | --- |
| build | - | `scripts/build_tasks.py --split eval` writes `tasks/eval/<item>/` |
| reference | `fh-eval-reference` | Harbor's `oracle` agent runs `solution/solve.sh`: the scripted policy, MCP tools only |
| do-nothing | `fh-eval-nop` | Harbor's `nop` agent; never submits |
| stop-early | `fh-eval-stop-early` | `framehunt/baselines/stop_early.py`: `info`, one low-res look, a well-formed guess |
| agents | `fh-eval-opencode-run1..3` | `harbor run -a opencode -m anthropic/claude-opus-5-5`, three runs |
| summary | - | `scripts/summarize.py` writes `results/summary.md` and `results/summary.json` |

On local Docker at the default `-n 4` this takes several hours: each baseline stage
takes roughly 5-25 min and each Opus run about 35-50 min (wall times of our earlier
30- and 40-item runs in `results/stage_times.txt`; the 30-item Opus runs took 35-49 min). The three Opus runs cost about $13 of API credits; `--runs 1` does a
single agent run (~$4.4).

### On Modal

Modal runs every job remotely, so the local machine does almost no work and the run is
much faster:

```bash
uv tool install --force 'harbor[modal]==0.25.0'
export MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=...
scripts/eval.sh --env modal -n 30
```

### Parts and options

```bash
scripts/eval.sh --stage baselines   # build + reference + nop + stop-early (no API spend)
scripts/eval.sh --stage agents      # only the agent runs (uses API credits)
scripts/eval.sh --stage summarize   # rebuild the summary from runs/
scripts/eval.sh --runs 1 -n 8       # one agent run, 8 concurrent trials
```

## Fresh-clone check

On 2026-11-14 we cloned the repository and ran
`scripts/eval.sh --env modal -n 40 --runs 1`. The clone made a new salt, so the items
were new instances. At that time EVAL still had 10 toasts items (since removed), so the
run covered 40 items. The whole pipeline took 14 min: build 5 s, reference 139 s,
do-nothing 132 s, stop-early 327 s, one Opus run 241 s, then the summary. API cost:
$5.72 for the 40 items.

Rescored on the 30 current items (`scripts/summarize.py --split eval`):

| Fresh clone | Score | Per kind (logstream / dialogs / scandoc) | Mean pixels |
| --- | --- | --- | --- |
| Reference | 1.000 (30/30) | 1.0 / 1.0 / 1.0 | 53k |
| Do-nothing | 0.000 | | 0 |
| Stop-early | 0.000 | | 32k |
| Opus 5.5, one run | 0.600 (18/30) | 0.5 / 0.6 / 0.7 | 112k |

- One stop-early trial hit a Modal Docker-in-Docker build error before the agent
  started. It scored 0, like every stop-early trial.
- We also smoke-tested the default local-Docker path from the same clone at `-n 2` on
  3 items, 2 of which are in the current EVAL (logstream-7001, scandoc-7601). On those
  2: reference 2/2, do-nothing 0, stop-early 0, Opus 1/2.
- A single Opus run varies: our three runs ranged from 0.50 to 0.70.

## Example run logs

[examples/harbor-runs/](../examples/harbor-runs/) holds full Harbor job outputs with scores
(`result.json`, per-trial `verifier/reward.txt`, the MCP server's call log, the agent's
trajectory):

- [`fresh-clone-modal-opus-run1/`](../examples/harbor-runs/fresh-clone-modal-opus-run1/):
  the Opus run from the fresh-clone check above (throwaway salt; 40 items including
  the 10 toasts items since removed), 0.600 (18/30) on the current 30 items.
- [`fh-eval-opencode-run1/`](../examples/harbor-runs/fh-eval-opencode-run1/): Opus 5.5 run 1
  on the **EVAL v1** set (30 items, retired salt), 0.700 (21/30).

Neither salt is the one behind our reported results, so publishing their answers is safe.

## Other models and providers

`--model <provider/model>` is passed to `harbor run -a opencode -m <provider/model>`,
with OpenCode pinned to 1.19.5. The script forwards only `ANTHROPIC_API_KEY` to the
agent. For another hosted provider, run Harbor directly with an env file that holds the
provider's key under the name OpenCode expects:

```bash
harbor run -p tasks/eval -a opencode -m <provider/model> --ak version=1.19.5 \
  --env-file <file with the provider key> --job-name fh-eval-<name>-run1 -o runs -n 4 -y
```

Qwen3.8-27B, base or with our adapter, runs on our own vLLM endpoints, one Verda H100
per run. See [infra/verda/README.md](../infra/verda/README.md); it needs Verda keys.

```bash
scripts/eval.sh --stage agents --model qwen --fleet                            # base, 3 runs in parallel
scripts/eval.sh --stage agents --model qwen-fh --lora train/out/<run> --fleet  # with a LoRA adapter
```

## Harness details

- Each item is a Harbor task: `tasks/eval/<item>/` holds `instruction.md`, `task.toml`,
  `environment/docker-compose.yaml`, `solution/` and `tests/`.
- Docker Compose runs two containers: the agent's `main` container and the `fh-mcp`
  sidecar, which serves MCP at `http://fh-mcp:8000/mcp`. The agent container has public
  network access for its model API. The sidecar holds the seed; the agent never sees it.
- The agent timeout is 1800 s (`[agent] timeout_sec`). Only agent-install timeouts are
  retried, because they happen before the agent sees anything. Any later failure counts
  as a failure.
- The verifier runs separately (`[verifier] environment_mode = "separate"`) and writes
  0 or 1 to `/logs/verifier/reward.txt`.
- Jobs run on local Docker (the default) or on Modal (`--env modal`, Docker-in-Docker
  inside a Modal sandbox).

## Reproducibility

- **New instances per checkout.** A fresh checkout makes a fresh salt, so judges get new
  instances of the same 30 item ids. Generators and distributions (kinds, K, event
  times, layouts) are the same. Expect the same overall picture as our numbers, not the
  same per-item outcomes.
- **The reference is deterministic** for a given salt. It scored 30/30 on our salt and
  on the fresh clone's salt. The tests also run it on 100-200 random seeds each of
  toasts, leaderboard and scandoc.
- **Which salt each run used.** The reference, the baselines, Qwen base, all SFT and GRPO runs
  and Opus's 10 scandoc trials used the current salt. Opus's 20 video trials ran under
  the previous EVAL v2 salt. We rotated it after a summary that included answers was
  pushed by mistake; summaries no longer contain answers. Same generators and items,
  new instances.
- **Infrastructure failures.** Base Qwen reruns kept 14 trials that had finished
  validly before an infrastructure network failure. Trials lost to infrastructure
  (Docker network pool exhaustion, a relay outage) were rerun and never counted.
- **Serving speed.** With the 1800 s timeout, serving speed matters for Qwen. Base and
  fine-tuned models used identical serving.
- **Logs.** Every Harbor job is logged under `runs/<job>/`: config, per-trial result,
  the server's call log, the agent's trajectory. `scripts/summarize.py` rebuilds the
  tables from those folders.
