# Verda tooling: Qwen3.8-27B endpoint and LoRA training

We serve Qwen3.8-27B (vision + tool calls) as an OpenAI-compatible vLLM endpoint on a Verda spot GPU, and
train LoRA adapters for it. [SkyPilot](https://github.com/jonibeach/skypilot) (our fork, pinned by commit in
`fh`) launches the boxes. The weights, venvs and runs live on the 300 GB block volume `fh-qwen` in FIN-01.
The fork adds Verda block volumes: it attaches the volume when it creates a box, waits for a deleted box to
let go of it, and mounts it at `/mnt/fh`.

Run every command from the repo root:

```bash
infra/verda/fh <command>
```

`fh` runs `fh.py` with uv in a side environment that holds the fork, so the project's `.venv` stays as it is
(SkyPilot and Harbor need different `uvicorn` versions). SkyPilot reads Verda keys from
`~/.verda/config.json` (`sky check verda` tells you if they work). `fh` writes `QWEN_API_KEY` to `.env`
(made once) and `QWEN_BASE_URL` (when the endpoint is up).

## Files

| file | what it is |
| --- | --- |
| `volume.yaml` | The `fh-qwen` volume (`use_existing: true`). |
| `download.yaml` | CPU box: downloads both checkpoints at pinned revisions, checks every file with `verify_weights.py`, builds the vLLM venv on the volume. |
| `serve.yaml` | GPU box with 80 GB or more, spot by default. vLLM picks BF16 or FP8 from the GPU memory it finds. |
| `train.yaml` | GPU box for `train/sft_lora.py`. |
| `eval.yaml` | One eval run per box: its own weights, vLLM and Harbor (see Parallel evals). |
| `fh.py` | Lifecycle, tunnel, watchdog steps, `test`, `harbor-agent`. |

## Commands

| command | what it does |
| --- | --- |
| `volume` | `sky volumes apply volume.yaml`. Run once per machine, so SkyPilot knows the volume. |
| `download` | Runs `download.yaml` and deletes the box. Skips a checkpoint whose verified revision is already on the volume. |
| `serve-up [--lora PATH] [--on-demand] [--no-wait] [--keep]` | Launches `fh-serve`, opens the tunnel and waits until `/v1/models` lists every served name. If that fails or times out (`--timeout-min`, 40), it deletes the box unless `--keep`. `--lora` also serves the adapter at `PATH` (on the volume) as `qwen3.8-27b-fh`. Refused while a training run is active. |
| `serve-ensure [--wait-min N]` | Watchdog step. Relaunches the box when it is gone (spot loss), when vLLM has exited, or when it is not healthy (through the tunnel) 40 min after launch; after 2 relaunches in a row for a failed launch, a failed job or an unhealthy box, it deletes the box and stops. A launch that finds no capacity leaves no box and is retried on the next tick. Restarts a dead tunnel. After `serve-down` it only deletes an `fh-serve` box that shows up late (e.g. from an interrupted launch). |
| `serve-down` | Deletes the box, stops the tunnel, clears `QWEN_BASE_URL`. |
| `status` | Serve and train state, tunnel, health. |
| `test` | Through the tunnel: a chat request with a PNG, a tool call and a streamed throughput probe. Writes `last_test.json`. |
| `harbor-agent [--served-name qwen3.8-27b\|qwen3.8-27b-fh] [--opencode-version V]` | Prints the Harbor JobConfig fragment `{"agents": [entry]}` (see OpenCode below). Makes no cloud calls, so `uv run python infra/verda/fh.py harbor-agent` also works. |
| `train-up --run NAME --data DIR [--epochs --lr --grad-accum --lora-r --save-every --trainer-arg X] [--max-usd 45] [--on-demand]` | Starts or resumes a run. Refused while serving is on (the volume attaches to one box). |
| `train-ensure` | Watchdog step. Relaunches after spot loss. Counts failed jobs and failed launches that left a box, and gives up after 4. Stops the run when SkyPilot's cost estimate for the run reaches `--max-usd`. When the job succeeds it copies the adapter to `train/out/<run>/` and deletes the box. |
| `train-down` | Stops the run and deletes the box. `train-up` with the same `--run` and arguments resumes from the last checkpoint. |
| `train-status` / `train-fetch [--run NAME]` | Progress (loss, eval, spend) / copy `final/` and the logs from any box that has the volume. |

Watchdogs: `scripts/watchdog.sh serve|train <logfile> <stopfile>` calls `<cmd>-ensure` every 2 minutes.
`fh` serializes `serve-*`, `train-*` through a lock file, and a watchdog step skips its turn while
another command holds it.

## Parallel evals

Each Qwen eval run gets its own box (`eval.yaml`, one SkyPilot cluster per run): the box downloads the pinned
weights, starts vLLM, and runs Harbor against it. Three runs take about as long as one.

```bash
uv run python scripts/build_tasks.py --split eval                 # tasks/eval (secret seeds; never commit)
infra/verda/fh eval-up --tasks tasks/eval --runs 3 -n 10          # fleet eval-qwen: boxes fh-ev-eval-qwen-{1,2,3}
infra/verda/fh eval-wait                                          # fetch each run as it ends, delete its box
infra/verda/fh eval-up --tasks tasks/eval --lora train/out/<run> --runs 3 -n 10   # fleet eval-qwen-fh
```

`scripts/eval.sh --stage agents --model qwen --fleet` (or `--model qwen-fh --lora train/out/<run>`) does both
steps and then summarizes. It runs for 1-2 hours, so start it with `systemd-run --user`.

| command | what it does |
| --- | --- |
| `eval-up --tasks DIR [--lora DIR] [--runs 3] [-n 10] [--only I] [--on-demand] [--max-usd 60] [--max-run-hours 4]` | Stages `.stage/eval-<fleet>/` (tasks with 0700 dirs, Harbor config, `verify_weights.py`, adapter, `meta.json`) and runs `sky launch -d` for every box in parallel (logs in `.stage/`). A failed launch (no capacity) is recorded and retried with `--only I`. Refuses to add a run whose tasks, adapter, instance type, `-n`, model or `eval.yaml` differ from the fleet's first runs (`--new` starts over). |
| `eval-status [--fleet F]` | Per box: cluster and job state, phase (`venv_ready`, `weights_ready`, `vllm_healthy`, `harbor_start`), trials done and rewards written, vLLM running/waiting requests, KV-cache use, preemptions, cost. |
| `eval-wait [--fleet F]` | Every 2 minutes: fetches a box whose job ended and deletes it; reports a box that is gone (spot loss) as `lost`; at the fleet cost cap it fetches what is there and deletes every box. Exits when no box is left (status 1 unless every run is `done`). |
| `eval-fetch [--fleet F] [--only I] [--partial]` | `rsync` of `~/fh/runs/<job>/` to `runs/<job>/`. `--partial` also fetches running jobs. |
| `eval-down [--fleet F] [--only I] [--no-fetch]` | Fetches what is there (unless `--no-fetch`), then deletes the boxes. |
| `fh sky <args>` | The SkyPilot CLI: `fh sky status`, `fh sky queue fh-ev-eval-qwen-1`, `fh sky logs fh-ev-eval-qwen-1`, `fh sky cost-report`. |

Comparability: every box uses the instance type pinned in `eval.yaml` (`1H100.80S.30V`, spot unless
`--on-demand`), the same vLLM flags and the same `-n`; speed matters because of the 30-minute agent timeout.
`runs/<job>/fh-eval-meta.json` records the fleet definition (instance type, spot, `-n`, model and revision,
vLLM/Harbor/OpenCode versions, tasks and adapter hashes, `eval.yaml` hash, git commit), the box's GPUs and
exact `vllm serve` command, phase timings and the cost. The box also leaves `fh-vllm.log` (tail),
`fh-metrics.txt` (vLLM metrics at the end) and `fh-timings.txt` there. A LoRA eval adds only
`--enable-lora --max-lora-rank 16 --lora-modules qwen3.8-27b-fh=<adapter>` to the base eval's flags.

Weights: each box downloads `Qwen/Qwen3.8-27B` at the pinned revision from Hugging Face to its own 250 GB
boot disk and checks every file with `verify_weights.py` (about 10 minutes, in parallel with the vLLM
install). `fh-qwen` is not used: a block volume attaches to one box, and it serves the endpoint.

Spot loss: the run is lost (Harbor has no trial resume). `eval-wait` reports it; `eval-up --tasks ... --only I`
starts it again from scratch on a new box.

Cost backstops: `eval-wait` enforces `--max-usd` for the fleet (SkyPilot's cost estimate, as `sky cost-report`
shows it); the job on the box ends after `--max-run-hours`; a box whose job has ended deletes itself after
60 idle minutes (`autostop` in `eval.yaml`), in case nobody runs `eval-wait`. Results not fetched by then are
lost.

Trial containers reach vLLM through `socat` on the Docker bridge (`http://172.17.0.1:18000/v1`); vLLM itself
listens on `127.0.0.1:8000`, with the key from the `QWEN_API_KEY` SkyPilot secret. Docker needs
`docker-buildx` on the box: Harbor builds the separate verifier images with buildx.

## Endpoint and tunnel

vLLM listens on `127.0.0.1:8000` on the box only. `fh` reaches it with `ssh -N -L
127.0.0.1:18000:127.0.0.1:8000 fh-serve` (SkyPilot writes the ssh config for `fh-serve`; pid in
`tunnel.pid`, log in `tunnel.log`). The tunnel pins each box's host key in `known_hosts` on first use, under
an alias per launch. Docker here is rootless (slirp4netns, host loopback enabled), so
containers reach the host's `127.0.0.1` as `10.0.2.2`: `QWEN_BASE_URL=http://10.0.2.2:18000/v1`. The URL
stays the same when the box changes. Override with `FH_TUNNEL_PORT`, `FH_TUNNEL_BIND`,
`FH_TUNNEL_CONTAINER_HOST` (rootful Docker: bind and container host `172.17.0.1`). The API key still guards
the port, since every container on this machine can reach it. vLLM gets the key through `VLLM_API_KEY`
from a SkyPilot secret, so it never shows on a command line.

## Training runs

`train-up` hashes the dataset, the trainer code and its requirements, and stores the run definition in
`.fh-state.json`. The box gets a copy of `train/` code and the dataset as its workdir. The job:

- writes `/mnt/fh/runs/<run>/manifest.json` (data, code and requirements hashes, trainer arguments) on
  first start, and refuses to run when the manifest differs or when the directory holds a run without one;
- copies the dataset to `/mnt/fh/data/<name>-<hash>` and the code to `/mnt/fh/runs/<run>/code` when the
  copies on the volume don't match;
- builds `/mnt/fh/envs/train-<requirements hash>` once;
- runs `sft_lora.py`, which resumes from the newest complete checkpoint;
- publishes `final/` to `/mnt/fh/adapters/<run>`.

`train-up` refuses to restage a run when the local data or code changed since it started. LoRA rank must be
1..16 (serving's `--max-lora-rank`), and `--trainer-arg` takes only listed flags with exact names.

## Layout on the volume (`/mnt/fh`)

```
models/Qwen3.8-27B/            BF16, 55.6 GB (rev 3756181)
models/Qwen3.8-27B-FP8/        FP8,  30.9 GB (rev c3a462c)
models/*.REVISION              "repo revision" of a downloaded and verified checkpoint
envs/vllm-0.33.0/              vLLM venv (python under uv/python)
envs/train-<hash>/             trainer venv
cache/                         vLLM / triton compile caches (faster restarts)
data/, runs/, adapters/        training inputs, runs, published adapters
```

## Serving config

- vLLM 0.33.0, served as `qwen3.8-27b`. Flags: `--enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3 --max-model-len 65536 --max-num-seqs 32 --kv-cache-dtype fp8 --gpu-memory-utilization 0.92
  --limit-mm-per-prompt '{"image": 64, "video": 0}'`, plus tensor parallel across all GPUs.
- 75 GB of GPU memory or more uses BF16, less uses the official FP8 checkpoint, and GPUs under 44 GB also
  get `--max-model-len 32768 --max-num-seqs 4`. `serve.yaml` lists only 80 GB+ types; add 40-48 GB types
  to `any_of` to allow FP8.
- Thinking is on by default. To turn it off for a request, send `chat_template_kwargs: {"enable_thinking": false}`.

## Gotchas

- **Use a block volume, not a shared filesystem.** FIN-01 boxes have only a public NIC, so an
  `NVMe_Shared` volume can't be mounted there. A block volume attaches only when the box is created and to
  one box at a time, so training and serving can't share it at the same time.
- **Verda deletes boxes without unmounting.** The fork cuts the page cache writeback delay to about 3 s;
  writes in the last seconds before a delete or a spot loss can still be lost. The trainer's checkpoints
  are written atomically, so a lost write costs at most one checkpoint interval.
- **CPU capacity in FIN-01 runs out often.** `download` then fails; retry later.
- **A6000 has no native FP8.** vLLM falls back to Marlin weight-only FP8. Prefill is slower than on an H100.
- **vLLM needs the venv on PATH** (FlashInfer JIT calls `ninja`). `serve.yaml` also sets
  `VLLM_USE_FLASHINFER_SAMPLER=0`, so a new box doesn't JIT-compile the sampler.

## OpenCode

`opencode-qwen.json` is a sample provider config. It reads `{env:QWEN_BASE_URL}` and `{env:QWEN_API_KEY}`,
and declares `modalities.input: [text, image]` and `attachment: true`. vLLM returns thinking in the
`reasoning` field, so the config sets `interleaved.field` to match.

**Harbor needs this override.** Harbor's OpenCode agent registers a custom model as `"models": {"<id>": {}}`,
which OpenCode treats as text-only, and OpenCode then replaces MCP images with `ERROR: Cannot read image
(this model does not support image input)`. Pass the model fields through `agents[].kwargs.opencode_config`,
which Harbor deep-merges into its generated provider. `harbor-agent` prints the full entry:

```bash
set -a; . ./.env; set +a          # Harbor reads the key from QWEN_API_KEY (model_api_key_name)
uv run python infra/verda/fh.py harbor-agent --opencode-version 1.19.5 > cfg.json
harbor run -c cfg.json -p tasks/eval --print-config   # check that it resolves
```

OpenCode sends MCP images back as a user message ("Attached media from tool result:" plus an `image_url`
part). Each frame counts toward `--limit-mm-per-prompt` (64 images per request), and vLLM returns a 400 error
past that, so raise it in `serve.yaml` if episodes run longer.

`ledger.jsonl` is the spend log of the earlier hand-written CLI; `sky cost-report` covers spend from here on.
