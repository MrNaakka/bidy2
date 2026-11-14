# Verda tooling: Qwen3.8-27B endpoint

`verda_cli.py` serves Qwen3.8-27B (vision + tool calls) as an OpenAI-compatible vLLM endpoint on a Verda spot GPU.
The weights live on a persistent shared volume. Run every command from the repo root:

```bash
uv run python infra/verda/verda_cli.py <command>
```

The CLI reads `VERDA_CLIENT_ID` and `VERDA_CLIENT_SECRET` from `./.env` and never prints them.
It writes `QWEN_API_KEY` to `.env` (generated once) and `QWEN_BASE_URL` (on every serve-up).

## Commands

| command | what it does |
| --- | --- |
| `inventory [--available]` | Lists instance types (GPU and CPU) with on-demand and spot €/h, VRAM, and current availability per location. Writes `inventory.json` with the cheapest BF16 and FP8 fits. |
| `volume-create` | Creates the 300 GB block `NVMe` volume `fh-qwen` in FIN-01 (idempotent). It is formatted ext4 (label `fhqwen`) on first use and mounted at `/mnt/fh`. |
| `download [--type T] [--keep]` | Boots a CPU box (spot first, then on-demand) with the volume attached. It downloads both checkpoints with `hf download` (hf_transfer, pinned revisions) and checks every file's size and LFS sha256 against the Hub. It also builds the vLLM venv on the volume, then deletes the box. |
| `serve-up [--type T] [--variant bf16\|fp8] [--on-demand] [--mtp]` | Starts the cheapest available spot candidate with the volume attached, through an SDK startup script. It waits until `/v1/models` answers from this box, then writes `QWEN_BASE_URL`. |
| `serve-ensure [--wait]` | Idempotent watchdog step. It does nothing if the endpoint is healthy, and leaves a box alone while it boots. If the box is gone (spot preemption), broken, or not healthy after 40 min, it deletes it and starts a new one. A file lock makes concurrent calls safe. |
| `serve-status` / `serve-down` | Shows or deletes the serving box. Down also clears `QWEN_BASE_URL`. |
| `test` | From this box: one chat request with a PNG data-URL image, one tool-call request, and a streamed throughput probe. Writes `last_test.json`. |
| `harbor-agent` | Prints a Harbor `agents[]` entry for OpenCode with image input enabled (see below). |
| `instances`, `delete ID`, `volume-status` | Low-level helpers. |
| `ledger [--balance]` | Total spend from `ledger.jsonl`. Each create and delete appends a line with time, type, €/h and spot flag. |

Watchdog example:

```bash
while true; do uv run python infra/verda/verda_cli.py serve-ensure; sleep 120; done
```

## Layout on the volume (`/mnt/fh`)

```
models/Qwen3.8-27B/            BF16, 55.6 GB (rev 3756181)
models/Qwen3.8-27B-FP8/        FP8,  30.9 GB (rev c3a462c)
models/*.verify.json           per-file size + sha256 check reports
envs/vllm-0.33.0/              venv used by every serving box (python from uv/python)
cache/                         vLLM / triton compile caches (faster restarts)
logs/serve-<hostname>.log      vLLM server logs; logs/download.log
```

## Serving config

- vLLM 0.33.0 (the recipe requires >= 0.17.0), served under the stable name `qwen3.8-27b` on port 8000 with `--api-key`.
- Flags: `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --max-model-len 65536 --max-num-seqs 16 --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image": 64, "video": 0}'`.
- GPUs with 80 GB or more use BF16. 40 to 48 GB GPUs use the official FP8 checkpoint. The candidate list and its order are in `CANDIDATES` in `verda_cli.py`.
- Thinking is on by default. To turn it off for a request, send `chat_template_kwargs: {"enable_thinking": false}`.

## Gotchas found while building this

- **Use a block volume, not a shared filesystem.** An `NVMe_Shared` (SFS) volume reports `exported` with a pseudo path, but FIN-01 instances (CPU and GPU) have only a public NIC. The NFS server answers `No such file or directory` for the export. A block volume attaches only at instance creation (`existing_volumes`) and to one instance at a time. So a training box and the serving box can't share it at the same time.
- **CPU spot creation fails in FIN-01** with "Not enough resources", even when `/instance-availability` lists the type. `download` therefore falls back to on-demand CPU (€0.086/h).
- **vLLM inside systemd needs PATH** to include the venv (FlashInfer JIT calls `ninja`). The unit also sets `VLLM_USE_FLASHINFER_SAMPLER=0`, so a new box doesn't JIT-compile the sampler.
- **A6000 has no native FP8.** vLLM falls back to Marlin weight-only FP8 (28.8 GiB of weights, about 12.8 GiB left for KV). Quality is that of the official FP8 checkpoint. Compute-heavy prefill is slower than on an H100.
- **The endpoint is plain HTTP on a public IP.** The API key (`--api-key`, 401 without it) is the only protection, and it travels unencrypted. Rotate it by deleting `QWEN_API_KEY` from `.env` and running serve-down then serve-up.

## OpenCode

`opencode-qwen.json` is a sample provider config. It reads `{env:QWEN_BASE_URL}` and `{env:QWEN_API_KEY}`, and declares `modalities.input: [text, image]` and `attachment: true`. vLLM returns thinking in the `reasoning` field, so the config sets `interleaved.field` to match.

**Harbor needs this override.** Harbor's OpenCode agent registers a custom model as `"models": {"<id>": {}}`, which OpenCode treats as text-only. The step0 probe showed the effect: OpenCode replaced the MCP image with `ERROR: Cannot read image (this model does not support image input)`. Pass the model fields through `agents[].kwargs.opencode_config`, which Harbor deep-merges into its generated provider. `harbor-agent` prints the full entry:

```bash
set -a; . ./.env; set +a          # Harbor reads the key from QWEN_API_KEY (model_api_key_name)
uv run python infra/verda/verda_cli.py harbor-agent   # paste into the job config's agents[]
```

OpenCode sends MCP images back as a user message ("Attached media from tool result:" plus an `image_url` part). Each frame counts toward `--limit-mm-per-prompt` (64 images per request). If a long episode passes 64 images, vLLM returns a 400 error, so raise the limit in `BASE_VLLM_ARGS` if episodes run that long.
