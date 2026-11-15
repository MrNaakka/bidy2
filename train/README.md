# FrameHunt training (tiebreaker: a smaller model trained on our environment)

We fine-tune **Qwen3.8-27B** on FrameHunt trajectories, then rerun the Harbor eval with the
adapter to show that the score goes up. Qwen3.8-27B is a vision-language model, served by
vLLM 0.33.0 inside Harbor's OpenCode agent.

- **Stage 1: LoRA SFT on teacher trajectories.** Done: v2 and v3 (results below).
- **RL stage:** in progress on a separate branch.

The SFT data format and tooling are the ones an RL stage (rejection sampling / GRPO on our
binary reward) would use.

## Results (EVAL v2, 30 items, Harbor, 3 runs each)

| model | data | mean reward | submitted | mean pixels |
| --- | --- | --- | --- | --- |
| Qwen3.8-27B base | - | 0.211 (0.200 / 0.233 / 0.200) | 13-16 of 30 (15 of 90 trials hit the 1800 s timeout) | 91-96k |
| + SFT v3 | 1,200 rows, including scandoc | 0.522 (0.533 / 0.467 / 0.567) | 29-30 of 30 | 70-84k |

An earlier adapter, SFT v2 (900 rows: logstream, dialogs, toasts; no scandoc), was run on
the 30 items EVAL had then; rescored on its 20 logstream and dialogs items: 0.617 (0.650 / 0.550 / 0.650), 20 of 20
submitted, 65-72k pixels.

SFT v3 settings: LoRA r=16, 1 epoch (145 optimizer steps), lr 1e-4, 4xH100 DDP. Held-out eval
loss was 0.015 (logs in `docs/results/data/`).

Per kind, v3 vs base:

| kind | v3 | base |
| --- | --- | --- |
| logstream | 0.90 | 0.33 |
| dialogs | 0.40 | 0.30 |
| scandoc | 0.27 | 0.00 |

The scandoc rows were generated with the video tools' descriptions, which is a small mismatch
with the eval tasks.

Charts and tables: [docs/results/](../docs/results/tables.md).

## Files

| file | what it does |
| --- | --- |
| `capture/` | The real OpenCode 1.19.5 request format, captured from a Harbor run against a logging mock LLM (`mock_opencode.py`, `run_capture.sh`, `job_capture.json`): `requests/req_NN.json`, `opencode_system_prompt.txt`, `opencode_tools.json`, `opencode_request_meta.json`, `harbor_trial/`. |
| `fhdata.py` | The SFT row format (= OpenCode's request body) and its conversion to Qwen tokens with assistant-only labels. Shared by every script below. |
| `make_sft.py` | Generates conversations on the TRAIN split with a varied teacher policy; keeps only correct ones. |
| `token_stats.py` | Token counts per conversation with the real Qwen3.8 processor (tokenizer files only). |
| `sft_lora.py` | LoRA SFT, checkpoint every N steps, automatic resume; `--init-adapter` continues from a trained adapter. |
| `rollouts_to_sft.py` | Expert iteration: rewarded fleet rollouts (logged by `infra/verda/log_relay.py`) -> rows in `make_sft.py`'s format. |
| `merge_sft.py` | Merges data dirs (teacher + rollouts) into one for `sft_lora.py` / `fh train-up`. |
| `make_tiny_model.py` | Tiny random-weight `Qwen3_5ForConditionalGeneration` for CPU validation. |
| `requirements.txt`, `requirements-gpu.txt` | Pinned training stack (transformers >= 5.8.0 is required for `qwen3_5`). |

## What OpenCode sends the model (captured)

Source: `capture/requests/` (OpenCode 1.19.5 via Harbor 0.25.0, provider overlay from
`infra/verda/opencode-qwen.json`).

- **Request:** `POST /v1/chat/completions`, `stream: true`, `max_tokens: 16384`,
  `tool_choice: "auto"`. There is no `reasoning_effort` or `chat_template_kwargs`, so Qwen's
  defaults apply: thinking on, effort `xhigh`, `preserve_thinking` on. Each run also makes one
  extra request with no tools. That is OpenCode's title generator, and it is not training data.
- **Messages:** one system message, then one user message. The system message is OpenCode's
  generic coding-agent prompt (9.5 kB), with an `<env>` block holding the date, `/app` and linux.
  The user message is the task's `instruction.md` **wrapped in double quotes**.
- **Tools:** 17, in this order: `bash, edit, framehunt_contact_sheet, framehunt_info, framehunt_submit,
  framehunt_view, glob, grep, list_mcp_resource_templates, list_mcp_resources, read,
  read_mcp_resource, skill, task, todowrite, webfetch, write`. MCP tools are named
  `<server>_<tool>` and keep the FastMCP JSON schemas and docstrings.
- **Assistant turns** come back as `{"role":"assistant","content":"...","reasoning":"...",
  "tool_calls":[{"id","type":"function","function":{"name","arguments":"<compact JSON>"}}]}`.
  OpenCode re-serializes arguments with no spaces. The reasoning goes in vLLM's `reasoning` field,
  which vLLM 0.33 mirrors to `reasoning_content` for the chat template.
- **Tool results:** `{"role":"tool","tool_call_id":...,"content":"<the MCP text>"}`. For `view` and
  `contact_sheet` the image follows as a separate user message:
  `[{"type":"text","text":"Attached media from tool result:"},{"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}]`.
  The PNG is passed through unchanged (pixel-identical to the server's output).
- **End of run:** after `submit`, the model's text-only reply ends the run.

Qwen's template renders this as `<tool_response>` blocks in a user turn, followed by a second
user turn holding the image. Every past assistant turn keeps its `<think>` block.

`make_sft.py --check-capture` replays the captured run in-process. It asserts that the system
prompt, user turn, tool list, every tool text and every image match the capture.

### Pixel parity gotcha

The sidecar's Pillow (python:3.12-slim) has no libraqm, so it lays out text with Pillow's BASIC
engine. A host with libraqm renders glyphs slightly differently (up to 63 grey levels on captured
frames). `make_sft.py` forces BASIC layout before any font loads, which makes its frames
pixel-identical to the sidecar's.

PNG bytes can still differ by zlib build (zlib-ng on Fedora). The decoded pixels, which are all
the model sees, are identical. For byte-identical files too, run it in the sidecar's base image
(below).

Note that the reference policy's OCR templates are also rendered with whatever layout engine the
host has.

## 1. Generate the SFT data

```bash
uv run python train/make_sft.py --out train/data/sft-v1 --per-kind 300 --workers 4   # ~2.5 min on 4 cores
uv run python train/make_sft.py --check-capture                                      # format parity check (needs secrets/eval.salt)
# byte-identical PNGs: run inside the sidecar image
docker run --rm -v "$PWD":/w -w /w python:3.12-slim sh -c \
  "pip install -q pillow==13.0.0 numpy==2.5.4 mcp==1.30.0 && python train/make_sft.py --out train/data/sft-v1"
```

### Items

`framehunt.splits.train_items(300)` gives 300 items per kind (`logstream-100001..`,
`dialogs-110001..`, `dashboard-120001..`). The scene seed is `derive_seed("train", item_id)`,
using the TRAIN split's own secret salt (`secrets/train.salt`, created on first run, git-ignored).

The HMAC uses a different split name and a different salt from EVAL/DEV, so no training scene
coincides with an eval scene. TRAIN is not in `SPLITS`, so the Harbor tooling never builds it.

### Teacher policy

One conversation per item, randomised per item from `--seed`. For the video kinds, each
conversation runs these steps (scandoc navigates by the contents page, running headers and table
dates instead, without the overview and event search):

1. Optional free `info` (~55 %).
2. A low-res overview, 120-240 px wide. For dashboard this is the last frame, where the named
   tile is already red.
3. A contact sheet of the trace crop (counter / bell badge / tile strip) over the whole video,
   with n in {6..16}. Tile widths are 1/8-1/6 (logstream, dialogs) or a few pixels (dashboard).
   **Every tile's value is spelled out with its tile number and time.**
4. Sometimes a second, narrower sheet.
5. Binary search with single cheap views down to the event's first frame.
6. **Always a verification look at that frame at 2-3x the search resolution** (a single view, or
   a 2-tile sheet of frames f-1 and f). This is the step Opus skipped when it misindexed a sheet.
7. A cheap locating look: red ERROR row in the level column, dialog card at 1/16-1/12, or KPI row
   label at ~1/2.
8. One native crop of the answer with random padding (sometimes including `req=` or the KPI
   label).
9. `submit`, then a one-line final message.

Perception is the reference policy's template matching and font OCR, so every reasoning sentence
describes what the returned image actually shows. Only trajectories the server marks correct are
kept.

### Dataset `sft-v1` (seed 0)

**900 items, 900 kept** (the teacher solved all of them), with 13 to 17 tool calls each. Mean
pixels used: 66k (logstream), 61k (dialogs), 48k (dashboard); max 101k of 120k.

Token counts below use the Qwen3.8 processor and include the OpenCode prompt and the 17 tools
(`token_stats.py`; full table in `train/data/sft-v1/token_stats.json`):

| per conversation | mean | min | p95 | max | total (900) |
| --- | --- | --- | --- | --- | --- |
| tokens | 14,340 | 12,384 | 14,806 | 15,205 | 12.9M |
| trained (assistant) tokens | 2,414 | 1,435 | 2,617 | 2,829 | 2.17M |
| image tokens | 998 | 480 | 1,136 | 1,231 | 0.90M |
| images | 13.2 | 6 | 15 | 15 | 11,844 |
| assistant turns | 15.7 | 9 | 17 | 18 | 14,134 |

9,282 tokens of every row are the shared prefix (system prompt + 17 tool schemas + instruction).
Images are ~75 tokens each, because Qwen's processor upsamples every small image to >= 65,536 px.

## 2. Train on a Verda H100

`infra/verda/fh` does all of the steps below on a SkyPilot box (`infra/verda/train.yaml`, see
`infra/verda/README.md`). The manual commands further down are what its job runs.

```bash
infra/verda/fh volume                     # once per machine: SkyPilot learns the fh-qwen volume
infra/verda/fh download                   # once: BF16 + FP8 checkpoints and the vLLM venv onto fh-qwen
infra/verda/fh train-up --run sft-v1 --data train/data/sft-v1 --epochs 1 --max-usd 45
scripts/watchdog.sh train <logfile> <stopfile>   # train-ensure every 2 min: spot loss, budget, fetch at the end
infra/verda/fh train-status               # loss, eval loss, spend
infra/verda/fh train-fetch --run sft-v1   # final/ and logs to train/out/sft-v1/ (train-ensure does it when done)
infra/verda/fh train-down                 # stop; train-up with the same --run and arguments resumes
```

`train-up` refuses to start while `serve-*` holds `fh-qwen` (one box per block volume). The eval
boxes use their own volumes (`fh-eval-<i>`), so evals can run while training does.

### Memory plan (bf16 LoRA, no quantization)

| item | estimate |
| --- | --- |
| Weights | 27B x 2 B = 54 GB |
| LoRA r=16 on full-attention q/k/v/o (16 layers) and MLP gate/up/down (64 layers) | ~80M parameters; ~1 GB with fp32 AdamW states |
| Gradient checkpointing: one 5120-wide bf16 hidden state per layer and token | 64 x 10 kB = 0.66 MB/token, ~10 GB at 15k tokens |
| One layer's recompute (MLP intermediates are 17408 wide) | ~2-4 GB |
| Logits, only at the ~2.5k trained positions, in checkpointed chunks of 1024 | ~1 GB instead of 15k x 248k x 4 B = 15 GB |
| Vision tower forward (~1k image tokens, no grad) | small |
| **Total at batch 1** | **~70 GB** |

So **bf16 LoRA fits one 80 GB GPU**. QLoRA (`--quant 4bit`, weights ~16 GB) is the fallback for
40-48 GB GPUs or much longer rows. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

### Volume layout

The block volume attaches to one instance at a time, so the serving box must be down while
training, and vice versa.

```
/mnt/fh/models/Qwen3.8-27B/        BF16 base (train from this, not the FP8 checkpoint)
/mnt/fh/data/sft-v1/               copy of train/data/sft-v1 (rsync it up, ~85 MB)
/mnt/fh/envs/train/                training venv
/mnt/fh/runs/sft-v1/               checkpoints, loss.jsonl, eval.jsonl, final/
```

### Manual commands on the GPU box

```bash
python3 -m venv /mnt/fh/envs/train && . /mnt/fh/envs/train/bin/activate
pip install -r train/requirements.txt -r train/requirements-gpu.txt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train/sft_lora.py --model /mnt/fh/models/Qwen3.8-27B --data /mnt/fh/data/sft-v1 \
    --out /mnt/fh/runs/sft-v1 --epochs 2 --grad-accum 8 --lr 1e-4 --lora-r 16 --save-every 10
```

Defaults:

- 3 % of rows (27, stratified by kind) held out for eval loss at every checkpoint.
- Cosine LR with 3 % warmup; grad-norm clip 1.0.
- LoRA alpha 32, dropout 0.05.
- 873 train rows x 2 epochs / 8 = 218 optimizer steps.

**Spot preemption: rerun the same command.** It loads the newest complete `ckpt-*` (adapter,
optimizer, LR schedule, RNG, position in the per-epoch shuffled order), truncates `loss.jsonl` to
that step and continues.

- Checkpoints are written to a temp dir and then renamed, so a kill mid-save never leaves a half
  checkpoint.
- SIGTERM checkpoints at the next optimizer step and exits 143.
- A changed argument in the same `--out` is refused.

### Several GPUs on one box (data parallel)

Each H100 holds a full replica (~70 GB peak), so
`torchrun` splits every optimizer step's `--grad-accum` conversations across the GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train/sft_lora.py --model /mnt/fh/models/Qwen3.8-27B \
    --data /mnt/fh/data/sft-v2 --out /mnt/fh/runs/sft-v2 <the run's other arguments, unchanged>
# continue a checkpoint written elsewhere (e.g. by the single-GPU run) in a fresh --out:
torchrun --standalone --nproc_per_node=4 train/sft_lora.py ... --out /mnt/fh/runs/sft-v2-dp \
    --resume-from /path/to/sft-v2/ckpt-000020
```

Same optimization as one GPU with the same arguments: step s still trains global micro indices
`[s*accum, (s+1)*accum)` of the same per-epoch shuffle, rank r takes every K-th of them (accum need not
divide by K), the step's trained-token total is all-reduced before backward so the token-weighted loss
normalisation is unchanged, and the LoRA grads are summed with one all_reduce before clipping and AdamW.
Checkpoints keep only the global step/micro, so they resume on any K, including single-GPU
checkpoints (`--resume-from` checks the source run's `args.json` for the arguments that fix data order
and schedule). Rank 0 writes logs and checkpoints; eval is sharded. Only the LoRA dropout masks differ
(per-rank RNG). `tests/test_sft_ddp.py` checks 2 gloo processes against 1 on the tiny model: same loss,
grad norm and trained tokens per step, same eval loss and adapter, and a single-process checkpoint
resumed on 2 ranks ends at the same adapter.

### Time and cost estimate

Assumptions: H100 80 GB SXM, 12.9M tokens per epoch. With LoRA and recompute, training costs
~6 x 27e9 FLOPs per token, so 2.1e18 FLOPs per epoch.

- At 25-35 % MFU (250-350 TFLOP/s for batch-1 HF training with the fla kernels): **1.7-2.4 h per
  epoch, 3.5-5 h for 2 epochs**, plus ~15 min setup (venv, loading 55 GB from the volume).
- At the spot price we paid (1H100.80S, ~EUR 2.34/h): **~EUR 9-13 for 2 epochs, ~EUR 5-7 for
  one**.

Run 1 epoch first and evaluate. The 13M-token epoch is dominated by the 9.3k-token prefix that
every row must include to match OpenCode.

Without `flash-linear-attention`, the 48 Gated DeltaNet layers use the PyTorch reference kernels
and the run is several times slower.

## 3. Serve the adapter and evaluate with Harbor

### Parallel eval

Three boxes, each serving the base model and the adapter shipped from `train/out/<run>/final`:

```bash
infra/verda/fh eval-up --tasks tasks/eval --lora train/out/sft-v1 --runs 3 -n 15 && infra/verda/fh eval-wait
# or: scripts/eval.sh --stage agents --model qwen-fh --lora train/out/sft-v1 --fleet
```

### Single endpoint on `fh-qwen`

A finished run's `final/` (PEFT adapter, ~300 MB) lands in `/mnt/fh/adapters/<run>/`.
`infra/verda/fh serve-up --lora /mnt/fh/adapters/sft-v1` adds these flags to the vLLM command in
`infra/verda/serve.yaml`. The other flags stay as they are, in particular
`--tool-call-parser qwen3_coder --reasoning-parser qwen3`.

```bash
--enable-lora --max-lora-rank 16 \
--lora-modules qwen3.8-27b-fh=/mnt/fh/adapters/sft-v1
```

The base model stays available as `qwen3.8-27b` and the adapter is served as `qwen3.8-27b-fh`,
so the baseline and the trained model run on the same box. `--max-lora-rank` must be >= `--lora-r`.

Default LoRA targets are standard vLLM LoRA layers (qkv_proj, o_proj, gate_up_proj, down_proj).
`--lora-linear-attn` also adapts the Gated DeltaNet projections, which vLLM packs as
`in_proj_qkvz`. Test-load such an adapter in vLLM before training a long run with it.

Harbor agent entry: use the one that `fh.py harbor-agent --served-name qwen3.8-27b-fh` prints.
It has `"model_name": "verda-qwen/qwen3.8-27b-fh"`, and the model key under
`opencode_config.provider.verda-qwen.models` is renamed to `qwen3.8-27b-fh` (same fields: image
input, reasoning field, limits).

Pin the agent's OpenCode version to the captured one (`"version": "1.19.5"` in the agents[] entry)
so the prompt and tool list match the training data. Then run the eval split three times for both
`qwen3.8-27b` (before) and `qwen3.8-27b-fh` (after):

```bash
harbor run -c job-qwen-fh.json -p tasks/eval --job-name fh-eval-qwen-fh-run1 -o runs -n 4 -y
uv run python scripts/summarize.py --runs runs --out results
```

## CPU validation (done here)

```bash
train/.venv/bin/python train/make_tiny_model.py          # 32M-param qwen3_5 VLM from the real config, real tokenizer/processor
train/.venv/bin/python train/sft_lora.py --model train/out/tiny-qwen3.8 --data train/data/sft-v1 \
    --out train/out/cpu-b --limit 40 --eval-frac 0.05 --grad-accum 2 --epochs 0.25 --save-every 2 \
    --dtype fp32 --lr 1e-3 --threads 2
```

Results on real rows (14.3k tokens, 13 images each), tiny model:

- **Run A**, uninterrupted, 4 optimizer steps: loss 12.448 -> 12.409 -> 12.369 -> 12.343. It
  starts at ln(248,320) = 12.42, as a random model should. Eval loss 12.374 (step 2) -> 12.336
  (step 4). Checkpoints `ckpt-000002`, `ckpt-000004`; adapter in `final/`.
- **Run B**, same command, SIGTERM during step 1: it finished step 2, checkpointed and exited 143.
  The rerun printed `resumed from ckpt-000002: step 2, micro 4` and finished. Steps 3-4 losses,
  the eval loss and **all 32 adapter tensors are bit-identical to run A**.
- **Label masking:** `token_stats.py --show 1` prints the trained segments. They are exactly each
  assistant turn after `<think>\n` (reasoning, `</think>`, the `<tool_call>` block, `<|im_end|>`)
  and the final text turn. System prompt, tools, instruction, tool results and images stay in
  context, but their positions are excluded from the loss.
- **LoRA placement:** LoRA lands only on language-model layers (asserted at load).
  `--lora-linear-attn` is also validated (25 adapted modules in the tiny model: 12 MLP,
  4 attention, 9 GDN).

On a busy CPU box, pass `--threads 1` or `2`. In our test, thread oversubscription made the
PyTorch Gated DeltaNet fallback ~10x slower.

## Toward RL

Rows are complete OpenCode requests, so the same pieces serve an RL loop:

1. Roll out the current adapter against `FrameHuntEnv` in-process. Tool calls map to
   `env.view/contact_sheet/submit`, and results are appended exactly as `make_sft.Episode` does.
2. Score with `env.result()["reward"]` (binary, the same reward the verifier uses).
3. Either keep the correct rollouts as new SFT rows (rejection sampling / expert iteration,
   reusing `sft_lora.py` unchanged), or feed reward-weighted sequences to a policy-gradient
   trainer.

`fhdata.encode` already gives the assistant-token mask a policy-gradient loss needs.

## Expert iteration (optional; not run yet)

Reward-filtered on-policy fine-tuning: the SFT adapter plays fresh TRAIN items on the eval fleet, the
verifier's reward picks the trajectories it solved, and those become SFT rows (same format as
`make_sft.py`'s, so the same trainer and chat template apply) that are mixed with the teacher data for a
second LoRA round that starts from the SFT adapter. Nothing here changes a default command: logging is off
unless `--log-requests` is given, and `--split train` / `--init-adapter` are new options.

```bash
# 1. TRAIN-split tasks: items 301-340 of each EVAL kind (make_sft.py used 1-300), TRAIN salt, never EVAL/DEV
uv run python scripts/build_tasks.py --split train --per-kind 40 --offset 300 --out tasks/train-ei
# 2. rollouts with the SFT adapter, every chat request logged per trial (2 boxes = 2 attempts per item)
infra/verda/fh eval-up --tasks tasks/train-ei --lora train/out/sft-v1 --label sft --runs 2 -n 15 \
    --log-requests --max-usd 30
infra/verda/fh eval-wait --fleet train-ei-sft     # fetches runs/fh-train-ei-sft-run{1,2}/ incl. requests/
# 3. reward-1 trajectories -> SFT rows (add --processor <Qwen3.8 dir> under train/.venv to also compare
#    the chat-template text)
uv run python train/rollouts_to_sft.py --out train/data/ei-1 --tasks tasks/train-ei \
    runs/fh-train-ei-sft-run1 runs/fh-train-ei-sft-run2
# 4. mix with the teacher data
uv run python train/merge_sft.py --out train/data/ei-1-mix train/data/sft-v1 train/data/ei-1
# 5. continue training from the SFT adapter (published on the volume by train-up as /mnt/fh/adapters/sft-v1)
infra/verda/fh train-up --run ei-1 --data train/data/ei-1-mix --epochs 1 --lr 5e-5 \
    --trainer-arg=--init-adapter=/mnt/fh/adapters/sft-v1
infra/verda/fh train-ensure                       # until done; the adapter lands in train/out/ei-1
# 6. evaluate like any adapter
infra/verda/fh eval-up --tasks tasks/eval --lora train/out/ei-1 --label qwen-ei --runs 3 -n 10
infra/verda/fh eval-wait --fleet eval-qwen-ei
```

**Logging relay** (`infra/verda/log_relay.py`, stdlib only). With `LOG_REQUESTS=1` (set by
`eval-up --log-requests`) `eval_job.sh` runs it on `172.17.0.1:18000` instead of socat. Request bodies and
response bytes pass through unchanged (SSE streamed chunk by chunk, one thread per connection); after each
chat completion it appends the request and the response reassembled from the stream (content, reasoning,
tool calls, finish reason, usage) to `~/fh/requests/<job>/<trial>/log.jsonl`. Images are stored once per
trial under `blobs/<sha256>` and the tool list under `tools/` (lossless; the converter restores the exact
request). The Authorization header is never logged. OpenCode sends no trial id, so the relay maps the client
IP to its container with `docker inspect` and takes the Compose project, which Harbor names
`<trial_name>__env`; lookups are cached per (IP, OpenCode session), so a new trial on a reused IP is looked
up again. `eval-fetch` copies the logs to `runs/<job>/requests/`. The flag is part of the fleet definition.

**Converter checks** (`rollouts_to_sft.py`; a failing trial goes to `rejected.jsonl` with the reason):
only TRAIN item ids (an EVAL/DEV/unknown item anywhere in the inputs aborts before writing); reward 1;
one OpenCode session with tools; every request a prefix of the last one, and every logged response equal to
the assistant turn OpenCode sent back next; the MCP server's call log (`verifier/calls.jsonl`) is an
in-order subsequence of the conversation's `framehunt_*` calls with the same arguments (a log can't be
pinned on the wrong trial); OpenCode's captured system prompt and 17 tools exactly; exactly one tool call
per assistant turn; a submit. The row is the last request plus its response as the final assistant turn,
images saved as the exact PNG bytes, and `to_wire(row)` must equal the logged request, `to_template` must
give the same messages and pixels (and with `--processor` the same template text). Rows carry
`provenance` (run, trial, model, reward, session, adapter hash). Identical trajectories are dropped and the
cheapest `--max-per-item` (default 1: fewest pixels, then fewest calls) are kept, so easy items don't
dominate.

**Mixing and init.** `merge_sft.py` concatenates data dirs (unique row ids, identical `tools.json`,
images hard-linked) and tags rows with `data_source`. `sft_lora.py --init-adapter DIR` loads a trained
adapter's LoRA weights before step 0 (rank and alpha must match; every LoRA tensor must load); optimizer
and schedule start fresh, and a resumed run uses its own checkpoint. `fh train-up` allows the flag via
`--trainer-arg`. CPU check: one step at LR 1e-30 from `train/out/cpu-a/final` saved a checkpoint
bit-identical to it; a rank mismatch is refused.

Cost and time (estimates, not measured): rollouts are an eval fleet on other tasks. Per box ~30-40 min
setup (weights, vLLM) plus 120 trials at `-n 15` (8 waves of up to 30 min; ~1.5-2.5 h), so ~2-3 h and
~EUR 5-8 per spot H100 box; 2 boxes ~EUR 10-16. Yield is the adapter's solve rate on TRAIN items (at 30 %,
~70 trajectories from 240 attempts, ~50-60 items after one-per-item). Training on ~950-1000 rows x 1 epoch
is ~2-2.7 h on one H100 (~EUR 5-7); the eval is the usual 3-box fleet (~1.5-2 h, ~EUR 10-15). End to end
~6-8 h wall clock and ~EUR 25-40.

Tests (`tests/test_expert_iteration.py`, all local and mocked): byte-identical forwarding of a 2 MB
body and of the SSE stream, live streaming (the client sees the first event while the upstream is still
generating), 15 concurrent trials, the key never logged, per-session trial lookup; rollouts played through
the real relay against a fake vLLM and converted (row equals the logged request, exact PNGs, a trial whose
server log disagrees is rejected, dedupe and one-per-item, `--tasks` instruction check); the split guard;
format rejections; merge; `--split train` task building; `--log-requests` staying opt-in. With
`FH_CAPTURE_REQUESTS=train/capture/requests` a captured OpenCode request round-trips too (also checked with
the Qwen processor: identical template text, and the same 35k-character system+tools prefix as a
`make_sft.py` row).

## GRPO (on-policy RL from the SFT adapter)

A small GRPO-style stage on top of the v3 SFT adapter, run on the 8xH100 Verda box
(`infra/verda/grpo/`). Each iteration:

1. **Rollouts.** `serve.sh` serves the current adapter (vLLM 0.33, the eval fleet's flags, LoRA served
   as `qwen3.8-27b-fh`) on every GPU behind `log_relay.py` (172.17.0.1:18100+g). 40 fresh TRAIN items
   (10 per kind: logstream, dialogs, toasts, scandoc; iteration k uses items 301+10(k-1)..310+10(k-1),
   `build_tasks.py --split train --per-kind 10 --offset 300+10(k-1)`; make_sft used 1-300) are dealt to
   the GPUs; one Harbor job per GPU, `-k 8` attempts per item, the evals' agent config (OpenCode 1.19.5,
   `cfg-ft-g*.json` with the relay's URL). Reward = `verifier/reward.txt`.
2. **Advantages.** `rollouts_to_pg.py` turns EVERY rollout (not only the successes, and also runs that
   never submit) into a row holding the exact logged conversation (the checks of `rollouts_to_sft.py`),
   with `advantage = (r - mean_group) / (std_group + 1e-6)` over the item's G attempts (population std).
   Groups with zero std are dropped; harness failures (setup/verifier errors) are left out of their
   group, agent timeouts count as reward 0. `pg_stats.json` holds the mean reward per kind (the
   training-reward curve) and the number of groups with signal.
3. **Update.** `sft_lora.py --pg --init-adapter <current>`: loss = sum over rows of A * sum_t CE_t over
   the assistant tokens, divided by the step's trained tokens (the SFT normalization with a signed row
   weight), i.e. the REINFORCE/GRPO gradient with token-level averaging. One pass over the rows
   (<= `STEPS` optimizer steps, every row once), constant LR, grad clip 1.0, LoRA dropout 0, torchrun
   data parallel on 8 GPUs. No KL penalty, no importance ratio (each row is used once; later steps of
   an iteration are slightly off-policy).

```bash
# on the box (code rsynced to /root/fh/grpo-code, tasks/grpo-it{1..4} built locally and copied)
systemd-run --unit=fh-grpo-loop --setenv=ITERS=4 --setenv=DEADLINE=2026-11-15T01:30Z \
    bash -c "/root/fh/grpo-code/infra/verda/grpo/loop.sh >> /root/fhdata/grpo/loop.log 2>&1"
# one iteration by hand:
ITER=1 INIT=/root/fhdata/runs/sft-v3/final TASKS=/root/fh/tasks/grpo-it1 infra/verda/grpo/iterate.sh
```

`loop.sh` evaluates the last adapter like every fine-tuned eval (`eval.sh`: GPUs 0-2, socat relays, 40
EVAL items, 3 runs `fh-eval-qwen-grpo-run{1,2,3}`, `-n 15`). Per-iteration logs: `/root/fhdata/grpo/`
(`times.txt`, `curve.jsonl`, `it<k>/{runs,data/pg_stats.json,train/loss.jsonl,train/final}`).
Tests: `tests/test_grpo.py` (advantages, conversion of all rollouts, and on the tiny model: A>0 lowers
the row's NLL, A<0 raises it, zero-advantage rows change nothing, 2-rank DDP equals one process).

### Async GRPO (`infra/verda/grpo_async/`, `train/async_grpo.py`)

The same objective without the synchronous rollout -> update barrier:

* **Inference** (`serve.sh`): long-lived vLLMs (fleet flags + `VLLM_ALLOW_RUNTIME_LORA_UPDATING`,
  `--max-loras 3`) serving the policy as `fh-v<N>`; a new version is hot-loaded with
  `/v1/load_lora_adapter`, never a restart.
* **Relay** (`log_relay.py --policy-file`): pins every trial to the version current at its first request,
  rewrites `model` to it, adds `logprobs`/`return_tokens_as_token_ids`, and hands OpenCode the reply
  with the log-probs removed and the public model name restored. The log keeps OpenCode's request plus
  the behaviour log-prob of every generated token (checked against `usage.completion_tokens`;
  `check_logprobs.py` confirms reasoning, content and tool-call tokens all get one under the qwen3
  parsers).
* **Scheduler** (`scheduler.py`): keeps ~24 trials per inference GPU in flight as one small Harbor job
  per TRAIN item (`-k 8` = one group, 8 min agent timeout); a finished job is converted by
  `rollouts_to_pg.py --behavior` and, with signal, queued for the trainer; every group is logged to
  `events.jsonl`; new adapters are loaded into every vLLM before the policy switches; adapters are
  unloaded only when no running job can be pinned to them.
* **Trainer** (`async_grpo.py`, torchrun): takes 4-12 ready groups sampled by the last <= 2 versions,
  PPO-clipped loss `-min(r A, clip(r, 0.8, 1.2) A)` with `r = exp(logp_theta - logp_behaviour)` per
  assistant token (behaviour log-probs aligned to the trainer's tokens by id; unmatched tokens use ratio
  1), token-normalized, one optimizer step, publishes `adapters/v<N+1>`. `metrics.jsonl`: reward per kind,
  staleness histogram, clip fraction, k3 KL(behaviour || current), matched fraction, grad norm, samples/min.
* `start.sh` starts it all as systemd units; `cutover.sh` switches from the sync loop at an iteration
  boundary and runs the final evals. Pipeline test without a real policy: `make_tiny_model.py --vllm`
  (133M random model with the real head dims, servable by vLLM) and `scheduler.py --synthetic` (coin
  rewards).

Operational notes from the first run (2026-11-14):
* Trial startup, not the GPUs, limits rollouts. Every pool task has identical build contexts, so
  `prebuilt_pool.py` points them at three images built once (task.toml `docker_image` for the agent and
  verifier environments, `image:` for the fh-mcp sidecar); environment setup dropped from minutes to
  ~15 s. With ~150 trial containers the box is CPU-bound (dockerd ~24 cores, OpenCode's npm install
  step), so more in-flight trials do not help.
* `scheduler.py --adopt` takes over a frozen (SIGSTOP) scheduler's Harbor jobs, so the scheduler can be
  changed without killing trials (`rebalance.sh` used it to move from 4+4 to 6 inference + 2 trainer
  GPUs). Never move a group out of `queue/` by hand while the trainer may be using it.
* Rows over 18432 tokens are skipped (OOM on the 27B at ~24k with the reference Gated DeltaNet kernels).
