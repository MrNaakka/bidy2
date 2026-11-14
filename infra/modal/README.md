# Modal tooling: Qwen3.8-27B endpoints for Harbor evals, LoRA training

The Modal path next to `infra/verda`: same weights (pinned revision, verified file by file), same vLLM version
and flags, same OpenCode agent config, same trainer. Harbor trials run on Modal sandboxes (`harbor run -e modal`).

```bash
infra/modal/fh <command>      # run from the repo root; Modal token from env, ~/.modal.toml or ~/.local/state/fh/modal.env
```

| file | what it is |
| --- | --- |
| `app.py` | The Modal app `fh-qwen`: `download_weights`, `serve` (one vLLM container per call), `train`. |
| `fh_modal.py` | Local CLI (`fh` wraps it with `uv run --with modal==1.6.4`). Reuses `infra/verda/fh.py` for the Harbor agent entry, hashing and trainer flags. |
| `probe_gpus.py` | nvidia-smi on each GPU type, with time to start. |

Modal objects: volume `fh-qwen-weights` (`models/Qwen3.8-27B` + `.REVISION` marker + `.verify.json`,
`adapters/<sha>/`), volume `fh-train` (`data/<name>-<sha>/`, `runs/<run>/`, `adapters/<run>/`), Dict
`fh-qwen-state` (endpoint URL/phase/GPU/vLLM command, training phase, stop flags), Secret `fh-qwen-key`
(`QWEN_API_KEY` from `.env`; made by `fh secret` or on the first deploy; never printed).

## Commands

| command | what it does |
| --- | --- |
| `weights` | Deploys, then downloads `Qwen/Qwen3.8-27B@3756181…` to `fh-qwen-weights` and runs `infra/verda/verify_weights.py`. Skips when the marker says the verified revision is there. |
| `serve-up [--tag T] [--lora DIR] [--gpu H100]` | One endpoint; prints its `base_url` (`https://….modal.host/v1`). `--lora` uploads `adapter_config.json` + `adapter_model.safetensors` (content-addressed) and also serves `qwen3.8-27b-fh`. |
| `test [--tag T]` | Vision + tool-call request per served name and a streamed throughput probe through the public URL; writes `last_test.json`. |
| `serve-down [--tag T \| --all]`, `status` | Stop endpoints; show endpoint and training state. |
| `eval --tasks DIR [--lora DIR] [--runs 3] [-n 15] [--label L] [--only 2,3] [--sequential]` | Per run: its own endpoint, wait until healthy, Harbor JobConfig with `base_url` = endpoint + `/v1`, `harbor run -e modal -n N` as job `fh-<split>-<label>-run<i>` in `runs/`, then stop the endpoint. Runs go in parallel (20 s apart). Default label `qwen-modal` / `qwen-fh-modal`, so Verda runs in `runs/` are not overwritten. |
| `train-up --run R --data DIR [--epochs 1 --lr 1e-4 --grad-accum 8 --lora-r 16 --save-every 10] [--gpu H100]` | Uploads the data (content-addressed), spawns `train` on the GPU. The same `--run` with the same arguments resumes from the newest checkpoint; a different definition is refused (`manifest.json`). |
| `train-status [--run R]`, `train-fetch --run R`, `train-down --run R` | Loss / s per step; copy `final/` and logs to `train/out/R/`; stop (SIGTERM -> checkpoint at the next step -> exit, then cancel). |
| `stop-all` | Stops every endpoint and training call this machine started, then `modal app stop fh-qwen`. |

## Serving and comparability

Each `serve` call runs in its own container (`single_use_containers`), so one endpoint is exactly one
replica, and it is reached through `modal.forward(8000)`: a raw HTTPS tunnel to that container. Modal does
not route, queue or time out requests on it (no 150 s web-endpoint limit for long streams); vLLM does all
the batching. The vLLM command is `eval_job.sh`'s, flag for flag (`--host 0.0.0.0` instead of 127.0.0.1 for
the tunnel; the API key still guards `/v1`). A LoRA eval adds only
`--enable-lora --max-lora-rank 16 --lora-modules qwen3.8-27b-fh=/w/adapters/<sha>`.

`runs/<job>/fh-eval-meta.json` records the definition (tasks hash, adapter hash, GPU type, `-n`, model and
revision, vLLM/Harbor/OpenCode versions, `app.py` hash, git commit), the endpoint's GPUs (nvidia-smi), the exact
`vllm serve` command, region/cloud, load time and phase timings; `vllm-metrics.txt` (end of run) and
`vllm-tail.log` sit next to it. Harbor's own log is `runs/<job>.harbor.log`.

Backstops so nothing bills forever: an endpoint exits on its stop flag, after `--idle-min` (30) minutes with no
running or waiting requests, after `--max-hours` (5), or when vLLM dies; the function timeout (6 h) is the hard
limit. `eval` stops its endpoints when Harbor ends, on error and on SIGTERM/SIGINT. Training calls time out after
24 h (Modal's maximum; `train-up` again resumes).

Long commands must survive this host's T3 restarts: start them as user units, e.g.

```bash
systemd-run --user --unit=fh-modal-eval-qwen --collect --same-dir --setenv=PATH=$PATH --setenv=HOME=$HOME \
  infra/modal/fh eval --tasks tasks/eval --runs 3 -n 15
journalctl --user -u fh-modal-eval-qwen -f
```

`eval` runs Harbor with the first `harbor` on PATH whose Python has the modal extra (`modal`, `dockerfile_parse`;
Harbor's separate verifier sandbox needs it), or `$FH_HARBOR`. The repo `.venv`'s harbor lacks it, and `uv run`
puts that one first on PATH: with it every trial fails at the verifier with `No module named 'dockerfile_parse'`.

## Status (2026-11-14, backup to the Verda fleet)

GPU probe (`probe_gpus.py`, time from call to nvidia-smi result): H100 3.7 s, H200 3.6 s, A100-80GB 3.4 s,
H100:2 3.5 s, B200 115 s. Weights: download 233 s + verify 30 s onto `fh-qwen-weights`.

Smoke results: an endpoint loads in 174-211 s from spawn to healthy (H100 80GB, eu-south). `test`: no key -> 401,
tool call correct (`view(t=12.5, out_width=320)`, reasoning present), 49.6 decode tok/s single stream, TTFT 0.2 s.
The vision check misread the code (`K7QZP`/`K702P` for `K7Q2ZP`, also at temperature 0) where the Verda box
read it right twice: not investigated (the image's resolved transformers/vLLM deps may differ from the fleet's
venv). Train: first logged step after 207 s (loss 0.424, sft-v2, 109 steps/epoch), `train-down` stopped it cleanly.

## Training

`train` runs `train/sft_lora.py` (code baked into the image) with `/w/models/Qwen3.8-27B` and the data on
`fh-train`, output in `fh-train:runs/<run>`. A thread commits the volume every minute so checkpoints survive a
lost container; Modal retries a failed call twice, and every attempt resumes from the newest complete
checkpoint. When `DONE` is written the adapter is copied to `fh-train:adapters/<run>`; `train-fetch` brings
it to `train/out/<run>/final` (what `eval --lora train/out/<run>` takes). The image installs
`train/requirements.txt` + `requirements-gpu.txt` (same pins as `infra/verda/train.yaml`, so
flash-linear-attention is in). causal-conv1d is not: 1.7.0 has no prebuilt wheel for CUDA 13 / torch 2.15 and
the source build failed, so the GDN conv uses transformers' PyTorch fallback, as on Verda.
