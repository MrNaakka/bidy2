"""FrameHunt on Modal: Qwen3.8-27B weights, a vLLM endpoint per eval run, LoRA training.

Deployed as the Modal apps `fh-qwen` (weights, serving) and `fh-qwen-train` (training) by `infra/modal/fh`
(every command deploys the app it uses first).
The local side (spawning, waiting, Harbor, fetching) is fh_modal.py; see README.md.

- download_weights: Qwen/Qwen3.8-27B at the fleet's pinned revision onto the volume fh-qwen-weights,
  checked file by file with infra/verda/verify_weights.py; marker models/Qwen3.8-27B.REVISION.
- serve: ONE container per call: `vllm serve` with exactly the Verda fleet's flags (eval_job.sh), exposed
  through a Modal tunnel (a raw HTTPS tunnel to the container: no Modal request routing, queueing or
  per-request timeout; vLLM does the batching). Publishes url/phase/GPU/vllm command in the Dict
  fh-qwen-state under the call's tag. Exits on a stop flag, after idle_min without requests, after
  max_hours, or when vLLM dies; the function timeout is the hard backstop.
- train: train/sft_lora.py against the weights volume, data and runs on the volume fh-train,
  resuming from the newest checkpoint (Modal retries the call after a preemption).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] if len(HERE.parents) > 1 else HERE   # in the container app.py is /root/app.py

APP_NAME = "fh-qwen"
MODEL_REPO = "Qwen/Qwen3.8-27B"
MODEL_REV = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"   # infra/verda/eval.yaml
VLLM_VERSION = "0.33.0"
SERVED_NAME = "qwen3.8-27b"
LORA_NAME = "qwen3.8-27b-fh"
LORA_MAX_RANK = 16
SERVE_GPU = "H100"          # pinned like the fleet's 1H100.80S.30V; fh_modal --gpu overrides per call (recorded)
TRAIN_GPU = "H100"
SERVE_TIMEOUT_H = 6         # hard backstop per endpoint (the fleet's MAX_RUN_HOURS is 4)
TRAIN_TIMEOUT_H = 24        # Modal's maximum; a longer run continues on the next call (resume)

W = "/w"                    # fh-qwen-weights: models/, adapters/<sha>/
T = "/t"                    # fh-train: data/<name>-<sha>/, runs/<run>/, adapters/<run>/
MODEL_DIR = f"{W}/models/{MODEL_REPO.split('/')[1]}"

app = modal.App(APP_NAME)
train_app = modal.App(APP_NAME + "-train")   # separate, so its image builds never hold up serving
weights = modal.Volume.from_name("fh-qwen-weights", create_if_missing=True)
trainvol = modal.Volume.from_name("fh-train", create_if_missing=True)
state = modal.Dict.from_name("fh-qwen-state", create_if_missing=True)
key_secret = modal.Secret.from_name("fh-qwen-key")      # QWEN_API_KEY, made by `fh secret` from .env

CUDA_BASE = "nvidia/cuda:13.1.2-devel-ubuntu24.04"     # CUDA_HOME for Triton/FlashInfer JIT, as on the fleet

dl_image = (modal.Image.debian_slim(python_version="3.12")
            .uv_pip_install("huggingface_hub==2.4.0")
            .add_local_file(ROOT / "infra/verda/verify_weights.py", "/opt/fh/verify_weights.py"))

vllm_image = (modal.Image.from_registry(CUDA_BASE, add_python="3.12")
              .apt_install("curl")
              .uv_pip_install(f"vllm=={VLLM_VERSION}")
              .env({"HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    "CUDA_HOME": "/usr/local/cuda", "HF_HUB_DISABLE_TELEMETRY": "1"}))

_TRAIN_CODE = ("sft_lora.py", "fhdata.py", "requirements.txt", "requirements-gpu.txt")
train_base = (modal.Image.from_registry(CUDA_BASE, add_python="3.12")
              .apt_install("git", "build-essential", "ninja-build")
              .uv_pip_install(requirements=[str(ROOT / "train/requirements.txt"),
                                            str(ROOT / "train/requirements-gpu.txt")])
              .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HUB_OFFLINE": "1",
                    "TOKENIZERS_PARALLELISM": "false", "CUDA_HOME": "/usr/local/cuda"}))
# causal-conv1d: not installed. 1.7.0 is sdist-only on PyPI, GitHub has no prebuilt wheel for cu13/torch2.15/cp312,
# and the source build in this image failed (tried 2026-11-14), so the GDN conv uses transformers' PyTorch fallback.


def _with_code(img: modal.Image) -> modal.Image:
    for f in _TRAIN_CODE:
        img = img.add_local_file(ROOT / "train" / f, f"/opt/fh/train/{f}")
    return img


train_image = _with_code(train_base)


# ---------------------------------------------------------------------------------------------- weights
@app.function(image=dl_image, volumes={W: weights}, timeout=3 * 3600, cpu=8, memory=8192)
def download_weights(force: bool = False) -> dict:
    marker = Path(f"{MODEL_DIR}.REVISION")
    want = f"{MODEL_REPO} {MODEL_REV}"
    if not force and marker.exists() and marker.read_text().strip() == want:
        return {"status": "present", "marker": want}
    marker.unlink(missing_ok=True)
    t0 = time.time()
    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    subprocess.run(["hf", "download", MODEL_REPO, "--revision", MODEL_REV, "--local-dir", MODEL_DIR,
                    "--max-workers", "16"], check=True, env=env, stdout=subprocess.DEVNULL)
    t1 = time.time()
    subprocess.run([sys.executable, "/opt/fh/verify_weights.py", MODEL_REPO, MODEL_REV, MODEL_DIR], check=True)
    marker.write_text(want + "\n")
    weights.commit()
    return {"status": "downloaded", "download_s": round(t1 - t0), "verify_s": round(time.time() - t1)}


# ---------------------------------------------------------------------------------------------- serving
def vllm_args(lora_dir: str | None) -> list[str]:
    """Exactly eval_job.sh's flags (host/port aside: the tunnel needs 0.0.0.0)."""
    args = [MODEL_DIR, "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", "8000",
            "--max-model-len", "65536", "--gpu-memory-utilization", "0.92", "--max-num-seqs", "32",
            "--kv-cache-dtype", "fp8",
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
            "--limit-mm-per-prompt", '{"image": 64, "video": 0}']
    if lora_dir:
        args += ["--enable-lora", "--max-lora-rank", str(LORA_MAX_RANK), "--lora-modules", f"{LORA_NAME}={lora_dir}"]
    return args


def _get(url: str, key: str | None = None, timeout: float = 10) -> bytes | None:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:  # noqa: BLE001
        return None


def _load(metrics: bytes | None) -> float | None:
    if metrics is None:
        return None
    tot = 0.0
    for line in metrics.decode(errors="replace").splitlines():
        if line.startswith(("vllm:num_requests_running", "vllm:num_requests_waiting")):
            tot += float(line.rsplit(" ", 1)[1])
    return tot


@app.function(image=vllm_image, gpu=SERVE_GPU, volumes={W: weights}, secrets=[key_secret],
              timeout=SERVE_TIMEOUT_H * 3600, max_containers=10, single_use_containers=True)
def serve(tag: str, lora_sha: str | None = None, idle_min: float = 30, health_min: float = 40,
          max_hours: float = 5) -> dict:
    t_start = time.time()
    key = os.environ["QWEN_API_KEY"]
    names = [SERVED_NAME] + ([LORA_NAME] if lora_sha else [])
    rec: dict = {"tag": tag, "call_id": modal.current_function_call_id(), "phase": "starting",
                 "started_at": t_start, "lora_sha": lora_sha, "served_names": names}

    def publish(**kw):
        rec.update(kw, updated_at=time.time())
        state[f"serve:{tag}"] = rec

    publish()
    marker = Path(f"{MODEL_DIR}.REVISION")
    if not marker.exists() or marker.read_text().strip() != f"{MODEL_REPO} {MODEL_REV}":
        publish(phase="failed", error="weights missing on fh-qwen-weights (run `fh weights`)")
        return rec
    lora_dir = f"{W}/adapters/{lora_sha}" if lora_sha else None
    args = vllm_args(lora_dir)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    cache = Path("/root/.cache")
    log_path = Path("/tmp/vllm.log")
    env = {**os.environ, "VLLM_API_KEY": key, "VLLM_CACHE_ROOT": str(cache / "vllm")}
    env.pop("QWEN_API_KEY", None)
    logf = log_path.open("wb")
    proc = subprocess.Popen(["vllm", "serve", *args], env=env, stdout=logf, stderr=subprocess.STDOUT)
    publish(gpus=smi, vllm_cmd=["vllm", "serve", *args], vllm_version=VLLM_VERSION,
            model=f"{MODEL_REPO}@{MODEL_REV}",
            region=os.environ.get("MODAL_REGION"), cloud=os.environ.get("MODAL_CLOUD_PROVIDER"))
    reason = None
    with modal.forward(8000) as tunnel:
        publish(url=tunnel.url, phase="loading")
        last_busy = time.time()
        while reason is None:
            time.sleep(10)
            now = time.time()
            if proc.poll() is not None:
                reason = f"vllm exited {proc.returncode}"
            elif state.get(f"stop:{tag}"):
                reason = "stopped"
            elif now - t_start > max_hours * 3600:
                reason = "max_hours"
            elif rec["phase"] == "loading":
                body = _get("http://127.0.0.1:8000/v1/models", key)
                if body is not None and set(names) <= {m["id"] for m in json.loads(body)["data"]}:
                    last_busy = now
                    publish(phase="healthy", healthy_at=now, load_s=round(now - t_start))
                elif now - t_start > health_min * 60:
                    reason = "not healthy in time"
            else:
                load = _load(_get("http://127.0.0.1:8000/metrics"))
                if load:
                    last_busy = now
                elif now - last_busy > idle_min * 60:
                    reason = f"idle {idle_min} min"
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(60)
    except subprocess.TimeoutExpired:
        proc.kill()
    logf.close()
    tail = log_path.read_text(errors="replace").splitlines()[-150:]
    print("\n".join(tail))
    publish(phase="ended", reason=reason, ended_at=time.time(), log_tail=tail)
    return {k: v for k, v in rec.items() if k != "log_tail"}


# --------------------------------------------------------------------------------------------- training
def _train_impl(run: str, trainer_args: list[str], manifest: str) -> dict:
    out = Path(f"{T}/runs/{run}")
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "manifest.json"
    if mf.exists():
        if mf.read_text() != manifest:
            raise SystemExit(f"{mf} belongs to a different run definition")
    elif any(out.glob("ckpt-*")) or (out / "args.json").exists() or (out / "DONE").exists():
        raise SystemExit(f"{out} holds a run without manifest.json")
    else:
        mf.write_text(manifest)
    state[f"train:{run}"] = {"phase": "running", "call_id": modal.current_function_call_id(),
                             "started_at": time.time()}
    state.pop(f"trainstop:{run}", None)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    rc = 0
    if not (out / "DONE").exists():
        with (out / "train.log").open("ab") as logf:
            logf.write(f"=== modal call {modal.current_function_call_id()} on {smi} at {time.ctime()}\n".encode())
            logf.flush()
            proc = subprocess.Popen([sys.executable, "-u", "/opt/fh/train/sft_lora.py", *trainer_args],
                                    cwd="/opt/fh/train", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            stop = threading.Event()

            def watch():  # commit progress to the volume; forward a stop request as SIGTERM (checkpoint + exit)
                sent = False
                while not stop.wait(60):
                    try:
                        trainvol.commit()
                    except Exception as e:  # noqa: BLE001
                        print(f"commit failed: {e!r}", flush=True)
                    if not sent and state.get(f"trainstop:{run}"):
                        proc.send_signal(signal.SIGTERM)
                        sent = True

            th = threading.Thread(target=watch, daemon=True)
            th.start()
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
                logf.write(line)
                logf.flush()
            rc = proc.wait()
            stop.set()
            th.join()
    if (out / "DONE").exists():
        import shutil
        dst = Path(f"{T}/adapters/{run}")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(out / "final", dst)
    trainvol.commit()
    phase = "done" if (out / "DONE").exists() else ("stopped" if rc == 143 else f"exit {rc}")
    state[f"train:{run}"] = {"phase": phase, "ended_at": time.time(), "gpu": smi}
    if phase.startswith("exit"):
        raise RuntimeError(f"trainer {phase}")   # Modal retries the call (resume from the last checkpoint)
    return {"phase": phase}


_TRAIN_KW = dict(gpu=TRAIN_GPU, volumes={W: weights, T: trainvol}, timeout=TRAIN_TIMEOUT_H * 3600,
                 retries=modal.Retries(max_retries=2, initial_delay=30.0), single_use_containers=True,
                 cpu=8, memory=65536)


@train_app.function(image=train_image, **_TRAIN_KW)
def train(run: str, trainer_args: list[str], manifest: str) -> dict:
    return _train_impl(run, trainer_args, manifest)

