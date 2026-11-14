#!/usr/bin/env bash
# The job of eval.yaml (one eval run on this box): prepare, serve Qwen with vLLM, run Harbor.
# eval.yaml runs it under `timeout` (MAX_RUN_HOURS) in its own process group; env vars come from eval.yaml.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PATH="$HOME/.local/bin:$PATH" HF_HUB_DISABLE_TELEMETRY=1
FH=~/fh; IN=$FH/in; ST=$FH/status
mkdir -p "$ST" "$FH/runs" "$FH/models"
deadline=$(( $(date +%s) + MAX_RUN_HOURS * 3600 - 300 ))   # Harbor stops before the outer limit hits
rm -f "$ST/timings.txt" "$ST/done.json"
mark() { echo "$1 $(date +%s)" >> "$ST/timings.txt"; }
mark job_start
vllm_pid=; socat_pid=
# Nothing may outlive the job: SkyPilot waits for the job's output to close, and `timeout` (our parent, whose
# process group this is) stops watching once this shell exits. So on every exit, stop the whole group but
# `timeout` and this shell. Nested timeouts use --foreground to stay in the group.
cleanup() {
  local pg p sig
  pg=$(ps -o pgid= -p $$ | tr -d ' ')
  for sig in TERM KILL; do
    for p in $(pgrep -g "$pg"); do [ "$p" = $$ ] || [ "$p" = "$PPID" ] || kill "-$sig" "$p" 2>/dev/null || true; done
    [ "$sig" = KILL ] || sleep 5
  done
  true
}
trap cleanup EXIT
# The key goes to curl in a private header file, never on a command line.
( umask 077; printf 'Authorization: Bearer %s\n' "$QWEN_API_KEY" > "$FH/auth.hdr" )

# --- prepare: packages, Harbor, vLLM venv, weights (all idempotent) ---
if ! { command -v socat && command -v rsync && docker buildx version && docker compose version; } >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y -qq docker.io docker-compose-v2 docker-buildx socat rsync >/dev/null
fi
systemctl start docker
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install -q "harbor==$HARBOR_VERSION"
model=$FH/models/${MODEL_REPO#*/}
(
  if [ "$(cat "$model.REVISION" 2>/dev/null)" != "$MODEL_REPO $MODEL_REV" ]; then
    rm -f "$model.REVISION"
    HF_HUB_ENABLE_HF_TRANSFER=1 timeout --foreground 90m uvx -q --python 3.12 --from 'huggingface_hub[cli,hf_transfer]' \
      hf download "$MODEL_REPO" --revision "$MODEL_REV" --local-dir "$model" --max-workers 16 >/dev/null
    uv run -q --no-project --python 3.12 python "$IN/verify_weights.py" "$MODEL_REPO" "$MODEL_REV" "$model"
    echo "$MODEL_REPO $MODEL_REV" > "$model.REVISION"
  fi
) &
dl=$!
VENV=$FH/vllm-$VLLM_VERSION
if [ ! -x "$VENV/bin/vllm" ]; then
  rm -rf "$VENV"
  uv venv -q --managed-python --python 3.12 "$VENV"   # system python lacks Python.h, which Triton compiles against
  VIRTUAL_ENV="$VENV" timeout --foreground 30m uv pip install -q "vllm==$VLLM_VERSION"
fi
mark venv_ready
wait "$dl"
mark weights_ready

# --- serve: identical flags on every box; LoRA adds the adapter ---
args=(--served-model-name "$SERVED_NAME" --host 127.0.0.1 --port 8000
      --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 32 --kv-cache-dtype fp8
      --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
      --limit-mm-per-prompt '{"image": 64, "video": 0}')
names="$SERVED_NAME"
if [ "$LORA" = 1 ]; then
  args+=(--enable-lora --max-lora-rank "$LORA_MAX_RANK" --lora-modules "$LORA_NAME=$IN/lora")
  names="$names $LORA_NAME"
fi
pkill -f '[v]llm serve' || true
pkill -f '[s]ocat TCP-LISTEN:18000' || true
export VLLM_API_KEY="$QWEN_API_KEY" HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_HOME=/usr/local/cuda PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"
vllm serve "$model" "${args[@]}" > "$ST/vllm.log" 2>&1 &
vllm_pid=$!
# Trial containers reach vLLM through the Docker bridge only (http://172.17.0.1:18000/v1).
socat TCP-LISTEN:18000,bind=172.17.0.1,fork,reuseaddr TCP:127.0.0.1:8000 &
socat_pid=$!
python3 - "$model" "${args[@]}" > "$ST/box.json" <<'PY'
import json, os, socket, subprocess, sys
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip().splitlines()
print(json.dumps({"hostname": socket.gethostname(), "gpus": gpu, "vllm_cmd": ["vllm", "serve", *sys.argv[1:]],
                  "cluster_info": json.loads(os.environ.get("SKYPILOT_CLUSTER_INFO") or "{}")}))
PY
served() {  # every name in $names is listed by /v1/models
  curl -sf -m 10 -H @"$FH/auth.hdr" http://127.0.0.1:8000/v1/models | python3 -c \
    'import json, sys; ids = {m["id"] for m in json.load(sys.stdin)["data"]}; sys.exit(not set(sys.argv[1:]) <= ids)' $names 2>/dev/null
}
health_deadline=$(( $(date +%s) + HEALTH_TIMEOUT_MIN * 60 ))
until served; do
  if ! kill -0 "$vllm_pid" 2>/dev/null || [ "$(date +%s)" -gt "$health_deadline" ]; then
    echo "vLLM did not become healthy" >&2; tail -n 80 "$ST/vllm.log" >&2; exit 1
  fi
  sleep 10
done
mark vllm_healthy
# the relay as a trial container sees it (root in the container, to read the 0600 header file)
timeout --foreground 300 docker run --rm --user 0:0 -v "$FH/auth.hdr:/auth.hdr:ro" curlimages/curl:8.10.1 -sf -m 10 -o /dev/null \
  -H @/auth.hdr http://172.17.0.1:18000/v1/models

# --- Harbor ---
( umask 077; printf 'QWEN_API_KEY=%s\n' "$QWEN_API_KEY" > "$FH/key.env" )
cd "$FH"
rm -rf "runs/$JOB"
left=$(( deadline - $(date +%s) ))
[ "$left" -gt 600 ] || { echo "MAX_RUN_HOURS used up before Harbor started" >&2; exit 1; }
mark harbor_start
rc=0
timeout --foreground --signal=INT --kill-after=240 "$left" harbor run -p "$IN/tasks" --job-name "$JOB" -o runs \
  -n "$N_CONCURRENT" -r 2 --retry-include AgentSetupTimeoutError -y -c "$IN/cfg.json" --env-file "$FH/key.env" || rc=$?
mark harbor_end
curl -s -m 30 http://127.0.0.1:8000/metrics > "$ST/metrics.txt" || true
printf '{"exit": %d, "finished_at": %d}\n' "$rc" "$(date +%s)" > "$ST/done.json"
exit "$rc"   # fh eval-fetch copies runs/$JOB and ~/fh/status (box.json, timings, vLLM log and metrics)
