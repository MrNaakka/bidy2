#!/bin/bash
# serve.sh <gpu> <lora_dir> <log_dir>: GRPO rollout server on GPU g. vLLM with the eval fleet's exact flags
# (serve_gpu.sh) and the LoRA served as qwen3.8-27b-fh on 127.0.0.1:(8100+g), plus the request-logging
# relay (infra/verda/log_relay.py) on 172.17.0.1:(18100+g) logging to <log_dir>. Units fh-grpo-vllm-g and
# fh-grpo-relay-g (the eval units fh-vllm-g / fh-relay-g on 8000+g / 18000+g are never touched).
set -euo pipefail
G=$1; LORA=$2; LOGDIR=$3
FH=/root/fh; DATA=/root/fhdata; VENV=$DATA/envs/vllm-0.33.0; model=$DATA/models/Qwen3.8-27B
CODE=${GRPO_CODE:-/root/fh/grpo-code}
P=$((8100+G)); RP=$((18100+G))
[ -f "$LORA/adapter_model.safetensors" ] || { echo "no adapter in $LORA" >&2; exit 1; }
mkdir -p "$LOGDIR" $FH/grpo/logs
systemctl stop fh-grpo-vllm-$G fh-grpo-relay-$G 2>/dev/null || true
systemctl reset-failed fh-grpo-vllm-$G fh-grpo-relay-$G 2>/dev/null || true
systemd-run --unit=fh-grpo-vllm-$G --property=EnvironmentFile=$FH/vllm.env --setenv=CUDA_VISIBLE_DEVICES=$G \
  --setenv=PATH=$VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin --setenv=VLLM_CACHE_ROOT=$DATA/cache/vllm-$G \
  --setenv=TRITON_CACHE_DIR=$DATA/cache/triton-$G --setenv=FLASHINFER_WORKSPACE_BASE=$DATA/cache/flashinfer-$G \
  --setenv=VLLM_USE_FLASHINFER_SAMPLER=0 --setenv=HF_HUB_OFFLINE=1 \
  bash -c "exec $VENV/bin/vllm serve $model --served-model-name qwen3.8-27b --host 127.0.0.1 --port $P \
    --max-model-len 65536 --gpu-memory-utilization 0.92 --max-num-seqs 32 --kv-cache-dtype fp8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{\"image\": 64, \"video\": 0}' \
    --enable-lora --max-lora-rank 16 --lora-modules qwen3.8-27b-fh=$LORA > $FH/grpo/logs/vllm-$G.log 2>&1"
systemd-run --unit=fh-grpo-relay-$G --property=Restart=always --property=StartLimitIntervalSec=0 \
  --setenv=PATH=/usr/bin:/bin \
  bash -c "exec python3 $CODE/infra/verda/log_relay.py --listen 172.17.0.1:$RP --upstream 127.0.0.1:$P --log-dir $LOGDIR >> $FH/grpo/logs/relay-$G.log 2>&1"
echo "grpo serving gpu $G on $P, logging relay $RP -> $LOGDIR, lora=$LORA"
