#!/bin/bash
# serve.sh <gpu> <model_dir> <lora_v0_dir>: long-lived async-GRPO inference server on GPU g.
# vLLM with the eval fleet's flags (serve_gpu.sh) plus runtime LoRA updating (new policy versions are
# hot-loaded as fh-v<N> via /v1/load_lora_adapter, never a restart) and up to 3 adapters per batch.
# Starts with fh-v0 = <lora_v0_dir>. 127.0.0.1:(8200+g); unit fh-agrpo-vllm-g.
set -euo pipefail
G=$1; MODEL=$2; LORA=$3
FH=/root/fh; DATA=/root/fhdata; VENV=$DATA/envs/vllm-0.33.0
P=$((8200+G)); MAXLEN=${MAXLEN:-65536}; GMU=${GMU:-0.92}; EXTRA=${EXTRA:-}
mkdir -p $FH/agrpo/logs
systemctl stop fh-agrpo-vllm-$G 2>/dev/null || true
systemctl reset-failed fh-agrpo-vllm-$G 2>/dev/null || true
systemd-run --unit=fh-agrpo-vllm-$G --property=EnvironmentFile=$FH/vllm.env --setenv=CUDA_VISIBLE_DEVICES=$G \
  --setenv=PATH=$VENV/bin:/usr/local/cuda/bin:/usr/bin:/bin --setenv=VLLM_CACHE_ROOT=$DATA/cache/vllm-$G \
  --setenv=TRITON_CACHE_DIR=$DATA/cache/triton-$G --setenv=FLASHINFER_WORKSPACE_BASE=$DATA/cache/flashinfer-$G \
  --setenv=VLLM_USE_FLASHINFER_SAMPLER=0 --setenv=HF_HUB_OFFLINE=1 --setenv=VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
  bash -c "exec $VENV/bin/vllm serve $MODEL --served-model-name qwen3.8-27b --host 127.0.0.1 --port $P \
    --max-model-len $MAXLEN --gpu-memory-utilization $GMU --max-num-seqs 32 --kv-cache-dtype fp8 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{\"image\": 64, \"video\": 0}' \
    --enable-lora --max-lora-rank 16 --max-loras 3 --max-cpu-loras 6 --lora-modules fh-v0=$LORA $EXTRA \
    > $FH/agrpo/logs/vllm-$G.log 2>&1"
echo "async-grpo vLLM gpu $G on $P, fh-v0=$LORA"
