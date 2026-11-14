#!/bin/bash
# eval.sh <adapter> <tag>: evaluate an adapter exactly like the fine-tuned evals (ft3_chain.sh): served on
# GPUs 0-2 by serve_gpu.sh (fleet flags, plain socat relay on 172.17.0.1:1800g), all 40 EVAL items
# (tasks/eval40), 3 runs fh-eval-qwen-<tag>-run{1,2,3}, one per GPU, -n 15, cfg-ft-g{0,1,2}.json.
set -uo pipefail
export HOME=/root PATH=/root/.local/bin:/usr/bin:/bin; cd /root/fh
A=$1; TAG=$2
[ -f $A/adapter_model.safetensors ] || { echo "no adapter in $A"; exit 1; }
[ -d tasks/eval40 ] || { echo "no tasks/eval40"; exit 1; }
for g in 0 1 2; do ./serve_gpu.sh $g $A; done
K=$(cut -d= -f2- vllm.env)
for g in 0 1 2; do for i in $(seq 1 120); do curl -sf -m 5 -H "Authorization: Bearer $K" localhost:$((8000+g))/v1/models | grep -q qwen3.8-27b-fh && break; sleep 10; done; done
docker network create fhgrpoev >/dev/null 2>&1
for g in 0 1 2; do c=$(docker run --rm --network fhgrpoev curlimages/curl:8.10.1 -s -m 8 -o /dev/null -w %{http_code} http://172.17.0.1:$((18000+g))/v1/models); echo "$TAG relay $g: $c" >> logs/times.txt; done
docker network rm fhgrpoev >/dev/null 2>&1
for i in 1 2 3; do g=$((i-1))
  ( t0=$SECONDS; harbor run -p tasks/eval40 --job-name fh-eval-qwen-$TAG-run$i -o runs -n 15 -r 2 --retry-include AgentSetupTimeoutError -y -c cfg-ft-g$g.json --env-file key.env > logs/fh-eval-qwen-$TAG-run$i.log 2>&1; echo "fh-eval-qwen-$TAG-run$i $((SECONDS-t0))s" >> logs/times.txt ) &
  sleep 10; done; wait
