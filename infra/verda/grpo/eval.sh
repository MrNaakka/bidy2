#!/bin/bash
# eval.sh <adapter> <tag> [gpus]: evaluate an adapter exactly like the fine-tuned evals (ft3_chain.sh): served
# by serve_gpu.sh (fleet flags, plain socat relay on 172.17.0.1:1800g), all 40 EVAL items (tasks/eval40),
# one run per GPU fh-eval-qwen-<tag>-run{1,2,3} (default GPUs 0 1 2), -n 15, cfg-ft-g<g>.json (written from
# cfg-ft-g0.json with the GPU's relay port if missing).
set -uo pipefail
export HOME=/root PATH=/root/.local/bin:/usr/bin:/bin; cd /root/fh
A=$1; TAG=$2; GPUS=${3:-"0 1 2"}
[ -f $A/adapter_model.safetensors ] || { echo "no adapter in $A"; exit 1; }
TASKS=${EVAL_TASKS:-tasks/eval40}   # EVAL_TASKS=tasks/eval30: the scored kinds only (no toasts)
[ -d $TASKS ] || { echo "no $TASKS"; exit 1; }
for g in $GPUS; do
  [ -f cfg-ft-g$g.json ] || python3 -c "
import json; c = json.load(open('cfg-ft-g0.json')); c['agents'][0]['base_url'] = 'http://172.17.0.1:$((18000+g))/v1'
json.dump(c, open('cfg-ft-g$g.json', 'w'), indent=1)"
  ./serve_gpu.sh $g $A
done
K=$(cut -d= -f2- vllm.env)
for g in $GPUS; do for i in $(seq 1 120); do curl -sf -m 5 -H "Authorization: Bearer $K" localhost:$((8000+g))/v1/models | grep -q qwen3.8-27b-fh && break; sleep 10; done; done
docker network create fhgrpoev-$TAG >/dev/null 2>&1
for g in $GPUS; do c=$(docker run --rm --network fhgrpoev-$TAG curlimages/curl:8.10.1 -s -m 8 -o /dev/null -w %{http_code} http://172.17.0.1:$((18000+g))/v1/models); echo "$TAG relay $g: $c" >> logs/times.txt; done
docker network rm fhgrpoev-$TAG >/dev/null 2>&1
i=0
for g in $GPUS; do i=$((i+1))
  ( t0=$SECONDS; harbor run -p $TASKS --job-name fh-eval-qwen-$TAG-run$i -o runs -n 15 -r 2 --retry-include AgentSetupTimeoutError -y -c cfg-ft-g$g.json --env-file key.env > logs/fh-eval-qwen-$TAG-run$i.log 2>&1; echo "fh-eval-qwen-$TAG-run$i $((SECONDS-t0))s" >> logs/times.txt ) &
  sleep 10; done; wait
