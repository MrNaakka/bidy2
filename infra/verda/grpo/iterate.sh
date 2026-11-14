#!/bin/bash
# iterate.sh: one GRPO iteration on the Verda box (run as a systemd unit; rerunning resumes by phase).
#
#   ITER=1 INIT=/root/fhdata/runs/sft-v3/final TASKS=/root/fh/tasks/grpo-it1 \
#   ROLLOUT_GPUS="0 1 2 3 4 5 6 7" TRAIN_GPUS="0,1,2,3,4,5,6,7" K=8 N=20 STEPS=8 LR=2e-5 \
#       infra/verda/grpo/iterate.sh
#
# 1. rollouts: serve INIT (vLLM LoRA qwen3.8-27b-fh) on every rollout GPU behind the request-logging relay
#    (grpo/serve.sh); the TASKS items are dealt round-robin to the GPUs; one Harbor job per GPU with
#    -k K attempts per item and -n N concurrent trials, the eval's agent config (OpenCode 1.19.5,
#    cfg-ft-g*.json) pointed at that GPU's relay.
# 2. convert: every rollout -> rows with group-normalized advantages (train/rollouts_to_pg.py).
# 3. update: torchrun sft_lora.py --pg --init-adapter INIT on TRAIN_GPUS, one pass, STEPS optimizer steps.
# Output: $OUT/it$ITER/{runs,requests-g*,data,train/final}; curve line in $OUT/curve.jsonl.
set -euo pipefail
export HOME=/root PATH=/root/.local/bin:/usr/bin:/bin HF_HUB_OFFLINE=1
: "${ITER:?}" "${INIT:?}" "${TASKS:?}"
ROLLOUT_GPUS=${ROLLOUT_GPUS:-"0 1 2 3 4 5 6 7"}; TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3,4,5,6,7}
K=${K:-8}; N=${N:-20}; STEPS=${STEPS:-8}; LR=${LR:-2e-5}; MAXLEN=${MAXLEN:-32768}
FH=/root/fh; CODE=${GRPO_CODE:-$FH/grpo-code}; OUT=${GRPO_OUT:-/root/fhdata/grpo}; NAME=${NAME:-it$ITER}
MODEL=/root/fhdata/models/Qwen3.8-27B; PY=/root/fhdata/envs/train/bin/python
D=$OUT/$NAME; mkdir -p $D/runs $D/tasks; TIMES=$OUT/times.txt
stamp() { echo "$(date -u +%FT%TZ) $NAME $*" | tee -a $TIMES; }
cd $FH
K_API=$(cut -d= -f2- vllm.env)

# ------------------------------------------------------------------ 1. rollouts
if [ ! -f $D/rollout.done ]; then
  stamp "rollouts start (gpus $ROLLOUT_GPUS, K=$K, N=$N, init=$INIT)"
  gpus=($ROLLOUT_GPUS); ng=${#gpus[@]}
  i=0
  for g in "${gpus[@]}"; do rm -rf $D/tasks/g$g; mkdir -p $D/tasks/g$g; done
  for t in $(ls $TASKS | sort); do g=${gpus[$((i % ng))]}; cp -r $TASKS/$t $D/tasks/g$g/; i=$((i+1)); done
  for g in "${gpus[@]}"; do
    $CODE/infra/verda/grpo/serve.sh $g $INIT $D/requests-g$g
    python3 - $FH/cfg-ft-g0.json $D/cfg-g$g.json $((18100+g)) <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
c["agents"][0]["base_url"] = f"http://172.17.0.1:{sys.argv[3]}/v1"
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
  done
  for g in "${gpus[@]}"; do
    ok=0
    for j in $(seq 1 90); do
      if curl -sf -m 5 -H "Authorization: Bearer $K_API" localhost:$((8100+g))/v1/models | grep -q qwen3.8-27b-fh; then ok=1; break; fi
      systemctl is-active -q fh-grpo-vllm-$g || { stamp "vllm $g died"; exit 1; }
      sleep 10
    done
    [ $ok = 1 ] || { stamp "vllm $g not healthy"; exit 1; }
  done
  docker network create fhgrpochk >/dev/null 2>&1 || true
  for g in "${gpus[@]}"; do
    c=$(docker run --rm --network fhgrpochk curlimages/curl:8.10.1 -s -m 8 -o /dev/null -w %{http_code} http://172.17.0.1:$((18100+g))/v1/models || true)
    stamp "relay $g from a container: $c"
    [ "$c" = 401 ] || { docker network rm fhgrpochk >/dev/null 2>&1; exit 1; }
  done
  docker network rm fhgrpochk >/dev/null 2>&1 || true
  stamp "servers up"
  pids=()
  for g in "${gpus[@]}"; do
    job=fh-grpo-$NAME-g$g
    rm -rf $D/runs/$job
    harbor run -p $D/tasks/g$g --job-name $job -o $D/runs -k $K -n $N -r 2 --retry-include AgentSetupTimeoutError \
      -y -c $D/cfg-g$g.json --env-file $FH/key.env > $D/harbor-g$g.log 2>&1 &
    pids+=($!)
    sleep 5
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  stamp "rollouts harbor done (rc $rc)"
  for g in "${gpus[@]}"; do systemctl stop fh-grpo-relay-$g fh-grpo-vllm-$g || true; done
  for g in "${gpus[@]}"; do ln -sfn $D/requests-g$g $D/runs/fh-grpo-$NAME-g$g/requests; done
  touch $D/rollout.done
  stamp "rollouts done"
fi

# ------------------------------------------------------------------ 2. convert
if [ ! -f $D/convert.done ]; then
  stamp "convert start"
  $PY $CODE/train/rollouts_to_pg.py $D/runs/fh-grpo-$NAME-g* --out $D/data --tasks $TASKS --processor $MODEL \
    --force > $D/convert.log 2>&1
  touch $D/convert.done
  stamp "convert done: $(python3 -c "import json;s=json.load(open('$D/data/pg_stats.json'));print({k:s[k] for k in ('trials','counted_trials','mean_reward','groups','groups_with_signal','rows','rows_pos','rows_neg')})")"
fi

# ------------------------------------------------------------------ 3. update
if [ ! -f $D/train.done ]; then
  for g in ${TRAIN_GPUS//,/ }; do
    systemctl is-active -q fh-grpo-vllm-$g && systemctl stop fh-grpo-vllm-$g
  done
  nrows=$(wc -l < $D/data/sft.jsonl)
  [ "$nrows" -gt 0 ] || { stamp "no training rows"; exit 1; }
  ga=$(( nrows / STEPS )); [ $ga -ge 1 ] || ga=1
  nproc=$(echo $TRAIN_GPUS | tr , '\n' | wc -l)
  stamp "update start ($nrows rows, grad_accum $ga, lr $LR, gpus $TRAIN_GPUS)"
  cd $CODE/train
  CUDA_VISIBLE_DEVICES=$TRAIN_GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /root/fhdata/envs/train/bin/torchrun --standalone --nproc_per_node=$nproc sft_lora.py --pg \
    --model $MODEL --data $D/data --out $D/train --init-adapter $INIT --epochs 1.0 --lr $LR --warmup-frac 0 \
    --grad-accum $ga --max-grad-norm 1.0 --weight-decay 0.0 --lora-r 16 --lora-alpha 32 --lora-dropout 0 \
    --dtype bf16 --attn sdpa --max-len $MAXLEN --loss-chunk 1024 --eval-frac 0 --save-every 1000 --keep 1 \
    --seed $ITER > $D/train.log 2>&1
  cd $FH
  [ -f $D/train/final/adapter_model.safetensors ] || { stamp "no final adapter"; exit 1; }
  touch $D/train.done
  stamp "update done"
fi

python3 - $D $OUT/curve.jsonl $NAME <<'PY'
import json, sys
d, curve, name = sys.argv[1:]
s = json.load(open(f"{d}/data/pg_stats.json"))
steps = [json.loads(l) for l in open(f"{d}/train/loss.jsonl") if l.strip()]
rec = {"iter": name, "mean_reward": s["mean_reward"], "counted_trials": s["counted_trials"],
       "by_kind": {k: v["mean_reward"] for k, v in s["by_kind"].items()},
       "solved_any": {k: v["solved_any"] for k, v in s["by_kind"].items()},
       "groups": s["groups"], "groups_with_signal": s["groups_with_signal"], "rows": s["rows"],
       "rows_pos": s["rows_pos"], "rows_neg": s["rows_neg"], "rejected": s["rejected"],
       "steps": len(steps), "loss": [x["loss"] for x in steps], "grad_norm": [x["grad_norm"] for x in steps],
       "nll_pos": [x.get("nll_pos") for x in steps], "nll_neg": [x.get("nll_neg") for x in steps]}
lines = [l for l in open(curve)] if __import__("os").path.exists(curve) else []
lines = [l for l in lines if json.loads(l)["iter"] != name] + [json.dumps(rec) + "\n"]
open(curve, "w").writelines(lines)
print(json.dumps(rec))
PY
stamp "iteration done: adapter $D/train/final"
