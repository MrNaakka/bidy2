#!/bin/bash
# loop.sh: the whole GRPO stage on the Verda box, unattended (systemd unit fh-grpo-loop; rerun resumes).
#
#   ITERS=4 START=/root/fhdata/runs/sft-v3/final DEADLINE=2026-11-15T01:30Z infra/verda/grpo/loop.sh
#
# Waits until no eval unit (fh-h-*) is active, frees the GPUs (stops the eval fleet's vLLM/socat units
# fh-vllm-* / fh-relay-*; their results in /root/fh/runs are not touched), then runs iterate.sh for
# iterations 1..ITERS on tasks/grpo-it<k> (fresh TRAIN items every iteration, offset 300 + 10 (k-1)),
# each starting from the previous iteration's adapter. An iteration is not started if it could not end
# (with the final eval) before DEADLINE. Then the last adapter is evaluated like every other eval:
# eval.sh (40 EVAL items, 3 runs, GPUs 0-2, socat relays, cfg-ft-g*.json, -n 15).
set -uo pipefail
export HOME=/root PATH=/root/.local/bin:/usr/bin:/bin
FH=/root/fh; CODE=${GRPO_CODE:-$FH/grpo-code}; OUT=${GRPO_OUT:-/root/fhdata/grpo}
ITERS=${ITERS:-4}; START=${START:-/root/fhdata/runs/sft-v3/final}; DEADLINE=${DEADLINE:-2026-11-15T01:30Z}
ITER_MIN=${ITER_MIN:-45}; EVAL_MIN=${EVAL_MIN:-35}
mkdir -p $OUT; TIMES=$OUT/times.txt
stamp() { echo "$(date -u +%FT%TZ) loop $*" | tee -a $TIMES; }

while [ -n "$(systemctl list-units 'fh-h-*' --state=active --no-legend --plain | grep -v '^$')" ]; do sleep 30; done
stamp "no eval unit active; freeing GPUs"
for u in $(systemctl list-units 'fh-vllm-*' 'fh-relay-*' --state=active --no-legend --plain | awk '{print $1}'); do
  systemctl stop $u; stamp "stopped $u"
done
sleep 10

init=$START
for k in $(seq 1 $ITERS); do
  D=$OUT/it$k
  if [ ! -f $D/train.done ]; then
    left=$(( ($(date -d $DEADLINE +%s) - $(date +%s)) / 60 ))
    if [ $left -lt $((ITER_MIN + EVAL_MIN)) ]; then stamp "stop before it$k: $left min to the deadline"; break; fi
    ITER=$k INIT=$init TASKS=$FH/tasks/grpo-it$k $CODE/infra/verda/grpo/iterate.sh >> $OUT/it$k.log 2>&1 \
      || { stamp "it$k failed (see $OUT/it$k.log)"; exit 1; }
  fi
  init=$D/train/final
done
echo "$init" > $OUT/final_adapter.txt
stamp "final adapter $init; eval"
$CODE/infra/verda/grpo/eval.sh "$init" grpo >> $OUT/eval.log 2>&1
stamp "eval done"
