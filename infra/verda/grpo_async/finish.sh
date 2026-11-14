#!/bin/bash
# finish.sh: at END stop the async run and evaluate (the tail of cutover.sh, for a restarted async run).
#   ROOT=/root/fhdata/agrpo/run1 END=2026-11-15T00:35Z SYNC_ADAPTER=/root/fhdata/grpo/it2/train/final finish.sh
# Waits until END even if the trainer stops early (it is logged, never a reason to start the evals early).
set -uo pipefail
export HOME=/root PATH=/root/.local/bin:/usr/bin:/bin
CODE=${GRPO_CODE:-/root/fh/grpo-code}
: "${ROOT:?}" "${END:?}" "${SYNC_ADAPTER:?}"
INFER_GPUS=${INFER_GPUS:-"0 1 2 3"}
stamp() { echo "$(date -u +%FT%TZ) finish $*" | tee -a $ROOT/times.txt; }
warned=0
while [ $(date +%s) -lt $(date -d $END +%s) ]; do
  if ! systemctl is-active -q fh-agrpo-train && [ $warned = 0 ]; then stamp "trainer not active before END"; warned=1; fi
  sleep 30
done
touch $ROOT/STOP
stamp "END: stopping"
for i in $(seq 1 60); do systemctl is-active -q fh-agrpo-train || break; sleep 10; done   # an update may be running
for i in $(seq 1 20); do systemctl is-active -q fh-agrpo-sched || break; sleep 10; done   # interrupts its jobs
systemctl stop fh-agrpo-train fh-agrpo-sched
for g in $INFER_GPUS; do systemctl stop fh-agrpo-relay-$g fh-agrpo-vllm-$g; done
N=$(cat $ROOT/adapters/LATEST)
FINAL=$ROOT/adapters/v$N
echo "$FINAL" > $ROOT/final_adapter.txt
stamp "async stopped at v$N; evals"
$CODE/infra/verda/grpo/eval.sh $FINAL grpo "0 1 2" >> $ROOT/eval.log 2>&1 &
sleep 20
$CODE/infra/verda/grpo/eval.sh $SYNC_ADAPTER grpo-sync "3 4 5" >> $ROOT/eval-sync.log 2>&1 &
wait
stamp "evals done"
python3 $CODE/infra/verda/grpo/summarize_evals.py /root/fh/runs fh-eval-qwen-fh3 fh-eval-qwen-grpo-sync fh-eval-qwen-grpo \
  | tee -a $ROOT/times.txt
