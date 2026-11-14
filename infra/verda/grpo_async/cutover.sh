#!/bin/bash
# cutover.sh: sync GRPO loop -> async GRPO -> final evals, unattended (systemd unit fh-agrpo-orch).
#
#   AFTER=it2 END=2026-11-15T00:40Z infra/verda/grpo_async/cutover.sh
#
# 1. waits until the sync loop logs "<AFTER> iteration done" (its adapter is saved), touches the loop's STOP
#    file (iterate.sh refuses to start the next iteration, the loop exits without its eval) and frees the GPUs;
# 2. starts the async system (start.sh) from that adapter: inference on INFER_GPUS, trainer on TRAIN_GPUS;
# 3. at END: STOP for the trainer (it finishes an update in progress), SIGTERM to the scheduler (Harbor jobs
#    interrupted), all async units stopped;
# 4. evaluates the newest async adapter (fh-eval-qwen-grpo-run{1,2,3} on GPUs 0-2) and, alongside, the
#    sync adapter it started from (fh-eval-qwen-grpo-sync-run{1,2,3} on GPUs 3-5).
set -uo pipefail
export HOME=/root PATH=/root/.local/bin:/usr/bin:/bin
CODE=${GRPO_CODE:-/root/fh/grpo-code}; SYNC=/root/fhdata/grpo
AFTER=${AFTER:-it2}; END=${END:-2026-11-15T00:40Z}
ROOT=${ROOT:-/root/fhdata/agrpo/run1}; INFER_GPUS=${INFER_GPUS:-"0 1 2 3"}; TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7}
stamp() { mkdir -p $ROOT; echo "$(date -u +%FT%TZ) cutover $*" | tee -a $ROOT/times.txt; }

until grep -q "^.* $AFTER iteration done" $SYNC/times.txt 2>/dev/null; do
  systemctl is-active -q fh-grpo-loop || { stamp "sync loop is not running and $AFTER never finished"; exit 1; }
  sleep 15
done
touch $SYNC/STOP
INIT=$SYNC/$AFTER/train/final
stamp "$AFTER done; sync loop told to stop; init $INIT"
for i in $(seq 1 40); do systemctl is-active -q fh-grpo-loop || break; sleep 3; done
systemctl stop fh-grpo-loop 2>/dev/null
for u in $(systemctl list-units 'fh-grpo-vllm-*' 'fh-grpo-relay-*' 'fh-agrpo-*' --state=active --no-legend --plain | awk '{print $1}'); do
  [ "$u" = fh-agrpo-orch.service ] && continue
  systemctl stop $u
done
sleep 5
stamp "GPUs free; starting async"
ROOT=$ROOT MODEL=/root/fhdata/models/Qwen3.8-27B INIT=$INIT POOL=/root/fh/tasks/agrpo-pool INFER_GPUS="$INFER_GPUS" \
  TRAIN_GPUS=$TRAIN_GPUS DEADLINE=$END SCHED_ARGS="${SCHED_ARGS:---inflight 24 --agent-timeout-min 8}" \
  TRAIN_ARGS="${TRAIN_ARGS:---min-groups 4 --max-groups 12 --max-stale 2 --lr 2e-5 --clip 0.2 --max-len 24576}" \
  $CODE/infra/verda/grpo_async/start.sh >> $ROOT/times.txt 2>&1 || { stamp "start failed"; exit 1; }
stamp "async running"

while [ $(date +%s) -lt $(date -d $END +%s) ]; do
  systemctl is-active -q fh-agrpo-train || { stamp "trainer not active"; break; }
  sleep 30
done
touch $ROOT/STOP
stamp "END: stopping"
for i in $(seq 1 60); do systemctl is-active -q fh-agrpo-train || break; sleep 10; done   # an update may be running
systemctl stop fh-agrpo-train fh-agrpo-sched
for g in $INFER_GPUS; do systemctl stop fh-agrpo-relay-$g fh-agrpo-vllm-$g; done
N=$(cat $ROOT/adapters/LATEST)
FINAL=$ROOT/adapters/v$N
echo "$FINAL" > $ROOT/final_adapter.txt
stamp "async stopped at v$N; evals"
$CODE/infra/verda/grpo/eval.sh $FINAL grpo "0 1 2" >> $ROOT/eval.log 2>&1 &
sleep 20
$CODE/infra/verda/grpo/eval.sh $INIT grpo-sync "3 4 5" >> $ROOT/eval-sync.log 2>&1 &
wait
stamp "evals done"
python3 $CODE/infra/verda/grpo/summarize_evals.py /root/fh/runs fh-eval-qwen-fh3 fh-eval-qwen-grpo-sync fh-eval-qwen-grpo \
  | tee -a $ROOT/times.txt
