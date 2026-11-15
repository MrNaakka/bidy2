#!/bin/bash
# rebalance.sh: move the running async run from 4 inference + 4 trainer GPUs to 6 + 2, between updates,
# without killing trials (systemd unit; ROOT, MODEL, INIT, END as for start.sh / finish.sh).
#   1. waits for the next published update (metrics line for version AFTER_V), then stops the trainer
#      (queued groups stay in queue/, the trainer resumes from adapters/LATEST);
#   2. vLLM + log-prob relay on GPUs 4 and 5 (fh-v0 = INIT; the new scheduler loads the current policy);
#   3. trainer on GPUs 6,7 (same hyperparameters);
#   4. relays checked from a container (401), then the scheduler is frozen (SIGSTOP, its Harbor jobs keep
#      running) and fh-agrpo-sched4 takes over all jobs (--adopt) on GPUs 0-5;
#   5. at END (ROOT/STOP, written by finish.sh) stops vLLM/relays 4-5 so the evals can use those GPUs.
set -uo pipefail
export HOME=/root PATH=/root/.local/bin:/usr/bin:/bin
: "${ROOT:?}" "${MODEL:?}" "${INIT:?}" "${END:?}" "${AFTER_V:?}"
CODE=${GRPO_CODE:-/root/fh/grpo-code}; FH=/root/fh
stamp() { echo "$(date -u +%FT%TZ) rebalance $*" | tee -a $ROOT/times.txt; }
TRAIN_ARGS=${TRAIN_ARGS:---min-groups 4 --max-groups 12 --max-stale 2 --lr 2e-5 --clip 0.2 --max-len 18432}
until grep -q "\"version\": $AFTER_V," $ROOT/metrics.jsonl; do sleep 2; done
until [ "$(cat $ROOT/adapters/LATEST)" -ge $AFTER_V ]; do sleep 1; done
sleep 3   # the trainer moves the used groups to consumed/ right after publishing
systemctl stop fh-agrpo-train
stamp "trainer stopped after v$AFTER_V (queue: $(ls $ROOT/queue | wc -l) groups)"
for g in 4 5; do
  $CODE/infra/verda/grpo_async/serve.sh $g $MODEL $INIT
  mkdir -p $ROOT/requests-g$g
  systemctl stop fh-agrpo-relay-$g 2>/dev/null; systemctl reset-failed fh-agrpo-relay-$g 2>/dev/null
  systemd-run --unit=fh-agrpo-relay-$g --property=Restart=always --property=StartLimitIntervalSec=0 \
    --setenv=PATH=/usr/bin:/bin bash -c "exec python3 $CODE/infra/verda/log_relay.py --listen 172.17.0.1:$((18200+g)) \
      --upstream 127.0.0.1:$((8200+g)) --log-dir $ROOT/requests-g$g --policy-file $ROOT/policy.json >> $ROOT/logs/relay-$g.log 2>&1"
done
systemctl reset-failed fh-agrpo-train 2>/dev/null
systemd-run --unit=fh-agrpo-train --setenv=CUDA_VISIBLE_DEVICES=6,7 --setenv=HF_HUB_OFFLINE=1 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --working-directory=$CODE/train \
  bash -c "exec /root/fhdata/envs/train/bin/torchrun --standalone --nproc_per_node=2 async_grpo.py --model $MODEL \
    --init-adapter $INIT --root $ROOT --deadline $END $TRAIN_ARGS >> $ROOT/logs/train.log 2>&1"
stamp "trainer restarted on 6,7"
K=$(cut -d= -f2- $FH/vllm.env)
for g in 4 5; do
  for i in $(seq 1 120); do curl -sf -m 5 -H "Authorization: Bearer $K" localhost:$((8200+g))/v1/models | grep -q fh-v0 && break
    systemctl is-active -q fh-agrpo-vllm-$g || { stamp "vllm $g died"; exit 1; }; sleep 5; done
done
docker network create fhagchk2 >/dev/null 2>&1 || true
for g in 4 5; do
  c=$(docker run --rm --network fhagchk2 curlimages/curl:8.10.1 -s -m 8 -o /dev/null -w %{http_code} http://172.17.0.1:$((18200+g))/v1/models || true)
  stamp "relay $g from a container: $c"; [ "$c" = 401 ] || { docker network rm fhagchk2 >/dev/null 2>&1; exit 1; }
done
docker network rm fhagchk2 >/dev/null 2>&1 || true
systemctl kill -s SIGSTOP --kill-whom=main fh-agrpo-sched
systemd-run --unit=fh-agrpo-sched4 --property=TimeoutStopSec=200 --setenv=HOME=/root --setenv=PATH=/root/.local/bin:/usr/bin:/bin \
  bash -c "exec /root/fhdata/envs/train/bin/python $CODE/infra/verda/grpo_async/scheduler.py --root $ROOT \
    --pool /root/fh/tasks/agrpo-pool2 --gpus 0 1 2 3 4 5 --deadline $END --inflight 24 --agent-timeout-min 8 \
    --launch-gap-s 20 --adopt >> $ROOT/logs/sched4.log 2>&1"
stamp "scheduler handed over to fh-agrpo-sched4 (gpus 0-5)"
until [ -f $ROOT/STOP ]; do sleep 10; done
for i in $(seq 1 20); do systemctl is-active -q fh-agrpo-sched4 || break; sleep 10; done
systemctl stop fh-agrpo-sched4 fh-agrpo-relay-4 fh-agrpo-relay-5 fh-agrpo-vllm-4 fh-agrpo-vllm-5
stamp "END: GPUs 4-5 freed"
