#!/bin/bash
# start.sh: the async GRPO system on the Verda box (all parts are systemd units; rerun-safe).
#
#   ROOT=/root/fhdata/agrpo/run1 MODEL=/root/fhdata/models/Qwen3.8-27B INIT=<adapter v0> POOL=/root/fh/tasks/agrpo-pool \
#   INFER_GPUS="0 1 2 3" TRAIN_GPUS=4,5,6,7 DEADLINE=2026-11-15T00:50Z infra/verda/grpo_async/start.sh
#
# Units: fh-agrpo-vllm-<g> (serve.sh: long-lived vLLM, runtime LoRA loading, fh-v0=INIT),
# fh-agrpo-relay-<g> (log_relay.py --policy-file ROOT/policy.json on 172.17.0.1:18200+g),
# fh-agrpo-sched (scheduler.py), fh-agrpo-train (torchrun train/async_grpo.py on TRAIN_GPUS).
# Extra env: SCHED_ARGS (e.g. "--synthetic --inflight 16"), TRAIN_ARGS (e.g. "--min-groups 2"),
# GMU/MAXLEN/EXTRA for serve.sh. Stop: touch ROOT/STOP (scheduler drains, trainer stops), then
# systemctl stop the units.
set -euo pipefail
: "${ROOT:?}" "${MODEL:?}" "${INIT:?}" "${POOL:?}" "${INFER_GPUS:?}" "${TRAIN_GPUS:?}"
DEADLINE=${DEADLINE:-}
CODE=${GRPO_CODE:-/root/fh/grpo-code}; PY=/root/fhdata/envs/train/bin/python; FH=/root/fh
mkdir -p $ROOT/logs $ROOT/adapters
[ -f $ROOT/policy.json ] || echo '{"version": 0, "name": "fh-v0"}' > $ROOT/policy.json
for g in $INFER_GPUS; do
  $CODE/infra/verda/grpo_async/serve.sh $g $MODEL $INIT
  mkdir -p $ROOT/requests-g$g
  systemctl stop fh-agrpo-relay-$g 2>/dev/null || true; systemctl reset-failed fh-agrpo-relay-$g 2>/dev/null || true
  systemd-run --unit=fh-agrpo-relay-$g --property=Restart=always --property=StartLimitIntervalSec=0 \
    --setenv=PATH=/usr/bin:/bin bash -c "exec python3 $CODE/infra/verda/log_relay.py --listen 172.17.0.1:$((18200+g)) \
      --upstream 127.0.0.1:$((8200+g)) --log-dir $ROOT/requests-g$g --policy-file $ROOT/policy.json >> $ROOT/logs/relay-$g.log 2>&1"
done
K=$(cut -d= -f2- $FH/vllm.env)
for g in $INFER_GPUS; do
  for i in $(seq 1 120); do curl -sf -m 5 -H "Authorization: Bearer $K" localhost:$((8200+g))/v1/models | grep -q fh-v0 && break
    systemctl is-active -q fh-agrpo-vllm-$g || { echo "vllm $g died"; exit 1; }; sleep 5; done
done
docker network create fhagchk >/dev/null 2>&1 || true
for g in $INFER_GPUS; do
  c=$(docker run --rm --network fhagchk curlimages/curl:8.10.1 -s -m 8 -o /dev/null -w %{http_code} http://172.17.0.1:$((18200+g))/v1/models || true)
  echo "relay $g from a container: $c"; [ "$c" = 401 ] || { docker network rm fhagchk >/dev/null 2>&1; exit 1; }
done
docker network rm fhagchk >/dev/null 2>&1 || true
nproc=$(echo $TRAIN_GPUS | tr , '\n' | wc -l)
for u in fh-agrpo-sched fh-agrpo-train; do systemctl stop $u 2>/dev/null || true; systemctl reset-failed $u 2>/dev/null || true; done
systemd-run --unit=fh-agrpo-train --setenv=CUDA_VISIBLE_DEVICES=$TRAIN_GPUS --setenv=HF_HUB_OFFLINE=1 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --working-directory=$CODE/train \
  bash -c "exec /root/fhdata/envs/train/bin/torchrun --standalone --nproc_per_node=$nproc async_grpo.py --model $MODEL \
    --init-adapter $INIT --root $ROOT ${DEADLINE:+--deadline $DEADLINE} ${TRAIN_ARGS:-} >> $ROOT/logs/train.log 2>&1"
systemd-run --unit=fh-agrpo-sched --property=TimeoutStopSec=200 --setenv=HOME=/root --setenv=PATH=/root/.local/bin:/usr/bin:/bin \
  bash -c "exec $PY $CODE/infra/verda/grpo_async/scheduler.py --root $ROOT --pool $POOL --gpus $INFER_GPUS \
    ${DEADLINE:+--deadline $DEADLINE} ${SCHED_ARGS:-} >> $ROOT/logs/sched.log 2>&1"
echo "async GRPO started: infer $INFER_GPUS, train $TRAIN_GPUS, root $ROOT"
