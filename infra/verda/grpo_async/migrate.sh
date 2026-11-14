#!/bin/bash
# migrate.sh: preserve the old Verda box's GRPO results here, and continue async GRPO on a fresh box.
# Run from this host. Every step is idempotent (rsync, marker files, "already done" checks).
#
#   migrate.sh fetch            old box -> this host (BEFORE the old box is deleted):
#                               eval runs -> $ST/box-runs/, adapters + curves + run state -> $ST/box-adapters/,
#                               tasks + box config (secrets, 0600) -> $ST/box-state/
#   migrate.sh setup            this host -> new box (IP from $ST/swap/box.json): docker daemon.json (only on a
#                               box without containers), packages, uv, Harbor 0.25.0, verified weights (HF, pinned
#                               revision), vLLM 0.33.0 venv, train venv, code, tasks, prebuilt images, run state
#   migrate.sh start [END]      new box: async GRPO from the latest policy (6 inference + 2 trainer GPUs) until
#                               END (UTC ISO, default 2026-11-15T02:20Z), then the final adapter's eval
#                               (fh-eval-qwen-grpo2-run1..3, finish.sh); `migrate.sh fetch-new` brings it here
#   migrate.sh status           new box: units, GPUs, latest version
#
# Env: OLD_SSH (default reserve/ssh.sh), NEW_IP (overrides box.json), ST (default ~/.local/state/fh).
set -euo pipefail
ST=${ST:-/home/user/.local/state/fh}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
OLD_SSH=${OLD_SSH:-$ST/reserve/ssh.sh}
OLD_HOST=root@86.38.238.27
KEY=$HOME/.ssh/verda_framehunt
SSH_OLD_OPTS="-F none -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o UserKnownHostsFile=$ST/reserve/known_hosts -o LogLevel=ERROR"
NEW_KH=$ST/swap/known_hosts
RUN=/root/fhdata/agrpo/run1          # async run root (same path on both boxes)
SYNC_IT2=/root/fhdata/grpo/it2/train/final
log() { echo "$(date -u +%H:%M:%S) migrate: $*" >&2; }

new_ip() {
  if [ -n "${NEW_IP:-}" ]; then echo "$NEW_IP"; return; fi
  python3 - "$ST/swap/box.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))
for k in ("ip", "ipv4", "public_ip", "host", "hostname", "address"):
    if b.get(k):
        print(b[k]); break
else:
    sys.exit(f"no ip/host in {sys.argv[1]}: keys {sorted(b)}")
PY
}
SSH_NEW_OPTS() { echo "-F none -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$NEW_KH -o LogLevel=ERROR -o ConnectTimeout=20"; }
new() { ssh $(SSH_NEW_OPTS) root@"$(new_ip)" "$@"; }
to_new() { rsync -a --info=stats0 -e "ssh $(SSH_NEW_OPTS)" "$@"; }       # to_new SRC... root@IP:DST
from_old() { rsync -a --info=stats0 -e "ssh $SSH_OLD_OPTS" "$@"; }

# ------------------------------------------------------------------ fetch (old box -> here)
fetch() {
  mkdir -p $ST/box-runs $ST/box-adapters $ST/box-state
  chmod 700 $ST/box-state
  log "eval runs (/root/fh/runs, all jobs incl. fh-eval-qwen-grpo*)"
  from_old $OLD_HOST:/root/fh/runs/ $ST/box-runs/
  log "sync GRPO: every iteration's adapter, data stats, curve, logs (not the rollout request logs)"
  from_old --exclude 'requests-g*' --exclude 'runs/' --exclude 'tasks/' --exclude 'data/images/' \
    --exclude 'ckpt-*' $OLD_HOST:/root/fhdata/grpo/ $ST/box-adapters/grpo-sync/
  log "async run: adapters v0..vN, metrics, events, policy, scheduler state, queue, logs"
  from_old --exclude 'requests-g*' --exclude 'runs/' --exclude 'jobs/' --exclude 'convert/' \
    $OLD_HOST:$RUN/ $ST/box-adapters/agrpo-run1/
  log "async rollout trajectories (relay logs + Harbor trial dirs, ~2 GB; training data provenance)"
  from_old $OLD_HOST:$RUN/requests-g0 $OLD_HOST:$RUN/requests-g1 $OLD_HOST:$RUN/requests-g2 \
    $OLD_HOST:$RUN/requests-g3 $OLD_HOST:$RUN/requests-g4 $OLD_HOST:$RUN/requests-g5 $OLD_HOST:$RUN/runs \
    $ST/box-adapters/agrpo-run1/ 2>/dev/null || log "(some request dirs missing; fine)"
  log "SFT v3 adapter and its training logs"
  from_old --exclude 'ckpt-*' $OLD_HOST:/root/fhdata/runs/sft-v3/ $ST/box-adapters/sft-v3-run/
  log "tasks (EVAL dirs, TRAIN pools) and box config incl. secrets (0600, never printed)"
  from_old $OLD_HOST:/root/fh/tasks/eval $OLD_HOST:/root/fh/tasks/eval-scandoc $OLD_HOST:/root/fh/tasks/eval40 \
    $OLD_HOST:/root/fh/tasks/agrpo-pool2 $ST/box-state/tasks/
  from_old $OLD_HOST:/root/fh/serve_gpu.sh $OLD_HOST:/root/fh/prepare.sh $OLD_HOST:/root/fh/verify_weights.py \
    $OLD_HOST:/root/fh/vllm.env $OLD_HOST:/root/fh/key.env $ST/box-state/
  from_old --include 'cfg-*.json' --exclude '*' $OLD_HOST:/root/fh/ $ST/box-state/
  from_old $OLD_HOST:/root/fhdata/models/Qwen3.8-27B.REVISION $OLD_HOST:/root/fhdata/models/Qwen3.8-27B.verify.json \
    $ST/box-state/ 2>/dev/null || true
  chmod 600 $ST/box-state/vllm.env $ST/box-state/key.env
  log "done: $(du -sh $ST/box-runs $ST/box-adapters $ST/box-state | tr '\n' ' ')"
}

# ------------------------------------------------------------------ setup (here -> new box)
setup() {
  local ip; ip=$(new_ip); mkdir -p $(dirname $NEW_KH)
  log "new box $ip"
  new 'nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8; nproc; df -h / | tail -1'
  # docker: the address pools and bip the eval fleet uses, set before any container exists
  new 'bash -s' <<'EOS'
set -euo pipefail
want='{"runtimes":{"nvidia":{"args":[],"path":"nvidia-container-runtime"}},"default-address-pools":[{"base":"10.200.0.0/14","size":24}],"bip":"172.17.0.1/16"}'
cur=$(python3 -c "import json;print(json.dumps(json.load(open('/etc/docker/daemon.json')),sort_keys=True))" 2>/dev/null || echo none)
new=$(python3 -c "import json,sys;print(json.dumps(json.loads(sys.argv[1]),sort_keys=True))" "$want")
if [ "$cur" != "$new" ]; then
  if [ -n "$(docker ps -aq 2>/dev/null)" ]; then echo "daemon.json differs but containers exist: not touching docker" >&2; exit 1; fi
  python3 - "$want" <<'PY'
import json, sys
p = "/etc/docker/daemon.json"
try:
    d = json.load(open(p))
except Exception:
    d = {}
w = json.loads(sys.argv[1])
d.setdefault("runtimes", w["runtimes"])
d["default-address-pools"] = w["default-address-pools"]
d["bip"] = w["bip"]
json.dump(d, open(p, "w"), indent=1)
PY
  systemctl restart docker   # a fresh box with no containers: the only time this is done
fi
docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'
EOS
  new 'mkdir -p /root/fh/tasks /root/fh/logs /root/fh/runs /root/fhdata/models /root/fhdata/envs /root/fhdata/cache; chmod 700 /root/fh'
  log "box config, scripts, code"
  to_new $ST/box-state/serve_gpu.sh $ST/box-state/prepare.sh $ST/box-state/verify_weights.py \
    $ST/box-state/vllm.env $ST/box-state/key.env $ST/box-state/cfg-*.json root@$ip:/root/fh/
  new 'chmod 600 /root/fh/vllm.env /root/fh/key.env; chmod +x /root/fh/*.sh'
  to_new --delete --exclude .git --exclude 'train/data' --exclude 'train/out' --exclude '.venv' --exclude '__pycache__' \
    --exclude results --exclude examples --exclude reviews $REPO/ root@$ip:/root/fh/grpo-code/
  log "prepare (packages, uv, Harbor 0.25.0, weights at the pinned revision + verify, vLLM 0.33.0): ~15 min first time"
  new 'systemctl reset-failed fh-prepare 2>/dev/null; systemctl is-active -q fh-prepare || systemd-run --unit=fh-prepare bash -c "/root/fh/prepare.sh > /root/fh/logs/prepare.log 2>&1"'
  log "tasks + run state (in parallel with prepare)"
  to_new $ST/box-state/tasks/ root@$ip:/root/fh/tasks/
  new "mkdir -p $RUN /root/fhdata/grpo/it2/train /root/fhdata/runs/sft-v3"
  to_new $ST/box-adapters/grpo-sync/it2/train/final root@$ip:/root/fhdata/grpo/it2/train/
  to_new $ST/box-adapters/sft-v3/ root@$ip:/root/fhdata/runs/sft-v3/final/
  to_new --exclude 'requests-g*' --exclude 'runs/' --exclude 'jobs/' --exclude 'convert/' --exclude STOP \
    $ST/box-adapters/agrpo-run1/ root@$ip:$RUN/
  new "rm -f $RUN/STOP; ls $RUN/adapters; cat $RUN/adapters/LATEST $RUN/policy.json"
  log "train venv (torch 2.15.0, transformers 5.22.2, peft 0.21.2, fla 0.6.0)"
  new 'bash -s' <<'EOS'
set -euo pipefail
export PATH=/root/.local/bin:$PATH UV_PYTHON_INSTALL_DIR=/root/fhdata/uv/python UV_LINK_MODE=copy
until command -v uv >/dev/null; do sleep 10; done
V=/root/fhdata/envs/train
if [ ! -f $V/.fh-ok ]; then
  rm -rf $V; uv venv -q --managed-python --python 3.12 $V
  VIRTUAL_ENV=$V uv pip install -q -r /root/fh/grpo-code/train/requirements.txt flash-linear-attention==0.6.0
  $V/bin/python -c "import torch, transformers, peft; assert torch.cuda.is_available()"; touch $V/.fh-ok
fi
echo train venv ok
EOS
  log "prebuilt images (agent, MCP sidecar, verifier) from one pool task"
  new 'bash -s' <<'EOS'
set -euo pipefail
cd /root/fh/tasks/agrpo-pool2/dialogs-110401
# the pool2 compose file names the image, so build from the unmodified contexts (identical in every task)
docker image inspect fh-grpo-main:v1 >/dev/null 2>&1 || docker build -q -t fh-grpo-main:v1 environment
docker image inspect fh-grpo-mcp:v1 >/dev/null 2>&1 || docker build -q -t fh-grpo-mcp:v1 environment/mcp
docker image inspect fh-grpo-verifier:v1 >/dev/null 2>&1 || docker build -q -t fh-grpo-verifier:v1 tests
docker images | grep fh-grpo
EOS
  log "waiting for prepare (weights + vLLM venv)"
  until new 'grep -q PREPARED /root/fh/logs/prepare.log 2>/dev/null'; do
    new 'systemctl is-failed -q fh-prepare' && { new 'tail -20 /root/fh/logs/prepare.log'; exit 1; }
    sleep 30
  done
  log "setup done"
}

# ------------------------------------------------------------------ start (new box)
start() {
  local END=${1:-2026-11-15T02:20Z}
  new "ROOT=$RUN MODEL=/root/fhdata/models/Qwen3.8-27B INIT=$SYNC_IT2 POOL=/root/fh/tasks/agrpo-pool2 \
    INFER_GPUS='0 1 2 3 4 5' TRAIN_GPUS=6,7 DEADLINE=$END \
    SCHED_ARGS='--inflight 24 --agent-timeout-min 8 --launch-gap-s 20' \
    TRAIN_ARGS='--min-groups 4 --max-groups 12 --max-stale 2 --lr 2e-5 --clip 0.2 --max-len 18432' \
    /root/fh/grpo-code/infra/verda/grpo_async/start.sh"
  # at END: stop (trainer finishes its update), free GPUs 4-5 too, then both evals on 6 GPUs
  new "systemctl reset-failed fh-agrpo-finish fh-agrpo-free45 2>/dev/null; \
    systemd-run --unit=fh-agrpo-free45 bash -c 'until [ -f $RUN/STOP ]; do sleep 2; done; systemctl stop fh-agrpo-relay-4 fh-agrpo-relay-5 fh-agrpo-vllm-4 fh-agrpo-vllm-5'; \
    systemd-run --unit=fh-agrpo-finish --setenv=ROOT=$RUN --setenv=END=$END --setenv=SYNC_ADAPTER=none \
      --setenv=TAG=grpo2 --setenv=INFER_GPUS='0 1 2 3' bash -c '/root/fh/grpo-code/infra/verda/grpo_async/finish.sh >> /root/fhdata/agrpo/finish.log 2>&1'"
  log "async running until $END; evals after"
}

fetch_new() {   # the continued run's results from the new box
  local ip; ip=$(new_ip)
  rsync -a -e "ssh $(SSH_NEW_OPTS)" root@$ip:/root/fh/runs/ $ST/box-runs/
  rsync -a -e "ssh $(SSH_NEW_OPTS)" --exclude 'requests-g*' --exclude 'runs/' --exclude 'jobs/' --exclude 'convert/' \
    root@$ip:$RUN/ $ST/box-adapters/agrpo-run1-cont/
}

status() {
  new "date; systemctl list-units 'fh-*' --state=active --no-legend --plain | cut -c1-50; \
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; cat $RUN/adapters/LATEST; tail -3 $RUN/times.txt"
}

case "${1:-}" in
  fetch) fetch ;;
  setup) setup ;;
  start) shift; start "$@" ;;
  status) status ;;
  fetch-new) fetch_new ;;
  *) echo "usage: $0 fetch|setup|start [END]|status" >&2; exit 2 ;;
esac
