# Job body for the LoRA training instance (appended to common.sh by verda_cli.py).
# Placeholders (double-underscore wrapped): RUN, REQ_HASH, REQS (requirements.txt +
#   requirements-gpu.txt), TRAIN_ARGS (sft_lora.py arguments, persisted per run).
# This script only prepares the box: mount the volume, build/reuse the training venv on
# the volume, write the fh-train systemd unit. verda_cli.py then uploads the dataset and
# code over SSH and starts the unit (train-up / train-ensure). The trainer resumes from
# the newest checkpoint in /mnt/fh/runs/<run>, so a re-provisioned box continues the run.
mount_data || exit 1
install_uv
RUN="__RUN__"
OUT="$MNT/runs/$RUN"
CODE="$OUT/code"
VENV="$MNT/envs/train-__REQ_HASH__"
mkdir -p "$OUT" /var/lib/fh "$MNT/envs"

if [ ! -f "$VENV/.fh-ok" ]; then
  echo "[fh] building $VENV"
  cat > /tmp/fh-train-requirements.txt <<'REQEOF'
__REQS__
REQEOF
  rm -rf "$VENV"
  uv venv --python 3.12 "$VENV" \
    && VIRTUAL_ENV="$VENV" uv pip install -r /tmp/fh-train-requirements.txt \
    && "$VENV/bin/python" -c "import torch, transformers, peft; assert torch.cuda.is_available(); print('[fh] torch', torch.__version__, 'transformers', transformers.__version__)" \
    && touch "$VENV/.fh-ok" \
    || { echo "[fh] FATAL: training venv build failed"; exit 1; }
fi

cat > /etc/systemd/system/fh-train.service <<EOF
[Unit]
Description=FrameHunt LoRA SFT ($RUN)
After=network-online.target local-fs.target

[Service]
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Environment=HF_HUB_OFFLINE=1
Environment=TOKENIZERS_PARALLELISM=false
WorkingDirectory=$CODE
ExecStart=$VENV/bin/python $CODE/sft_lora.py __TRAIN_ARGS__
# SIGTERM: the trainer checkpoints at the next optimizer step and exits 143
KillSignal=SIGTERM
TimeoutStopSec=900
Restart=no
StandardOutput=append:$OUT/train.log
StandardError=append:$OUT/train.log
EOF
systemctl daemon-reload
echo "venv=$VENV" > /var/lib/fh/train-ready
echo "[fh] training box ready; waiting for verda_cli.py to upload data and start fh-train"
