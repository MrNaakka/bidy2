# Job body for the vLLM serving instance (appended to common.sh by verda_cli.py).
# Placeholders (double-underscore wrapped): VLLM_VERSION, MODEL_DIR, SERVED_NAME, PORT,
#   VLLM_ARGS, API_KEY
# vLLM binds 127.0.0.1 only; verda_cli.py reaches it through an SSH tunnel.
mount_data || exit 1
VENV="$MNT/envs/vllm-__VLLM_VERSION__"
mkdir -p "$MNT/logs"
LOG="$MNT/logs/serve-$(hostname).log"
[ -x "$VENV/bin/vllm" ] || { echo "[fh] FATAL: $VENV missing; run the download job first"; exit 1; }

# The key reaches vLLM only through its environment (VLLM_API_KEY, read by vLLM's auth
# middleware when --api-key is absent), never on the command line where ps/procfs show it.
install -m 600 /dev/null /etc/fh-vllm.env
cat > /etc/fh-vllm.env <<'EOF'
VLLM_API_KEY=__API_KEY__
EOF

cat > /etc/systemd/system/fh-vllm.service <<EOF
[Unit]
Description=FrameHunt vLLM server
After=network-online.target local-fs.target
Wants=network-online.target

[Service]
EnvironmentFile=/etc/fh-vllm.env
Environment=HF_HUB_OFFLINE=1
Environment=VLLM_NO_USAGE_STATS=1
Environment=VLLM_CACHE_ROOT=$MNT/cache/vllm
Environment=TRITON_CACHE_DIR=$MNT/cache/triton
# venv bin has ninja (needed by any FlashInfer JIT build); CUDA toolkit ships with the image
Environment=PATH=$VENV/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=CUDA_HOME=/usr/local/cuda
Environment=FLASHINFER_WORKSPACE_BASE=$MNT/cache/flashinfer
# PyTorch-native top-k/top-p sampler: avoids a multi-minute FlashInfer JIT on every new box
Environment=VLLM_USE_FLASHINFER_SAMPLER=0
ExecStart=$VENV/bin/vllm serve __MODEL_DIR__ --served-model-name __SERVED_NAME__ --host 127.0.0.1 --port __PORT__ __VLLM_ARGS__
Restart=on-failure
RestartSec=15
StandardOutput=append:$LOG
StandardError=append:$LOG

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now fh-vllm.service
echo "[fh] fh-vllm started; log $LOG"
