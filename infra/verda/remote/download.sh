# Job body for the weight-download instance (appended to common.sh by verda_cli.py).
# Placeholders (double-underscore wrapped): MODELS ("repo revision" lines), VLLM_VERSION, VERIFY_PY
mount_data || exit 1
install_uv
STATUS="$MNT/logs/download.status"
mkdir -p "$MNT/models" "$MNT/logs" "$MNT/envs" "$MNT/bin"
echo "running $(date -Is)" > "$STATUS"
exec > >(tee -a "$MNT/logs/download.log") 2>&1

cat > "$MNT/bin/verify_weights.py" <<'PYEOF'
__VERIFY_PY__
PYEOF

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/root/.cache/huggingface   # keep the HF cache off the SFS
fail=0
while read -r repo rev; do
  [ -z "$repo" ] && continue
  name="${repo#*/}"
  echo "[fh] downloading $repo@$rev -> $MNT/models/$name"
  t0=$(date +%s)
  uvx --python 3.12 --from 'huggingface_hub[cli,hf_transfer]' \
      hf download "$repo" --revision "$rev" --local-dir "$MNT/models/$name" --max-workers 16 || fail=1
  echo "[fh] downloaded $repo in $(( $(date +%s) - t0 ))s"
  uv run --no-project --python 3.12 python "$MNT/bin/verify_weights.py" "$repo" "$rev" "$MNT/models/$name" || fail=1
  echo "$repo $rev" > "$MNT/models/$name.REVISION"
done <<'EOF'
__MODELS__
EOF

# vLLM lives in a venv on the SFS so serving instances skip the install.
VENV="$MNT/envs/vllm-__VLLM_VERSION__"
if [ ! -x "$VENV/bin/vllm" ]; then
  echo "[fh] building $VENV"
  uv python install 3.12
  uv venv --python 3.12 "$VENV" && \
  VIRTUAL_ENV="$VENV" uv pip install "vllm==__VLLM_VERSION__" || fail=1
fi
"$VENV/bin/python" -c "import vllm; print('[fh] vllm', vllm.__version__)" || fail=1

if [ "$fail" = 0 ]; then echo "done $(date -Is)" > "$STATUS"; else echo "failed $(date -Is)" > "$STATUS"; fi
du -sh "$MNT"/models/* "$VENV" || true
echo "[fh] download job finished fail=$fail"
