#!/bin/bash
# Shared prelude for Verda startup scripts. Rendered by verda_cli.py
# (placeholders are double-underscore wrapped): MNT = mount point of the data
# volume, VOL_GB = its size in GB (used to find the disk on first use).
set -uo pipefail
MNT="__MNT__"
VOL_GB="__VOL_GB__"
LABEL=fhqwen
mkdir -p /var/log/fh
exec > >(tee -a /var/log/fh/startup.log) 2>&1
echo "[fh] startup $(date -Is) host=$(hostname)"

# The persistent data volume is a block NVMe volume attached at instance creation.
# It is formatted ext4 with label "fhqwen" the first time it is seen.
mount_data() {
  mkdir -p "$MNT"
  if mountpoint -q "$MNT"; then return 0; fi
  local dev=""
  for i in $(seq 1 30); do
    dev="$(blkid -L "$LABEL" 2>/dev/null || true)"
    [ -n "$dev" ] && break
    # first use: an unformatted, unpartitioned disk of the volume's size
    for d in $(lsblk -dnpo NAME,TYPE | awk '$2=="disk"{print $1}'); do
      size_gb=$(( $(blockdev --getsize64 "$d") / 1000000000 ))
      if [ "$size_gb" -ge $((VOL_GB - 5)) ] && [ "$size_gb" -le $((VOL_GB + 25)) ] \
         && [ -z "$(blkid -o value -s TYPE "$d" 2>/dev/null)" ] \
         && [ "$(lsblk -no NAME "$d" | wc -l)" = 1 ]; then
        echo "[fh] formatting new data volume $d (${size_gb} GB)"
        mkfs.ext4 -q -L "$LABEL" -m 0 "$d" && dev="$d"
        break
      fi
    done
    [ -n "$dev" ] && break
    echo "[fh] data volume not found yet (attempt $i)"; sleep 10
  done
  [ -n "$dev" ] || { echo "[fh] FATAL: data volume not found"; lsblk; return 1; }
  mount -o noatime "$dev" "$MNT" || { echo "[fh] FATAL: mount $dev failed"; return 1; }
  grep -q "LABEL=$LABEL" /etc/fstab || echo "LABEL=$LABEL $MNT ext4 defaults,noatime,nofail 0 2" >> /etc/fstab
  echo "[fh] mounted $dev at $MNT"; df -h "$MNT" | tail -1
}

install_uv() {
  export PATH="/root/.local/bin:$PATH"
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="/root/.local/bin:$PATH"
  # Python interpreters live on the volume so venvs built there work on every instance.
  export UV_PYTHON_INSTALL_DIR="$MNT/uv/python"
  export UV_LINK_MODE=copy
}
