"""Verda cloud tooling for FrameHunt: serve Qwen3.8-27B (vision) behind an
OpenAI-compatible vLLM endpoint on cheap spot GPUs, with weights on a persistent
block volume.

Run from the repo root:  uv run python infra/verda/verda_cli.py <command> [...]
Credentials come from ./.env (VERDA_CLIENT_ID, VERDA_CLIENT_SECRET). Secrets are
never printed. See infra/verda/README.md.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import io
import json
import re
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
from verda.exceptions import APIException

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV_PATH = ROOT / ".env"
STATE_PATH = HERE / "state.json"
LEDGER_PATH = HERE / "ledger.jsonl"
INVENTORY_PATH = HERE / "inventory.json"
REMOTE = HERE / "remote"
SSH_KEY = Path.home() / ".ssh" / "verda_framehunt"
KNOWN_HOSTS = HERE / "known_hosts"  # host keys, pinned per instance id (HostKeyAlias)
LOCK_PATH = HERE / ".serve-ensure.lock"
STATE_LOCK = HERE / ".state.lock"
TUNNEL_PIDFILE = HERE / "tunnel.pid"
TUNNEL_LOG = HERE / "tunnel.log"

# ---- fixed choices (pinned for reproducibility) --------------------------------
LOCATION = "FIN-01"  # most spot GPU types + CPU spot; see inventory.json
VOLUME_NAME = "fh-qwen"
VOLUME_TYPE = "NVMe"  # block volume, one instance at a time (NVMe_Shared/SFS is not mountable from FIN-01 instances)
VOLUME_SIZE_GB = 300
MNT = "/mnt/fh"
IMAGE = "24.04.cuda13.4"  # Ubuntu 24.04 + CUDA 13.4 driver
VLLM_VERSION = "0.33.0"
MODELS = {  # repo -> pinned revision
    "Qwen/Qwen3.8-27B": "37561816e66b5b9153b1b64c10968c18c1e6f4a0",
    "Qwen/Qwen3.8-27B-FP8": "c3a462cc3d38cf351f3b26cbd763469eeb74ef81",
}
VARIANT_REPO = {"bf16": "Qwen/Qwen3.8-27B", "fp8": "Qwen/Qwen3.8-27B-FP8"}
SERVED_NAME = "qwen3.8-27b"
LORA_NAME = "qwen3.8-27b-fh"  # LoRA adapter served next to the base model (serve-up --lora PATH)
LORA_MAX_RANK = 16
PORT = 8000
# Every instance this CLI creates is named fhbot-<serve|dl>-<12 hex> (machine-generated,
# nobody types that by hand). Reconcile sweeps untracked instances only if their name
# matches OWN_HOSTNAME_RE exactly; every other name (including other fh-* names and
# legacy fh-qwen-*/fh-serve-*/fh-dl-* boxes) is report-only unless tracked by id.
SERVE_PREFIX = "fhbot-serve-"
DL_PREFIX = "fhbot-dl-"
TRAIN_PREFIX = "fhbot-train-"
ROLE_PREFIX = {"serve": SERVE_PREFIX, "download": DL_PREFIX, "train": TRAIN_PREFIX}
OWN_HOSTNAME_RE = re.compile(r"^fhbot-(serve|dl|train)-[0-9a-f]{12}$")
ROLES = ("serve", "download", "train")  # state[role] holds the tracked instance of each role
DOWNLOAD_FALLBACK = ["CPU.8V.32G", "CPU.16V.64G", "CPU.4V.16G", "CPU.32V.128G"]
STARTUP_GRACE_S = 40 * 60  # a serving box that is not healthy after this is replaced
GONE_STATUSES = ("deleted", "discontinued")  # get_by_id status of an instance that no longer exists/bills

# vLLM listens on 127.0.0.1:PORT on the instance only. This box reaches it through an
# SSH local forward (TUNNEL_BIND:TUNNEL_PORT -> instance 127.0.0.1:PORT), so the Bearer
# key never crosses the internet in clear text. Docker on this box is rootless
# (slirp4netns with host loopback enabled): containers on any compose network reach the
# host's 127.0.0.1 as 10.0.2.2, so QWEN_BASE_URL is http://10.0.2.2:TUNNEL_PORT/v1.
# On rootful Docker set FH_TUNNEL_BIND=172.17.0.1 and FH_TUNNEL_CONTAINER_HOST=172.17.0.1.
TUNNEL_BIND = os.environ.get("FH_TUNNEL_BIND", "127.0.0.1")
TUNNEL_PORT = int(os.environ.get("FH_TUNNEL_PORT", "18000"))
TUNNEL_CONTAINER_HOST = os.environ.get("FH_TUNNEL_CONTAINER_HOST", "10.0.2.2")

# Serving candidates. Weights: BF16 55.6 GB, FP8 30.9 GB. The model is a hybrid
# (Gated DeltaNet + 16 full-attention layers with 4 KV heads x 256), so KV cache
# is ~64 KB/token and long contexts are cheap once the weights fit.
# (instance_type, variant, extra vLLM args)
CANDIDATES: list[tuple[str, str, list[str]]] = [
    ("1A6000.10V", "fp8", []),
    ("1RTX6000ADA.10V", "fp8", []),
    ("2A6000.20V", "bf16", ["--tensor-parallel-size", "2"]),
    ("1A100.40S.22V", "fp8", ["--max-model-len", "32768", "--max-num-seqs", "4"]),
    ("1L40S.20V", "fp8", []),
    ("1A100.22V", "bf16", []),
    ("1RTXPRO6000.30V", "bf16", []),
    ("1H100.80S.30V", "bf16", []),
    ("1H100.80S.32V", "bf16", []),
    ("1H200.141S.44V", "bf16", []),
]
BASE_VLLM_ARGS = [
    "--max-model-len", "65536",
    "--gpu-memory-utilization", "0.92",
    "--max-num-seqs", "16",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--reasoning-parser", "qwen3",
    "--limit-mm-per-prompt", "'{\"image\": 64, \"video\": 0}'",
]


# ---- env / state / ledger ---------------------------------------------------------
def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def set_env_var(key: str, value: str) -> None:
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n")
    ENV_PATH.chmod(0o600)


# ---- state: every write is a short locked transaction ------------------------------
# All commands (and the watchdog) share state.json. A write is always
# lock -> load fresh -> mutate -> save, never a save of an older in-memory snapshot,
# so a concurrent command can't erase e.g. a creation intent written meanwhile.
# Transactions only touch the file; network calls happen outside them.
_txn_depth = 0


def load_state() -> dict:
    """Read-only snapshot. Never pass it back to a writer; use update_state()."""
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def _write_state(state: dict) -> None:
    tmp = STATE_PATH.with_name(STATE_PATH.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(STATE_PATH)  # atomic: a crash never leaves a half-written state file


@contextlib.contextmanager
def state_lock(timeout_s: float = 120):
    global _txn_depth
    if _txn_depth:
        raise RuntimeError("nested state transaction")  # the outer one would save a stale copy
    with STATE_LOCK.open("a") as f:
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 > timeout_s:
                    raise TimeoutError(f"state lock {STATE_LOCK} held for over {timeout_s:.0f}s")
                time.sleep(0.05)
        _txn_depth += 1
        try:
            yield
        finally:
            _txn_depth -= 1


def update_state(fn) -> dict:
    """Atomically load the current state, apply fn(state) and save it. Returns the new state."""
    with state_lock():
        st = load_state()
        fn(st)
        _write_state(st)
        return st


def save_state(state: dict) -> None:
    """Replace the whole state (bootstrap/tests only; commands use update_state)."""
    with state_lock():
        _write_state(state)


def _proc_start(pid: int) -> str | None:
    """Kernel start time of a live (non-zombie) process; with the pid it identifies the process."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    return None if fields[0] in ("Z", "X") else fields[19]


def proc_identity() -> dict:
    return {"pid": os.getpid(), "pid_start": _proc_start(os.getpid())}


def owner_alive(rec: dict) -> bool:
    """True if the process recorded in rec (proc_identity()) is still running."""
    pid = rec.get("pid")
    return bool(pid) and rec.get("pid_start") is not None and _proc_start(pid) == rec.get("pid_start")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ledger_append(**rec) -> None:
    rec = {"ts": now_iso(), **rec}  # a caller may pass ts= (backfill with the original creation time)
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(rec) + "\n")


# The Verda SDK calls requests.request / requests.post (auth) without a timeout, so a
# stalled connection could hang a command (and hold the serve-ensure lock) forever.
# Every requests call in this process gets a finite (connect, read) timeout instead.
HTTP_TIMEOUT = (10, 60)


def install_http_timeouts() -> None:
    import requests

    orig = requests.sessions.Session.request
    if getattr(orig, "_fh_default_timeout", False):
        return

    def request(self, method, url, *args, **kwargs):
        if kwargs.get("timeout") is None and len(args) < 7:  # timeout is positional #7 after url
            kwargs["timeout"] = HTTP_TIMEOUT
        return orig(self, method, url, *args, **kwargs)

    request._fh_default_timeout = True
    requests.sessions.Session.request = request


install_http_timeouts()


def client():
    from verda import VerdaClient

    install_http_timeouts()

    env = load_env()
    cid = os.environ.get("VERDA_CLIENT_ID") or env.get("VERDA_CLIENT_ID")
    sec = os.environ.get("VERDA_CLIENT_SECRET") or env.get("VERDA_CLIENT_SECRET")
    if not cid or not sec:
        sys.exit("VERDA_CLIENT_ID / VERDA_CLIENT_SECRET missing from .env")
    return VerdaClient(cid, sec)


def eur_prices(c) -> dict[str, dict]:
    raw = c._http_client.get("/instance-types", params={"currency": "eur"}).json()
    return {t["instance_type"]: t for t in raw}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---- inventory ------------------------------------------------------------------
def gpu_family(t: dict) -> str:
    return (t.get("model") or t["gpu"].get("description") or "").upper()


def fits(t: dict, variant: str) -> bool:
    n = t["gpu"].get("number_of_gpus") or 0
    vram = (t["gpu_memory"] or {}).get("size_in_gigabytes") or 0
    if n == 0 or "V100" in gpu_family(t) or ".CC" in t["instance_type"]:
        return False  # V100 lacks BF16/modern kernels; CC = confidential-computing SKUs
    if variant == "bf16":
        return vram >= 72 and n in (1, 2, 4)
    return vram >= 40 and n in (1, 2, 4)


def cmd_inventory(args) -> None:
    c = client()
    types = eur_prices(c)
    avail = {}
    for spot in (True, False):
        for loc in c.instances.get_availabilities(is_spot=spot):
            for it in loc["availabilities"]:
                avail.setdefault(it, {"spot": [], "on_demand": []})[
                    "spot" if spot else "on_demand"
                ].append(loc["location_code"])
    rows = []
    for name in sorted(set(types) | set(avail)):
        t = types.get(name)
        a = avail.get(name, {"spot": [], "on_demand": []})
        if t is None:  # available but missing from the price list (e.g. CPU-TURIN)
            rows.append({"instance_type": name, "listed": False, **a})
            continue
        rows.append({
            "instance_type": name,
            "listed": True,
            "gpu": t["gpu"].get("description") or "",
            "n_gpus": t["gpu"].get("number_of_gpus") or 0,
            "vram_gb": (t["gpu_memory"] or {}).get("size_in_gigabytes") or 0,
            "vcpus": t["cpu"].get("number_of_cores"),
            "ram_gb": t["memory"].get("size_in_gigabytes"),
            "eur_per_h": float(t["price_per_hour"]),
            "eur_per_h_spot": float(t["spot_price"]),
            "spot": a["spot"],
            "on_demand": a["on_demand"],
            "fits_bf16": fits(t, "bf16"),
            "fits_fp8": fits(t, "fp8"),
        })
    rows.sort(key=lambda r: (r.get("n_gpus", 0) > 0, r.get("eur_per_h_spot", 1e9)))

    def best(variant, loc=None):
        opts = []
        for r in rows:
            if not r.get(f"fits_{variant}"):
                continue
            if (loc is None and r["spot"]) or (loc and loc in r["spot"]):
                opts.append(("spot", r["eur_per_h_spot"], r))
            if (loc is None and r["on_demand"]) or (loc and loc in r["on_demand"]):
                opts.append(("on_demand", r["eur_per_h"], r))
        if not opts:
            return None
        kind, price, r = min(opts, key=lambda o: o[1])
        locs = r[kind] if loc is None else [loc]
        return {"instance_type": r["instance_type"], "contract": kind, "eur_per_h": price,
                "locations": locs, "gpu": r["gpu"]}

    rec = {
        "bf16_any_location": best("bf16"),
        "fp8_any_location": best("fp8"),
        f"bf16_{LOCATION}": best("bf16", LOCATION),
        f"fp8_{LOCATION}": best("fp8", LOCATION),
    }
    snap = {"taken_at": now_iso(), "currency": "eur", "volume_location": LOCATION,
            "recommendation": rec, "types": rows}
    INVENTORY_PATH.write_text(json.dumps(snap, indent=1) + "\n")

    print(f"{'type':<20}{'GPU':<26}{'VRAM':>5}{'vCPU':>5}{'RAM':>5}{'€/h':>8}{'spot€/h':>9}  "
          f"{'spot avail':<22}{'on-demand avail':<22}fit")
    for r in rows:
        if not r["listed"]:
            print(f"{r['instance_type']:<20}{'(not in price list)':<26}{'':>5}{'':>5}{'':>5}"
                  f"{'':>8}{'':>9}  {','.join(r['spot']) or '-':<22}{','.join(r['on_demand']) or '-':<22}")
            continue
        if args.available and not (r["spot"] or r["on_demand"]):
            continue
        fit = "bf16" if r["fits_bf16"] else ("fp8" if r["fits_fp8"] else "")
        print(f"{r['instance_type']:<20}{r['gpu'][:25]:<26}{r['vram_gb']:>5}{r['vcpus']:>5}"
              f"{r['ram_gb']:>5}{r['eur_per_h']:>8.3f}{r['eur_per_h_spot']:>9.3f}  "
              f"{','.join(r['spot']) or '-':<22}{','.join(r['on_demand']) or '-':<22}{fit}")
    print("\nrecommendation:")
    print(json.dumps(rec, indent=1))
    print(f"snapshot -> {INVENTORY_PATH.relative_to(ROOT)}")


# ---- ssh -------------------------------------------------------------------------
def ensure_ssh_key(c, state) -> str:
    if state.get("ssh_key_id"):
        return state["ssh_key_id"]
    if not SSH_KEY.exists():
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-C", "framehunt-verda",
                        "-f", str(SSH_KEY)], check=True)
    pub = SSH_KEY.with_suffix(".pub").read_text().strip()
    for k in c.ssh_keys.get():
        if k.public_key.strip() == pub:
            key_id = k.id
            break
    else:
        key_id = c.ssh_keys.create("framehunt-verda", pub).id
    update_state(lambda st: st.__setitem__("ssh_key_id", key_id))
    state["ssh_key_id"] = key_id
    return key_id


def ssh_opts(alias: str) -> list[str]:
    """Trust-on-first-use per instance: the host key is pinned under the instance id
    (HostKeyAlias) in KNOWN_HOSTS, so a later box that reuses the IP gets its own entry."""
    # ControlMaster/ControlPath off: a user-level "ControlMaster auto" + ControlPersist would
    # daemonize the tunnel into a mux master that we can neither track nor stop.
    return ["-F", "none", "-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ControlMaster=no", "-o", "ControlPath=none",
            "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={KNOWN_HOSTS}",
            "-o", f"HostKeyAlias={alias}", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10"]


def ssh(ip: str, cmd: str, *, alias: str, timeout: int = 60, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *ssh_opts(alias), f"root@{ip}", cmd],
                          capture_output=True, text=True, timeout=timeout, check=check)


# ---- volume ---------------------------------------------------------------------
def get_volume(c, state):
    vid = state.get("volume", {}).get("id")
    if not vid:
        return None
    try:
        return c.volumes.get_by_id(vid)
    except Exception:
        return None


def cmd_volume_create(args) -> None:
    c = client()
    state = load_state()
    vol = get_volume(c, state)
    if vol:
        log(f"volume exists: {vol.id} {vol.name} {vol.size}GB {vol.type} {vol.location} {vol.status}")
        return
    for v in c.volumes.get():
        if v.name == VOLUME_NAME and v.location == LOCATION:
            vol = v
            break
    else:
        vol = c.volumes.create(VOLUME_TYPE, VOLUME_NAME, VOLUME_SIZE_GB, location=LOCATION)
        vol = c.volumes.get_by_id(vol.id)
        ledger_append(event="create", kind="volume", id=vol.id, type=VOLUME_TYPE,
                      size_gb=VOLUME_SIZE_GB, location=LOCATION,
                      price_per_hour=round(vol.base_hourly_cost or 0, 5), currency=vol.currency or "eur")
    rec = {"id": vol.id, "name": vol.name, "size_gb": vol.size, "type": vol.type,
           "location": vol.location, "mount": MNT}
    update_state(lambda st: st.__setitem__("volume", rec))
    log(f"volume {vol.id} {vol.type} {vol.size}GB in {vol.location}")


def cmd_volume_status(args) -> None:
    c = client()
    for v in c.volumes.get():
        print(json.dumps({k: getattr(v, k) for k in ("id", "name", "status", "size", "type",
                                                     "location", "instance_id", "pseudo_path",
                                                     "mount_command", "instances")}))


# ---- instances ------------------------------------------------------------------
def render(body_file: str, subs: dict[str, str], vol: dict) -> str:
    text = (REMOTE / "common.sh").read_text() + "\n" + (REMOTE / body_file).read_text()
    subs = {"__MNT__": vol["mount"], "__VOL_GB__": str(vol["size_gb"]), **subs}
    for k, v in subs.items():
        text = text.replace(k, v)
    return text


def new_hostname(role: str) -> str:
    """Globally unique hostname for one create request (intents are keyed by it)."""
    return f"{ROLE_PREFIX[role]}{uuid.uuid4().hex[:12]}"


def _drop_intent(hostname: str) -> None:
    def f(st):
        st.get("pending_creates", {}).pop(hostname, None)
        if not st.get("pending_creates"):
            st.pop("pending_creates", None)
    update_state(f)


def create_instance(c, state, *, role, itype, hostname, script_name, script, spot, os_gb=80, record=None):
    """Create an instance and track it in state[role] ("serve" or "download").

    Before the request, an intent {creator pid, hostname, desired config} is saved, so
    `reconcile` leaves the box alone while this process lives and can adopt it with
    that config if it dies. The create POST is made directly (no SDK polling): its
    response carries the instance id, which is persisted at once. Only an error from
    the POST itself counts as a rejection; polling happens later (wait_running).
    """
    from verda.instances import OSVolume
    from verda.instances._instances import INSTANCES_ENDPOINT

    vol = state["volume"]
    key_id = ensure_ssh_key(c, state)
    wait_volume_free(c, vol["id"])
    for s in c.startup_scripts.get():  # one script per role; replace stale copies
        if s.name == script_name:
            c.startup_scripts.delete_by_id(s.id)
    sc = c.startup_scripts.create(script_name, script)
    prices = eur_prices(c)
    intent = {"role": role, "type": itype, "ts": time.time(), "record": record or {}, **proc_identity()}
    update_state(lambda st: st.setdefault("pending_creates", {}).__setitem__(hostname, intent))
    payload = {
        "instance_type": itype, "image": IMAGE, "ssh_key_ids": [key_id], "startup_script_id": sc.id,
        "hostname": hostname, "description": hostname, "location_code": vol["location"],
        "os_volume": OSVolume(name=f"{hostname}-os", size=os_gb,
                              on_spot_discontinue="delete_permanently" if spot else None).to_dict(),
        "volumes": [], "existing_volumes": [vol["id"]], "is_spot": spot, "coupon": None,
    }
    try:
        inst_id = c._http_client.post(INSTANCES_ENDPOINT, json=payload).text.strip().strip('"')
    except APIException:
        _drop_intent(hostname)  # the POST was answered with an error: nothing was created
        raise
    # (any other failure, e.g. a lost response, leaves the intent for reconcile's sweep)
    eur_h = float(prices.get(itype, {}).get("spot_price" if spot else "price_per_hour", 0) or 0)
    tracked = {"id": inst_id, "hostname": hostname, "type": itype, "spot": spot, "created": time.time(),
               "price_per_hour": eur_h, **(record or {})}

    def track(st):  # ledger row and tracking in one state transaction (ledger first)
        if not _ledger_has("create", inst_id):
            ledger_append(event="create", kind="instance", id=inst_id, type=itype, hostname=hostname,
                          location=vol["location"], spot=spot, price_per_hour=eur_h, currency="eur")
        st[role] = tracked
        st.get("pending_creates", {}).pop(hostname, None)
        if not st.get("pending_creates"):
            st.pop("pending_creates", None)
    update_state(track)
    state[role] = tracked
    log(f"created {itype} {'spot' if spot else 'on-demand'} {inst_id} ({hostname}) €{eur_h}/h")
    return SimpleNamespace(id=inst_id, hostname=hostname, instance_type=itype, is_spot=spot)


class DeleteFailed(RuntimeError):
    """An instance could not be confirmed deleted; it may still be running and billing."""


def _not_found(e: BaseException) -> bool:
    return isinstance(e, APIException) and e.code == "not_found"


def instance_gone(c, inst_id: str) -> bool:
    """True only on confirmed absence (API 404 or a terminal status). API errors propagate."""
    try:
        inst = c.instances.get_by_id(inst_id)
    except APIException as e:
        if _not_found(e):
            return True
        raise
    return inst.status in GONE_STATUSES


def delete_instance(c, inst_id: str, reason: str = "", *, attempts: int = 5, backoff_s: float = 5.0,
                    confirm_s: float = 300.0) -> None:
    """Delete an instance and wait until the API confirms it is gone.

    Transient failures (API errors, network errors, a delete that does not take) are
    retried with exponential backoff. Only a confirmed absence is recorded in the
    ledger; otherwise DeleteFailed is raised and the caller must keep its state.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            inst = c.instances.get_by_id(inst_id)
            if inst.status not in GONE_STATUSES:
                if inst.status != "deleting":
                    c.instances.action(inst_id, "delete",
                                       volume_ids=[inst.os_volume_id] if inst.os_volume_id else None,
                                       delete_permanently=True)
                deadline = time.time() + confirm_s
                while not instance_gone(c, inst_id):
                    if time.time() > deadline:
                        raise TimeoutError(f"still present {confirm_s:.0f}s after the delete request")
                    time.sleep(5)
            break
        except Exception as e:
            if _not_found(e):  # already gone (e.g. spot discontinued)
                break
            last = e
            log(f"delete {inst_id} attempt {attempt}/{attempts} failed: {type(e).__name__}: {str(e)[:200]}")
            if attempt < attempts:
                time.sleep(backoff_s * 2 ** (attempt - 1))
    else:
        raise DeleteFailed(f"could not confirm deletion of instance {inst_id} after {attempts} attempts "
                           f"({type(last).__name__}: {str(last)[:200]}); it may still be billing") from last
    ledger_append(event="delete", kind="instance", id=inst_id, reason=reason)
    log(f"deleted instance {inst_id} {reason}")


@contextlib.contextmanager
def no_interrupt():
    """Let a cleanup finish even if Ctrl-C is pressed again (it would leak a paid box)."""
    try:
        old = signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:  # not in the main thread
        yield
        return
    remaining = signal.alarm(0)  # a Deadline must not cut a cleanup short either
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old)
        if remaining:
            signal.alarm(remaining)


def delete_by_hostnames(c, hostnames: list[str], reason: str, skip: set[str] = frozenset()) -> None:
    """Best-effort immediate cleanup after an interrupted create. Anything missed (lookup
    failed, order still processing) is caught later by reconcile's fh- sweep."""
    if not hostnames:
        return
    try:
        insts = c.instances.get()
    except Exception as e:
        log(f"cleanup lookup failed ({type(e).__name__}); the next command's reconcile sweeps it")
        return
    for i in insts:
        if i.hostname in hostnames and i.id not in skip and i.status not in GONE_STATUSES:
            delete_instance(c, i.id, reason=reason)
            _drop_intent(i.hostname)


PENDING_TTL_S = 3600  # an intent whose creator died and whose instance never appeared is dropped after this


def _ledger_has(event: str, inst_id: str) -> bool:
    if not LEDGER_PATH.exists():
        return False
    recs = (json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip())
    return any(r.get("event") == event and r.get("id") == inst_id for r in recs)


def _role_of(hostname: str) -> str | None:
    """'serve'/'download' for a hostname this CLI generated, else None (not ours to sweep)."""
    m = OWN_HOSTNAME_RE.match(hostname or "")
    return None if m is None else {"serve": "serve", "dl": "download", "train": "train"}[m.group(1)]


def live_creating_hostnames(st: dict) -> set[str]:
    """Hostnames of creates whose creator process is still running."""
    return {hn for hn, rec in (st.get("pending_creates") or {}).items() if owner_alive(rec)}


def reconcile(c) -> dict:
    """Bring state in line with the account. Runs at the start of every cloud command.

    Ownership rule: this Verda account is ours alone, so every instance whose hostname
    matches OWN_HOSTNAME_RE (fhbot-<role>-<12 hex>) is ours. Such an instance that is neither the tracked
    serve/download box nor being created by a live process is an orphan: it is adopted
    as the serving box (with the config from its intent, if any) when it is a serve box
    and none is tracked and serving is wanted (serve_desired, not cancelled by serve-down),
    and deleted otherwise. Instances without the prefix are only reported, never touched.

    Also: tracked records whose instance is gone are dropped (closing the ledger row);
    a tracked download box whose owner died (and not --keep) is deleted; an intent is
    dropped once its creator is dead and its instance is tracked, deleted, or has not
    appeared within PENDING_TTL_S.

    The listing and the deletes happen outside the state lock; state changes are short
    transactions that re-read the file.
    """
    out = {"adopted": [], "deleted": [], "dropped": [], "unknown": []}
    t_list = time.time()  # records created after this may legitimately be missing from the listing
    try:
        insts = c.instances.get()
    except Exception as e:
        log(f"reconcile: cannot list instances ({type(e).__name__}: {str(e)[:120]}); skipped")
        return out
    live = {i.id: i for i in insts if i.status not in GONE_STATUSES}
    to_delete: list = []
    gone_serve: list[str] = []

    def decide(st):
        to_delete.clear()
        gone_serve.clear()
        for k in out:
            out[k].clear()
        pend = st.get("pending_creates") or {}
        creating = live_creating_hostnames(st)
        for role in ROLES:  # tracked records whose box is gone
            rec = st.get(role) or {}
            if rec.get("id") and rec["id"] not in live and rec.get("created", 0) < t_list:
                if not _ledger_has("delete", rec["id"]):
                    ledger_append(event="delete", kind="instance", id=rec["id"], reason=f"reconcile: {role} box gone")
                st.pop(role)
                out["dropped"].append(rec["id"])
                if role == "serve":
                    gone_serve.append(rec["id"])
        dl = st.get("download") or {}
        if dl.get("id") in live and not dl.get("keep") and not owner_alive(dl):
            to_delete.append(live[dl["id"]])
        tracked = {(st.get(r) or {}).get("id") for r in ROLES}
        for role in ROLES:  # tracked boxes always have a ledger row (spend accounting / budget)
            rec = st.get(role) or {}
            if rec.get("id") in live and not _ledger_has("create", rec["id"]):
                i = live[rec["id"]]
                ledger_append(ts=datetime.fromtimestamp(rec.get("created", t_list), UTC).isoformat(timespec="seconds"),
                              event="create", kind="instance", id=i.id, type=i.instance_type,
                              hostname=i.hostname, location=i.location, spot=i.is_spot,
                              price_per_hour=float(rec.get("price_per_hour") or i.price_per_hour or 0),
                              currency="eur", note="backfilled by reconcile")
        for i in live.values():  # the fh- sweep
            if i.id in tracked or i.hostname in creating:
                continue
            if _role_of(i.hostname) is None:  # not a name we generate: report only
                out["unknown"].append(i.id)
                continue
            if not _ledger_has("create", i.id):
                ledger_append(event="create", kind="instance", id=i.id, type=i.instance_type, hostname=i.hostname,
                              location=i.location, spot=i.is_spot, price_per_hour=float(i.price_per_hour or 0),
                              currency="eur", note="found by reconcile")
            intent = pend.get(i.hostname) or {}
            if (_role_of(i.hostname) == "serve" and not (st.get("serve") or {}).get("id")
                    and st.get("serve_desired", True) and not intent.get("cancelled")):
                cfg = (pend.get(i.hostname) or {}).get("record") or {}
                st["serve"] = {"id": i.id, "hostname": i.hostname, "type": i.instance_type, "spot": i.is_spot,
                               "created": (pend.get(i.hostname) or {}).get("ts", t_list), "adopted": True, **cfg}
                if "lora" in cfg:
                    st["serve_lora"] = cfg["lora"]
                tracked.add(i.id)
                out["adopted"].append(i.id)
            elif (_role_of(i.hostname) == "train" and not (st.get("train") or {}).get("id")
                  and (st.get("train_run") or {}).get("desired") and not intent.get("cancelled")
                  and (intent.get("record") or {}).get("run") == st["train_run"].get("run")):
                st["train"] = {"id": i.id, "hostname": i.hostname, "type": i.instance_type, "spot": i.is_spot,
                               "created": intent.get("ts", t_list), "adopted": True, **intent["record"]}
                tracked.add(i.id)
                out["adopted"].append(i.id)
            else:
                to_delete.append(i)
        live_hosts = {i.hostname for i in live.values()}
        tracked_hosts = {(st.get(r) or {}).get("hostname") for r in ROLES}
        for hn, rec in list(pend.items()):
            if hn in creating:
                continue
            if hn in tracked_hosts or (hn not in live_hosts and t_list - rec.get("ts", 0) > PENDING_TTL_S):
                pend.pop(hn)
        if not pend:
            st.pop("pending_creates", None)

    update_state(decide)
    for iid in out["adopted"]:
        log(f"reconcile: adopted orphaned serving instance {iid} with its requested config")
    info = tunnel_info()
    if info and info.get("inst_id") in gone_serve:
        stop_tunnel()
    for i in to_delete:
        try:
            delete_instance(c, i.id, reason="reconcile: orphaned fh- instance")
        except DeleteFailed as e:
            log(f"reconcile: {e}")  # retried by the next command
            continue
        out["deleted"].append(i.id)

        def forget(st, i=i):
            if (st.get("download") or {}).get("id") == i.id:
                st.pop("download")
            st.get("pending_creates", {}).pop(i.hostname, None)
            if not st.get("pending_creates"):
                st.pop("pending_creates", None)
        update_state(forget)
    for iid in out["unknown"]:
        i = live[iid]
        log(f"WARNING: instance {i.id} ({i.hostname}, {i.instance_type}, {i.status}) is not ours "
            f"(not tracked, name not fhbot-<role>-<id>); left alone")
    return out


def wait_volume_free(c, vol_id: str, timeout_s: int = 300) -> None:
    """A block volume can only be attached to one instance; wait until the last one let go."""
    t0 = time.time()
    while True:
        v = c.volumes.get_by_id(vol_id)
        if not v.instance_id:
            return
        try:
            holder = c.instances.get_by_id(v.instance_id)
            hs = holder.status
        except Exception:
            hs = "gone"
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"volume {vol_id} still attached to {v.instance_id} ({hs})")
        log(f"volume attached to {v.instance_id[:8]} ({hs}); waiting for detach")
        time.sleep(15)


def wait_running(c, inst_id: str, timeout_s: int = 1200):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        inst = c.instances.get_by_id(inst_id)
        if inst.status != last:
            log(f"instance {inst_id[:8]} status={inst.status} ip={inst.ip}")
            last = inst.status
        if inst.status == "running" and inst.ip:
            return inst
        if inst.status in ("error", "discontinued", "deleted"):
            raise RuntimeError(f"instance {inst_id} entered {inst.status}")
        time.sleep(10)
    raise TimeoutError(f"instance {inst_id} not running after {timeout_s}s")


def role_instances(c, role: str):
    """Instances of a role: the one tracked in state by id (whatever its hostname, e.g. a
    legacy fh-qwen-serve-* box), plus any with a generated name for that role."""
    tracked = (load_state().get(role) or {}).get("id")
    return [i for i in c.instances.get() if i.id == tracked or _role_of(i.hostname) == role]


def cmd_instances(args) -> None:
    c = client()
    for i in c.instances.get():
        print(f"{i.id}  {i.hostname:<28} {i.instance_type:<18} {i.status:<14} spot={i.is_spot} "
              f"ip={i.ip} {i.location} price={i.price_per_hour}")


def cmd_delete(args) -> None:
    c = client()
    delete_instance(c, args.id, reason="manual")


# ---- download -------------------------------------------------------------------
_VERIFY_SUMMARY_PY = f"""import glob, json
for f in sorted(glob.glob("{MNT}/models/*.verify.json")):
    d = json.load(open(f)); d.pop("files", None); print(json.dumps(d))
"""
VERIFY_SUMMARY_CMD = ("python3 -c \"$(echo " + base64.b64encode(_VERIFY_SUMMARY_PY.encode()).decode()
                      + " | base64 -d)\"")

def download_status_path(attempt: str) -> str:
    return f"{MNT}/logs/download-{attempt}.status"


def download_verified_path(attempt: str) -> str:
    return f"{MNT}/logs/download-{attempt}.verified"


def parse_verified(text: str, request: dict[str, str]) -> dict[str, str]:
    """'repo revision' lines written by the box after a successful download + check;
    keep only entries that match this attempt's request exactly."""
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and request.get(parts[0]) == parts[1]:
            out[parts[0]] = parts[1]
    return out


def parse_download_status(line: str, attempt: str) -> str:
    """'running|done|failed <attempt> <date>' -> state word, but only for *this* attempt.
    A status line from any other attempt (e.g. a stale 'done') counts as not started."""
    parts = line.split()
    if len(parts) >= 2 and parts[0] in ("running", "done", "failed") and parts[1] == attempt:
        return parts[0]
    return ""


def cmd_download(args) -> None:
    c = client()
    state = load_state()
    if not get_volume(c, state):
        sys.exit("no volume; run volume-create first")
    vol = state["volume"]
    unknown = [r for r in args.repos if r not in MODELS]
    if unknown:
        sys.exit(f"unknown repos {unknown}; known: {list(MODELS)}")
    request = {r: MODELS[r] for r in args.repos}  # exactly what this attempt must download + verify
    inst, hostnames, status, verified = None, [], "", {}
    # Refuse (without touching it) a box that is busy with, or kept for, a different request.
    rec = state.get("download") or {}
    for i in role_instances(c, "download"):
        if i.id == rec.get("id") and rec.get("request") != request and (rec.get("keep") or owner_alive(rec)):
            sys.exit(f"download box {i.id} is {'kept' if rec.get('keep') else 'in use'} for a different request "
                     f"{sorted(rec.get('request') or {})}; rerun with the same --repos, or `delete {i.id}` first")
    # Everything from provisioning on is inside try/finally, so a failure or Ctrl-C while
    # creating or booting the box still deletes it (unless --keep).
    try:
        attempt = None
        for i in role_instances(c, "download"):
            if i.id == rec.get("id") and rec.get("attempt") and rec.get("request") == request:
                inst, attempt = i, rec["attempt"]
                log(f"reusing download instance {inst.id} ({inst.status}), attempt {attempt}")

                def claim(st):
                    if (st.get("download") or {}).get("id") == i.id:
                        st["download"].update(keep=args.keep, **proc_identity())
                update_state(claim)
            else:  # unknown or different request: its status can't be trusted, so replace it
                log(f"download instance {i.id} has no matching attempt record; deleting it")
                delete_instance(c, i.id, reason="download: untracked attempt")
        if inst is None:
            attempt = secrets.token_hex(6)
            models = "\n".join(f"{r} {rev}" for r, rev in request.items())
            script = render("download.sh", {
                "__MODELS__": models, "__VLLM_VERSION__": VLLM_VERSION, "__ATTEMPT__": attempt,
                "__VERIFY_PY__": (REMOTE / "verify_weights.py").read_text()}, vol)
            types = [args.type] if args.type else DOWNLOAD_FALLBACK
            options = [(t, True) for t in types] + [(t, False) for t in types]
            if args.on_demand:
                options = [(t, False) for t in types]
            for itype, spot in options:
                hostname = new_hostname("download")
                hostnames.append(hostname)
                try:
                    inst = create_instance(c, state, role="download", itype=itype, hostname=hostname,
                                           script_name=DL_PREFIX, script=script, spot=spot, os_gb=120,
                                           record={"attempt": attempt, "request": request, "keep": args.keep,
                                                   **proc_identity()})
                    break
                except APIException as e:
                    log(f"{itype} {'spot' if spot else 'on-demand'} unavailable: {str(e).splitlines()[-1][:120]}")
            if inst is None:
                sys.exit("no CPU capacity for the download instance")
        inst = wait_running(c, inst.id)
        spath = download_status_path(attempt)
        log(f"download instance {inst.ip}; polling {spath}")
        t0 = time.time()
        while time.time() - t0 < args.timeout_min * 60:
            r = ssh(inst.ip, f"echo \"$(cat {spath} 2>/dev/null)\"; "
                             f"du -sh {MNT}/models 2>/dev/null | cut -f1; tail -n 2 /var/log/fh/startup.log",
                    alias=inst.id)
            out = r.stdout.splitlines()  # line 1 is always the (possibly empty) status line
            if out:
                status = parse_download_status(out[0], attempt)
                log(" | ".join(x[-120:] for x in out[:4]))
            if status in ("done", "failed"):
                break
            if any("FATAL" in x for x in out):
                status = "failed (FATAL in startup log)"
                break
            time.sleep(60)
        r = ssh(inst.ip, f"{VERIFY_SUMMARY_CMD}; grep -E '^\\[fh\\]' {MNT}/logs/download.log | tail -n 20",
                alias=inst.id, timeout=120)
        print(r.stdout)
        if status == "done":
            r = ssh(inst.ip, f"cat {download_verified_path(attempt)} 2>/dev/null", alias=inst.id, timeout=60)
            verified = parse_verified(r.stdout, request)
    finally:
        if not args.keep:
            reason = f"download {status or 'incomplete'}"
            with no_interrupt():
                if inst is not None:
                    delete_instance(c, inst.id, reason=reason)
                    update_state(lambda st: st.pop("download", None)
                                 if (st.get("download") or {}).get("id") == inst.id else None)
                delete_by_hostnames(c, hostnames, reason=reason, skip={inst.id} if inst is not None else set())
                for sc in c.startup_scripts.get():
                    if sc.name == DL_PREFIX:
                        c.startup_scripts.delete_by_id(sc.id)
    if verified:  # record only what this attempt verified, keeping earlier checkpoints
        def done(st):
            st.setdefault("weights", {}).update(
                {r: {"revision": rev, "path": f"{MNT}/models/{r.split('/')[1]}"} for r, rev in verified.items()})
            if status == "done":
                st["vllm_env"] = f"{MNT}/envs/vllm-{VLLM_VERSION}"
        update_state(done)
        log(f"verified: {sorted(verified)}")
    if status != "done":
        sys.exit(f"download job status: {status or 'timeout'}")
    if set(verified) != set(request):
        sys.exit(f"download reported done but verified only {sorted(verified)} of {sorted(request)}")


# ---- serving --------------------------------------------------------------------
def api_key() -> str:
    env = load_env()
    key = env.get("QWEN_API_KEY")
    if not key:
        key = "fhq-" + secrets.token_urlsafe(32)
        set_env_var("QWEN_API_KEY", key)
        log("generated QWEN_API_KEY in .env")
    return key


def pick_candidate(c, args):
    loc = load_state()["volume"]["location"]
    prices = eur_prices(c)
    spot_ok = set()
    od_ok = set()
    for l in c.instances.get_availabilities(is_spot=True, location_code=loc):
        if l["location_code"] == loc:
            spot_ok |= set(l["availabilities"])
    for l in c.instances.get_availabilities(is_spot=False, location_code=loc):
        if l["location_code"] == loc:
            od_ok |= set(l["availabilities"])
    cands = [x for x in CANDIDATES if (not args.type or x[0] == args.type)
             and (not args.variant or x[1] == args.variant)]
    opts = []
    for itype, variant, extra in cands:
        if itype not in prices:
            continue
        if itype in spot_ok:
            opts.append((float(prices[itype]["spot_price"]), itype, variant, extra, True))
        if args.on_demand and itype in od_ok:
            opts.append((float(prices[itype]["price_per_hour"]), itype, variant, extra, False))
    return sorted(opts, key=lambda o: o[0])


def vllm_args(variant: str, extra: list[str], mtp: bool, lora: str | None = None) -> list[str]:
    args = list(BASE_VLLM_ARGS)
    # later flags override earlier ones in argparse
    args += extra
    if mtp:
        args += ["--speculative-config", "'{\"method\": \"mtp\", \"num_speculative_tokens\": 2}'"]
    if lora:  # adapter dir on the instance (e.g. on the volume under /mnt/fh)
        if any(ch in lora for ch in " '\"$`;&|\\") or "=" in lora:
            raise SystemExit(f"--lora path must be a plain path, got {lora!r}")
        args += ["--enable-lora", "--max-lora-rank", str(LORA_MAX_RANK), "--lora-modules", f"{LORA_NAME}={lora}"]
    return args


def served_names(lora: str | None) -> list[str]:
    return [SERVED_NAME] + ([LORA_NAME] if lora else [])


def health(base_url: str, key: str, timeout: float = 10, names: list[str] | None = None) -> bool:
    """200 from /v1/models listing every expected model id (base, plus the LoRA if served)."""
    try:
        r = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        ids = {m["id"] for m in r.json()["data"]} if r.status_code == 200 else set()
        return r.status_code == 200 and set(names or [SERVED_NAME]) <= ids
    except Exception:
        return False


def local_base_url() -> str:
    """URL of the tunnel as seen from this box (host-side health checks, `test`)."""
    host = "127.0.0.1" if TUNNEL_BIND in ("127.0.0.1", "0.0.0.0", "localhost") else TUNNEL_BIND
    return f"http://{host}:{TUNNEL_PORT}/v1"


def container_base_url() -> str:
    """QWEN_BASE_URL: the tunnel as seen from Docker containers on this box. It does not
    depend on the instance, so it survives preemption and re-provisioning."""
    return f"http://{TUNNEL_CONTAINER_HOST}:{TUNNEL_PORT}/v1"


# ---- ssh tunnel -----------------------------------------------------------------
def tunnel_cmd(ip: str, alias: str) -> list[str]:
    return ["ssh", *ssh_opts(alias), "-N",
            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
            "-o", "ExitOnForwardFailure=yes",
            "-L", f"{TUNNEL_BIND}:{TUNNEL_PORT}:127.0.0.1:{PORT}", f"root@{ip}"]


def _pid_alive(pid: int) -> bool:
    with contextlib.suppress(ChildProcessError, OSError):
        os.waitpid(pid, os.WNOHANG)  # reap it if it is our own exited child
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def _is_tunnel_pid(pid: int) -> bool:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return (bool(argv) and argv[0].endswith(b"ssh")
            and f"{TUNNEL_BIND}:{TUNNEL_PORT}:127.0.0.1:{PORT}".encode() in argv)


def _read_pidfile() -> dict | None:
    try:
        return json.loads(TUNNEL_PIDFILE.read_text())
    except (OSError, ValueError):
        return None


def tunnel_info() -> dict | None:
    """The pidfile record if that tunnel process is alive, else None."""
    info = _read_pidfile()
    if info and _pid_alive(info["pid"]) and _is_tunnel_pid(info["pid"]):
        return info
    return None


def _port_open(host: str, port: int) -> bool:
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=1):
        return True
    return False


def stop_tunnel() -> None:
    info = _read_pidfile()
    if info and _pid_alive(info["pid"]) and _is_tunnel_pid(info["pid"]):
        os.kill(info["pid"], signal.SIGTERM)
        for _ in range(50):
            if not _pid_alive(info["pid"]):
                break
            time.sleep(0.1)
        else:
            os.kill(info["pid"], signal.SIGKILL)
        log(f"stopped tunnel pid {info['pid']} -> {info.get('ip')}")
    TUNNEL_PIDFILE.unlink(missing_ok=True)


def start_tunnel(inst_id: str, ip: str, timeout_s: float = 30) -> bool:
    """(Re)start the detached ssh -N -L process to this instance. True once it listens."""
    stop_tunnel()
    host = local_base_url().split("//")[1].rsplit(":", 1)[0]
    if _port_open(host, TUNNEL_PORT):
        raise RuntimeError(f"{host}:{TUNNEL_PORT} is in use by another process; set FH_TUNNEL_PORT")
    with TUNNEL_LOG.open("a") as lf:
        lf.write(f"--- {now_iso()} tunnel to {ip} ({inst_id})\n")
        lf.flush()
        p = subprocess.Popen(tunnel_cmd(ip, inst_id), stdin=subprocess.DEVNULL, stdout=lf, stderr=lf,
                             start_new_session=True)  # survives this CLI (and its watchdog loop)
    TUNNEL_PIDFILE.write_text(json.dumps({"pid": p.pid, "inst_id": inst_id, "ip": ip, "bind": TUNNEL_BIND,
                                          "port": TUNNEL_PORT, "started": now_iso()}) + "\n")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if p.poll() is not None:  # ssh refused / sshd not up yet / forward failed
            TUNNEL_PIDFILE.unlink(missing_ok=True)
            log(f"tunnel to {ip} exited rc={p.returncode} (see {TUNNEL_LOG.relative_to(ROOT)})")
            return False
        if _port_open(host, TUNNEL_PORT):
            log(f"tunnel up: {TUNNEL_BIND}:{TUNNEL_PORT} -> {ip}:127.0.0.1:{PORT} (pid {p.pid})")
            return True
        time.sleep(0.5)
    stop_tunnel()
    log(f"tunnel to {ip} not listening after {timeout_s:.0f}s")
    return False


def ensure_tunnel(inst_id: str, ip: str) -> bool:
    """Keep a live tunnel that points at this instance; restart a dead one, re-point a stale one."""
    info = tunnel_info()
    if info and info.get("inst_id") == inst_id and info.get("ip") == ip:
        return True
    if info:
        log(f"tunnel points at {info.get('ip')} ({str(info.get('inst_id'))[:8]}); re-pointing to {ip}")
    elif TUNNEL_PIDFILE.exists():
        log("tunnel process is dead; restarting")
    return start_tunnel(inst_id, ip)


# ---- serving --------------------------------------------------------------------
def _write_base_url() -> str:
    url = container_base_url()
    if load_env().get("QWEN_BASE_URL") != url:
        set_env_var("QWEN_BASE_URL", url)
    return url


def cmd_serve_up(args) -> None:
    with lifecycle_lock(args, busy_code=1):
        if training_active(load_state()):
            sys.exit("serve-up refused: a training run is active (train-down first, or wait for it to finish)")
        update_state(lambda st: st.__setitem__("serve_desired", True))
        c = client()
        reconcile(c)
        _serve_up(c, args)


def _serve_up(c, args) -> None:
    state = load_state()
    if not get_volume(c, state) or "weights" not in state:
        sys.exit("volume/weights missing; run volume-create and download first")
    existing = [i for i in role_instances(c, "serve") if i.status not in ("error", *GONE_STATUSES)]
    if existing:
        inst = existing[0]
        log(f"serving instance already exists: {inst.id} {inst.instance_type} {inst.status}")
        if args.no_wait:
            return
        wait_healthy(c, inst.id, args.timeout_min)  # not ours to delete on failure
        return
    opts = pick_candidate(c, args)
    if not opts:
        sys.exit("no candidate GPU available in the volume's location (try --on-demand)")
    key = api_key()
    inst, hostnames = None, []
    # Desired adapter, remembered before anything is created so serve-ensure (and an
    # adopted orphan) keeps serving it after a preemption.
    update_state(lambda st: st.__setitem__("serve_lora", args.lora))
    try:
        for price, itype, variant, extra, spot in opts:
            model_dir = state["weights"].get(VARIANT_REPO[variant], {}).get("path")
            if not model_dir:
                log(f"skip {itype}: {variant} weights not on the volume")
                continue
            va = vllm_args(variant, extra, args.mtp, args.lora)
            script = render("serve.sh", {
                "__VLLM_VERSION__": VLLM_VERSION, "__MODEL_DIR__": model_dir,
                "__SERVED_NAME__": SERVED_NAME, "__PORT__": str(PORT),
                "__VLLM_ARGS__": " ".join(va), "__API_KEY__": key}, state["volume"])
            hostname = new_hostname("serve")
            hostnames.append(hostname)
            try:
                inst = create_instance(c, state, role="serve", itype=itype, hostname=hostname,
                                       script_name=SERVE_PREFIX, script=script, spot=spot, os_gb=60,
                                       record={"variant": variant, "eur_per_h": price, "vllm_args": va,
                                               "model_dir": model_dir, "lora": args.lora})
                break
            except APIException as e:
                log(f"{itype} {'spot' if spot else 'on-demand'} unavailable: {str(e).splitlines()[-1][:120]}")
        if inst is None:
            sys.exit("all candidates failed to deploy")
        if args.no_wait:
            return
        wait_healthy(c, inst.id, args.timeout_min)
    except BaseException as e:  # includes sys.exit() from a health timeout and Ctrl-C
        if args.keep or (inst is None and not hostnames):
            raise
        log(f"serve startup failed ({type(e).__name__}: {str(e)[:160]}); deleting the new instance "
            "(use --keep to keep it)")
        with no_interrupt():
            if inst is not None:
                delete_instance(c, inst.id, reason="serve-up startup failed")
            delete_by_hostnames(c, hostnames, reason="serve-up startup failed",
                                skip={inst.id} if inst is not None else set())
            if inst is not None:
                update_state(lambda st: st.pop("serve", None)
                             if (st.get("serve") or {}).get("id") == inst.id else None)
                info = tunnel_info()
                if info and info.get("inst_id") == inst.id:
                    stop_tunnel()
        raise


def wait_healthy(c, inst_id: str, timeout_min: float) -> None:
    key = load_env()["QWEN_API_KEY"]
    inst = wait_running(c, inst_id)
    url = local_base_url()
    t0 = time.time()
    last_tail = ""
    while time.time() - t0 < timeout_min * 60:
        names = served_names((load_state().get("serve") or {}).get("lora"))
        if ensure_tunnel(inst.id, inst.ip) and health(url, key, names=names):
            cburl = _write_base_url()
            def ready(st):
                sv = st.get("serve") or {}
                if sv.get("id") == inst.id:
                    sv["ready_after_s"] = round(time.time() - sv.get("created", t0))
                    sv["base_url"] = cburl
            update_state(ready)
            log(f"healthy via tunnel {url}; QWEN_BASE_URL={cburl} (for containers)")
            return
        with contextlib.suppress(APIException, OSError):  # a failed status poll is not fatal
            cur = c.instances.get_by_id(inst.id)
            if cur.status in ("error", *GONE_STATUSES):  # e.g. spot preemption while booting
                raise RuntimeError(f"instance {inst.id} entered {cur.status} before becoming healthy")
        try:
            r = ssh(inst.ip, f"tail -n 1 {MNT}/logs/serve-$(hostname).log 2>/dev/null || tail -n 1 /var/log/fh/startup.log",
                    alias=inst.id, timeout=30)
            tail = r.stdout.strip()[-160:]
            if tail and tail != last_tail:
                log(f"... {tail}")
                last_tail = tail
        except Exception:
            pass
        time.sleep(20)
    sys.exit(f"endpoint not healthy after {timeout_min} min")


def cmd_serve_down(args) -> None:
    with lifecycle_lock(args, busy_code=1):
        # Persist the shutdown before deleting anything: from now on reconcile deletes any
        # late-arriving serve box instead of adopting it, and serve-ensure stays idle.
        def cancel(st):
            st["serve_desired"] = False
            for rec in (st.get("pending_creates") or {}).values():
                if rec.get("role") == "serve":
                    rec["cancelled"] = True
        update_state(cancel)
        c = client()
        reconcile(c)
        _serve_down(c)
        unresolved = [hn for hn, rec in (load_state().get("pending_creates") or {}).items()
                      if rec.get("role") == "serve"]
        if unresolved:
            log(f"{len(unresolved)} creates unresolved ({', '.join(unresolved)}); "
                "will be deleted when they appear (by any later command's reconcile)")


def _serve_down(c) -> None:
    failed = []
    for inst in role_instances(c, "serve"):
        try:
            delete_instance(c, inst.id, reason="serve-down")
        except DeleteFailed as e:
            log(str(e))
            failed.append(inst.id)
    if failed:  # keep state, tunnel and QWEN_BASE_URL: the box may still be running
        raise DeleteFailed(f"serve-down could not delete {failed}; state kept, rerun serve-down")
    for s in c.startup_scripts.get():
        if s.name == SERVE_PREFIX:
            c.startup_scripts.delete_by_id(s.id)
    stop_tunnel()
    remaining = {i.id for i in c.instances.get() if i.status not in GONE_STATUSES}
    # drop the serve record unless it points at a box created meanwhile by another command
    update_state(lambda st: st.pop("serve", None)
                 if (st.get("serve") or {}).get("id") not in remaining else None)
    set_env_var("QWEN_BASE_URL", "")


def cmd_serve_status(args) -> None:
    c = client()
    env = load_env()
    state = load_state()
    insts = role_instances(c, "serve")
    info = tunnel_info()
    if not insts:
        print("no serving instance")
    for i in insts:
        via = bool(info and info.get("inst_id") == i.id)
        ok = via and health(local_base_url(), env.get("QWEN_API_KEY", ""),
                            names=served_names(state.get("serve", {}).get("lora")))
        print(json.dumps({"id": i.id, "type": i.instance_type, "status": i.status, "spot": i.is_spot,
                          "ip": i.ip, "tunnel": via, "base_url": env.get("QWEN_BASE_URL"), "healthy": ok,
                          "variant": state.get("serve", {}).get("variant"),
                          "price_per_hour": i.price_per_hour}))
    print(json.dumps({"tunnel": info, "local_url": local_base_url()}))


@contextlib.contextmanager
def file_lock(path: Path, ttl_s: float, busy_code: int = 0):
    """Exclusive non-blocking lifecycle lock that survives a hung holder.

    flock is released by the kernel when the holder dies, so a dead holder never
    blocks. The holder records its pid, start time and deadline (now + ttl_s) in the
    file; a later caller that finds the lock held past deadline + LOCK_GRACE_S by the
    same live process terminates it and takes over, so the watchdog keeps working.
    """
    with path.open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.seek(0)
            try:
                holder = json.loads(f.read() or "{}")
            except ValueError:
                holder = {}
            overdue = time.time() > holder.get("deadline", float("inf")) + LOCK_GRACE_S
            if not (overdue and owner_alive(holder)):
                log(f"another serve-up/serve-ensure/serve-down is running (pid {holder.get('pid')}); exiting")
                sys.exit(busy_code)
            log(f"serve-ensure pid {holder['pid']} is past its deadline; terminating it")
            terminate_process(holder["pid"])
            for _ in range(60):
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.5)
            else:
                sys.exit(f"could not take the serve-ensure lock from pid {holder['pid']}")
        f.seek(0)
        f.truncate()
        f.write(json.dumps({**proc_identity(), "acquired": time.time(), "deadline": time.time() + ttl_s}))
        f.flush()
        yield


LOCK_GRACE_S = 300


def terminate_process(pid: int, wait_s: float = 15) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    t0 = time.time()
    while time.time() - t0 < wait_s and _proc_start(pid) is not None:
        time.sleep(0.5)
    if _proc_start(pid) is not None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


class Deadline(BaseException):
    """serve-ensure ran past its own deadline (BaseException so retry loops don't swallow it)."""


@contextlib.contextmanager
def deadline(seconds: float):
    """SIGALRM-based overall time limit for this process (main thread only)."""
    def fire(signum, frame):
        raise Deadline(f"exceeded {seconds:.0f}s deadline")
    try:
        old = signal.signal(signal.SIGALRM, fire)
    except ValueError:
        yield
        return
    signal.alarm(max(1, int(seconds)))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def lifecycle_ttl_s(args) -> float:
    waits = getattr(args, "wait", False) or (hasattr(args, "no_wait") and not args.no_wait)
    return 45 * 60 + (args.timeout_min * 60 if waits else 0)


@contextlib.contextmanager
def lifecycle_lock(args, busy_code: int):
    """One lock for serve-up, serve-ensure and serve-down: they never run concurrently.
    The holder stops itself at its deadline; a later caller kills it past that (file_lock)."""
    ttl = lifecycle_ttl_s(args)
    with file_lock(LOCK_PATH, ttl, busy_code=busy_code), deadline(ttl):
        yield


def cmd_serve_ensure(args) -> None:
    """Idempotent: healthy -> no-op; booting -> wait (or return); gone/broken -> replace.
    Also keeps the SSH tunnel alive and pointed at the current box. QWEN_BASE_URL never
    changes across re-provisioning, so running jobs pick up the new box transparently."""
    with lifecycle_lock(args, busy_code=0):
        c = client()
        reconcile(c)  # also drops the record of a preempted box and sweeps orphans
        state = load_state()
        if not state.get("serve_desired", True):
            log("serving is switched off (serve-down); nothing to ensure. Run serve-up to start again.")
            return
        if training_active(state):
            log("a training run owns the volume; serving stays off")
            return
        key = load_env().get("QWEN_API_KEY", "")
        creating = live_creating_hostnames(state)
        insts = [i for i in role_instances(c, "serve") if i.status not in GONE_STATUSES]
        for i in insts:
            if i.hostname in creating:  # a live create owns it (never ours to replace)
                log(f"{i.id} is being created by another process; leaving it alone")
                return
            if i.ip and i.status == "running":
                was_up = tunnel_info() is not None
                up = ensure_tunnel(i.id, i.ip)
                names = served_names(state.get("serve", {}).get("lora"))
                ok = up and health(local_base_url(), key, names=names)
                if up and not ok and was_up:  # rule out a wedged tunnel before judging the box
                    ok = start_tunnel(i.id, i.ip) and health(local_base_url(), key, names=names)
                if ok:
                    url = _write_base_url()
                    log(f"healthy {i.instance_type} {i.ip} via tunnel; QWEN_BASE_URL={url}")
                    return
            age = time.time() - state.get("serve", {}).get("created", 0) if state.get("serve", {}).get("id") == i.id else 1e9
            if i.status in ("error", "discontinued", "offline", "deleting") or age > STARTUP_GRACE_S:
                log(f"replacing {i.id} status={i.status} age={age:.0f}s")
                delete_instance(c, i.id, reason=f"ensure: status={i.status} age={int(age)}")
            else:
                log(f"{i.id} booting (status={i.status}, age={age:.0f}s)")
                if args.wait:
                    wait_healthy(c, i.id, args.timeout_min)
                return
        args.no_wait = not args.wait
        if args.lora is None:  # keep serving the adapter the last serve-up asked for
            args.lora = state.get("serve_lora")
        _serve_up(c, args)


# ---- LoRA training (train-up / train-ensure / train-status / train-down / train-fetch) ----
# One run at a time. state["train_run"] holds the run definition (name, trainer arguments,
# data/code hashes, budget, desired flag, restart count); state["train"] holds the tracked
# training instance (role "train", hostname fhbot-train-<hex>). The block volume attaches
# to one box at a time, so training and serving exclude each other (serve-down first).
TRAIN_DIR = ROOT / "train"
TRAIN_CODE_FILES = ("sft_lora.py", "fhdata.py", "requirements.txt", "requirements-gpu.txt")
TRAIN_REQ_FILES = ("requirements.txt", "requirements-gpu.txt")
TRAIN_BASE_REPO = "Qwen/Qwen3.8-27B"  # BF16 base; train from this, not the FP8 checkpoint
# Single-GPU types with >= 80 GB (bf16 LoRA peaks at ~70 GB) and the assumed training
# throughput for this workload in tokens/s (bf16 LoRA + recompute, batch 1 x ~14k tokens).
# H100: train/README.md estimates 1.7-2.4 h per 12.9M-token epoch; we take the slow end.
# A100: ~1/3 of H100's dense bf16 FLOPs, partly offset by memory-bound kernels -> 0.4x.
TRAIN_GPUS = {
    "1H100.80S.30V": 1500, "1H100.80S.32V": 1500, "1H200.141S.44V": 1650,
    "1A100.22V": 600, "1RTXPRO6000.30V": 700,
}
TRAIN_SETUP_H = 0.5        # boot, venv (first time only), upload, loading 55 GB of weights
TRAIN_SPOT_OVERHEAD = 1.15  # rework after spot preemptions (checkpoint every --save-every steps)
TRAIN_GRACE_S = 60 * 60    # a training box not ready/started after this is replaced
TRAIN_MAX_RESTARTS = 3     # trainer process crashes tolerated per run before giving up
TRAIN_MAX_FAILURES = 4     # failed setup/upload/start/status attempts (persisted) before giving up


class NoCapacity(RuntimeError):
    """No GPU could be created right now (transient; the next tick tries again)."""


def _sha_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def train_req_hash() -> str:
    return _sha_files([TRAIN_DIR / f for f in TRAIN_REQ_FILES])[:12]


def train_code_sha() -> str:
    return _sha_files([TRAIN_DIR / f for f in TRAIN_CODE_FILES])


def data_sha(data_dir: Path) -> str:
    """Every file under data_dir, recursively: relative path, length and contents, in sorted
    path order (the .fh_sha stamp written on the volume is excluded)."""
    h = hashlib.sha256()
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.name != ".fh_sha")
    for f in files:
        b = f.read_bytes()
        h.update(f"{f.relative_to(data_dir).as_posix()}\0{len(b)}\0".encode())
        h.update(b)
    return h.hexdigest()[:16]


def epoch_tokens(data_dir: Path) -> int:
    ts = data_dir / "token_stats.json"
    if ts.exists():
        return int(json.loads(ts.read_text())["summary"]["tokens"]["total"])
    rows = sum(1 for l in (data_dir / "sft.jsonl").open() if l.strip())
    return rows * 16000  # conservative: rows are ~14.3k tokens


def train_plan(prices: dict, spot_ok: set, od_ok: set, tokens: float, *, only_type=None,
               on_demand_only=False) -> list[dict]:
    """Options ordered by preference: spot ones by expected cost, then on-demand ones."""
    opts = []
    for itype, tps in TRAIN_GPUS.items():
        if (only_type and itype != only_type) or itype not in prices:
            continue
        hours = TRAIN_SETUP_H + tokens / tps / 3600
        if itype in spot_ok and not on_demand_only:
            eur_h = float(prices[itype]["spot_price"])
            opts.append({"type": itype, "spot": True, "eur_h": eur_h, "hours": round(hours, 2),
                         "expected_eur": round(eur_h * hours * TRAIN_SPOT_OVERHEAD, 2)})
        if itype in od_ok:
            eur_h = float(prices[itype]["price_per_hour"])
            opts.append({"type": itype, "spot": False, "eur_h": eur_h, "hours": round(hours, 2),
                         "expected_eur": round(eur_h * hours, 2)})
    return sorted(opts, key=lambda o: (not o["spot"], o["expected_eur"]))


def _availability(c, loc: str) -> tuple[set, set]:
    spot_ok, od_ok = set(), set()
    for spot, acc in ((True, spot_ok), (False, od_ok)):
        for l in c.instances.get_availabilities(is_spot=spot, location_code=loc):
            if l["location_code"] == loc:
                acc |= set(l["availabilities"])
    return spot_ok, od_ok


# Extra sft_lora.py flags allowed through --trainer-arg (exact names; the trainer's argparse
# accepts abbreviations, so anything else could silently override e.g. --model or --out).
TRAINER_EXTRA_FLAGS = {"--warmup-frac", "--max-grad-norm", "--weight-decay", "--lora-alpha", "--lora-dropout",
                       "--quant", "--attn", "--max-len", "--loss-chunk", "--eval-frac", "--keep", "--seed",
                       "--max-steps", "--limit", "--no-grad-ckpt", "--lora-r"}


class TrainFatal(RuntimeError):
    """A training problem retrying cannot fix (identity mismatch, changed inputs, budget)."""


def trainer_args(run: str, model_dir: str, data_remote: str, a) -> list[str]:
    args = ["--model", model_dir, "--data", data_remote, "--out", f"{MNT}/runs/{run}",
            "--epochs", str(a.epochs), "--lr", str(a.lr), "--grad-accum", str(a.grad_accum),
            "--lora-r", str(a.lora_r), "--save-every", str(a.save_every)]
    for extra in a.trainer_arg or []:
        if not re.fullmatch(r"[A-Za-z0-9_.=/:+-]+", extra):
            raise SystemExit(f"--trainer-arg {extra!r}: only plain flags/values are allowed")
        if extra.startswith("-") and extra.split("=", 1)[0] not in TRAINER_EXTRA_FLAGS:
            raise SystemExit(f"--trainer-arg {extra!r}: flag not allowed (allowed: {sorted(TRAINER_EXTRA_FLAGS)})")
        args.append(extra)
    validate_trainer_args(args)
    return args


def validate_trainer_args(args: list[str]) -> None:
    """Reject effective trainer arguments whose adapter vLLM (serve-up --lora) cannot load."""
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-linear-attn", action="store_true")
    ns, _ = p.parse_known_args(args)  # last occurrence wins, as in the trainer
    if not 1 <= ns.lora_r <= LORA_MAX_RANK:
        raise SystemExit(f"--lora-r {ns.lora_r}: serving uses --max-lora-rank {LORA_MAX_RANK}, "
                         f"so the rank must be 1..{LORA_MAX_RANK}")
    if ns.lora_linear_attn:
        raise SystemExit("--lora-linear-attn adapters are not loadable by vLLM's packed in_proj_qkvz; not allowed")
    for flag in ("--model", "--data", "--out"):
        n = sum(1 for x in args if x == flag or x.startswith(flag + "="))
        if n != 1:
            raise SystemExit(f"{flag} must appear exactly once in the trainer arguments")


def run_manifest(run: dict) -> dict:
    """The immutable identity of a run, stored once at /mnt/fh/runs/<run>/manifest.json."""
    return {k: run[k] for k in ("run", "data_sha", "code_sha", "req_hash", "trainer_args")}


# Files whose presence means the output dir already holds (part of) a run.
RUN_STATE_GLOBS = "{out}/ckpt-* {out}/.tmp-ckpt-* {out}/final {out}/DONE {out}/args.json {out}/loss.jsonl"
REMOTE_HASHES_CMD = ("printf 'data=%s\\ncode=%s\\nmanifest=%s\\nrunstate=%s\\n' \"$(cat {data}/.fh_sha 2>/dev/null)\" "
                     "\"$(cat {out}/code/.fh_sha 2>/dev/null)\" \"$(base64 -w0 {out}/manifest.json 2>/dev/null)\" "
                     "\"$(ls -d " + RUN_STATE_GLOBS + " 2>/dev/null | wc -l)\"")


def parse_labelled(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        k, sep, v = line.partition("=")
        if sep:
            out[k.strip()] = v.strip()
    return out


def remote_run_identity(ip: str, alias: str, run: dict) -> dict:
    """{'data': sha|'', 'code': sha|'', 'manifest': dict|None} as found on the volume."""
    out = f"{MNT}/runs/{run['run']}"
    r = ssh(ip, REMOTE_HASHES_CMD.replace("{out}", out).replace("{data}", run["data_remote"]),
            alias=alias, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"reading run identity failed: {r.stderr.strip()[:200]}")
    kv = parse_labelled(r.stdout)
    if "data" not in kv or "code" not in kv:
        raise RuntimeError(f"unexpected identity output: {r.stdout[:200]!r}")
    m = None
    if kv.get("manifest"):
        try:
            m = json.loads(base64.b64decode(kv["manifest"]))
        except ValueError as e:
            raise TrainFatal(f"unreadable manifest.json on the volume: {e}") from e
    runstate = kv.get("runstate", "")
    if not runstate.isdigit():
        raise RuntimeError(f"unexpected identity output: {r.stdout[:200]!r}")
    return {"data": kv["data"], "code": kv["code"], "manifest": m, "runstate": int(runstate)}


def check_manifest(found: dict | None, run: dict, *, required: bool = False) -> None:
    if found is None:
        if required:
            raise TrainFatal(f"/mnt/fh/runs/{run['run']} has no manifest.json, so its checkpoints/adapter can't be "
                             "matched to this run definition; use a new --run")
        return
    if found != run_manifest(run):
        diff = sorted(k for k in set(found) | set(run_manifest(run)) if found.get(k) != run_manifest(run).get(k))
        raise TrainFatal(f"/mnt/fh/runs/{run['run']} belongs to a run with different {diff}; use a new --run")


def _ssh_pipe(ip: str, alias: str, remote_cmd: str, local_cmd: list[str], timeout: int = 1800,
              upload: bool = True) -> None:
    """tar stream between this box and the instance over the pinned SSH connection."""
    ssh_cmd = ["ssh", *ssh_opts(alias), f"root@{ip}", remote_cmd]
    src, dst = (local_cmd, ssh_cmd) if upload else (ssh_cmd, local_cmd)
    p1 = subprocess.Popen(src, stdout=subprocess.PIPE)
    p2 = subprocess.Popen(dst, stdin=p1.stdout, stdout=subprocess.DEVNULL)
    p1.stdout.close()
    try:
        rc2 = p2.wait(timeout=timeout)
        rc1 = p1.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p1.kill()
        p2.kill()
        raise RuntimeError(f"transfer to/from {ip} timed out")
    if rc1 or rc2:
        raise RuntimeError(f"transfer to/from {ip} failed (rc {rc1}/{rc2})")


def _upload_dir(ip: str, alias: str, src: Path, members: list[str], dest: str, sha: str) -> None:
    """Atomically replace dest on the volume with members of src; stamp it with sha."""
    remote = (f"set -e; rm -rf {dest}.tmp; mkdir -p {dest}.tmp; tar -C {dest}.tmp -xf -; "
              f"echo {sha} > {dest}.tmp/.fh_sha; rm -rf {dest}; mv {dest}.tmp {dest}; sync")
    _ssh_pipe(ip, alias, remote, ["tar", "-C", str(src), "-cf", "-", *members])


def train_prepare(ip: str, alias: str, run: dict) -> None:
    """Make the volume hold exactly this run's data and code, and its manifest.

    The manifest is written once (first start) and checked on every later start: a
    resume always uses the data, code and arguments the run began with. A copy on the
    volume that already matches is used even if the local files changed since; a missing
    or different copy is uploaded only if the local files still match the run."""
    ident = remote_run_identity(ip, alias, run)
    if ident["manifest"] is None and ident["runstate"]:
        raise TrainFatal(f"/mnt/fh/runs/{run['run']} already holds checkpoints/logs but no manifest.json "
                         "(a run from before manifests, or a foreign one): refusing to resume it. "
                         "Use a new --run name (or move that directory away on the volume).")
    check_manifest(ident["manifest"], run)
    if ident["data"] != run["data_sha"]:
        local = Path(run["data_dir"])
        if data_sha(local) != run["data_sha"]:
            raise TrainFatal(f"{local} changed since the run started (sha {run['data_sha']}) and the volume has "
                             "no matching copy; start a new --run")
        log(f"uploading {local} -> {run['data_remote']}")
        _upload_dir(ip, alias, local, ["."], run["data_remote"], run["data_sha"])
    out = f"{MNT}/runs/{run['run']}"
    if ident["code"] != run["code_sha"]:
        if train_code_sha() != run["code_sha"]:
            raise TrainFatal(f"train/ code changed since the run started (sha {run['code_sha']}) and the volume "
                             "has no matching copy; start a new --run")
        log(f"uploading trainer code -> {out}/code")
        _upload_dir(ip, alias, TRAIN_DIR, list(TRAIN_CODE_FILES), f"{out}/code", run["code_sha"])
    if ident["manifest"] is None:  # fresh output dir only (checked above, and again on the box)
        b64 = base64.b64encode(json.dumps(run_manifest(run), sort_keys=True).encode()).decode()
        fresh = "! ls -d " + RUN_STATE_GLOBS.replace("{out}", out) + " >/dev/null 2>&1"
        r = ssh(ip, f"mkdir -p {out} && {{ test -f {out}/manifest.json || {{ {fresh} && "
                    f"echo {b64} | base64 -d > {out}/.manifest.tmp && mv -n {out}/.manifest.tmp {out}/manifest.json; }}; }}",
                alias=alias, timeout=60)
        if r.returncode != 0:
            raise RuntimeError(f"writing manifest failed: {r.stderr.strip()[:200] or 'output dir not fresh'}")
        check_manifest(remote_run_identity(ip, alias, run)["manifest"], run, required=True)


TRAIN_STATUS_CMD = """OUT={out}
echo "run=$(cat /var/lib/fh/train-run 2>/dev/null)"
echo "ready=$(test -f /var/lib/fh/train-ready && echo 1 || echo 0)"
echo "service=$(systemctl is-active fh-train 2>/dev/null || true)"
echo "exit=$(systemctl show fh-train -p ExecMainStatus --value 2>/dev/null)"
echo "done=$(test -f $OUT/DONE && echo 1 || echo 0)"
echo "fatal=$(grep -c 'FATAL' /var/log/fh/startup.log 2>/dev/null || echo 0)"
echo "total_steps=$(grep -o '[0-9]* optimizer steps' $OUT/train.log 2>/dev/null | tail -n 1 | cut -d' ' -f1)"
echo "eval=$(tail -n 1 $OUT/eval.jsonl 2>/dev/null)"
tail -n 10 $OUT/loss.jsonl 2>/dev/null | sed 's/^/loss=/'
"""


def parse_train_status(text: str) -> dict:
    st = {"run": None, "ready": False, "service": "", "exit": "", "done": False, "fatal": 0, "total_steps": None,
          "eval": None, "losses": []}
    for line in text.splitlines():
        k, _, v = line.partition("=")
        v = v.strip()
        if k == "loss" and v:
            with contextlib.suppress(ValueError):
                st["losses"].append(json.loads(v))
        elif k in ("ready", "done"):
            st[k] = v == "1"
        elif k == "fatal":
            st[k] = int(v or 0) if v.isdigit() else 0
        elif k == "total_steps":
            st[k] = int(v) if v.isdigit() else None
        elif k == "eval" and v:
            with contextlib.suppress(ValueError):
                st["eval"] = json.loads(v)
        elif k in ("service", "exit"):
            st[k] = v
        elif k == "run":
            st[k] = v or None
    last = st["losses"][-1] if st["losses"] else None
    st["step"] = last["step"] if last else 0
    st["loss"] = last["loss"] if last else None
    secs = [r["seconds"] for r in st["losses"][-10:] if "seconds" in r]
    st["sec_per_step"] = sum(secs) / len(secs) if secs else None
    st["eta_h"] = (round((st["total_steps"] - st["step"]) * st["sec_per_step"] / 3600, 2)
                   if st["total_steps"] and st["sec_per_step"] else None)
    return st


def train_remote_status(ip: str, alias: str, run: str) -> dict:
    r = ssh(ip, TRAIN_STATUS_CMD.format(out=f"{MNT}/runs/{run}"), alias=alias, timeout=60)
    if r.returncode != 0 and "ready=" not in r.stdout:
        raise RuntimeError(f"ssh status failed: {r.stderr.strip()[:200]}")
    return parse_train_status(r.stdout)


def train_fetch_from(ip: str, alias: str, run: str) -> Path:
    """Copy the final adapter (without the processor copy) and the logs to train/out/<run>/."""
    dest = TRAIN_DIR / "out" / run
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    out = f"{MNT}/runs/{run}"
    remote = (f"cd {out} && tar -cf - --exclude=final/processor final loss.jsonl "
              f"$(ls eval.jsonl args.json DONE 2>/dev/null)")
    _ssh_pipe(ip, alias, remote, ["tar", "-C", str(tmp), "-xf", "-"], timeout=1800, upload=False)
    st_file = tmp / "final" / "adapter_model.safetensors"
    if not st_file.exists() or st_file.stat().st_size == 0 or not (tmp / "final" / "adapter_config.json").exists():
        raise RuntimeError(f"fetched adapter for {run} is incomplete")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return dest


def _train_box_ip(c) -> tuple[str, str] | None:
    """(ip, instance id) of a running tracked box that has the volume (train or serve)."""
    st = load_state()
    for role in ("train", "serve"):
        rec = st.get(role) or {}
        if rec.get("id"):
            with contextlib.suppress(Exception):
                i = c.instances.get_by_id(rec["id"])
                if i.status == "running" and i.ip:
                    return i.ip, i.id
    return None


def _train_mark(**kw) -> None:
    update_state(lambda st: st.get("train_run", {}).update(kw) if st.get("train_run") else None)


def _train_box_mark(box_id: str, **kw) -> None:
    update_state(lambda st: st["train"].update(kw) if (st.get("train") or {}).get("id") == box_id else None)


def _train_provision(c, run: dict, *, on_demand_only=False, only_type=None) -> str:
    """Create a training box for the run (budget-checked). Returns the instance id."""
    st = load_state()
    loc = st["volume"]["location"]
    prices = eur_prices(c)
    spot_ok, od_ok = _availability(c, loc)
    remaining = run["tokens"] * (1 - float(run.get("progress", 0)))
    opts = train_plan(prices, spot_ok, od_ok, remaining, only_type=only_type, on_demand_only=on_demand_only)
    if not opts:
        raise NoCapacity("no 80 GB GPU available for training in the volume's location")
    spent = ledger_total_eur()
    script = render("train.sh", {"__RUN__": run["run"], "__REQ_HASH__": run["req_hash"],
                                 "__REQS__": "\n".join((TRAIN_DIR / f).read_text() for f in TRAIN_REQ_FILES),
                                 "__TRAIN_ARGS__": " ".join(run["trainer_args"])}, st["volume"])
    for o in opts:
        if spent + o["expected_eur"] > run["max_eur"]:
            log(f"skip {o['type']} {'spot' if o['spot'] else 'on-demand'}: ledger €{spent:.2f} + expected "
                f"€{o['expected_eur']:.2f} > --max-eur {run['max_eur']}")
            continue
        log(f"training on {o['type']} {'spot' if o['spot'] else 'on-demand'}: €{o['eur_h']}/h x {o['hours']} h "
            f"-> expected €{o['expected_eur']:.2f} (ledger so far €{spent:.2f}, cap €{run['max_eur']})")
        try:
            inst = create_instance(c, st, role="train", itype=o["type"], hostname=new_hostname("train"),
                                   script_name=TRAIN_PREFIX, script=script, spot=o["spot"], os_gb=100,
                                   record={"run": run["run"], "expected_eur": o["expected_eur"]})
            return inst.id
        except APIException as e:
            log(f"{o['type']} unavailable: {str(e).splitlines()[-1][:120]}")
    if all(spent + o["expected_eur"] > run["max_eur"] for o in opts):
        raise TrainFatal(f"no training option fits the budget (--max-eur {run['max_eur']}, ledger €{spent:.2f})")
    raise NoCapacity("all training creates failed (no capacity right now)")


def _drop_train_box(box_id: str) -> None:
    update_state(lambda s: s.pop("train", None) if (s.get("train") or {}).get("id") == box_id else None)


def train_tick(c) -> str:
    """One idempotent step of the training state machine. Every outcome is bounded:
    the budget is checked first on every tick; failed attempts are counted in state and
    the run is given up (box deleted, data and checkpoints kept on the volume) after
    TRAIN_MAX_FAILURES; after DONE the box is always deleted.

    Returns: idle | provisioned | no-capacity | booting | setup | running | restarted |
    retry | done | failed | budget."""
    st = load_state()
    run = st.get("train_run") or {}
    box = st.get("train") or {}
    active = bool(run.get("desired")) and not run.get("done")
    if box.get("id") and (not active or box.get("run") != run.get("run")):
        return _train_cleanup(c, box)  # box of an inactive run, or of another run: delete it
    if run.get("unverified"):
        return _train_retry_verify(c, run)
    if active and ledger_total_eur() >= run["max_eur"]:
        return _train_stop_budget(c, box, run, ledger_total_eur())
    if not active:
        return "idle"
    if not box.get("id"):
        try:
            _train_provision(c, run)
        except NoCapacity as e:
            log(str(e))
            return "no-capacity"
        except TrainFatal as e:
            _train_mark(desired=False, failed=str(e))
            log(f"training {run['run']} disabled: {e}")
            return "failed"
        return "provisioned"
    age = time.time() - box.get("created", 0)
    try:
        inst = c.instances.get_by_id(box["id"])
    except APIException as e:
        if _not_found(e):
            return "booting"  # reconcile drops the record next time
        raise
    if inst.status in ("error", "offline", *GONE_STATUSES) or (inst.status != "running" and age > TRAIN_GRACE_S):
        log(f"training box {inst.id} is {inst.status} (age {age:.0f}s); replacing it")
        delete_instance(c, inst.id, reason=f"train: box {inst.status}")
        _drop_train_box(inst.id)
        return _train_count_failure(c, None, run, f"box {inst.status}")
    if inst.status != "running" or not inst.ip:
        return "booting"
    try:
        rs = train_remote_status(inst.ip, inst.id, run["run"])
    except Exception as e:
        log(f"training box {inst.id}: {e}")
        if age > TRAIN_GRACE_S:
            return _train_count_failure(c, inst, run, f"unreachable: {e}", replace=True)
        return "booting"
    if rs["total_steps"]:
        _train_mark(progress=round(rs["step"] / rs["total_steps"], 4), step=rs["step"],
                    total_steps=rs["total_steps"], loss=rs["loss"])
    if rs["run"] != run["run"]:  # the unit on the box runs something else
        log(f"training box {inst.id} runs {rs['run']!r}, not {run['run']!r}; deleting it")
        return _train_cleanup(c, box)
    if rs["done"]:
        return _train_finish(c, inst, run)
    if rs["fatal"]:
        return _train_count_failure(c, inst, run, "box setup failed (FATAL in startup log)", replace=True)
    if not rs["ready"]:
        if age > TRAIN_GRACE_S:
            return _train_count_failure(c, inst, run, f"box not ready after {TRAIN_GRACE_S // 60} min",
                                        replace=True)
        return "setup"
    if rs["service"] in ("active", "activating", "deactivating"):
        if run.get("failures"):
            _train_mark(failures=0)
        log(f"training {run['run']}: step {rs['step']}/{rs['total_steps'] or '?'} loss {rs['loss']} "
            f"ETA {rs['eta_h']} h")
        return "running"
    restarted = bool(box.get("started"))
    if restarted:  # the trainer exited without DONE: crash, OOM, or SIGTERM (143)
        n = int(run.get("restarts", 0))
        if n >= TRAIN_MAX_RESTARTS:
            return _train_fail(c, inst, run, f"trainer exited (status {rs['exit']}) {n + 1} times")
        log(f"trainer exited with status {rs['exit']}; restart {n + 1}/{TRAIN_MAX_RESTARTS} (resumes)")
        _train_mark(restarts=n + 1)
    try:
        train_prepare(inst.ip, inst.id, run)
        r = ssh(inst.ip, "systemctl reset-failed fh-train 2>/dev/null; systemctl start fh-train && "
                         "systemctl is-active fh-train", alias=inst.id, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"could not start fh-train: {r.stdout.strip()} {r.stderr.strip()[:200]}")
    except TrainFatal as e:
        return _train_fail(c, inst, run, str(e))
    except Exception as e:
        return _train_count_failure(c, inst, run, f"upload/start: {e}")
    _train_box_mark(inst.id, started=True)
    log(f"started fh-train for {run['run']} on {inst.id}")
    return "restarted" if restarted else "running"


def _train_cleanup(c, box: dict) -> str:
    """Delete the tracked box of a run that is no longer active. Tried on every tick
    (with delete_instance's own bounded backoff) until it is confirmed gone."""
    log(f"training box {box['id']} belongs to an inactive run; deleting it")
    try:
        delete_instance(c, box["id"], reason="train: run inactive", attempts=3, backoff_s=2.0)
    except DeleteFailed as e:
        log(f"cleanup of {box['id']} failed ({e}); retrying on the next tick")
        return "cleanup-retry"
    _drop_train_box(box["id"])
    return "cleaned"


def _train_count_failure(c, inst, run: dict, why: str, replace: bool = False) -> str:
    """Persist one more failed attempt; give up after TRAIN_MAX_FAILURES. With replace, the
    box is deleted now (the next tick provisions a fresh one)."""
    n = int(load_state().get("train_run", {}).get("failures", 0)) + 1
    _train_mark(failures=n, last_failure=why)
    log(f"training {run['run']}: {why} (failure {n}/{TRAIN_MAX_FAILURES})")
    if n >= TRAIN_MAX_FAILURES:
        if inst is not None:
            return _train_fail(c, inst, run, f"{n} failures, last: {why}")
        _train_mark(desired=False, failed=f"{n} failures, last: {why}")
        return "failed"
    if replace and inst is not None:
        delete_instance(c, inst.id, reason=f"train: {why[:80]}")
        _drop_train_box(inst.id)
    return "retry"


def _train_fail(c, inst, run: dict, why: str) -> str:
    log(f"training {run['run']} failed: {why}; deleting the box (data, code and checkpoints stay on the volume)")
    delete_instance(c, inst.id, reason=f"train failed: {why[:80]}")
    _drop_train_box(inst.id)
    _train_mark(desired=False, failed=why, checkpoints=f"{MNT}/runs/{run['run']}")
    return "failed"


def _train_stop_budget(c, box: dict, run: dict, spent: float) -> str:
    """Budget exhausted: ask the trainer to checkpoint (bounded wait), delete the box and
    disable the run. Checkpoints stay on the volume; train-up with a higher --max-eur resumes."""
    why = f"budget: ledger €{spent:.2f} >= --max-eur {run['max_eur']}"
    log(f"training {run['run']} stopped: {why}")
    if box.get("id"):
        with contextlib.suppress(Exception):
            inst = c.instances.get_by_id(box["id"])
            if inst.status == "running" and inst.ip:
                ssh(inst.ip, "timeout 300 systemctl stop fh-train", alias=inst.id, timeout=330)
        delete_instance(c, box["id"], reason="train: budget exhausted")
        _drop_train_box(box["id"])
    _train_mark(desired=False, disabled=why, checkpoints=f"{MNT}/runs/{run['run']}")
    return "budget"


def _publish_copy_fetch(ip: str, alias: str, name: str) -> dict:
    """Copy runs/<run>/final to adapters/<run> on the volume and fetch it here (best effort)."""
    res = {"adapter": None, "fetched": None, "error": None}
    r = ssh(ip, f"set -e; test -f {MNT}/runs/{name}/final/adapter_model.safetensors; mkdir -p {MNT}/adapters; "
                f"rm -rf {MNT}/adapters/{name}.tmp; cp -r {MNT}/runs/{name}/final {MNT}/adapters/{name}.tmp; "
                f"rm -rf {MNT}/adapters/{name}; mv {MNT}/adapters/{name}.tmp {MNT}/adapters/{name}; sync",
            alias=alias, timeout=600)
    if r.returncode != 0:
        res["error"] = f"publish failed: {r.stderr.strip()[:200] or r.returncode}"
    else:
        res["adapter"] = f"{MNT}/adapters/{name}"
    try:
        res["fetched"] = str(train_fetch_from(ip, alias, name).relative_to(ROOT))
    except Exception as e:
        res["error"] = (res["error"] + "; " if res["error"] else "") + f"fetch failed: {e}"
    return res


def train_verify_and_publish(ip: str, alias: str, run: dict) -> str:
    """Accept a DONE only after the volume's manifest was read and matches the run.

    - manifest read failed (ssh/API error): run stays 'unverified' (retryable through any
      box: next tick or train-publish), never 'done'.
    - manifest missing/different, or no DONE marker: run failed (DONE rejected).
    - verified: run done; then publish + fetch once (a failure there is recorded for
      train-publish; the run is still done because its identity was verified).
    Returns 'done' | 'unverified' | 'failed'."""
    name = run["run"]
    try:
        ident = remote_run_identity(ip, alias, run)
        check_manifest(ident["manifest"], run, required=True)
        r = ssh(ip, f"test -f {MNT}/runs/{name}/DONE", alias=alias, timeout=60)
        if r.returncode == 1:
            raise TrainFatal(f"no DONE marker in {MNT}/runs/{name}")
        if r.returncode != 0:
            raise RuntimeError(f"DONE check failed (ssh rc {r.returncode})")
    except TrainFatal as e:
        _train_mark(desired=False, unverified=False, failed=f"DONE rejected: {e}", finished=now_iso())
        log(f"training {name}: DONE not accepted ({e})")
        return "failed"
    except Exception as e:
        _train_mark(desired=False, unverified=True, last_failure=f"verification failed: {e}")
        log(f"training {name}: could not verify DONE ({e}); kept as unverified, retried via any box "
            "(next train-ensure / train-publish)")
        return "unverified"
    res = _publish_copy_fetch(ip, alias, name)
    _train_mark(done=True, unverified=False, desired=False, progress=1.0, adapter=res["adapter"],
                fetched=res["fetched"], final=f"{MNT}/runs/{name}/final", publish_error=res["error"],
                finished=now_iso())
    if res["error"]:
        log(f"training {name} done (verified); {res['error']}. The adapter is at {MNT}/runs/{name}/final; "
            "run `train-publish` while any box has the volume")
    else:
        log(f"training {name} done; adapter {res['adapter']} (serve-up --lora {res['adapter']}); "
            f"local copy: {res['fetched']}")
    return "done"


def _train_finish(c, inst, run: dict) -> str:
    """DONE seen on the box: verify + publish through it, then ALWAYS delete the box.
    serve_desired is left as it is."""
    phase = "unverified"
    try:
        phase = train_verify_and_publish(inst.ip, inst.id, run)
    finally:
        delete_instance(c, inst.id, reason=f"train {run['run']} done ({phase})")
        _drop_train_box(inst.id)
    return phase


def _train_retry_verify(c, run: dict) -> str:
    box = _train_box_ip(c)
    if not box:
        return "unverified"  # waits for any box with the volume (serving or training)
    return train_verify_and_publish(box[0], box[1], run)


def _serving_blocks_training(st: dict) -> str | None:
    if st.get("serve_desired", True):
        return "serving is switched on; run serve-down first (the volume attaches to one box at a time)"
    if (st.get("serve") or {}).get("id"):
        return f"serving instance {st['serve']['id']} is still tracked; run serve-down first"
    return None


def training_active(st: dict) -> bool:
    run = st.get("train_run") or {}
    return bool(run.get("desired")) and not run.get("done")


def cmd_train_up(args) -> None:
    with lifecycle_lock(args, busy_code=1):
        c = client()
        reconcile(c)
        st = load_state()
        why = _serving_blocks_training(st)
        if why:
            sys.exit(f"train-up refused: {why}")
        model_dir = (st.get("weights") or {}).get(TRAIN_BASE_REPO, {}).get("path")
        if not model_dir:
            sys.exit(f"{TRAIN_BASE_REPO} weights are not on the volume; run download first")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.run):
            sys.exit("--run must be [A-Za-z0-9_.-]{1,64}")
        data_dir = Path(args.data_dir).resolve()
        dsha = data_sha(data_dir)
        data_remote = f"{MNT}/data/{data_dir.name}-{dsha}"  # per content: runs never overwrite each other's data
        targs = trainer_args(args.run, model_dir, data_remote, args)  # validated for vLLM loading
        run = {"run": args.run, "trainer_args": targs, "data_dir": str(data_dir), "data_name": data_dir.name,
               "data_remote": data_remote, "data_sha": dsha, "code_sha": train_code_sha(),
               "req_hash": train_req_hash(), "tokens": epoch_tokens(data_dir) * float(args.epochs),
               "max_eur": float(args.max_eur), "desired": True, "restarts": 0, "failures": 0, "created": now_iso()}
        spent = ledger_total_eur()
        if spent >= float(args.max_eur):
            sys.exit(f"train-up refused: ledger €{spent:.2f} already >= --max-eur {args.max_eur}")
        prev = st.get("train_run") or {}
        tbox = st.get("train") or {}
        if tbox.get("id") and tbox.get("run") != args.run:
            sys.exit(f"training box {tbox['id']} of run {tbox.get('run')!r} is still tracked; wait until "
                     "train-ensure/train-down has confirmed its deletion before starting another run")
        if prev.get("run") == args.run and prev.get("unverified"):
            sys.exit(f"run {args.run} finished but is unverified; run train-publish (needs any box with the volume)")
        if prev.get("run") == args.run:
            same = {k: prev.get(k) == run[k] for k in ("trainer_args", "data_sha", "code_sha", "req_hash")}
            if not all(same.values()):
                sys.exit(f"run {args.run} exists with different {[k for k, v in same.items() if not v]}; "
                         "use a new --run name")
            if prev.get("done"):
                sys.exit(f"run {args.run} is already done (adapter {prev.get('adapter')})")
            run = {**prev, "desired": True, "max_eur": float(args.max_eur), "failed": None, "disabled": None,
                   "failures": 0, "restarts": 0}
            log(f"resuming run {args.run} (progress {prev.get('progress', 0):.0%})")
        elif training_active(st):
            sys.exit(f"run {prev.get('run')} is active; train-down it first")
        update_state(lambda s: s.__setitem__("train_run", run))
        created = None
        try:
            if not (load_state().get("train") or {}).get("id"):
                try:
                    created = _train_provision(c, load_state()["train_run"], on_demand_only=args.on_demand,
                                               only_type=args.type)
                except (NoCapacity, TrainFatal) as e:
                    _train_mark(desired=False)
                    sys.exit(f"train-up: {e}")
            if args.no_wait:
                return
            t0 = time.time()
            while time.time() - t0 < args.timeout_min * 60:
                phase = train_tick(c)
                if phase in ("running", "restarted", "done"):
                    log(f"train-up: {phase}; keep `train-ensure` running in a loop (watchdog)")
                    return
                if phase in ("failed", "idle", "budget"):
                    sys.exit(f"train-up: {phase} ({load_state()['train_run'].get('failed') or ''})")
                time.sleep(30)
            sys.exit(f"training did not start within {args.timeout_min} min")
        except BaseException as e:
            if args.keep or created is None:
                raise
            log(f"train-up failed ({type(e).__name__}: {str(e)[:160]}); deleting the new box (use --keep to keep it)")
            with no_interrupt():
                delete_instance(c, created, reason="train-up failed")
                update_state(lambda s: s.pop("train", None) if (s.get("train") or {}).get("id") == created else None)
                _train_mark(desired=False, failed=f"train-up: {type(e).__name__}")
            raise


def cmd_train_ensure(args) -> None:
    """Watchdog step: provision / set up / start / restart / finish the desired run."""
    with lifecycle_lock(args, busy_code=0):
        c = client()
        reconcile(c)
        phase = train_tick(c)
        log(f"train-ensure: {phase}")


def cmd_train_down(args) -> None:
    with lifecycle_lock(args, busy_code=1):
        def cancel(st):
            if st.get("train_run"):
                st["train_run"]["desired"] = False
            for rec in (st.get("pending_creates") or {}).values():
                if rec.get("role") == "train":
                    rec["cancelled"] = True
        update_state(cancel)
        c = client()
        reconcile(c)
        failed = []
        for inst in role_instances(c, "train"):
            if inst.status in GONE_STATUSES:
                continue
            try:  # the trainer gets SIGTERM via shutdown and checkpoints if it can; the run resumes later
                delete_instance(c, inst.id, reason="train-down")
                update_state(lambda s, i=inst: s.pop("train", None)
                             if (s.get("train") or {}).get("id") == i.id else None)
            except DeleteFailed as e:
                log(str(e))
                failed.append(inst.id)
        if failed:
            raise DeleteFailed(f"train-down could not delete {failed}; rerun train-down")
        unresolved = [hn for hn, rec in (load_state().get("pending_creates") or {}).items()
                      if rec.get("role") == "train"]
        if unresolved:
            log(f"{len(unresolved)} creates unresolved; will be deleted when they appear")
        log("training stopped; `train-up` with the same --run and arguments resumes from the last checkpoint")


def cmd_train_status(args) -> None:
    st = load_state()
    run = st.get("train_run")
    if not run:
        print("no training run")
        return
    info = {k: run.get(k) for k in ("run", "desired", "done", "failed", "progress", "step", "total_steps",
                                    "loss", "restarts", "max_eur", "adapter", "fetched")}
    info["trainer_args"] = " ".join(run["trainer_args"])
    info["ledger_eur"] = round(ledger_total_eur(), 2)
    box = st.get("train") or {}
    info["box"] = {k: box.get(k) for k in ("id", "type", "spot", "started", "expected_eur")} if box else None
    if box.get("id"):
        c = client()
        with contextlib.suppress(Exception):
            inst = c.instances.get_by_id(box["id"])
            info["box"]["status"] = inst.status
            if inst.status == "running" and inst.ip:
                rs = train_remote_status(inst.ip, inst.id, run["run"])
                info.update({k: rs[k] for k in ("ready", "service", "done", "step", "total_steps", "loss",
                                                "sec_per_step", "eta_h", "eval")})
    print(json.dumps(info, indent=1))


def cmd_train_publish(args) -> None:
    """Retry publication of a finished run's adapter (copy to /mnt/fh/adapters/<run> and fetch
    it here) through whichever tracked box has the volume (training or serving)."""
    with lifecycle_lock(args, busy_code=1):
        c = client()
        reconcile(c)
        run = load_state().get("train_run") or {}
        if not run or (args.run and args.run != run.get("run")):
            sys.exit("only the current train_run can be published (its manifest is checked)")
        if not (run.get("done") or run.get("unverified")):
            sys.exit(f"run {run['run']} has not finished")
        box = _train_box_ip(c)
        if not box:
            sys.exit(f"no running box has the volume attached; the adapter is at {MNT}/runs/{run['run']}/final. "
                     "Run this again while the serving box is up (serve-up) or during another training run.")
        phase = train_verify_and_publish(box[0], box[1], run)
        run = load_state()["train_run"]
        if phase != "done" or run.get("publish_error"):
            sys.exit(f"train-publish: {phase}: {run.get('publish_error') or run.get('failed') or run.get('last_failure')}")
        log(f"published {run['adapter']}; local copy {run['fetched']}")


def cmd_train_fetch(args) -> None:
    st = load_state()
    name = args.run or (st.get("train_run") or {}).get("run")
    if not name:
        sys.exit("no run given")
    box = _train_box_ip(client())
    if not box:
        sys.exit("no running box has the volume attached (training box or serving box); the adapter is at "
                 f"{MNT}/adapters/{name} once training finished")
    dest = train_fetch_from(box[0], box[1], name)
    log(f"fetched {name} -> {dest.relative_to(ROOT)}")
    if (st.get("train_run") or {}).get("run") == name:
        _train_mark(fetched=str(dest.relative_to(ROOT)))


# ---- smoke test -----------------------------------------------------------------
def test_png_data_url(text: str) -> str:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    d = ImageDraw.Draw(img)
    font = None
    for fp in (ROOT / "framehunt/assets/fonts").glob("DejaVuSans*.ttf"):
        font = ImageFont.truetype(str(fp), 48)
        break
    d.text((20, 50), text, fill="black", font=font or ImageFont.load_default(size=48))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(url, key, body) -> tuple[dict, float]:
    t0 = time.time()
    r = httpx.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                   json=body, timeout=600)
    r.raise_for_status()
    return r.json(), time.time() - t0


def stream_tps(url, key, max_tokens=512) -> dict:
    """Streams a text completion to measure time-to-first-token and decode tokens/s."""
    body = {"model": SERVED_NAME, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0.7, "top_p": 0.8,
            "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True,
            "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}]}
    t0 = time.time()
    first = None
    usage = None
    with httpx.stream("POST", f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                      json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("choices") and first is None and (ev["choices"][0].get("delta") or {}).get("content"):
                first = time.time()
            if ev.get("usage"):
                usage = ev["usage"]
    t1 = time.time()
    n = usage["completion_tokens"] if usage else 0
    return {"ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 2),
            "completion_tokens": n, "decode_tok_per_s": round((n - 1) / max(t1 - (first or t0), 1e-6), 1)}


def cmd_test(args) -> None:
    env = load_env()
    key = env.get("QWEN_API_KEY")
    if not env.get("QWEN_BASE_URL") or not tunnel_info():
        sys.exit("no endpoint/tunnel; run serve-up or serve-ensure")
    url = local_base_url()  # QWEN_BASE_URL is the container-side address of the same tunnel
    results = {"base_url": env["QWEN_BASE_URL"], "via_tunnel_to": tunnel_info().get("ip")}
    # unauthenticated request must fail
    results["no_key_status"] = httpx.get(f"{url}/models", timeout=10).status_code

    code = "K7Q2ZP"
    body = {"model": SERVED_NAME, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What code is written in this image? Reply with the code only."},
                {"type": "image_url", "image_url": {"url": test_png_data_url(f"CODE: {code}")}}]}]}
    resp, dt = chat(url, key, body)
    ans = resp["choices"][0]["message"]["content"] or ""
    results["vision"] = {"answer": ans.strip()[:80], "correct": code in ans.upper().replace(" ", ""),
                         "latency_s": round(dt, 2), "usage": resp.get("usage")}

    tools = [{"type": "function", "function": {
        "name": "view", "description": "Return one video frame at time t (seconds).",
        "parameters": {"type": "object", "properties": {
            "t": {"type": "number", "description": "time in seconds"},
            "out_width": {"type": "integer", "description": "output width in pixels"}},
            "required": ["t"]}}}]
    body = {"model": SERVED_NAME, "max_tokens": 2048, "tools": tools, "tool_choice": "auto",
            "messages": [{"role": "user", "content":
                          "Look at the frame at 12.5 seconds with an output width of 320. Use the tool."}]}
    resp, dt = chat(url, key, body)
    msg = resp["choices"][0]["message"]
    tc = msg.get("tool_calls") or []
    results["tool_call"] = {"latency_s": round(dt, 2), "n_tool_calls": len(tc),
                            "call": tc[0]["function"] if tc else None,
                            "has_reasoning": bool(msg.get("reasoning_content") or msg.get("reasoning")),
                            "finish_reason": resp["choices"][0]["finish_reason"], "usage": resp.get("usage")}
    results["throughput"] = stream_tps(url, key)
    state = load_state()
    results["server"] = {k: state.get("serve", {}).get(k) for k in ("type", "variant", "spot", "eur_per_h", "ready_after_s")}
    print(json.dumps(results, indent=1))
    (HERE / "last_test.json").write_text(json.dumps({"ts": now_iso(), **results}, indent=1) + "\n")


def harbor_agent_entry(base_url: str, served_name: str = SERVED_NAME, opencode_version: str | None = None) -> dict:
    """Harbor `agents[]` entry for OpenCode against our endpoint.

    Harbor registers custom models as `{}` (text-only), and OpenCode then replaces
    every MCP image result with "ERROR: Cannot read image (this model does not
    support image input)". The opencode_config overlay below is merged into
    Harbor's generated provider and declares image input.
    """
    # The LoRA adapter has the base model's capabilities (vision, tools, reasoning, context).
    model = dict(json.loads((HERE / "opencode-qwen.json").read_text())["provider"]["verda-qwen"]["models"][SERVED_NAME])
    if served_name == LORA_NAME:
        model["name"] = f"{model['name']} (FrameHunt LoRA)"
    kwargs: dict = {"opencode_config": {"provider": {"verda-qwen": {"models": {served_name: model}}}}}
    if opencode_version:
        kwargs["version"] = opencode_version  # Harbor installs opencode-ai@<version>
    return {
        "name": "opencode",
        "model_name": f"verda-qwen/{served_name}",
        "base_url": base_url,
        "model_api_key_name": "QWEN_API_KEY",
        "model_api_format": "openai_chat_completions",
        "kwargs": kwargs,
    }


def cmd_harbor_agent(args) -> None:
    """Prints a Harbor JobConfig fragment ({"agents": [...]}) for `harbor run -c`. Local only."""
    url = load_env().get("QWEN_BASE_URL") or container_base_url()
    entry = harbor_agent_entry(url, served_name=args.served_name, opencode_version=args.opencode_version)
    print(json.dumps({"agents": [entry]}, indent=1))


# ---- ledger ---------------------------------------------------------------------
def ledger_rows() -> list[tuple[dict, datetime | None, float, float]]:
    """(create record, end time or None if still open, hours, cost) per ledger item."""
    if not LEDGER_PATH.exists():
        return []
    recs = [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]
    open_: dict[str, dict] = {}
    pairs = []
    for r in recs:
        if r["event"] == "create":
            open_[r["id"]] = r
        elif r["event"] == "delete" and r["id"] in open_:
            cr = open_.pop(r["id"])
            pairs.append((cr, datetime.fromisoformat(r["ts"])))
    pairs += [(cr, None) for cr in open_.values()]
    now = datetime.now(UTC)
    out = []
    for cr, end in pairs:
        hrs = ((end or now) - datetime.fromisoformat(cr["ts"])).total_seconds() / 3600
        out.append((cr, end, hrs, hrs * cr["price_per_hour"]))
    return out


def ledger_total_eur() -> float:
    return sum(cost for cr, _, _, cost in ledger_rows() if cr.get("currency", "eur") == "eur")


def cmd_ledger(args) -> None:
    rows = ledger_rows()
    if not rows:
        print("empty ledger")
        return
    total = {"eur": 0.0, "usd": 0.0}
    print(f"{'kind':<9}{'type':<18}{'spot':<6}{'price/h':>9} {'hours':>7} {'cost':>8}  state")
    for cr, end, hrs, cost in rows:
        cur = cr.get("currency", "eur")
        total[cur] = total.get(cur, 0) + cost
        print(f"{cr['kind']:<9}{cr['type']:<18}{str(cr.get('spot', '')):<6}{cr['price_per_hour']:>9.4f} "
              f"{hrs:>7.3f} {cost:>8.4f}  {'closed' if end else 'OPEN'} {cur} {cr['id'][:8]}")
    print(f"total: €{total['eur']:.3f}" + (f" + ${total['usd']:.3f}" if total.get("usd") else "")
          + "  (OPEN rows accrue until deleted; volumes are billed while they exist)")
    if args.balance:
        b = client().balance.get()
        st = load_state()
        print(f"account balance now: {b.amount:.3f} {b.currency}; at start: {st.get('initial_balance')}")


LOCAL_CMDS = ("ledger", "harbor-agent", "test")  # never touch the cloud API
LIFECYCLE_CMDS = ("serve-up", "serve-ensure", "serve-down", "train-up", "train-ensure", "train-down",
                  "train-publish")


# ---- main -----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inventory", help="instance types, prices, availability -> inventory.json")
    s.add_argument("--available", action="store_true", help="only rows available somewhere")
    s.set_defaults(fn=cmd_inventory)
    sub.add_parser("volume-create", help="create the 300 GB data volume").set_defaults(fn=cmd_volume_create)
    sub.add_parser("volume-status", help="list volumes").set_defaults(fn=cmd_volume_status)
    s = sub.add_parser("download", help="download + verify weights and build the vLLM venv on the volume")
    s.add_argument("--repos", nargs="+", default=list(MODELS))
    s.add_argument("--type", help=f"instance type (default: try {DOWNLOAD_FALLBACK})")
    s.add_argument("--on-demand", action="store_true")
    s.add_argument("--keep", action="store_true", help="keep the download instance")
    s.add_argument("--timeout-min", type=float, default=90)
    s.set_defaults(fn=cmd_download)
    for name, fn in (("serve-up", cmd_serve_up), ("serve-ensure", cmd_serve_ensure)):
        s = sub.add_parser(name)
        s.add_argument("--type", help="force an instance type from the candidate list")
        s.add_argument("--variant", choices=["bf16", "fp8"])
        s.add_argument("--on-demand", action="store_true", help="also consider on-demand capacity")
        s.add_argument("--mtp", action="store_true", help="enable MTP speculative decoding")
        s.add_argument("--timeout-min", type=float, default=40)
        s.add_argument("--keep", action="store_true",
                       help="keep a newly created instance if it fails to become healthy")
        s.add_argument("--lora", metavar="PATH",
                       help=f"also serve a LoRA adapter (dir on the instance, e.g. under {MNT}) as {LORA_NAME}"
                            + ("" if name == "serve-up" else "; default: the adapter of the last serve-up"))
        if name == "serve-up":
            s.add_argument("--no-wait", action="store_true")
        else:
            s.add_argument("--wait", action="store_true", help="block until healthy")
        s.set_defaults(fn=fn)
    sub.add_parser("serve-down", help="delete serving instances").set_defaults(fn=cmd_serve_down)
    sub.add_parser("serve-status", help="show serving instance + health").set_defaults(fn=cmd_serve_status)
    sub.add_parser("test", help="vision + tool-call + throughput smoke test").set_defaults(fn=cmd_test)
    s = sub.add_parser("harbor-agent", help="print a Harbor JobConfig fragment {agents: [OpenCode + image input]}")
    s.add_argument("--served-name", choices=[SERVED_NAME, LORA_NAME], default=SERVED_NAME,
                   help=f"model to target: the base model or the LoRA adapter ({LORA_NAME})")
    s.add_argument("--opencode-version", help="pin the OpenCode version Harbor installs (kwargs.version)")
    s.set_defaults(fn=cmd_harbor_agent)
    sub.add_parser("instances", help="list all instances on the account").set_defaults(fn=cmd_instances)
    s = sub.add_parser("delete", help="delete one instance by id")
    s.add_argument("id")
    s.set_defaults(fn=cmd_delete)
    s = sub.add_parser("train-up", help="start (or resume) a LoRA training run on an 80 GB GPU")
    s.add_argument("--run", required=True, help="run name; output in /mnt/fh/runs/<run>")
    s.add_argument("--data-dir", default=str(ROOT / "train/data/sft-v1"))
    s.add_argument("--epochs", type=float, default=1.0)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--grad-accum", type=int, default=8)
    s.add_argument("--lora-r", type=int, default=16)
    s.add_argument("--save-every", type=int, default=10)
    s.add_argument("--trainer-arg", action="append", help="extra sft_lora.py argument (repeatable)")
    s.add_argument("--type", choices=sorted(TRAIN_GPUS), help="force a GPU type")
    s.add_argument("--on-demand", action="store_true", help="on-demand only (default: spot first)")
    s.add_argument("--max-eur", type=float, default=40.0,
                   help="refuse if ledger total + expected cost exceeds this (also checked on re-provision)")
    s.add_argument("--timeout-min", type=float, default=60, help="wait this long for training to start")
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--keep", action="store_true", help="keep a new box if startup fails")
    s.set_defaults(fn=cmd_train_up)
    sub.add_parser("train-ensure", help="watchdog step for the training run").set_defaults(fn=cmd_train_ensure)
    sub.add_parser("train-down", help="stop training (resumable) and delete the box").set_defaults(fn=cmd_train_down)
    sub.add_parser("train-status", help="run state, step/loss, ETA").set_defaults(fn=cmd_train_status)
    s = sub.add_parser("train-publish", help="retry copying a finished adapter to /mnt/fh/adapters/<run> + fetch")
    s.add_argument("--run")
    s.set_defaults(fn=cmd_train_publish)
    s = sub.add_parser("train-fetch", help="copy the final adapter + logs to train/out/<run>/")
    s.add_argument("--run")
    s.set_defaults(fn=cmd_train_fetch)
    s = sub.add_parser("ledger", help="print spend from ledger.jsonl")
    s.add_argument("--balance", action="store_true")
    s.set_defaults(fn=cmd_ledger)
    args = p.parse_args()
    if "initial_balance" not in load_state() and args.cmd not in LOCAL_CMDS:
        b = client().balance.get()
        val = f"{b.amount:.3f} {b.currency} @ {now_iso()}"
        update_state(lambda st: st.setdefault("initial_balance", val))
    if args.cmd not in LOCAL_CMDS and args.cmd not in LIFECYCLE_CMDS:  # those reconcile under their lock
        reconcile(client())
    try:
        args.fn(args)
    except DeleteFailed as e:
        sys.exit(f"ERROR: {e}")


if __name__ == "__main__":
    main()
