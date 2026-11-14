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
import io
import json
import os
import secrets
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

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
PORT = 8000
SERVE_PREFIX = "fh-qwen-serve"
DL_PREFIX = "fh-qwen-dl"
DOWNLOAD_FALLBACK = ["CPU.8V.32G", "CPU.16V.64G", "CPU.4V.16G", "CPU.32V.128G"]
STARTUP_GRACE_S = 40 * 60  # a serving box that is not healthy after this is replaced

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


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=1) + "\n")


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def ledger_append(**rec) -> None:
    rec = {"ts": now_iso(), **rec}
    with LEDGER_PATH.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def client():
    from verda import VerdaClient

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
            state["ssh_key_id"] = k.id
            break
    else:
        state["ssh_key_id"] = c.ssh_keys.create("framehunt-verda", pub).id
    save_state(state)
    return state["ssh_key_id"]


def ssh(ip: str, cmd: str, timeout: int = 60, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-i", str(SSH_KEY), "-o", "StrictHostKeyChecking=no", "-o",
         "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10",
         f"root@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout, check=check,
    )


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
    state["volume"] = {"id": vol.id, "name": vol.name, "size_gb": vol.size, "type": vol.type,
                       "location": vol.location, "mount": MNT}
    save_state(state)
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


def create_instance(c, state, *, itype, hostname, script_name, script, spot, os_gb=80):
    from verda.instances import OSVolume

    vol = state["volume"]
    key_id = ensure_ssh_key(c, state)
    wait_volume_free(c, vol["id"])
    for s in c.startup_scripts.get():  # one script per role; replace stale copies
        if s.name == script_name:
            c.startup_scripts.delete_by_id(s.id)
    sc = c.startup_scripts.create(script_name, script)
    prices = eur_prices(c)
    try:
        inst = c.instances.create(
            instance_type=itype, image=IMAGE, hostname=hostname, description=hostname,
            ssh_key_ids=[key_id], location=vol["location"], startup_script_id=sc.id,
            existing_volumes=[vol["id"]], is_spot=spot,
            os_volume=OSVolume(name=f"{hostname}-os", size=os_gb,
                               on_spot_discontinue="delete_permanently" if spot else None),
            max_wait_time=600,
        )
    except TimeoutError:  # order accepted but still queued: it exists and bills once running
        inst = next(i for i in c.instances.get() if i.hostname == hostname)
    p = prices.get(itype, {})
    eur_h = float(p.get("spot_price" if spot else "price_per_hour", inst.price_per_hour))
    ledger_append(event="create", kind="instance", id=inst.id, type=itype, hostname=hostname,
                  location=vol["location"], spot=spot, price_per_hour=eur_h, currency="eur",
                  api_price_per_hour=inst.price_per_hour)
    log(f"created {itype} {'spot' if spot else 'on-demand'} {inst.id} ({hostname}) €{eur_h}/h")
    return inst


def delete_instance(c, inst_id: str, reason: str = "") -> None:
    try:
        inst = c.instances.get_by_id(inst_id)
        c.instances.action(inst_id, "delete", volume_ids=[inst.os_volume_id] if inst.os_volume_id else None,
                           delete_permanently=True)
    except Exception as e:  # already gone (e.g. spot discontinued)
        log(f"delete {inst_id}: {type(e).__name__}: {str(e)[:200]}")
    ledger_append(event="delete", kind="instance", id=inst_id, reason=reason)
    log(f"deleted instance {inst_id} {reason}")


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


def our_instances(c, prefix: str):
    return [i for i in c.instances.get() if i.hostname.startswith(prefix)]


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

def cmd_download(args) -> None:
    c = client()
    state = load_state()
    if not get_volume(c, state):
        sys.exit("no volume; run volume-create first")
    vol = state["volume"]
    insts = our_instances(c, DL_PREFIX)
    if insts:
        inst = insts[0]
        log(f"reusing download instance {inst.id} ({inst.status})")
    else:
        models = "\n".join(f"{r} {rev}" for r, rev in MODELS.items() if r in args.repos)
        script = render("download.sh", {
            "__MODELS__": models, "__VLLM_VERSION__": VLLM_VERSION,
            "__VERIFY_PY__": (REMOTE / "verify_weights.py").read_text()}, vol)
        types = [args.type] if args.type else DOWNLOAD_FALLBACK
        options = [(t, True) for t in types] + [(t, False) for t in types]
        if args.on_demand:
            options = [(t, False) for t in types]
        inst = None
        for itype, spot in options:
            try:
                inst = create_instance(c, state, itype=itype, hostname=f"{DL_PREFIX}-{int(time.time())}",
                                       script_name=DL_PREFIX, script=script, spot=spot, os_gb=120)
                break
            except APIException as e:
                log(f"{itype} {'spot' if spot else 'on-demand'} unavailable: {str(e).splitlines()[-1][:120]}")
        if inst is None:
            sys.exit("no CPU capacity for the download instance")
    inst = wait_running(c, inst.id)
    log(f"download instance {inst.ip}; polling {MNT}/logs/download.status")
    t0, status = time.time(), ""
    try:
        while time.time() - t0 < args.timeout_min * 60:
            r = ssh(inst.ip, f"cat {MNT}/logs/download.status 2>/dev/null; "
                             f"du -sh {MNT}/models 2>/dev/null | cut -f1; tail -n 2 /var/log/fh/startup.log")
            out = r.stdout.strip().splitlines()
            if out:
                status = out[0]
                log(" | ".join(x[-120:] for x in out[:4]))
            if status.startswith(("done", "failed")):
                break
            if any("FATAL" in x for x in out):
                status = "failed (FATAL in startup log)"
                break
            time.sleep(60)
        r = ssh(inst.ip, f"{VERIFY_SUMMARY_CMD}; grep -E '^\\[fh\\]' {MNT}/logs/download.log | tail -n 20",
                timeout=120)
        print(r.stdout)
    finally:
        if not args.keep:
            delete_instance(c, inst.id, reason=f"download {status or 'timeout'}")
            for sc in c.startup_scripts.get():
                if sc.name == DL_PREFIX:
                    c.startup_scripts.delete_by_id(sc.id)
    if not status.startswith("done"):
        sys.exit(f"download job status: {status or 'timeout'}")
    state["weights"] = {r: {"revision": MODELS[r], "path": f"{MNT}/models/{r.split('/')[1]}"}
                        for r in args.repos}
    state["vllm_env"] = f"{MNT}/envs/vllm-{VLLM_VERSION}"
    save_state(state)


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


def vllm_args(variant: str, extra: list[str], mtp: bool) -> list[str]:
    args = list(BASE_VLLM_ARGS)
    # later flags override earlier ones in argparse
    args += extra
    if mtp:
        args += ["--speculative-config", "'{\"method\": \"mtp\", \"num_speculative_tokens\": 2}'"]
    return args


def health(base_url: str, key: str, timeout: float = 10) -> bool:
    try:
        r = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
        return r.status_code == 200 and any(m["id"] == SERVED_NAME for m in r.json()["data"])
    except Exception:
        return False


def base_url_for(ip: str) -> str:
    return f"http://{ip}:{PORT}/v1"


def cmd_serve_up(args) -> None:
    c = client()
    state = load_state()
    if not get_volume(c, state) or "weights" not in state:
        sys.exit("volume/weights missing; run volume-create and download first")
    existing = [i for i in our_instances(c, SERVE_PREFIX) if i.status not in ("error", "discontinued")]
    if existing:
        inst = existing[0]
        log(f"serving instance already exists: {inst.id} {inst.instance_type} {inst.status}")
    else:
        opts = pick_candidate(c, args)
        if not opts:
            sys.exit("no candidate GPU available in the volume's location (try --on-demand)")
        key = api_key()
        inst = None
        for price, itype, variant, extra, spot in opts:
            model_dir = state["weights"].get(VARIANT_REPO[variant], {}).get("path")
            if not model_dir:
                log(f"skip {itype}: {variant} weights not on the volume")
                continue
            va = vllm_args(variant, extra, args.mtp)
            script = render("serve.sh", {
                "__VLLM_VERSION__": VLLM_VERSION, "__MODEL_DIR__": model_dir,
                "__SERVED_NAME__": SERVED_NAME, "__PORT__": str(PORT),
                "__VLLM_ARGS__": " ".join(va), "__API_KEY__": key}, state["volume"])
            hostname = f"{SERVE_PREFIX}-{int(time.time())}"
            try:
                inst = create_instance(c, state, itype=itype, hostname=hostname, script_name=SERVE_PREFIX,
                                       script=script, spot=spot, os_gb=60)
                break
            except APIException as e:
                log(f"{itype} {'spot' if spot else 'on-demand'} unavailable: {str(e).splitlines()[-1][:120]}")
        if inst is None:
            sys.exit("all candidates failed to deploy")
        state["serve"] = {"id": inst.id, "type": itype, "variant": variant, "spot": spot,
                          "eur_per_h": price, "created": time.time(), "vllm_args": va,
                          "model_dir": model_dir}
        save_state(state)
    if args.no_wait:
        return
    wait_healthy(c, inst.id, args.timeout_min)


def wait_healthy(c, inst_id: str, timeout_min: float) -> None:
    key = load_env()["QWEN_API_KEY"]
    inst = wait_running(c, inst_id)
    url = base_url_for(inst.ip)
    t0 = time.time()
    last_tail = ""
    while time.time() - t0 < timeout_min * 60:
        if health(url, key):
            set_env_var("QWEN_BASE_URL", url)
            state = load_state()
            state.setdefault("serve", {})["ready_after_s"] = round(time.time() - state["serve"].get("created", t0))
            state["serve"]["base_url"] = url
            save_state(state)
            log(f"healthy: {url} (QWEN_BASE_URL written to .env)")
            return
        try:
            r = ssh(inst.ip, f"tail -n 1 {MNT}/logs/serve-$(hostname).log 2>/dev/null || tail -n 1 /var/log/fh/startup.log",
                    timeout=30)
            tail = r.stdout.strip()[-160:]
            if tail and tail != last_tail:
                log(f"... {tail}")
                last_tail = tail
        except Exception:
            pass
        time.sleep(20)
    sys.exit(f"endpoint not healthy after {timeout_min} min")


def cmd_serve_down(args) -> None:
    c = client()
    state = load_state()
    for inst in our_instances(c, SERVE_PREFIX):
        delete_instance(c, inst.id, reason="serve-down")
    for s in c.startup_scripts.get():
        if s.name == SERVE_PREFIX:
            c.startup_scripts.delete_by_id(s.id)
    state.pop("serve", None)
    save_state(state)
    set_env_var("QWEN_BASE_URL", "")


def cmd_serve_status(args) -> None:
    c = client()
    env = load_env()
    state = load_state()
    insts = our_instances(c, SERVE_PREFIX)
    if not insts:
        print("no serving instance")
    for i in insts:
        url = base_url_for(i.ip) if i.ip else None
        ok = bool(url and health(url, env.get("QWEN_API_KEY", "")))
        print(json.dumps({"id": i.id, "type": i.instance_type, "status": i.status, "spot": i.is_spot,
                          "ip": i.ip, "base_url": url, "healthy": ok,
                          "variant": state.get("serve", {}).get("variant"),
                          "price_per_hour": i.price_per_hour}))


@contextlib.contextmanager
def file_lock(path: Path):
    with path.open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another serve-ensure is running; exiting")
            sys.exit(0)
        yield


def cmd_serve_ensure(args) -> None:
    """Idempotent: healthy -> no-op; booting -> wait (or return); gone/broken -> replace."""
    with file_lock(HERE / ".serve-ensure.lock"):
        c = client()
        state = load_state()
        key = load_env().get("QWEN_API_KEY", "")
        insts = our_instances(c, SERVE_PREFIX)
        for i in insts:
            url = base_url_for(i.ip) if i.ip else None
            if url and health(url, key):
                if load_env().get("QWEN_BASE_URL") != url:
                    set_env_var("QWEN_BASE_URL", url)
                log(f"healthy {i.instance_type} {url}")
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
        if state.get("serve", {}).get("id") and not any(i.id == state["serve"]["id"] for i in insts):
            ledger_append(event="delete", kind="instance", id=state["serve"]["id"], reason="gone (spot preempted?)")
            log(f"instance {state['serve']['id']} is gone (preempted?)")
            state.pop("serve")
            save_state(state)
        args.no_wait = not args.wait
        cmd_serve_up(args)


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
    url, key = env.get("QWEN_BASE_URL"), env.get("QWEN_API_KEY")
    if not url:
        sys.exit("QWEN_BASE_URL not set; run serve-up")
    results = {"base_url_host": url.split("//")[1].split(":")[0]}
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


def harbor_agent_entry(base_url: str) -> dict:
    """Harbor `agents[]` entry for OpenCode against our endpoint.

    Harbor registers custom models as `{}` (text-only), and OpenCode then replaces
    every MCP image result with "ERROR: Cannot read image (this model does not
    support image input)". The opencode_config overlay below is merged into
    Harbor's generated provider and declares image input.
    """
    model = json.loads((HERE / "opencode-qwen.json").read_text())["provider"]["verda-qwen"]["models"][SERVED_NAME]
    return {
        "name": "opencode",
        "model_name": f"verda-qwen/{SERVED_NAME}",
        "base_url": base_url,
        "model_api_key_name": "QWEN_API_KEY",
        "model_api_format": "openai_chat_completions",
        "kwargs": {"opencode_config": {"provider": {"verda-qwen": {"models": {SERVED_NAME: model}}}}},
    }


def cmd_harbor_agent(args) -> None:
    url = load_env().get("QWEN_BASE_URL") or "http://<ip>:8000/v1"
    print(json.dumps(harbor_agent_entry(url), indent=1))


# ---- ledger ---------------------------------------------------------------------
def cmd_ledger(args) -> None:
    if not LEDGER_PATH.exists():
        print("empty ledger")
        return
    recs = [json.loads(l) for l in LEDGER_PATH.read_text().splitlines() if l.strip()]
    open_: dict[str, dict] = {}
    rows = []
    now = datetime.now(UTC)
    for r in recs:
        if r["event"] == "create":
            open_[r["id"]] = r
        elif r["event"] == "delete" and r["id"] in open_:
            cr = open_.pop(r["id"])
            rows.append((cr, datetime.fromisoformat(r["ts"])))
    for cr in open_.values():
        rows.append((cr, None))
    total = {"eur": 0.0, "usd": 0.0}
    print(f"{'kind':<9}{'type':<18}{'spot':<6}{'price/h':>9} {'hours':>7} {'cost':>8}  state")
    for cr, end in rows:
        start = datetime.fromisoformat(cr["ts"])
        hrs = ((end or now) - start).total_seconds() / 3600
        cost = hrs * cr["price_per_hour"]
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
        if name == "serve-up":
            s.add_argument("--no-wait", action="store_true")
        else:
            s.add_argument("--wait", action="store_true", help="block until healthy")
        s.set_defaults(fn=fn)
    sub.add_parser("serve-down", help="delete serving instances").set_defaults(fn=cmd_serve_down)
    sub.add_parser("serve-status", help="show serving instance + health").set_defaults(fn=cmd_serve_status)
    sub.add_parser("test", help="vision + tool-call + throughput smoke test").set_defaults(fn=cmd_test)
    sub.add_parser("harbor-agent", help="print a Harbor agents[] entry (OpenCode + image input)").set_defaults(fn=cmd_harbor_agent)
    sub.add_parser("instances", help="list all instances on the account").set_defaults(fn=cmd_instances)
    s = sub.add_parser("delete", help="delete one instance by id")
    s.add_argument("id")
    s.set_defaults(fn=cmd_delete)
    s = sub.add_parser("ledger", help="print spend from ledger.jsonl")
    s.add_argument("--balance", action="store_true")
    s.set_defaults(fn=cmd_ledger)
    args = p.parse_args()
    st = load_state()
    if "initial_balance" not in st and args.cmd not in ("ledger",):
        b = client().balance.get()
        st["initial_balance"] = f"{b.amount:.3f} {b.currency} @ {now_iso()}"
        save_state(st)
    args.fn(args)


if __name__ == "__main__":
    main()
