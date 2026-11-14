"""FrameHunt's Verda tooling on top of SkyPilot: serve Qwen3.8-27B with vLLM and train LoRA adapters.

Run from the repo root:  uv run --group infra python infra/verda/fh.py <command>
See infra/verda/README.md.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import io
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV_PATH = ROOT / ".env"
STATE_PATH = HERE / ".fh-state.json"
LOCK_PATH = HERE / ".fh.lock"
STAGE_DIR = HERE / ".stage"
TUNNEL_PIDFILE = HERE / "tunnel.pid"
TUNNEL_LOG = HERE / "tunnel.log"
KNOWN_HOSTS = HERE / "known_hosts"
TRAIN_DIR = ROOT / "train"

SERVE, TRAIN, DOWNLOAD = "fh-serve", "fh-train", "fh-download"
MNT = "/mnt/fh"
SERVED_NAME = "qwen3.8-27b"
LORA_NAME = "qwen3.8-27b-fh"
LORA_MAX_RANK = 16
SERVE_GRACE_S = 40 * 60
SERVE_MAX_FAILED_RELAUNCHES = 2
TRAIN_MAX_FAILURES = 4
CHECKPOINT_WAIT_S = 15 * 60
FETCH_TIMEOUT_S = 30 * 60

TUNNEL_BIND = os.environ.get("FH_TUNNEL_BIND", "127.0.0.1")
TUNNEL_PORT = int(os.environ.get("FH_TUNNEL_PORT", "18000"))
TUNNEL_CONTAINER_HOST = os.environ.get("FH_TUNNEL_CONTAINER_HOST", "10.0.2.2")

TRAIN_CODE_FILES = ("sft_lora.py", "fhdata.py", "requirements.txt", "requirements-gpu.txt")
TRAIN_REQ_FILES = ("requirements.txt", "requirements-gpu.txt")
TRAINER_EXTRA_FLAGS = {"--warmup-frac", "--max-grad-norm", "--weight-decay", "--lora-alpha", "--lora-dropout",
                       "--quant", "--attn", "--max-len", "--loss-chunk", "--eval-frac", "--keep", "--seed",
                       "--max-steps", "--limit", "--no-grad-ckpt", "--lora-r"}
JOB_ACTIVE = {"INIT", "PENDING", "SETTING_UP", "RUNNING"}


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC):%H:%M:%S}] {msg}", flush=True)


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def set_env_var(key: str, value: str | None) -> None:
    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    out = [line for line in lines if line.split("=", 1)[0].strip() != key]
    if value is not None:
        out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n")
    ENV_PATH.chmod(0o600)


def api_key() -> str:
    key = load_env().get("QWEN_API_KEY")
    if not key:
        key = "fhq-" + secrets.token_urlsafe(32)
        set_env_var("QWEN_API_KEY", key)
        log("generated QWEN_API_KEY in .env")
    return key


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    tmp.replace(STATE_PATH)


@contextlib.contextmanager
def lifecycle_lock(wait: bool):
    with LOCK_PATH.open("w") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError:
            log("another fh.py lifecycle command holds the lock; skipping")
            sys.exit(0)
        yield


def sky(*args: str, check: bool = True, env: dict | None = None) -> int:
    cmd = [str(Path(sys.executable).parent / "sky"), *args]
    return subprocess.run(cmd, check=check, env={**os.environ, **(env or {})}).returncode


def cluster_status(name: str) -> str | None:
    import sky as sky_sdk

    records = sky_sdk.get(sky_sdk.status([name], refresh=sky_sdk.StatusRefreshMode.FORCE))
    return records[0]["status"].value if records else None


def last_job_status(name: str) -> str | None:
    import sky as sky_sdk

    try:
        jobs = sky_sdk.get(sky_sdk.queue(name))
    except (sky_sdk.exceptions.SkyPilotExcludeArgsBaseException, sky_sdk.exceptions.ClusterNotUpError,
            ValueError, RuntimeError) as e:
        log(f"{name}: job queue unreadable ({type(e).__name__}: {e})")
        return "UNREADABLE"
    return max(jobs, key=lambda j: j["job_id"])["status"].value if jobs else None


def cluster_cost_since(name: str, since: float) -> float:
    import sky as sky_sdk

    return sum(r["total_cost"] for r in sky_sdk.get(sky_sdk.cost_report())
               if r["name"] == name and (r["launched_at"] or 0) >= since - 60)


def down(name: str) -> None:
    if cluster_status(name) is not None:
        sky("down", "-y", name)


def ssh_base(cluster: str) -> list[str]:
    import sky as sky_sdk

    records = sky_sdk.get(sky_sdk.status([cluster]))
    launched = records[0]["launched_at"] if records else 0
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={KNOWN_HOSTS}", "-o", f"HostKeyAlias={cluster}-{launched}"]


def ssh_out(cluster: str, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run([*ssh_base(cluster), cluster, cmd],
                          capture_output=True, text=True, timeout=timeout, check=False)


def local_base_url() -> str:
    host = "127.0.0.1" if TUNNEL_BIND in ("127.0.0.1", "0.0.0.0", "localhost") else TUNNEL_BIND
    return f"http://{host}:{TUNNEL_PORT}/v1"


def container_base_url() -> str:
    return f"http://{TUNNEL_CONTAINER_HOST}:{TUNNEL_PORT}/v1"


def served_names(lora: str | None) -> list[str]:
    return [SERVED_NAME] + ([LORA_NAME] if lora else [])


def healthy(names: list[str]) -> bool:
    try:
        r = httpx.get(f"{local_base_url()}/models", headers={"Authorization": f"Bearer {api_key()}"}, timeout=10)
        return r.status_code == 200 and set(names) <= {m["id"] for m in r.json()["data"]}
    except (httpx.HTTPError, ValueError, KeyError):
        return False


def tunnel_pid() -> int | None:
    if not TUNNEL_PIDFILE.exists():
        return None
    pid = int(TUNNEL_PIDFILE.read_text().strip() or 0)
    ps = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True, check=False)
    return pid if ps.returncode == 0 and SERVE in ps.stdout and "-N" in ps.stdout else None


def stop_tunnel() -> None:
    pid = tunnel_pid()
    if pid:
        os.kill(pid, 15)
    TUNNEL_PIDFILE.unlink(missing_ok=True)


def start_tunnel() -> None:
    stop_tunnel()
    with TUNNEL_LOG.open("a") as logf:
        proc = subprocess.Popen(
            [*ssh_base(SERVE), "-N", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
             "-o", "ExitOnForwardFailure=yes", "-L", f"{TUNNEL_BIND}:{TUNNEL_PORT}:127.0.0.1:8000", SERVE],
            stdout=logf, stderr=logf, stdin=subprocess.DEVNULL, start_new_session=True)
    TUNNEL_PIDFILE.write_text(str(proc.pid))
    time.sleep(3)
    if proc.poll() is not None:
        raise RuntimeError(f"ssh tunnel exited with {proc.returncode}; see {TUNNEL_LOG}")


def ensure_tunnel() -> bool:
    if tunnel_pid():
        return True
    try:
        start_tunnel()
    except RuntimeError as e:
        log(f"serve: {e}")
        return False
    return True


def launch_serve(state: dict) -> None:
    cfg = state["serve"]
    down(SERVE)
    stop_tunnel()
    cfg["launched_at"] = time.time()
    save_state(state)
    args = ["launch", "-y", "-d", "-c", SERVE, str(HERE / "serve.yaml"), "--secret", "QWEN_API_KEY",
            "--env", f"LORA={cfg.get('lora') or ''}"]
    if cfg.get("on_demand"):
        args.append("--no-use-spot")
    sky(*args, env={"QWEN_API_KEY": api_key()})


def give_up_serving(state: dict) -> None:
    log("serve: still failing after relaunches; giving up (see tunnel.log and `sky logs fh-serve`)")
    down(SERVE)
    state.pop("serve")
    save_state(state)
    stop_tunnel()


def serve_tick(state: dict, wait_s: float) -> bool:
    cfg = state["serve"]
    names = served_names(cfg.get("lora"))
    status = cluster_status(SERVE)
    job = last_job_status(SERVE) if status == "UP" else None
    past_grace = time.time() - cfg.get("launched_at", 0) > SERVE_GRACE_S
    broken = status == "UP" and (
        job not in JOB_ACTIVE or (not (ensure_tunnel() and healthy(names)) and past_grace))
    if broken:
        cfg["failed_relaunches"] = cfg.get("failed_relaunches", 0) + 1
        if cfg["failed_relaunches"] > SERVE_MAX_FAILED_RELAUNCHES:
            give_up_serving(state)
            return False
        status = f"{status}, job {job}, unhealthy or failed"
        job = None
    if status != "UP" or job not in JOB_ACTIVE:
        log(f"serve: cluster {status}, job {job}; launching")
        try:
            launch_serve(state)
        except subprocess.CalledProcessError:
            if cluster_status(SERVE) is None:
                raise
            cfg["failed_relaunches"] = cfg.get("failed_relaunches", 0) + 1
            if cfg["failed_relaunches"] > SERVE_MAX_FAILED_RELAUNCHES:
                give_up_serving(state)
            save_state(state)
            return False
    deadline = time.time() + wait_s
    while not (ensure_tunnel() and healthy(names)):
        if time.time() > deadline:
            return False
        time.sleep(15)
    cfg["failed_relaunches"] = 0
    save_state(state)
    set_env_var("QWEN_BASE_URL", container_base_url())
    return True


def cmd_serve_up(args) -> None:
    with lifecycle_lock(wait=True):
        state = load_state()
        if active_train(state):
            sys.exit("serve-up refused: a training run is active (train-down first)")
        state["serve"] = {"lora": args.lora, "on_demand": args.on_demand}
        ok = False
        try:
            launch_serve(state)
            if args.no_wait:
                return
            ok = serve_tick(state, args.timeout_min * 60)
        finally:
            if not ok and not args.no_wait and not args.keep:
                log("serve: not healthy; deleting the box (use --keep to keep it)")
                down(SERVE)
                state.pop("serve", None)
                save_state(state)
                stop_tunnel()
    log(f"serve: {'healthy at ' + container_base_url() if ok else 'not healthy; box kept, run serve-ensure'}")


def cmd_serve_ensure(args) -> None:
    with lifecycle_lock(wait=False):
        state = load_state()
        if not state.get("serve"):
            if cluster_status(SERVE) is not None:
                log("serve: not desired, but fh-serve exists; deleting it")
                down(SERVE)
            log("serve: not desired; idle")
            return
        ok = serve_tick(state, args.wait_min * 60)
    log(f"serve: {'healthy' if ok else 'not healthy yet'}")


def cmd_serve_down(args) -> None:
    with lifecycle_lock(wait=True):
        state = load_state()
        down(SERVE)
        state.pop("serve", None)
        save_state(state)
        stop_tunnel()
        set_env_var("QWEN_BASE_URL", None)
    log("serve: down")


def sha_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def data_sha(data_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in data_dir.rglob("*") if p.is_file()):
        b = f.read_bytes()
        h.update(f"{f.relative_to(data_dir).as_posix()}\0{len(b)}\0".encode())
        h.update(b)
    return h.hexdigest()[:16]


def trainer_args(run: str, data_remote: str, a) -> list[str]:
    args = ["--model", f"{MNT}/models/Qwen3.8-27B", "--data", data_remote, "--out", f"{MNT}/runs/{run}",
            "--epochs", str(a.epochs), "--lr", str(a.lr), "--grad-accum", str(a.grad_accum),
            "--lora-r", str(a.lora_r), "--save-every", str(a.save_every)]
    for extra in a.trainer_arg or []:
        if not re.fullmatch(r"[A-Za-z0-9_.=/:+-]+", extra):
            raise SystemExit(f"--trainer-arg {extra!r}: only plain flags and values are allowed")
        if extra.startswith("-") and extra.split("=", 1)[0] not in TRAINER_EXTRA_FLAGS:
            raise SystemExit(f"--trainer-arg {extra!r}: flag not allowed (allowed: {sorted(TRAINER_EXTRA_FLAGS)})")
        args.append(extra)
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--lora-r", type=int)
    ns, _ = p.parse_known_args(args)
    if not 1 <= ns.lora_r <= LORA_MAX_RANK:
        raise SystemExit(f"--lora-r {ns.lora_r}: serving loads ranks 1..{LORA_MAX_RANK}")
    return args


def run_definition(a) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", a.run):
        raise SystemExit(f"--run {a.run!r}: use letters, digits, '.', '_' and '-'")
    data_dir = Path(a.data).resolve()
    dsha = data_sha(data_dir)
    data_remote = f"{MNT}/data/{data_dir.name}-{dsha}"
    run = {"run": a.run, "data_sha": dsha, "code_sha": sha_files([TRAIN_DIR / f for f in TRAIN_CODE_FILES]),
           "req_hash": sha_files([TRAIN_DIR / f for f in TRAIN_REQ_FILES])[:12],
           "trainer_args": trainer_args(a.run, data_remote, a)}
    return {**run, "manifest": json.dumps(run, sort_keys=True), "data_dir": str(data_dir),
            "data_remote": data_remote, "max_usd": a.max_usd, "on_demand": a.on_demand}


def stage(run: dict) -> Path:
    if data_sha(Path(run["data_dir"])) != run["data_sha"]:
        raise SystemExit(f"{run['data_dir']} changed since run {run['run']} started; start a new --run")
    if sha_files([TRAIN_DIR / f for f in TRAIN_CODE_FILES]) != run["code_sha"]:
        raise SystemExit(f"train/ changed since run {run['run']} started; start a new --run")
    dest = STAGE_DIR / run["run"]
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "code").mkdir(parents=True)
    for f in TRAIN_CODE_FILES:
        shutil.copy2(TRAIN_DIR / f, dest / "code" / f)
    shutil.copytree(run["data_dir"], dest / "data")
    return dest


def launch_train(run: dict) -> None:
    down(TRAIN)
    envs = {"RUN": run["run"], "MANIFEST": run["manifest"], "DATA_SHA": run["data_sha"],
            "CODE_SHA": run["code_sha"], "REQ_HASH": run["req_hash"], "DATA_REMOTE": run["data_remote"],
            "TRAINER_ARGS": " ".join(run["trainer_args"])}
    args = ["launch", "-y", "-d", "-c", TRAIN, str(HERE / "train.yaml"), "--workdir", str(stage(run))]
    for k in envs:
        args += ["--env", k]
    if run.get("on_demand"):
        args.append("--no-use-spot")
    sky(*args, env=envs)


def active_train(state: dict) -> bool:
    t = state.get("train")
    return bool(t) and not (t.get("done") or t.get("failed") or t.get("stopped"))


def fetch_run(cluster: str, run: str) -> Path:
    dest = TRAIN_DIR / "out" / run
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    out = f"{MNT}/runs/{run}"
    remote = f"cd {out} && tar -cf - --exclude=final/processor final loss.jsonl $(ls eval.jsonl args.json DONE 2>/dev/null)"
    pipe = f"set -o pipefail; {shlex.join([*ssh_base(cluster), cluster, remote])} | tar -C {shlex.quote(str(tmp))} -xf -"
    subprocess.run(["bash", "-c", pipe], check=True, timeout=FETCH_TIMEOUT_S)
    if not (tmp / "final" / "adapter_model.safetensors").exists():
        raise RuntimeError(f"fetched adapter for {run} is incomplete")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.replace(dest)
    return dest


def checkpoint_and_stop() -> None:
    if cluster_status(TRAIN) != "UP":
        return
    log("train: asking the trainer to checkpoint before the box goes")
    ssh_out(TRAIN, "pkill -TERM -f '[s]ft_lora.py'")
    deadline = time.time() + CHECKPOINT_WAIT_S
    while time.time() < deadline:
        if ssh_out(TRAIN, "pgrep -f '[s]ft_lora.py'").returncode == 1:
            return
        time.sleep(15)
    log("train: the trainer did not exit in time; deleting the box anyway")


def train_tick(state: dict) -> str:
    run = state["train"]
    if not active_train(state):
        down(TRAIN)
        return "inactive"
    spent = cluster_cost_since(TRAIN, run["started_at"])
    if spent >= run["max_usd"]:
        run["failed"] = f"budget: ${spent:.2f} >= ${run['max_usd']}"
        save_state(state)
        checkpoint_and_stop()
        down(TRAIN)
        return run["failed"]
    status = cluster_status(TRAIN)
    job = last_job_status(TRAIN) if status == "UP" else None
    if job in JOB_ACTIVE:
        return f"running (job {job}, ${spent:.2f} spent)"
    if job == "SUCCEEDED":
        run["done"] = True
        run["adapter"] = f"{MNT}/adapters/{run['run']}"
        save_state(state)
        try:
            run["fetched"] = str(fetch_run(TRAIN, run["run"]).relative_to(ROOT))
        except (OSError, RuntimeError, subprocess.SubprocessError) as e:
            run["fetch_error"] = f"{type(e).__name__}: {e}"
        save_state(state)
        down(TRAIN)
        if "fetch_error" in run:
            return f"done: adapter {run['adapter']}; fetch failed ({run['fetch_error']}), retry with train-fetch"
        return f"done: adapter {run['adapter']}, local copy {run['fetched']}"
    if job is not None:
        fatal = status == "UP" and ssh_out(TRAIN, f"test -f {MNT}/runs/{run['run']}/FATAL").returncode == 0
        run["failures"] = run.get("failures", 0) + 1
        if fatal or run["failures"] >= TRAIN_MAX_FAILURES:
            run["failed"] = "manifest mismatch" if fatal else f"job {job} {run['failures']} times"
            save_state(state)
            down(TRAIN)
            return f"failed: {run['failed']}"
    log(f"train: cluster {status}, job {job}; launching")
    try:
        launch_train(run)
    except subprocess.CalledProcessError:
        if cluster_status(TRAIN) is None:
            raise
        run["failures"] = run.get("failures", 0) + 1
        if run["failures"] >= TRAIN_MAX_FAILURES:
            run["failed"] = f"launch failed {run['failures']} times"
            save_state(state)
            down(TRAIN)
            return f"failed: {run['failed']}"
        save_state(state)
        return f"launch failed ({run['failures']}/{TRAIN_MAX_FAILURES}); retrying on the next tick"
    save_state(state)
    return "launched"


def cmd_train_up(args) -> None:
    with lifecycle_lock(wait=True):
        state = load_state()
        if state.get("serve") or cluster_status(SERVE) is not None:
            sys.exit("train-up refused: serving is on, and the volume attaches to one box (serve-down first)")
        run = run_definition(args)
        prev = state.get("train") or {}
        if prev.get("run") == run["run"] and prev.get("manifest") != run["manifest"]:
            sys.exit(f"run {run['run']} exists with a different definition; use a new --run")
        if prev.get("run") not in (None, run["run"]) and (active_train(state) or cluster_status(TRAIN) is not None):
            sys.exit(f"run {prev['run']} is active or its box still exists; train-down first")
        state["train"] = {**run, "started_at": prev.get("started_at", time.time()) if prev.get("run") == run["run"]
                          else time.time(), "failures": 0}
        save_state(state)
        log(f"train {run['run']}: {train_tick(state)}")


def cmd_train_ensure(args) -> None:
    with lifecycle_lock(wait=False):
        state = load_state()
        if not state.get("train"):
            log("train: no run; idle")
            return
        log(f"train {state['train']['run']}: {train_tick(state)}")


def cmd_train_down(args) -> None:
    with lifecycle_lock(wait=True):
        state = load_state()
        if state.get("train"):
            state["train"]["stopped"] = True
            save_state(state)
        checkpoint_and_stop()
        down(TRAIN)
    log("train: stopped; train-up with the same --run and arguments resumes from the last checkpoint")


def cmd_train_status(args) -> None:
    run = load_state().get("train")
    if not run:
        sys.exit("no training run")
    print(json.dumps({k: run.get(k) for k in ("run", "trainer_args", "failures", "done", "failed", "stopped",
                                               "adapter", "fetched", "max_usd")}, indent=1))
    print(f"spent: ${cluster_cost_since(TRAIN, run['started_at']):.2f}")
    if cluster_status(TRAIN) == "UP":
        r = ssh_out(TRAIN, f"tail -n 3 {MNT}/runs/{run['run']}/loss.jsonl; tail -n 1 {MNT}/runs/{run['run']}/eval.jsonl")
        print(r.stdout)


def cmd_train_fetch(args) -> None:
    run = args.run or load_state().get("train", {}).get("run")
    cluster = next((c for c in (TRAIN, SERVE) if cluster_status(c) == "UP"), None)
    if not run or not cluster:
        sys.exit("need --run and an UP cluster with the volume (fh-train or fh-serve)")
    log(f"fetched {fetch_run(cluster, run).relative_to(ROOT)}")


def cmd_download(args) -> None:
    state = load_state()
    if state.get("serve") or active_train(state):
        sys.exit("download refused: serving or training holds the volume")
    try:
        sky("launch", "-y", "--down", "-c", DOWNLOAD, str(HERE / "download.yaml"), "--workdir", str(HERE))
    finally:
        down(DOWNLOAD)


def cmd_volume(args) -> None:
    sky("volumes", "apply", "-y", str(HERE / "volume.yaml"))


def cmd_status(args) -> None:
    state = load_state()
    serve = state.get("serve")
    print(json.dumps({"serve": serve, "serve_cluster": cluster_status(SERVE), "tunnel_pid": tunnel_pid(),
                      "healthy": healthy(served_names(serve.get("lora"))) if serve else None,
                      "train": (state.get("train") or {}).get("run"), "train_cluster": cluster_status(TRAIN)},
                     indent=1))


def test_png_data_url(text: str) -> str:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (480, 160), "white")
    d = ImageDraw.Draw(img)
    fonts = sorted((ROOT / "framehunt/assets/fonts").glob("DejaVuSans*.ttf"))
    d.text((20, 50), text, fill="black",
           font=ImageFont.truetype(str(fonts[0]), 48) if fonts else ImageFont.load_default(size=48))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def chat(url: str, key: str, body: dict) -> tuple[dict, float]:
    t0 = time.time()
    r = httpx.post(f"{url}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=600)
    r.raise_for_status()
    return r.json(), time.time() - t0


def stream_tps(url: str, key: str, max_tokens: int = 512) -> dict:
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
            usage = ev.get("usage") or usage
    t1 = time.time()
    n = usage["completion_tokens"] if usage else 0
    return {"ttft_s": round((first or t1) - t0, 3), "total_s": round(t1 - t0, 2), "completion_tokens": n,
            "decode_tok_per_s": round((n - 1) / max(t1 - (first or t0), 1e-6), 1)}


def cmd_test(args) -> None:
    key = api_key()
    url = local_base_url()
    results: dict = {"base_url": load_env().get("QWEN_BASE_URL"),
                     "no_key_status": httpx.get(f"{url}/models", timeout=10).status_code}
    code = "K7Q2ZP"
    body = {"model": SERVED_NAME, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False},
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
    print(json.dumps(results, indent=1))
    (HERE / "last_test.json").write_text(json.dumps({"ts": datetime.now(UTC).isoformat(), **results}, indent=1) + "\n")


def harbor_agent_entry(base_url: str, served_name: str = SERVED_NAME, opencode_version: str | None = None) -> dict:
    model = dict(json.loads((HERE / "opencode-qwen.json").read_text())["provider"]["verda-qwen"]["models"][SERVED_NAME])
    if served_name == LORA_NAME:
        model["name"] = f"{model['name']} (FrameHunt LoRA)"
    kwargs: dict = {"opencode_config": {"provider": {"verda-qwen": {"models": {served_name: model}}}}}
    if opencode_version:
        kwargs["version"] = opencode_version
    return {"name": "opencode", "model_name": f"verda-qwen/{served_name}", "base_url": base_url,
            "model_api_key_name": "QWEN_API_KEY", "model_api_format": "openai_chat_completions", "kwargs": kwargs}


def cmd_harbor_agent(args) -> None:
    url = load_env().get("QWEN_BASE_URL") or container_base_url()
    print(json.dumps({"agents": [harbor_agent_entry(url, args.served_name, args.opencode_version)]}, indent=1))


# ---- parallel evals: one Harbor run per box (eval.yaml) ---------------------------------------------
EVAL_YAML = HERE / "eval.yaml"
EVAL_STATE_PATH = HERE / ".fh-eval.json"
EVAL_LOCK_PATH = HERE / ".fh-eval.lock"
RUNS_DIR = ROOT / "runs"
EVAL_RELAY_URL = "http://172.17.0.1:18000/v1"   # socat on the box's Docker bridge -> vLLM 127.0.0.1:8000
OPENCODE_VERSION = "1.19.5"
EVAL_TERMINAL = {"SUCCEEDED", "FAILED", "FAILED_SETUP", "FAILED_DRIVER", "FAILED_PRECHECKS", "CANCELLED"}
EVAL_NAME_RE = r"[a-z0-9][a-z0-9-]{0,15}"
LORA_FILES = ("adapter_config.json", "adapter_model.safetensors")
EVAL_PROBE = r"""
J=~/fh/runs/$1
echo "trials=$(ls -d "$J"/*/ 2>/dev/null | wc -l)"
echo "done=$(ls "$J"/*/result.json 2>/dev/null | wc -l)"
echo "rewards=$(ls "$J"/*/verifier/reward.txt 2>/dev/null | wc -l)"
echo "tasks=$(ls -d ~/fh/in/tasks/*/ 2>/dev/null | wc -l)"
echo "phase=$(tail -n 1 ~/fh/status/timings.txt 2>/dev/null | cut -d' ' -f1)"
curl -s -m 5 http://127.0.0.1:8000/metrics 2>/dev/null |
  awk '/^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)[{ ]/ {
         split($1, a, "{"); s[a[1]] += $NF } END { for (k in s) print k "=" s[k] }'
"""


@contextlib.contextmanager
def _eval_lock():
    with EVAL_LOCK_PATH.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def load_evals() -> dict:
    return json.loads(EVAL_STATE_PATH.read_text()) if EVAL_STATE_PATH.exists() else {}


def save_evals(evals: dict) -> None:
    tmp = EVAL_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(evals, indent=1) + "\n")
    tmp.replace(EVAL_STATE_PATH)


def eval_yaml_resources() -> dict:
    import yaml

    return yaml.safe_load(EVAL_YAML.read_text())["resources"]


def eval_yaml_envs() -> dict:
    import yaml

    return yaml.safe_load(EVAL_YAML.read_text())["envs"]


def eval_names(split: str, label: str, i: int) -> tuple[str, str]:
    return f"fh-ev-{split}-{label}-{i}", f"fh-{split}-{label}-run{i}"


def git_commit() -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return r.stdout.strip() or None


def lora_rank(lora: Path) -> int:
    return int(json.loads((lora / "adapter_config.json").read_text())["r"])


def eval_definition(a) -> dict:
    """Everything that must be identical across a fleet's boxes (and is recorded with every run)."""
    tasks = Path(a.tasks).resolve()
    if not tasks.is_dir() or not any(p.is_dir() for p in tasks.iterdir()):
        raise SystemExit(f"{tasks}: not a directory of Harbor tasks (build it with scripts/build_tasks.py)")
    split = a.split or tasks.name
    lora = Path(a.lora).resolve() if a.lora else None
    if lora:
        if (lora / "final").is_dir():
            lora = lora / "final"
        missing = [f for f in LORA_FILES if not (lora / f).is_file()]
        if missing:
            raise SystemExit(f"{lora}: missing {missing}")
        if not 1 <= lora_rank(lora) <= LORA_MAX_RANK:
            raise SystemExit(f"{lora}: LoRA rank {lora_rank(lora)} is outside 1..{LORA_MAX_RANK}")
    label = a.label or ("qwen-fh" if lora else "qwen")
    for name, v in (("split", split), ("label", label)):
        if not re.fullmatch(EVAL_NAME_RE, v):
            raise SystemExit(f"{name} {v!r}: use lowercase letters, digits and '-' (max 16)")
    res, envs = eval_yaml_resources(), eval_yaml_envs()
    return {
        "fleet": f"{split}-{label}", "split": split, "label": label,
        "tasks_dir": str(tasks), "tasks_sha": data_sha(tasks), "n_tasks": sum(p.is_dir() for p in tasks.iterdir()),
        "lora_dir": str(lora) if lora else None,
        "lora_sha": sha_files([lora / f for f in LORA_FILES]) if lora else None,
        "served_name": LORA_NAME if lora else SERVED_NAME,
        "instance_type": a.instance_type or res["instance_type"],
        "use_spot": not a.on_demand, "disk_size": a.disk_size or res["disk_size"],
        "n_concurrent": a.n, "model_repo": a.model_repo or envs["MODEL_REPO"],
        "model_rev": a.model_rev or envs["MODEL_REV"], "vllm_version": envs["VLLM_VERSION"],
        "harbor_version": envs["HARBOR_VERSION"], "opencode_version": OPENCODE_VERSION,
        "max_run_hours": a.max_run_hours, "eval_yaml_sha": sha_files([EVAL_YAML]), "git_commit": git_commit(),
    }


COMPARABLE_KEYS = ("tasks_sha", "lora_sha", "served_name", "instance_type", "disk_size", "n_concurrent",
                   "model_repo", "model_rev", "vllm_version", "harbor_version", "opencode_version", "eval_yaml_sha")


def stage_eval(defn: dict) -> Path:
    """The dir eval.yaml mounts at ~/fh/in: tasks (secret seeds, kept 0700), Harbor config, LoRA, metadata."""
    dest = STAGE_DIR / f"eval-{defn['fleet']}"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, mode=0o700)
    dest.chmod(0o700)

    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(defn["tasks_dir"], dest / "tasks", copy_function=link_or_copy)
    if data_sha(dest / "tasks") != defn["tasks_sha"]:
        raise SystemExit(f"{defn['tasks_dir']} changed while staging; try again")
    for d in [dest / "tasks", *(p for p in (dest / "tasks").rglob("*") if p.is_dir())]:
        d.chmod(0o700)
    if defn["lora_dir"]:
        (dest / "lora").mkdir(mode=0o700)
        for f in LORA_FILES:
            shutil.copy2(Path(defn["lora_dir"]) / f, dest / "lora" / f)
        if sha_files([dest / "lora" / f for f in LORA_FILES]) != defn["lora_sha"]:
            raise SystemExit(f"{defn['lora_dir']} changed while staging; try again")
    cfg = {"agents": [harbor_agent_entry(EVAL_RELAY_URL, defn["served_name"], defn["opencode_version"])]}
    (dest / "cfg.json").write_text(json.dumps(cfg, indent=1) + "\n")
    shutil.copy2(HERE / "verify_weights.py", dest / "verify_weights.py")
    (dest / "meta.json").write_text(json.dumps(defn, indent=1) + "\n")
    (dest / ".skyignore").write_text("")   # sync everything (tasks/ is git-ignored in the repo)
    return dest


def eval_launch_args(defn: dict, stage: Path, i: int) -> list[str]:
    cluster, job = eval_names(defn["split"], defn["label"], i)
    envs = {"JOB": job, "STAGE": str(stage), "N_CONCURRENT": str(defn["n_concurrent"]),
            "LORA": "1" if defn["lora_dir"] else "0", "MODEL_REPO": defn["model_repo"],
            "MODEL_REV": defn["model_rev"], "MAX_RUN_HOURS": str(defn["max_run_hours"])}
    args = ["launch", "-y", "-d", "-c", cluster, str(EVAL_YAML), "--secret", "QWEN_API_KEY",
            "--instance-type", defn["instance_type"], "--disk-size", str(defn["disk_size"]),
            "--use-spot" if defn["use_spot"] else "--no-use-spot"]
    for k, v in envs.items():
        args += ["--env", f"{k}={v}"]
    return args


def launch_eval_boxes(defn: dict, stage: Path, runs: list[int]) -> dict[int, int]:
    """`sky launch -d` for every box at once; each returns when its job is submitted. Logs in the stage dir."""
    sky_bin = str(Path(sys.executable).parent / "sky")
    env = {**os.environ, "QWEN_API_KEY": api_key()}
    procs = {}
    for i in runs:
        logf = (STAGE_DIR / f"eval-{defn['fleet']}-launch-{i}.log").open("w")
        procs[i] = (subprocess.Popen([sky_bin, *eval_launch_args(defn, stage, i)], env=env, stdout=logf,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL), logf)
        log(f"eval: launching {eval_names(defn['split'], defn['label'], i)[0]} (log {logf.name})")
    rcs = {}
    for i, (p, logf) in procs.items():
        rcs[i] = p.wait()
        logf.close()
    return rcs


def cmd_eval_up(args) -> None:
    defn = eval_definition(args)
    runs = args.only or list(range(1, args.runs + 1))
    with _eval_lock():
        evals = load_evals()
        fleet = evals.get(defn["fleet"])
        if fleet:
            diff = [k for k in COMPARABLE_KEYS if fleet["meta"].get(k) != defn.get(k)]
            if diff and not args.new:
                raise SystemExit(f"fleet {defn['fleet']} exists with different {diff}; its runs would not be "
                                 f"comparable. Use --new to start over (old results in runs/ are replaced).")
            if args.new:
                fleet = None
        if fleet is None:
            fleet = {"meta": {**defn, "created_at": time.time()}, "boxes": {}, "max_usd": args.max_usd}
        fleet["max_usd"] = args.max_usd
        for i in runs:
            cluster, _ = eval_names(defn["split"], defn["label"], i)
            box = fleet["boxes"].get(str(i))
            if box and box["state"] == "running" and cluster_status(cluster) is not None:
                raise SystemExit(f"{cluster} is running; eval-down --only {i} first to relaunch it")
            if cluster_status(cluster) is not None:
                down(cluster)
        evals[defn["fleet"]] = fleet
        save_evals(evals)
        stage = stage_eval(defn)
        for i in runs:
            cluster, job = eval_names(defn["split"], defn["label"], i)
            fleet["boxes"][str(i)] = {"cluster": cluster, "job": job, "state": "launching",
                                      "launched_at": time.time(),
                                      "launches": fleet["boxes"].get(str(i), {}).get("launches", 0) + 1}
        save_evals(evals)
    rcs = launch_eval_boxes(defn, stage, runs)
    with _eval_lock():
        evals = load_evals()
        fleet = evals[defn["fleet"]]
        for i, rc in rcs.items():
            box = fleet["boxes"][str(i)]
            if rc == 0:
                box["state"] = "running"
            else:
                box["state"] = "launch_failed"
                if cluster_status(box["cluster"]) is not None:
                    down(box["cluster"])
        save_evals(evals)
    for i in runs:
        b = fleet["boxes"][str(i)]
        log(f"eval {defn['fleet']} run {i}: {b['cluster']} {b['state']}")
    if any(rc != 0 for rc in rcs.values()):
        log(f"eval: some launches failed (no capacity?); see the launch logs in {STAGE_DIR}; "
            f"retry with eval-up ... --only N")


def pick_fleet(evals: dict, name: str | None) -> str:
    if name:
        if name not in evals:
            raise SystemExit(f"no eval fleet {name!r} (have {sorted(evals)})")
        return name
    if not evals:
        raise SystemExit("no eval fleets; run eval-up first")
    return max(evals, key=lambda k: evals[k]["meta"]["created_at"])


def fleet_cost(fleet: dict) -> dict[str, float]:
    return {b["cluster"]: cluster_cost_since(b["cluster"], fleet["meta"]["created_at"])
            for b in fleet["boxes"].values()}


def parse_probe(out: str) -> dict:
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            with contextlib.suppress(ValueError):
                d[k.removeprefix("vllm:")] = float(v) if "." in v or "e" in v else int(v)
            d.setdefault(k.removeprefix("vllm:"), v)
    return d


def probe_box(box: dict) -> dict:
    r = ssh_out(box["cluster"], f"bash -s {shlex.quote(box['job'])} <<'EOF'\n{EVAL_PROBE}\nEOF", timeout=60)
    return parse_probe(r.stdout) if r.returncode == 0 else {"probe_error": r.stderr.strip()[-200:]}


def fetch_eval(fleet: dict, box: dict, job_status: str | None) -> Path:
    """rsync ~/fh/runs/<job>/ from the box to runs/<job>/ and write runs/<job>/fh-eval-meta.json."""
    dest = RUNS_DIR / box["job"]
    dest.mkdir(parents=True, exist_ok=True)
    rsh = shlex.join(ssh_base(box["cluster"]))
    subprocess.run(["rsync", "-a", "--delete", "-e", rsh, f"{box['cluster']}:fh/runs/{box['job']}/", f"{dest}/"],
                   check=True, timeout=FETCH_TIMEOUT_S)
    remote = ssh_out(box["cluster"], "cat ~/fh/status/box.json; echo; cat ~/fh/status/timings.txt", timeout=60)
    lines = remote.stdout.splitlines()
    boxinfo = {}
    with contextlib.suppress(ValueError, IndexError):
        boxinfo = json.loads(lines[0])
    timings = {}
    for line in lines[1:]:
        k, _, v = line.partition(" ")
        if v.isdigit():
            timings[k] = int(v)
    meta = {**fleet["meta"], "run": box["job"], "cluster": box["cluster"], "job_status": job_status,
            "complete": job_status == "SUCCEEDED", "fetched_at": time.time(), "launches": box.get("launches"),
            "launched_at": box.get("launched_at"), "cost_so_far": cluster_cost_since(box["cluster"],
                                                                                      fleet["meta"]["created_at"]),
            "box": boxinfo, "timings": timings}
    if timings.get("job_start"):
        meta["timings_rel_s"] = {k: v - timings["job_start"] for k, v in timings.items()}
    (dest / "fh-eval-meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return dest


def eval_tick(fleet: dict) -> list[str]:
    """One watchdog step: fetch and delete finished boxes, report lost ones, enforce the cost cap."""
    notes = []
    costs = fleet_cost(fleet)
    total = sum(costs.values())
    over = total >= fleet["max_usd"]
    for i, box in sorted(fleet["boxes"].items(), key=lambda kv: int(kv[0])):
        if box["state"] not in ("running", "launching"):
            continue
        status = cluster_status(box["cluster"])
        if status is None and box["state"] == "launching" and time.time() - box["launched_at"] < 1800:
            continue
        if status is None:
            box["state"] = "lost"
            notes.append(f"run {i}: {box['cluster']} is gone (spot preemption or autodown) before its results "
                         f"were fetched; relaunch with eval-up ... --only {i}")
            continue
        job = last_job_status(box["cluster"]) if status == "UP" else None
        if job in EVAL_TERMINAL or (over and status == "UP"):
            try:
                fetch_eval(fleet, box, job)
                box["fetched_at"] = time.time()
            except (OSError, subprocess.SubprocessError) as e:
                box["fetch_error"] = f"{type(e).__name__}: {e}"
                if not over:
                    notes.append(f"run {i}: fetch failed ({box['fetch_error']}); retrying next tick")
                    continue
            box["job_status"] = job
            box["state"] = "budget" if over and job not in EVAL_TERMINAL else "done"
            down(box["cluster"])
            notes.append(f"run {i}: job {job}; fetched to runs/{box['job']}/ and deleted {box['cluster']}")
        elif over:
            down(box["cluster"])
            box["state"] = "budget"
            notes.append(f"run {i}: cost cap; deleted {box['cluster']} ({status})")
    if over:
        notes.append(f"cost cap reached: {total:.2f} >= {fleet['max_usd']}")
    return notes


def cmd_eval_status(args) -> None:
    evals = load_evals()
    name = pick_fleet(evals, args.fleet)
    fleet = evals[name]
    costs = fleet_cost(fleet)
    m = fleet["meta"]
    print(f"fleet {name}: {m['instance_type']} {'spot' if m['use_spot'] else 'on-demand'}, -n {m['n_concurrent']}, "
          f"{m['n_tasks']} tasks, model {m['model_repo']}@{m['model_rev'][:7]}, served {m['served_name']}; "
          f"cost {sum(costs.values()):.2f} of cap {fleet['max_usd']}")
    for i, box in sorted(fleet["boxes"].items(), key=lambda kv: int(kv[0])):
        status = cluster_status(box["cluster"])
        line = f"  run {i} {box['cluster']}: {box['state']}, cluster {status}"
        if status == "UP":
            line += f", job {last_job_status(box['cluster'])}"
            p = probe_box(box)
            if "probe_error" in p:
                line += f", probe failed: {p['probe_error']}"
            else:
                line += (f", phase {p.get('phase') or '-'}, trials {p.get('done', 0)}/{p.get('tasks', '?')} done "
                         f"({p.get('rewards', 0)} rewards), vLLM running {p.get('num_requests_running', '-')} "
                         f"waiting {p.get('num_requests_waiting', '-')} kv {p.get('kv_cache_usage_perc', '-')} "
                         f"preemptions {p.get('num_preemptions_total', '-')}")
        elif box["state"] == "running" and status is None:
            line += " (gone: preempted? eval-wait reports it; relaunch with --only)"
        line += f", cost {costs.get(box['cluster'], 0):.2f}"
        print(line)


def cmd_eval_fetch(args) -> None:
    with _eval_lock():
        evals = load_evals()
        fleet = evals[pick_fleet(evals, args.fleet)]
        for i, box in sorted(fleet["boxes"].items(), key=lambda kv: int(kv[0])):
            if args.only and int(i) not in args.only:
                continue
            if cluster_status(box["cluster"]) != "UP":
                log(f"run {i}: {box['cluster']} is not UP; nothing to fetch")
                continue
            job = last_job_status(box["cluster"])
            if job not in EVAL_TERMINAL and not args.partial:
                log(f"run {i}: job {job} still running; --partial fetches what is there")
                continue
            log(f"run {i}: fetched {fetch_eval(fleet, box, job).relative_to(ROOT)} (job {job})")
        save_evals(evals)


def cmd_eval_wait(args) -> None:
    while True:
        with _eval_lock():
            evals = load_evals()
            name = pick_fleet(evals, args.fleet)
            fleet = evals[name]
            for note in eval_tick(fleet):
                log(f"eval {name}: {note}")
            save_evals(evals)
            active = [i for i, b in fleet["boxes"].items() if b["state"] in ("running", "launching")]
        if not active:
            states = {i: b["state"] for i, b in fleet["boxes"].items()}
            log(f"eval {name}: no box left running; {states}")
            if any(s != "done" for s in states.values()):
                sys.exit(1)
            return
        time.sleep(args.interval)


def cmd_eval_down(args) -> None:
    with _eval_lock():
        evals = load_evals()
        fleet = evals[pick_fleet(evals, args.fleet)]
        for i, box in sorted(fleet["boxes"].items(), key=lambda kv: int(kv[0])):
            if args.only and int(i) not in args.only:
                continue
            status = cluster_status(box["cluster"])
            if status == "UP" and not args.no_fetch and not box.get("fetched_at"):
                with contextlib.suppress(OSError, subprocess.SubprocessError):
                    fetch_eval(fleet, box, last_job_status(box["cluster"]))
                    box["fetched_at"] = time.time()
            if status is not None:
                down(box["cluster"])
            if box["state"] in ("running", "launching"):
                box["state"] = "stopped"
            log(f"run {i}: {box['cluster']} deleted ({box['state']})")
        save_evals(evals)


def add_eval_parsers(sub) -> None:
    s = sub.add_parser("eval-up", help="launch one box per run (eval.yaml); see README")
    s.add_argument("--tasks", required=True, help="Harbor tasks dir, e.g. tasks/eval")
    s.add_argument("--split", help="name in job/cluster names (default: the tasks dir name)")
    s.add_argument("--label", help="default qwen, or qwen-fh with --lora")
    s.add_argument("--lora", help=f"adapter dir (train/out/<run> or its final/), served as {LORA_NAME}")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--only", type=int, action="append", help="(re)launch only this run index (repeatable)")
    s.add_argument("-n", type=int, default=10, help="concurrent trials per box")
    s.add_argument("--instance-type", help="default: eval.yaml's pinned type")
    s.add_argument("--disk-size", type=int)
    s.add_argument("--on-demand", action="store_true")
    s.add_argument("--model-repo", help="smoke tests only (recorded in the metadata)")
    s.add_argument("--model-rev")
    s.add_argument("--max-usd", type=float, default=60, help="cost cap for the whole fleet (eval-wait enforces it)")
    s.add_argument("--max-run-hours", type=int, default=4, help="hard limit per box job")
    s.add_argument("--new", action="store_true", help="replace an existing fleet with a different definition")
    s.set_defaults(fn=cmd_eval_up)
    for name, fn in (("eval-status", cmd_eval_status), ("eval-fetch", cmd_eval_fetch), ("eval-wait", cmd_eval_wait),
                     ("eval-down", cmd_eval_down)):
        s = sub.add_parser(name)
        s.add_argument("--fleet", help="<split>-<label> (default: the newest)")
        if name in ("eval-fetch", "eval-down"):
            s.add_argument("--only", type=int, action="append")
        if name == "eval-fetch":
            s.add_argument("--partial", action="store_true", help="also fetch running jobs (snapshot)")
        if name == "eval-down":
            s.add_argument("--no-fetch", action="store_true")
        if name == "eval-wait":
            s.add_argument("--interval", type=float, default=120)
        s.set_defaults(fn=fn)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("volume").set_defaults(fn=cmd_volume)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    s = sub.add_parser("serve-up")
    s.add_argument("--lora", help=f"adapter dir on the volume, e.g. {MNT}/adapters/<run>; served as {LORA_NAME}")
    s.add_argument("--on-demand", action="store_true")
    s.add_argument("--no-wait", action="store_true")
    s.add_argument("--keep", action="store_true", help="keep the box when it does not become healthy")
    s.add_argument("--timeout-min", type=float, default=40)
    s.set_defaults(fn=cmd_serve_up)
    s = sub.add_parser("serve-ensure")
    s.add_argument("--wait-min", type=float, default=0)
    s.set_defaults(fn=cmd_serve_ensure)
    sub.add_parser("serve-down").set_defaults(fn=cmd_serve_down)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    s = sub.add_parser("harbor-agent")
    s.add_argument("--served-name", default=SERVED_NAME, choices=[SERVED_NAME, LORA_NAME])
    s.add_argument("--opencode-version")
    s.set_defaults(fn=cmd_harbor_agent)
    s = sub.add_parser("train-up")
    s.add_argument("--run", required=True)
    s.add_argument("--data", required=True, help="dir with sft.jsonl, tools.json, images/")
    s.add_argument("--epochs", type=float, default=1)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--grad-accum", type=int, default=8)
    s.add_argument("--lora-r", type=int, default=16)
    s.add_argument("--save-every", type=int, default=10)
    s.add_argument("--trainer-arg", action="append")
    s.add_argument("--max-usd", type=float, default=45)
    s.add_argument("--on-demand", action="store_true")
    s.set_defaults(fn=cmd_train_up)
    add_eval_parsers(sub)
    sub.add_parser("train-ensure").set_defaults(fn=cmd_train_ensure)
    sub.add_parser("train-down").set_defaults(fn=cmd_train_down)
    sub.add_parser("train-status").set_defaults(fn=cmd_train_status)
    s = sub.add_parser("train-fetch")
    s.add_argument("--run")
    s.set_defaults(fn=cmd_train_fetch)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
