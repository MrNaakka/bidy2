"""FrameHunt's Modal tooling (local side): weights, vLLM endpoints, Harbor evals, LoRA training.

Run from the repo root:  infra/modal/fh <command>      (see infra/modal/README.md)
The Modal side is app.py (deployed as the app `fh-qwen`).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATE_PATH = HERE / ".state.json"
RUNS_DIR = ROOT / "runs"
APP = "fh-qwen"
OPENCODE_VERSION = "1.19.5"
HARBOR_VERSION = "0.25.0"
LORA_FILES = ("adapter_config.json", "adapter_model.safetensors")
LORA_MAX_RANK = 16
NAME_RE = r"[a-z0-9][a-z0-9-]{0,23}"


def _load_verda():
    spec = importlib.util.spec_from_file_location("fh_verda", ROOT / "infra/verda/fh.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


verda = _load_verda()   # harbor_agent_entry, data_sha, sha_files, chat, test_png_data_url, trainer flags


def log(msg: str) -> None:
    print(f"[fh-modal {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_state() -> dict:
    return json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}


def save_state(st: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1) + "\n")
    tmp.replace(STATE_PATH)


def api_key() -> str:
    key = os.environ.get("QWEN_API_KEY") or verda.load_env().get("QWEN_API_KEY")
    if not key:
        raise SystemExit("QWEN_API_KEY is not set (env or .env)")
    return key


def git_commit() -> str | None:
    return verda.git_commit()


def remote_dict() -> modal.Dict:
    return modal.Dict.from_name("fh-qwen-state", create_if_missing=True)


def fn(name: str) -> modal.Function:
    return modal.Function.from_name(f"{APP}-train" if name.startswith("train") else APP, name)


# ------------------------------------------------------------------------------------------- setup
def cmd_secret(args) -> None:
    """Modal Secret fh-qwen-key from QWEN_API_KEY (env or .env); the value is never printed."""
    modal.Secret.objects.create("fh-qwen-key", {"QWEN_API_KEY": api_key()}, allow_existing=True)
    log("Modal secret fh-qwen-key is set")


def deploy(which: str = "app") -> None:
    """which: "app" (fh-qwen: weights, serve) or "train_app" (fh-qwen-train). Image builds show in the output."""
    try:
        modal.Secret.from_name("fh-qwen-key").hydrate()
    except modal.exception.NotFoundError:
        cmd_secret(None)
    modal_cli = Path(sys.executable).parent / "modal"
    name = APP if which == "app" else f"{APP}-train"
    log(f"deploying {name} (app.py::{which})")
    subprocess.run([str(modal_cli), "deploy", f"{HERE / 'app.py'}::{which}", "--name", name], check=True)


def cmd_deploy(args) -> None:
    deploy("app")
    if args.train:
        deploy("train_app")


def cmd_weights(args) -> None:
    if not args.no_deploy:
        deploy()
    log("weights: download + verify on fh-qwen-weights (skips when the verified revision is there)")
    log(json.dumps(fn("download_weights").remote(force=args.force)))


# ------------------------------------------------------------------------------------------- serving
def upload_lora(lora: Path) -> str:
    """Adapter files to fh-qwen-weights:adapters/<sha>/ (content-addressed); returns <sha>."""
    sha = verda.sha_files([lora / f for f in LORA_FILES])
    vol = modal.Volume.from_name("fh-qwen-weights")
    have = {e.path.rsplit("/", 1)[-1] for e in vol.listdir(f"adapters/{sha}")} if _vol_exists(vol, f"adapters/{sha}") else set()
    if not set(LORA_FILES) <= have:
        log(f"uploading adapter {lora} as adapters/{sha}")
        with vol.batch_upload(force=True) as b:
            for f in LORA_FILES:
                b.put_file(str(lora / f), f"adapters/{sha}/{f}")
    return sha


def _vol_exists(vol: modal.Volume, path: str) -> bool:
    try:
        vol.listdir(path)
        return True
    except Exception:  # noqa: BLE001
        return False


def resolve_lora(lora: str | None) -> Path | None:
    if not lora:
        return None
    p = Path(lora).resolve()
    if (p / "final").is_dir():
        p = p / "final"
    missing = [f for f in LORA_FILES if not (p / f).is_file()]
    if missing:
        raise SystemExit(f"{p}: missing {missing}")
    r = int(json.loads((p / "adapter_config.json").read_text())["r"])
    if not 1 <= r <= LORA_MAX_RANK:
        raise SystemExit(f"{p}: LoRA rank {r} is outside 1..{LORA_MAX_RANK}")
    return p


def spawn_endpoint(tag: str, lora_sha: str | None, gpu: str, idle_min: float, max_hours: float) -> str:
    d = remote_dict()
    d.pop(f"stop:{tag}", None)
    d.pop(f"serve:{tag}", None)
    f = fn("serve")
    if gpu != "H100":
        f = f.with_options(gpu=gpu)
    call = f.spawn(tag=tag, lora_sha=lora_sha, idle_min=idle_min, max_hours=max_hours)
    st = load_state()
    st.setdefault("endpoints", {})[tag] = {"call_id": call.object_id, "gpu": gpu, "lora_sha": lora_sha,
                                           "spawned_at": time.time()}
    save_state(st)
    log(f"endpoint {tag}: spawned call {call.object_id} on {gpu}")
    return call.object_id


def endpoint_info(tag: str) -> dict:
    return remote_dict().get(f"serve:{tag}") or {}


def wait_healthy(tag: str, timeout_min: float = 45) -> dict:
    """Waits until the endpoint reports healthy AND answers /v1/models through its public URL."""
    key = api_key()
    deadline = time.time() + timeout_min * 60
    last = None
    while time.time() < deadline:
        info = endpoint_info(tag)
        ph = info.get("phase")
        if ph != last:
            log(f"endpoint {tag}: {ph}")
            last = ph
        if ph in ("failed", "ended"):
            tail = "\n".join((info.get("log_tail") or [])[-40:])
            raise RuntimeError(f"endpoint {tag} {ph}: {info.get('error') or info.get('reason')}\n{tail}")
        if ph == "healthy" and info.get("url"):
            try:
                r = httpx.get(f"{info['url']}/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=20)
                if r.status_code == 200 and set(info["served_names"]) <= {m["id"] for m in r.json()["data"]}:
                    log(f"endpoint {tag}: healthy at {info['url']} (load {info.get('load_s')} s)")
                    return info
            except httpx.HTTPError:
                pass
        time.sleep(15)
    raise RuntimeError(f"endpoint {tag}: not healthy after {timeout_min} min")


def stop_endpoint(tag: str, grace_s: float = 0) -> None:
    """Stop flag (the container sees it within 10 s, stops vLLM, publishes `ended` with the log tail), then
    after up to grace_s for that, cancel the call."""
    st = load_state()
    ep = st.get("endpoints", {}).get(tag)
    remote_dict()[f"stop:{tag}"] = True
    deadline = time.time() + grace_s
    while time.time() < deadline and endpoint_info(tag).get("phase") not in ("ended", "failed"):
        time.sleep(5)
    if ep:
        time.sleep(1)
        with contextlib.suppress(Exception):
            modal.FunctionCall.from_id(ep["call_id"]).cancel()
        st["endpoints"].pop(tag, None)
        save_state(st)
    log(f"endpoint {tag}: stopped")


def cmd_serve_up(args) -> None:
    deploy()
    lora = resolve_lora(args.lora)
    sha = upload_lora(lora) if lora else None
    spawn_endpoint(args.tag, sha, args.gpu, args.idle_min, args.max_hours)
    try:
        info = wait_healthy(args.tag)
    except BaseException:
        stop_endpoint(args.tag)
        raise
    print(json.dumps({"tag": args.tag, "base_url": info["url"] + "/v1", "served_names": info["served_names"],
                      "gpus": info.get("gpus")}, indent=1))


def cmd_serve_down(args) -> None:
    tags = list(load_state().get("endpoints", {})) if args.all else [args.tag]
    for t in tags:
        stop_endpoint(t)


def cmd_status(args) -> None:
    st = load_state()
    out = {"endpoints": {t: {k: v for k, v in endpoint_info(t).items() if k not in ("log_tail", "vllm_cmd")}
                         for t in st.get("endpoints", {})},
           "train": {r: remote_dict().get(f"train:{r}") for r in st.get("train", {})}}
    print(json.dumps(out, indent=1, default=str))


def cmd_test(args) -> None:
    """Vision + tool-call requests through the endpoint's public URL (like `infra/verda/fh test`)."""
    info = endpoint_info(args.tag)
    if info.get("phase") != "healthy":
        raise SystemExit(f"endpoint {args.tag} is not healthy: {info.get('phase')}")
    url, key = info["url"] + "/v1", api_key()
    res: dict = {"base_url": url, "no_key_status": httpx.get(f"{url}/models", timeout=20).status_code}
    code = "K7Q2ZP"
    for name in info["served_names"]:
        body = {"model": name, "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What code is written in this image? Reply with the code only."},
                    {"type": "image_url", "image_url": {"url": verda.test_png_data_url(f"CODE: {code}")}}]}]}
        resp, dt = verda.chat(url, key, body)
        ans = resp["choices"][0]["message"]["content"] or ""
        res[f"vision[{name}]"] = {"answer": ans.strip()[:80], "correct": code in ans.upper().replace(" ", ""),
                                  "latency_s": round(dt, 2)}
        tools = [{"type": "function", "function": {
            "name": "view", "description": "Return one video frame at time t (seconds).",
            "parameters": {"type": "object", "properties": {
                "t": {"type": "number", "description": "time in seconds"},
                "out_width": {"type": "integer", "description": "output width in pixels"}}, "required": ["t"]}}}]
        body = {"model": name, "max_tokens": 2048, "tools": tools, "tool_choice": "auto",
                "messages": [{"role": "user", "content":
                              "Look at the frame at 12.5 seconds with an output width of 320. Use the tool."}]}
        resp, dt = verda.chat(url, key, body)
        msg = resp["choices"][0]["message"]
        tc = msg.get("tool_calls") or []
        res[f"tool_call[{name}]"] = {"latency_s": round(dt, 2), "call": tc[0]["function"] if tc else None,
                                     "has_reasoning": bool(msg.get("reasoning_content") or msg.get("reasoning")),
                                     "finish_reason": resp["choices"][0]["finish_reason"]}
    res["throughput"] = verda.stream_tps(url, key)
    print(json.dumps(res, indent=1))
    (HERE / "last_test.json").write_text(json.dumps({"ts": datetime.now(UTC).isoformat(), "tag": args.tag,
                                                     "gpus": info.get("gpus"), **res}, indent=1) + "\n")


# ------------------------------------------------------------------------------------------- evals
def run_one(i: int, tag: str, job: str, defn: dict, tasks: Path, keyenv: Path, args) -> int:
    """One eval run: its own endpoint, then Harbor (trials on Modal sandboxes), then metadata, then stop."""
    meta: dict = {"definition": defn, "job": job, "run": i, "endpoint_tag": tag, "timings": {}}
    meta["timings"]["spawn"] = time.time()
    rc = 1
    try:
        spawn_endpoint(tag, defn["lora_sha"], defn["gpu"], args.idle_min, args.max_hours)
        info = wait_healthy(tag)
        meta["timings"]["healthy"] = time.time()
        meta["endpoint"] = {k: v for k, v in info.items() if k not in ("log_tail", "url")}
        cfg = {"agents": [verda.harbor_agent_entry(info["url"] + "/v1", defn["served_name"], OPENCODE_VERSION)]}
        cfg_path = Path(tempfile.mkdtemp(prefix=f"fh-modal-{job}-")) / "cfg.json"
        cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
        shutil.rmtree(RUNS_DIR / job, ignore_errors=True)
        cmd = [args.harbor, "run", "-p", str(tasks), "--job-name", job, "-o", str(RUNS_DIR), "-n", str(args.n),
               "-r", "2", "--retry-include", "AgentSetupTimeoutError", "-y", "-e", "modal",
               "-c", str(cfg_path), "--env-file", str(keyenv)]
        meta["harbor_cmd"] = cmd
        logf = (ROOT / "runs" / f"{job}.harbor.log")
        log(f"{job}: harbor run (log {logf.relative_to(ROOT)})")
        meta["timings"]["harbor_start"] = time.time()
        with logf.open("w") as lf:
            rc = subprocess.run(["nice", "-n", "10", *cmd], stdout=lf, stderr=subprocess.STDOUT, cwd=ROOT,
                                timeout=args.max_hours * 3600).returncode
        meta["timings"]["harbor_end"] = time.time()
        meta["harbor_exit"] = rc
        with contextlib.suppress(httpx.HTTPError):
            m = httpx.get(f"{info['url']}/metrics", timeout=30).text
            (RUNS_DIR / job).mkdir(parents=True, exist_ok=True)
            (RUNS_DIR / job / "vllm-metrics.txt").write_text(m)
        log(f"{job}: harbor exit {rc}")
    except Exception as e:  # noqa: BLE001
        meta["error"] = repr(e)
        log(f"{job}: FAILED {e!r}")
    finally:
        stop_endpoint(tag, grace_s=120)   # lets the container publish its vLLM log tail before the cancel
        info = endpoint_info(tag)
        meta["endpoint_end"] = {k: info.get(k) for k in ("phase", "reason", "ended_at", "started_at", "healthy_at")}
        (RUNS_DIR / job).mkdir(parents=True, exist_ok=True)
        if info.get("log_tail"):
            (RUNS_DIR / job / "vllm-tail.log").write_text("\n".join(info["log_tail"]) + "\n")
        (RUNS_DIR / job / "fh-eval-meta.json").write_text(json.dumps(meta, indent=1, default=str) + "\n")
    return rc


def harbor_bin() -> str:
    """The Harbor CLI for `-e modal`: $FH_HARBOR, else the first `harbor` on PATH whose Python has the modal
    extra (modal + dockerfile_parse, used for the separate verifier sandbox). The repo .venv's harbor lacks it
    and `uv run` puts that one first on PATH; with it every trial fails at the verifier."""
    def ok(h: str) -> bool:
        first = Path(h).resolve().read_bytes()[:300].split(b"\n")[0]
        py = first[2:].decode().strip() if first.startswith(b"#!") else sys.executable
        return subprocess.run([py, "-c", "import modal, dockerfile_parse"], capture_output=True).returncode == 0

    if os.environ.get("FH_HARBOR"):
        cands = [os.environ["FH_HARBOR"]]
    else:
        cands = [str(Path(d) / "harbor") for d in os.environ.get("PATH", "").split(os.pathsep)
                 if d and os.access(Path(d) / "harbor", os.X_OK)]
    for h in cands:
        if ok(h):
            return h
    raise SystemExit(f"no harbor with harbor[modal] (modal, dockerfile_parse) among {cands}; set FH_HARBOR")


def cmd_eval(args) -> None:
    tasks = Path(args.tasks).resolve()
    if not tasks.is_dir() or not any(p.is_dir() for p in tasks.iterdir()):
        raise SystemExit(f"{tasks}: not a directory of Harbor tasks (scripts/build_tasks.py)")
    split = args.split or tasks.name
    lora = resolve_lora(args.lora)
    label = args.label or ("qwen-fh-modal" if lora else "qwen-modal")
    for v in (split, label):
        if not re.fullmatch(NAME_RE, v):
            raise SystemExit(f"{v!r}: use lowercase letters, digits and '-'")
    deploy()
    sha = upload_lora(lora) if lora else None
    served = verda.LORA_NAME if lora else verda.SERVED_NAME
    app_src = (HERE / "app.py").read_bytes()
    defn = {"backend": "modal", "split": split, "label": label, "tasks_dir": str(tasks),
            "tasks_sha": verda.data_sha(tasks), "n_tasks": sum(p.is_dir() for p in tasks.iterdir()),
            "lora_dir": str(lora) if lora else None, "lora_sha": sha, "served_name": served,
            "gpu": args.gpu, "n_concurrent": args.n, "harbor_env": "modal", "model_repo": "Qwen/Qwen3.8-27B",
            "model_rev": "37561816e66b5b9153b1b64c10968c18c1e6f4a0", "vllm_version": "0.33.0",
            "harbor_version": HARBOR_VERSION, "opencode_version": OPENCODE_VERSION,
            "app_py_sha": hashlib.sha256(app_src).hexdigest()[:16], "git_commit": git_commit(),
            "started": datetime.now(UTC).isoformat()}
    args.harbor = harbor_bin()
    hv = subprocess.run([args.harbor, "--version"], capture_output=True, text=True).stdout.strip().splitlines()
    if not hv or hv[-1] != HARBOR_VERSION:
        log(f"WARNING: local harbor is {hv[-1] if hv else '?'}, the fleet uses {HARBOR_VERSION}")
    runs = [int(x) for x in args.only.split(",")] if args.only else list(range(1, args.runs + 1))
    RUNS_DIR.mkdir(exist_ok=True)
    kd = Path(tempfile.mkdtemp(prefix="fh-modal-key-"))
    keyenv = kd / "key.env"
    with open(os.open(keyenv, os.O_WRONLY | os.O_CREAT, 0o600), "w") as f:
        f.write(f"QWEN_API_KEY={api_key()}\n")
    tags = {i: f"{split}-{label}-run{i}" for i in runs}

    def on_sig(signum, frame):
        log(f"signal {signum}: stopping endpoints")
        for t in tags.values():
            with contextlib.suppress(Exception):
                stop_endpoint(t)
        shutil.rmtree(kd, ignore_errors=True)
        os._exit(130)

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    rcs: dict[int, int] = {}
    threads = []
    try:
        for k, i in enumerate(runs):
            job = f"fh-{split}-{label}-run{i}"
            th = threading.Thread(target=lambda i=i, job=job: rcs.__setitem__(
                i, run_one(i, tags[i], job, defn, tasks, keyenv, args)), daemon=True)
            th.start()
            threads.append(th)
            if args.sequential:
                th.join()
            elif k + 1 < len(runs):
                time.sleep(20)   # stagger Harbor's start-up (image builds, sandbox creation)
        for th in threads:
            th.join()
    finally:
        shutil.rmtree(kd, ignore_errors=True)
    log(f"done: {json.dumps(rcs)}")
    sys.exit(0 if rcs and all(v == 0 for v in rcs.values()) else 1)


# ------------------------------------------------------------------------------------------- training
def upload_data(data: Path) -> tuple[str, str]:
    dsha = verda.data_sha(data)
    remote = f"data/{data.name}-{dsha}"
    vol = modal.Volume.from_name("fh-train", create_if_missing=True)
    if _vol_exists(vol, f"{remote}/.fh_sha"):
        return dsha, remote
    log(f"uploading {data} to fh-train:{remote}")
    with vol.batch_upload(force=True) as b:
        b.put_directory(str(data), remote)
        tmp = Path(tempfile.mkdtemp()) / ".fh_sha"
        tmp.write_text(dsha + "\n")
        b.put_file(str(tmp), f"{remote}/.fh_sha")
    return dsha, remote


def cmd_train_up(args) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run):
        raise SystemExit(f"--run {args.run!r}: use letters, digits, '.', '_' and '-'")
    deploy("train_app")
    data = Path(args.data).resolve()
    dsha, remote = upload_data(data)
    targs = verda.trainer_args(args.run, f"/t/{remote}", args)
    targs = [a.replace(f"{verda.MNT}/models/", "/w/models/").replace(f"{verda.MNT}/runs/", "/t/runs/") for a in targs]
    code_sha = verda.sha_files([ROOT / "train" / f for f in verda.TRAIN_CODE_FILES])
    run = {"run": args.run, "data_sha": dsha, "code_sha": code_sha, "trainer_args": targs, "backend": "modal"}
    manifest = json.dumps(run, sort_keys=True)
    f = fn("train")
    if args.gpu != "H100":
        f = f.with_options(gpu=args.gpu)
    remote_dict().pop(f"trainstop:{args.run}", None)
    call = f.spawn(run=args.run, trainer_args=targs, manifest=manifest)
    st = load_state()
    st.setdefault("train", {})[args.run] = {"call_id": call.object_id, "gpu": args.gpu, "spawned_at": time.time(),
                                            **run}
    save_state(st)
    log(f"train {args.run}: call {call.object_id} on {args.gpu} ({' '.join(targs)})")


def _read(vol: modal.Volume, path: str) -> str | None:
    try:
        return b"".join(vol.read_file(path)).decode()
    except Exception:  # noqa: BLE001
        return None


def cmd_train_status(args) -> None:
    vol = modal.Volume.from_name("fh-train")
    runs = [args.run] if args.run else list(load_state().get("train", {}))
    for r in runs:
        info = remote_dict().get(f"train:{r}")
        loss = [json.loads(x) for x in (_read(vol, f"runs/{r}/loss.jsonl") or "").splitlines() if x.strip()]
        ev = (_read(vol, f"runs/{r}/eval.jsonl") or "").strip().splitlines()
        secs = [x["seconds"] for x in loss[1:]]
        print(json.dumps({"run": r, "remote": info, "steps": len(loss), "last": loss[-1] if loss else None,
                          "mean_s_per_step_excl_first": round(sum(secs) / len(secs), 1) if secs else None,
                          "eval": ev[-3:]}, indent=1, default=str))


def cmd_train_fetch(args) -> None:
    out = ROOT / "train/out" / args.run
    out.mkdir(parents=True, exist_ok=True)
    modal_cli = Path(sys.executable).parent / "modal"
    for p in ("final", "loss.jsonl", "eval.jsonl", "train.log", "args.json", "manifest.json", "DONE"):
        subprocess.run([str(modal_cli), "volume", "get", "--force", "fh-train", f"runs/{args.run}/{p}", str(out) + "/"],
                       check=False, stdout=subprocess.DEVNULL)
    log(f"fetched to {out}: {sorted(x.name for x in out.iterdir())}")


def cmd_train_down(args) -> None:
    """Stop flag -> SIGTERM -> sft_lora.py checkpoints at the next optimizer step and exits; then cancel."""
    st = load_state()
    tr = st.get("train", {}).get(args.run)
    d = remote_dict()
    d[f"trainstop:{args.run}"] = True
    log(f"train {args.run}: stop requested; waiting up to {args.wait_min} min for the checkpoint")
    deadline = time.time() + args.wait_min * 60
    while time.time() < deadline:
        info = d.get(f"train:{args.run}") or {}
        if info.get("phase") in ("stopped", "done") or (info.get("phase") or "").startswith("exit"):
            break
        time.sleep(15)
    if tr:
        with contextlib.suppress(Exception):
            modal.FunctionCall.from_id(tr["call_id"]).cancel()
    log(f"train {args.run}: {d.get(f'train:{args.run}')}")


def cmd_stop_all(args) -> None:
    st = load_state()
    for t in list(st.get("endpoints", {})):
        stop_endpoint(t)
    for r in list(st.get("train", {})):
        with contextlib.suppress(Exception):
            remote_dict()[f"trainstop:{r}"] = True
            modal.FunctionCall.from_id(st["train"][r]["call_id"]).cancel()
    modal_cli = Path(sys.executable).parent / "modal"
    for a in (APP, f"{APP}-train"):
        subprocess.run([str(modal_cli), "app", "stop", a], check=False)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="infra/modal/fh", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("secret").set_defaults(fn=cmd_secret)
    s = sub.add_parser("deploy")
    s.add_argument("--train", action="store_true", help="also fh-qwen-train")
    s.set_defaults(fn=cmd_deploy)
    s = sub.add_parser("weights")
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-deploy", action="store_true")
    s.set_defaults(fn=cmd_weights)

    def serve_opts(s):
        s.add_argument("--gpu", default="H100", help="Modal GPU type (recorded with the run)")
        s.add_argument("--idle-min", type=float, default=30, help="endpoint exits after this long without requests")
        s.add_argument("--max-hours", type=float, default=5, help="endpoint (and Harbor) hard limit")

    s = sub.add_parser("serve-up")
    s.add_argument("--tag", default="manual")
    s.add_argument("--lora")
    serve_opts(s)
    s.set_defaults(fn=cmd_serve_up)
    s = sub.add_parser("serve-down")
    s.add_argument("--tag", default="manual")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_serve_down)
    s = sub.add_parser("test")
    s.add_argument("--tag", default="manual")
    s.set_defaults(fn=cmd_test)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("eval")
    s.add_argument("--tasks", required=True)
    s.add_argument("--lora")
    s.add_argument("--runs", type=int, default=3)
    s.add_argument("--only", help="comma-separated run numbers, e.g. 2,3")
    s.add_argument("-n", type=int, default=15)
    s.add_argument("--split")
    s.add_argument("--label")
    s.add_argument("--sequential", action="store_true")
    serve_opts(s)
    s.set_defaults(fn=cmd_eval)
    s = sub.add_parser("train-up")
    s.add_argument("--run", required=True)
    s.add_argument("--data", required=True)
    s.add_argument("--epochs", type=float, default=1)
    s.add_argument("--lr", type=float, default=1e-4)
    s.add_argument("--grad-accum", type=int, default=8)
    s.add_argument("--lora-r", type=int, default=16)
    s.add_argument("--save-every", type=int, default=10)
    s.add_argument("--trainer-arg", action="append")
    s.add_argument("--gpu", default="H100")
    s.set_defaults(fn=cmd_train_up)
    s = sub.add_parser("train-status")
    s.add_argument("--run")
    s.set_defaults(fn=cmd_train_status)
    s = sub.add_parser("train-fetch")
    s.add_argument("--run", required=True)
    s.set_defaults(fn=cmd_train_fetch)
    s = sub.add_parser("train-down")
    s.add_argument("--run", required=True)
    s.add_argument("--wait-min", type=float, default=10)
    s.set_defaults(fn=cmd_train_down)
    sub.add_parser("stop-all").set_defaults(fn=cmd_stop_all)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
