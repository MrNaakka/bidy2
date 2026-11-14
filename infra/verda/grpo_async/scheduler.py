"""Async GRPO rollout scheduler (runs on the Verda box; train-env python, stdlib + the repo).

    python scheduler.py --root R --pool TASKS --gpus 0 1 2 [--k 8] [--inflight 24] [--deadline ISO] [--synthetic]

Keeps about --inflight Harbor trials per inference GPU in flight: one small Harbor job per TRAIN item
(-k K attempts = one GRPO group, -n K), against that GPU's request-logging relay (log_relay.py
--policy-file R/policy.json on 172.17.0.1:18200+g -> vLLM 127.0.0.1:8200+g), the evals' agent config
(cfg-ft-g0.json, OpenCode 1.19.5) and a shorter agent timeout (--agent-timeout-min). Items are taken from
--pool (TRAIN items only, built with build_tasks.py --split train --offset >= 300) in a fixed order,
kinds interleaved, each used once.

When a job ends, train/rollouts_to_pg.py --behavior converts its trials (exact logged conversations,
group-normalized advantages, behaviour log-probs, policy versions) and, if the group has signal, the
result goes to R/queue/<job>/ with group.json and READY for train/async_grpo.py. Every finished group
(with or without signal) is a line in R/events.jsonl: item, kind, rewards, policy versions, durations.

Policy versions: when the trainer publishes R/adapters/v<N> (LATEST = N), every vLLM loads it at runtime
as fh-v<N> (/v1/load_lora_adapter; no restart); once all have it, R/policy.json switches to it, so new
trials are pinned to v<N> by the relays (running trials finish on the version they started with).
Versions older than N-3 are unloaded. Stops launching at --deadline or when R/STOP exists, waits for
the running jobs (killing any that exceed --job-max-min), then exits.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CODE = Path(__file__).resolve().parents[3]
FH = Path("/root/fh")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def api(port: int, path: str, body: dict | None, key: str, timeout: float = 120) -> tuple[int, str]:
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method="POST" if body is not None else "GET",
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()[:500]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]
    except OSError as e:
        return 0, str(e)


def write_json(p: Path, obj) -> None:
    t = p.with_name(f".{p.name}.tmp")
    t.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(t, p)


def item_order(pool: Path) -> list[str]:
    by_kind: dict[str, list[str]] = {}
    for d in sorted(pool.iterdir()):
        if (d / "task.toml").is_file():
            by_kind.setdefault(d.name.rsplit("-", 1)[0], []).append(d.name)
    out, i = [], 0
    while any(i < len(v) for v in by_kind.values()):
        out += [v[i] for _, v in sorted(by_kind.items()) if i < len(v)]
        i += 1
    return out


class Job:
    def __init__(self, name, item, gpu, proc, version):
        self.name, self.item, self.gpu, self.proc, self.launch_version = name, item, gpu, proc, version
        self.t0 = time.time()
        self.t_done = None
        self.conv = None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--inflight", type=int, default=24, help="target trials in flight per GPU")
    ap.add_argument("--agent-timeout-min", type=float, default=8.0)
    ap.add_argument("--job-max-min", type=float, default=25.0)
    ap.add_argument("--deadline", default=None)
    ap.add_argument("--synthetic", action="store_true", help="pipeline test: synthetic rewards, no call-log check")
    ap.add_argument("--python", default="/root/fhdata/envs/train/bin/python")
    ap.add_argument("--cfg", type=Path, default=FH / "cfg-ft-g0.json")
    a = ap.parse_args(argv)
    R = a.root
    for sub in ("queue", "jobs", "runs", "convert", "nosignal", "adapters"):
        (R / sub).mkdir(parents=True, exist_ok=True)
    key = (FH / "vllm.env").read_text().strip().split("=", 1)[1]
    deadline = datetime.fromisoformat(a.deadline.replace("Z", "+00:00")).timestamp() if a.deadline else None
    tmult = round(a.agent_timeout_min * 60 / 1800.0, 4)    # the tasks' agent timeout is 1800 s
    for g in a.gpus:
        c = json.loads(a.cfg.read_text())
        c["agents"][0]["base_url"] = f"http://172.17.0.1:{18200 + g}/v1"
        write_json(R / f"cfg-g{g}.json", c)
    pol_file = R / "policy.json"
    if not pol_file.exists():
        write_json(pol_file, {"version": 0, "name": "fh-v0"})
    version = json.loads(pol_file.read_text())["version"]
    loaded = {g: {0} | ({version} if version else set()) for g in a.gpus}
    state_f = R / "sched_state.json"
    st = json.loads(state_f.read_text()) if state_f.exists() else {"next": 0, "launched": 0}
    items = item_order(a.pool)
    log(f"{len(items)} pool items, next {st['next']}, version {version}, gpus {a.gpus}, k {a.k}, "
        f"inflight {a.inflight}/gpu, agent timeout x{tmult}")
    running: list[Job] = []
    converting: list[Job] = []
    stopping = False

    def on_term(signum, frame):
        nonlocal stopping
        stopping = True
        log(f"signal {signum}: no new jobs")

    signal.signal(signal.SIGTERM, on_term)
    while True:
        # ---- new policy version from the trainer
        lat = R / "adapters" / "LATEST"
        if lat.exists():
            n = int(lat.read_text())
            if n > version:
                ok = True
                for g in a.gpus:
                    if n in loaded[g]:
                        continue
                    code, msg = api(8200 + g, "/v1/load_lora_adapter",
                                    {"lora_name": f"fh-v{n}", "lora_path": str(R / "adapters" / f"v{n}")}, key)
                    if code == 200 or (code == 400 and "already" in msg):
                        loaded[g].add(n)
                    else:
                        ok = False
                        log(f"gpu {g}: load fh-v{n} failed: {code} {msg}")
                if ok:
                    version = n
                    write_json(pol_file, {"version": n, "name": f"fh-v{n}", "time": now_iso()})
                    log(f"policy -> fh-v{n}")
                    for g in a.gpus:
                        for old in sorted(v for v in loaded[g] if v < n - 3):
                            code, msg = api(8200 + g, "/v1/unload_lora_adapter", {"lora_name": f"fh-v{old}"}, key)
                            loaded[g].discard(old)
        # ---- finished jobs -> conversion
        for j in list(running):
            age = time.time() - j.t0
            if j.proc.poll() is None and age > a.job_max_min * 60:
                log(f"{j.name}: running {age / 60:.0f} min, killing")
                j.proc.send_signal(signal.SIGINT)
                try:
                    j.proc.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    j.proc.kill()
            if j.proc.poll() is None:
                continue
            running.remove(j)
            j.t_done = time.time()
            run = R / "runs" / j.name
            if run.is_dir():
                req = run / "requests"
                if not req.exists():
                    req.symlink_to(R / f"requests-g{j.gpu}")
                cmd = [a.python, str(CODE / "train" / "rollouts_to_pg.py"), str(run), "--out",
                       str(R / "convert" / j.name), "--tasks", str(R / "jobs" / j.name / "tasks"), "--behavior",
                       "--force"] + (["--synthetic-reward", "--no-call-check"] if a.synthetic else [])
                j.conv = subprocess.Popen(cmd, stdout=open(R / "jobs" / j.name / "convert.log", "w"),
                                          stderr=subprocess.STDOUT)
                converting.append(j)
            else:
                log(f"{j.name}: no run dir (harbor rc {j.proc.returncode})")
        for j in list(converting):
            if j.conv.poll() is None:
                continue
            converting.remove(j)
            out = R / "convert" / j.name
            ev = {"job": j.name, "item": j.item, "gpu": j.gpu, "launch_version": j.launch_version,
                  "harbor_rc": j.proc.returncode, "convert_rc": j.conv.returncode,
                  "rollout_s": round(j.t_done - j.t0, 1), "done": now_iso()}
            if j.conv.returncode == 0 and (out / "pg_stats.json").exists():
                s = json.loads((out / "pg_stats.json").read_text())
                rows = [json.loads(l) for l in (out / "sft.jsonl").read_text().splitlines() if l.strip()]
                runs = R / "runs" / j.name
                rewards = []
                for td in sorted(runs.iterdir()):
                    rf = td / "verifier" / "reward.txt"
                    if td.is_dir() and (td / "config.json").exists():
                        rewards.append(None if not rf.exists() else float(rf.read_text().strip() or 0))
                kind = j.item.rsplit("-", 1)[0]
                versions = sorted({v for r in rows for v in r.get("policy_versions", [])})
                ev.update(kind=kind, mean_reward=s["mean_reward"], counted=s["counted_trials"], trials=s["trials"],
                          signal=s["groups_with_signal"], rows=s["rows"], rejected=s["rejected"],
                          versions=versions, verifier_rewards=rewards,
                          group_rewards=[r["reward"] for r in rows])
                if s["groups_with_signal"] and rows:
                    gj = {"item": j.item, "kind": kind, "policy_version": min(r["policy_version"] for r in rows),
                          "rewards": [r["reward"] for r in rows], "group_mean": rows[0]["group_mean"],
                          "group_n": rows[0]["group_n"],
                          "job": j.name, "launch_version": j.launch_version}
                    write_json(out / "group.json", gj)
                    dst = R / "queue" / j.name
                    os.replace(out, dst)
                    (dst / "READY").write_text(now_iso() + "\n")
                else:
                    os.replace(out, R / "nosignal" / j.name)
            with open(R / "events.jsonl", "a") as f:
                f.write(json.dumps(ev) + "\n")
            log(f"{j.name}: {ev.get('mean_reward')} signal={ev.get('signal')} rows={ev.get('rows')} "
                f"rollout {ev['rollout_s']:.0f}s")
        # ---- launch
        if (deadline and time.time() > deadline) or (R / "STOP").exists():
            stopping = True
        if not stopping:
            for g in a.gpus:
                while sum(a.k for j in running if j.gpu == g) + a.k <= a.inflight and st["next"] < len(items):
                    item = items[st["next"]]
                    st["next"] += 1
                    st["launched"] += 1
                    name = f"ag-{item}"
                    jd = R / "jobs" / name
                    if jd.exists():
                        shutil.rmtree(jd)
                    (jd / "tasks").mkdir(parents=True)
                    shutil.copytree(a.pool / item, jd / "tasks" / item)
                    shutil.rmtree(R / "runs" / name, ignore_errors=True)
                    cmd = ["/root/.local/bin/harbor", "run", "-p", str(jd / "tasks"), "--job-name", name,
                           "-o", str(R / "runs"), "-k", str(a.k), "-n", str(a.k), "-r", "2",
                           "--retry-include", "AgentSetupTimeoutError", "-y", "-c", str(R / f"cfg-g{g}.json"),
                           "--env-file", str(FH / "key.env"), "--agent-timeout-multiplier", str(tmult)]
                    env = dict(os.environ, HOME="/root", PATH="/root/.local/bin:/usr/bin:/bin")
                    p = subprocess.Popen(cmd, cwd=str(FH), stdout=open(jd / "harbor.log", "w"),
                                         stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    running.append(Job(name, item, g, p, version))
                    write_json(state_f, st)
                    log(f"launched {name} on gpu {g} (policy v{version}); in flight "
                        f"{sum(a.k for j in running if j.gpu == g)} on gpu {g}")
        if stopping and not running and not converting:
            log("stopped")
            return 0
        time.sleep(5)


if __name__ == "__main__":
    sys.exit(main())
