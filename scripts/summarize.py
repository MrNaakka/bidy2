"""Summarize FrameHunt Harbor jobs into results/summary.md and results/summary.json.

    uv run python scripts/summarize.py [--runs runs] [--out results] [JOB ...]

Reads each job dir (default: every job under runs/, in a fixed order: reference,
baselines, then agent runs). Per trial it uses Harbor's result.json (reward, exception,
agent cost/tokens) and the verifier evidence that test.sh saves in verifier/:
result.json (submitted, answer, correct, pixels_used, n_calls) and calls.jsonl (every
tool call). API cost comes from Harbor's agent_result.cost_usd, or, if that is empty,
from `cost` fields of OpenCode step_finish events in the agent logs.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("logstream", "dialogs", "toasts", "leaderboard", "dashboard")
ORDER = ["reference", "nop", "stop-early"]


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _jsonl(p: Path) -> list[dict]:
    out = []
    try:
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def _ts(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _log_cost(agent_dir: Path) -> tuple[float, int, int]:
    """Sum cost/tokens from OpenCode-style step_finish events in any agent log file."""
    cost, tin, tout = 0.0, 0, 0
    if not agent_dir.is_dir():
        return cost, tin, tout
    for p in agent_dir.rglob("*"):
        if not p.is_file() or p.stat().st_size > 50_000_000 or p.suffix not in ("", ".txt", ".json", ".jsonl", ".log"):
            continue
        for ev in _jsonl(p):
            if isinstance(ev, dict) and ev.get("type") == "step_finish":
                part = ev.get("part") or {}
                cost += float(part.get("cost") or 0)
                tok = part.get("tokens") or {}
                tin += int(tok.get("input") or 0)
                tout += int(tok.get("output") or 0)
    return cost, tin, tout


def trial_row(tdir: Path) -> dict | None:
    res = _load(tdir / "result.json")
    if not isinstance(res, dict) or "task_name" not in res:
        return None
    item = res["task_name"].split("/")[-1]
    vr = (res.get("verifier_result") or {}).get("rewards") or {}
    ev = _load(tdir / "verifier" / "result.json") or {}
    calls = _jsonl(tdir / "verifier" / "calls.jsonl")
    ar = res.get("agent_result") or {}
    cost = ar.get("cost_usd")
    tin, tout = ar.get("n_input_tokens"), ar.get("n_output_tokens")
    if not cost:
        c, i, o = _log_cost(tdir / "agent")
        cost = c or None
        tin, tout = tin or i or None, tout or o or None
    exc = res.get("exception_info")
    t0, t1 = _ts(res.get("started_at")), _ts(res.get("finished_at"))
    return {
        "item": item,
        "kind": next((k for k in KINDS if item.startswith(k + "-")), "other"),
        "trial": res.get("trial_name", tdir.name),
        "reward": float(vr.get("reward", 0.0) or 0.0),
        "submitted": bool(ev.get("submitted", False)),
        "answer": ev.get("answer"),
        "correct": bool(ev.get("correct", False)),
        "pixels_used": int(ev.get("pixels_used", 0) or 0),
        "n_calls": int(ev.get("n_calls", len(calls)) or 0),
        "tools": dict(Counter(c.get("tool") for c in calls)),
        "refused_calls": sum(1 for c in calls if not c.get("ok", True)),
        "evidence": bool(ev),
        "exception": (exc or {}).get("exception_type") if exc else None,
        "cost_usd": cost,
        "input_tokens": tin,
        "output_tokens": tout,
        "seconds": round((t1 - t0).total_seconds(), 1) if t0 and t1 else None,
    }


def job_summary(jdir: Path) -> dict:
    cfg = _load(jdir / "config.json") or {}
    agents = cfg.get("agents")
    if not agents:  # Harbor leaves the default (oracle) out of the job config; ask a trial
        trial = next((_load(t / "result.json") for t in jdir.iterdir() if (t / "result.json").exists()), None)
        agents = [((trial or {}).get("config") or {}).get("agent") or {"name": "oracle"}]
    agent = agents[0].get("name") or agents[0].get("import_path") or "?"
    model = agents[0].get("model_name")
    rows = sorted(filter(None, (trial_row(t) for t in jdir.iterdir() if t.is_dir())), key=lambda r: r["item"])
    jres = _load(jdir / "result.json") or {}
    t0, t1 = _ts(jres.get("started_at")), _ts(jres.get("finished_at"))

    def mean(xs):
        xs = list(xs)
        return round(statistics.fmean(xs), 4) if xs else None

    costs = [r["cost_usd"] for r in rows if r["cost_usd"]]
    return {
        "job": jdir.name, "agent": agent, "model": model, "n_trials": len(rows),
        "mean_reward": mean(r["reward"] for r in rows),
        "n_correct": sum(r["reward"] >= 1.0 for r in rows),
        "n_submitted": sum(r["submitted"] for r in rows),
        "n_exceptions": sum(bool(r["exception"]) for r in rows),
        "n_missing_evidence": sum(not r["evidence"] for r in rows),
        "mean_pixels_used": mean(r["pixels_used"] for r in rows),
        "max_pixels_used": max((r["pixels_used"] for r in rows), default=None),
        "mean_calls": mean(r["n_calls"] for r in rows),
        "per_kind": {k: {"n": sum(r["kind"] == k for r in rows),
                         "mean_reward": mean(r["reward"] for r in rows if r["kind"] == k),
                         "mean_pixels_used": mean(r["pixels_used"] for r in rows if r["kind"] == k)}
                     for k in KINDS if any(r["kind"] == k for r in rows)},
        "cost_usd": round(sum(costs), 4) if costs else None,
        "wall_seconds": round((t1 - t0).total_seconds()) if t0 and t1 else None,
        "trials": rows,
    }


def _order(name: str) -> tuple:
    for i, key in enumerate(ORDER):
        if name.endswith(key):
            return (i, name)
    return (len(ORDER), name)


def fmt(x, nd=3):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def to_markdown(jobs: list[dict]) -> str:
    L = ["# FrameHunt results", "",
         "Reward is 1 if the first submitted answer is exactly right, else 0. Pixels = image pixels the",
         "run was charged (budget per item: see each task.toml, default 120000). Generated by `scripts/summarize.py`.", "",
         "## Per run", "",
         "| job | agent | model | trials | mean reward | correct | submitted | exceptions | mean px | max px | mean calls | API cost (USD) | wall time (s) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for j in jobs:
        L.append(f"| {j['job']} | {j['agent']} | {j['model'] or '-'} | {j['n_trials']} | {fmt(j['mean_reward'])} | "
                 f"{j['n_correct']} | {j['n_submitted']} | {j['n_exceptions']} | {fmt(j['mean_pixels_used'], 0)} | "
                 f"{fmt(j['max_pixels_used'])} | {fmt(j['mean_calls'], 1)} | {fmt(j['cost_usd'], 2)} | "
                 f"{fmt(j['wall_seconds'])} |")
    total = [j["cost_usd"] for j in jobs if j["cost_usd"]]
    if total:
        L += ["", f"Total API cost: ${sum(total):.2f}"]
    present = [k for k in KINDS if any(k in j["per_kind"] for j in jobs)]
    L += ["", "## Per kind (mean reward / mean pixels)", "",
          "| job | " + " | ".join(present) + " |", "| --- |" + " --- |" * len(present)]
    for j in jobs:
        cells = []
        for k in present:
            pk = j["per_kind"].get(k)
            cells.append(f"{fmt(pk['mean_reward'], 2)} / {fmt(pk['mean_pixels_used'], 0)}" if pk else "-")
        L.append(f"| {j['job']} | " + " | ".join(cells) + " |")
    items = sorted({r["item"] for j in jobs for r in j["trials"]},
                   key=lambda i: (next((n for n, k in enumerate(KINDS) if i.startswith(k)), 9), i))
    L += ["", "## Per item (reward, pixels used, calls; `-` = not submitted)", "",
          "| item | " + " | ".join(j["job"] for j in jobs) + " |", "| --- |" + " --- |" * len(jobs)]
    for it in items:
        cells = []
        for j in jobs:
            r = next((r for r in j["trials"] if r["item"] == it), None)
            if r is None:
                cells.append("")
            else:
                s = f"{r['reward']:.0f}, {r['pixels_used']} px, {r['n_calls']} calls"
                cells.append(s if r["submitted"] else f"{s}, -")
        L.append(f"| {it} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", help="job names under --runs (default: all)")
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    a = ap.parse_args(argv)
    names = a.jobs or [p.name for p in a.runs.iterdir() if (p / "config.json").exists()] if a.runs.is_dir() else []
    if not names:
        print(f"no jobs found under {a.runs}", file=sys.stderr)
        return 1
    jobs = [job_summary(a.runs / n) for n in sorted(names, key=_order)]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps({"jobs": jobs}, indent=1))
    md = to_markdown(jobs)
    (a.out / "summary.md").write_text(md)
    print(md.split("## Per kind")[0].rstrip())
    print(f"\nwrote {a.out / 'summary.md'} and {a.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
