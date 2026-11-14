"""GRPO stage: every on-policy rollout -> a policy-gradient row with a group-normalized advantage.

    python train/rollouts_to_pg.py --out DIR RUN [RUN ...] [--tasks TASKS] [--processor DIR] [--group-size 8]

Inputs are Harbor run dirs (one trial dir per rollout, `-k G` attempts per item) whose request logs from
infra/verda/log_relay.py are in <run>/requests/ (see train/rollouts_to_sft.py, whose checks this reuses).

Reward of a trial: verifier/reward.txt. A trial without one counts as reward 0 if the agent itself failed
(no exception, or AgentTimeoutError: the policy ran out of time). A trial with any other exception is a
harness failure (environment/setup/verifier errors are not the policy's doing) and is left out of its
group, even if it wrote a reward.

Groups: all counted trials of one item. Advantage A = (r - mean_group) / (std_group + eps) with the
population std over the whole group (also the trials whose conversation could not be converted, so a
trial's advantage does not depend on which other rows happen to convert). Groups with zero std carry no
signal and are dropped; so are groups with fewer than --min-group counted trials.

Rows: like rollouts_to_sft.py's (the exact logged conversation, the same render checks), for EVERY counted
trial of a group with signal, not only successes, and without requiring a framehunt_submit call (a run
that never submits is a legitimate failure). Extra fields: reward, advantage, group_mean, group_std,
group_n. A trial whose log cannot be converted is skipped (rejected.jsonl) but still counts in its
group's statistics. Output: <out>/sft.jsonl (for train/sft_lora.py --pg), images/, tools.json,
pg_stats.json (mean reward per kind = the training-reward curve, groups with signal, rejections).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fhdata  # noqa: E402
import rollouts_to_sft as r2s  # noqa: E402

AGENT_FAULTS = {None, "AgentTimeoutError"}   # exception types that still count as the policy's reward 0
EPS = 1e-6


def trial_reward(tdir: Path) -> tuple[float | None, str | None]:
    """(reward or None if the trial does not count, exception type)."""
    exc = None
    if (tdir / "result.json").is_file():
        res = json.loads((tdir / "result.json").read_text())
        exc = (res.get("exception_info") or {}).get("exception_type")
    if exc not in AGENT_FAULTS:          # harness failure: not counted, even if a reward was written
        return None, exc
    rf = tdir / "verifier" / "reward.txt"
    if rf.is_file():
        try:
            return float(rf.read_text().strip()), exc
        except ValueError:
            return 0.0, exc
    if exc in AGENT_FAULTS:
        return 0.0, exc
    return None, exc


def advantages(rewards: list[float]) -> tuple[float, float, list[float]]:
    n = len(rewards)
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return mean, std, [(r - mean) / (std + EPS) for r in rewards]


def synthetic_reward(trial: str) -> float:
    """Pipeline tests with a random model (real rewards ~0): a deterministic coin per trial name."""
    import hashlib
    return float(hashlib.sha256(trial.encode()).digest()[0] & 1)


def behavior(records: list[dict], msgs: list[dict]) -> tuple[list, list]:
    """Behaviour log-probs per assistant message of msgs (async GRPO relay, --policy-file), and the policy
    versions that served the trajectory.

    The response of a request with n messages is assistant message n (OpenCode sends it back next); a
    record counts only if its reassembled response equals that message (r2s.same_turn), so a retried or
    aborted request never lends its log-probs to the turn that was kept. Each entry: {"ids": token ids,
    "lp": log-probs, "complete": every generated token has one (count == usage.completion_tokens)} or None.
    """
    asst = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    by_idx: dict[int, dict] = {}
    versions = set()
    for r in records:
        if "_req" not in r or not r2s.usable(r):
            continue
        if r.get("policy_version") is not None:
            versions.add(r["policy_version"])
        idx = len(r["_req"]["messages"])
        resp = r["response"]
        lps = resp.get("logprobs")
        if idx >= len(msgs) or idx in by_idx or not lps or not r2s.same_turn(resp, msgs[idx]):
            continue
        ids = [x[0] for x in lps]
        if not all(isinstance(t, int) for t in ids):
            continue
        by_idx[idx] = {"ids": ids, "lp": [float(x[1]) for x in lps], "version": r.get("policy_version"),
                       "complete": resp.get("completion_tokens") == len(ids)}
    return [by_idx.get(i) for i in asst], sorted(versions)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="Harbor run dirs with requests/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, help="the TRAIN tasks dir the runs used: also check each instruction")
    ap.add_argument("--processor", help="HF processor dir: also compare chat-template renderings")
    ap.add_argument("--min-group", type=int, default=4, help="drop groups with fewer counted trials")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    ap.add_argument("--behavior", action="store_true",
                    help="async GRPO: attach behaviour log-probs and policy versions from the relay log")
    ap.add_argument("--synthetic-reward", action="store_true",
                    help="pipeline tests only: reward = a deterministic coin per trial (not the verifier's)")
    ap.add_argument("--no-call-check", action="store_true",
                    help="pipeline tests only: skip the MCP call-log check (a random model makes no tool calls)")
    a = ap.parse_args(argv)
    if len({p.resolve() for p in a.runs}) != len(a.runs):
        ap.error("a run dir is given twice (its rewards would be counted twice)")
    out_final = a.out.resolve()
    for p in [*a.runs, *([a.tasks] if a.tasks else [])]:
        q = p.resolve()
        if q == out_final or out_final in q.parents or q in out_final.parents:
            ap.error(f"--out {a.out} overlaps the input {p}")
    if a.out.exists() and not a.force:
        ap.error(f"{a.out} exists (--force replaces it)")
    processor = None
    if a.processor:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(a.processor)

    trials = []
    for run in a.runs:
        if not run.is_dir():
            ap.error(f"{run}: not a directory")
        meta = {}
        if (run / "fh-eval-meta.json").is_file():
            meta = json.loads((run / "fh-eval-meta.json").read_text())
        for td in r2s.find_trials(run):
            t = r2s.load_trial(td)
            t["reward"], t["exception"] = trial_reward(td)
            if a.synthetic_reward and t["reward"] is not None:
                t["reward"] = synthetic_reward(t["trial"])
            trials.append((run, meta, t))
    bad = sorted({f"{run.name}/{t['trial']} ({t['item_id']})" for run, _, t in trials
                  if r2s.train_kind(t["item_id"]) is None})
    if bad:
        raise SystemExit(f"refusing: {len(bad)} trial(s) are not on TRAIN items (EVAL/DEV/unknown): {bad[:10]}")
    keys = collections.Counter((run, r2s.trial_key(r2s.harbor_project(t["trial"]))) for run, _, t in trials)

    groups: dict[str, list] = collections.defaultdict(list)
    excluded = collections.Counter()
    for run, meta, t in trials:
        if t["reward"] is None:
            excluded[t["exception"]] += 1
            continue
        groups[t["item_id"]].append((run, meta, t))

    final_out = a.out
    final_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{final_out.name}.", suffix=".tmp", dir=final_out.parent))
    (tmp / "images").mkdir()
    shutil.copy(fhdata.TOOLS_FILE, tmp / "tools.json")
    rows, rejected = [], []
    kinds: dict[str, dict] = {}
    stats = {"runs": [str(r) for r in a.runs], "trials": len(trials), "excluded_harness_errors": dict(excluded),
             "groups": len(groups), "groups_with_signal": 0, "groups_small": 0, "rows": 0, "rows_pos": 0,
             "rows_neg": 0, "rejected": collections.Counter(), "rejected_by_reward": collections.Counter()}
    for item_id in sorted(groups):
        g = sorted(groups[item_id], key=lambda x: (x[0].name, x[2]["trial"]))
        kind = r2s.train_kind(item_id)
        rewards = [t["reward"] for _, _, t in g]
        ks = kinds.setdefault(kind, {"trials": 0, "reward_sum": 0.0, "items": 0, "items_with_signal": 0,
                                     "rows": 0, "solved_any": 0, "solved_all": 0})
        ks["trials"] += len(g)
        ks["reward_sum"] += sum(rewards)
        ks["items"] += 1
        ks["solved_any"] += any(r > 0 for r in rewards)
        ks["solved_all"] += all(r >= 1 for r in rewards)
        if len(g) < a.min_group:
            stats["groups_small"] += 1
            continue
        mean, std, adv = advantages(rewards)
        if std < 1e-9:
            continue
        stats["groups_with_signal"] += 1
        ks["items_with_signal"] += 1
        for (run, meta, t), A in zip(g, adv):
            try:
                if keys[(run, r2s.trial_key(r2s.harbor_project(t["trial"])))] > 1:
                    raise r2s.Reject("two trials map to the same request log (trial names differ only in case)")
                c = r2s.convert_trial(t, run, meta, a.tasks, require_submit=False, check_calls=not a.no_call_check)
                row_id = f"{run.name}__{t['trial']}"
                body, pngs = r2s.make_row(row_id, c["wire"], tmp)
                row = {"id": row_id, "item_id": item_id, "kind": kind, "split": "train", "answer": c["answer"],
                       "correct": t["reward"] >= 1, "reward": t["reward"], "advantage": A,
                       "group_mean": mean, "group_std": std, "group_n": len(g),
                       "pixels_used": c["pixels_used"], "budget": c["budget"], "n_calls": len(c["tools_used"]),
                       "n_images": len(pngs), "tools_used": c["tools_used"], "date": c["date"],
                       "provenance": c["provenance"], "messages": body["messages"]}
                if a.behavior:
                    beh, versions = behavior(c["records"], c["wire"])
                    if not versions:
                        raise r2s.Reject("no policy version in the relay log")
                    row.update(behavior=beh, policy_versions=versions, policy_version=min(versions))
                img_dir = tmp / "images" / row_id
                img_dir.mkdir(parents=True)
                for i, png in enumerate(pngs):
                    (img_dir / f"{i:02d}.png").write_bytes(png)
                try:
                    r2s.assert_renders(row, c["wire"], c["tools"], tmp, processor)
                except (AssertionError, ValueError) as e:
                    shutil.rmtree(img_dir, ignore_errors=True)
                    raise r2s.Reject(f"render check: {str(e)[:200]}")
            except r2s.Reject as e:
                reason = str(e)
                stats["rejected"][re.sub(r"\d+", "N", reason)] += 1
                stats["rejected_by_reward"][str(t["reward"])] += 1
                rejected.append({"run": run.name, "trial": t["trial"], "item_id": item_id, "reward": t["reward"],
                                 "advantage": A, "reason": reason})
                continue
            rows.append(row)
            ks["rows"] += 1
            stats["rows_pos" if A > 0 else "rows_neg"] += 1
    with open(tmp / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(tmp / "rejected.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    for ks in kinds.values():
        ks["mean_reward"] = round(ks.pop("reward_sum") / max(ks["trials"], 1), 4)
    n_counted = sum(len(g) for g in groups.values())
    stats.update(rows=len(rows), counted_trials=n_counted,
                 mean_reward=round(sum(t["reward"] for g in groups.values() for _, _, t in g) / max(n_counted, 1), 4),
                 by_kind=dict(sorted(kinds.items())), rejected=dict(stats["rejected"]),
                 rejected_by_reward=dict(stats["rejected_by_reward"]), rendered_with_processor=bool(processor))
    (tmp / "pg_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    if final_out.exists():
        shutil.rmtree(final_out)
    tmp.rename(final_out)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
