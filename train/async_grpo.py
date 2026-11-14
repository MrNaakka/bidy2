"""Async GRPO trainer: consumes finished rollout groups, PPO-clipped update, publishes policy versions.

    torchrun --standalone --nproc_per_node=K train/async_grpo.py --model M --init-adapter A0 --root R \\
        [--min-groups 8] [--max-groups 16] [--max-stale 2] [--lr 2e-5] [--clip 0.2] [--deadline ISO]

Layout under R (shared with infra/verda/grpo_async/scheduler.py):
  queue/<group>/        one finished item (G rollouts) converted by train/rollouts_to_pg.py --behavior:
                        sft.jsonl (rows with advantage, behavior log-probs, policy_version), images/,
                        pg_stats.json; READY marks it complete. Groups without signal are not queued.
  adapters/v<N>/        policy version N (v0 = --init-adapter, copied); LATEST holds the newest N.
  consumed/, stale/     groups moved here after use / when too old (version < N - max_stale).
  metrics.jsonl         one line per update.

Each update (one optimizer step, i.e. one pass over the rows of the groups taken):
  * groups: the oldest ready groups whose policy version is >= current - max_stale (others -> stale/),
    at least --min-groups (waits otherwise), at most --max-groups;
  * per assistant token t of row i: ratio r = exp(logp_theta(t) - logp_behaviour(t)), loss
    -min(r A_i, clip(r, 1-eps, 1+eps) A_i), summed over all tokens and divided by the step's token count
    (the SFT normalization, as in sft_lora.py --pg). The behaviour log-probs are the ones vLLM returned
    while sampling (the relay logs them). They are matched to the trainer's tokenization of each assistant
    turn by token id (difflib alignment of the generated ids and the template's ids); a token without a
    match (re-rendering differences) uses its own detached log-prob (ratio 1, plain policy gradient);
    the matched fraction is logged;
  * DDP: rows dealt round-robin to ranks, token count all-reduced, LoRA grads summed (sft_lora.py's
    scheme), grad clip, AdamW (state kept across updates), then rank 0 saves adapters/v<N+1>.
Metrics: reward per kind, groups used, staleness histogram, clip fraction, approx KL (k3, behaviour ->
current), ratio mean, grad norm, matched fraction, tokens, seconds, samples/min.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fhdata  # noqa: E402
import sft_lora as S  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--init-adapter", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--min-groups", type=int, default=8)
    ap.add_argument("--max-groups", type=int, default=16)
    ap.add_argument("--max-stale", type=int, default=2, help="use groups sampled by versions >= current - this")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--loss-chunk", type=int, default=1024)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--max-updates", type=int, default=None)
    ap.add_argument("--deadline", default=None, help="ISO time (UTC): no update starts after it")
    ap.add_argument("--poll", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


# ---------------------------------------------------------------------- per-token log-probs
def token_logprobs(core, batch: dict, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """log pi(label) at every trained position (in order) and the positions' label ids."""
    labels = batch.pop("labels")
    out = core.model(**batch, use_cache=False)
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
    tgt = labels[:, 1:]
    pos = (tgt != fhdata.IGNORE).nonzero(as_tuple=False)
    h = hidden[:, :-1][pos[:, 0], pos[:, 1]]
    y = tgt[pos[:, 0], pos[:, 1]]
    head = core.lm_head

    def lp(hc, yc):
        return -F.cross_entropy(head(hc).float(), yc, reduction="none")

    parts = []
    for s in range(0, y.numel(), chunk):
        hc, yc = h[s:s + chunk], y[s:s + chunk]
        parts.append(checkpoint(lp, hc, yc, use_reentrant=False) if torch.is_grad_enabled() else lp(hc, yc))
    return (torch.cat(parts) if parts else h.new_zeros(0, dtype=torch.float32)), y


def behaviour_per_token(enc: dict, row: dict, processor) -> tuple[list[float | None], int]:
    """The behaviour log-prob of every trained token (None where no generated token matches), #matched."""
    ids = enc["input_ids"][0].tolist()
    spans = fhdata.assistant_spans(ids, processor.tokenizer)
    beh = row.get("behavior") or []
    out: list[float | None] = []
    matched = 0
    for j, (s, e) in enumerate(spans):
        tmpl = ids[s:e]
        b = beh[j] if j < len(beh) else None
        vals: list[float | None] = [None] * len(tmpl)
        if b:
            sm = difflib.SequenceMatcher(None, tmpl, b["ids"], autojunk=False)
            for blk in sm.get_matching_blocks():
                for k in range(blk.size):
                    vals[blk.a + k] = b["lp"][blk.b + k]
                    matched += 1
        out += vals
    return out, matched


# ---------------------------------------------------------------------- queue
def version_of(d: Path) -> int:
    return int(json.loads((d / "group.json").read_text())["policy_version"])


def ready_groups(root: Path) -> list[Path]:
    q = root / "queue"
    if not q.is_dir():
        return []
    return sorted((d for d in q.iterdir() if (d / "READY").exists()), key=lambda d: (d / "READY").stat().st_mtime)


def publish(model, root: Path, n: int) -> Path:
    d = root / "adapters" / f"v{n}"
    tmp = root / "adapters" / f".v{n}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    t = root / "adapters" / ".LATEST.tmp"
    t.write_text(f"{n}\n")
    os.replace(t, root / "adapters" / "LATEST")
    return d


def main(argv=None) -> int:
    a = parse_args(argv)
    sa = S.parse_args(["--model", a.model, "--data", str(a.root), "--out", str(a.root), "--lora-r", str(a.lora_r),
                       "--lora-alpha", str(a.lora_alpha), "--lora-dropout", "0", "--dtype", a.dtype,
                       "--device", a.device, "--attn", a.attn])
    S.init_dist(sa)
    rank, world = S.DIST["rank"], S.DIST["world"]
    a.device = sa.device
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    deadline = datetime.fromisoformat(a.deadline.replace("Z", "+00:00")).timestamp() if a.deadline else None
    model, core, processor = S.load_model(sa)
    S.init_from_adapter(model, a.init_adapter, sa)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    model.train()
    for sub in ("queue", "consumed", "stale", "adapters"):
        (a.root / sub).mkdir(parents=True, exist_ok=True)
    latest = a.root / "adapters" / "LATEST"
    if rank == 0 and not latest.exists():
        shutil.copytree(a.init_adapter, a.root / "adapters" / "v0", dirs_exist_ok=True)
        latest.write_text("0\n")
    S.barrier()
    version = int(latest.read_text())
    if version != 0:
        S.init_from_adapter(model, a.root / "adapters" / f"v{version}", sa)   # restart: continue from the newest
    tools = fhdata.tools()
    updates = 0
    t_prev = time.time()
    while True:
        # ---- rank 0 picks the groups, everyone gets the list
        plan = None
        if rank == 0:
            if (a.max_updates is not None and updates >= a.max_updates) or (deadline and time.time() > deadline) \
                    or (a.root / "STOP").exists():
                plan = {"stop": True}
            else:
                fresh, stale = [], []
                for d in ready_groups(a.root):
                    (fresh if version_of(d) >= version - a.max_stale else stale).append(d)
                for d in stale:
                    os.replace(d, a.root / "stale" / d.name)
                if len(fresh) >= a.min_groups:
                    plan = {"groups": [str(d) for d in fresh[:a.max_groups]]}
        if world > 1:
            box = [plan]
            torch.distributed.broadcast_object_list(box, src=0)
            plan = box[0]
        if plan is None:
            time.sleep(a.poll)
            continue
        if plan.get("stop"):
            break
        t0 = time.time()
        groups = [Path(g) for g in plan["groups"]]
        rows = []
        for g in groups:
            for r in fhdata.read_jsonl(g / "sft.jsonl"):
                rows.append((g, r))
        mine = rows[rank::world]
        encs = []
        for g, r in mine:
            enc = fhdata.encode(r, g, processor, tools)
            if enc["n_tokens"] > a.max_len:
                continue
            encs.append((enc, r))
        tok_total = int(S.all_sum([sum(e["n_train_tokens"] for e, _ in encs)], a.device)[0]) or 1
        opt.zero_grad(set_to_none=True)
        st = collections.Counter()
        for enc, r in encs:
            beh, matched = behaviour_per_token(enc, r, processor)
            batch = S.to_device(enc, a.device, dtype)
            lp, _ = token_logprobs(core, batch, a.loss_chunk)
            n = lp.numel()
            assert n == len(beh), (n, len(beh))
            have = torch.tensor([b is not None for b in beh], device=lp.device)
            old = torch.tensor([b if b is not None else 0.0 for b in beh], device=lp.device, dtype=torch.float32)
            old = torch.where(have, old, lp.detach())
            A = float(r["advantage"])
            ratio = torch.exp(lp - old)
            clipped = torch.clamp(ratio, 1 - a.clip, 1 + a.clip)
            loss_tok = -torch.minimum(ratio * A, clipped * A)
            (loss_tok.sum() / tok_total).backward()
            with torch.no_grad():
                d = (old - lp.detach())[have]          # log(pi_b / pi_theta) on matched tokens
                st["tokens"] += n
                st["matched"] += int(have.sum())
                st["loss"] += float(loss_tok.detach().sum())
                st["clipped"] += int(((ratio - clipped).abs() > 1e-6).sum())
                st["ratio_sum"] += float(ratio.detach()[have].sum())
                st["k3_sum"] += float((torch.exp(d) - 1 - d).sum())
                st["rows"] += 1
                st["rows_complete"] += int(all(b is None or b.get("complete") for b in (r.get("behavior") or [])))
        keys = ["tokens", "matched", "loss", "clipped", "ratio_sum", "k3_sum", "rows", "rows_complete"]
        tot = dict(zip(keys, S.all_sum([float(st[k]) for k in keys], a.device)))
        if world > 1:
            S.all_reduce_grads(params)
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        opt.step()
        version += 1
        updates += 1
        if rank == 0:
            publish(model, a.root, version)
            by_kind, stale_h, n_samples = collections.defaultdict(list), collections.Counter(), 0
            for g in groups:
                gj = json.loads((g / "group.json").read_text())
                by_kind[gj["kind"]] += gj["rewards"]
                stale_h[version - 1 - gj["policy_version"]] += 1
                n_samples += len(gj["rewards"])
            now = time.time()
            rec = {"version": version, "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "groups": len(groups), "rows": int(tot["rows"]), "tokens": int(tot["tokens"]),
                   "reward_by_kind": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_kind.items())},
                   "staleness": dict(sorted(stale_h.items())),
                   "loss": round(tot["loss"] / max(tot["tokens"], 1), 6),
                   "clip_frac": round(tot["clipped"] / max(tot["tokens"], 1), 5),
                   "matched_frac": round(tot["matched"] / max(tot["tokens"], 1), 5),
                   "rows_complete_frac": round(tot["rows_complete"] / max(tot["rows"], 1), 4),
                   "ratio_mean": round(tot["ratio_sum"] / max(tot["matched"], 1), 5),
                   "kl_k3": round(tot["k3_sum"] / max(tot["matched"], 1), 6),
                   "grad_norm": round(float(gn), 5), "seconds": round(now - t0, 1),
                   "samples_per_min_since_last": round(n_samples / max((now - t_prev) / 60, 1e-6), 2)}
            t_prev = now
            with open(a.root / "metrics.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            S.log(json.dumps(rec))
            for g in groups:
                os.replace(g, a.root / "consumed" / g.name)
        S.barrier()
    S.barrier()
    return 0


if __name__ == "__main__":
    rc = main()
    if S.DIST["world"] > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    sys.exit(rc)
