"""LoRA SFT of Qwen3.8-27B on FrameHunt conversations (OpenCode format), resumable.

    python train/sft_lora.py --model /mnt/fh/models/Qwen3.8-27B --data /mnt/fh/data/sft-v1 \
        --out /mnt/fh/runs/sft-v1 [--epochs 2] [--save-every 25]

Rerunning the same command resumes from the newest complete checkpoint in --out
(spot preemption): adapter, optimizer, LR schedule, RNG and position in the data
order are restored, and loss.jsonl is truncated to the resumed step.

What is trained
  * Tokens: each row is rendered with the model's own chat template exactly as vLLM
    0.33 renders OpenCode's request (train/fhdata.py), tools included. Only the
    assistant tokens the model generates are trained: after
    "<|im_start|>assistant\\n<think>\\n" up to and including "<|im_end|>" (reasoning,
    the </think> close, the <tool_call> block or final text). System prompt, tool
    list, instruction, tool results and images are context only.
  * Weights: LoRA on the language model only (regex on model.language_model.*):
    full-attention q/k/v/o and MLP gate/up/down projections in every layer
    (--lora-linear-attn adds the Gated DeltaNet in_proj_qkv/in_proj_z/out_proj).
    The vision tower, the patch merger, embeddings and lm_head stay frozen.

Memory (H100 80 GB, bf16 base): weights 54 GB (27B x 2 B); LoRA r=16 ~80M params +
AdamW fp32 states ~1 GB; gradient checkpointing keeps one 5120-wide bf16 hidden state
per layer per token (64 x 10 KB = 0.66 MB/token, ~10 GB at 15k tokens) plus one
layer's recompute (~2-4 GB); the loss computes logits only at trained positions, in
checkpointed chunks (~1 GB instead of 15k x 248k x 4 B = 15 GB). Total ~70 GB, so
bf16 LoRA fits one 80 GB GPU at batch 1 x 15k tokens; --quant 4bit (QLoRA, needs
bitsandbytes) is the fallback for 40-48 GB GPUs or longer sequences.

Multi-GPU data parallel (one box, a full model replica per GPU):

    torchrun --standalone --nproc_per_node=4 train/sft_lora.py <same args>

gives the same optimization as one GPU with the same args. Every optimizer step still
consumes the same grad_accum conversations of the same per-epoch shuffled order (global
micro indices start..start+grad_accum-1); rank r takes those with (i - start) % K == r,
so grad_accum need not be a multiple of K. The trained-token total of the whole step is
all-reduced before backward, each rank weights its micro-batch losses by k / global
total, and the LoRA gradients are summed across ranks (manual all_reduce, no DDP
wrapper), which is exactly the single-GPU gradient. Clipping, AdamW and the schedule then
run identically on every rank. Rank 0 alone writes logs, checkpoints and final/; eval is
sharded and reduced. Checkpoints hold the global step and micro position only, so a
checkpoint resumes on any K (including one written by a single-GPU run, and vice
versa). --resume-from <ckpt dir> starts a fresh --out from another run's checkpoint.
LoRA dropout masks differ from the single-GPU run (per-rank RNG, reseeded each step).

Continued training (expert iteration): --init-adapter DIR starts from a trained PEFT adapter (e.g. the
SFT run's final/) instead of a fresh LoRA. Its rank and alpha must match this run's and it must adapt
the same modules (every LoRA tensor is loaded, none left over). Optimizer and LR schedule start fresh.
A resumed run loads its own checkpoint instead.

Policy gradient (GRPO stage, train/rollouts_to_pg.py): --pg trains on rollout rows that carry an
`advantage` A (group-normalized reward, any sign). Each row's trained tokens get weight A, so the step
loss is sum_rows A * sum_t CE_t / (trained tokens of the step), the SFT normalization with a signed row
weight: its gradient is -sum A * grad log pi(token) / tokens, the on-policy REINFORCE/GRPO gradient
(DAPO-style token-level averaging). A > 0 raises the sampled tokens' log-probs, A < 0 lowers them, A = 0
contributes nothing (the row is not run and its tokens are not in the normalization). One pass over the batch (--epochs 1) with the rows
sampled from the --init-adapter policy keeps it on-policy, so no importance ratio or clipping is used.
--pg implies a constant LR after warmup and no held-out eval (--eval-frac 0). The logged `loss` is the
advantage-weighted loss; `nll_pos`/`nll_neg` are the token-mean NLL of the A>0 / A<0 rows (they
separate as the policy moves), `adv_abs` the token-weighted mean |A|.

Logs: <out>/loss.jsonl (one line per optimizer step: step, epoch, loss, lr, tokens,
grad_norm, seconds), <out>/eval.jsonl (held-out loss at each checkpoint),
<out>/args.json. Final adapter: <out>/final (PEFT format, loadable by vLLM --lora-modules).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fhdata

STOP = {"flag": False}
DIST = {"rank": 0, "world": 1}


def init_dist(a) -> None:
    """torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; without them this is a no-op (single process)."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return
    import torch.distributed as dist
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    if a.device.startswith("cuda"):
        torch.cuda.set_device(local)
        a.device = f"cuda:{local}"
        backend = "nccl"
    else:
        backend = "gloo"
    # Long timeout: rank 0 writes checkpoints while the others wait at the next collective.
    dist.init_process_group(backend, timeout=timedelta(minutes=60))
    DIST.update(rank=rank, world=world)


def is_main() -> bool:
    return DIST["rank"] == 0


def barrier() -> None:
    if DIST["world"] > 1:
        torch.distributed.barrier()


def all_sum(vals: list[float], device) -> list[float]:
    """Sum a few scalars over ranks (float64)."""
    if DIST["world"] == 1:
        return list(vals)
    t = torch.tensor(vals, dtype=torch.float64, device=device)
    torch.distributed.all_reduce(t)
    return t.tolist()


def all_reduce_grads(params: list[torch.Tensor]) -> None:
    """Sum trainable grads over ranks (one fp32 flat buffer). A rank that trained no rows
    this step contributes zeros."""
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])
    torch.distributed.all_reduce(flat)
    off = 0
    for p in params:
        n = p.numel()
        g = flat[off:off + n].view_as(p).to(p.dtype)
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
        off += n


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model dir (Qwen3.8-27B BF16) or the tiny test model")
    ap.add_argument("--data", type=Path, required=True, help="dir with sft.jsonl, tools.json, images/")
    ap.add_argument("--out", type=Path, required=True, help="checkpoints, logs and the final adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--grad-accum", type=int, default=8, help="conversations per optimizer step (batch 1 each)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-linear-attn", action="store_true",
                    help="also adapt the Gated DeltaNet projections (check vLLM loads it before relying on it)")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none", help="4bit = QLoRA via bitsandbytes")
    ap.add_argument("--init-adapter", type=Path, default=None,
                    help="start from this PEFT adapter dir (adapter_config.json + adapter_model.safetensors), "
                         "e.g. the SFT run's final/; same --lora-r/--lora-alpha/targets required")
    ap.add_argument("--pg", action="store_true",
                    help="policy-gradient mode: rows carry `advantage`; loss = sum A * CE / tokens (see docstring)")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--attn", default="sdpa", help="attn_implementation (sdpa, flash_attention_2, eager)")
    ap.add_argument("--max-len", type=int, default=32768, help="skip rows longer than this (tokens)")
    ap.add_argument("--loss-chunk", type=int, default=1024, help="trained positions per checkpointed logits chunk")
    ap.add_argument("--eval-frac", type=float, default=0.03, help="held-out rows for eval loss")
    ap.add_argument("--save-every", type=int, default=25, help="optimizer steps between checkpoints")
    ap.add_argument("--keep", type=int, default=2, help="checkpoints to keep")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (testing)")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows (testing)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", type=Path, default=None,
                    help="checkpoint dir (ckpt-NNNNNN, e.g. from a single-GPU run) to start from when --out has none; "
                         "the arguments that set the data order and schedule must match its run's args.json")
    ap.add_argument("--threads", type=int, default=None,
                    help="torch CPU threads (CPU runs on a busy box: 1-2 avoids oversubscription, which made the "
                         "reference Gated DeltaNet kernels ~10x slower in our test)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------- model
def load_model(a):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    kw = {"dtype": dtype, "attn_implementation": a.attn}
    if a.quant == "4bit":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"])
        kw["device_map"] = {"": a.device}
    elif DIST["world"] > 1 and a.device.startswith("cuda"):
        kw["device_map"] = {"": a.device}   # straight to this rank's GPU (K copies may not fit in host RAM)
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    if a.quant == "none" and "device_map" not in kw:
        model.to(a.device)
    processor = AutoProcessor.from_pretrained(a.model)
    for p in model.parameters():
        p.requires_grad_(False)
    # Plain module names (vLLM-friendly adapter_config). They only exist in the language
    # model: the vision tower uses qkv/proj/linear_fc1/linear_fc2; the assert below checks it.
    target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if a.lora_linear_attn:
        target += ["in_proj_qkv", "in_proj_z", "out_proj"]
    cfg = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, bias="none",
                     target_modules=target, task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    if not a.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    core = model.get_base_model()      # Qwen3_5ForConditionalGeneration with LoRA layers injected
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(re.search(r"\.language_model\.layers\.\d+\.(self_attn|mlp|linear_attn)\.", n) and "lora_" in n
                         for n in names), "LoRA must touch only the language model"
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    if is_main():
        log(f"model {type(core).__name__}: {n_all / 1e6:.1f}M params, LoRA trainable {n_train / 1e6:.2f}M "
            f"({len(names)} tensors), target={target}")
    return model, core, processor


def init_from_adapter(model, d: Path, a) -> None:
    """Load a trained adapter's LoRA weights into the fresh LoRA layers (continued training)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = json.loads((d / "adapter_config.json").read_text())
    ours = model.peft_config["default"]
    # everything that sets the shapes or the scale (lora_alpha / r, or / sqrt(r) with rsLoRA, per-module
    # patterns, DoRA) must match; the target modules are checked by the loaded keys below
    defaults = {"use_rslora": False, "use_dora": False, "rank_pattern": {}, "alpha_pattern": {}}
    mismatch = {}
    for k in ("r", "lora_alpha", *defaults):
        theirs = cfg.get(k, defaults.get(k))
        mine = getattr(ours, k, defaults.get(k))
        if (theirs or defaults.get(k)) != (mine or defaults.get(k)):
            mismatch[k] = (theirs, mine)
    if mismatch:
        raise SystemExit(f"--init-adapter {d}: config differs from this run (adapter, run): {mismatch}")
    sd = load_file(str(d / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model, sd)
    unexpected = list(getattr(res, "unexpected_keys", []))
    missing = [k for k in getattr(res, "missing_keys", []) if "lora_" in k]
    if unexpected or missing:
        raise SystemExit(f"--init-adapter {d}: unexpected keys {unexpected[:5]}, missing LoRA keys {missing[:5]}")
    log(f"initialized LoRA from {d} ({len(sd)} tensors)")


# ---------------------------------------------------------------------- loss
def row_weight(row: dict, pg: bool) -> float:
    """Per-row loss weight: 1 for SFT, the row's advantage for --pg (must be a finite number)."""
    if not pg:
        return 1.0
    w = row.get("advantage")
    if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
        raise SystemExit(f"--pg: row {row.get('id')} has no finite `advantage` ({w!r})")
    return float(w)


def lm_loss(core, batch: dict, chunk: int) -> tuple[torch.Tensor, int]:
    """Mean CE over trained positions. Logits only at those positions, in checkpointed chunks."""
    labels = batch.pop("labels")
    out = core.model(**batch, use_cache=False)
    hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state      # [1, L, H]
    tgt = labels[:, 1:]
    pos = (tgt != fhdata.IGNORE).nonzero(as_tuple=False)                      # positions predicting a label
    h = hidden[:, :-1][pos[:, 0], pos[:, 1]]                                  # [N, H]
    y = tgt[pos[:, 0], pos[:, 1]]
    n = y.numel()
    head = core.lm_head

    def ce_sum(hc, yc):
        return F.cross_entropy(head(hc).float(), yc, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, n, chunk):
        hc, yc = h[s:s + chunk], y[s:s + chunk]
        total = total + (checkpoint(ce_sum, hc, yc, use_reentrant=False) if torch.is_grad_enabled() else ce_sum(hc, yc))
    return total / max(n, 1), n


def to_device(enc: dict, device, dtype) -> dict:
    keep = {}
    for k, v in enc.items():
        if not torch.is_tensor(v):
            continue
        v = v.to(device)
        if k == "pixel_values":
            v = v.to(dtype)
        keep[k] = v
    return keep


# ---------------------------------------------------------------------- checkpoints
def ckpt_dirs(out: Path) -> list[Path]:
    ds = [d for d in out.glob("ckpt-*") if (d / "COMPLETE").exists()]
    return sorted(ds, key=lambda d: int(d.name.split("-")[1]))


def save_ckpt(out: Path, model, opt, sched, state: dict, keep: int) -> Path:
    step = state["step"]
    final = out / f"ckpt-{step:06d}"
    tmp = out / f".tmp-ckpt-{step:06d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp / "adapter")
    torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng": {"python": random.getstate(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() and DIST["world"] == 1
                        else None}},
               tmp / "trainer_state.pt")
    (tmp / "state.json").write_text(json.dumps(state, indent=1))
    (tmp / "COMPLETE").write_text("ok\n")
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    for d in ckpt_dirs(out)[:-keep]:
        shutil.rmtree(d, ignore_errors=True)
    return final


def load_ckpt(d: Path, model, opt, sched) -> dict:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd = load_file(str(d / "adapter" / "adapter_model.safetensors"))
    res = set_peft_model_state_dict(model, sd)
    unexpected = getattr(res, "unexpected_keys", [])
    assert not unexpected, f"unexpected adapter keys: {unexpected[:5]}"
    ts = torch.load(d / "trainer_state.pt", map_location="cpu", weights_only=False)
    opt.load_state_dict(ts["opt"])
    sched.load_state_dict(ts["sched"])
    random.setstate(ts["rng"]["python"])
    torch.set_rng_state(ts["rng"]["torch"])
    # Data-parallel runs reseed per rank and step instead (main loop); restoring here would also
    # create CUDA contexts on other ranks' GPUs. A single process restores the devices it has.
    if ts["rng"].get("cuda") is not None and torch.cuda.is_available() and DIST["world"] == 1:
        for i, st in enumerate(ts["rng"]["cuda"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(st, i)
    return json.loads((d / "state.json").read_text())


def truncate_jsonl(p: Path, max_step: int) -> None:
    if not p.exists():
        return
    keep = [l for l in p.read_text().splitlines() if l.strip() and json.loads(l)["step"] <= max_step]
    p.write_text("".join(l + "\n" for l in keep))


# ---------------------------------------------------------------------- data
def split_rows(rows: list[dict], eval_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministic held-out split, stratified by kind."""
    tr, ev = [], []
    for kind in sorted({r["kind"] for r in rows}):
        kr = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["id"])
        random.Random(f"{seed}:{kind}").shuffle(kr)
        n_ev = round(len(kr) * eval_frac)
        ev += kr[:n_ev]
        tr += kr[n_ev:]
    return tr, ev


def epoch_order(n: int, epoch: int, seed: int) -> list[int]:
    idx = list(range(n))
    random.Random(f"{seed}:epoch{epoch}").shuffle(idx)
    return idx


# ---------------------------------------------------------------------- main
# Arguments that fix the data order, the step count and the schedule: a checkpoint from
# another --out (--resume-from) is only meaningful if they match.
RESUME_MUST_MATCH = ("epochs", "lr", "warmup_frac", "grad_accum", "weight_decay", "lora_r", "lora_alpha",
                     "lora_linear_attn", "quant", "max_len", "eval_frac", "limit", "seed")


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.pg and a.eval_frac:
        raise SystemExit("--pg needs --eval-frac 0 (rollout rows are not a held-out set)")
    init_dist(a)
    rank, world = DIST["rank"], DIST["world"]
    if is_main():
        a.out.mkdir(parents=True, exist_ok=True)
    barrier()
    args_path = a.out / "args.json"
    args_d = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if args_path.exists():
        prev = json.loads(args_path.read_text())
        diff = {k: (prev.get(k), v) for k, v in args_d.items()
                if k not in ("max_steps", "device", "save_every", "keep", "threads", "resume_from")
                and prev.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} holds a run with different arguments: {diff}. Use a new --out.")
    if a.resume_from is not None:
        src_args = a.resume_from.parent / "args.json"
        if src_args.exists():
            prev = json.loads(src_args.read_text())
            diff = {k: (prev.get(k), args_d.get(k)) for k in RESUME_MUST_MATCH if k in prev and prev.get(k) != args_d.get(k)}
            if diff:
                raise SystemExit(f"--resume-from {a.resume_from}: its run used different arguments {diff}")
        elif is_main():
            log(f"warning: no {src_args}; cannot check that --resume-from used the same data order and schedule")
    barrier()
    if is_main():
        args_path.write_text(json.dumps(args_d, indent=1))

    if a.threads:
        torch.set_num_threads(a.threads)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = fhdata.read_jsonl(a.data / "sft.jsonl")[: a.limit]
    for r in rows:
        row_weight(r, a.pg)
    tools = json.loads((a.data / "tools.json").read_text())
    train_rows, eval_rows = split_rows(rows, a.eval_frac, a.seed)
    if is_main():
        (a.out / "eval_ids.json").write_text(json.dumps(sorted(r["id"] for r in eval_rows)))

    model, core, processor = load_model(a)
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.999))
    micro_per_epoch = len(train_rows)
    total_micro = math.floor(a.epochs * micro_per_epoch)
    total_steps = max(1, total_micro // a.grad_accum)
    warm = max(1, round(a.warmup_frac * total_steps))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        if a.pg:
            return 1.0
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    state = {"step": 0, "micro": 0, "skipped": []}
    cks = ckpt_dirs(a.out)
    src = cks[-1] if cks else None
    if src is None and a.resume_from is not None:
        src = a.resume_from
        if not (src / "COMPLETE").exists():
            raise SystemExit(f"--resume-from {src}: no COMPLETE marker")
    if src is not None:
        state = load_ckpt(src, model, opt, sched)
        assert state["micro"] == state["step"] * a.grad_accum, f"checkpoint position mismatch: {state}"
        if is_main():
            log(f"resumed from {src}: step {state['step']}, micro {state['micro']} (world {world})")
    elif a.init_adapter is not None:
        init_from_adapter(model, a.init_adapter, a)
    if is_main():
        truncate_jsonl(a.out / "loss.jsonl", state["step"])
        truncate_jsonl(a.out / "eval.jsonl", state["step"])
        if state["step"] >= total_steps:
            log("already finished")
        log(f"{len(train_rows)} train rows, {len(eval_rows)} eval rows, {total_steps} optimizer steps "
            f"({a.epochs} epochs x {micro_per_epoch} / accum {a.grad_accum}), warmup {warm}, data-parallel ranks {world}")
    barrier()

    def on_term(signum, frame):
        STOP["flag"] = True
        log(f"rank {rank} signal {signum}: will checkpoint at the next optimizer step and exit")

    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)

    def row_at(micro: int) -> dict:
        ep, i = divmod(micro, micro_per_epoch)
        return train_rows[epoch_order(micro_per_epoch, ep, a.seed)[i]]

    def evaluate() -> float | None:
        if not eval_rows:
            return None
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for r in eval_rows[rank::world]:
                enc = fhdata.encode(r, a.data, processor, tools)
                if enc["n_tokens"] > a.max_len:
                    continue
                loss, k = lm_loss(core, to_device(enc, a.device, dtype), a.loss_chunk)
                tot += loss.item() * k
                n += k
        model.train()
        tot, n = all_sum([tot, n], a.device)
        return tot / max(n, 1)

    model.train()
    loss_f = open(a.out / "loss.jsonl", "a") if is_main() else None
    t_step = time.time()
    while state["step"] < total_steps:
        opt.zero_grad(set_to_none=True)
        if world > 1:
            # LoRA dropout: distinct per rank, reproducible across resumes.
            torch.manual_seed((a.seed * 1_000_003 + state["step"]) * 1024 + rank)
        acc_loss, acc_tok, acc_seq = 0.0, 0, 0
        # --pg diagnostics: NLL sums / tokens of A>0 and A<0 rows, sum |A| * tokens
        nll_pos, tok_pos, nll_neg, tok_neg, adv_abs = 0.0, 0, 0.0, 0, 0.0
        # Each micro-batch's loss is a mean over its trained tokens; weight by tokens so the
        # step's gradient is the mean over all trained tokens of the step (over all ranks).
        # The step owns global micro indices [start, start + grad_accum); rank r takes every
        # world-th of them, so the rows and their order match a single-GPU run.
        start = state["micro"]
        state["micro"] += a.grad_accum
        encs, skipped = [], []
        for m in range(start + rank, start + a.grad_accum, world):
            r = row_at(m)
            enc = fhdata.encode(r, a.data, processor, tools)
            if enc["n_tokens"] > a.max_len:
                skipped.append((m, r["id"]))
                continue
            encs.append((enc, row_weight(r, a.pg)))
        tok_local = sum(e["n_train_tokens"] for e, w in encs if w != 0.0)
        tok_total, n_encs, n_skipped = all_sum([tok_local, sum(w != 0.0 for _, w in encs), len(skipped)], a.device)
        tok_total = int(tok_total) or 1
        if n_skipped:
            if world > 1:
                gathered = [None] * world
                torch.distributed.all_gather_object(gathered, skipped)
                skipped = sorted(x for g in gathered for x in g)
            state["skipped"].extend(i for _, i in skipped)
        for enc, w in encs:
            if w == 0.0:        # zero advantage: no gradient and not in the token normalization
                continue
            loss, k = lm_loss(core, to_device(enc, a.device, dtype), a.loss_chunk)
            (loss * (w * k / tok_total)).backward()
            acc_loss += w * loss.item() * k
            acc_tok += k
            acc_seq += enc["n_tokens"]
            if w > 0:
                nll_pos, tok_pos = nll_pos + loss.item() * k, tok_pos + k
            else:
                nll_neg, tok_neg = nll_neg + loss.item() * k, tok_neg + k
            adv_abs += abs(w) * k
        if world > 1:
            if n_encs:          # as on one GPU: a step with no trained rows leaves grads None (no update)
                all_reduce_grads(params)
            # Signals reach ranks at different times: every rank acts on the reduced flag.
            acc_loss, acc_tok, acc_seq, sig, nll_pos, tok_pos, nll_neg, tok_neg, adv_abs = all_sum(
                [acc_loss, acc_tok, acc_seq, float(STOP["flag"]), nll_pos, tok_pos, nll_neg, tok_neg, adv_abs], a.device)
            acc_tok, acc_seq = int(acc_tok), int(acc_seq)
            stop_synced = sig > 0
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        opt.step()
        sched.step()
        state["step"] += 1
        rec = {"step": state["step"], "epoch": round(state["micro"] / micro_per_epoch, 4),
               "loss": round(acc_loss / max(acc_tok, 1), 5), "lr": sched.get_last_lr()[0],
               "grad_norm": round(float(gn), 4), "train_tokens": acc_tok, "seq_tokens": acc_seq,
               "seconds": round(time.time() - t_step, 2)}
        if a.pg:
            rec.update(nll_pos=round(nll_pos / tok_pos, 5) if tok_pos else None,
                       nll_neg=round(nll_neg / tok_neg, 5) if tok_neg else None,
                       tok_pos=int(tok_pos), tok_neg=int(tok_neg), adv_abs=round(adv_abs / max(acc_tok, 1), 4))
        t_step = time.time()
        if is_main():
            loss_f.write(json.dumps(rec) + "\n")
            loss_f.flush()
            log(json.dumps(rec))
        stop = stop_synced if world > 1 else STOP["flag"]
        last = state["step"] >= total_steps or (a.max_steps is not None and state["step"] >= a.max_steps)
        if state["step"] % a.save_every == 0 or last or stop:
            ev = evaluate() if (state["step"] % a.save_every == 0 or last) and not stop else None
            if ev is not None and is_main():
                with open(a.out / "eval.jsonl", "a") as f:
                    f.write(json.dumps({"step": state["step"], "eval_loss": round(ev, 5)}) + "\n")
                log(f"eval loss {ev:.5f}")
            if is_main():
                d = save_ckpt(a.out, model, opt, sched, state, a.keep)
                log(f"saved {d.name}")
            barrier()
        if stop:
            if loss_f:
                loss_f.close()
            return 143
        if a.max_steps is not None and state["step"] >= a.max_steps:
            break
    if loss_f:
        loss_f.close()
    if state["step"] >= total_steps and is_main():
        final = a.out / "final"
        if final.exists():
            shutil.rmtree(final)
        model.save_pretrained(final)
        processor.save_pretrained(final / "processor")
        # Completion marker for unattended runs (infra/verda train-ensure): written last,
        # atomically, only after the final adapter is fully saved.
        tmp = a.out / ".DONE.tmp"
        tmp.write_text(json.dumps({"step": state["step"], "total_steps": total_steps,
                                   "final": str(final), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
        os.replace(tmp, a.out / "DONE")
        log(f"done: adapter in {final}")
    barrier()
    return 0


if __name__ == "__main__":
    rc = main()
    if DIST["world"] > 1 and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    sys.exit(rc)
