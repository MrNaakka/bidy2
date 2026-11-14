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

Continued training (expert iteration): --init-adapter DIR starts from a trained PEFT adapter (e.g. the
SFT run's final/) instead of a fresh LoRA. Its rank and alpha must match this run's and it must adapt
the same modules (every LoRA tensor is loaded, none left over). Optimizer and LR schedule start fresh.
A resumed run loads its own checkpoint instead.

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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fhdata

STOP = {"flag": False}


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
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    if a.quant == "none":
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
    log(f"model {type(core).__name__}: {n_all / 1e6:.1f}M params, LoRA trainable {n_train / 1e6:.2f}M "
        f"({len(names)} tensors), target={target}")
    return model, core, processor


def init_from_adapter(model, d: Path, a) -> None:
    """Load a trained adapter's LoRA weights into the fresh LoRA layers (continued training)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    cfg = json.loads((d / "adapter_config.json").read_text())
    ours = model.peft_config["default"]
    # rank and alpha set the scale; the target modules are checked by the loaded keys below
    mismatch = {k: (cfg.get(k), v) for k, v in (("r", ours.r), ("lora_alpha", ours.lora_alpha)) if cfg.get(k) != v}
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
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}},
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
    if ts["rng"].get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ts["rng"]["cuda"])
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
def main(argv=None) -> int:
    a = parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    args_path = a.out / "args.json"
    args_d = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    if args_path.exists():
        prev = json.loads(args_path.read_text())
        diff = {k: (prev.get(k), v) for k, v in args_d.items()
                if k not in ("max_steps", "device", "save_every", "keep", "threads") and prev.get(k) != v}
        if diff:
            raise SystemExit(f"{a.out} holds a run with different arguments: {diff}. Use a new --out.")
    args_path.write_text(json.dumps(args_d, indent=1))

    if a.threads:
        torch.set_num_threads(a.threads)
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    rows = fhdata.read_jsonl(a.data / "sft.jsonl")[: a.limit]
    tools = json.loads((a.data / "tools.json").read_text())
    train_rows, eval_rows = split_rows(rows, a.eval_frac, a.seed)
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
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total_steps - warm))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    state = {"step": 0, "micro": 0, "skipped": []}
    cks = ckpt_dirs(a.out)
    if cks:
        state = load_ckpt(cks[-1], model, opt, sched)
        log(f"resumed from {cks[-1].name}: step {state['step']}, micro {state['micro']}")
    elif a.init_adapter is not None:
        init_from_adapter(model, a.init_adapter, a)
    truncate_jsonl(a.out / "loss.jsonl", state["step"])
    truncate_jsonl(a.out / "eval.jsonl", state["step"])
    if state["step"] >= total_steps:
        log("already finished")
    log(f"{len(train_rows)} train rows, {len(eval_rows)} eval rows, {total_steps} optimizer steps "
        f"({a.epochs} epochs x {micro_per_epoch} / accum {a.grad_accum}), warmup {warm}")

    def on_term(signum, frame):
        STOP["flag"] = True
        log(f"signal {signum}: will checkpoint at the next optimizer step and exit")

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
            for r in eval_rows:
                enc = fhdata.encode(r, a.data, processor, tools)
                if enc["n_tokens"] > a.max_len:
                    continue
                loss, k = lm_loss(core, to_device(enc, a.device, dtype), a.loss_chunk)
                tot += loss.item() * k
                n += k
        model.train()
        return tot / max(n, 1)

    model.train()
    loss_f = open(a.out / "loss.jsonl", "a")
    t_step = time.time()
    while state["step"] < total_steps:
        opt.zero_grad(set_to_none=True)
        acc_loss, acc_tok, acc_seq = 0.0, 0, 0
        # Each micro-batch's loss is a mean over its trained tokens; weight by tokens so the
        # step's gradient is the mean over all trained tokens of the step.
        encs = []
        for _ in range(a.grad_accum):
            r = row_at(state["micro"])
            state["micro"] += 1
            enc = fhdata.encode(r, a.data, processor, tools)
            if enc["n_tokens"] > a.max_len:
                state["skipped"].append(r["id"])
                continue
            encs.append(enc)
        tok_total = sum(e["n_train_tokens"] for e in encs) or 1
        for enc in encs:
            loss, k = lm_loss(core, to_device(enc, a.device, dtype), a.loss_chunk)
            (loss * (k / tok_total)).backward()
            acc_loss += loss.item() * k
            acc_tok += k
            acc_seq += enc["n_tokens"]
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        opt.step()
        sched.step()
        state["step"] += 1
        rec = {"step": state["step"], "epoch": round(state["micro"] / micro_per_epoch, 4),
               "loss": round(acc_loss / max(acc_tok, 1), 5), "lr": sched.get_last_lr()[0],
               "grad_norm": round(float(gn), 4), "train_tokens": acc_tok, "seq_tokens": acc_seq,
               "seconds": round(time.time() - t_step, 2)}
        t_step = time.time()
        loss_f.write(json.dumps(rec) + "\n")
        loss_f.flush()
        log(json.dumps(rec))
        last = state["step"] >= total_steps or (a.max_steps is not None and state["step"] >= a.max_steps)
        if state["step"] % a.save_every == 0 or last or STOP["flag"]:
            ev = evaluate() if (state["step"] % a.save_every == 0 or last) and not STOP["flag"] else None
            if ev is not None:
                with open(a.out / "eval.jsonl", "a") as f:
                    f.write(json.dumps({"step": state["step"], "eval_loss": round(ev, 5)}) + "\n")
                log(f"eval loss {ev:.5f}")
            d = save_ckpt(a.out, model, opt, sched, state, a.keep)
            log(f"saved {d.name}")
        if STOP["flag"]:
            loss_f.close()
            return 143
        if a.max_steps is not None and state["step"] >= a.max_steps:
            break
    loss_f.close()
    if state["step"] >= total_steps:
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
