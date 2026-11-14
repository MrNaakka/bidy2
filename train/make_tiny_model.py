"""Build a tiny random-weight Qwen3.8 (qwen3_5 architecture) for CPU validation of sft_lora.py.

    train/.venv/bin/python train/make_tiny_model.py --src train/hf/Qwen3.8-27B --out train/out/tiny-qwen3.8

Same architecture family as Qwen/Qwen3.8-27B (Qwen3_5ForConditionalGeneration:
gated-delta-net linear attention + gated full attention every 4th layer, mRoPE,
Qwen3-VL-style vision tower and patch merger), same tokenizer, chat template and
image processor, but tiny dimensions (4 text layers, hidden 64, 2 vision blocks).
The real config is read from --src (config.json only, no weights).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROC_FILES = ["chat_template.jinja", "merges.txt", "preprocessor_config.json", "tokenizer.json",
              "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json", "generation_config.json"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path(__file__).resolve().parent / "hf" / "Qwen3.8-27B")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "out" / "tiny-qwen3.8")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText

    cfg = json.loads((a.src / "config.json").read_text())
    t, v = cfg["text_config"], cfg["vision_config"]
    t.update({
        "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "mtp_num_hidden_layers": 0,
    })
    t["rope_parameters"] = dict(t["rope_parameters"], mrope_section=[2, 1, 1])   # sums to head_dim*0.25/2
    v.update({"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64})
    a.out.mkdir(parents=True, exist_ok=True)
    tmp = a.out / "config.json"
    tmp.write_text(json.dumps(cfg, indent=1))
    config = AutoConfig.from_pretrained(a.out)
    torch.manual_seed(a.seed)
    model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
    model.save_pretrained(a.out)
    for f in PROC_FILES:
        if (a.src / f).exists():
            shutil.copy(a.src / f, a.out / f)
    n = sum(p.numel() for p in model.parameters())
    print(f"tiny {type(model).__name__}: {n / 1e6:.1f}M params -> {a.out}")


if __name__ == "__main__":
    main()
