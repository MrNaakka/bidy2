"""Data-parallel sft_lora.py (torchrun, gloo on CPU) matches the single-process run.

Needs the tiny model (train/make_tiny_model.py) and train/data/sft-v1 (both git-ignored);
skipped without them. Rows are shortened (system prompt and tool list cut, 3 assistant
turns) so a CPU step takes seconds. ~3-5 min with one thread per process:

    nice -n 19 train/.venv/bin/python -m pytest tests/test_sft_ddp.py -q
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TINY = Path(os.environ.get("FH_TINY_MODEL", ROOT / "train" / "out" / "tiny-qwen3.8"))
DATA = Path(os.environ.get("FH_SFT_DATA", ROOT / "train" / "data" / "sft-v1"))

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytestmark = pytest.mark.skipif(not (TINY / "config.json").exists() or not (DATA / "sft.jsonl").exists(),
                                reason="needs the tiny model and train/data/sft-v1")


def short_data(dst: Path, n: int = 16) -> Path:
    dst.mkdir(parents=True)
    out = []
    with open(DATA / "sft.jsonl") as f:
        for line in f:
            r = json.loads(line)
            msgs, n_asst = [], 0
            for m in r["messages"]:
                if m["role"] == "system":
                    m = dict(m, content=m["content"][:300])
                msgs.append(m)
                n_asst += m["role"] == "assistant"
                if n_asst == 3:
                    break
            out.append(dict(r, messages=msgs))
            if len(out) == n:
                break
    (dst / "sft.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    (dst / "tools.json").write_text("[]")
    (dst / "images").symlink_to(DATA / "images")
    return dst


def common(data: Path, out: Path, *extra: str) -> list[str]:
    return [str(ROOT / "train" / "sft_lora.py"), "--model", str(TINY), "--data", str(data), "--out", str(out),
            "--limit", "16", "--eval-frac", "0.13", "--grad-accum", "3", "--epochs", "1", "--save-every", "2",
            "--dtype", "fp32", "--lr", "1e-2", "--lora-dropout", "0", "--threads", "1", "--device", "cpu", *extra]


def run(args: list[str], nproc: int) -> None:
    env = dict(os.environ, OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
    for k in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        env.pop(k, None)
    pre = [sys.executable] if nproc == 1 else [sys.executable, "-m", "torch.distributed.run", "--standalone",
                                               f"--nproc_per_node={nproc}"]
    p = subprocess.run(pre + args, env=env, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-6000:]


def losses(out: Path) -> list[tuple[int, float, int, float]]:
    return [(r["step"], r["loss"], r["train_tokens"], r["grad_norm"])
            for r in map(json.loads, (out / "loss.jsonl").read_text().splitlines())]


def adapter(d: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(d / "adapter" / "adapter_model.safetensors"))


def close(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=1e-4, atol=1e-6, msg=k)


def same_log(x, y) -> None:
    assert [r[0] for r in x] == [r[0] for r in y]
    for (s, l1, t1, g1), (_, l2, t2, g2) in zip(x, y):
        assert t1 == t2, s                      # same rows trained at every step
        assert abs(l1 - l2) < 1e-4 and abs(g1 - g2) < 1e-3, (s, l1, l2, g1, g2)


def test_ddp_matches_single(tmp_path):
    data = short_data(tmp_path / "data")
    one, two, res = tmp_path / "one", tmp_path / "two", tmp_path / "res"
    # 14 train rows / accum 3 = 4 steps; accum 3 over 2 ranks exercises the uneven split.
    run(common(data, one), 1)
    run(common(data, two), 2)
    same_log(losses(one), losses(two))
    assert len(losses(one)) == 4
    ev1 = [json.loads(l) for l in (one / "eval.jsonl").read_text().splitlines()]
    ev2 = [json.loads(l) for l in (two / "eval.jsonl").read_text().splitlines()]
    assert [e["step"] for e in ev1] == [e["step"] for e in ev2] == [2, 4]
    for e1, e2 in zip(ev1, ev2):
        assert abs(e1["eval_loss"] - e2["eval_loss"]) < 1e-4
    close(adapter(one / "ckpt-000004"), adapter(two / "ckpt-000004"))
    assert (two / "DONE").exists() and (two / "final" / "adapter_model.safetensors").exists()
    assert json.loads((two / "ckpt-000004" / "state.json").read_text())["micro"] == 12

    # A checkpoint written by the single-process run continues on 2 ranks (fresh --out).
    src = tmp_path / "src"
    shutil.copytree(one / "ckpt-000002", src / "ckpt-000002")
    shutil.copy(one / "args.json", src / "args.json")
    run(common(data, res, "--resume-from", str(src / "ckpt-000002")), 2)
    assert [r[0] for r in losses(res)] == [3, 4]
    same_log(losses(one)[2:], losses(res))
    close(adapter(one / "ckpt-000004"), adapter(res / "ckpt-000004"))
