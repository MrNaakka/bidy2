"""GRPO stage: sft_lora.py --pg (advantage-weighted loss) on the tiny model, and rollouts_to_pg.py.

The loss tests need the tiny model (train/make_tiny_model.py) and train/data/sft-v1 (git-ignored) and are
skipped without them; they run on CPU in a few minutes:

    nice -n 19 train/.venv/bin/python -m pytest tests/test_grpo.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import rollouts_to_pg as r2pg  # noqa: E402


# ---------------------------------------------------------------------- advantages / conversion
def test_advantages_are_group_normalized():
    mean, std, adv = r2pg.advantages([1, 0, 0, 1, 1, 1, 0, 1])
    assert mean == 0.625 and abs(std - (0.625 * 0.375) ** 0.5) < 1e-12
    assert abs(sum(adv)) < 1e-9
    assert all(a > 0 for a, r in zip(adv, [1, 0, 0, 1, 1, 1, 0, 1]) if r == 1)
    assert all(a < 0 for a, r in zip(adv, [1, 0, 0, 1, 1, 1, 0, 1]) if r == 0)
    _, std, adv = r2pg.advantages([1.0] * 8)
    assert std == 0 and adv == [0.0] * 8


def test_trial_reward_counts_agent_failures_only(tmp_path):
    def trial(name, reward=None, exc=None):
        d = tmp_path / name
        (d / "verifier").mkdir(parents=True)
        if reward is not None:
            (d / "verifier" / "reward.txt").write_text(str(reward))
        (d / "result.json").write_text(json.dumps({"exception_info": {"exception_type": exc} if exc else None}))
        return d
    assert r2pg.trial_reward(trial("a", 1)) == (1.0, None)
    assert r2pg.trial_reward(trial("b", None, "AgentTimeoutError")) == (0.0, "AgentTimeoutError")
    assert r2pg.trial_reward(trial("c", None, None)) == (0.0, None)
    assert r2pg.trial_reward(trial("d", None, "EnvironmentStartTimeoutError")) == (None, "EnvironmentStartTimeoutError")
    assert r2pg.trial_reward(trial("e", 1, "VerifierTimeoutError")) == (None, "VerifierTimeoutError")


def test_all_rollouts_become_rows_with_advantages(tmp_path):
    tei = pytest.importorskip("test_expert_iteration")
    good, bad, nosubmit = tei.script(), tei.script("WRONG"), tei.script()[:4] + [("Out of ideas.", "Gave up.", None,
                                                                                 None, None)]
    item, other, flat = "logstream-100301", "dialogs-110301", "toasts-130301"
    trials = [(f"{item}__T{i}", item, r, s) for i, (r, s) in enumerate(
        [(1.0, good), (0.0, bad), (0.0, nosubmit), (1.0, good)])]
    trials += [(f"{other}__U{i}", other, 0.0, bad) for i in range(4)]          # zero std: dropped
    trials += [(f"{flat}__V{i}", flat, 1.0, good) for i in range(2)]           # too small a group
    run, wires = tei.rollout_run(tmp_path, "fh-grpo-it1-g0", trials)
    # a harness failure (no reward, setup error) is left out of the group
    t = run / f"{item}__T3"
    (t / "verifier" / "reward.txt").unlink()
    (t / "result.json").write_text(json.dumps({"task_name": f"framehunt/{item}", "trial_name": f"{item}__T3",
                                               "exception_info": {"exception_type": "AgentSetupTimeoutError"}}))
    out = tmp_path / "pg"
    assert r2pg.main([str(run), "--out", str(out), "--min-group", "3"]) == 0
    rows = {r["id"].split("__", 1)[1]: r for r in map(json.loads, (out / "sft.jsonl").read_text().splitlines())}
    assert sorted(rows) == [f"{item}__T0", f"{item}__T1", f"{item}__T2"]
    mean, std, adv = r2pg.advantages([1.0, 0.0, 0.0])
    for i, a in enumerate(adv):
        r = rows[f"{item}__T{i}"]
        assert abs(r["advantage"] - a) < 1e-12 and r["group_n"] == 3 and abs(r["group_mean"] - mean) < 1e-12
    assert rows[f"{item}__T0"]["correct"] and not rows[f"{item}__T1"]["correct"]
    # the run without a submit call is a training row too (a negative one), exactly as logged
    r = rows[f"{item}__T2"]
    assert "framehunt_submit" not in r["tools_used"] and r["advantage"] < 0
    stats = json.loads((out / "pg_stats.json").read_text())
    assert stats["groups"] == 3 and stats["groups_with_signal"] == 1 and stats["groups_small"] == 1
    assert stats["excluded_harness_errors"] == {"AgentSetupTimeoutError": 1}
    assert stats["by_kind"]["logstream"]["mean_reward"] == round(1 / 3, 4)
    assert stats["by_kind"]["dialogs"]["mean_reward"] == 0.0 and stats["by_kind"]["toasts"]["mean_reward"] == 1.0
    assert stats["rows_pos"] == 1 and stats["rows_neg"] == 2


# ---------------------------------------------------------------------- the loss (tiny model)
try:
    import peft  # noqa: F401
    import torch
    import test_sft_ddp as ddp
    HAVE_TINY = (ddp.TINY / "config.json").exists() and (ddp.DATA / "sft.jsonl").exists()
except (ImportError, pytest.skip.Exception):
    HAVE_TINY = False

needs_tiny = pytest.mark.skipif(not HAVE_TINY, reason="needs torch, peft, the tiny model and train/data/sft-v1")


def pg_data(dst: Path, advs: list[float]) -> Path:
    src = ddp.short_data(dst.with_name(dst.name + "-src"), n=len(advs))
    rows = [json.loads(l) for l in (src / "sft.jsonl").read_text().splitlines()]
    dst.mkdir(parents=True)
    (dst / "sft.jsonl").write_text("".join(json.dumps(dict(r, advantage=a)) + "\n" for r, a in zip(rows, advs)))
    (dst / "tools.json").write_text("[]")
    (dst / "images").symlink_to(ddp.DATA / "images")
    return dst


def pg_args(data: Path, out: Path, n: int, *extra: str) -> list[str]:
    return [str(ROOT / "train" / "sft_lora.py"), "--model", str(ddp.TINY), "--data", str(data), "--out", str(out),
            "--pg", "--eval-frac", "0", "--grad-accum", str(n), "--epochs", "1", "--save-every", "100",
            "--warmup-frac", "0", "--dtype", "fp32", "--lr", "1e-2", "--lora-dropout", "0", "--threads", "1",
            "--device", "cpu", *extra]


def nll(adapter: Path | None, data: Path) -> list[float]:
    """Token-mean NLL of every row of data/sft.jsonl under the tiny model + adapter (None: fresh LoRA)."""
    import fhdata
    import sft_lora
    a = sft_lora.parse_args(["--model", str(ddp.TINY), "--data", str(data), "--out", str(data / "x"),
                             "--dtype", "fp32", "--device", "cpu", "--lora-dropout", "0"])
    model, core, processor = sft_lora.load_model(a)
    if adapter is not None:
        sft_lora.init_from_adapter(model, adapter, a)
    model.eval()
    out = []
    with torch.no_grad():
        for r in fhdata.read_jsonl(data / "sft.jsonl"):
            enc = fhdata.encode(r, data, processor, [])
            loss, _ = sft_lora.lm_loss(core, sft_lora.to_device(enc, "cpu", torch.float32), 1024)
            out.append(loss.item())
    return out


@needs_tiny
def test_pg_moves_log_probs_in_the_advantage_direction(tmp_path):
    data = pg_data(tmp_path / "d", [1.0, -1.0])
    pos, neg = tmp_path / "pos", tmp_path / "neg"
    ddp.run(pg_args(data, pos, 1, "--limit", "1"), 1)              # row 0, A = +1
    one = tmp_path / "d1"                                           # row 1 alone, A = -1
    one.mkdir()
    (one / "sft.jsonl").write_text((data / "sft.jsonl").read_text().splitlines()[1] + "\n")
    (one / "tools.json").write_text("[]")
    (one / "images").symlink_to(ddp.DATA / "images")
    ddp.run(pg_args(one, neg, 1), 1)
    before = nll(None, data)
    after_pos, after_neg = nll(pos / "final", data), nll(neg / "final", data)
    assert after_pos[0] < before[0] - 1e-3, (before, after_pos)    # A > 0: sampled tokens more likely
    assert after_neg[1] > before[1] + 1e-3, (before, after_neg)    # A < 0: less likely
    rec = json.loads((neg / "loss.jsonl").read_text().splitlines()[0])
    assert rec["nll_pos"] is None and rec["nll_neg"] > 0 and rec["loss"] < 0 and rec["adv_abs"] == 1.0


@needs_tiny
def test_zero_advantage_rows_contribute_nothing(tmp_path):
    with_zero = pg_data(tmp_path / "z", [1.0, 0.0])
    a, b = tmp_path / "a", tmp_path / "b"
    ddp.run(pg_args(with_zero, a, 2), 1)                            # one step: rows 0 (A=1) and 1 (A=0)
    ddp.run(pg_args(with_zero, b, 1, "--limit", "1"), 1)           # one step: row 0 only
    ddp.close(ddp.adapter(a / "ckpt-000001"), ddp.adapter(b / "ckpt-000001"))


@needs_tiny
def test_pg_ddp_matches_single(tmp_path):
    # 7 rows, accum 3: steps of 3, 3 and 1 rows (--pg uses every row once)
    data = pg_data(tmp_path / "d", [1.2, -0.8, 0.5, -1.5, 0.0, 2.0, -0.7])
    one, two = tmp_path / "one", tmp_path / "two"
    ddp.run(pg_args(data, one, 3), 1)
    ddp.run(pg_args(data, two, 3), 2)
    l1 = [json.loads(l) for l in (one / "loss.jsonl").read_text().splitlines()]
    l2 = [json.loads(l) for l in (two / "loss.jsonl").read_text().splitlines()]
    assert len(l1) == len(l2) == 3
    for x, y in zip(l1, l2):
        assert x["train_tokens"] == y["train_tokens"] and abs(x["loss"] - y["loss"]) < 1e-4
        assert abs(x["grad_norm"] - y["grad_norm"]) < 1e-3
    ddp.close(ddp.adapter(one / "ckpt-000003"), ddp.adapter(two / "ckpt-000003"))
    assert json.loads((two / "ckpt-000003" / "state.json").read_text())["micro"] == 7


def test_system_prompt_may_name_the_served_model():
    import rollouts_to_sft as r2s
    import fhdata
    d = "Sat Nov 14 2026"
    s = r2s.expected_system_prompt(d, "qwen3.8-27b-fh")
    assert "named qwen3.8-27b-fh. The exact model ID is verda-qwen/qwen3.8-27b-fh\n" in s
    assert s.replace("qwen3.8-27b-fh", "qwen3.8-27b") == fhdata.system_prompt(d)
    assert r2s.expected_system_prompt(d, "qwen3.8-27b") == fhdata.system_prompt(d)
