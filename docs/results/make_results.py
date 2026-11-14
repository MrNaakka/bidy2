"""Build docs/results/{results.json, tables.md, *.png} from results/summary.json.

    uv run --with matplotlib python docs/results/make_results.py

Inputs: results/summary.json (scripts/summarize.py over our runs/) and the SFT v3 training
logs copied to docs/results/data/. No answers are read or written: summary.json holds
per-trial reward, pixels and calls only.
"""
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
KINDS = ["logstream", "dialogs", "scandoc"]
N_ITEMS = 30

# Agent groups: label, job prefix, colour (fixed per entity in every chart).
GROUPS = [
    ("Reference (scripted)", ["fh-eval-reference-all"], "#6250d6"),
    ("Opus 5.5 (OpenCode)", [f"fh-eval-opencode-all-run{i}" for i in (1, 2, 3)], "#2a78d6"),
    ("Qwen3.8-27B + SFT v3", [f"fh-eval-qwen-sft3-run{i}" for i in (1, 2, 3)], "#1baf7a"),
    ("Qwen3.8-27B base", [f"fh-eval-qwen-run{i}" for i in (1, 2, 3)], "#eb6834"),
    ("Stop-early baseline", ["fh-eval-stop-early-all"], "#8a8984"),
    ("Do-nothing baseline", ["fh-eval-nop-all"], "#b5b4ae"),
]
COLOR = {g[0]: g[2] for g in GROUPS}
# The earlier SFT v2 adapter was trained without scandoc and run on the 30 items EVAL had then (incl. toasts);
# summary.json (--split eval) keeps its 20 logstream + dialogs trials.
SFT_V2 = ("Qwen3.8-27B + SFT v2 (20 items, no scandoc)", [f"fh-eval-qwen-sft2-run{i}" for i in (1, 2, 3)])
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# EVAL v1 (history; 30 items: logstream, dialogs, dashboard). From the v1 runs' result.json.
V1 = {
    "items": 30,
    "kinds": ["logstream", "dialogs", "dashboard"],
    "runs": {
        "Reference (scripted)": {"scores": [1.0], "per_kind_correct": [[10, 10, 10]]},
        "Opus 5.5 (OpenCode)": {"scores": [0.70, 0.70, 0.70],
                                "per_kind_correct": [[3, 9, 9], [3, 8, 10], [4, 7, 10]]},
        "Qwen3.8-27B base": {"scores": [round(8 / 30, 3), round(4 / 30, 3), round(6 / 30, 3)],
                             "per_kind_correct": [[4, 1, 3], [0, 1, 3], [0, 1, 5]],
                             "note": "heavier serving contention than the v2 Qwen runs"},
        "Do-nothing baseline": {"scores": [0.0]},
        "Stop-early baseline": {"scores": [0.0]},
    },
}


def load_jobs():
    d = json.loads((ROOT / "results/summary.json").read_text())
    return {j["job"]: j for j in d["jobs"]}


def group_stats(jobs, label, names):
    runs = [jobs[n] for n in names]
    trials = [t for r in runs for t in r["trials"]]
    kinds = [k for k in KINDS if any(t["kind"] == k for t in trials)]
    per_kind = {}
    for k in kinds:
        per_run = [mean(t["reward"] for t in r["trials"] if t["kind"] == k) for r in runs]
        px = [t["pixels_used"] for t in trials if t["kind"] == k]
        per_kind[k] = {"mean_reward": round(mean(per_run), 3), "per_run": [round(x, 3) for x in per_run],
                       "mean_pixels": round(mean(px))}
    cost = [r["cost_usd"] for r in runs if r["cost_usd"] is not None]
    return {
        "label": label,
        "jobs": names,
        "model": runs[0]["model"],
        "n_items": runs[0]["n_trials"],
        "scores": [round(r["mean_reward"], 3) for r in runs],
        "mean_score": round(mean(r["mean_reward"] for r in runs), 3),
        "correct": [r["n_correct"] for r in runs],
        "submitted": [r["n_submitted"] for r in runs],
        "agent_timeouts": [sum(t["exception"] == "AgentTimeoutError" for t in r["trials"]) for r in runs],
        "mean_pixels": [round(r["mean_pixels_used"]) for r in runs],
        "mean_pixels_correct": round(mean([t["pixels_used"] for t in trials if t["correct"]] or [0])),
        "mean_pixels_wrong": round(mean([t["pixels_used"] for t in trials if not t["correct"]] or [0])),
        "max_pixels": max(r["max_pixels_used"] for r in runs),
        "mean_calls": round(mean(r["mean_calls"] for r in runs), 1),
        "api_cost_usd": [round(c, 2) for c in cost] or None,
        "api_cost_usd_total": round(sum(cost), 2) if cost else None,
        "per_kind": per_kind,
    }


def short(label):
    return (label.replace("Qwen3.8-27B + ", "Qwen3.8-27B\n+ ").replace("Qwen3.8-27B base", "Qwen3.8-27B\nbase")
            .replace(" (", "\n(").replace(" baseline", "\nbaseline"))


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=12)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def chart_overall(S):
    order = ["Reference (scripted)", "Opus 5.5 (OpenCode)", "Qwen3.8-27B + SFT v3", "Qwen3.8-27B base",
             "Stop-early baseline", "Do-nothing baseline"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, lab in enumerate(order):
        s = S[lab]
        ax.bar(i, s["mean_score"], width=0.62, color=COLOR[lab], zorder=2)
        xs = [i + d for d in ((-0.13, 0, 0.13) if len(s["scores"]) == 3 else (0,))]
        ax.scatter(xs, s["scores"], s=46, color="white", edgecolor=INK, linewidth=1.3, zorder=3)
        ax.text(i, max([s["mean_score"]] + s["scores"]) + 0.035, f"{s['mean_score']:.2f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=INK)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short(o) for o in order], fontsize=11.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean reward (fraction of 30 items exactly right)", fontsize=12.5)
    ax.set_title("FrameHunt EVAL (30 items, 120k-pixel budget): bar = mean, dots = individual runs",
                 fontsize=14, color=INK, loc="left")
    style(ax)
    save(fig, "overall.png")


def chart_per_kind(S):
    labs = ["Opus 5.5 (OpenCode)", "Qwen3.8-27B base", "Qwen3.8-27B + SFT v3"]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    w = 0.26
    for j, lab in enumerate(labs):
        for i, k in enumerate(KINDS):
            pk = S[lab]["per_kind"][k]
            x = i + (j - 1) * w
            ax.bar(x, pk["mean_reward"], width=w - 0.03, color=COLOR[lab], zorder=2,
                   label=lab if i == 0 else None)
            ax.scatter([x] * len(pk["per_run"]), pk["per_run"], s=22, color="white", edgecolor=INK,
                       linewidth=1, zorder=3)
            ax.text(x, max([pk["mean_reward"]] + pk["per_run"]) + 0.03, f"{pk['mean_reward']:.2f}", ha="center", fontsize=10.5,
                    color=INK)
    ax.axhline(1.0, color=COLOR["Reference (scripted)"], linestyle="--", linewidth=1.5, zorder=1)
    ax.text(len(KINDS) - 0.5, 1.015, "reference 1.00 on every kind", ha="right", va="bottom", fontsize=10.5,
            color=COLOR["Reference (scripted)"])
    ax.set_xticks(range(len(KINDS)))
    ax.set_xticklabels(["logstream\n(terminal log)", "dialogs\n(error dialog)",
                        "scandoc\n(scanned report)"], fontsize=12)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Mean reward per kind (10 items each)", fontsize=12.5)
    ax.set_title("Per kind: bar = mean of 3 runs, dots = runs", fontsize=14, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    style(ax)
    save(fig, "per_kind.png")


def chart_pixels(S, jobs):
    labs = ["Reference (scripted)", "Opus 5.5 (OpenCode)", "Qwen3.8-27B + SFT v3", "Qwen3.8-27B base",
            "Stop-early baseline"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    for lab in labs:
        s = S[lab]
        a1.scatter([p / 1000 for p in s["mean_pixels"]], s["scores"], s=110, color=COLOR[lab],
                   edgecolor="white", linewidth=1.5, zorder=3, label=lab)
    a1.axvline(120, color=MUTED, linestyle=":", linewidth=1.3)
    a1.text(119, 1.02, "budget 120k", ha="right", fontsize=10.5, color=MUTED)
    a1.set_xlim(0, 128)
    a1.set_ylim(-0.04, 1.1)
    a1.set_xlabel("Mean pixels used per item (thousands)", fontsize=12.5)
    a1.set_ylabel("Mean reward (one dot per run)", fontsize=12.5)
    a1.set_title("Spending more pixels does not buy accuracy", fontsize=14, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=10.5, loc="lower left")
    style(a1)
    a1.grid(axis="x", color=GRID)

    labs2 = ["Opus 5.5 (OpenCode)", "Qwen3.8-27B base", "Qwen3.8-27B + SFT v3"]
    for i, lab in enumerate(labs2):
        s = S[lab]
        sub = mean(s["submitted"]) / s["n_items"]
        a2.bar(i, sub, width=0.6, color=COLOR[lab], zorder=2)
        a2.text(i, sub + 0.02, f"{mean(s['submitted']):.1f}/{s['n_items']}".replace(".0/", "/"), ha="center", fontsize=12,
                color=INK, fontweight="bold")
        to = sum(s["agent_timeouts"])
        a2.text(i, 0.04, f"timeouts:\n{to} of {len(s['scores']) * s['n_items']}", ha="center",
                fontsize=10.5, color="white" if sub > 0.2 else INK)
    a2.set_xticks(range(3))
    a2.set_xticklabels([short(l) for l in labs2], fontsize=11)
    a2.set_ylim(0, 1.12)
    a2.set_ylabel("Items with a submitted answer (mean of 3 runs)", fontsize=12.5)
    a2.set_title("Submission rate", fontsize=14, color=INK, loc="left")
    style(a2)
    fig.tight_layout(w_pad=3)
    save(fig, "pixels_and_submissions.png")


def chart_training():
    loss = [json.loads(l) for l in (OUT / "data/sft-v3-loss.jsonl").read_text().splitlines() if l.strip()]
    ev = [json.loads(l) for l in (OUT / "data/sft-v3-eval.jsonl").read_text().splitlines() if l.strip()]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot([r["step"] for r in loss], [r["loss"] for r in loss], color="#86b6ef", linewidth=2,
            label="train loss (per optimizer step)")
    ax.plot([r["step"] for r in ev], [r["eval_loss"] for r in ev], color="#1c5cab", linewidth=2,
            marker="o", markersize=7, label="held-out eval loss (3% of rows)")
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step (1 epoch, 8 rows per step)", fontsize=12.5)
    ax.set_ylabel("Cross-entropy on assistant tokens (log scale)", fontsize=12.5)
    ax.set_title("SFT v3: LoRA r=16 on Qwen3.8-27B, 1,200 teacher trajectories", fontsize=14, color=INK,
                 loc="left")
    last = ev[-1]
    ax.annotate(f"eval {last['eval_loss']:.3f}", (last["step"], last["eval_loss"]),
                xytext=(-10, 22), textcoords="offset points", ha="right", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=11)
    style(ax)
    save(fig, "sft_v3_loss.png")


def tables(S):
    L = ["# FrameHunt results (EVAL v2)", "",
         "- **Set:** 30 items, 10 each of logstream, dialogs and scandoc.",
         "- **Budget:** 120,000 pixels per item.",
         "- **Reward:** 1 only for an exact first submission, else 0.",
         "- **Charts:** [overall.png](overall.png) (score per agent, dots = runs), "
         "[per_kind.png](per_kind.png), [pixels_and_submissions.png](pixels_and_submissions.png), "
         "[sft_v3_loss.png](sft_v3_loss.png).",
         "- **Data:** [results.json](results.json), built from `results/summary.json` by "
         "`make_results.py`. Regenerate with `uv run --with matplotlib python docs/results/make_results.py`.", "",
         "## Overall", "",
         "| Agent | Runs | Mean | Correct per run | Submitted per run | Timeouts | Mean pixels per run |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for lab, _, _ in GROUPS:
        s = S[lab]
        L.append(f"| {lab} | {' / '.join(f'{x:.3f}' for x in s['scores'])} | **{s['mean_score']:.3f}** | "
                 f"{' / '.join(map(str, s['correct']))} (of {s['n_items']}) | "
                 f"{' / '.join(map(str, s['submitted']))} | {sum(s['agent_timeouts'])} | "
                 f"{' / '.join(f'{p / 1000:.0f}k' for p in s['mean_pixels'])} |")
    L += ["", "## Per kind (mean reward over runs; mean pixels)", "",
          "| Agent | " + " | ".join(KINDS) + " |", "| --- |" + " --- |" * len(KINDS)]
    for lab, _, _ in GROUPS:
        pk = S[lab]["per_kind"]
        L.append(f"| {lab} | " + " | ".join(
            f"{pk[k]['mean_reward']:.2f} ({pk[k]['mean_pixels'] / 1000:.0f}k)" if k in pk else "-" for k in KINDS)
            + " |")
    o = S["Opus 5.5 (OpenCode)"]
    L += ["", "## Pixels on correct vs wrong answers", "",
          "| Agent | Mean px, correct | Mean px, wrong | Max px |", "| --- | --- | --- | --- |"]
    for lab in ("Reference (scripted)", "Opus 5.5 (OpenCode)", "Qwen3.8-27B base", "Qwen3.8-27B + SFT v3"):
        s = S[lab]
        L.append(f"| {lab} | {s['mean_pixels_correct']:,} | "
                 f"{s['mean_pixels_wrong']:,} | {s['max_pixels']:,} |".replace("| 0 |", "| - |"))
    L += ["", f"Opus API cost: {' + '.join(f'${c:.2f}' for c in o['api_cost_usd'])} "
              f"= ${o['api_cost_usd_total']:.2f} for 3 x 30 trials.", "",
          "## EVAL v1 (history)", "",
          "The first EVAL set had 30 items (10 each of logstream, dialogs and `dashboard`) and ran under "
          "the first salt, now retired. Dashboard was too easy (Opus 29/30 over three runs), so EVAL v2 "
          "replaced it with the scandoc document kind. The Qwen base v1 runs had "
          "heavier serving contention than the v2 runs. Rebuild v1 with "
          "`scripts/build_tasks.py --split eval --v1`.", "",
          "Correct answers per run, out of 10 per kind:", "",
          "| Agent | Runs | logstream | dialogs | dashboard |", "| --- | --- | --- | --- | --- |"]
    for lab, r in V1["runs"].items():
        pk = r.get("per_kind_correct")
        cells = (" | ".join("/".join(str(p[i]) for p in pk) for i in range(3))) if pk else "0 | 0 | 0"
        L.append(f"| {lab} | {' / '.join(f'{x:.2f}' for x in r['scores'])} | {cells} |")
    (OUT / "tables.md").write_text("\n".join(L) + "\n")


def main():
    jobs = load_jobs()
    S = {lab: group_stats(jobs, lab, names) for lab, names, _ in GROUPS}
    (OUT / "results.json").write_text(json.dumps({
        "eval": {"split": "EVAL v2", "n_items": N_ITEMS, "kinds": KINDS, "items_per_kind": 10,
                 "budget_px": 120000, "reward": "1 if the first submitted answer is exact, else 0"},
        "agents": S,
        "earlier_sft_v2": group_stats(jobs, *SFT_V2),
        "eval_v1_history": V1,
        # Judge-path check: fresh clone of main, its own new salt, scripts/eval.sh --env modal -n 40 --runs 1.
        # That run still had 40 items (10 toasts items, since removed from EVAL); the scores below are
        # rescored on the 30 current items with scripts/summarize.py --split eval. Wall times and the total
        # API cost ($5.72) are for the 40-item run.
        "fresh_clone_check": {
            "date": "2026-11-14", "command": "scripts/eval.sh --env modal -n 40 --runs 1",
            "wall_seconds_40_items": {"build": 5, "reference": 139, "nop": 132, "stop_early": 327, "opus_run1": 241,
                                      "total": 845},
            "reference": 1.0, "nop": 0.0, "stop_early": 0.0,
            "opus_5_5": {"score": 0.6, "correct": 18, "per_kind": {"logstream": 0.5, "dialogs": 0.6, "scandoc": 0.7},
                         "mean_pixels": 112414, "api_cost_usd": 4.40},
            "log": "examples/harbor-runs/fresh-clone-modal-opus-run1/",
        },
        "source": "results/summary.json (scripts/summarize.py over runs/); no answers included",
    }, indent=1) + "\n")
    tables(S)
    chart_overall(S)
    chart_per_kind(S)
    chart_pixels(S, jobs)
    chart_training()


if __name__ == "__main__":
    main()
