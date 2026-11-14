# Handoff: finding more failure scenarios (written 2026-11-14, ~15:20 local)

> **Superseded.** This was the working handoff for finding new scene kinds. It is kept for
> history. For the final state (EVAL v2: logstream, dialogs, scandoc, 10 each; results;
> training), see [README.md](../README.md) and [docs/results/](results/tables.md).

This file lets a fresh agent on a fresh machine start work right away. Read it first. Then read:

- `AGENTS.md`: judging criteria.
- `README.md`: results, failure analysis, how to run.
- `docs/design.md`: item rules, interfaces, reward integrity.

`docs/framing.md` is the demo pitch.

## 1. The job

FrameHunt is our entry to AfterQuery's "Break the Agent" challenge (Junction 2026). The agent
answers a question about a long synthetic screen recording. It can only request single frames
(cropped, downscaled) through an MCP server. Every returned pixel counts against a
120,000-pixel budget. Reward is 1 for an exact first answer, else 0.

**Your task: find more scene kinds ("failure scenarios") where Opus 5.5 fails reliably for
reasons we can explain, while the scripted reference still scores 100%.** The logstream kind
already does this. We want failures of the same quality in other settings.

The scoring criteria (AGENTS.md) are the gap (30), meaningful difficulty (30), reward integrity
(25) and craft (15). Difficulty must come from the agent's decisions. Hidden information, trick
wording, impossible timing or broken tools score zero.

## 2. What we know so far

### Results on EVAL v1 (30 items, 10 per kind, 120k budget, through Harbor)

| Run | logstream | dialogs | dashboard | Total |
| --- | --- | --- | --- | --- |
| Reference (scripted) | 10/10 | 10/10 | 10/10 | 30/30 |
| Do-nothing | 0 | 0 | 0 | 0/30 |
| Stop-early | 0 | 0 | 0 | 0/30 |
| Opus 5.5 run 1 | 3/10 | 9/10 | 9/10 | 21/30 |
| Opus 5.5 run 2 | 3/10 | 8/10 | 10/10 | 21/30 |
| Opus 5.5 run 3 | 4/10 | 7/10 | 10/10 | 21/30 |
| Qwen3.8-27B base run 1 | 4/10 | 1/10 | 3/10 | 8/30 |
| Qwen3.8-27B base run 2 | 0/10 | 1/10 | 3/10 | 4/30 |
| Qwen3.8-27B base run 3 | 0/10 | 1/10 | 5/10 | 6/30 |

Run 3 and the Qwen rows are tallied from `runs/*/verifier/result.json`. The README table
still says run 3 is "in progress" and should be updated.

Logstream items 7001, 7004, 7007, 7008 and 7009 failed in both Opus runs 1 and 2. Opus failed
7008 in all three runs, with a different wrong id each time and almost the whole budget spent.

### Why logstream breaks Opus and the other two kinds don't

Logstream is the only v1 kind with all three of these properties:

1. **No margin before the event.** One frame early, the K-th ERROR line doesn't exist yet,
   but the (K-1)-th ERROR line is on screen and looks like a valid answer. In 9 of 14
   logstream failures the agent read 1–4 frames early. It had already bought a contact sheet
   that showed K-1 on that frame and misread the tiny tiles (8–18 px tall). It never spent a
   few hundred pixels on one direct look to check.
2. **The answer moves after the event.** Lines arrive 3–5 per second, so "newest line =
   answer" only holds for a few frames. In 5 of 14 failures the agent had the right moment
   but read the wrong row.
3. **Look-alike distractors everywhere.** Every row has an 8-hex id, so a wrong row still
   gives a well-formed answer.

Opus also wastes budget:

- It spends a median of 62k pixels before its first look at the log. The reference needs at
  most 58.6k for a whole item.
- In 14 of 60 trials its first call was a full frame at the default width, costing 57,600
  pixels (48% of the budget).

So one slip can't be recovered. In 17 of 18 failures, the final message admits it is guessing.

Dialogs and dashboard are easy for three reasons. The answer stays on screen for seconds on
both sides of the event, it sits at a fixed spot, and nothing on screen looks like it. An
imprecise search still lands on it.

**The recipe for a new failure scenario:** being slightly wrong about *when* or *where* must
still produce a well-formed wrong answer that the agent trusts, while checking costs only a
few hundred pixels. Full analysis: README.md, "Why it's hard".

### Already built (don't redo)

`toasts` and `leaderboard` carry the logstream mechanism into two new settings. Both are
implemented, tested and reviewed. Descriptions are in `docs/design.md`, "Second round".

**Superseded: the final EVAL v2 is logstream + dialogs + scandoc, 10 each; toasts and
leaderboard are DEV/TRAIN only. See README "Results" and docs/design.md "Second round"; pilot
results are there too.**

- Reference on 200 random seeds per new kind: 100% correct. Maximum spend was 41,048 px
  (toasts) and 47,648 px (leaderboard).
- Reference through Harbor on the 12 pilot items (DEV ids toasts-3401..3406,
  leaderboard-3501..3506): 12/12. One trial first failed on a container error under host
  load and passed on rerun.
- **Not yet known: how Opus scores on toasts and leaderboard.** The Opus pilot on those 12
  items had not run when this was written. The other machine (see section 7) is about to
  run it. Until then, "toasts and leaderboard break Opus" is a hypothesis.

### Ideas that are planned but not built (your starting list)

These come from the planning pass in the earlier thread. None has been implemented or tested.

- **`incident` (two-hop logstream), the best next candidate.**
  - Setup: the terminal gets a status-bar badge that stays on screen. It switches from
    `v4.12` (grey) to `v4.13 LIVE` (green) at deploy frame D.
  - Question: "request id on the last ERROR line printed before the deploy".
  - Solution path: find D, read the counter at D to learn N, binary-search back to where the
    counter first read N, then find the row and read the id.
  - Constraints: the N-th ERROR line has scrolled off by D (at least 15 s earlier), and no
    ERROR line falls within 3 s of D.
  - What it tests: budget planning (two searches must fit) and "adapting when new
    information arrives" (K is only known after the first observation).
  - Estimates: reference cost 56–63k; 3–4 h of work as a subclass of `LogStreamScene`.
  - Risk: judges may see it as "more logstream".
- **Shorter dialogs (open 1.0–1.5 s instead of 3–6 s).** 30 minutes of work. Expected impact
  is low because Opus already refines to 0.1 s. Pilot only.
- **Leaderboard variant: "the last frame *before* tile S turned red".** Forces reading f-1,
  a boundary check.
- **Multi-pane terminals** (four log panes, question names one): not recommended. It is
  mostly a "where" puzzle with an easy trace.
- Anything new you come up with that follows the recipe above. The organizers' named
  starting points: planning several steps ahead, exploring an unfamiliar system, using
  tools, managing limited resources, adapting when new information arrives.

## 3. Hard rules for any new scene kind (docs/design.md)

1. Answers are random, drawn from the secret seed, from a space large enough that guessing
   succeeds ~0 of the time.
2. The event leaves a **lasting, monotonic trace** readable at 1/8 scale (counter, badge).
3. Answers are exact-match after `normalize_answer`.
4. Answer text is unreadable at ≤ 1/4 native width and readable from a native crop.
5. No hidden information. The instruction states the question, budget and costs, and every
   element named in the question is visible on every frame. `answer_format` must state the
   exact alphabet. (An earlier fairness bug: codes excluded I and O but the format didn't
   say so.)
6. The scripted reference must solve 100% using only the four MCP tools, well under 120k.
7. `render(frame)` is a pure function of (seed, frame). Use `random.Random(seed)` or
   `numpy.random.default_rng(seed)`, never global state. Use the bundled fonts.
8. Keep the trace font at least 44 px (an existing test in `tests/test_scenes.py` checks this).

## 4. How to add a scene kind (every file that must change)

1. `framehunt/scenes/<kind>.py`: the scene class. See `base.py` for the interface. Copy the
   structure of `toasts.py` or `leaderboard.py`, including `debug_info()`.
2. `framehunt/scenes/__init__.py`: add the kind to `KINDS` and `make_scene`.
3. `framehunt/reference.py`: a solver class (like `Toasts`, `Board`), plus
   `parse_question` and `overview_kind`, so the reference recognises the kind.
4. `framehunt/splits.py`: new ids in DEV (3xxx) and, when promoted, EVAL (7xxx). **Never
   edit existing entries.** Ranges used so far: logstream 70xx, dialogs 71xx, dashboard 72xx,
   toasts 34xx/74xx, leaderboard 35xx/75xx. Pick an unused range.
5. `scripts/summarize.py`: the kind list.
6. `tests/test_scenes.py`, `tests/test_antihack.py`, `tests/test_reference.py`: add the
   kind to the window, legibility, anti-hack and reference checks.
7. `framehunt/baselines/stop_early_client.py`: confirm the new `answer_format` string
   parses, so the stop-early baseline still submits a well-formed guess.
8. `train/make_sft.py`: a teacher solver in `SOLVERS` (only needed for the fine-tuning
   tiebreaker).
9. `docs/design.md` and `README.md`: describe the kind.

## 5. Validation protocol, cheapest first

1. **Eyeball it:** `uv run python scripts/preview_scene.py <kind> 7 --out /tmp/preview`
2. **Unit tests:** `uv run pytest -m "not reference"`
3. **Reference on many seeds (no Docker, no API):** `uv run pytest -m reference -s`.
   Pass: 100% correct on 200 seeds per new kind, max spend ≤ 70k.
4. **Window tests per item:**
   - At f-1, the answer is absent from the target location and a well-formed distractor is
     there.
   - The answer is present for the whole valid window.
   - Shortly after, it has moved or changed.
5. **Hacking checks:** nop and stop-early score 0. A "smart cheap" script that reads the
   newest candidate on the last frame scores ≤ 2%.
6. **Review:** `scripts/codex_review.sh <prompt-file> <out.md>` runs a headless Codex
   review (`gpt-6.1-sol`). Act on real design or correctness findings and repeat until the
   review is clean. Optional if Codex isn't installed locally.
7. **Oracle through Harbor (free):**
   ```bash
   uv run python scripts/build_tasks.py --split dev --only <id> <id> ... --out tasks/pilot
   harbor run -p tasks/pilot --job-name fh-pilot-oracle -o runs -n 3 -y -a oracle
   ```
8. **Opus pilot (costs API credits, about $0.15–0.20 per trial):** 6 DEV items per new kind.
   ```bash
   grep '^ANTHROPIC_API_KEY=' .env > /tmp/anthropic.env
   harbor run -p tasks/pilot --job-name fh-pilot-scenes -o runs -n 3 -r 2 \
     --retry-include AgentSetupTimeoutError -y -a opencode -m anthropic/claude-opus-5-5 \
     --ak version=1.19.5 --env-file /tmp/anthropic.env
   ```
   A kind passes when all three hold:
   - Opus is at most 60% correct.
   - At least half of the failures are "read too early" or "wrong row/target" when you
     compare `calls.jsonl` with the scene's `debug_info()`.
   - Opus succeeds at least once (this shows the kind is solvable, not broken).
9. **Full evaluation:** `scripts/eval.sh` (see README). A full Opus run of 30 items costs
   about $5 and takes 35–48 minutes at `-n 4`.

To analyse a failure, look at:

- `runs/<job>/<trial>/verifier/result.json`: answer, correct, pixels.
- `verifier/calls.jsonl`: every tool call with its cost.
- `agent/`: the agent's own log and final message.

Compare these with `make_scene(kind, derive_seed(split, item_id)).debug_info()`.

## 6. Setting up a new machine

Requirements: Docker (running), `uv`, Python 3.12+, Harbor 0.25.0 (`uv tool install harbor`).
Harbor installs OpenCode 1.19.5 inside the task container, so there is nothing to install
locally for it. Codex CLI is optional (review loop only).

```bash
git clone git@github.com:MrNaakka/bidy2.git && cd bidy2
uv sync
# copy the files listed below, then:
uv run pytest -m "not reference"                # sanity check
scripts/eval.sh --stage baselines               # builds tasks/eval, runs reference + baselines, no API spend
```

### Files that are NOT in git and must be copied by hand

Copy these from the hackathon host (`/home/user/bidy2/`) to the same paths in the clone:

| Path | Needed? | Why |
| --- | --- | --- |
| `.env` | **Required** for Opus runs | Holds `ANTHROPIC_API_KEY` (plus the Qwen and Verda keys). |
| `secrets/eval.salt`, `secrets/dev.salt`, `secrets/train.salt` | **Required** to reproduce the same items | Item seeds are HMAC(salt, item id). Without these files a build generates new salts, so `logstream-7008` etc. become different videos and the existing run logs no longer match the items. |
| `runs/` (118 MB) | **Strongly recommended** | Every Harbor run so far: 3 Opus runs, 3 Qwen runs, reference, baselines, the oracle pilot. This is the evidence for the failure analysis. One Opus run is also committed in `examples/harbor-runs/`. |
| `train/data/` (85 MB) | Only for fine-tuning | The SFT dataset. Can be regenerated with `train/make_sft.py` if `secrets/train.salt` is present. |

Not needed: `tasks/` (rebuilt by `scripts/build_tasks.py` / `eval.sh`), `.venv/`,
`train/out/` and `train/hf/` (CPU validation artefacts), `reviews/`, `jobs/`.

Never commit `.env` or `secrets/`. Don't post item answers or seeds publicly.

## 7. Coordination with the hackathon host (important)

The original host is still working on this repo:

- A thread there keeps committing to `main` (scenes agent, SkyPilot/Verda eval agent). It is
  about to run the Opus pilot on toasts and leaderboard.
- **Work on a branch** on the new machine and merge deliberately. Pull `main` often.
- Verda tooling:
  - It lives in `infra/verda/fh` (SkyPilot; see `infra/verda/README.md`). The hand-written
    `verda_cli.py` and its state files are gone.
  - Qwen evals run in parallel, one box per run:
    `scripts/eval.sh --stage agents --model qwen --fleet` (or `fh eval-up` + `fh eval-wait`).
    Each box mounts its own volume `fh-eval-<i>`
    (FIN-01, $20/month each, `fh eval-volumes`).
  - LoRA training: `fh train-up` (train/README.md).
  - SkyPilot keeps its state in `~/.sky` on the machine that launched a box. Drive each
    fleet or training run from a single machine.

## 8. Lessons from the host (so you don't repeat them)

- The host has 6 cores and 8 GB of RAM.
  - Running 16 Harbor trials at once crashed and rebooted it.
  - Later, parallel Docker builds, tests and a video encode pushed load to ~100 and killed
    the agent session twice.
  - The rule there: one Harbor job at a time, `-n 3` or `-n 4`, about 0.55 GB RAM per trial.
    A bigger machine can go higher; scale by RAM.
- Under load, Harbor trials can fail on infrastructure (`Could not resolve container for
  compose service 'main'`, `AgentSetupTimeoutError`). These are not policy failures. Rerun them.
- A full `build_tasks.py` build removes stale item directories, and `eval.sh` rebuilds when
  `tasks/<split>` doesn't match the split (fixes from reviews r15 and r16).
- Scored runs must use the API key, not a subscription login.
- Sonnet 4.5 failed in OpenCode with "Unexpected server error". Opus 5.5 works.
- Budget choice: at 500k pixels Opus solved everything while wasting most of the budget. At
  120k (about 2x what the reference needs), careless spending is punished. Keep 120k.

## 9. Open items

1. Opus pilot on toasts and leaderboard (the host is doing this). If either kind is too easy
   (Opus above 60%), it needs hardening or replacing.
2. Build and pilot `incident` and any further kinds. This is the main task of this handoff.
3. Full EVAL v2 runs: reference, baselines and three Opus runs. Then update the README
   results tables, which still show v1 and list run 3 as "in progress".
4. Tiebreaker: LoRA fine-tune of Qwen3.8-27B on reference trajectories (`train/README.md`).
   Base Qwen scores 4–8 of 30. Training has not produced a result yet.
