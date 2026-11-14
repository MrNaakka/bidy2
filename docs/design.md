# FrameHunt design (v0)

FrameHunt is an RL environment for **active video sampling under a pixel budget**.
The agent gets a question about a long screen recording. It can't watch the video.
It can only ask the MCP server for individual frames, optionally cropped and downscaled,
and every returned pixel is charged against a fixed budget. Answering takes
planning: find *when* the event happened using cheap low-resolution looks, then
spend pixels on *one* sharp crop of *where* the answer is written.

## Hard rules for items

1. **Random answers.** Every answer is drawn from the item's (secret) seed, from a space large
   enough that guessing is ~0 (codes like `K7Q2ZP`, request ids, 3–5 digit numbers).
   Common sense must not help.
2. **Trace rule.** The event the question asks about is brief, but it leaves a
   **lasting, monotonic trace** on screen that is legible at low resolution (a counter,
   a badge, a tile that stays red). This makes efficient binary search possible:
   the information is always there for an agent that plans.
3. **Exact answers.** The answer comes from the generator, never from labels. Checking
   is an exact string match after `normalize_answer` (strip, uppercase, drop
   whitespace).
4. **Answer text needs resolution.** The answer text is unreadable when the frame is
   downscaled to ≤ 1/4 of native width, and readable from a native-resolution crop.
   The trace is readable at 1/8 native width (240×135 for a 1920×1080 video).
   "Unreadable" means for a reader that looks at the image (a vision model, a person,
   generic OCR; tested by OCR on the 1/4 crop blown back up). A scripted matched filter
   that knows the bundled font and the exact sub-pixel origin of the text *can* decode
   most 1/4-scale crops, on every kind. That only saves pixels: it still has to find the
   event frame and the right row or toast first, and pixels used don't change the
   reward. Fonts small enough to defeat it would also be unreadable at native.
5. **No hidden information.** The instruction states the budget, the tool costs and
   the question. Screen elements named in the question are visible on every frame.

## Synthetic scene kinds (first slice)

All 1920×1080, 10 fps, 3–8 minutes long (n_frames chosen from the seed).

- `logstream`: a terminal tailing a service log. Lines scroll (several per second),
  with levels INFO/DEBUG/WARN/ERROR and request ids. A status bar at the bottom shows
  `errors: N`, which counts ERROR lines printed so far. Question: "What is the request
  id on the K-th ERROR line?" An ERROR line stays visible ≥ 6 s before scrolling off.
  WARN lines with ids act as distractors.
- `dialogs`: a desktop app. Over time, modal error dialogs pop up with an error code,
  each for a few seconds, then close. A notification bell in the top bar shows a badge
  count of dialogs shown so far. Question: "What error code was shown in the K-th error
  dialog?" Dialog codes look alike (same prefix) so low-res reading fails.
- `dashboard`: a monitoring dashboard with service status tiles, KPI numbers that
  update every 2 s, and line charts. At some moment one named service tile turns red
  and stays red. Question: "What did KPI X read on the first frame where tile S is
  red?" KPI X holds its value for ≥ 1 s on both sides of that frame.

K and event times are random. Events sit in roughly the 10–90 % range of the video.

### Second slice: toasts and leaderboard; the EVAL v2 set

Opus 5.5 scored 3/10 on logstream but 8–10/10 on dialogs and dashboard (EVAL v1).
Logstream is hard because being slightly wrong still yields a confident wrong answer:
one frame early, the (K-1)-th ERROR line is on screen as a convincing distractor; a few
frames late, newer lines have scrolled in, so "the newest line" is wrong. In dialogs and
dashboard the answer holds for seconds on both sides of the event, sits at a fixed spot
and has no look-alike on screen, so an imprecise search still lands on it. The two new
kinds carry logstream's mechanism into other settings.

**EVAL v2 (final): logstream + dialogs + toasts, 10 each.** The setting is triage when a
screen recording is the only artifact: E2E test videos (Playwright, Cypress), users'
bug-report recordings, terminal recordings of a failed deploy. Logstream is the hard
case: verifying the event frame and finding the row need extra looks, which Opus
doesn't spend. Dialogs is the common E2E case. Toasts is the explicit **control**: it
has the same mechanism (a lasting counter pins the event, the answer is only briefly
where you'd look, look-alikes before and after), but verification is cheap and in the
same place (the badge right above the stack). Opus verifies there and succeeds (pilots
below). That fits the hypothesis that its logstream failures come from verification cost
rather than from the mechanism. It isn't proof: the kinds also differ in layout and
distractors, and only a matched intervention would isolate the cause. Leaderboard stays in DEV and TRAIN as a replication experiment
(Opus DEV pilot 2/6, below); dashboard is in EVAL_V1, DEV and TRAIN.

- `toasts`: the same desktop app as `dialogs`, with non-modal toast notifications
  stacking at the top right: Info (blue), Warning (amber), Error (red), each with a
  6-character code (all codes in a scene share a 2-character prefix). The newest toast
  takes the top slot and pushes the others down; at most 5 are visible; each lives
  4–6 s (or until 5 newer ones arrive). Toasts come in bursts (an incident: 0.2–0.9 s
  apart, mostly Errors) separated by calm stretches (1–3 s apart, mostly Info), so 2–4
  red Error toasts with the same prefix are often on screen together. The bell
  badge counts Error toasts only. Question: "What code was shown on the K-th Error
  toast?" Guarantees: K ≥ 2 and on frame f-1 the (K-1)-th Error toast is still on
  screen as the newest red toast (same prefix); the next toast arrives 0.2–0.6 s after
  f, so a few frames late the target is no longer on top (it stays visible lower in the
  stack, for at least 1 s), and the (K+1)-th Error arrives within 1.5 s, so soon after
  f "the newest red toast" is a look-alike, not the target. The level of every toast is visible at 1/8 from its 24-px colour stripe.
- `leaderboard`: an API error tracker. The "Endpoints by errors / min" table lists 10
  endpoints (all always visible), sorted by value, and refreshes every 0.5 s (5 frames):
  values change and rows move (an endpoint changes row on about a third of the
  refreshes). Endpoint names are sans-bold 18 (readable at 1/2); values
  are small monospace digits (mono 13), readable only at native resolution. Trace: a
  big `alerts fired: N` counter (top right). Question: "What errors/min value did
  endpoint E show on the first frame where the alerts counter reached K?" Every alert
  fires on a table refresh, so the value is shown for exactly [f, f+4] (0.5 s). E's
  value and E's row both differ on f-1 and on f+5, and no other row shows the same number on
  f. Unlike `dashboard` (KPI holds ≥ 1 s on both sides), there is no slack before f.

What the agent must decide in both: find the exact event frame on the cheap trace and
**verify it** before reading (one frame early is a confident wrong answer), then read on
**that same frame**, after **re-finding the moving target** there (the top toast is
only the target for 0.2–0.6 s; the endpoint's row moves every half second). The reference
does exactly this: binary search on the trace at ≤ 1/8, a 1/8 check of the top toast's
stripe (toasts) or a 1/4 look at the name column to find E's row (leaderboard), then one
native crop. It needs at most 41k (toasts) and 48k (leaderboard) pixels of the 120k.

Pilot and tuning (DEV, Opus 5.5, 6 items per kind): the first version (toasts: next
toast 0.3–1.5 s after f; leaderboard: 1 s refresh, sans-bold 22 names) scored 5/6 on
both kinds. Both failures were the targeted mechanism (read one frame early: the
(K-1)-th Error toast's code; E's previous value). The successes found the exact frame
with dense contact sheets and then read up to 5 frames late, which those wide windows
forgave, and found E's row with a cheap look. Tuned to the parameters above: a 0.2–0.6 s
top-slot window with a red successor half the time, a 0.5 s value window, and names that
need a deliberate 1/2-scale crop of the name column.

Re-pilot after tuning (6 fresh DEV items per kind): leaderboard 2/6. All 4 failures are
the targeted decisions: read 1 to 3 frames early (E's previous value, twice), read 9
frames late (E's next value), and one misread of the counter on 24-px contact-sheet
tiles that was never checked at a legible scale. Toasts 6/6: even when Opus read 1 to 7
frames late, it combined the badge count with the stack order and the red stripe to pick
the right toast lower in the stack (Error toasts are a minority, so usually only one or
two are visible). Toasts were then reworked: 4–6 s lives (a realistic default), and
Error-heavy bursts, so several red look-alikes surround the target and the (K+1)-th
Error follows within 1.5 s.

Toasts re-pilot after that rework (6 fresh DEV items, on Modal): 5/6, so toasts still
does not show a gap. Opus pinned f to within 1 to 5 frames on every item and usually
re-read the badge on the frame it read the toast from, which tells it which red toast is
the K-th even after the stack moves. The one failure is the targeted one: it read 3
frames late, the badge already showed K+1, and it took the top toast, which was the
(K+1)-th Error. The look-alikes only catch an agent that skips that same-frame check.

## Prototype: scanned documents (`scandoc`, DEV and TRAIN only)

The same tools, budget and reward on a different medium: a **long scanned PDF** instead
of a video. The setting is a reviewer who has only a scan of a company's annual report
(no text layer) and needs one figure from the notes to the financial statements. Frame
index = 0-based PDF page index (fps 1, so `t` is the page index); the instruction and
`info()` state this mapping plainly, and say that printed page numbers are the
document's own numbering, not PDF indexes. `scandoc` is not in EVAL.

**The document** (`framehunt/scenes/scandoc.py`, all planned from the seed): 150–400
pages of 1240×1754 (A4 at 150 dpi). Front matter numbered in lower-case roman numerals
(cover, optional notice and blank pages, the Contents page, 2–4 front sections);
main matter numbered from 1 (3–4 report sections, the consolidated statements, 8–11
numbered notes, auditor's report, shareholder information); 2–4 **unnumbered inserted
sheets** (correction slips, letters) that carry no running header and no page number and
do not count in the printed numbering. Every numbered page has a running header: section
or note title at the top left, printed page number at the top right. Each note after
note 1 has a table "Table n.1 <caption>" by country, printed **twice on facing pages**:
at 31 December of the report year and of the comparative year (same caption, rows and
columns; different amounts). All amounts in the document are distinct. Pages are
rendered with PIL from the bundled DejaVu fonts, then a seeded, deterministic scan
degradation: shift ≤ 4 px, skew ≤ 0.12°, a 3×3 softening blur, sensor noise from a
cached noise field, a per-page paper tint, margin speckles, and a red "RECEIVED" stamp
on the cover and inserts.

**Question:** "In the notes ..., the note titled T has a table C (amounts in EUR
thousand). What amount does that table give for row R in column K at 31 December Y?"
Answer: the amount (4–6 digits; the thousands space doesn't matter).

**Hard rules, adapted:**

1. *Random answers*: amounts are drawn from the seed, 4–6 digits, unique in the document.
2. *Navigation cues instead of a trace*: everything needed to navigate is legible at 1/4
   scale: the Contents entries (36 px), the running header and printed page number
   (36 px bold), the table caption, date line, column headers and row labels (36 px).
   Every numbered page shows its printed number; inserts visibly have none.
3. *Exact answers* from the generator.
4. *Answer needs resolution*: amounts are 12-px monospace digits. At 1/4 they are a
   blur (a matched filter with the bundled font, searching 1-px shifts, gets ~40 % of
   the digits and almost never the whole amount; `tests/test_scandoc.py`); a native
   crop reads exactly (they are also readable at 1/2).
5. *No hidden information*: the instruction states the page-index mapping, the cost
   rules and the question; the Contents, headers and dates are on the pages.

**Where the difficulty comes from (decisions, all on visible information):**

- *Navigation planning.* One full page at 1/4 costs 136k px, more than the 120k budget,
  and at 1/8 (34k) the cues are illegible. Skimming is unaffordable; the agent has to
  read the Contents (a narrow crop of it), then look only at small crops (header strips,
  one column).
- *Printed page ≠ PDF index.* The roman front matter shifts the numbering, and at least
  one unnumbered insert sits in the report sections and at least one in the notes before
  the target note (`inserts_before ≥ 2`). An offset calibrated once at the start of the
  main matter is wrong by the time you reach the note. Jumping and then reading the
  printed number on the landing page (36 px, top right) fixes it in a hop or two.
- *Look-alike neighbour.* The comparative-year table is on the facing page with the same
  header, caption, rows and columns. Reading the amount from it gives a well-formed wrong
  answer; the date line under the caption (36 px) tells them apart. Both orders occur
  (current year first in ~70 % of notes), and the asked year is the comparative one in
  ~1/3 of items.
- *Row and column.* The amount is 12 px; the row labels and column headers are 36 px
  and far to the left/top. The agent must locate the row and column at low resolution
  and then crop the right cell at native.

Not implemented (possible extensions): superseded clauses and two-hop references
(clause -> schedule). They would add the same kind of trap (a well-formed look-alike that
a check on visible text rules out).

**Reference** (`reference.Doc`, MCP tools only): header-title crops of PDF pages 1–3
at 1/4 to find the Contents; the Contents' title column at 1/4 to find the note's row;
that row's printed page number at 1/4; jump to PDF index = printed number and correct
by the printed number read on the landing page (1/4 header crop) until they agree;
read only the date line of the following pages until the asked year; row labels and
column headers at 1/4; one native crop of the amount (template OCR). Every look searches
small shifts because of the scan. 100 % on 100 random seeds; 51.6k px on average, at most
56.1k (`tests/test_reference.py`, `FH_REF_N_DOC`).

**Cheap shortcuts** (perfect readers, `tests/test_scandoc.py`): using the printed number
as the PDF index, or the front-matter offset only, then reading the first table found
within 3 pages: 0 %; the other year's table: 0 %; the first or last table of the document:
≤ 5 %. A slower "jump with the front-matter offset, take the first table found from there,
never check a header or date" strategy hits ~13 % (when it happens to land in the note
before the target and the asked year is printed first). Stop-early (a well-formed random
4–6 digit guess) and do-nothing score 0.

**Budget**: 120k, as for the videos, which is about 2.1× the reference's maximum.

## Interfaces

### Package `framehunt/` (scene generation)

```python
# framehunt/scenes/base.py
@dataclass(frozen=True)
class SceneMeta:
    width: int; height: int; fps: int; n_frames: int
    @property
    def duration_s(self) -> float: ...

class Scene(ABC):
    kind: str
    seed: int
    meta: SceneMeta
    question: str        # natural language; names on-screen elements
    answer: str          # canonical exact answer (already normalized)
    answer_format: str   # shown to the agent; states the exact alphabet, e.g.
                         # "6 characters: uppercase letters A-Z except I and O, and digits 0-9"
    def render(self, frame: int) -> PIL.Image.Image: ...
        # RGB, meta.width × meta.height. Pure function of (seed, frame).
    def debug_info(self) -> dict: ...
        # key frames of events etc. For tests/analysis only; the server never exposes it.

# framehunt/scenes/__init__.py
KINDS: tuple[str, ...] = ("logstream", "dialogs", "dashboard", "toasts", "leaderboard", "scandoc")
DOCUMENT_KINDS = ("scandoc",)   # a scanned PDF: frame index = page index, fps 1
def make_scene(kind: str, seed: int) -> Scene: ...

# framehunt/answers.py
def normalize_answer(s: str) -> str: ...
```

Fonts are bundled in `framehunt/assets/fonts/` (DejaVu, free license) so rendering is
identical on every machine. All randomness uses `random.Random(seed)` /
`numpy.random.default_rng(seed)`, never global state.

### MCP server `framehunt/server.py`

Configured by env vars: `FH_KIND`, `FH_SEED`, `FH_BUDGET_PX` (default 120000),
`VERIFY_TOKEN`, `FH_LOG_PATH` (optional). Streamable HTTP on `0.0.0.0:8000/mcp`.

Agent-facing tools (the agent's whole action space):

- `info()` → text: question, answer_format, width, height, fps, n_frames, duration_s,
  budget_total, budget_left, cost rules.
- `view(t, x=0, y=0, w=None, h=None, out_width=320)` → image + one text line
  (actual frame time, frame index, output size, cost, budget_left). `t` in seconds is
  snapped to the nearest frame. The crop (x, y, w, h) is in native pixels and
  clipped to the frame. Output is resized to `out_width` wide, keeping aspect ratio.
  No upscaling: `out_width ≤ w`. **Cost = output width × output height.**
- `contact_sheet(t_start, t_end, n, x=0, y=0, w=None, h=None, out_width=160)` → one grid
  image of `n` evenly spaced frames (2 ≤ n ≤ 16) plus a text list of frame times.
  Cost = n × per-tile output pixels. Thin separators between tiles are free.
- `submit(answer)` → records the answer. Only the first submission counts. Returns
  no correctness feedback.

A request that costs more than `budget_left` is refused with an error and charges
nothing. Budget accounting is under a lock, so concurrent calls can't overspend.

Hidden HTTP routes (token-protected, for the verifier only):
`/_result?token=` → JSON `{submitted, answer, correct, reward, pixels_used, n_calls}`,
`/_log?token=` → JSONL of every tool call (args, cost, budget_left, timestamp).
`/_health` is public and returns `ok`.

### Reward

`reward = 1.0` if the first submitted answer matches exactly, else `0.0`. Not
submitting gives 0. The budget is a hard constraint, not a reward term. Keeping the
reward binary leaves no term to farm.

### Harbor tasks

`scripts/build_tasks.py --split eval` writes `tasks/eval/<item_id>/` for the split's
fixed, committed item ids in `framehunt/splits.py` (`--split eval --v1` builds the first
EVAL set, `EVAL_V1`, into `tasks/eval-v1/`). The scene seed of each item is
secret (see "Item secrecy"). Each task follows the step0 probe layout: `task.toml`
with the MCP server `framehunt` at `http://fh-mcp:8000/mcp`, `instruction.md`
(question + rules + budget), `environment/docker-compose.yaml` (the `fh-mcp` sidecar
with `FH_KIND`, `FH_SEED` = the derived secret seed, `VERIFY_TOKEN`),
`environment/mcp/` (Dockerfile plus a copy of `framehunt/`), `tests/` (the verifier
image, see below), `solution/solve.sh` (runs the reference policy). The token is
random per task and lives only in the compose file, which the agent's container
never sees.

**The verifier never runs in the agent's container.** The agent is root in its own
container, so anything there (binaries, interpreters, files under `/logs`) can be
replaced, and a verifier running there could be made to print any reward. So
`task.toml` sets `[verifier] environment_mode = "separate"`, and the chain is:

1. The agent phase ends. Harbor stops the agent's (`main`) container, which also
   kills anything the agent left running.
2. A `[[verifier.collect]]` hook runs `python -m framehunt.evidence /fh-evidence` in
   the `fh-mcp` sidecar, which is still running. It reads `/_result` and `/_log` from
   the server process with the sidecar's own `VERIFY_TOKEN` and writes `result.json`
   and `calls.jsonl`. The agent can only reach the sidecar through the four MCP tools.
3. Harbor collects `/fh-evidence` from the sidecar (an `[[artifacts]]` entry with
   `service = "fh-mcp"`) and recreates it at `/fh-evidence` in a fresh verifier
   container built from `tests/Dockerfile` (python:3.12-slim + `test.sh` +
   `framehunt/verify.py`, no network).
4. `tests/test.sh` runs `verify.py`: reward 1 only if the sidecar's result has
   `submitted: true` and `correct: true` (that is, the first submission was right).
   A missing or malformed dump gives 0. It copies `result.json` and `calls.jsonl` to
   `/logs/verifier`.

Nothing written by the agent's container is read by the verifier. (The agent's
`/logs/artifacts` is also copied into the verifier container, at `/logs/artifacts`,
and ignored. Harbor clears `/logs/verifier`, including reward files, before the
verifier runs.)

### Item secrecy

The generators are public, and so is this repository. If the committed item list
determined the scenes, a policy could render every eval item offline, fingerprint
the item from free `info()` facts (every eval item has a different `n_frames`), and
submit the answer without looking at a single pixel. A review did exactly that and
scored 30/30.

So the **generator is public, the instances are private.** Each item keeps a public
id (`logstream-7001`), but the scene is rendered from

    seed = int.from_bytes(HMAC-SHA256(salt, f"{split}:{item_id}")[:8], "big")

(`framehunt.splits.derive_seed`, the only place seeds come from; the task builder,
tests and any training-data generator use it). The salt is per split and secret:
`FH_SALT_<SPLIT>` (e.g. `FH_SALT_EVAL`) or the git-ignored `secrets/<split>.salt`.
If neither exists, `build_tasks.py` generates one with `secrets.token_hex(32)` and
saves it. Consequences:

- Our reruns in this checkout reproduce the exact same items (same salt file).
- A fresh checkout (the judges') gets a new salt, so new items that nobody,
  including us, has seen. Item difficulty is the same by construction: same
  generators, same distributions over kinds, K and event times.
- The derived seed only goes into the sidecar's compose environment. The agent
  doesn't see the compose file, `task.toml` metadata has no seed, and
  `instruction.md` holds only the question and the video facts. With a 64-bit secret
  seed, `info()` fingerprints (`n_frames` etc.) identify nothing useful: there is no
  table to look them up in.

Published run logs (`examples/harbor-runs/`) contain answers for the items of the salt
they were run with. So after publishing EVAL v1 logs, EVAL v2 got a fresh salt in this
checkout; the old one is kept privately as `secrets/eval-v1.salt`, and
`build_tasks.py --split eval --v1` uses it to rebuild the v1 items. Never publish logs of
the salt you are still evaluating with.

This doesn't hide anything the task needs. The agent gets exactly the same
`info()`, instruction and tools as before. Rule 5 ("no hidden information") is about
what's needed to answer, and every answer is on screen. The salt only stops
answering *without looking*.

### Reference run and baselines

- Reference: a scripted coarse-to-fine policy that uses only the agent's MCP tools.
  It takes one low-res overview, binary-searches the trace at ≤ 1/8 scale, then reads
  one native-res crop. Reading is pluggable: deterministic OCR from the bundled font
  (template matching) by default, or a model reader.
- Baselines: do-nothing (never submits) and stop-early (one look, then submits a
  well-formed guess).
- `scripts/eval.sh`: one command that builds tasks, then runs the reference, three
  agent runs, and both baselines through Harbor, and prints a results table.

## Budget choice (120k pixels)

Pilot runs with Opus 5.5 (2026-11-14) set this number. At 500k pixels Opus solved 3 of 3 items while spending 60–97 % of the budget. Its first 480-wide overview frame alone cost 130k, and none of that waste was ever punished. The reference needs at most 58.6k on any eval item. At 120k, about 2× the reference, careless spending stops being free. One full native frame (2.07 Mpx) is far out of reach, but there is ample room for a planned coarse-to-fine search. In the 120k pilot Opus scored 5 of 6. The failure was a grid-indexing slip on a contact sheet, never checked before it spent the remaining budget.
