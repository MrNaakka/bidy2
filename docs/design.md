# FrameHunt design

> Results (EVAL v2, 40 items: reference 1.000, Opus 5.5 0.650, Qwen3.8-27B 0.250 base and
> 0.617 after our SFT) are in [README.md](../README.md) and [docs/results/](results/tables.md).
> Pilot numbers quoted below are from DEV items during design and tuning.

FrameHunt is an RL environment for active video sampling under a pixel budget.

The agent gets a question about a long screen recording. It can't watch the video.
It can only ask the MCP server for single frames, optionally cropped and downscaled,
and every pixel it gets back counts against a fixed budget.

To answer, the agent has to plan. It uses cheap low-resolution looks to find *when* the
event happened, then spends pixels on *one* sharp crop of *where* the answer is written.

## Item sets

EVAL v2 has 40 items, 10 per kind:

| Kind | Medium | Role in EVAL v2 |
| --- | --- | --- |
| `logstream` | video | the hard case |
| `dialogs` | video | the common E2E case |
| `toasts` | video | the control |
| `scandoc` | scanned PDF | added later |

Two more video kinds are not in EVAL v2. `dashboard` is in EVAL_V1, DEV and TRAIN.
`leaderboard` is in DEV and TRAIN, as a replication experiment.

The video kinds were built in two rounds: `logstream`, `dialogs` and `dashboard` first,
then `toasts` and `leaderboard`.

## Hard rules for items

1. **Random answers.** Every answer is drawn from the item's (secret) seed, from a space
   large enough that guessing is ~0 (codes like `K7Q2ZP`, request ids, 3–5 digit
   numbers). Common sense must not help.
2. **Trace rule.** The event the question asks about is brief, but it leaves a
   **lasting, monotonic trace** on screen that is legible at low resolution (a counter,
   a badge, a tile that stays red). This makes efficient binary search possible: the
   information is always there for an agent that plans.
3. **Exact answers.** The answer comes from the generator, never from labels. Checking
   is an exact string match after `normalize_answer` (strip, uppercase, drop
   whitespace).
4. **Answer text needs resolution.** The answer text is unreadable when the frame is
   downscaled to ≤ 1/4 of native width, and readable from a native-resolution crop.
   The trace is readable at 1/8 native width (240×135 for a 1920×1080 video).
   - "Unreadable" means unreadable for a reader that looks at the image: a vision model,
     a person, or generic OCR. We test it by running OCR on the 1/4 crop scaled back up.
   - A scripted matched filter that knows the bundled font and the exact sub-pixel
     origin of the text *can* decode most 1/4-scale crops, on every kind. That only
     saves pixels. It still has to find the event frame and the right row or toast
     first, and pixels used don't change the reward. Fonts small enough to defeat it
     would also be unreadable at native.
5. **No hidden information.** The instruction states the budget, the tool costs and
   the question. Screen elements named in the question are visible on every frame.

## Video scenes

All 1920×1080, 10 fps, 3–8 minutes long (n_frames chosen from the seed).

### First round: logstream, dialogs, dashboard

In these three kinds, K and event times are random, and events fall roughly in the
10–90 % range of the video.

- `logstream`: a terminal tailing a service log. Lines scroll (several per second),
  with levels INFO/DEBUG/WARN/ERROR and request ids. A status bar at the bottom shows
  `errors: N`, the number of ERROR lines printed so far. Question: "What is the request
  id on the K-th ERROR line?" An ERROR line stays visible ≥ 6 s before scrolling off.
  WARN lines with ids act as distractors.
- `dialogs`: a desktop app. Modal error dialogs with an error code pop up over time,
  stay a few seconds, then close. A notification bell in the top bar shows a badge
  count of dialogs shown so far. Question: "What error code was shown in the K-th error
  dialog?" Dialog codes look alike (same prefix), so low-res reading fails.
- `dashboard`: a monitoring dashboard with service status tiles, KPI numbers that
  update every 2 s, and line charts. At some moment one named service tile turns red
  and stays red. Question: "What did KPI X read on the first frame where tile S is
  red?" KPI X holds its value for ≥ 1 s on both sides of that frame.

### Why logstream is hard

In EVAL v1, Opus 5.5 scored 3/10 on logstream but 8–10/10 on dialogs and dashboard.
On logstream, being slightly off still gives a confident wrong answer:

- One frame early, the (K-1)-th ERROR line is on screen as a convincing distractor.
- A few frames late, newer lines have scrolled in, so "the newest line" is wrong.

In dialogs and dashboard the answer holds for seconds on both sides of the event, sits
at a fixed spot and has no look-alike on screen. An imprecise search still lands on it.
The second-round kinds, toasts and leaderboard, carry logstream's mechanism into other
settings.

### What EVAL v2 tests

The setting is triage when a screen recording is the only artifact: E2E test videos
(Playwright, Cypress), users' bug-report recordings, terminal recordings of a failed
deploy.

- **Logstream** is the hard case. Verifying the event frame and finding the row take
  extra looks, which Opus doesn't spend.
- **Dialogs** is the common E2E case.
- **Toasts** is the explicit control. It has the same mechanism (a lasting counter pins
  the event, the answer is only briefly where you'd look, look-alikes appear before and
  after), but verification is cheap and in the same place: the badge right above the
  stack. Opus verifies there and succeeds (pilots below; on EVAL, toasts 0.80 vs
  logstream 0.43 over three runs).
- **Scandoc** is a scanned PDF, added later; see
  [Scanned documents](#scanned-documents-scandoc).

The toasts result fits the hypothesis that Opus's logstream failures come from
verification cost rather than from the mechanism. It isn't proof: the kinds also differ
in layout and distractors, and only a matched intervention would isolate the cause.

Leaderboard stays in DEV and TRAIN as a replication experiment (Opus DEV pilot 2/6,
below).

### Second round: toasts and leaderboard

In the guarantees below, f is the event frame.

**`toasts`**: the same desktop app as `dialogs`, with non-modal toast notifications
stacking at the top right.

- Levels: Info (blue), Warning (amber), Error (red). Each toast has a 6-character code;
  all codes in a scene share a 2-character prefix. The level of every toast is visible
  at 1/8 from its 24-px colour stripe.
- Stack: the newest toast takes the top slot and pushes the others down. At most 5 are
  visible. Each lives 4–6 s (or until 5 newer ones arrive).
- Timing: toasts come in bursts (an incident: 0.2–0.9 s apart, mostly Errors) separated
  by calm stretches (1–3 s apart, mostly Info). So 2–4 red Error toasts with the same
  prefix are often on screen together.
- Trace: the bell badge counts Error toasts only.
- Question: "What code was shown on the K-th Error toast?"
- Guarantees:
  - K ≥ 2, and on frame f-1 the (K-1)-th Error toast is still on screen as the newest
    red toast (same prefix).
  - The next toast arrives 0.2–0.6 s after f, so a few frames late the target is no
    longer on top. It stays visible lower in the stack for at least 1 s.
  - The (K+1)-th Error arrives within 1.5 s, so soon after f "the newest red toast" is a
    look-alike, not the target.

**`leaderboard`**: an API error tracker.

- Table: "Endpoints by errors / min" lists 10 endpoints (all always visible), sorted by
  value. It refreshes every 0.5 s (5 frames): values change and rows move (an endpoint
  changes row on about a third of the refreshes).
- Text sizes: endpoint names are sans-bold 18 (readable at 1/2). Values are small
  monospace digits (mono 13), readable only at native resolution.
- Trace: a big `alerts fired: N` counter (top right).
- Question: "What errors/min value did endpoint E show on the first frame where the
  alerts counter reached K?"
- Guarantees: every alert fires on a table refresh, so the value is shown for exactly
  [f, f+4] (0.5 s). E's value and E's row both differ on f-1 and on f+5, and no other
  row shows the same number on f. Unlike `dashboard` (KPI holds ≥ 1 s on both sides),
  there is no slack before f.

### What the agent must decide

In both kinds the agent has to:

1. Find the exact event frame on the cheap trace and **verify it** before reading. One
   frame early gives a confident wrong answer.
2. Read on **that same frame**, after **re-finding the moving target** there. The top
   toast is the target for only 0.2–0.6 s; the endpoint's row moves every half second.

The reference does exactly this: binary search on the trace at ≤ 1/8, then a 1/8 check
of the top toast's stripe (toasts) or a 1/4 look at the name column to find E's row
(leaderboard), then one native crop. It needs at most 41k (toasts) and 48k
(leaderboard) pixels of the 120k.

### Pilots and tuning (DEV, Opus 5.5)

| Round | Items | toasts | leaderboard |
| --- | --- | --- | --- |
| First version | 6 per kind | 5/6 | 5/6 |
| Re-pilot after tuning | 6 fresh per kind | 6/6 | 2/6 |
| Toasts re-pilot after rework (on Modal) | 6 fresh, toasts only | 5/6 | – |

**First version.** Toasts had the next toast 0.3–1.5 s after f; leaderboard had a 1 s
refresh and sans-bold 22 names. Both failures were the targeted mechanism: reading one
frame early (the (K-1)-th Error toast's code; E's previous value). The successes found
the exact frame with dense contact sheets, then read up to 5 frames late, which those
wide windows forgave. They found E's row with a cheap look.

We tuned to the parameters above: a 0.2–0.6 s top-slot window with a red successor half
the time, a 0.5 s value window, and names that need a deliberate 1/2-scale crop of the
name column.

**Re-pilot after tuning.** Leaderboard 2/6. All 4 failures are the targeted decisions:

- read 1 to 3 frames early (E's previous value), twice;
- read 9 frames late (E's next value);
- misread the counter on 24-px contact-sheet tiles and never checked it at a legible
  scale.

Toasts 6/6. Even when Opus read 1 to 7 frames late, it combined the badge count with
the stack order and the red stripe to pick the right toast lower in the stack. (In that
version Error toasts were a minority, so usually only one or two were visible.)

So toasts were reworked: 4–6 s lives (a realistic default) and Error-heavy bursts.
Several red look-alikes now surround the target, and the (K+1)-th Error follows within
1.5 s.

**Toasts re-pilot after the rework.** 5/6, so toasts still shows no gap. Opus pinned f
to within 1 to 5 frames on every item. It usually re-read the badge on the frame it read
the toast from, which tells it which red toast is the K-th even after the stack moves.

The one failure is the targeted one: it read 3 frames late, the badge already showed
K+1, and it took the top toast, which was the (K+1)-th Error. The look-alikes only catch
an agent that skips that same-frame check.

## Scanned documents (`scandoc`)

Scandoc uses the same tools, budget and reward on a different medium: a **long scanned
PDF** instead of a video. The setting is a reviewer who has only a scan of a company's
annual report (no text layer) and needs one figure from the notes to the financial
statements.

Frame index = 0-based PDF page index (fps 1, so `t` is the page index). The instruction
and `info()` state this mapping plainly, and say that printed page numbers are the
document's own numbering, not PDF indexes.

### Status in EVAL v2

`scandoc` started as a DEV/TRAIN prototype and is now an EVAL v2 kind: 10 items
`scandoc-7601..7610` (fresh public seeds, the same EVAL salt). They were added after
the 30 video items had been run. The addition changes nothing about those 30:

- The scandoc instruction, `info()` text and tool docstrings are separate templates
  selected by `DOCUMENT_KINDS`, and the video code paths are unchanged.
- Before and after the merge, the question/answer/format/meta hashes, rendered frames,
  `instruction.md`, `task.toml`, `info()` text, MCP tool list and the reference's full
  trace and pixel use were identical for all 30.

EVAL results for scandoc (reference 10/10, Opus 5.5 0.60 / 0.50 / 0.40 over three runs)
are in the [README](../README.md) and [docs/results/](results/tables.md). The numbers
below are from the DEV pilot.

### The document

`framehunt/scenes/scandoc.py` plans everything from the seed.

- Pages: 150–400 pages of 1240×1754 (A4 at 150 dpi).
- Front matter, numbered in lower-case roman numerals: cover, optional notice and blank
  pages, the Contents page, 2–4 front sections.
- Main matter, numbered from 1: 3–4 report sections, the consolidated statements, 8–11
  numbered notes, auditor's report, shareholder information.
- 2–4 **unnumbered inserted sheets** (correction slips, letters). They carry no running
  header and no page number and do not count in the printed numbering.
- Running header on every numbered page: section or note title at the top left, printed
  page number at the top right.
- Tables: each note after note 1 has a table "Table n.1 <caption>" by country, printed
  **twice on facing pages**: at 31 December of the report year and of the comparative
  year (same caption, rows and columns; different amounts). All amounts printed in the
  document, Total rows included, are distinct.
- Rendering: PIL with the bundled DejaVu fonts, then a seeded, deterministic scan
  degradation: shift ≤ 4 px, skew ≤ 0.12°, a 3×3 softening blur, sensor noise from a
  cached noise field, a per-page paper tint, margin speckles, and a red "RECEIVED" stamp
  on the cover and inserts.

### Question

"In the notes ..., the note titled T has a table C (amounts in EUR thousand). What
amount does that table give for row R in column K at 31 December Y?"

Answer: the amount (4–6 digits; the thousands space doesn't matter).

### Hard rules, adapted

1. *Random answers*: amounts are drawn from the seed, 4–6 digits, unique in the
   document.
2. *Navigation cues instead of a trace*: everything needed to navigate is legible at
   1/4 scale: the Contents entries (36 px), the running header and printed page number
   (36 px bold), the table caption, date line, column headers and row labels (36 px).
   Every numbered page shows its printed number; inserts visibly have none.
3. *Exact answers* from the generator.
4. *Answer needs resolution*: amounts are 12-px monospace digits. At 1/4 they are a
   blur: a matched filter with the bundled font, searching 1-px shifts, gets ~40 % of
   the digits and almost never the whole amount (`tests/test_scandoc.py`). A native
   crop reads exactly (they are also readable at 1/2).
5. *No hidden information*: the instruction states the page-index mapping, the cost
   rules and the question; the Contents, headers and dates are on the pages.

### Where the difficulty comes from

All of these are decisions made on visible information.

- **Navigation planning.** One full page at 1/4 costs 136k px, more than the 120k
  budget, and at 1/8 (34k) the cues are illegible. Skimming is unaffordable. The agent
  has to read the Contents (a narrow crop of it), then look only at small crops (header
  strips, one column).
- **Printed page ≠ PDF index.** The roman front matter shifts the numbering. At least
  one unnumbered insert sits in the report sections and at least one in the notes
  before the target note (`inserts_before ≥ 2`). An offset calibrated once at the start
  of the main matter is wrong by the time you reach the note. Jumping and then reading
  the printed number on the landing page (36 px, top right) fixes it in a hop or two.
- **Look-alike neighbour.** The comparative-year table is on the facing page with the
  same header, caption, rows and columns. Reading the amount from it gives a
  well-formed wrong answer; the date line under the caption (36 px) tells them apart.
  Both orders occur (current year first in ~70 % of notes), and the asked year is the
  comparative one in ~1/3 of items.
- **Row and column.** The amount is 12 px; the row labels and column headers are 36 px
  and far to the left/top. The agent must locate the row and column at low resolution
  and then crop the right cell at native.

Not implemented (possible extensions): superseded clauses and two-hop references
(clause -> schedule). They would add the same kind of trap: a well-formed look-alike
that a check on visible text rules out.

### Reference

`reference.Doc` uses only the MCP tools:

1. Header-title crops of PDF pages 1–3 at 1/4 to find the Contents.
2. The Contents' title column at 1/4 to find the note's row, then that row's printed
   page number at 1/4.
3. Jump to PDF index = printed number, and correct by the printed number read on the
   landing page (1/4 header crop) until they agree.
4. Read only the date line of the following pages until the asked year.
5. Row labels and column headers at 1/4.
6. One native crop of the amount (template OCR).

Every look searches small shifts because of the scan. The reference scores 100 % on 100
random seeds and uses 51.6k px on average, at most 56.1k (`tests/test_reference.py`,
`FH_REF_N_DOC`).

### Cheap shortcuts

Scores with perfect readers (`tests/test_scandoc.py`):

| Shortcut | Score |
| --- | --- |
| Use the printed number as the PDF index, or the front-matter offset only, then read the first table found within 3 pages | 0 % |
| Read the other year's table | 0 % |
| Read the first or last table of the document | ≤ 5 % |
| Slower: jump with the front-matter offset, take the first table found from there, never check a header or date | ~13 % |

The ~13 % comes from cases where the jump happens to land in the note before the
target and the asked year is printed first. Stop-early (a well-formed random 4–6 digit
guess) and do-nothing score 0.

### Budget

120k, as for the videos, which is about 2.1× the reference's maximum.

### DEV pilot

Modal, 8 items `scandoc-3601..3608`, Opus 5.5 via OpenCode, one run. The reference
scored 8/8, Opus **3/8**. The failures:

- **3601** read the facing comparative-year table (2019 instead of 2020), from a
  0.4-scale crop.
- **3602** found the right table but the wrong row (Latvia for Netherlands); the amount
  was read at 1/2.
- **3607** found the right table and row, but misread the digits from a 1/3-scale crop
  when the budget was nearly gone.
- **3605** first assumed printed = PDF page, searched the wrong stretch, and ran out of
  budget before the note (placeholder guess).
- **3608** spent 46k on one unreadable full view of PDF page 2 (the contents is page 1),
  never read the contents, skimmed headers with contact sheets, and ran out of budget
  (guess).

6 of 8 trials read the contents page (3604 skipped it and still won by skimming
headers). None ended with more than 12k left; the reference uses 50–56k.

The main mistakes are budget planning (broad looks before narrow ones, then amounts read
below native) and skipping the date or row check on the look-alike table. Opus was
≤ 6/8, so no tuning was done.

## Budget choice (120k pixels)

Pilot runs with Opus 5.5 (2026-11-14) set this number.

At 500k pixels Opus solved 3 of 3 items while spending 60–97 % of the budget. Its first
480-wide overview frame alone cost 130k, and none of that waste was ever punished.

The reference needs at most 58.6k on any eval item. At 120k, about 2× the reference,
careless spending stops being free. One full native frame (2.07 Mpx) is far out of
reach, but there is ample room for a planned coarse-to-fine search.

In the 120k pilot Opus scored 5 of 6. The failure was a grid-indexing slip on a contact
sheet, never checked before it spent the remaining budget.

## Implementation

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

Fonts are bundled in `framehunt/assets/fonts/` (DejaVu, free license), so rendering is
identical on every machine. All randomness uses `random.Random(seed)` /
`numpy.random.default_rng(seed)`, never global state.

### MCP server `framehunt/server.py`

Configured by env vars: `FH_KIND`, `FH_SEED`, `FH_BUDGET_PX` (default 120000),
`VERIFY_TOKEN`, `FH_LOG_PATH` (optional). Streamable HTTP on `0.0.0.0:8000/mcp`.

Agent-facing tools (the agent's whole action space):

- `info()` → text: question, answer_format, width, height, fps, n_frames, duration_s,
  budget_total, budget_left, cost rules.
- `view(t, x=0, y=0, w=None, h=None, out_width=320)` → image + one text line
  (actual frame time, frame index, output size, cost, budget_left).
  - `t` in seconds is snapped to the nearest frame.
  - The crop (x, y, w, h) is in native pixels and clipped to the frame.
  - Output is resized to `out_width` wide, keeping aspect ratio. No upscaling:
    `out_width ≤ w`.
  - **Cost = output width × output height.**
- `contact_sheet(t_start, t_end, n, x=0, y=0, w=None, h=None, out_width=160)` → one grid
  image of `n` evenly spaced frames (2 ≤ n ≤ 16) plus a text list of frame times.
  Cost = n × per-tile output pixels. Thin separators between tiles are free.
- `submit(answer)` → records the answer. Only the first submission counts. Returns
  no correctness feedback.

A request that costs more than `budget_left` is refused with an error and charges
nothing. Budget accounting is under a lock, so concurrent calls can't overspend.

Hidden HTTP routes (token-protected, for the verifier only):

- `/_result?token=` → JSON `{submitted, answer, correct, reward, pixels_used, n_calls}`
- `/_log?token=` → JSONL of every tool call (args, cost, budget_left, timestamp)

`/_health` is public and returns `ok`.

### Reward

`reward = 1.0` if the first submitted answer matches exactly, else `0.0`. Not
submitting gives 0. The budget is a hard constraint, not a reward term. A binary reward
leaves no term to farm.

### Harbor tasks

`scripts/build_tasks.py --split eval` writes `tasks/eval/<item_id>/` for the split's
fixed, committed item ids in `framehunt/splits.py`. `--split eval --v1` builds the
first EVAL set, `EVAL_V1`, into `tasks/eval-v1/`. The scene seed of each item is secret
(see [Item secrecy](#item-secrecy)).

Each task follows the step0 probe layout:

- `task.toml` with the MCP server `framehunt` at `http://fh-mcp:8000/mcp`
- `instruction.md` (question + rules + budget)
- `environment/docker-compose.yaml`: the `fh-mcp` sidecar with `FH_KIND`, `FH_SEED` =
  the derived secret seed, `VERIFY_TOKEN`
- `environment/mcp/`: Dockerfile plus a copy of `framehunt/`
- `tests/`: the verifier image, see below
- `solution/solve.sh`: runs the reference policy

The token is random per task and lives only in the compose file, which the agent's
container never sees.

### Verifier isolation

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

Nothing written by the agent's container is read by the verifier. The agent's
`/logs/artifacts` is also copied into the verifier container, at `/logs/artifacts`,
and ignored. Harbor clears `/logs/verifier`, including reward files, before the
verifier runs.

### Item secrecy

The generators are public, and so is this repository. If the committed item list
determined the scenes, a policy could render every eval item offline, fingerprint
the item from free `info()` facts (every eval item has a different `n_frames`), and
submit the answer without looking at a single pixel. A review did exactly that and
scored 30/30.

So the **generator is public, the instances are private.** Each item keeps a public
id (`logstream-7001`), but the scene is rendered from

    seed = int.from_bytes(HMAC-SHA256(salt, f"{split}:{item_id}")[:8], "big")

`framehunt.splits.derive_seed` is the only place seeds come from; the task builder,
tests and any training-data generator use it. The salt is per split and secret:
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

This doesn't hide anything the task needs. The agent gets exactly the same `info()`,
instruction and tools as before. Rule 5 ("no hidden information") is about what's
needed to answer, and every answer is on screen. The salt only stops answering
*without looking*.

#### Salt rotations

Published run logs (`examples/harbor-runs/`) contain answers for the items of the salt
they were run with. So after the EVAL v1 logs were published, EVAL v2 got a fresh salt
in this checkout. The old one is kept privately as `secrets/eval-v1.salt`, and
`build_tasks.py --split eval --v1` uses it to rebuild the v1 items. Never publish logs
of the salt you are still evaluating with.

The EVAL v2 salt was rotated once more, after a results summary that listed submitted
answers was accidentally pushed (`scripts/summarize.py` no longer writes answers).
Opus's 30 video-item runs were made under the previous v2 salt; everything else
(reference, baselines, Opus scandoc, all Qwen runs) under the current one. Same
generators and item ids, different instances.

### Reference run and baselines

- Reference: a scripted coarse-to-fine policy that uses only the agent's MCP tools.
  It takes one low-res overview, binary-searches the trace at ≤ 1/8 scale, then reads
  one native-res crop. Reading is pluggable: deterministic OCR from the bundled font
  (template matching) by default, or a model reader.
- Baselines: do-nothing (never submits) and stop-early (one look, then submits a
  well-formed guess).
- `scripts/eval.sh`: one command that builds tasks, then runs the reference, three
  agent runs, and both baselines through Harbor, and prints a results table.
