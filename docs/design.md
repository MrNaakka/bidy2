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
    answer_format: str   # shown to the agent, e.g. "6 characters, A-Z and 0-9"
    def render(self, frame: int) -> PIL.Image.Image: ...
        # RGB, meta.width × meta.height. Pure function of (seed, frame).
    def debug_info(self) -> dict: ...
        # key frames of events etc. For tests/analysis only; the server never exposes it.

# framehunt/scenes/__init__.py
KINDS: tuple[str, ...] = ("logstream", "dialogs", "dashboard")
def make_scene(kind: str, seed: int) -> Scene: ...

# framehunt/answers.py
def normalize_answer(s: str) -> str: ...
```

Fonts are bundled in `framehunt/assets/fonts/` (DejaVu, free license) so rendering is
identical on every machine. All randomness uses `random.Random(seed)` /
`numpy.random.default_rng(seed)`, never global state.

### MCP server `framehunt/server.py`

Configured by env vars: `FH_KIND`, `FH_SEED`, `FH_BUDGET_PX` (default 500000),
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
fixed, committed item ids in `framehunt/splits.py`. The scene seed of each item is
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
