# FrameHunt design (v0)

FrameHunt is an RL environment for **active video sampling under a pixel budget**.
The agent gets a question about a long screen recording. It can't watch the video.
It can only ask the MCP server for individual frames, optionally cropped and downscaled,
and every returned pixel is charged against a fixed budget. Answering takes
planning: find *when* the event happened using cheap low-resolution looks, then
spend pixels on *one* sharp crop of *where* the answer is written.

## Hard rules for items

1. **Random answers.** Every answer is drawn from the item seed, from a space large
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

`scripts/build_tasks.py --split eval` writes `tasks/eval/<item_id>/` for a fixed list
of `(kind, seed)` items (the split's seeds are in `framehunt/splits.py`, fixed and
committed). Each task follows the step0 probe layout: `task.toml` with the MCP server
`framehunt` at `http://fh-mcp:8000/mcp`, `instruction.md` (question + rules + budget),
`environment/docker-compose.yaml` (the `fh-mcp` sidecar with `FH_KIND`, `FH_SEED`,
`VERIFY_TOKEN`), `environment/mcp/` (Dockerfile plus a copy of `framehunt/`),
`tests/test.sh` (reads `/_result` and `/_log`, writes reward + logs to `/logs/verifier`),
`solution/solve.sh` (runs the reference policy). The token is random per task and lives
only in the compose file and `tests/`, which the agent's container never sees.

### Reference run and baselines

- Reference: a scripted coarse-to-fine policy that uses only the agent's MCP tools.
  It takes one low-res overview, binary-searches the trace at ≤ 1/8 scale, then reads
  one native-res crop. Reading is pluggable: deterministic OCR from the bundled font
  (template matching) by default, or a model reader.
- Baselines: do-nothing (never submits) and stop-early (one look, then submits a
  well-formed guess).
- `scripts/eval.sh`: one command that builds tasks, then runs the reference, three
  agent runs, and both baselines through Harbor, and prints a results table.
