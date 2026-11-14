"""FrameHunt MCP server: active video sampling under a pixel budget.

The agent's whole action space is four MCP tools: ``info``, ``view``,
``contact_sheet`` and ``submit``. Every returned pixel is charged against a fixed
budget. Hidden, token-protected HTTP routes let the verifier read the result and the
call log; ``/_health`` is public.

The logic lives in :class:`FrameHuntEnv` (plain Python, testable without HTTP).
:func:`build_mcp` wires an env into a FastMCP app.

Env vars: FH_KIND, FH_SEED, FH_BUDGET_PX (default 500000), VERIFY_TOKEN,
FH_LOG_PATH (optional JSONL copy of the call log).
"""

from __future__ import annotations

import hmac
import io
import json
import math
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from PIL import Image as PILImage

DEFAULT_BUDGET_PX = 500_000
MIN_OUT_WIDTH = 8
MIN_SHEET_N = 2
MAX_SHEET_N = 16
SHEET_SEPARATOR_PX = 2
SHEET_SEPARATOR_RGB = (128, 128, 128)
MAX_ANSWER_CHARS = 256
RESAMPLE = PILImage.Resampling.LANCZOS
FRAME_CACHE_SIZE = 48


class FrameHuntError(Exception):
    """A refused tool call. Nothing is charged."""


def _num(name: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise FrameHuntError(f"{name} must be a number, got {v!r}")
    try:
        f = float(v)
    except OverflowError:
        f = math.inf
    if not math.isfinite(f):
        raise FrameHuntError(f"{name} must be a finite number, got {v!r}")
    return f


def _jsonable(v: Any) -> Any:
    if isinstance(v, float) and not math.isfinite(v):
        return repr(v)
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)[:MAX_ANSWER_CHARS]


def _int(name: str, v: Any) -> int:
    if isinstance(v, bool):
        raise FrameHuntError(f"{name} must be an integer, got {v!r}")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        return int(v)
    raise FrameHuntError(f"{name} must be an integer, got {v!r}")


class FrameHuntEnv:
    """Budget accounting, argument validation, rendering and the result record."""

    def __init__(
        self,
        scene: Any,
        budget_px: int = DEFAULT_BUDGET_PX,
        verify_token: str = "",
        log_path: str | None = None,
        normalize: Callable[[str], str] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if normalize is None:
            from framehunt.answers import normalize_answer as normalize
        budget_px = _int("budget_px", budget_px)
        if budget_px <= 0:
            raise ValueError("budget_px must be positive")
        self.scene = scene
        self.meta = scene.meta
        self.budget_total = budget_px
        self._budget_left = budget_px
        self._token = verify_token or ""
        self._normalize = normalize
        self._clock = clock
        self._expected = normalize(scene.answer)

        self._lock = threading.Lock()  # budget, submission, log
        self._render_lock = threading.Lock()  # scene.render + cache
        self._cache: OrderedDict[int, PILImage.Image] = OrderedDict()

        self._submitted = False
        self._answer: str | None = None
        self._correct = False
        self._log: list[dict] = []
        self._log_file = open(log_path, "a", encoding="utf-8") if log_path else None

    # ------------------------------------------------------------------ state
    @property
    def budget_left(self) -> int:
        with self._lock:
            return self._budget_left

    @property
    def pixels_used(self) -> int:
        with self._lock:
            return self.budget_total - self._budget_left

    def _record(self, tool: str, args: dict, ok: bool, cost: int = 0, error: str | None = None, **extra) -> dict:
        # Caller holds self._lock.
        rec = {
            "i": len(self._log),
            "ts": self._clock(),
            "tool": tool,
            "args": _jsonable(args),
            "ok": ok,
            "cost": cost,
            "budget_left": self._budget_left,
        }
        if error is not None:
            rec["error"] = error
        rec.update(extra)
        self._log.append(rec)
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps(rec) + "\n")
                self._log_file.flush()
            except OSError:
                pass
        return rec

    def _refuse(self, tool: str, args: dict, msg: str) -> FrameHuntError:
        with self._lock:
            self._record(tool, args, ok=False, error=msg)
        return FrameHuntError(msg)

    # ------------------------------------------------------------- validation
    def _frame_index(self, name: str, t: Any) -> int:
        t = _num(name, t)
        m = self.meta
        if t < 0:
            raise FrameHuntError(f"{name}={t} is negative; valid range is 0 to {m.duration_s:g} s")
        if t > m.duration_s + 1e-9:
            raise FrameHuntError(f"{name}={t} is past the end of the video; valid range is 0 to {m.duration_s:g} s")
        return min(int(math.floor(t * m.fps + 0.5)), m.n_frames - 1)

    def _crop(self, x: Any, y: Any, w: Any, h: Any) -> tuple[int, int, int, int]:
        m = self.meta
        x, y = _int("x", x), _int("y", y)
        if not 0 <= x < m.width:
            raise FrameHuntError(f"x={x} is outside the frame; must be 0 to {m.width - 1}")
        if not 0 <= y < m.height:
            raise FrameHuntError(f"y={y} is outside the frame; must be 0 to {m.height - 1}")
        w = m.width - x if w is None else _int("w", w)
        h = m.height - y if h is None else _int("h", h)
        if w <= 0 or h <= 0:
            raise FrameHuntError(f"w and h must be positive, got w={w}, h={h}")
        w, h = min(w, m.width - x), min(h, m.height - y)
        if w < MIN_OUT_WIDTH:
            raise FrameHuntError(f"crop is {w} px wide after clipping to the frame; it must be at least {MIN_OUT_WIDTH}")
        return x, y, w, h

    @staticmethod
    def _out_size(w: int, h: int, out_width: Any) -> tuple[int, int]:
        ow = _int("out_width", out_width)
        if ow < MIN_OUT_WIDTH:
            raise FrameHuntError(f"out_width must be at least {MIN_OUT_WIDTH}, got {ow}")
        ow = min(ow, w)  # no upscaling
        oh = max(1, int(math.floor(h * ow / w + 0.5)))
        return ow, oh

    # -------------------------------------------------------------- rendering
    def _render(self, idx: int) -> PILImage.Image:
        with self._render_lock:
            img = self._cache.get(idx)
            if img is not None:
                self._cache.move_to_end(idx)
                return img
            img = self.scene.render(idx)
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size != (self.meta.width, self.meta.height):
                raise RuntimeError(f"scene rendered {img.size}, expected {(self.meta.width, self.meta.height)}")
            self._cache[idx] = img
            if len(self._cache) > FRAME_CACHE_SIZE:
                self._cache.popitem(last=False)
            return img

    def _tile(self, idx: int, box: tuple[int, int, int, int], out: tuple[int, int]) -> PILImage.Image:
        x, y, w, h = box
        img = self._render(idx).crop((x, y, x + w, y + h))
        if (w, h) != out:
            img = img.resize(out, RESAMPLE)
        return img

    @staticmethod
    def _png(img: PILImage.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=6)
        return buf.getvalue()

    def _charge(self, tool: str, args: dict, cost: int, **extra) -> int:
        """Atomically check and charge. Raises (and charges nothing) if over budget."""
        with self._lock:
            if cost > self._budget_left:
                msg = f"refused: this request costs {cost} px but only {self._budget_left} px are left; nothing was charged"
                self._record(tool, args, ok=False, error=msg)
                raise FrameHuntError(msg)
            self._budget_left -= cost
            self._record(tool, args, ok=True, cost=cost, **extra)
            return self._budget_left

    def _precheck(self, tool: str, args: dict, cost: int) -> None:
        with self._lock:
            left = self._budget_left
        if cost > left:
            raise self._refuse(tool, args, f"refused: this request costs {cost} px but only {left} px are left; nothing was charged")

    def _frame_time(self, idx: int) -> float:
        return idx / self.meta.fps

    # ------------------------------------------------------------------ tools
    def info(self) -> str:
        m = self.meta
        with self._lock:
            left = self._budget_left
            self._record("info", {}, ok=True)
        return "\n".join([
            f"question: {self.scene.question}",
            f"answer_format: {self.scene.answer_format}",
            f"width: {m.width}",
            f"height: {m.height}",
            f"fps: {m.fps}",
            f"n_frames: {m.n_frames}",
            f"duration_s: {m.duration_s:g}",
            f"budget_total: {self.budget_total}",
            f"budget_left: {left}",
            "cost rules:",
            "  view: cost = output width x output height (pixels returned).",
            f"  contact_sheet: cost = n x tile output width x tile output height; the {SHEET_SEPARATOR_PX}-px separators are free.",
            "  info and submit are free. A request that costs more than budget_left is refused and charges nothing.",
            "  Frame times are snapped to the nearest frame. Crops (x, y, w, h) are in native pixels and clipped to the frame.",
            f"  out_width must be at least {MIN_OUT_WIDTH} and is capped at the crop width (no upscaling); height keeps the aspect ratio.",
            "  Only the first submit counts. Submitting gives no feedback on correctness.",
        ])

    def view(self, t: Any, x: Any = 0, y: Any = 0, w: Any = None, h: Any = None, out_width: Any = 320) -> tuple[bytes, str]:
        args = {"t": t, "x": x, "y": y, "w": w, "h": h, "out_width": out_width}
        try:
            idx = self._frame_index("t", t)
            box = self._crop(x, y, w, h)
            out = self._out_size(box[2], box[3], out_width)
        except FrameHuntError as e:
            raise self._refuse("view", args, str(e))
        cost = out[0] * out[1]
        self._precheck("view", args, cost)
        png = self._png(self._tile(idx, box, out))
        left = self._charge("view", args, cost, frames=[idx], crop=list(box), out=list(out))
        text = (f"t={self._frame_time(idx):.3f}s frame={idx} crop=(x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}) "
                f"output={out[0]}x{out[1]} cost={cost} budget_left={left}")
        return png, text

    def contact_sheet(self, t_start: Any, t_end: Any, n: Any, x: Any = 0, y: Any = 0, w: Any = None, h: Any = None,
                      out_width: Any = 160) -> tuple[bytes, str]:
        args = {"t_start": t_start, "t_end": t_end, "n": n, "x": x, "y": y, "w": w, "h": h, "out_width": out_width}
        try:
            f0 = self._frame_index("t_start", t_start)
            f1 = self._frame_index("t_end", t_end)
            n = _int("n", n)
            if not MIN_SHEET_N <= n <= MAX_SHEET_N:
                raise FrameHuntError(f"n must be between {MIN_SHEET_N} and {MAX_SHEET_N}, got {n}")
            if f1 <= f0:
                raise FrameHuntError("t_end must be later than t_start (after snapping to frames)")
            if f1 - f0 + 1 < n:
                raise FrameHuntError(f"the range holds only {f1 - f0 + 1} frames; n must not exceed that")
            box = self._crop(x, y, w, h)
            tw, th = self._out_size(box[2], box[3], out_width)
        except FrameHuntError as e:
            raise self._refuse("contact_sheet", args, str(e))
        cost = n * tw * th
        self._precheck("contact_sheet", args, cost)

        frames = [f0 + int(math.floor(i * (f1 - f0) / (n - 1) + 0.5)) for i in range(n)]
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        s = SHEET_SEPARATOR_PX
        sheet = PILImage.new("RGB", (cols * tw + (cols - 1) * s, rows * th + (rows - 1) * s), SHEET_SEPARATOR_RGB)
        for i, idx in enumerate(frames):
            r, c = divmod(i, cols)
            sheet.paste(self._tile(idx, box, (tw, th)), (c * (tw + s), r * (th + s)))
        png = self._png(sheet)
        left = self._charge("contact_sheet", args, cost, frames=frames, crop=list(box), out=[tw, th])
        lines = [f"{n} frames, {cols} columns x {rows} rows, read left to right then top to bottom; "
                 f"crop=(x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}) tile={tw}x{th} cost={cost} budget_left={left}"]
        lines += [f"  tile {i + 1}: t={self._frame_time(idx):.3f}s frame={idx}" for i, idx in enumerate(frames)]
        return png, "\n".join(lines)

    def submit(self, answer: Any) -> str:
        if not isinstance(answer, str):
            raise self._refuse("submit", {"answer": repr(answer)[:MAX_ANSWER_CHARS]}, "answer must be a string")
        args = {"answer": answer[:MAX_ANSWER_CHARS]}
        if len(answer) > MAX_ANSWER_CHARS:
            raise self._refuse("submit", args, f"answer is longer than {MAX_ANSWER_CHARS} characters; not recorded")
        if not answer.strip():
            raise self._refuse("submit", args, "answer is empty; not recorded")
        try:
            correct = self._normalize(answer) == self._expected
        except Exception:
            correct = False
        with self._lock:
            if self._submitted:
                msg = "refused: an answer was already submitted; only the first submission counts"
                self._record("submit", args, ok=False, error=msg)
                raise FrameHuntError(msg)
            self._submitted = True
            self._answer = answer
            self._correct = correct
            self._record("submit", args, ok=True)
        return "Answer recorded. Only the first submission counts, so this answer is final. No correctness feedback is given."

    # --------------------------------------------------------------- verifier
    def check_token(self, token: str | None) -> bool:
        if not self._token or not isinstance(token, str):
            return False
        return hmac.compare_digest(token.encode("utf-8", "surrogatepass"), self._token.encode("utf-8"))

    def result(self) -> dict:
        with self._lock:
            return {
                "submitted": self._submitted,
                "answer": self._answer,
                "correct": self._correct,
                "reward": 1.0 if self._correct else 0.0,
                "pixels_used": self.budget_total - self._budget_left,
                "n_calls": len(self._log),
            }

    def log_jsonl(self) -> str:
        with self._lock:
            return "".join(json.dumps(r) + "\n" for r in self._log)


# ---------------------------------------------------------------------- MCP wiring
def build_mcp(env: FrameHuntEnv, host: str = "0.0.0.0", port: int = 8000):
    import anyio
    from mcp.server.fastmcp import FastMCP, Image
    from mcp.server.fastmcp.exceptions import ToolError
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.requests import Request
    from starlette.responses import JSONResponse, PlainTextResponse, Response

    mcp = FastMCP(
        "framehunt", host=host, port=port, log_level="WARNING",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    async def run(fn, *a):
        try:
            return await anyio.to_thread.run_sync(fn, *a)
        except FrameHuntError as e:
            raise ToolError(str(e)) from None

    @mcp.tool(structured_output=False)
    async def info() -> str:
        """Show the question, answer format, video size/fps/length, the pixel budget left and the cost rules. Free."""
        return await run(env.info)

    @mcp.tool(structured_output=False)
    async def view(t: float, x: int = 0, y: int = 0, w: int | None = None, h: int | None = None,
                   out_width: int = 320):
        """Return one video frame as a PNG image, optionally cropped and downscaled.

        t: time in seconds, snapped to the nearest frame.
        x, y, w, h: crop rectangle in native pixels (default: whole frame), clipped to the frame.
        out_width: output width in pixels (min 8, capped at the crop width); height keeps the aspect ratio.
        Cost: output width x output height pixels.
        """
        png, text = await run(env.view, t, x, y, w, h, out_width)
        return [Image(data=png, format="png"), text]

    @mcp.tool(structured_output=False)
    async def contact_sheet(t_start: float, t_end: float, n: int, x: int = 0, y: int = 0, w: int | None = None,
                            h: int | None = None, out_width: int = 160):
        """Return one grid image of n evenly spaced frames from t_start to t_end (inclusive), plus their times.

        n: 2 to 16 frames. x, y, w, h: crop applied to every tile, in native pixels.
        out_width: width of each tile (min 8, capped at the crop width).
        Cost: n x tile width x tile height pixels; separators between tiles are free.
        """
        png, text = await run(env.contact_sheet, t_start, t_end, n, x, y, w, h, out_width)
        return [Image(data=png, format="png"), text]

    @mcp.tool(structured_output=False)
    async def submit(answer: str) -> str:
        """Submit the final answer. Only the first submission counts. No correctness feedback is given."""
        return await run(env.submit, answer)

    def _authorized(req: Request) -> bool:
        return env.check_token(req.query_params.get("token"))

    @mcp.custom_route("/_health", methods=["GET"])
    async def health(req: Request):
        return PlainTextResponse("ok")

    @mcp.custom_route("/_result", methods=["GET"])
    async def result(req: Request):
        if not _authorized(req):
            return PlainTextResponse("forbidden", status_code=403)
        return JSONResponse(env.result())

    @mcp.custom_route("/_log", methods=["GET"])
    async def log(req: Request):
        if not _authorized(req):
            return PlainTextResponse("forbidden", status_code=403)
        return Response(env.log_jsonl(), media_type="application/x-ndjson")

    return mcp


def env_from_environ(environ=os.environ, scene_factory=None) -> FrameHuntEnv:
    if scene_factory is None:
        from framehunt.scenes import KINDS, make_scene
        kind = environ.get("FH_KIND", "")
        if kind not in KINDS:
            raise SystemExit(f"FH_KIND must be one of {KINDS}, got {kind!r}")
        scene_factory = make_scene
    kind = environ.get("FH_KIND", "")
    try:
        seed = int(environ["FH_SEED"])
        budget = int(environ.get("FH_BUDGET_PX", DEFAULT_BUDGET_PX))
    except (KeyError, ValueError) as e:
        raise SystemExit(f"bad FH_SEED / FH_BUDGET_PX: {e}")
    token = environ.get("VERIFY_TOKEN", "")
    if not token:
        print("warning: VERIFY_TOKEN is empty; /_result and /_log are disabled", flush=True)
    env = FrameHuntEnv(scene_factory(kind, seed), budget_px=budget, verify_token=token,
                       log_path=environ.get("FH_LOG_PATH") or None)
    env._render(0)  # fail fast if the scene can't render; also warms the cache
    return env


def main() -> None:
    env = env_from_environ()
    build_mcp(env).run(transport="streamable-http")


if __name__ == "__main__":
    main()
