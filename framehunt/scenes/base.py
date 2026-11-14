"""Scene base class, metadata and shared drawing helpers.

Every scene is a pure function of (seed, frame): all randomness comes from local
`random.Random` / `numpy.random.default_rng` instances seeded from the scene seed, the
event timeline is precomputed in `__init__`, and `render(f)` never depends on which
frames were rendered before. Internal caches only store values that are themselves
pure functions of (seed, something), so they cannot change the output.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
FONT_FILES = {
    "sans": "DejaVuSans.ttf",
    "sans-bold": "DejaVuSans-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
    "mono-bold": "DejaVuSansMono-Bold.ttf",
}

_tls = threading.local()


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    """Bundled DejaVu font. Cached per thread (FreeType faces are not thread-safe)."""
    cache = getattr(_tls, "fonts", None)
    if cache is None:
        cache = _tls.fonts = {}
    key = (name, size)
    f = cache.get(key)
    if f is None:
        f = cache[key] = ImageFont.truetype(str(FONT_DIR / FONT_FILES[name]), size)
    return f


def draw_text(d: ImageDraw.ImageDraw, xy, text: str, font_name: str, size: int, fill) -> None:
    """The one way scenes draw text (default 'la' anchor), so tests/OCR can reproduce it."""
    d.text(xy, text, font=font(font_name, size), fill=fill)


def text_width(text: str, font_name: str, size: int) -> float:
    return font(font_name, size).getlength(text)


@dataclass(frozen=True)
class SceneMeta:
    width: int
    height: int
    fps: int
    n_frames: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps


class _LRU:
    """Small thread-safe LRU for pure intermediate results (e.g. rendered text strips)."""

    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key, make):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
                return v
        v = make()
        with self._lock:
            self._d[key] = v
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        return v


class Scene(ABC):
    kind: str
    seed: int
    meta: SceneMeta
    question: str
    answer: str
    answer_format: str

    WIDTH = 1920
    HEIGHT = 1080
    FPS = 10

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._cache = _LRU(256)
        self._static_lock = threading.RLock()
        self._static: dict = {}

    def rng(self, purpose: str) -> random.Random:
        """Independent deterministic stream per purpose (str seeds hash with SHA-512)."""
        return random.Random(f"framehunt/{self.kind}/{self.seed}/{purpose}")

    def np_seed(self, purpose: str) -> int:
        return self.rng(purpose).getrandbits(63)

    def static(self, name: str, make):
        """Lazily built, immutable per-scene layer (e.g. the background image)."""
        with self._static_lock:
            v = self._static.get(name)
            if v is None:
                v = self._static[name] = make()
            return v

    def frame_time_s(self, frame: int) -> float:
        return frame / self.meta.fps

    def _check_frame(self, frame: int) -> int:
        frame = int(frame)
        if not 0 <= frame < self.meta.n_frames:
            raise IndexError(f"frame {frame} out of range [0, {self.meta.n_frames})")
        return frame

    @abstractmethod
    def render(self, frame: int) -> Image.Image:
        """RGB image, meta.width x meta.height. Pure function of (seed, frame)."""

    @abstractmethod
    def debug_info(self) -> dict:
        """Key frames etc. For tests/analysis only; never exposed to the agent."""


def pick_event_frames(rng: random.Random, n: int, lo: int, hi: int, min_gap: int) -> list[int]:
    """n sorted distinct frames in [lo, hi] with pairwise gaps >= min_gap."""
    span = hi - lo - (n - 1) * min_gap
    if span < 0:
        raise ValueError("not enough room for events")
    base = sorted(rng.randint(0, span) for _ in range(n))
    return [lo + b + i * min_gap for i, b in enumerate(base)]


class GlyphAtlas:
    """Fixed-cell glyph masks for fast terminal-style text, composed with numpy.

    Like a real terminal, each character sits in an integer cell (cell_w x cell_h) and
    is rasterised on its own, so a row of text is a pure function of its characters.
    """

    _registry: dict = {}
    _lock = threading.Lock()

    def __init__(self, font_name: str, size: int, cell_w: int, cell_h: int, y_off: int):
        self.font_name, self.size = font_name, size
        self.cell_w, self.cell_h, self.y_off = cell_w, cell_h, y_off
        masks = np.zeros((128, cell_h, cell_w), np.uint8)
        f = font(font_name, size)
        for c in range(32, 127):
            im = Image.new("L", (cell_w, cell_h), 0)
            ImageDraw.Draw(im).text((0, y_off), chr(c), font=f, fill=255)
            masks[c] = np.asarray(im)
        masks.setflags(write=False)
        self.masks = masks

    @classmethod
    def get(cls, font_name: str, size: int, cell_w: int, cell_h: int, y_off: int) -> "GlyphAtlas":
        key = (font_name, size, cell_w, cell_h, y_off)
        with cls._lock:
            a = cls._registry.get(key)
            if a is None:
                a = cls._registry[key] = cls(*key)
            return a

    @staticmethod
    def encode(text: str, width: int) -> np.ndarray:
        b = text[:width].ljust(width).encode("ascii", "replace")
        return np.frombuffer(b, np.uint8).copy()

    def colored(self, palette, bg) -> np.ndarray:
        """Pre-blended RGB glyph cells: (len(palette), 128, cell_h, cell_w, 3) uint8."""
        key = ("colored", tuple(map(tuple, np.asarray(palette).tolist())), tuple(bg))
        with self._lock:
            v = self._registry.get((id(self), key))
            if v is not None:
                return v
        bgv = np.asarray(bg, np.int32)
        col = np.asarray(palette, np.int32)[:, None, None, None, :] - bgv     # P,1,1,1,3
        a = self.masks.astype(np.int32)[None, ..., None]                        # 1,128,h,w,1
        out = ((col * a + 127) // 255 + bgv).astype(np.uint8)
        out.setflags(write=False)
        with self._lock:
            self._registry[(id(self), key)] = out
        return out

    def compose(self, codes: np.ndarray, cidx: np.ndarray, palette, bg) -> np.ndarray:
        """codes, cidx (R, C) uint8 -> RGB array (R*cell_h, C*cell_w, 3).

        Equivalent to blending each glyph mask over `bg` with colour palette[cidx]:
        out = bg + round((fg - bg) * alpha / 255).
        """
        R, C = codes.shape
        cells = self.colored(palette, bg)[cidx, codes]          # R,C,h,w,3
        return np.ascontiguousarray(cells.transpose(0, 2, 1, 3, 4)).reshape(
            R * self.cell_h, C * self.cell_w, 3)
