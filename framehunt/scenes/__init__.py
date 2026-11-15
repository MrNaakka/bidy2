"""Synthetic screen-recording scenes. `make_scene(kind, seed)` is the entry point."""
from __future__ import annotations

from .base import Scene, SceneMeta

KINDS: tuple[str, ...] = ("logstream", "dialogs", "dashboard", "toasts", "leaderboard", "scandoc", "mathanim")
# Kinds whose "video" is a scanned PDF (frame index = page index, fps 1).
DOCUMENT_KINDS: tuple[str, ...] = ("scandoc",)
# Kinds whose video is rendered with manim at build time and decoded from a file.
PRERENDERED_KINDS: tuple[str, ...] = ("mathanim",)
# Video kinds drawn frame by frame in Python.
VIDEO_KINDS: tuple[str, ...] = tuple(k for k in KINDS if k not in DOCUMENT_KINDS + PRERENDERED_KINDS)


def make_scene(kind: str, seed: int) -> Scene:
    if kind == "logstream":
        from .logstream import LogStreamScene
        return LogStreamScene(seed)
    if kind == "dialogs":
        from .dialogs import DialogsScene
        return DialogsScene(seed)
    if kind == "dashboard":
        from .dashboard import DashboardScene
        return DashboardScene(seed)
    if kind == "toasts":
        from .toasts import ToastsScene
        return ToastsScene(seed)
    if kind == "leaderboard":
        from .leaderboard import LeaderboardScene
        return LeaderboardScene(seed)
    if kind == "scandoc":
        from .scandoc import ScanDocScene
        return ScanDocScene(seed)
    if kind == "mathanim":
        from .mathanim import MathAnimScene
        return MathAnimScene(seed)
    raise ValueError(f"unknown scene kind {kind!r}; expected one of {KINDS}")


__all__ = ["DOCUMENT_KINDS", "KINDS", "PRERENDERED_KINDS", "VIDEO_KINDS", "Scene", "SceneMeta", "make_scene"]
