"""Synthetic screen-recording scenes. `make_scene(kind, seed)` is the entry point."""
from __future__ import annotations

from .base import Scene, SceneMeta

KINDS: tuple[str, ...] = ("logstream", "dialogs", "dashboard")


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
    raise ValueError(f"unknown scene kind {kind!r}; expected one of {KINDS}")


__all__ = ["KINDS", "Scene", "SceneMeta", "make_scene"]
