"""Synthetic screen-recording scenes. `make_scene(kind, seed)` is the entry point."""
from __future__ import annotations

from .base import Scene, SceneMeta

KINDS: tuple[str, ...] = ("logstream", "dialogs", "dashboard", "toasts", "leaderboard")


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
    raise ValueError(f"unknown scene kind {kind!r}; expected one of {KINDS}")


__all__ = ["KINDS", "Scene", "SceneMeta", "make_scene"]
