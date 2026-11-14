"""Fixed, committed item lists, and the secret seed derivation. Each item is
(item_id, kind, public_seed).

EVAL (v2): 30 items, 10 each of logstream, dialogs and toasts (the control). EVAL_V1 is
the first evaluation set (10 each of logstream, dialogs and dashboard); the v1 results
in the README were measured on it. EVAL shares its logstream and dialogs ids with
EVAL_V1 (same derived seeds; dialogs' answer_format now states the exact alphabet).
leaderboard is in DEV and TRAIN only. DEV: 100 items (20 per kind). item_id
is "<kind>-<public_seed:04d>". Public seeds are disjoint between splits (EVAL uses 7xxx,
DEV uses 3xxx). Do not edit existing entries: results are reported against these ids.

The public seed is only part of the item's name. The scene seed actually rendered is
``derive_seed(split, item_id, salt)`` = the first 8 bytes of
HMAC-SHA256(salt, "<split>:<item_id>") as a big-endian int. The salt is secret: it comes
from the env var ``FH_SALT_<SPLIT>`` (e.g. FH_SALT_EVAL) or the git-ignored file
``secrets/<split>.salt``. Without the salt, the public generators and the public item
list don't reveal any answer. With the same salt, every item reproduces exactly.
See docs/design.md, "Item secrecy".
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets as _secrets
from pathlib import Path

SECRETS_DIR = Path(__file__).resolve().parent.parent / "secrets"

# The first evaluation set; reported as "v1" in the README. Kept for the record.
EVAL_V1: tuple[tuple[str, str, int], ...] = (
    ("logstream-7001", "logstream", 7001),
    ("logstream-7002", "logstream", 7002),
    ("logstream-7003", "logstream", 7003),
    ("logstream-7004", "logstream", 7004),
    ("logstream-7005", "logstream", 7005),
    ("logstream-7006", "logstream", 7006),
    ("logstream-7007", "logstream", 7007),
    ("logstream-7008", "logstream", 7008),
    ("logstream-7009", "logstream", 7009),
    ("logstream-7010", "logstream", 7010),
    ("dialogs-7101", "dialogs", 7101),
    ("dialogs-7102", "dialogs", 7102),
    ("dialogs-7103", "dialogs", 7103),
    ("dialogs-7104", "dialogs", 7104),
    ("dialogs-7105", "dialogs", 7105),
    ("dialogs-7106", "dialogs", 7106),
    ("dialogs-7107", "dialogs", 7107),
    ("dialogs-7108", "dialogs", 7108),
    ("dialogs-7109", "dialogs", 7109),
    ("dialogs-7110", "dialogs", 7110),
    ("dashboard-7201", "dashboard", 7201),
    ("dashboard-7202", "dashboard", 7202),
    ("dashboard-7203", "dashboard", 7203),
    ("dashboard-7204", "dashboard", 7204),
    ("dashboard-7205", "dashboard", 7205),
    ("dashboard-7206", "dashboard", 7206),
    ("dashboard-7207", "dashboard", 7207),
    ("dashboard-7208", "dashboard", 7208),
    ("dashboard-7209", "dashboard", 7209),
    ("dashboard-7210", "dashboard", 7210),
)

EVAL: tuple[tuple[str, str, int], ...] = (
    ("logstream-7001", "logstream", 7001),
    ("logstream-7002", "logstream", 7002),
    ("logstream-7003", "logstream", 7003),
    ("logstream-7004", "logstream", 7004),
    ("logstream-7005", "logstream", 7005),
    ("logstream-7006", "logstream", 7006),
    ("logstream-7007", "logstream", 7007),
    ("logstream-7008", "logstream", 7008),
    ("logstream-7009", "logstream", 7009),
    ("logstream-7010", "logstream", 7010),
    ("dialogs-7101", "dialogs", 7101),
    ("dialogs-7102", "dialogs", 7102),
    ("dialogs-7103", "dialogs", 7103),
    ("dialogs-7104", "dialogs", 7104),
    ("dialogs-7105", "dialogs", 7105),
    ("dialogs-7106", "dialogs", 7106),
    ("dialogs-7107", "dialogs", 7107),
    ("dialogs-7108", "dialogs", 7108),
    ("dialogs-7109", "dialogs", 7109),
    ("dialogs-7110", "dialogs", 7110),
    ("toasts-7401", "toasts", 7401),
    ("toasts-7402", "toasts", 7402),
    ("toasts-7403", "toasts", 7403),
    ("toasts-7404", "toasts", 7404),
    ("toasts-7405", "toasts", 7405),
    ("toasts-7406", "toasts", 7406),
    ("toasts-7407", "toasts", 7407),
    ("toasts-7408", "toasts", 7408),
    ("toasts-7409", "toasts", 7409),
    ("toasts-7410", "toasts", 7410),
)

DEV: tuple[tuple[str, str, int], ...] = (
    ("logstream-3001", "logstream", 3001),
    ("logstream-3002", "logstream", 3002),
    ("logstream-3003", "logstream", 3003),
    ("logstream-3004", "logstream", 3004),
    ("logstream-3005", "logstream", 3005),
    ("logstream-3006", "logstream", 3006),
    ("logstream-3007", "logstream", 3007),
    ("logstream-3008", "logstream", 3008),
    ("logstream-3009", "logstream", 3009),
    ("logstream-3010", "logstream", 3010),
    ("logstream-3011", "logstream", 3011),
    ("logstream-3012", "logstream", 3012),
    ("logstream-3013", "logstream", 3013),
    ("logstream-3014", "logstream", 3014),
    ("logstream-3015", "logstream", 3015),
    ("logstream-3016", "logstream", 3016),
    ("logstream-3017", "logstream", 3017),
    ("logstream-3018", "logstream", 3018),
    ("logstream-3019", "logstream", 3019),
    ("logstream-3020", "logstream", 3020),
    ("dialogs-3101", "dialogs", 3101),
    ("dialogs-3102", "dialogs", 3102),
    ("dialogs-3103", "dialogs", 3103),
    ("dialogs-3104", "dialogs", 3104),
    ("dialogs-3105", "dialogs", 3105),
    ("dialogs-3106", "dialogs", 3106),
    ("dialogs-3107", "dialogs", 3107),
    ("dialogs-3108", "dialogs", 3108),
    ("dialogs-3109", "dialogs", 3109),
    ("dialogs-3110", "dialogs", 3110),
    ("dialogs-3111", "dialogs", 3111),
    ("dialogs-3112", "dialogs", 3112),
    ("dialogs-3113", "dialogs", 3113),
    ("dialogs-3114", "dialogs", 3114),
    ("dialogs-3115", "dialogs", 3115),
    ("dialogs-3116", "dialogs", 3116),
    ("dialogs-3117", "dialogs", 3117),
    ("dialogs-3118", "dialogs", 3118),
    ("dialogs-3119", "dialogs", 3119),
    ("dialogs-3120", "dialogs", 3120),
    ("dashboard-3201", "dashboard", 3201),
    ("dashboard-3202", "dashboard", 3202),
    ("dashboard-3203", "dashboard", 3203),
    ("dashboard-3204", "dashboard", 3204),
    ("dashboard-3205", "dashboard", 3205),
    ("dashboard-3206", "dashboard", 3206),
    ("dashboard-3207", "dashboard", 3207),
    ("dashboard-3208", "dashboard", 3208),
    ("dashboard-3209", "dashboard", 3209),
    ("dashboard-3210", "dashboard", 3210),
    ("dashboard-3211", "dashboard", 3211),
    ("dashboard-3212", "dashboard", 3212),
    ("dashboard-3213", "dashboard", 3213),
    ("dashboard-3214", "dashboard", 3214),
    ("dashboard-3215", "dashboard", 3215),
    ("dashboard-3216", "dashboard", 3216),
    ("dashboard-3217", "dashboard", 3217),
    ("dashboard-3218", "dashboard", 3218),
    ("dashboard-3219", "dashboard", 3219),
    ("dashboard-3220", "dashboard", 3220),
    ("toasts-3401", "toasts", 3401),
    ("toasts-3402", "toasts", 3402),
    ("toasts-3403", "toasts", 3403),
    ("toasts-3404", "toasts", 3404),
    ("toasts-3405", "toasts", 3405),
    ("toasts-3406", "toasts", 3406),
    ("toasts-3407", "toasts", 3407),
    ("toasts-3408", "toasts", 3408),
    ("toasts-3409", "toasts", 3409),
    ("toasts-3410", "toasts", 3410),
    ("toasts-3411", "toasts", 3411),
    ("toasts-3412", "toasts", 3412),
    ("toasts-3413", "toasts", 3413),
    ("toasts-3414", "toasts", 3414),
    ("toasts-3415", "toasts", 3415),
    ("toasts-3416", "toasts", 3416),
    ("toasts-3417", "toasts", 3417),
    ("toasts-3418", "toasts", 3418),
    ("toasts-3419", "toasts", 3419),
    ("toasts-3420", "toasts", 3420),
    ("leaderboard-3501", "leaderboard", 3501),
    ("leaderboard-3502", "leaderboard", 3502),
    ("leaderboard-3503", "leaderboard", 3503),
    ("leaderboard-3504", "leaderboard", 3504),
    ("leaderboard-3505", "leaderboard", 3505),
    ("leaderboard-3506", "leaderboard", 3506),
    ("leaderboard-3507", "leaderboard", 3507),
    ("leaderboard-3508", "leaderboard", 3508),
    ("leaderboard-3509", "leaderboard", 3509),
    ("leaderboard-3510", "leaderboard", 3510),
    ("leaderboard-3511", "leaderboard", 3511),
    ("leaderboard-3512", "leaderboard", 3512),
    ("leaderboard-3513", "leaderboard", 3513),
    ("leaderboard-3514", "leaderboard", 3514),
    ("leaderboard-3515", "leaderboard", 3515),
    ("leaderboard-3516", "leaderboard", 3516),
    ("leaderboard-3517", "leaderboard", 3517),
    ("leaderboard-3518", "leaderboard", 3518),
    ("leaderboard-3519", "leaderboard", 3519),
    ("leaderboard-3520", "leaderboard", 3520),
)

SPLITS: dict[str, tuple[tuple[str, str, int], ...]] = {"eval": EVAL, "dev": DEV}


# TRAIN: generated, not committed item by item, and deliberately not in SPLITS (so
# build_tasks.py and the eval tooling never see it). Used only by training-data
# generators (train/make_sft.py). Public seeds start at 100001 (logstream), 110001
# (dialogs), 120001 (dashboard), 130001 (toasts), 140001 (leaderboard), so ids never
# collide with EVAL (7xxx) or DEV (3xxx).
# Scene seeds are derive_seed("train", item_id) with the split's own secret salt
# (FH_SALT_TRAIN or secrets/train.salt): a different split name and a different salt
# from EVAL/DEV, so no training scene can coincide with an evaluation scene.
TRAIN_KINDS: tuple[str, ...] = ("logstream", "dialogs", "dashboard", "toasts", "leaderboard")
TRAIN_SEED_BASE = 100_000
TRAIN_MAX_PER_KIND = 9_999


def train_items(n_per_kind: int = 300) -> tuple[tuple[str, str, int], ...]:
    """The first ``n_per_kind`` TRAIN items of each kind, as (item_id, kind, public_seed).

    Stable: item i of a kind is the same for every n_per_kind > i.
    """
    if not 0 < n_per_kind <= TRAIN_MAX_PER_KIND:
        raise ValueError(f"n_per_kind must be 1..{TRAIN_MAX_PER_KIND}, got {n_per_kind}")
    out = []
    for ki, kind in enumerate(TRAIN_KINDS):
        for j in range(n_per_kind):
            s = TRAIN_SEED_BASE + ki * 10_000 + j + 1
            out.append((f"{kind}-{s}", kind, s))
    return tuple(out)


def salt_env_var(split: str) -> str:
    return f"FH_SALT_{split.upper()}"


def salt_path(split: str, secrets_dir: Path | None = None) -> Path:
    return (secrets_dir or SECRETS_DIR) / f"{split}.salt"


def load_salt(split: str, *, create: bool = False, secrets_dir: Path | None = None,
              environ=os.environ) -> str | None:
    """The split's secret salt: env FH_SALT_<SPLIT>, else secrets/<split>.salt.

    If neither exists: with ``create=True`` a fresh random salt (64 hex chars) is
    written to secrets/<split>.salt (mode 0600) and returned; otherwise returns None.
    """
    v = environ.get(salt_env_var(split), "").strip()
    if v:
        return v
    p = salt_path(split, secrets_dir)
    if p.is_file():
        v = p.read_text().strip()
        if not v:
            raise ValueError(f"{p} is empty; delete it to generate a new salt, or restore the old one")
        return v
    if not create:
        return None
    v = _secrets.token_hex(32)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(v + "\n")
    return v


def derive_seed(split: str, item_id: str, salt: str | bytes | None = None) -> int:
    """The scene seed for an item: first 8 bytes of HMAC-SHA256(salt, f"{split}:{item_id}").

    The single place where scene seeds come from (task builder, tests, training data).
    ``salt=None`` loads the split's salt, creating it if missing (see load_salt).
    """
    if salt is None:
        salt = load_salt(split, create=True)
    key = salt.encode("utf-8") if isinstance(salt, str) else bytes(salt)
    if not key:
        raise ValueError("salt must not be empty")
    mac = hmac.new(key, f"{split}:{item_id}".encode(), hashlib.sha256).digest()
    return int.from_bytes(mac[:8], "big")


def _check() -> None:
    for name, items in SPLITS.items():
        ids = [i for i, _, _ in items]
        assert len(ids) == len(set(ids)), f"duplicate item ids in {name}"
    assert not {s for _, _, s in EVAL + EVAL_V1} & {s for _, _, s in DEV}, "EVAL and DEV seeds overlap"
    v1 = {i: (k, s) for i, k, s in EVAL_V1}
    assert all(v1[i] == (k, s) for i, k, s in EVAL if i in v1), "shared EVAL/EVAL_V1 ids must match"
    assert "train" not in SPLITS


_check()
