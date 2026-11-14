"""Fixed, committed item lists. Each item is (item_id, kind, seed).

EVAL: 30 items (10 per kind). DEV: 60 items (20 per kind). Seeds are disjoint between
splits (EVAL uses 7xxx, DEV uses 3xxx). item_id is "<kind>-<seed:04d>". Do not edit
existing entries: results are reported against these exact items.
"""

from __future__ import annotations

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
)

SPLITS: dict[str, tuple[tuple[str, str, int], ...]] = {"eval": EVAL, "dev": DEV}


def _check() -> None:
    for name, items in SPLITS.items():
        ids = [i for i, _, _ in items]
        assert len(ids) == len(set(ids)), f"duplicate item ids in {name}"
    assert not {s for _, _, s in EVAL} & {s for _, _, s in DEV}, "EVAL and DEV seeds overlap"


_check()
