"""Copy a TRAIN task pool so Harbor starts trials from prebuilt images instead of building per trial.

    python3 prebuilt_pool.py SRC DST --main fh-grpo-main:v1 --mcp fh-grpo-mcp:v1 --verifier fh-grpo-verifier:v1

Every pool task has the same build contexts (environment/, environment/mcp/, tests/; the item is chosen
by FH_KIND/FH_SEED/... in docker-compose.yaml), so one image each serves all items. The images must be
built from one task's dirs first (docker build -t <tag> environment | environment/mcp | tests). Changes
per task, nothing else: task.toml [environment] and [verifier.environment] get docker_image (Harbor
then skips `docker compose build` and runs that image), and docker-compose.yaml's fh-mcp `build:
context: ./mcp` becomes `image: <mcp>`. The build contexts are checked to be identical to the source
task the images were built from (--ref).
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path


def tree_hash(d: Path, skip=("docker-compose.yaml",)) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name not in skip:
            h.update(str(p.relative_to(d)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--ref", required=True, help="the task whose dirs the images were built from")
    ap.add_argument("--main", required=True)
    ap.add_argument("--mcp", required=True)
    ap.add_argument("--verifier", required=True)
    a = ap.parse_args()
    ref = a.src / a.ref
    want = {s: tree_hash(ref / s) for s in ("environment", "environment/mcp", "tests")}
    a.dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for t in sorted(a.src.iterdir()):
        if not (t / "task.toml").is_file():
            continue
        for s, h in want.items():
            if tree_hash(t / s) != h:
                raise SystemExit(f"{t.name}: {s} differs from {a.ref}; the prebuilt image would not match")
        d = a.dst / t.name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(t, d)
        toml = (d / "task.toml").read_text()
        toml, k1 = re.subn(r"^\[environment\]\n", f'[environment]\ndocker_image = "{a.main}"\n', toml, flags=re.M)
        toml, k2 = re.subn(r"^\[verifier\.environment\]\n", f'[verifier.environment]\ndocker_image = "{a.verifier}"\n',
                           toml, flags=re.M)
        assert k1 == 1 and k2 == 1, (t.name, k1, k2)
        (d / "task.toml").write_text(toml)
        dc = (d / "environment" / "docker-compose.yaml").read_text()
        dc, k3 = re.subn(r"^    build:\n      context: \./mcp\n", f"    image: {a.mcp}\n", dc, flags=re.M)
        assert k3 == 1, t.name
        (d / "environment" / "docker-compose.yaml").write_text(dc)
        n += 1
    print(f"{n} tasks -> {a.dst}")


if __name__ == "__main__":
    main()
