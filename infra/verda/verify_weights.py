"""Verify a downloaded Hugging Face snapshot against the Hub's metadata.

Checks every file listed by the Hub at the pinned revision: size always, and the
LFS sha256 for large files. Writes a JSON report next to the model directory and
exits non-zero on any mismatch.

usage: python verify_weights.py <repo_id> <revision> <local_dir>
"""

import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    repo, rev, local = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    url = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
    meta = json.load(urllib.request.urlopen(url, timeout=60))
    files = meta["siblings"]
    t0 = time.time()

    def check(sib: dict) -> dict:
        name = sib["rfilename"]
        p = local / name
        rec = {"file": name, "expected_size": sib.get("size")}
        if not p.exists():
            return rec | {"ok": False, "error": "missing"}
        size = p.stat().st_size
        rec["size"] = size
        if sib.get("size") is not None and size != sib["size"]:
            return rec | {"ok": False, "error": "size mismatch"}
        lfs = sib.get("lfs")
        if lfs:
            digest = sha256_file(p)
            rec |= {"sha256": digest, "expected_sha256": lfs["sha256"]}
            if digest != lfs["sha256"]:
                return rec | {"ok": False, "error": "sha256 mismatch"}
        return rec | {"ok": True}

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(check, files))
    bad = [r for r in results if not r["ok"]]
    report = {
        "repo": repo,
        "revision": rev,
        "local_dir": str(local),
        "n_files": len(results),
        "n_lfs_sha256_checked": sum(1 for r in results if "sha256" in r),
        "total_bytes": sum(r.get("size", 0) for r in results),
        "n_bad": len(bad),
        "bad": bad,
        "seconds": round(time.time() - t0, 1),
        "files": results,
    }
    out = local.parent / f"{local.name}.verify.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "files"}))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
