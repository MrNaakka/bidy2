"""Verifier evidence: dump the running server's result and call log to files.

Runs inside the fh-mcp sidecar as a Harbor ``[[verifier.collect]]`` hook, after the
agent's container has been stopped:

    python -m framehunt.evidence /fh-evidence

It reads the token-protected ``/_result`` and ``/_log`` routes of the server process
in the same container (the token comes from the container's own VERIFY_TOKEN env var)
and writes ``result.json`` and ``calls.jsonl`` into the output directory. Harbor then
collects that directory as an artifact into the separate verifier container, which
computes the reward (see ``framehunt.verify``). The agent never has write access to
the sidecar, so nothing in this chain is under the agent's control.

Exit code 0 only if both files were written. On failure nothing is left behind, so
the verifier sees a missing artifact and scores 0.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_BASE = "http://127.0.0.1:8000"


def _get(base: str, route: str, token: str, timeout: float) -> bytes:
    url = f"{base}{route}?{urllib.parse.urlencode({'token': token})}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def dump(out_dir: str | Path, base: str = DEFAULT_BASE, token: str | None = None, timeout: float = 30.0) -> dict:
    token = os.environ.get("VERIFY_TOKEN", "") if token is None else token
    if not token:
        raise RuntimeError("VERIFY_TOKEN is not set")
    result_raw = _get(base, "/_result", token, timeout)
    result = json.loads(result_raw)  # must be valid JSON before we write anything
    if not isinstance(result, dict):
        raise RuntimeError(f"/_result returned {type(result).__name__}, expected an object")
    log_raw = _get(base, "/_log", token, timeout)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(out / "calls.jsonl", log_raw)
    _write_atomic(out / "result.json", result_raw)  # written last: its presence means the dump completed
    return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2):
        print("usage: python -m framehunt.evidence OUT_DIR [BASE_URL]", file=sys.stderr)
        return 2
    try:
        res = dump(argv[0], *argv[1:])
    except Exception as e:  # noqa: BLE001 - report and fail; the verifier scores a missing dump as 0
        print(f"evidence dump failed: {e!r}", file=sys.stderr)
        return 1
    print(f"evidence written to {argv[0]}: {json.dumps(res)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
