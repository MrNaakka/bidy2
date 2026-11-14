"""Stdlib-only MCP plumbing check (Harbor oracle placeholder).

Speaks MCP streamable HTTP to the FrameHunt server: initialize, list tools, call
`info`, one tiny `view`, then `submit` a dummy answer. It proves the agent-side MCP
path works end to end. The dummy answer is wrong on purpose, so reward 0 is expected.

Usage: python3 mcp_smoke.py [URL]   (default http://fh-mcp:8000/mcp)
"""

from __future__ import annotations

import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://fh-mcp:8000/mcp"
PROTOCOL = "2025-06-18"
_session: dict[str, str] = {}
_next_id = [0]


def rpc(method: str, params: dict | None = None, notify: bool = False) -> dict | None:
    body: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        body["params"] = params
    if not notify:
        _next_id[0] += 1
        body["id"] = _next_id[0]
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    headers.update(_session)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        sid = r.headers.get("mcp-session-id")
        if sid:
            _session["Mcp-Session-Id"] = sid
        ctype = r.headers.get("content-type", "")
        raw = r.read().decode()
    if notify:
        return None
    msgs = []
    if "text/event-stream" in ctype:
        msgs = [json.loads(line[5:].strip()) for line in raw.splitlines() if line.startswith("data:")]
    elif raw.strip():
        msgs = [json.loads(raw)]
    for m in msgs:
        if m.get("id") == body["id"]:
            if "error" in m:
                raise RuntimeError(f"{method}: {m['error']}")
            return m["result"]
    raise RuntimeError(f"{method}: no response in {raw[:500]!r}")


def call(tool: str, args: dict) -> dict:
    res = rpc("tools/call", {"name": tool, "arguments": args})
    kinds = [c.get("type") for c in res.get("content", [])]
    texts = [c["text"] for c in res.get("content", []) if c.get("type") == "text"]
    print(f"--- {tool}({json.dumps(args)}) isError={res.get('isError', False)} content={kinds}")
    for t in texts:
        print(t)
    return res


def main() -> int:
    init = rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                              "clientInfo": {"name": "framehunt-smoke", "version": "0"}})
    _session["MCP-Protocol-Version"] = init.get("protocolVersion", PROTOCOL)
    rpc("notifications/initialized", notify=True)
    tools = sorted(t["name"] for t in rpc("tools/list", {})["tools"])
    print("tools:", tools)
    ok = tools == ["contact_sheet", "info", "submit", "view"]
    ok &= not call("info", {}).get("isError")
    v = call("view", {"t": 0, "out_width": 64})
    ok &= not v.get("isError") and any(c.get("type") == "image" and c.get("mimeType") == "image/png"
                                       for c in v["content"])
    ok &= not call("submit", {"answer": "PLUMBING-CHECK"}).get("isError")
    ok &= bool(call("submit", {"answer": "SECOND"}).get("isError"))  # must be refused
    print("SMOKE", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
