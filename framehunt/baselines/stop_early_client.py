"""Stop-early client: runs inside the agent's container, stdlib only.

    python3 stop_early_client.py URL ITEM_ID

Calls `info`, takes one low-res look at the middle frame (1/8 scale), then submits a
well-formed guess that matches answer_format, drawn from random.Random(item_id). It
never reads the image: that is the point of the baseline (stopping before doing the
work). Prints one JSON line per tool call.
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.request

PROTOCOL = "2025-06-18"


class Client:
    def __init__(self, url: str):
        self.url, self.headers, self.n = url, {}, 0

    def rpc(self, method, params=None, notify=False):
        body = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            body["params"] = params
        if not notify:
            self.n += 1
            body["id"] = self.n
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.headers})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.headers.get("mcp-session-id"):
                self.headers["Mcp-Session-Id"] = r.headers["mcp-session-id"]
            ctype, raw = r.headers.get("content-type", ""), r.read().decode()
        if notify:
            return None
        msgs = ([json.loads(x[5:]) for x in raw.splitlines() if x.startswith("data:")]
                if "text/event-stream" in ctype else [json.loads(raw)])
        for m in msgs:
            if m.get("id") == body["id"]:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m["result"]
        raise RuntimeError(f"no response to {method}")

    def call(self, tool, args):
        res = self.rpc("tools/call", {"name": tool, "arguments": args})
        text = "\n".join(c["text"] for c in res.get("content", []) if c.get("type") == "text")
        print(json.dumps({"tool": tool, "args": args, "isError": res.get("isError", False),
                          "text": text[:300] if tool != "info" else "(info)"}), flush=True)
        return text


def guess(answer_format: str, rng: random.Random) -> str:
    """A well-formed but uninformed answer for the stated format."""
    fmt = answer_format.lower()
    if "integer" in fmt or "digits only" in fmt:
        m = re.search(r"(\d+) to (\d+) digits", fmt)
        lo, hi = (int(m.group(1)), int(m.group(2))) if m else (3, 5)
        nd = rng.randint(lo, hi)
        return str(rng.randint(10 ** (nd - 1), 10 ** nd - 1))
    m = re.search(r"(\d+) characters", fmt)
    n = int(m.group(1)) if m else 6
    alphabet = "0123456789ABCDEF" if "hex" in fmt else "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(rng.choice(alphabet) for _ in range(n))


def main(url: str, item_id: str) -> int:
    c = Client(url)
    init = c.rpc("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                "clientInfo": {"name": "framehunt-stop-early", "version": "1"}})
    c.headers["MCP-Protocol-Version"] = init.get("protocolVersion", PROTOCOL)
    c.rpc("notifications/initialized", notify=True)
    info = dict(line.split(":", 1) for line in c.call("info", {}).splitlines()
                if ":" in line and not line.startswith(" "))
    info = {k.strip(): v.strip() for k, v in info.items()}
    c.call("view", {"t": float(info["duration_s"]) / 2, "out_width": int(info["width"]) // 8})
    answer = guess(info["answer_format"], random.Random(f"stop-early/{item_id}"))
    c.call("submit", {"answer": answer})
    print(json.dumps({"summary": True, "item_id": item_id, "answer": answer}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unknown"))
