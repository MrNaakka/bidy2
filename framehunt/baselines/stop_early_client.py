"""Stop-early client: runs inside the agent's container.

    python3 stop_early_client.py URL ITEM_ID

Calls `info`, takes one low-res look at the middle frame (1/8 scale), then submits a
well-formed guess that matches answer_format, drawn from random.Random(item_id). It
never reads the image: that is the point of the baseline (stopping before doing the
work). Prints one JSON line per tool call.
"""

import asyncio
import json
import random
import re
import sys
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def guess(answer_format: str, rng: random.Random):
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


async def main(url: str, item_id: str):
    async with streamable_http_client(url) as (read, write, _), ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as session:
        await session.initialize()

        async def call(tool, args):
            res = await session.call_tool(tool, args)
            text = "\n".join(c.text for c in res.content if c.type == "text")
            print(json.dumps({"tool": tool, "args": args, "isError": res.isError,
                              "text": text[:300] if tool != "info" else "(info)"}), flush=True)
            return text

        info = {k.strip(): v.strip() for k, v in (line.split(":", 1) for line in (await call("info", {})).splitlines()
                                                  if ":" in line and not line.startswith(" "))}
        await call("view", {"t": float(info["duration_s"]) / 2, "out_width": int(info["width"]) // 8})
        answer = guess(info["answer_format"], random.Random(f"stop-early/{item_id}"))
        await call("submit", {"answer": answer})
        print(json.dumps({"summary": True, "item_id": item_id, "answer": answer}), flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "unknown"))
