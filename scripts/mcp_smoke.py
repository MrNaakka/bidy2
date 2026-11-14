"""MCP plumbing check (Harbor oracle placeholder).

Speaks MCP streamable HTTP to the FrameHunt server: initialize, list tools, call
`info`, one tiny `view`, then `submit` a dummy answer. It proves the agent-side MCP
path works end to end. The dummy answer is wrong on purpose, so reward 0 is expected.

Usage: python3 mcp_smoke.py [URL]   (default http://fh-mcp:8000/mcp)
"""

import asyncio
import json
import sys
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://fh-mcp:8000/mcp"


async def main():
    async with streamable_http_client(URL) as (read, write, _), ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as session:
        await session.initialize()

        async def call(tool, args):
            res = await session.call_tool(tool, args)
            print(f"--- {tool}({json.dumps(args)}) isError={res.isError} content={[c.type for c in res.content]}")
            for c in res.content:
                if c.type == "text":
                    print(c.text)
            return res

        tools = sorted(t.name for t in (await session.list_tools()).tools)
        print("tools:", tools)
        ok = tools == ["contact_sheet", "info", "submit", "view"]
        ok &= not (await call("info", {})).isError
        v = await call("view", {"t": 0, "out_width": 64})
        ok &= not v.isError and any(c.type == "image" and c.mimeType == "image/png" for c in v.content)
        ok &= not (await call("submit", {"answer": "PLUMBING-CHECK"})).isError
        ok &= (await call("submit", {"answer": "SECOND"})).isError  # must be refused
        print("SMOKE", "OK" if ok else "FAILED")
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
