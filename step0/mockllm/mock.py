"""Mock OpenAI-compatible chat-completions endpoint (streaming) for Step-0.

Turn 1: asks OpenCode to call the MCP tool `probe_show_card`.
Turn 2+: dumps the full request body to /log so we can see how the tool result
(the PNG) was forwarded to the "model", then ends the conversation.
"""
import itertools
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOG = "/log"
os.makedirs(LOG, exist_ok=True)
counter = itertools.count()


def chunk(delta, finish=None):
    return {"id": "c1", "object": "chat.completion.chunk", "created": int(time.time()),
            "model": "mock", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}]}).encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        n = next(counter)
        with open(f"{LOG}/req_{n:02d}.json", "w") as f:
            json.dump(body, f, indent=1)
        msgs = body.get("messages", [])
        tools = [t["function"]["name"] for t in body.get("tools", [])]
        has_tool_result = any(m.get("role") == "tool" for m in msgs)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream"); self.end_headers()
        if "probe_show_card" in tools and not has_tool_result:
            events = [chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                      "function": {"name": "probe_show_card", "arguments": "{}"}}]}),
                      chunk({}, "tool_calls")]
        else:
            events = [chunk({"role": "assistant", "content": "done"}), chunk({}, "stop")]
        for e in events:
            self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


ThreadingHTTPServer(("0.0.0.0", 9000), H).serve_forever()
