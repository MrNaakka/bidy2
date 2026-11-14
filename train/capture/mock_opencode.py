"""Scripted OpenAI-compatible mock LLM that logs every request OpenCode sends.

    python train/capture/mock_opencode.py --port 19190 --log train/capture/requests

Used to capture OpenCode's exact chat-completions format (system prompt, tool list,
how MCP tool results and images come back, assistant tool-call messages) so the SFT
data in train/make_sft.py matches what Qwen sees at eval time. Extends the idea of
step0/mockllm/mock.py.

Every POST body is written to <log>/req_NN.json. Requests that carry the FrameHunt
tools get the next scripted step; any other request (e.g. OpenCode's title
generation, which has no tools) gets a short text answer. The step is chosen from the
number of tool results already in the conversation, so the mock is stateless.

Each scripted reply streams a `reasoning` delta (vLLM's --reasoning-parser field),
a short `content` sentence and one tool call, like Qwen behind vLLM does.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# (reasoning, content, tool suffix, arguments). Arguments are valid for every
# FrameHunt item (1920x1080, video at least 3 minutes long).
SCRIPT = [
    ("I should read the question and budget first.", "Reading the task info.", "info", {}),
    ("Take a cheap overview of the first frame.", "Low-res overview of the first frame.", "view",
     {"t": 0, "out_width": 240}),
    ("Check a cheap contact sheet of the bottom-right corner over the first 20 s.",
     "Contact sheet of the bottom-right corner.", "contact_sheet",
     {"t_start": 0, "t_end": 20, "n": 4, "x": 1180, "y": 970, "w": 730, "h": 100, "out_width": 91}),
    ("Read a small native-resolution crop.", "Native crop.", "view",
     {"t": 10, "x": 0, "y": 0, "w": 160, "h": 40, "out_width": 160}),
    ("Submit a placeholder answer to capture the submit format.", "Submitting.", "submit",
     {"answer": "CAPTURE"}),
]

counter = itertools.count()


def chunk(delta: dict, finish: str | None = None, usage: dict | None = None) -> dict:
    c = {"id": "chatcmpl-mock", "object": "chat.completion.chunk", "created": int(time.time()),
         "model": "qwen3.8-27b", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        c["usage"] = usage
    return c


def find_tool(tools: list[dict], suffix: str) -> str | None:
    names = [t.get("function", {}).get("name", "") for t in tools]
    for n in names:
        if n == suffix or n.endswith("_" + suffix):
            return n
    return None


class Handler(BaseHTTPRequestHandler):
    log_dir = "."

    def log_message(self, fmt, *args):  # quiet
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"object": "list", "data": [{"id": "qwen3.8-27b", "object": "model"}]}).encode())

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("content-length", 0)))
        n = next(counter)
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            body = {"_raw": raw.decode("utf-8", "replace")}
        with open(os.path.join(self.log_dir, f"req_{n:02d}.json"), "w") as f:
            json.dump({"path": self.path, "headers": {k: v for k, v in self.headers.items()
                                                      if k.lower() not in ("authorization",)},
                       "body": body}, f, indent=1)
        msgs = body.get("messages", []) if isinstance(body, dict) else []
        tools = body.get("tools", []) if isinstance(body, dict) else []
        n_results = sum(1 for m in msgs if m.get("role") == "tool")
        events: list[dict] = []
        name = None
        if tools and n_results < len(SCRIPT):
            reasoning, content, suffix, args = SCRIPT[n_results]
            name = find_tool(tools, suffix)
        if name:
            events.append(chunk({"role": "assistant", "content": ""}))
            events.append(chunk({"reasoning": reasoning}))
            events.append(chunk({"content": content}))
            events.append(chunk({"tool_calls": [{"index": 0, "id": f"call_{n_results:02d}", "type": "function",
                                                 "function": {"name": name, "arguments": ""}}]}))
            events.append(chunk({"tool_calls": [{"index": 0, "function": {"arguments": json.dumps(args)}}]}))
            events.append(chunk({}, "tool_calls"))
        else:
            text = "FrameHunt capture" if not tools else "Done: the answer was submitted."
            events.append(chunk({"role": "assistant", "content": text}))
            events.append(chunk({}, "stop"))
        events.append(chunk({}, None, {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020}))
        events[-1]["choices"] = []
        if body.get("stream", True):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for e in events:
                self.wfile.write(f"data: {json.dumps(e)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:  # non-streaming fallback
            msg: dict = {"role": "assistant", "content": text if not name else content}
            if name:
                msg["reasoning"] = reasoning
                msg["tool_calls"] = [{"id": f"call_{n_results:02d}", "type": "function",
                                      "function": {"name": name, "arguments": json.dumps(args)}}]
            out = {"id": "chatcmpl-mock", "object": "chat.completion", "created": int(time.time()),
                   "model": "qwen3.8-27b",
                   "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if name else "stop"}],
                   "usage": {"prompt_tokens": 1000, "completion_tokens": 20, "total_tokens": 1020}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=19190)
    ap.add_argument("--log", default="train/capture/requests")
    a = ap.parse_args()
    os.makedirs(a.log, exist_ok=True)
    Handler.log_dir = a.log
    print(f"mock listening on {a.host}:{a.port}, logging to {a.log}", flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
