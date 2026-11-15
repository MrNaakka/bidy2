"""Replay logged OpenCode requests against a vLLM with logprobs on: does every generated token get a
log-prob (reasoning, content and tool-call tokens, under the qwen3 reasoning and qwen3_coder tool parsers)?

    python3 check_logprobs.py --port 8203 --model fh-v0 --n 6 /root/fhdata/grpo/it1/requests-g0
"""

import argparse
import http.client
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import log_relay  # noqa: E402


def one(port, key, model, body):
    b = {**body, "model": model, "stream": True, "logprobs": True, "top_logprobs": 0,
         "return_tokens_as_token_ids": True, "stream_options": {"include_usage": True}}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
    c.request("POST", "/v1/chat/completions", body=json.dumps(b).encode(),
              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    raw = c.getresponse().read()
    r = log_relay.assemble_sse(raw)
    return {"completion_tokens": (r.get("usage") or {}).get("completion_tokens"), "logprobs": len(r.get("logprobs") or []),
            "finish": r.get("finish_reason"), "tool_calls": [t["function"]["name"] for t in r["tool_calls"]],
            "reasoning_chars": len(r.get("reasoning") or ""), "content_chars": len(r.get("content") or "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--port", type=int, default=8203)
    ap.add_argument("--model", default="fh-v0")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    key = Path("/root/fh/vllm.env").read_text().strip().split("=", 1)[1]
    bodies = []
    for d in a.dirs:
        for t in sorted(d.iterdir()):
            recs = [json.loads(l) for l in (t / "log.jsonl").read_text().splitlines() if l.strip()]
            recs = [r for r in recs if isinstance(r.get("request"), dict) and r["request"].get("tools")]
            if recs:
                bodies.append(log_relay.restore(random.Random(t.name).choice(recs)["request"], t))
    random.Random(0).shuffle(bodies)
    with ThreadPoolExecutor(a.n) as ex:
        res = list(ex.map(lambda b: one(a.port, key, a.model, b), bodies[:a.n]))
    for r in res:
        print(json.dumps(r))
    ok = sum(r["logprobs"] == r["completion_tokens"] for r in res)
    print(f"{ok}/{len(res)} responses have a log-prob for every generated token")


if __name__ == "__main__":
    main()
