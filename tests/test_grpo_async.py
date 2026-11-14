"""Async GRPO: the relay's policy mode (pinning, log-prob injection, transparent replies) and the
behaviour log-prob plumbing into rows."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "train"))

import test_expert_iteration as tei  # noqa: E402
from test_expert_iteration import log_relay, r2s, fhdata  # noqa: E402

import rollouts_to_pg as r2pg  # noqa: E402


def with_logprobs(chunks: list[dict], start_id: int = 1000) -> list[dict]:
    """vLLM's stream when logprobs were requested: every content/reasoning/tool delta carries its tokens."""
    out, n = [], start_id
    for c in chunks:
        c = json.loads(json.dumps(c))
        for ch in c.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("reasoning") or d.get("content") or d.get("tool_calls"):
                ch["logprobs"] = {"content": [{"token": f"token_id:{n}", "logprob": -0.5, "bytes": [], "top_logprobs": []},
                                              {"token": f"token_id:{n + 1}", "logprob": -0.25, "bytes": [],
                                               "top_logprobs": []}]}
                n += 2
            else:
                ch["logprobs"] = None
        if c.get("usage") is not None:
            c["usage"] = dict(c["usage"], completion_tokens=n - start_id)
        out.append(c)
    return out


@pytest.fixture
def prelay(tmp_path):
    up = tei.Upstream()
    docker = tei.FakeDocker()
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"version": 3, "name": "fh-v3"}))
    srv = log_relay.make_server(("127.0.0.1", 0), ("127.0.0.1", up.port), tmp_path / "requests",
                                log_relay.TrialResolver(docker), policy_file=pol)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield type("P", (), dict(up=up, docker=docker, port=srv.server_address[1], logs=tmp_path / "requests", pol=pol))
    srv.shutdown()
    up.srv.shutdown()


def test_policy_mode_injects_strips_and_pins(prelay):
    plain = tei.turn_chunks("think", "hello", None)
    prelay.up.script.append(tei.sse(with_logprobs(plain)))
    body = {"model": "qwen3.8-27b-fh", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    raw = json.dumps(body).encode()
    got, _ = tei.post(prelay.port, raw, "s1")
    # upstream saw the pinned version and the logprob request
    sent = json.loads(prelay.up.bodies[0])
    assert sent["model"] == "fh-v3" and sent["logprobs"] is True and sent["top_logprobs"] == 0
    assert sent["return_tokens_as_token_ids"] is True and sent["messages"] == body["messages"]
    # OpenCode gets the stream it would get without logprobs (same events, logprobs null, model unchanged)
    events = [json.loads(l[5:]) for l in got.decode().splitlines() if l.startswith("data:") and "[DONE]" not in l]
    want = [dict(c, choices=[dict(ch, logprobs=None) for ch in c["choices"]]) for c in with_logprobs(plain)]
    assert events == want and got.decode().rstrip().endswith("data: [DONE]")
    # the log keeps OpenCode's request and the behaviour log-probs; the trial is pinned to v3
    rec = tei.wait_lines(prelay.logs / "proj__env" / "log.jsonl", 1)[0]
    assert rec["request"] == body and rec["policy_version"] == 3 and rec["policy_name"] == "fh-v3"
    assert rec["response"]["logprobs"] == [[1000, -0.5], [1001, -0.25], [1002, -0.5], [1003, -0.25],
                                           [1004, -0.5], [1005, -0.25]]
    assert rec["response"]["completion_tokens"] == 6 and rec["response"]["content"] == "hello"
    # a new version does not move a running trial; a new trial gets it
    prelay.pol.write_text(json.dumps({"version": 4, "name": "fh-v4"}))
    tei.post(prelay.port, raw, "s1")
    assert json.loads(prelay.up.bodies[1])["model"] == "fh-v3"
    prelay.docker.project = "other__env"
    tei.post(prelay.port, raw, "s2")
    assert json.loads(prelay.up.bodies[2])["model"] == "fh-v4"
    # other models pass through untouched
    tei.post(prelay.port, json.dumps(dict(body, model="x")).encode(), "s2")
    assert prelay.up.bodies[3] == json.dumps(dict(body, model="x")).encode()


def test_stripper_handles_split_events():
    ev = b'data: {"model":"fh-v1","choices":[{"index":0,"delta":{"content":"a"},"logprobs":{"content":[]}}]}\n\n'
    s = log_relay.SSEStripper("qwen3.8-27b-fh")
    out = s.feed(ev[:17]) + s.feed(ev[17:] + b"data: [DONE]\n\n") + s.flush()
    assert out == (b'data: {"model":"qwen3.8-27b-fh","choices":[{"index":0,"delta":{"content":"a"},"logprobs":null}]}'
                   b'\n\ndata: [DONE]\n\n')


def test_rows_carry_behaviour_logprobs_per_assistant_turn(tmp_path, monkeypatch):
    # rollout_run drives the relay with the default (non-policy) server; patch make_server to policy mode
    pol = tmp_path / "policy.json"
    pol.write_text(json.dumps({"version": 7, "name": "fh-v7"}))
    orig = log_relay.make_server
    monkeypatch.setattr(log_relay, "make_server", lambda *a, **k: orig(*a, **k, policy_file=pol))
    orig_sse = tei.sse
    monkeypatch.setattr(tei, "sse", lambda chunks: orig_sse(with_logprobs(chunks)))
    item = "logstream-100301"
    trials = [(f"{item}__T{i}", item, r, tei.script() if r else tei.script("WRONG")) for i, r in enumerate([1.0, 0.0])]
    run, _ = tei.rollout_run(tmp_path, "ag-x", trials)
    out = tmp_path / "pg"
    assert r2pg.main([str(run), "--out", str(out), "--min-group", "2", "--behavior"]) == 0
    rows = [json.loads(l) for l in (out / "sft.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        n_asst = sum(m["role"] == "assistant" for m in r["messages"])
        assert len(r["behavior"]) == n_asst and all(b is not None and b["complete"] for b in r["behavior"])
        assert r["policy_version"] == 7 and r["policy_versions"] == [7]
        assert r["behavior"][0]["lp"][:2] == [-0.5, -0.25] and isinstance(r["behavior"][0]["ids"][0], int)
