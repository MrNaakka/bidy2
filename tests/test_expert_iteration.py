"""Expert-iteration plumbing: the request-logging relay, rollouts -> SFT rows, merging, train-split tasks.

Everything is local and mocked: a fake vLLM upstream, a fake `docker inspect`, synthetic Harbor trial dirs.
With FH_CAPTURE_REQUESTS=<dir of captured OpenCode requests> (train/capture/requests, git-ignored) a
captured request is also round-tripped.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infra" / "verda"))
sys.path.insert(0, str(ROOT / "train"))
import fhdata  # noqa: E402
import fh  # noqa: E402
import log_relay  # noqa: E402
import merge_sft  # noqa: E402
import rollouts_to_sft as r2s  # noqa: E402
from test_fh_eval import eval_args, evalbox, fleet_state, verda  # noqa: E402,F401
from test_fh_infra import sandbox  # noqa: E402,F401

from framehunt import splits  # noqa: E402

KEY = "sk-secret-key-0123456789"


# ---------------------------------------------------------------------- fake vLLM
def sse(chunks: list[dict]) -> list[bytes]:
    return [f"data: {json.dumps(c)}\n\n".encode() for c in chunks] + [b"data: [DONE]\n\n"]


def turn_chunks(reasoning: str, content: str, call: tuple[str, str, dict] | None) -> list[dict]:
    """A Qwen-behind-vLLM style stream: reasoning deltas, content, one tool call split over deltas."""
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "qwen3.8-27b-fh"}
    out = [{**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}]
    half = len(reasoning) // 2
    for r in (reasoning[:half], reasoning[half:]):
        out.append({**base, "choices": [{"index": 0, "delta": {"reasoning": r, "reasoning_content": r}}]})
    if content:
        out.append({**base, "choices": [{"index": 0, "delta": {"content": content}}]})
    if call:
        cid, name, args = call
        a = json.dumps(args)   # vLLM's spacing; OpenCode re-serializes compactly
        out.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": cid, "type": "function", "function": {"name": name, "arguments": a[:5]}}]}}]})
        out.append({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": a[5:]}}]}}]})
    out.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if call else "stop"}]})
    out.append({**base, "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
    return out


class Upstream:
    """Records raw request bodies; answers with the next scripted SSE stream (chunked, flushed per event)."""

    def __init__(self):
        self.bodies, self.headers, self.script, self.gate = [], [], [], None
        up = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_GET(self):
                b = json.dumps({"data": [{"id": "qwen3.8-27b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                up.bodies.append(body)
                up.headers.append(dict(self.headers))
                events = up.script.pop(0) if up.script else sse(turn_chunks("ok", "done", None))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for i, e in enumerate(events):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(e), e))
                    self.wfile.flush()
                    if i == 0 and up.gate is not None:
                        assert up.gate.wait(10), "client never saw the first event"
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.port = self.srv.server_address[1]


class FakeDocker:
    def __init__(self):
        self.project, self.calls = "proj__env", 0

    def __call__(self):
        self.calls += 1
        return [{"Name": "/c-main-1", "Config": {"Labels": {"com.docker.compose.project": self.project}},
                 "NetworkSettings": {"Networks": {"n": {"IPAddress": "127.0.0.1"}}}}]


@pytest.fixture
def relay(tmp_path):
    up = Upstream()
    docker = FakeDocker()
    srv = log_relay.make_server(("127.0.0.1", 0), ("127.0.0.1", up.port), tmp_path / "requests",
                                log_relay.TrialResolver(docker))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield SimpleNamespace(up=up, docker=docker, port=srv.server_address[1], logs=tmp_path / "requests")
    srv.shutdown()
    up.srv.shutdown()


def post(port: int, body: bytes, session: str, conn=None) -> tuple[bytes, dict]:
    conn = conn or http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("POST", "/v1/chat/completions", body=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}", "x-opencode-session-id": session,
        "User-Agent": "opencode/1.19.5"})
    r = conn.getresponse()
    return r.read(), dict(r.getheaders())


def wait_lines(path: Path, n: int, timeout: float = 10) -> list[dict]:
    """The relay appends a request's log line right after the response has been passed on."""
    end = time.time() + timeout
    while time.time() < end:
        if path.is_file():
            lines = [l for l in path.read_text().splitlines() if l.strip()]
            if len(lines) >= n:
                return [json.loads(l) for l in lines]
        time.sleep(0.02)
    raise AssertionError(f"{path}: fewer than {n} log lines")


def png_bytes(seed: int, size=(40, 20)) -> bytes:
    img = Image.new("RGB", size, (seed * 37 % 256, seed * 91 % 256, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------- relay
def test_relay_forwards_bytes_unchanged_and_logs_the_reassembled_stream(relay):
    big = base64.b64encode(os.urandom(1_500_000)).decode()   # a large base64 image, like OpenCode sends
    req = {"model": "qwen3.8-27b-fh", "stream": True, "tools": fhdata.tools(), "messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": [{"type": "text", "text": fhdata.ATTACHED_MEDIA_TEXT},
                                     {"type": "image_url", "image_url": {"url": "data:image/png;base64," + big}}]}]}
    raw = json.dumps(req, indent=1).encode()   # exact bytes, not re-serialized
    events = sse(turn_chunks("Look at the counter. Then zoom.", "", ("call_7", "framehunt_view",
                                                                      {"t": 12.5, "out_width": 160})))
    relay.up.script.append(events)
    got, headers = post(relay.port, raw, "ses_a")
    assert relay.up.bodies == [raw]                                     # request body byte-identical
    assert relay.up.headers[0]["Authorization"] == f"Bearer {KEY}"       # forwarded ...
    assert got == b"".join(events)                                      # response bytes unchanged
    assert headers["Content-Type"] == "text/event-stream"
    d = relay.logs / "proj__env"
    rec = wait_lines(d / "log.jsonl", 1)[0]
    resp = rec["response"]
    assert resp["reasoning"] == "Look at the counter. Then zoom." and resp["content"] == ""
    assert resp["tool_calls"] == [{"id": "call_7", "type": "function", "function": {
        "name": "framehunt_view", "arguments": json.dumps({"t": 12.5, "out_width": 160})}}]
    assert resp["finish_reason"] == "tool_calls" and resp["done"] and resp["usage"]["completion_tokens"] == 5
    assert rec["session"] == "ses_a" and rec["container"] == "c-main-1" and rec["status"] == 200
    assert log_relay.restore(rec["request"], d) == req                  # lossless storage
    assert rec["request"]["tools"].startswith("fhtools:") and len(list((d / "blobs").iterdir())) == 1
    for p in relay.logs.rglob("*"):                                     # ... but never logged
        if p.is_file():
            assert KEY.encode() not in p.read_bytes() and b"Authorization" not in p.read_bytes()


def test_relay_streams_live(relay):
    """The client gets the first SSE event while vLLM is still generating (no buffering)."""
    relay.up.gate = threading.Event()
    relay.up.script.append(sse(turn_chunks("r", "hello", None)))
    conn = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=30)
    conn.request("POST", "/v1/chat/completions", body=b'{"messages": []}',
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    first = r.read1(65536)
    assert first.startswith(b"data: ")
    relay.up.gate.set()
    rest = r.read()
    assert b"[DONE]" in first + rest


def test_relay_handles_concurrent_trials(relay):
    n = 15
    bodies = {i: json.dumps({"i": i, "messages": [{"role": "user", "content": "x" * (1000 + i)}],
                             "tools": [{"type": "function", "function": {"name": "t"}}]}).encode() for i in range(n)}
    results = {}

    def go(i):
        results[i] = post(relay.port, bodies[i], f"ses_{i}")[0]

    ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert sorted(relay.up.bodies) == sorted(bodies.values())
    assert all(b"[DONE]" in results[i] for i in range(n))
    recs = wait_lines(relay.logs / "proj__env" / "log.jsonl", n)
    assert sorted(r["session"] for r in recs) == sorted(f"ses_{i}" for i in range(n))


def test_relay_keeps_get_unlogged_and_resolves_per_session(relay):
    conn = http.client.HTTPConnection("127.0.0.1", relay.port, timeout=30)
    conn.request("GET", "/v1/models", headers={"Authorization": f"Bearer {KEY}"})
    assert b"qwen3.8-27b" in conn.getresponse().read()
    assert not relay.logs.exists() or not any(relay.logs.iterdir())
    post(relay.port, b'{"messages": []}', "ses_1")
    wait_lines(relay.logs / "proj__env" / "log.jsonl", 1)
    post(relay.port, b'{"messages": []}', "ses_1")
    wait_lines(relay.logs / "proj__env" / "log.jsonl", 2)
    assert relay.docker.calls == 1                       # cached for the session
    relay.docker.project = "next-trial__env"             # the IP now belongs to another trial
    post(relay.port, b'{"messages": []}', "ses_2")
    assert relay.docker.calls == 2
    assert len(wait_lines(relay.logs / "proj__env" / "log.jsonl", 2)) == 2
    assert len(wait_lines(relay.logs / "next-trial__env" / "log.jsonl", 1)) == 1


def test_unresolved_clients_and_assembly_edge_cases(tmp_path):
    r = log_relay.TrialResolver(lambda: [])
    assert r.resolve("10.0.0.9", "s") == ("unresolved-10.0.0.9", None)
    raw = b"".join(sse([{"choices": [{"index": 0, "delta": {"reasoning_content": "a"}}]},
                        {"choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": 1, "id": "c2", "function": {"name": "framehunt_submit", "arguments": "{}"}},
                            {"index": 0, "id": "c1", "function": {"name": "framehunt_info", "arguments": ""}}]}}]},
                        {"choices": [{"index": 0, "delta": {"tool_calls": [
                            {"index": 0, "function": {"name": "framehunt_info", "arguments": "{}"}}]}}]}]))
    out = log_relay.assemble_sse(raw + b": keep-alive\n\ndata: {broken\n\n")
    assert out["reasoning"] == "a" and out["bad_lines"] == 1
    assert [c["function"]["name"] for c in out["tool_calls"]] == ["framehunt_info", "framehunt_submit"]
    assert out["tool_calls"][0]["function"]["arguments"] == "{}"   # the name is taken once, like the AI SDK
    j = log_relay.assemble_json(json.dumps({"choices": [{"message": {"content": "hi", "reasoning": "r"},
                                                         "finish_reason": "stop"}]}).encode())
    assert j["content"] == "hi" and j["reasoning"] == "r" and j["finish_reason"] == "stop"


# ---------------------------------------------------------------------- rollouts -> rows
ITEM = "logstream-100301"
DATE = "Tue Nov 10 2026"
INSTRUCTION = "# FrameHunt\n\nQuestion here.\n\nYou have 120000 pixels in total. For scale, ...\n"


def script(answer="a1b2c3d4"):
    """(reasoning, content, (name, args) | None, tool result text, png | None) per assistant turn."""
    return [
        ("Read the rules.", "", ("framehunt_info", {}), "question: ...", None),
        ("Overview first.", "", ("framehunt_view", {"t": 0, "out_width": 240}), "t=0.000s frame=0 ...", png_bytes(1)),
        ("Sheet of the counter.", "", ("framehunt_contact_sheet", {"t_start": 0, "t_end": 300, "n": 8, "x": 1700,
                                                                   "y": 1040, "w": 200, "h": 30, "out_width": 40}),
         "8 frames ...", png_bytes(2)),
        ("Read the line.", "", ("framehunt_view", {"t": 151.2, "x": 0, "y": 500, "w": 600, "h": 30,
                                                   "out_width": 600}), "t=151.200s ...", png_bytes(3)),
        ("Submit it.", "", ("framehunt_submit", {"answer": answer}), "Answer recorded. ...", None),
        ("Done.", f"Submitted {answer}.", None, None, None),
    ]


def play(port: int, upstream: Upstream, session: str, steps, title_request=True) -> list[dict]:
    """Drive the relay like OpenCode: resend the whole history every turn; return the last request's messages."""
    msgs = [{"role": "system", "content": fhdata.system_prompt(DATE)}, fhdata.user_turn(INSTRUCTION)]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    if title_request:   # OpenCode's title generator: no tools, not training data
        upstream.script.append(sse(turn_chunks("", "Title", None)))
        post(port, json.dumps({"model": "m", "messages": msgs[1:]}).encode(), session, conn)
    for k, (reasoning, content, call, result, png) in enumerate(steps):
        cid = f"call_{k:02d}"
        upstream.script.append(sse(turn_chunks(reasoning, content, (cid, *call) if call else None)))
        body = {"model": "qwen3.8-27b-fh", "max_tokens": 16384, "messages": msgs, "tools": fhdata.tools(),
                "tool_choice": "auto", "stream": True}
        post(port, json.dumps(body).encode(), session, conn)
        if call is None:
            break
        msgs = msgs + [{"role": "assistant", "content": content, "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": call[0], "arguments": fhdata.args_json(call[1])}}],
            "reasoning": reasoning}, {"role": "tool", "tool_call_id": cid, "content": result}]
        if png is not None:
            msgs.append({"role": "user", "content": [
                {"type": "text", "text": fhdata.ATTACHED_MEDIA_TEXT},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]})
    return msgs


def server_calls(steps) -> list[dict]:
    """What the MCP server logs: every call with defaults filled in."""
    defaults = {"view": {"x": 0, "y": 0, "w": None, "h": None, "out_width": 320},
                "contact_sheet": {"x": 0, "y": 0, "w": None, "h": None, "out_width": 160}}
    out = []
    for _, _, call, _, _ in steps:
        if call:
            tool = call[0][len("framehunt_"):]
            args = {**defaults.get(tool, {}), **call[1]}
            if "t" in args:
                args["t"] = float(args["t"])
            out.append({"i": len(out), "tool": tool, "args": args, "ok": True})
    return out


def make_trial(run: Path, trial: str, item: str, reward: float, steps, calls=None) -> None:
    d = run / trial
    (d / "verifier").mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"task": {"path": f"tasks/train-ei/{item}"}, "trial_name": trial}))
    (d / "result.json").write_text(json.dumps({"task_name": f"framehunt/{item}", "trial_name": trial}))
    (d / "verifier" / "reward.txt").write_text(f"{reward:g}")
    (d / "verifier" / "result.json").write_text(json.dumps(
        {"submitted": True, "answer": "a1b2c3d4", "correct": reward == 1, "reward": reward, "pixels_used": 4321,
         "n_calls": 5}))
    calls = server_calls(steps) if calls is None else calls
    (d / "verifier" / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))


def rollout_run(tmp_path: Path, name: str, trials: list[tuple[str, str, float, list]]):
    """A fetched run dir: trial dirs plus requests/ logged by the real relay against a fake vLLM."""
    run = tmp_path / "runs" / name
    up, docker = Upstream(), FakeDocker()
    srv = log_relay.make_server(("127.0.0.1", 0), ("127.0.0.1", up.port), run / "requests",
                                log_relay.TrialResolver(docker))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    wires = {}
    try:
        for trial, item, reward, steps in trials:
            docker.project = r2s.harbor_project(trial)
            wires[trial] = play(srv.server_address[1], up, f"ses_{trial}", steps)
            n_steps = next(i for i, st in enumerate(steps) if st[2] is None) + 1
            wait_lines(run / "requests" / log_relay.trial_key(r2s.harbor_project(trial)) / "log.jsonl", n_steps + 1)
            make_trial(run, trial, item, reward, steps)
    finally:
        srv.shutdown()
        up.srv.shutdown()
    (run / "fh-eval-meta.json").write_text(json.dumps({"served_name": "qwen3.8-27b-fh", "lora_sha": "abc",
                                                       "fleet": "train-ei-qwen-fh", "git_commit": "c0ffee"}))
    return run, wires


def test_rollouts_become_rows_that_render_like_the_logged_request(tmp_path):
    steps = script()
    run, wires = rollout_run(tmp_path, "fh-train-ei-qwen-fh-run1", [
        ("logstream-100301__AbCdEf1", ITEM, 1.0, steps),
        ("dialogs-110301__XyZ0001", "dialogs-110301", 0.0, script("WRONG")),
        ("logstream-100302__Mismat1", "logstream-100302", 1.0, steps),
    ])
    # the third trial's server log disagrees with its conversation (a log of another trial)
    calls = server_calls(steps)
    calls[2]["args"]["t_end"] = 299.0
    (run / "logstream-100302__Mismat1" / "verifier" / "calls.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in calls))
    out = tmp_path / "ei"
    assert r2s.main([str(run), "--out", str(out)]) == 0
    rows = fhdata.read_jsonl(out / "sft.jsonl")
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "fh-train-ei-qwen-fh-run1__logstream-100301__AbCdEf1" and row["item_id"] == ITEM
    assert row["kind"] == "logstream" and row["split"] == "train" and row["correct"] and row["date"] == DATE
    assert row["budget"] == 120000 and row["n_calls"] == 5 and row["n_images"] == 3
    assert row["tools_used"][-1] == "framehunt_submit"
    p = row["provenance"]
    assert p["run"] == "fh-train-ei-qwen-fh-run1" and p["reward"] == 1.0 and p["model"] == "qwen3.8-27b-fh"
    assert p["final_turn"] and p["lora_sha"] == "abc" and p["session"] == "ses_logstream-100301__AbCdEf1"
    # the stored conversation is the logged request plus the final response, images as exact PNG files
    final = {"role": "assistant", "content": "Submitted a1b2c3d4.", "reasoning": "Done."}
    wire = wires["logstream-100301__AbCdEf1"] + [final]
    assert fhdata.to_wire(row, out) == wire
    assert row["messages"][-1] == final and row["messages"][0]["content"] == fhdata.system_prompt(DATE)
    assert (out / "images" / row["id"] / "02.png").read_bytes() == png_bytes(3)
    assert all(len(m["tool_calls"]) == 1 for m in row["messages"] if m.get("tool_calls"))
    assert (out / "tools.json").read_bytes() == fhdata.TOOLS_FILE.read_bytes()
    stats = json.loads((out / "stats.json").read_text())
    assert stats["trials"] == 3 and stats["reward_1"] == 2 and stats["rows"] == 1
    rej = fhdata.read_jsonl(out / "rejected.jsonl")
    assert [r["trial"] for r in rej] == ["logstream-100302__Mismat1"] and "call log" in rej[0]["reason"]
    # renders exactly like a make_sft row: same template messages, same images
    msgs, images = fhdata.to_template(row, out)
    assert len(images) == 3 and msgs[2]["reasoning_content"] == "Read the rules."
    assert msgs[3]["role"] == "tool" and isinstance(msgs[2]["tool_calls"][0]["function"]["arguments"], dict)


def test_dedupe_max_per_item_and_tasks_check(tmp_path):
    steps = script()
    cheaper = script()[:1] + script()[3:]   # fewer calls, same answer
    run1, _ = rollout_run(tmp_path, "run1", [("logstream-100301__AAAAAA1", ITEM, 1.0, steps)])
    run2, _ = rollout_run(tmp_path, "run2", [("logstream-100301__BBBBBB2", ITEM, 1.0, steps),
                                             ("logstream-100301__CCCCCC3", ITEM, 1.0, cheaper)])
    for t in ("logstream-100301__CCCCCC3",):
        res = json.loads((run2 / t / "verifier" / "result.json").read_text())
        (run2 / t / "verifier" / "result.json").write_text(json.dumps({**res, "pixels_used": 100}))
    out = tmp_path / "all"
    assert r2s.main([str(run1), str(run2), "--out", str(out), "--max-per-item", "0"]) == 0
    s = json.loads((out / "stats.json").read_text())
    assert s["rows"] == 2 and s["duplicates"] == 1       # the same trajectory in two runs counts once
    out = tmp_path / "one"
    assert r2s.main([str(run1), str(run2), "--out", str(out)]) == 0
    rows = fhdata.read_jsonl(out / "sft.jsonl")
    assert [r["id"] for r in rows] == ["run2__logstream-100301__CCCCCC3"]   # the cheapest
    tasks = tmp_path / "tasks"
    (tasks / ITEM).mkdir(parents=True)
    (tasks / ITEM / "instruction.md").write_text("another instruction")
    (tasks / ITEM / "task.toml").write_text('[metadata]\nsplit = "train"\n')
    out = tmp_path / "tasks-checked"
    assert r2s.main([str(run1), "--out", str(out), "--tasks", str(tasks)]) == 0
    assert json.loads((out / "stats.json").read_text())["rows"] == 0
    (tasks / ITEM / "instruction.md").write_text(INSTRUCTION)
    assert r2s.main([str(run1), "--out", str(out), "--tasks", str(tasks), "--force"]) == 0
    assert json.loads((out / "stats.json").read_text())["rows"] == 1


def test_split_guard_refuses_eval_and_dev_items(tmp_path):
    for item in (splits.EVAL[-1][0], splits.DEV[-1][0], "logstream-7001", "logstream-99", "dashboard-100001"):
        run = tmp_path / item
        make_trial(run, f"{item}__Q1w2e3r", item, 1.0, script())
        with pytest.raises(SystemExit, match="not on TRAIN items"):
            r2s.main([str(run), "--out", str(tmp_path / "o")])
        assert not (tmp_path / "o").exists()
    assert r2s.train_kind("logstream-100001") == "logstream"
    assert r2s.train_kind("dialogs-110300") == "dialogs"
    assert r2s.train_kind("logstream-110001") is None    # outside logstream's seed range


def test_format_rejects_parallel_tool_calls_and_foreign_prompts(tmp_path):
    msgs = [{"role": "system", "content": fhdata.system_prompt(DATE)}, fhdata.user_turn(INSTRUCTION),
            {"role": "assistant", "content": "", "reasoning": "r", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "framehunt_info", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "framehunt_info", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "a", "content": "x"}, {"role": "tool", "tool_call_id": "b", "content": "x"},
            {"role": "assistant", "content": "", "reasoning": "r", "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "framehunt_submit", "arguments": "{\"answer\":\"x\"}"}}]}]
    with pytest.raises(r2s.Reject, match="exactly one"):
        r2s.check_format(msgs, fhdata.tools())
    ok = msgs[:2] + msgs[5:]
    assert r2s.check_format(ok, fhdata.tools()) == DATE   # last turn's call may lack a result
    with pytest.raises(r2s.Reject, match="tool list"):
        r2s.check_format(ok, fhdata.tools()[:-1])
    bad = [{"role": "system", "content": "You are another agent. Today's date: " + DATE}] + ok[1:]
    with pytest.raises(r2s.Reject, match="system prompt"):
        r2s.check_format(bad, fhdata.tools())


@pytest.mark.skipif(not os.environ.get("FH_CAPTURE_REQUESTS"), reason="set FH_CAPTURE_REQUESTS=train/capture/requests")
def test_captured_request_round_trips(tmp_path):
    """A real OpenCode request (captured, eval item, mock model) converts and renders identically."""
    reqs = sorted(Path(os.environ["FH_CAPTURE_REQUESTS"]).glob("req_*.json"))
    body = json.loads(reqs[-1].read_text())["body"]
    wire = body["messages"]
    assert r2s.check_format(wire, body["tools"]) == fhdata.CAPTURED_DATE
    row, pngs = r2s.make_row("cap", wire, tmp_path)
    row["id"] = "cap"
    (tmp_path / "images" / "cap").mkdir(parents=True)
    for i, png in enumerate(pngs):
        (tmp_path / "images" / "cap" / f"{i:02d}.png").write_bytes(png)
    r2s.assert_renders(row, wire, body["tools"], tmp_path)
    assert fhdata.to_wire(row, tmp_path) == wire and len(pngs) == 3


# ---------------------------------------------------------------------- merge
def test_merge_keeps_rows_and_images(tmp_path):
    a, b = tmp_path / "teacher", tmp_path / "ei"
    for d, rid in ((a, "logstream-100001"), (b, "run1__logstream-100301__X")):
        (d / "images" / rid).mkdir(parents=True)
        (d / "images" / rid / "00.png").write_bytes(png_bytes(len(rid)))
        (d / "tools.json").write_bytes(fhdata.TOOLS_FILE.read_bytes())
        row = {"id": rid, "kind": "logstream", "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"images/{rid}/00.png"}}]}]}
        (d / "sft.jsonl").write_text(json.dumps(row) + "\n")
    out = tmp_path / "mix"
    assert merge_sft.main([str(a), str(b), "--out", str(out)]) == 0
    rows = fhdata.read_jsonl(out / "sft.jsonl")
    assert [r["data_source"] for r in rows] == ["teacher", "ei"]
    assert all(fhdata.load_png(r["messages"][0]["content"][0]["image_url"]["url"], out) for r in rows)
    assert json.loads((out / "stats.json").read_text())["by_source"]["ei"]["rows"] == 1
    with pytest.raises(SystemExit, match="more than one input"):
        merge_sft.main([str(a), str(_copy(a, tmp_path / "a2")), "--out", str(tmp_path / "x")])
    (b / "tools.json").write_text("[]")
    with pytest.raises(SystemExit, match="tools.json differs"):
        merge_sft.main([str(a), str(b), "--out", str(tmp_path / "y")])


def _copy(src: Path, dst: Path) -> Path:
    import shutil
    shutil.copytree(src, dst)
    return dst


# ---------------------------------------------------------------------- train-split tasks
def _build_tasks():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import build_tasks
    finally:
        sys.path.pop(0)
    return build_tasks


def test_build_train_split_tasks(tmp_path, monkeypatch):
    bt = _build_tasks()
    monkeypatch.setenv("FH_SALT_TRAIN", "test-train-salt")
    items = bt.train_subset(2, 300, ["logstream", "dialogs"])
    assert [i for i, _, _ in items] == ["logstream-100301", "logstream-100302", "dialogs-110301", "dialogs-110302"]
    eval_kinds = list(dict.fromkeys(k for _, k, _ in splits.EVAL))
    out = tmp_path / "train-ei"
    assert bt.main(["--split", "train", "--per-kind", "1", "--offset", "300", "--out", str(out),
                    "--only", "logstream-100301"]) == 0
    d = out / "logstream-100301"
    assert 'split = "train"' in (d / "task.toml").read_text()
    seed = splits.derive_seed("train", "logstream-100301", "test-train-salt")
    assert f'FH_SEED: "{seed}"' in (d / "environment" / "docker-compose.yaml").read_text()
    with pytest.raises(SystemExit):
        bt.main(["--split", "train", "--out", str(out)])                       # --per-kind required
    with pytest.raises(SystemExit):
        bt.main(["--split", "eval", "--per-kind", "3", "--out", str(out)])     # train-only options
    assert all(k in splits.TRAIN_KINDS for k in eval_kinds)


# ---------------------------------------------------------------------- fleet flag
def test_log_requests_is_opt_in(evalbox):
    d = fh.eval_definition(eval_args(evalbox))
    assert "log_requests" not in d and "log_relay_sha" not in d
    args = fh.eval_launch_args(d, Path("/s"), 1)
    assert not any(x.startswith("LOG_REQUESTS") for x in args)
    stage = fh.stage_eval(d)
    assert not (stage / "log_relay.py").exists()

    d = fh.eval_definition(eval_args(evalbox, log_requests=True))
    assert d["log_requests"] and d["log_relay_sha"]
    assert "LOG_REQUESTS=1" in fh.eval_launch_args(d, Path("/s"), 1)
    stage = fh.stage_eval(d, "x")
    assert (stage / "log_relay.py").read_bytes() == fh.LOG_RELAY.read_bytes()
    job = fh.EVAL_JOB.read_text()
    assert 'if [ "${LOG_REQUESTS:-0}" = 1 ]' in job and "socat TCP-LISTEN:18000,bind=172.17.0.1" in job


def test_fetch_copies_request_logs_only_when_logging(evalbox, sandbox, monkeypatch):
    monkeypatch.setattr(fh, "ssh_base", lambda c: ["ssh"])
    monkeypatch.setattr(fh, "cluster_cost_since", lambda n, s: 0.0)
    for logging in (False, True):
        fleet = fleet_state(evalbox, ["running"])
        if logging:
            fleet["meta"]["log_requests"] = True
        calls = []
        monkeypatch.setattr(fh.subprocess, "run", lambda cmd, **k: calls.append(cmd) or SimpleNamespace(returncode=0))
        fh.fetch_eval(fleet, fleet["boxes"]["1"], "RUNNING")
        req = [c for c in calls if c[-1].endswith("/requests/")]
        if logging:
            assert len(req) == 1 and req[0][-2] == "fh-ev-dev-qwen-1:fh/requests/fh-dev-qwen-run1/"
            assert "requests/" in calls[0]      # the run-dir sync must not delete the fetched logs
        else:
            assert not req and "requests/" not in calls[0] and len(calls) == 2
