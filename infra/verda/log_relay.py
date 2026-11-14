#!/usr/bin/env python3
"""Request-logging relay between Harbor's trial containers and vLLM (opt-in; replaces socat).

    python3 log_relay.py --listen 172.17.0.1:18000 --upstream 127.0.0.1:8000 --log-dir ~/fh/requests/$JOB

Started by eval_job.sh when LOG_REQUESTS=1 (`fh eval-up --log-requests`); otherwise the job runs the plain
socat relay. Python standard library only (the box's system python3).

Forwarding: every request goes to vLLM with the same method, path, headers (minus hop-by-hop ones) and
the exact body bytes; the response status, headers and body bytes come back unchanged, streamed chunk by
chunk as vLLM produces them (SSE streams stay live). One thread per client connection (OpenCode keeps
connections alive), one upstream connection per client connection.

Logging (POST .../chat/completions only): after the response has been passed on, one JSON line is
appended to <log-dir>/<trial>/log.jsonl with the request body, the response reassembled from the stream
the way OpenCode's AI SDK reads it (content, reasoning, tool calls, finish reason, usage), timings and
sha256 of the raw request and response bytes. If the client disconnects mid-response the relay closes the
upstream connection (vLLM aborts the generation, as with socat) and logs the partial response with
client_ok=false. The API key (Authorization header) is never logged; only
User-Agent and OpenCode's session headers are.

To keep the logs linear in trial length (every request repeats the whole conversation, images
included), images are stored once per trial: a `data:image/...;base64,...` URL becomes
`fhblob:<mime>:<sha256>` and the decoded bytes go to <trial>/blobs/<sha256>, and the tool list becomes
`fhtools:<sha256>` with <trial>/tools/<sha256>.json. This is only done when it is lossless (re-encoding
gives the same base64 text and restoring gives the same JSON); `restore()` undoes it exactly.

Async GRPO mode (--policy-file F): every chat request that names the public model (qwen3.8-27b-fh) is
rewritten before forwarding: `model` becomes the policy version the trial is pinned to (F holds
{"version": N, "name": "fh-vN"}; a trial is pinned to the version current at its first request, so one
trajectory is sampled from one policy), and "logprobs": true, "top_logprobs": 0,
"return_tokens_as_token_ids": true are added. The response goes back to OpenCode with the logprobs removed
(set to null, as vLLM sends them when not asked) and `model` restored, so OpenCode sees a normal reply. The
log keeps the request exactly as OpenCode sent it, plus "policy_version" and the behaviour log-probs of
every generated token (response.logprobs = [[token_id, logprob], ...], response.completion_tokens).

Trial identification: OpenCode sends no trial id, so the relay maps the client's IP to its container
(`docker inspect` of the running containers) and uses the container's Docker Compose project, which
Harbor names after the trial (`<trial_name>__env`, lowercased), looked up when the request arrives. The lookup is cached per (IP, OpenCode
session id); a new session (a new trial, possibly reusing a finished trial's IP) always looks up again.
Unresolved clients log to `unresolved-<ip>/`. train/rollouts_to_sft.py maps <trial> back to Harbor's
trial dirs and checks every trajectory against the MCP server's own call log.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
import hashlib
import http.client
import json
import os
import re
import select
import socket
import subprocess
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection", "te",
              "trailer", "trailers", "transfer-encoding", "upgrade"}
LOGGED_HEADERS = ("user-agent", "x-opencode-session-id", "x-session-id", "x-session-affinity")
DATA_URL_RE = re.compile(r"data:(image/[A-Za-z0-9.+-]+);base64,(.*)", re.S)
BLOB_RE = re.compile(r"fhblob:(image/[A-Za-z0-9.+-]+):([0-9a-f]{64})")
TOOLS_RE = re.compile(r"fhtools:([0-9a-f]{64})")
KEY_RE = re.compile(r"[^a-z0-9_.-]")
CHUNK = 64 * 1024
UPSTREAM_TIMEOUT_S = 3600


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), "log_relay:", msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------- reassembly
def assemble_sse(raw: bytes) -> dict:
    """Reassemble an OpenAI-style chat.completion.chunk stream (choice 0) like the AI SDK does.

    Content and reasoning deltas are concatenated; reasoning is `delta.reasoning_content`, else
    `delta.reasoning` (vLLM may send both with the same text). A tool call's id, type and name come from
    its first delta (per `index`), its arguments from all deltas.
    """
    content, reasoning, calls, lps = [], [], {}, []
    out: dict = {"stream": True, "finish_reason": None, "usage": None, "done": False, "bad_lines": 0}
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            out["done"] = True
            continue
        try:
            ch = json.loads(data)
        except ValueError:
            out["bad_lines"] += 1
            continue
        if not isinstance(ch, dict):
            out["bad_lines"] += 1
            continue
        if ch.get("error") is not None:
            out["error"] = ch["error"]
        if ch.get("usage"):
            out["usage"] = ch["usage"]
        for key in ("id", "model"):
            if ch.get(key) and key not in out:
                out[key] = ch[key]
        for c in ch.get("choices") or []:
            if c.get("index", 0) != 0:
                continue
            d = c.get("delta") or {}
            if isinstance(d.get("content"), str):
                content.append(d["content"])
            r = d.get("reasoning_content")
            if not isinstance(r, str):
                r = d.get("reasoning")
            if isinstance(r, str):
                reasoning.append(r)
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                f = tc.get("function") or {}
                if i not in calls:
                    calls[i] = {"id": tc.get("id"), "type": tc.get("type") or "function",
                                "function": {"name": f.get("name") or "", "arguments": ""}}
                calls[i]["function"]["arguments"] += f.get("arguments") or ""
            for t in ((c.get("logprobs") or {}).get("content") or []):
                tok = t.get("token") or ""
                lps.append([int(tok[9:]) if tok.startswith("token_id:") else tok, t.get("logprob")])
            if c.get("finish_reason"):
                out["finish_reason"] = c["finish_reason"]
    if lps:
        out["logprobs"] = lps
    out["content"] = "".join(content)
    out["reasoning"] = "".join(reasoning) if reasoning else None
    out["tool_calls"] = [calls[i] for i in sorted(calls)]
    return out


def logprobs_json(raw: bytes) -> list | None:
    try:
        ch = (json.loads(raw).get("choices") or [{}])[0]
    except (ValueError, AttributeError):
        return None
    out = []
    for t in ((ch.get("logprobs") or {}).get("content") or []):
        tok = t.get("token") or ""
        out.append([int(tok[9:]) if tok.startswith("token_id:") else tok, t.get("logprob")])
    return out or None


def assemble_json(raw: bytes) -> dict:
    """A non-streamed chat.completion (choice 0) in the same shape as assemble_sse."""
    try:
        body = json.loads(raw)
    except ValueError:
        return {"stream": False, "error": "response is not JSON"}
    if not isinstance(body, dict):
        return {"stream": False, "error": "response is not a JSON object"}
    out = {"stream": False, "usage": body.get("usage"), "id": body.get("id"), "model": body.get("model")}
    if body.get("error") is not None:
        out["error"] = body["error"]
    ch = (body.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    r = m.get("reasoning_content")
    out.update(content=m.get("content") or "", reasoning=r if isinstance(r, str) else m.get("reasoning"),
               tool_calls=[{"id": t.get("id"), "type": t.get("type") or "function",
                            "function": {"name": (t.get("function") or {}).get("name") or "",
                                         "arguments": (t.get("function") or {}).get("arguments") or ""}}
                           for t in m.get("tool_calls") or []],
               finish_reason=ch.get("finish_reason"))
    return out


def decode_body(raw: bytes, encoding: str | None) -> bytes | None:
    enc = (encoding or "identity").strip().lower()
    try:
        if enc in ("", "identity"):
            return raw
        if enc == "gzip":
            return gzip.decompress(raw)
        if enc == "deflate":
            try:
                return zlib.decompress(raw)
            except zlib.error:
                return zlib.decompress(raw, -zlib.MAX_WBITS)
    except (OSError, EOFError, zlib.error):
        return None
    return None   # br / zstd: forwarded untouched, not parsed


# ---------------------------------------------------------------------- async GRPO rewriting
PUBLIC_MODEL = "qwen3.8-27b-fh"


class Policy:
    """The current policy version (from --policy-file) and the version each trial is pinned to."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._pins: dict[str, dict] = {}

    def current(self) -> dict:
        return json.loads(self.path.read_text())

    def for_trial(self, key: str) -> dict:
        with self._lock:
            if key not in self._pins:
                self._pins[key] = self.current()
            return self._pins[key]


def rewrite_request(body: bytes, pol: dict) -> bytes | None:
    """The body to forward (model -> the pinned version, logprobs requested), or None to forward as is."""
    try:
        b = json.loads(body)
    except ValueError:
        return None
    if not isinstance(b, dict) or b.get("model") != PUBLIC_MODEL:
        return None
    b = {**b, "model": pol["name"], "logprobs": True, "top_logprobs": 0, "return_tokens_as_token_ids": True}
    return json.dumps(b, ensure_ascii=False, separators=(",", ":")).encode()


def strip_event(ev: bytes, model: str) -> bytes:
    """One SSE event (or a JSON body) with the logprobs removed and `model` restored."""
    m = re.match(rb"(\s*data:\s?)(.*?)(\s*)$", ev, re.S)
    if not m or m.group(2).strip() in (b"", b"[DONE]") or not m.group(2).lstrip().startswith(b"{"):
        return ev
    try:
        ch = json.loads(m.group(2))
    except ValueError:
        return ev
    return m.group(1) + strip_obj(ch, model) + m.group(3)


def strip_obj(ch, model: str) -> bytes:
    if isinstance(ch, dict):
        if "model" in ch:
            ch["model"] = model
        for c in ch.get("choices") or []:
            if isinstance(c, dict) and "logprobs" in c:
                c["logprobs"] = None
    return json.dumps(ch, ensure_ascii=False, separators=(",", ":")).encode()


class SSEStripper:
    """Feed upstream bytes, get back whole events with the logprobs stripped (an event ends with a blank line)."""

    def __init__(self, model: str):
        self.model = model
        self.buf = b""

    def feed(self, piece: bytes) -> bytes:
        self.buf += piece
        out = []
        while True:
            i = self.buf.find(b"\n\n")
            if i < 0:
                break
            ev, self.buf = self.buf[:i], self.buf[i + 2:]
            out.append(strip_event(ev, self.model) + b"\n\n")
        return b"".join(out)

    def flush(self) -> bytes:
        rest, self.buf = self.buf, b""
        return strip_event(rest, self.model) if rest else b""


# ---------------------------------------------------------------------- blob store
def _write_once(path: Path, data: bytes) -> None:
    if path.exists():
        return
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def externalize(body, trial_dir: Path):
    """Images -> fhblob refs, tools -> fhtools ref (files under trial_dir). Lossless or unchanged."""
    if not isinstance(body, dict):
        return body
    blobs = trial_dir / "blobs"
    out = dict(body)
    if isinstance(body.get("tools"), list):
        tb = json.dumps(body["tools"], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        h = hashlib.sha256(tb).hexdigest()
        (trial_dir / "tools").mkdir(parents=True, exist_ok=True)
        _write_once(trial_dir / "tools" / f"{h}.json", tb)
        out["tools"] = f"fhtools:{h}"
    msgs = body.get("messages")
    if isinstance(msgs, list):
        new_msgs = []
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), list):
                parts = []
                for p in m["content"]:
                    url = (p.get("image_url") or {}).get("url") if isinstance(p, dict) and p.get("type") == "image_url" \
                        and isinstance(p.get("image_url"), dict) else None
                    mt = DATA_URL_RE.fullmatch(url) if isinstance(url, str) else None
                    if mt:
                        try:
                            data = base64.b64decode(mt.group(2), validate=True)
                        except (binascii.Error, ValueError):
                            data = None
                        if data is not None and base64.b64encode(data).decode("ascii") == mt.group(2):
                            h = hashlib.sha256(data).hexdigest()
                            blobs.mkdir(parents=True, exist_ok=True)
                            _write_once(blobs / h, data)
                            p = {**p, "image_url": {**p["image_url"], "url": f"fhblob:{mt.group(1)}:{h}"}}
                    parts.append(p)
                m = {**m, "content": parts}
            new_msgs.append(m)
        out["messages"] = new_msgs
    return out


def restore(body, trial_dir: Path):
    """Inverse of externalize(): the request body exactly as the client sent it (as JSON)."""
    if not isinstance(body, dict):
        return body
    out = dict(body)
    t = body.get("tools")
    if isinstance(t, str) and TOOLS_RE.fullmatch(t):
        out["tools"] = json.loads((trial_dir / "tools" / f"{TOOLS_RE.fullmatch(t).group(1)}.json").read_bytes())
    msgs = body.get("messages")
    if isinstance(msgs, list):
        new_msgs = []
        for m in msgs:
            if isinstance(m, dict) and isinstance(m.get("content"), list):
                parts = []
                for p in m["content"]:
                    url = (p.get("image_url") or {}).get("url") if isinstance(p, dict) and p.get("type") == "image_url" \
                        and isinstance(p.get("image_url"), dict) else None
                    mt = BLOB_RE.fullmatch(url) if isinstance(url, str) else None
                    if mt:
                        data = (trial_dir / "blobs" / mt.group(2)).read_bytes()
                        url = f"data:{mt.group(1)};base64," + base64.b64encode(data).decode("ascii")
                        p = {**p, "image_url": {**p["image_url"], "url": url}}
                    parts.append(p)
                m = {**m, "content": parts}
            new_msgs.append(m)
        out["messages"] = new_msgs
    return out


# ---------------------------------------------------------------------- trial lookup
def default_docker_inspect() -> list[dict]:
    ids = subprocess.run(["docker", "ps", "-q", "--no-trunc"], capture_output=True, text=True, timeout=30,
                         check=True).stdout.split()
    if not ids:
        return []
    r = subprocess.run(["docker", "inspect", *ids], capture_output=True, text=True, timeout=60, check=False)
    # a container can exit between ps and inspect: inspect still prints the others
    try:
        return json.loads(r.stdout or "[]")
    except ValueError:
        return []


class TrialResolver:
    """Client IP -> (trial key, container name), via the containers' IPs and compose project labels."""

    def __init__(self, inspect=default_docker_inspect):
        self._inspect = inspect
        self._lock = threading.Lock()
        self._by_ip: dict[str, tuple[str, str]] = {}
        self._sessions: dict[tuple[str, str], tuple[str, str]] = {}

    def _refresh(self) -> None:
        by_ip = {}
        try:
            containers = self._inspect()
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            log(f"docker inspect failed: {e}")
            containers = []
        for c in containers:
            labels = (c.get("Config") or {}).get("Labels") or {}
            project = labels.get("com.docker.compose.project")
            name = (c.get("Name") or "").lstrip("/")
            for net in ((c.get("NetworkSettings") or {}).get("Networks") or {}).values():
                ip = (net or {}).get("IPAddress")
                if ip and project:
                    by_ip[ip] = (project, name)
        self._by_ip = by_ip

    def resolve(self, ip: str, session: str | None) -> tuple[str, str | None]:
        with self._lock:
            if session and (ip, session) in self._sessions:
                return self._sessions[(ip, session)]
            self._refresh()
            hit = self._by_ip.get(ip)
            if hit is None:
                return f"unresolved-{ip}", None
            if session:
                self._sessions[(ip, session)] = hit
            return hit


def trial_key(project: str) -> str:
    """A safe directory name for a compose project (already [a-z0-9_-] when Harbor made it)."""
    k = KEY_RE.sub("-", project.lower()).strip(".") or "unknown"
    return k[:200]


# ---------------------------------------------------------------------- logger
class RequestLog:
    def __init__(self, root: Path):
        self.root = root
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock(self, key: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def write(self, key: str, rec: dict, raw_body: bytes) -> None:
        d = self.root / trial_key(key)
        with self._lock(key):
            d.mkdir(parents=True, exist_ok=True)
            try:
                body = json.loads(raw_body)
            except ValueError:
                body = None
            if body is None:
                rec["request_raw_b64"] = base64.b64encode(raw_body).decode("ascii")
            else:
                ext = externalize(body, d)
                rec["request"] = ext if restore(ext, d) == body else body
            with open(d / "log.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------- proxy
class ClientWatch:
    """While a request is upstream, watch the client socket; if it closes, shut the upstream socket down.

    So an aborted request (OpenCode cancelled, trial timed out) is cancelled in vLLM at once, also while
    vLLM is still queuing, prefilling or between stream events, as when socat closes its side.
    """

    def __init__(self, client: socket.socket, upstream: socket.socket | None):
        self.gone = threading.Event()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, args=(client, upstream), daemon=True)
        self._t.start()

    def _run(self, client, upstream) -> None:
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([client], [], [], 0.2)
            except (OSError, ValueError):
                return
            if not readable:
                continue
            try:
                data = client.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if data or self._stop.is_set():
                return   # the client sent its next request (or we are done): not a disconnect
            self.gone.set()
            if upstream is not None:
                try:
                    upstream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            return

    def stop(self) -> None:
        self._stop.set()   # no join: the thread ends within one select timeout


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream: tuple[str, int] = ("127.0.0.1", 8000)
    request_log: RequestLog | None = None
    resolver: TrialResolver | None = None
    policy: Policy | None = None

    def log_message(self, fmt, *args):  # quiet; errors go through log()
        pass

    def _read_body(self) -> bytes:
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            parts = []
            while True:
                size = int(self.rfile.readline().split(b";", 1)[0].strip() or b"0", 16)
                if size == 0:
                    while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    return b"".join(parts)
                parts.append(self.rfile.read(size))
                self.rfile.readline()
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _upstream(self) -> http.client.HTTPConnection:
        conn = getattr(self, "_conn", None)
        if conn is None:
            conn = http.client.HTTPConnection(*self.upstream, timeout=UPSTREAM_TIMEOUT_S)
            self._conn = conn
        return conn

    def _forward(self) -> None:
        t0 = time.time()
        body = self._read_body()
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers = {k: v for k, v in headers.items() if k.lower() != "content-length"}
            headers["Content-Length"] = str(len(body))
        logging = (self.request_log is not None and self.command == "POST"
                   and self.path.split("?", 1)[0].endswith("/chat/completions"))
        who = None
        if logging:
            # Attribute before forwarding, while the trial's container surely still holds this IP.
            ip = self.client_address[0]
            session = self.headers.get("x-opencode-session-id") or self.headers.get("x-session-id")
            try:
                who = self.resolver.resolve(ip, session) if self.resolver else (f"unresolved-{ip}", None)
            except Exception as e:   # never break the agent's run
                log(f"trial lookup failed: {type(e).__name__}: {e}")
                who = (f"unresolved-{ip}", None)
        pol, fwd = None, body
        if logging and self.policy is not None:
            try:
                pol = self.policy.for_trial(who[0])
                fwd = rewrite_request(body, pol)
            except Exception as e:   # never break the agent's run: forward unchanged
                log(f"policy rewrite failed: {type(e).__name__}: {e}")
                fwd = None
            if fwd is None:
                pol, fwd = None, body
            else:   # the reply is rewritten, so it must come back uncompressed
                headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
            headers["Content-Length"] = str(len(fwd))
        self._pol = pol
        watch = None
        conn = self._upstream()
        try:
            try:
                conn.request(self.command, self.path, body=fwd, headers=headers)
                watch = ClientWatch(self.connection, conn.sock)
                resp = conn.getresponse()
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                if watch is not None:
                    watch.stop()
                    if watch.gone.is_set():
                        raise
                conn.close()   # a stale keep-alive connection: one retry on a fresh one
                conn.request(self.command, self.path, body=fwd, headers=headers)
                watch = ClientWatch(self.connection, conn.sock)
                resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            if watch is not None:
                watch.stop()
            conn.close()
            self._conn = None
            if watch is not None and watch.gone.is_set():   # the client gave up while vLLM was busy
                self.close_connection = True
                return
            log(f"upstream error: {e}")
            msg = json.dumps({"error": {"message": f"relay: upstream unreachable: {e}"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        ctype_in = (resp.getheader("Content-Type") or "").lower()
        strip_sse = strip_json = None
        if pol is not None and (resp.getheader("Content-Encoding") or "identity").strip().lower() == "identity":
            if "text/event-stream" in ctype_in:
                strip_sse = SSEStripper(PUBLIC_MODEL)
            elif "json" in ctype_in:
                strip_json = True
        if strip_json:
            self._forward_json(t0, who, body, resp, watch, conn, logging)
            return
        self.send_response_only(resp.status, resp.reason)
        length = resp.getheader("Content-Length")
        if strip_sse is not None:
            length = None   # events are re-serialized: always chunked to the client
        chunked_out = length is None and self.command != "HEAD" and resp.status not in (204, 304)
        for k, v in resp.getheaders():
            if k.lower() not in HOP_BY_HOP and not (strip_sse is not None and k.lower() == "content-length"):
                self.send_header(k, v)
        if chunked_out:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        captured = [] if logging else None
        client_ok = True
        upstream_ok = True
        while True:
            try:
                piece = resp.read1(CHUNK)
            except (OSError, http.client.HTTPException) as e:
                if watch.gone.is_set():   # we shut it down: the client went away
                    client_ok = False
                else:                     # vLLM died mid-response
                    log(f"upstream read error: {e}")
                    upstream_ok = False
                break
            if not piece:
                if resp.length:           # EOF before Content-Length was satisfied (read1 doesn't raise)
                    log(f"upstream closed with {resp.length} bytes missing")
                    upstream_ok = False
                if strip_sse is not None and upstream_ok:
                    rest = strip_sse.flush()
                    if rest:
                        try:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(rest), rest))
                            self.wfile.flush()
                        except OSError:
                            client_ok = False
                break
            if captured is not None:
                captured.append(piece)
            if strip_sse is not None:
                piece = strip_sse.feed(piece)
                if not piece:
                    continue
            try:
                if chunked_out:
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
                else:
                    self.wfile.write(piece)
                self.wfile.flush()
            except OSError:
                # The client went away (OpenCode aborted): stop like socat would, closing the upstream
                # connection below so vLLM aborts the generation. The log keeps the partial response.
                client_ok = False
                break
        watch.stop()
        if watch.gone.is_set():
            client_ok = False
        if client_ok and chunked_out and upstream_ok:
            try:
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except OSError:
                client_ok = False
        if not (client_ok and upstream_ok) or resp.getheader("Connection", "").lower() == "close":
            self.close_connection = True
            conn.close()
            self._conn = None
        if logging:
            try:
                self._log(t0, who, body, resp, b"".join(captured), client_ok, upstream_ok)
            except Exception as e:   # logging must never break the agent's run
                log(f"logging failed: {type(e).__name__}: {e}")

    def _forward_json(self, t0, who, body, resp, watch, conn, logging) -> None:
        """A non-streamed reply in policy mode: read it whole, strip the logprobs, send it with its new length."""
        client_ok = upstream_ok = True
        try:
            raw = resp.read()
        except (OSError, http.client.HTTPException) as e:
            raw, upstream_ok = b"", False
            if watch.gone.is_set():
                client_ok = False
            else:
                log(f"upstream read error: {e}")
        watch.stop()
        if not upstream_ok or watch.gone.is_set():
            self.close_connection = True
            conn.close()
            self._conn = None
            if not client_ok or watch.gone.is_set():
                client_ok = False
            else:
                msg = json.dumps({"error": {"message": "relay: upstream failed mid-response"}}).encode()
                try:
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.end_headers()
                    self.wfile.write(msg)
                except OSError:
                    client_ok = False
        else:
            try:
                out = strip_obj(json.loads(raw), PUBLIC_MODEL)
            except ValueError:
                out = raw
            try:
                self.send_response_only(resp.status, resp.reason)
                for k, v in resp.getheaders():
                    if k.lower() not in HOP_BY_HOP and k.lower() != "content-length":
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                self.wfile.flush()
            except OSError:
                client_ok = False
                self.close_connection = True
        if resp.getheader("Connection", "").lower() == "close":
            self.close_connection = True
            conn.close()
            self._conn = None
        if logging:
            try:
                self._log(t0, who, body, resp, raw, client_ok, upstream_ok)
            except Exception as e:
                log(f"logging failed: {type(e).__name__}: {e}")

    def _log(self, t0: float, who: tuple[str, str | None], body: bytes, resp, raw: bytes, client_ok: bool,
             upstream_ok: bool) -> None:
        ip = self.client_address[0]
        session = self.headers.get("x-opencode-session-id") or self.headers.get("x-session-id")
        key, container = who
        decoded = decode_body(raw, resp.getheader("Content-Encoding"))
        ctype = (resp.getheader("Content-Type") or "").lower()
        if decoded is None:
            parsed = {"error": f"undecodable Content-Encoding {resp.getheader('Content-Encoding')!r}"}
        elif "text/event-stream" in ctype:
            parsed = assemble_sse(decoded)
        else:
            parsed = assemble_json(decoded)
        rec = {"ts": t0, "t_end": time.time(), "trial": key, "client_ip": ip, "container": container,
               "session": session, "method": self.command, "path": self.path, "status": resp.status,
               "client_ok": client_ok, "upstream_ok": upstream_ok,
               "headers": {h: self.headers.get(h) for h in LOGGED_HEADERS if self.headers.get(h) is not None},
               "body_len": len(body), "body_sha256": hashlib.sha256(body).hexdigest(),
               "resp_len": len(raw), "resp_sha256": hashlib.sha256(raw).hexdigest(), "response": parsed}
        pol = getattr(self, "_pol", None)
        if pol is not None:
            rec["policy_version"] = pol["version"]
            rec["policy_name"] = pol["name"]
            # completion_tokens from usage lets the converter check that every generated token has a log-prob
            rec["response"]["completion_tokens"] = ((parsed.get("usage") or {}).get("completion_tokens")
                                                    if isinstance(parsed, dict) else None)
        if isinstance(parsed, dict) and parsed.get("stream") is False and pol is not None:
            parsed["logprobs"] = logprobs_json(raw)
        self.request_log.write(key, rec, body)

    def finish(self):
        try:
            super().finish()
        finally:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()

    def do_GET(self):
        self._forward()

    do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_GET


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128


def make_server(listen: tuple[str, int], upstream: tuple[str, int], log_dir: Path | None,
                resolver: TrialResolver | None = None, policy_file: Path | None = None) -> Server:
    handler = type("BoundHandler", (Handler,), {
        "upstream": upstream,
        "policy": Policy(policy_file) if policy_file else None,
        "request_log": RequestLog(log_dir) if log_dir else None,
        "resolver": resolver if resolver is not None else (TrialResolver() if log_dir else None)})
    return Server(listen, handler)


def hostport(s: str) -> tuple[str, int]:
    host, _, port = s.rpartition(":")
    return host, int(port)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", type=hostport, required=True, help="host:port, e.g. 172.17.0.1:18000")
    ap.add_argument("--upstream", type=hostport, required=True, help="host:port of vLLM, e.g. 127.0.0.1:8000")
    ap.add_argument("--log-dir", type=Path, required=True)
    ap.add_argument("--policy-file", type=Path, help="async GRPO: pin trials to policy versions, log logprobs")
    a = ap.parse_args(argv)
    os.umask(0o077)
    a.log_dir.mkdir(parents=True, exist_ok=True)
    srv = make_server(a.listen, a.upstream, a.log_dir, policy_file=a.policy_file)
    log(f"listening on {a.listen[0]}:{a.listen[1]} -> {a.upstream[0]}:{a.upstream[1]}, logs in {a.log_dir}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
