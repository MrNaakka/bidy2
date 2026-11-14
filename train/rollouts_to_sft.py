"""Expert iteration: rewarded on-policy rollouts -> SFT rows in make_sft.py's exact format.

    uv run python train/rollouts_to_sft.py --out train/data/ei-1 runs/fh-train-ei-qwen-fh-run1 [runs/... ...] \\
        [--tasks tasks/train-ei] [--max-per-item 1] [--processor train/out/tiny-qwen3.8]

Inputs are Harbor run dirs fetched by `fh eval-fetch` from a fleet launched with `--log-requests`: one
dir per trial (config.json, result.json, verifier/reward.txt, verifier/result.json, verifier/calls.jsonl)
plus requests/<trial key>/ written by infra/verda/log_relay.py (log.jsonl, blobs/, tools/).

For every trial with reward 1:
  * item guard: the item id must be a TRAIN item (framehunt.splits.train_items' id scheme) and not an
    EVAL/DEV id; any other item in the inputs is an error (nothing is written);
  * the trial's request log is found by Harbor's compose project name (<trial_name>__env, lowercased);
    requests that carry tools must all come from one OpenCode session;
  * the conversation is the longest such request (OpenCode resends the whole history every turn) plus
    its logged response as the final assistant turn (if the response completed); every earlier request
    must be a prefix of it, and every logged response must equal the assistant turn OpenCode sent back
    next (content, reasoning, tool calls), which checks the relay's stream reassembly;
  * the trajectory must match the MCP server's own call log (verifier/calls.jsonl): the server's calls
    are an in-order subsequence of the conversation's framehunt_* calls with the same arguments, so a
    log can't be attributed to the wrong trial;
  * format: OpenCode's captured system prompt (with the row's date) and the captured tool list exactly,
    the quoted instruction (identical to <tasks>/<item>/instruction.md with --tasks), exactly one tool
    call in every assistant turn but the last (zero or one), each followed by its tool result, images
    only as PNG attachments, and a framehunt_submit call.

Rows are written like make_sft.py's: images saved as the exact PNG bytes the server returned
(images/<id>/NN.png, decoded from the request's base64), messages exactly as OpenCode sent them. Each
row is checked to render identically to the logged request: fhdata.to_wire(row) == the request's messages
(+ final turn), fhdata.to_template gives the same messages and pixel-identical images, and with
--processor the chat template text is identical too. Rows also carry provenance (run, trial, model,
reward, session). Identical trajectories are dropped; per item the cheapest --max-per-item are kept
(fewest pixels, then fewest calls). Output: <out>/sft.jsonl, images/, tools.json (the capture's),
stats.json, rejected.jsonl.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "infra" / "verda"))

import fhdata  # noqa: E402
from log_relay import restore, trial_key  # noqa: E402

from framehunt import splits  # noqa: E402

PNG_SIG = b"\x89PNG\r\n\x1a\n"
DATA_PNG = "data:image/png;base64,"
DATE_RE = re.compile(r"Today's date: (.+)")
BUDGET_RE = re.compile(r"You have (\d+) pixels in total")


class Reject(Exception):
    pass


# ---------------------------------------------------------------------- items
def train_kind(item_id: str) -> str | None:
    """The kind of a TRAIN item id, or None if the id is not a TRAIN item (or is an EVAL/DEV id)."""
    eval_ids = {i for items in splits.SPLITS.values() for i, _, _ in items}
    eval_ids |= {i for i, _, _ in getattr(splits, "EVAL_V1", ())}
    m = re.fullmatch(r"([a-z]+)-(\d+)", item_id)
    if not m or item_id in eval_ids or m.group(1) not in splits.TRAIN_KINDS:
        return None
    kind, seed = m.group(1), int(m.group(2))
    j = seed - splits.TRAIN_SEED_BASE - splits.TRAIN_KINDS.index(kind) * 10_000
    return kind if 1 <= j <= splits.TRAIN_MAX_PER_KIND else None


def harbor_project(trial_name: str) -> str:
    """Harbor's docker compose project name for a trial (session id <trial>__env, sanitized)."""
    name = f"{trial_name}__env".lower()
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    return re.sub(r"[^a-z0-9_-]", "-", name)


# ---------------------------------------------------------------------- trials
def load_trial(tdir: Path) -> dict:
    cfg = json.loads((tdir / "config.json").read_text())
    trial = cfg["trial_name"]
    item_id = Path(cfg["task"]["path"]).name
    res = {}
    if (tdir / "result.json").is_file():
        res = json.loads((tdir / "result.json").read_text())
    if res.get("task_name") and res["task_name"].split("/")[-1] != item_id:
        raise SystemExit(f"{tdir}: task_name {res['task_name']} != task path {cfg['task']['path']}")
    reward = 0.0
    rf = tdir / "verifier" / "reward.txt"
    if rf.is_file():
        try:
            reward = float(rf.read_text().strip())
        except ValueError:
            reward = 0.0
    vres = {}
    if (tdir / "verifier" / "result.json").is_file():
        vres = json.loads((tdir / "verifier" / "result.json").read_text())
    calls = []
    if (tdir / "verifier" / "calls.jsonl").is_file():
        calls = [json.loads(l) for l in (tdir / "verifier" / "calls.jsonl").read_text().splitlines() if l.strip()]
    return {"dir": tdir, "trial": trial, "item_id": item_id, "task_path": cfg["task"]["path"], "reward": reward,
            "verifier": vres, "calls": calls}


def find_trials(run: Path) -> list[Path]:
    return sorted(p for p in run.iterdir() if p.is_dir() and (p / "config.json").is_file()
                  and "trial_name" in json.loads((p / "config.json").read_text()))


# ---------------------------------------------------------------------- conversation
def usable(rec: dict) -> bool:
    """A logged response that completed: delivered to OpenCode in full, a clean stream that ended."""
    resp = rec.get("response") or {}
    return (rec.get("status") == 200 and rec.get("client_ok") is True and rec.get("upstream_ok") is True
            and not resp.get("error") and resp.get("finish_reason") in ("stop", "tool_calls")
            and (not resp.get("stream") or (resp.get("done") is True and resp.get("bad_lines") == 0)))


def final_turn(resp: dict) -> dict:
    """The assistant message OpenCode would send back for this response (compact JSON arguments)."""
    turn = {"role": "assistant", "content": resp.get("content") or "", "reasoning": resp.get("reasoning")}
    if resp.get("tool_calls"):
        calls = []
        for tc in resp["tool_calls"]:
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except ValueError:
                raise Reject("final tool call arguments are not JSON")
            calls.append({"id": tc["id"], "type": "function",
                          "function": {"name": tc["function"]["name"], "arguments": fhdata.args_json(args)}})
        turn["tool_calls"] = calls
    return turn


def _args(tc: dict):
    a = tc["function"].get("arguments")
    try:
        return json.loads(a) if isinstance(a, str) and a else (a or {})
    except ValueError:
        return None


def same_turn(resp: dict, msg: dict) -> bool:
    """A logged response equals the assistant message OpenCode sent back (texts up to surrounding whitespace)."""
    if msg.get("role") != "assistant":
        return False
    if (resp.get("content") or "").strip() != (msg.get("content") or "").strip():
        return False
    if (resp.get("reasoning") or "").strip() != (msg.get("reasoning") or "").strip():
        return False
    a, b = resp.get("tool_calls") or [], msg.get("tool_calls") or []
    return len(a) == len(b) and all(
        x.get("id") == y.get("id") and x["function"]["name"] == y["function"]["name"] and _args(x) == _args(y)
        for x, y in zip(a, b))


def conversation(records: list[dict], rdir: Path) -> tuple[list[dict], list[dict], dict, dict]:
    """(messages incl. final turn, tools, final record, info) from one trial's relay log."""
    recs = [r for r in records if r.get("method") == "POST" and isinstance(r.get("request"), dict)
            and r["request"].get("tools")]
    if not recs:
        raise Reject("no logged request with tools")
    sessions = {r.get("session") for r in recs}
    if len(sessions) != 1:
        raise Reject(f"{len(sessions)} OpenCode sessions with tools (subagent or mixed log)")
    ok = [r for r in recs if r.get("status") == 200]
    if not ok:
        raise Reject("no successful request")
    for r in ok:
        r["_req"] = restore(r["request"], rdir)
    n_max = max(len(r["_req"]["messages"]) for r in ok)
    longest = [r for r in ok if len(r["_req"]["messages"]) == n_max]
    final = max(longest, key=lambda r: (usable(r), r.get("ts", 0)))
    msgs = final["_req"]["messages"]
    tools = final["_req"]["tools"]
    for r in ok:
        m = r["_req"]["messages"]
        if m != msgs[:len(m)]:
            raise Reject("an earlier request is not a prefix of the last one (history rewritten)")
        if r["_req"]["tools"] != tools:
            raise Reject("tool list changed during the run")
        if len(m) < len(msgs) and usable(r) and not same_turn(r["response"], msgs[len(m)]):
            raise Reject("a logged response differs from the assistant turn OpenCode sent back")
    out = list(msgs)
    has_final = usable(final)
    if has_final:
        out.append(final_turn(final["response"]))
    return out, tools, final, {"session": final.get("session"), "final_turn": has_final,
                               "model": final["_req"].get("model"), "n_requests": len(recs)}


CAPTURED_MODEL_LINE = "You are powered by the model named qwen3.8-27b. The exact model ID is verda-qwen/qwen3.8-27b\n"


def expected_system_prompt(date: str, model: str | None) -> str:
    """The captured system prompt with this row's date; OpenCode names the served model in it, so a run
    against another served name (the LoRA's qwen3.8-27b-fh) differs from the capture in that line only."""
    s = fhdata.system_prompt(date)
    if model and model != "qwen3.8-27b":
        assert s.count(CAPTURED_MODEL_LINE) == 1
        s = s.replace(CAPTURED_MODEL_LINE, CAPTURED_MODEL_LINE.replace("qwen3.8-27b", model))
    return s


def check_format(msgs: list[dict], tools: list[dict], require_submit: bool = True, model: str | None = None) -> str:
    """Raise Reject unless msgs/tools have make_sft's exact shape; return the row's date."""
    if tools != fhdata.tools():
        raise Reject("tool list differs from the capture (OpenCode version or MCP server changed?)")
    if len(msgs) < 3 or msgs[0].get("role") != "system" or not isinstance(msgs[0].get("content"), str):
        raise Reject("no system prompt")
    m = DATE_RE.search(msgs[0]["content"])
    if not m or msgs[0]["content"] not in (fhdata.system_prompt(m.group(1)), expected_system_prompt(m.group(1), model)):
        raise Reject("system prompt differs from the capture")
    u = msgs[1]
    if u.get("role") != "user" or not isinstance(u.get("content"), str) or len(u["content"]) < 2 \
            or not (u["content"].startswith('"') and u["content"].endswith('"')):
        raise Reject("second message is not the quoted instruction")
    pending: list[str] = []
    last_asst = max(i for i, x in enumerate(msgs) if x.get("role") == "assistant") if any(
        x.get("role") == "assistant" for x in msgs) else -1
    if last_asst < 0:
        raise Reject("no assistant turn")
    submitted = False
    for i, x in enumerate(msgs[2:], start=2):
        role = x.get("role")
        if role == "assistant":
            if pending:
                raise Reject(f"message {i}: assistant turn before the previous tool result")
            calls = x.get("tool_calls") or []
            if i != last_asst and len(calls) != 1:
                raise Reject(f"message {i}: {len(calls)} tool calls (need exactly one)")
            if i == last_asst and len(calls) > 1:
                raise Reject(f"message {i}: {len(calls)} tool calls in the last turn")
            for tc in calls:
                if _args(tc) is None:
                    raise Reject(f"message {i}: tool arguments are not JSON")
                submitted |= tc["function"]["name"] == fhdata.TOOL_PREFIX + "submit"
            pending = [tc["id"] for tc in calls]
        elif role == "tool":
            if not pending or x.get("tool_call_id") != pending[0]:
                raise Reject(f"message {i}: tool result without its call")
            pending.pop(0)
        elif role == "user":
            c = x.get("content")
            if pending or not isinstance(c, list) or not c or any(
                    p.get("type") not in ("text", "image_url") for p in c) or not any(
                    p.get("type") == "image_url" for p in c):
                raise Reject(f"message {i}: unexpected user message")
            for p in c:
                if p.get("type") == "image_url" and not p["image_url"]["url"].startswith(DATA_PNG):
                    raise Reject(f"message {i}: image is not a PNG data URL")
        else:
            raise Reject(f"message {i}: unexpected role {role!r}")
    if pending and msgs[-1] is not msgs[last_asst]:   # only the last turn's call may lack a result
        raise Reject("a tool call has no result")
    if require_submit and not submitted:
        raise Reject("no framehunt_submit call")
    return m.group(1)


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def args_match(model: dict, server: dict) -> bool:
    for k, v in (model or {}).items():
        if v is None:
            continue
        if k not in server:
            return False
        s = server[k]
        if _num(v) or _num(s):
            try:
                if abs(float(v) - float(s)) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif v != s:
            return False
    return True


def check_server_log(msgs: list[dict], calls: list[dict]) -> int:
    """The server's calls must be an in-order subsequence of the conversation's framehunt calls."""
    conv = [(tc["function"]["name"][len(fhdata.TOOL_PREFIX):], _args(tc))
            for x in msgs if x.get("role") == "assistant" for tc in x.get("tool_calls") or []
            if tc["function"]["name"].startswith(fhdata.TOOL_PREFIX)]
    if not calls:
        raise Reject("the server logged no calls")
    j = 0
    for name, args in conv:
        if j < len(calls) and calls[j]["tool"] == name and args_match(args, calls[j].get("args") or {}):
            j += 1
    if j != len(calls):
        raise Reject(f"conversation does not match the server's call log (matched {j}/{len(calls)} calls)")
    return len(conv) - j


# ---------------------------------------------------------------------- rows
def wire_key(msgs: list[dict]) -> str:
    """Content hash of a conversation for dedupe: images as PNG hashes, tool-call ids (random per run)
    renumbered in order of appearance."""
    ids: dict[str, str] = {}

    def cid(i):
        return ids.setdefault(i, f"c{len(ids)}")

    def strip(m):
        if m.get("tool_calls"):
            m = {**m, "tool_calls": [{**tc, "id": cid(tc.get("id"))} for tc in m["tool_calls"]]}
        if m.get("role") == "tool":
            m = {**m, "tool_call_id": cid(m.get("tool_call_id"))}
        if isinstance(m.get("content"), list):
            m = {**m, "content": [{**p, "image_url": {"url": hashlib.sha256(
                base64.b64decode(p["image_url"]["url"][len(DATA_PNG):])).hexdigest()}}
                if p.get("type") == "image_url" else p for p in m["content"]]}
        return m
    return hashlib.sha256(json.dumps([strip(m) for m in msgs], sort_keys=True).encode()).hexdigest()


def make_row(row_id: str, msgs: list[dict], out: Path) -> tuple[dict, list[bytes]]:
    rmsgs, pngs = [], []
    for m in msgs:
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "image_url":
                    png = base64.b64decode(p["image_url"]["url"][len(DATA_PNG):], validate=True)
                    if not png.startswith(PNG_SIG):
                        raise Reject("an image is not a PNG")
                    p = {**p, "image_url": {**p["image_url"], "url": f"images/{row_id}/{len(pngs):02d}.png"}}
                    pngs.append(png)
                parts.append(p)
            m = {**m, "content": parts}
        rmsgs.append(m)
    return {"messages": rmsgs}, pngs


def assert_renders(row: dict, wire: list[dict], tools: list[dict], out: Path, processor=None) -> None:
    """The stored row renders exactly like the logged request (and like any make_sft row)."""
    if fhdata.to_wire(row, out) != wire:
        raise AssertionError(f"{row['id']}: to_wire(row) differs from the logged request")
    tm_row, im_row = fhdata.to_template(row, out)
    tm_wire, im_wire = fhdata.to_template({"messages": wire}, out)
    if tm_row != tm_wire or [(i.size, i.tobytes()) for i in im_row] != [(i.size, i.tobytes()) for i in im_wire]:
        raise AssertionError(f"{row['id']}: to_template differs from the logged request")
    if processor is not None:
        a = processor.apply_chat_template(tm_row, tools=tools, tokenize=False, add_generation_prompt=False)
        b = processor.apply_chat_template(tm_wire, tools=tools, tokenize=False, add_generation_prompt=False)
        if a != b:
            raise AssertionError(f"{row['id']}: chat template text differs")
        enc = fhdata.encode(row, out, processor, tools)
        n_asst = sum(m["role"] == "assistant" for m in row["messages"])
        if enc["n_assistant_turns"] != n_asst or enc["n_images"] != len(im_row):
            raise AssertionError(f"{row['id']}: {enc['n_assistant_turns']} trained turns for {n_asst} assistant "
                                 f"messages, {enc['n_images']} images for {len(im_row)}")


def convert_trial(t: dict, run: Path, meta: dict, tasks: Path | None, require_submit: bool = True) -> dict:
    rdir = run / "requests" / trial_key(harbor_project(t["trial"]))
    if not (rdir / "log.jsonl").is_file():
        raise Reject(f"no request log ({rdir.relative_to(run)})")
    records = [json.loads(l) for l in (rdir / "log.jsonl").read_text().splitlines() if l.strip()]
    msgs, tools, final, info = conversation(records, rdir)
    date = check_format(msgs, tools, require_submit, info["model"])
    if tasks is not None:
        inst = tasks / t["item_id"] / "instruction.md"
        if not inst.is_file():
            raise Reject(f"{inst} not found")
        if msgs[1] != fhdata.user_turn(inst.read_text()):
            raise Reject("instruction differs from the task's instruction.md")
        if 'split = "train"' not in (tasks / t["item_id"] / "task.toml").read_text():
            raise Reject("task.toml is not a train-split task")
    unmatched = check_server_log(msgs, t["calls"])
    tools_used = [tc["function"]["name"] for m in msgs if m.get("role") == "assistant" for tc in m.get("tool_calls") or []]
    bm = BUDGET_RE.search(msgs[1]["content"])
    v = t["verifier"]
    return {"wire": msgs, "date": date, "tools": tools, "tools_used": tools_used, "info": info,
            "unmatched_calls": unmatched, "budget": int(bm.group(1)) if bm else None,
            "answer": v.get("answer"), "pixels_used": v.get("pixels_used"),
            "provenance": {"source": "rollout", "run": run.name, "trial": t["trial"], "item_id": t["item_id"],
                           "reward": t["reward"], "model": info["model"], "session": info["session"],
                           "final_turn": info["final_turn"], "n_requests": info["n_requests"],
                           "server_calls": len(t["calls"]), "unmatched_calls": unmatched,
                           "served_name": meta.get("served_name"), "lora_sha": meta.get("lora_sha"),
                           "fleet": meta.get("fleet"), "git_commit": meta.get("git_commit")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="fetched Harbor run dirs (runs/<job>) with requests/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", type=Path, help="the TRAIN tasks dir the runs used: also check each instruction")
    ap.add_argument("--max-per-item", type=int, default=1, help="rows kept per item (0: every distinct one)")
    ap.add_argument("--processor", help="HF processor dir (train venv): also compare chat-template renderings")
    ap.add_argument("--force", action="store_true", help="replace an existing --out")
    a = ap.parse_args(argv)
    out_final = a.out.resolve()
    for p in [*a.runs, *([a.tasks] if a.tasks else [])]:
        q = p.resolve()
        if q == out_final or out_final in q.parents or q in out_final.parents:
            ap.error(f"--out {a.out} overlaps the input {p}")
    if a.out.exists() and not a.force:
        ap.error(f"{a.out} exists (--force replaces it)")
    processor = None
    if a.processor:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(a.processor)

    trials = []
    for run in a.runs:
        if not run.is_dir():
            ap.error(f"{run}: not a directory")
        meta = {}
        if (run / "fh-eval-meta.json").is_file():
            meta = json.loads((run / "fh-eval-meta.json").read_text())
        trials += [(run, meta, load_trial(td)) for td in find_trials(run)]
    bad = sorted({f"{run.name}/{t['trial']} ({t['item_id']})" for run, _, t in trials if train_kind(t["item_id"]) is None})
    if bad:
        raise SystemExit(f"refusing: {len(bad)} trial(s) are not on TRAIN items (EVAL/DEV/unknown): {bad[:10]}")
    keys = collections.Counter((run, trial_key(harbor_project(t["trial"]))) for run, _, t in trials)

    stats = {"runs": [str(r) for r in a.runs], "trials": len(trials), "reward_1": 0, "rejected": collections.Counter(),
             "duplicates": 0, "over_max_per_item": 0, "by_kind": {}, "models": collections.Counter()}
    rejected, cands, seen = [], collections.defaultdict(list), set()
    for run, meta, t in trials:
        if t["reward"] != 1.0:
            continue
        stats["reward_1"] += 1
        try:
            if keys[(run, trial_key(harbor_project(t["trial"])))] > 1:
                raise Reject("two trials map to the same request log (trial names differ only in case)")
            c = convert_trial(t, run, meta, a.tasks)
        except Reject as e:
            reason = str(e)
            stats["rejected"][re.sub(r"\d+", "N", reason)] += 1
            rejected.append({"run": run.name, "trial": t["trial"], "item_id": t["item_id"], "reason": reason})
            continue
        k = wire_key(c["wire"])
        if k in seen:
            stats["duplicates"] += 1
            continue
        seen.add(k)
        c["trial"], c["item_id"], c["run"] = t["trial"], t["item_id"], run.name
        cands[t["item_id"]].append(c)

    # build next to --out and swap it in at the end: a failure never leaves a half dataset or loses the old one
    final_out = a.out
    final_out.parent.mkdir(parents=True, exist_ok=True)
    a.out = Path(tempfile.mkdtemp(prefix=f".{final_out.name}.", suffix=".tmp", dir=final_out.parent))
    (a.out / "images").mkdir()
    shutil.copy(fhdata.TOOLS_FILE, a.out / "tools.json")
    rows = []
    for item_id in sorted(cands):
        cs = sorted(cands[item_id], key=lambda c: (c["pixels_used"] or 0, len(c["tools_used"]), c["run"], c["trial"]))
        if a.max_per_item > 0:
            stats["over_max_per_item"] += max(0, len(cs) - a.max_per_item)
            cs = cs[:a.max_per_item]
        for c in cs:
            row_id = f"{c['run']}__{c['trial']}"
            body, pngs = make_row(row_id, c["wire"], a.out)
            kind = train_kind(item_id)
            row = {"id": row_id, "item_id": item_id, "kind": kind, "split": "train", "answer": c["answer"],
                   "correct": True, "pixels_used": c["pixels_used"], "budget": c["budget"],
                   "n_calls": len(c["tools_used"]), "n_images": len(pngs), "tools_used": c["tools_used"],
                   "date": c["date"], "provenance": c["provenance"], "messages": body["messages"]}
            img_dir = a.out / "images" / row_id
            img_dir.mkdir(parents=True)
            for i, png in enumerate(pngs):
                (img_dir / f"{i:02d}.png").write_bytes(png)
            assert_renders(row, c["wire"], c["tools"], a.out, processor)
            rows.append(row)
            stats["models"][c["provenance"]["model"]] += 1
            ks = stats["by_kind"].setdefault(kind, {"rows": 0, "n_calls": 0, "pixels_used": 0})
            ks["rows"] += 1
            ks["n_calls"] += row["n_calls"]
            ks["pixels_used"] += row["pixels_used"] or 0
    with open(a.out / "sft.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(a.out / "rejected.jsonl", "w") as f:
        for r in rejected:
            f.write(json.dumps(r) + "\n")
    for ks in stats["by_kind"].values():
        ks["mean_calls"] = round(ks.pop("n_calls") / ks["rows"], 2)
        ks["mean_pixels"] = round(ks.pop("pixels_used") / ks["rows"])
    stats.update(rows=len(rows), items=len(cands), rendered_with_processor=bool(processor),
                 rejected=dict(stats["rejected"]), models=dict(stats["models"]))
    (a.out / "stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    if final_out.exists():
        shutil.rmtree(final_out)
    a.out.rename(final_out)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
