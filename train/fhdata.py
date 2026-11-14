"""FrameHunt SFT data format: OpenCode's wire format, and how it becomes Qwen tokens.

Stored format (one JSON object per line in sft.jsonl), captured from a real Harbor +
OpenCode 1.19.5 run against a logging mock LLM (train/capture/):

    {"id": ..., "item_id": ..., "kind": ..., "split": "train", "answer": ..., "correct": true,
     "pixels_used": ..., "n_calls": ..., "date": "Sat Nov 14 2026",
     "messages": [
       {"role": "system", "content": <OpenCode system prompt, with this row's date>},
       {"role": "user", "content": "\"" + instruction.md + "\""},       # OpenCode quotes it
       {"role": "assistant", "content": "", "reasoning": "...",
        "tool_calls": [{"id": "call_..", "type": "function",
                        "function": {"name": "framehunt_view", "arguments": "{\"t\":12.3,...}"}}]},
       {"role": "tool", "tool_call_id": "call_..", "content": "<the tool's text>"},
       {"role": "user", "content": [{"type": "text", "text": "Attached media from tool result:"},
                                    {"type": "image_url", "image_url": {"url": "images/<id>/03.png"}}]},
       ...,
       {"role": "assistant", "content": "Submitted ...", "reasoning": "..."}]}

That is exactly the request body OpenCode sends (tool calls with compact JSON-string
arguments, vLLM's `reasoning` field, MCP images as an extra user message), except
that image URLs are relative PNG paths instead of data: URLs. The tools list
(17 tools, OpenCode's own plus `framehunt_*`) is the same for every row and is stored
once in <data>/tools.json (a copy of train/capture/opencode_tools.json).

`to_wire()` turns a row back into the literal request (data: URLs) for replaying it
against a vLLM endpoint. `to_template()` turns it into what vLLM 0.33 hands to the
chat template: `reasoning` mirrored to `reasoning_content`, arguments parsed to dicts,
images as PIL images. `encode()` renders that with the model's own chat template
(preserve_thinking defaults to true, so every assistant turn keeps its <think> block,
as at inference) and builds labels: only assistant tokens the model generates are
trained, i.e. from after "<|im_start|>assistant\\n<think>\\n" (the generation prompt)
through "<|im_end|>" inclusive.
"""

from __future__ import annotations

import base64
import copy
import io
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "capture"
SYSTEM_PROMPT_FILE = CAPTURE / "opencode_system_prompt.txt"
TOOLS_FILE = CAPTURE / "opencode_tools.json"
CAPTURED_DATE = "Sat Nov 14 2026"
ATTACHED_MEDIA_TEXT = "Attached media from tool result:"
TOOL_PREFIX = "framehunt_"   # OpenCode names MCP tools <server>_<tool>
IGNORE = -100


def system_prompt(date: str | None = None) -> str:
    s = SYSTEM_PROMPT_FILE.read_text()
    if date is not None:
        assert f"Today's date: {CAPTURED_DATE}" in s
        s = s.replace(f"Today's date: {CAPTURED_DATE}", f"Today's date: {date}")
    return s


def tools() -> list[dict]:
    return json.loads(TOOLS_FILE.read_text())


def user_turn(instruction_md: str) -> dict:
    # OpenCode (via Harbor) sends the instruction wrapped in double quotes.
    return {"role": "user", "content": f'"{instruction_md}"'}


def args_json(args: dict) -> str:
    # OpenCode re-serializes tool arguments compactly (JSON.stringify).
    return json.dumps(args, separators=(",", ":"), ensure_ascii=False)


# ---------------------------------------------------------------------- conversions
def _image_parts(content):
    return [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"] \
        if isinstance(content, list) else []


def load_png(url: str, root: Path) -> bytes:
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    return (root / url).read_bytes()


def to_wire(row: dict, root: Path) -> list[dict]:
    """The row's messages exactly as OpenCode would send them (images as data: URLs)."""
    msgs = copy.deepcopy(row["messages"])
    for m in msgs:
        for p in _image_parts(m.get("content")):
            u = p["image_url"]["url"]
            if not u.startswith("data:"):
                p["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(load_png(u, root)).decode()
    return msgs


def to_template(row: dict, root: Path) -> tuple[list[dict], list]:
    """Messages as vLLM 0.33 passes them to the chat template, plus the images in order."""
    from PIL import Image

    out, images = [], []
    for m in row["messages"]:
        m = copy.deepcopy(m)
        if m["role"] == "assistant":
            if m.get("reasoning") is not None:
                m["reasoning_content"] = m.pop("reasoning")   # vLLM mirrors reasoning -> reasoning_content
            for tc in m.get("tool_calls") or []:
                a = tc["function"].get("arguments")
                tc["function"]["arguments"] = json.loads(a) if isinstance(a, str) and a else (a or {})
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "image_url":
                    img = Image.open(io.BytesIO(load_png(p["image_url"]["url"], root)))
                    img.load()
                    images.append(img.convert("RGB"))
                    parts.append({"type": "image"})
                else:
                    parts.append(p)
            m["content"] = parts
        out.append(m)
    return out, images


# ---------------------------------------------------------------------- tokenization
def assistant_spans(ids: list[int], tok) -> list[tuple[int, int]]:
    """[start, end) token spans to train on: each assistant turn's generated part.

    Starts after "<|im_start|>assistant\\n<think>\\n" (what the server puts in the
    generation prompt) and ends after the turn's "<|im_end|>".
    """
    im_start = tok.convert_tokens_to_ids("<|im_start|>")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    head = tok.encode("assistant\n<think>\n", add_special_tokens=False)
    spans = []
    i, n = 0, len(ids)
    while i < n:
        if ids[i] == im_start and ids[i + 1:i + 1 + len(head)] == head:
            s = i + 1 + len(head)
            e = s
            while e < n and ids[e] != im_end:
                e += 1
            if e == n:
                raise ValueError("assistant turn without <|im_end|>")
            spans.append((s, e + 1))
            i = e + 1
        else:
            i += 1
    return spans


def encode(row: dict, root: Path, processor, tools_list: list[dict] | None = None,
           max_len: int | None = None) -> dict:
    """Render one row with the model's chat template and build masked labels.

    Returns a dict of tensors for the model (input_ids, attention_mask, labels,
    pixel_values, image_grid_thw, mm_token_type_ids if the processor makes them) plus
    'n_tokens', 'n_train_tokens', 'n_images', 'n_image_tokens'.
    """
    import torch

    msgs, images = to_template(row, root)
    text = processor.apply_chat_template(msgs, tools=tools_list if tools_list is not None else tools(),
                                         tokenize=False, add_generation_prompt=False)
    enc = processor(text=[text], images=images or None, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    if max_len is not None and len(ids) > max_len:
        raise ValueError(f"{row.get('id')}: {len(ids)} tokens > max_len {max_len}")
    tok = processor.tokenizer
    labels = [IGNORE] * len(ids)
    spans = assistant_spans(ids, tok)
    for s, e in spans:
        labels[s:e] = ids[s:e]
    out = dict(enc)
    out["labels"] = torch.tensor([labels])
    image_pad = tok.convert_tokens_to_ids("<|image_pad|>")
    out["n_tokens"] = len(ids)
    out["n_train_tokens"] = sum(e - s for s, e in spans)
    out["n_images"] = len(images)
    out["n_image_tokens"] = sum(1 for t in ids if t == image_pad)
    out["n_assistant_turns"] = len(spans)
    return out


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
