"""Helpers for parsing and repairing JSON from LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from .ollama_client import ollama_chat
from .prompts import JSON_REPAIR_SYSTEM

CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
LINE_COMMENT_RE = re.compile(r"(?m)^\s*//.*?$")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def strip_code_fences(text: str) -> str:
    """Remove Markdown code fences from text."""
    return CODE_FENCE_RE.sub("", text).strip()


def strip_json_comments(text: str) -> str:
    """Remove line and block comments from JSON-like text."""
    text = BLOCK_COMMENT_RE.sub("", text)
    text = LINE_COMMENT_RE.sub("", text)
    return text


def remove_trailing_commas(text: str) -> str:
    """Remove trailing commas from JSON arrays/objects."""
    prev = None
    while prev != text:
        prev = text
        text = TRAILING_COMMA_RE.sub(r"\1", text)
    return text


def extract_first_json_value(text: str) -> Optional[str]:
    """Extract the first JSON object/array from raw text."""
    # Scan for the first JSON object/array, respecting quoted strings.
    text = text.strip()
    start_obj = text.find("{")
    start_arr = text.find("[")
    if start_obj == -1 and start_arr == -1:
        return None

    if start_obj == -1 or (start_arr != -1 and start_arr < start_obj):
        start = start_arr
        open_c, close_c = "[", "]"
    else:
        start = start_obj
        open_c, close_c = "{", "}"

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None


def best_effort_json_text(raw: str) -> Optional[str]:
    """Try to extract a valid JSON snippet from raw text."""
    s = strip_code_fences(raw)
    s = strip_json_comments(s).strip()
    extracted = extract_first_json_value(s)
    if extracted:
        s = extracted
    s = remove_trailing_commas(s)
    return s if s else None


def parse_or_repair_json(
    raw: str,
    *,
    ollama_base: str,
    model: str,
    timeout_s: int,
    num_predict: int,
    num_ctx: int,
    log: Optional[Callable[[str], None]] = None,
    label: str = "json_repair",
) -> Optional[dict]:
    """Parse JSON directly or fall back to LLM-based repair."""
    # Try direct parse, then light cleanup, then LLM repair as a last resort.
    try:
        return json.loads(raw)
    except Exception:
        pass

    cleaned = best_effort_json_text(raw)
    if cleaned:
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    repaired = ollama_chat(
        ollama_base, model,
        JSON_REPAIR_SYSTEM,
        "INPUT_TEXT:\n" + raw,
        temperature=0.0,
        timeout_s=timeout_s,
        num_predict=num_predict,
        num_ctx=num_ctx,
        log=log,
        label=label,
    )
    repaired_clean = best_effort_json_text(repaired) or repaired
    try:
        return json.loads(repaired_clean)
    except Exception:
        return None
