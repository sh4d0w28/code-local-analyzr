"""Load step definitions used to select and scan files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .config import get_steps_config


@dataclass(frozen=True)
class Step:
    """Represents a scanning step and its matching rules."""
    key: str
    title: str
    globs: List[str]
    path_keywords: List[str]
    snippet_regexes: List[re.Pattern]
    max_files: Optional[int] = None


def compile_rx(patterns: List[str]) -> List[re.Pattern]:
    """Compile regex patterns for snippet matching."""
    return [re.compile(p) for p in patterns]

def _require_str(value: object, label: str, idx: int) -> str:
    """Ensure required step fields are present and non-empty."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f'steps[{idx}].{label} is missing or not a string')
    return value


def _normalize_list(value: object) -> List[str]:
    """Normalize a list of strings from config values."""
    if value is None:
        return []
    if not isinstance(value, list):
        raise RuntimeError("Expected a list in steps config")
    out: List[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        if item.strip():
            out.append(item)
    return out


def load_steps() -> List[Step]:
    """Parse step definitions from configuration and return Step objects."""
    steps_cfg = get_steps_config()
    steps: List[Step] = []
    for idx, raw in enumerate(steps_cfg):
        if not isinstance(raw, dict):
            raise RuntimeError(f"steps[{idx}] is not a mapping")
        key = _require_str(raw.get("key"), "key", idx)
        title = _require_str(raw.get("title"), "title", idx)
        globs = _normalize_list(raw.get("globs"))
        path_keywords = _normalize_list(raw.get("path_keywords"))
        snippet_patterns = _normalize_list(raw.get("snippet_regexes"))
        max_files = raw.get("max_files")
        if max_files is not None and not isinstance(max_files, int):
            raise RuntimeError(f'steps[{idx}].max_files must be an int or null')
        steps.append(
            Step(
                key=key,
                title=title,
                globs=globs,
                path_keywords=path_keywords,
                snippet_regexes=compile_rx(snippet_patterns),
                max_files=max_files,
            )
        )
    return steps


STEPS: List[Step] = load_steps()
