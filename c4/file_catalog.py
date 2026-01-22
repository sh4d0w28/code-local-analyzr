"""LLM-driven file classification and catalog caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .json_tools import best_effort_json_text
from .ollama_client import ollama_chat
from .prompts import FILE_CLASSIFY_REPAIR_SYSTEM, FILE_CLASSIFY_SYSTEM
from .repo_scan import (
    IGNORE_DIRS,
    IGNORE_EXTS,
    TEXT_EXT_ALLOWLIST,
    is_probably_binary,
    is_special_text_filename,
    read_file_head_tail,
    redact,
    relposix,
)
from .steps import Step


def _has_ignored_dir(path_posix: str) -> bool:
    """Check if a path is under an ignored directory."""
    return any(seg in path_posix.lower().split("/") for seg in IGNORE_DIRS)


def file_is_text_candidate_for_catalog(p: Path) -> bool:
    """Decide whether a file should be classified."""
    if is_special_text_filename(p.name):
        return True
    ext = p.suffix.lower()
    if ext in IGNORE_EXTS:
        return False
    if is_probably_binary(p):
        return False
    if ext in TEXT_EXT_ALLOWLIST:
        return True
    return True


def read_file_head_tail_limited(
    p: Path,
    *,
    max_bytes: int,
    head_lines: int = 240,
    tail_lines: int = 120,
) -> str:
    """Read file head/tail with a stricter byte cap."""
    try:
        size = p.stat().st_size
    except Exception:
        size = None
    if size is None or size <= max_bytes:
        return read_file_head_tail(p, max_bytes=max_bytes, head_lines=head_lines, tail_lines=tail_lines)

    head_raw = b""
    tail_raw = b""
    try:
        with p.open("rb") as f:
            head_raw = f.read(max_bytes)
            try:
                f.seek(max(size - max_bytes, 0))
                tail_raw = f.read(max_bytes)
            except Exception:
                tail_raw = b""
    except Exception:
        return ""

    head_text = head_raw.decode("utf-8", errors="ignore")
    tail_text = tail_raw.decode("utf-8", errors="ignore")
    head = head_text.splitlines()[:head_lines]
    tail = tail_text.splitlines()[-tail_lines:] if tail_lines > 0 else []
    if not tail:
        return "\n".join(head)
    return "\n".join(head) + "\n\n[...TRUNCATED...]\n\n" + "\n".join(tail)


def _load_catalog(path: Path) -> Dict[str, dict]:
    """Load an existing JSONL catalog into memory."""
    if not path.exists():
        return {}
    out: Dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            continue
        p = entry.get("path")
        if isinstance(p, str) and p:
            out[p] = entry
    return out


def load_file_catalog(path: Path) -> Dict[str, dict]:
    """Public helper to load a JSONL file catalog."""
    return _load_catalog(path)


def _write_catalog(path: Path, entries: Iterable[dict]) -> None:
    """Write catalog entries as JSONL, sorted by path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(entries, key=lambda e: e.get("path", ""))
    lines = [json.dumps(e, ensure_ascii=False) for e in ordered]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_steps(steps: List[Step]) -> str:
    """Render step keys/titles for classification prompts."""
    return "\n".join(f"- {s.key}: {s.title}" for s in steps)


def _parse_classification(
    raw: str,
    *,
    allowed_categories: set[str],
    ollama_base: str,
    model: str,
    timeout_s: int,
    num_predict: int,
    num_ctx: int,
    available_steps_text: str,
    repair_system: Optional[str],
    log: Optional[Callable[[str], None]],
) -> Optional[dict]:
    """Parse or repair the classifier JSON response."""
    cleaned = best_effort_json_text(raw) or raw
    parsed: Optional[dict]
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = None

    if parsed is None and repair_system:
        repaired = ollama_chat(
            ollama_base, model,
            repair_system,
            "AVAILABLE_STEPS:\n" + available_steps_text + "\n\nINPUT_TEXT:\n" + raw,
            temperature=0.0,
            timeout_s=timeout_s,
            num_predict=num_predict,
            num_ctx=num_ctx,
            log=log,
            label="classify_repair",
        )
        repaired_clean = best_effort_json_text(repaired) or repaired
        try:
            parsed = json.loads(repaired_clean)
        except Exception:
            parsed = None

    if not isinstance(parsed, dict):
        return None

    categories = parsed.get("categories")
    if not isinstance(categories, list):
        categories = []
    filtered: List[str] = []
    seen: set[str] = set()
    for c in categories:
        if not isinstance(c, str):
            continue
        if c in allowed_categories and c not in seen:
            filtered.append(c)
            seen.add(c)

    summary = parsed.get("summary")
    if not isinstance(summary, str):
        summary = ""

    return {"categories": filtered, "summary": summary}


def build_file_catalog(
    repo: Path,
    files: List[Path],
    steps: List[Step],
    *,
    out_path: Path,
    ollama_base: str,
    model: str,
    timeout_s: int,
    num_predict: int,
    num_ctx: int,
    max_file_bytes: int,
    log: Optional[Callable[[str], None]] = None,
    llm_log: Optional[Callable[[str], None]] = None,
    verbose: bool = False,
) -> Dict[str, dict]:
    """Classify files with the LLM and write a JSONL catalog."""
    if not FILE_CLASSIFY_SYSTEM:
        raise RuntimeError('Missing "file_classify_system" prompt in config')
    allowed_categories = {s.key for s in steps}
    available_steps_text = _format_steps(steps)
    cached = _load_catalog(out_path)
    entries: List[dict] = []
    by_path: Dict[str, dict] = {}

    # Pre-filter candidates so progress reflects actual files scanned.
    candidates: List[Path] = []
    for p in files:
        rp = relposix(repo, p)
        if _has_ignored_dir(rp):
            continue
        if not file_is_text_candidate_for_catalog(p):
            continue
        candidates.append(p)

    total = len(candidates)
    if log:
        log(f"[CLASSIFY] start files={total} (0%)")

    reused = 0
    analyzed = 0

    progress_every = max(1, total // 20) if total else 1
    processed = 0

    def _log_progress() -> None:
        """Emit periodic progress during classification."""
        if not log:
            return
        if processed == total or processed % progress_every == 0:
            percent = int((processed / total) * 100) if total else 100
            remaining = max(total - processed, 0)
            log(f"[CLASSIFY] progress {processed}/{total} ({percent}%) remaining={remaining}")

    for p in candidates:
        processed += 1
        rp = relposix(repo, p)

        try:
            stat = p.stat()
        except Exception:
            _log_progress()
            continue
        fingerprint = f"{stat.st_size}:{int(stat.st_mtime)}"
        # Cache by size+mtime+model to avoid re-classifying unchanged files.
        cached_entry = cached.get(rp)
        if (
            cached_entry
            and cached_entry.get("fingerprint") == fingerprint
            and cached_entry.get("model") == model
        ):
            entries.append(cached_entry)
            by_path[rp] = cached_entry
            reused += 1
            if log and verbose:
                log(f"[CLASSIFY] reuse {rp}")
            _log_progress()
            continue

        content = read_file_head_tail_limited(p, max_bytes=max_file_bytes)
        content = redact(content)
        user = (
            f"REPO: {repo.name}\n"
            f"PATH: {repo}\n"
            f"FILE: {rp}\n\n"
            "AVAILABLE_STEPS:\n"
            f"{available_steps_text}\n\n"
            "FILE_CONTENT:\n"
            f"{content}"
        )
        raw = ollama_chat(
            ollama_base, model,
            FILE_CLASSIFY_SYSTEM,
            user,
            temperature=0.0,
            timeout_s=timeout_s,
            num_predict=num_predict,
            num_ctx=num_ctx,
            log=llm_log,
            label="classify",
        )
        parsed = _parse_classification(
            raw,
            allowed_categories=allowed_categories,
            ollama_base=ollama_base,
            model=model,
            timeout_s=timeout_s,
            num_predict=num_predict,
            num_ctx=num_ctx,
            available_steps_text=available_steps_text,
            repair_system=FILE_CLASSIFY_REPAIR_SYSTEM,
            log=llm_log,
        )
        categories: List[str] = []
        summary = ""
        if parsed:
            categories = parsed.get("categories", [])
            summary = parsed.get("summary", "")

        entry = {
            "path": rp,
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
            "fingerprint": fingerprint,
            "categories": categories,
            "summary": summary,
            "model": model,
        }
        entries.append(entry)
        by_path[rp] = entry
        analyzed += 1
        if log and verbose:
            log(f"[CLASSIFY] analyze {rp} categories={len(categories)}")

        _log_progress()

    _write_catalog(out_path, entries)

    if log:
        log(f"[CLASSIFY] done analyzed={analyzed} reused={reused} catalog={out_path} (100%)")

    return by_path
