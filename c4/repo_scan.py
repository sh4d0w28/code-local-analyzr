"""Repository scanning and evidence extraction helpers."""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from .config import get_scan_rules
from .steps import Step

SCAN_RULES = get_scan_rules()

IGNORE_DIRS = {
    str(d).lower()
    for d in SCAN_RULES.get("ignore_dirs", [])
    if isinstance(d, str) and d.strip()
}
IGNORE_EXTS = {
    str(e).lower()
    for e in SCAN_RULES.get("ignore_exts", [])
    if isinstance(e, str) and e.strip()
}
TEXT_EXT_ALLOWLIST = {
    str(e).lower()
    for e in SCAN_RULES.get("text_ext_allowlist", [])
    if isinstance(e, str) and e.strip()
}
SPECIAL_TEXT_FILENAME_PATTERNS = [
    str(p).lower()
    for p in SCAN_RULES.get("special_text_filename_patterns", [])
    if isinstance(p, str) and p.strip()
]

PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 \-]*PRIVATE KEY-----.*?-----END [A-Z0-9 \-]*PRIVATE KEY-----",
    re.DOTALL,
)
AUTH_BEARER_RE = re.compile(r"(?i)(authorization:\s*bearer\s+)([A-Za-z0-9\-._~+/]+=*)")
SIMPLE_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*([^\s'\"`]+)")


def redact(text: str) -> str:
    """Redact secrets from extracted text snippets."""
    text = PRIVATE_KEY_BLOCK_RE.sub("[REDACTED_PRIVATE_KEY_BLOCK]", text)
    text = AUTH_BEARER_RE.sub(r"\1[REDACTED]", text)

    def _repl(m: re.Match) -> str:
        """Mask key/value style secrets."""
        return f"{m.group(1)}=[REDACTED]"

    return SIMPLE_SECRET_RE.sub(_repl, text)


def is_probably_binary(p: Path) -> bool:
    """Heuristic to detect binary files by null bytes."""
    try:
        with p.open("rb") as f:
            chunk = f.read(4096)
        return b"\x00" in chunk
    except Exception:
        return True


def relposix(base: Path, p: Path) -> str:
    """Return a POSIX-style relative path."""
    return p.relative_to(base).as_posix()


def is_special_text_filename(name: str) -> bool:
    """Check for filenames that should be treated as text."""
    low = name.lower()
    for pat in SPECIAL_TEXT_FILENAME_PATTERNS:
        if fnmatch.fnmatch(low, pat):
            return True
    return False


def file_is_text_candidate(p: Path) -> bool:
    """Decide whether a file should be considered text for scanning."""
    if is_special_text_filename(p.name):
        return True
    ext = p.suffix.lower()
    if ext in IGNORE_EXTS:
        return False
    if ext == "":
        return False
    if ext in TEXT_EXT_ALLOWLIST:
        return True
    return False


def list_repo_files(repo: Path) -> List[Path]:
    """List candidate files, preferring git-tracked files when possible."""
    # Prefer git-tracked files to avoid build outputs and vendored noise.
    if (repo / ".git").exists():
        try:
            out = subprocess.check_output(
                ["git", "-C", str(repo), "ls-files", "-z"],
                stderr=subprocess.DEVNULL,
            )
            files: List[Path] = []
            for b in out.split(b"\x00"):
                if not b:
                    continue
                p = repo / b.decode("utf-8", errors="ignore")
                if p.is_file():
                    files.append(p)
            return files
        except Exception:
            pass

    files: List[Path] = []
    for root, dirs, filenames in os.walk(repo):
        dirs[:] = [d for d in dirs if d.lower() not in IGNORE_DIRS]
        for fn in filenames:
            p = Path(root) / fn
            if p.is_file():
                files.append(p)
    return files


def read_file_head_tail(
    p: Path,
    *,
    max_bytes: int,
    head_lines: int = 240,
    tail_lines: int = 120,
) -> str:
    """Read a truncated view of a file for evidence."""
    raw = p.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if len(raw) <= max_bytes:
        return text

    lines = text.splitlines()
    head = lines[:head_lines]
    tail = lines[-tail_lines:] if len(lines) > (head_lines + tail_lines) else []
    return "\n".join(head) + "\n\n[...TRUNCATED...]\n\n" + "\n".join(tail)


def iter_file_chunks(p: Path, *, chunk_bytes: int) -> Iterable[tuple[str, int, int]]:
    """Yield file chunks as text along with byte offsets (start, end)."""
    if chunk_bytes <= 0:
        return
    try:
        with p.open("rb") as f:
            remainder = b""
            offset = 0
            while True:
                data = f.read(chunk_bytes)
                if not data:
                    if remainder:
                        chunk = remainder
                        start = offset
                        end = offset + len(chunk)
                        yield chunk.decode("utf-8", errors="ignore"), start, end
                    break
                buf = remainder + data
                cut = buf.rfind(b"\n")
                if cut == -1:
                    chunk = buf
                    remainder = b""
                else:
                    chunk = buf[: cut + 1]
                    remainder = buf[cut + 1 :]
                if not chunk:
                    continue
                start = offset
                end = offset + len(chunk)
                yield chunk.decode("utf-8", errors="ignore"), start, end
                offset = end
    except Exception:
        return


def extract_snippets(
    p: Path,
    *,
    regexes: List[re.Pattern],
    max_snippets: int,
    context_lines: int,
    max_bytes: int,
) -> str:
    """Extract regex-based snippets or a head/tail fallback."""
    # Collect a few hit regions and merge overlaps to keep evidence compact.
    raw = p.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    hits: List[Tuple[int, str]] = []
    for i, line in enumerate(lines):
        for rx in regexes:
            if rx.search(line):
                hits.append((i, line))
                break
        if len(hits) >= max_snippets:
            break

    if not hits:
        return read_file_head_tail(p, max_bytes=max_bytes)

    spans: List[Tuple[int, int]] = []
    for i, _ in hits:
        start = max(0, i - context_lines)
        end = min(len(lines), i + context_lines + 1)
        spans.append((start, end))

    spans.sort()
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    out_parts: List[str] = []
    for s, e in merged:
        out_parts.append("\n".join(lines[s:e]))
        out_parts.append("\n---\n")

    out = "\n".join(out_parts)
    if len(out.encode("utf-8", errors="ignore")) > max_bytes:
        return read_file_head_tail(p, max_bytes=max_bytes)
    return out


def glob_match(path_posix: str, pattern: str) -> bool:
    """Match a path against a glob pattern."""
    return fnmatch.fnmatch(path_posix, pattern)


def select_step_files(repo: Path, files: List[Path], step: Step) -> List[Path]:
    """Select files relevant to a step based on globs/keywords."""
    selected: List[Path] = []

    for p in files:
        rp = relposix(repo, p)
        low = rp.lower()

        if any(seg in low.split("/") for seg in IGNORE_DIRS):
            continue

        if not file_is_text_candidate(p):
            continue
        if is_probably_binary(p):
            continue

        matched = False
        for g in step.globs:
            if glob_match(rp, g):
                matched = True
                break
        if not matched and step.path_keywords:
            if any(k in low for k in step.path_keywords):
                matched = True

        if matched:
            selected.append(p)

    selected.sort(key=lambda x: relposix(repo, x))
    if step.max_files is not None:
        selected = selected[: step.max_files]
    return selected


def build_step_evidence(
    repo: Path,
    files: List[Path],
    step: Step,
    *,
    max_step_bytes: int,
    max_file_bytes: int,
    max_snippets_per_file: int,
    snippet_context_lines: int,
    chunk_large_files: bool = False,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Build the evidence blob for a single step."""
    # Assemble per-step evidence, enforcing a total byte budget.
    parts: List[str] = []
    total = 0

    header = [
        f"REPO: {repo.name}",
        f"PATH: {repo}",
        f"STEP: {step.key} - {step.title}",
        "-----",
    ]
    header_text = "\n".join(header)
    parts.append(header_text)
    total = len(header_text.encode("utf-8", errors="ignore"))

    for p in files:
        rp = relposix(repo, p)
        if log is not None:
            log(f"[ANALYZE] {step.key} {rp}")
        try:
            if step.snippet_regexes:
                content = extract_snippets(
                    p,
                    regexes=step.snippet_regexes,
                    max_snippets=max_snippets_per_file,
                    context_lines=snippet_context_lines,
                    max_bytes=max_file_bytes,
                )
                content = redact(content)
                chunk = f"\n===== FILE: {rp} =====\n{content}\n"
                b = chunk.encode("utf-8", errors="ignore")
                if total + len(b) > max_step_bytes:
                    if log is not None:
                        log(f"[LIMIT] {step.key} {rp} step_bytes_cap={max_step_bytes}")
                    parts.append("\n[STEP_EVIDENCE_LIMIT_REACHED]\n")
                    break

                parts.append(chunk)
                total += len(b)
                continue

            use_chunks = False
            file_size = None
            if chunk_large_files:
                try:
                    file_size = p.stat().st_size
                except Exception:
                    file_size = None
                if file_size is not None and file_size > max_file_bytes:
                    use_chunks = True

            if use_chunks:
                est_total = None
                if file_size is not None and max_file_bytes > 0:
                    est_total = max(1, (file_size + max_file_bytes - 1) // max_file_bytes)
                chunk_index = 0
                for chunk_text, start, end in iter_file_chunks(p, chunk_bytes=max_file_bytes):
                    chunk_index += 1
                    if log is not None:
                        est_label = est_total if est_total is not None else "?"
                        log(
                            f"[CHUNK] {step.key} {rp} part={chunk_index}/{est_label} bytes={start}-{end}"
                        )
                    content = redact(chunk_text)
                    chunk_label = f"{rp} (chunk {chunk_index}"
                    if est_total is not None:
                        chunk_label += f"/{est_total}"
                    chunk_label += f", bytes {start}-{end})"
                    chunk = f"\n===== FILE: {chunk_label} =====\n{content}\n"
                    b = chunk.encode("utf-8", errors="ignore")
                    if total + len(b) > max_step_bytes:
                        if log is not None:
                            log(f"[LIMIT] {step.key} {rp} step_bytes_cap={max_step_bytes}")
                        parts.append("\n[STEP_EVIDENCE_LIMIT_REACHED]\n")
                        return "\n".join(parts)
                    parts.append(chunk)
                    total += len(b)
                continue

            content = read_file_head_tail(p, max_bytes=max_file_bytes)
        except Exception as e:
            if log is not None:
                log(f"[SKIP] {step.key} {rp} read_error={type(e).__name__}")
            continue

        content = redact(content)
        chunk = f"\n===== FILE: {rp} =====\n{content}\n"
        b = chunk.encode("utf-8", errors="ignore")
        if total + len(b) > max_step_bytes:
            if log is not None:
                log(f"[LIMIT] {step.key} {rp} step_bytes_cap={max_step_bytes}")
            parts.append("\n[STEP_EVIDENCE_LIMIT_REACHED]\n")
            break

        parts.append(chunk)
        total += len(b)

    return "\n".join(parts)


def build_step_sources(
    repo: Path,
    files: List[Path],
    step: Step,
    *,
    include_file_size: bool,
    include_mtime: bool,
    fmt: str,
) -> str:
    """Build a per-step sources list with optional metadata."""
    header = [
        f"REPO: {repo.name}",
        f"PATH: {repo}",
        f"STEP: {step.key} - {step.title}",
        "-----",
    ]
    lines = header[:]
    if fmt.lower() == "tsv":
        cols = ["path"]
        if include_file_size:
            cols.append("bytes")
        if include_mtime:
            cols.append("mtime_epoch")
        lines.append("\t".join(cols))

    total_bytes = 0
    for p in files:
        rp = relposix(repo, p)
        size = None
        mtime = None
        try:
            stat = p.stat()
            size = stat.st_size
            mtime = int(stat.st_mtime)
        except Exception:
            pass

        row = [rp]
        if include_file_size:
            row.append(str(size or 0))
            if size is not None:
                total_bytes += size
        if include_mtime:
            row.append(str(mtime or 0))

        lines.append("\t".join(row) if fmt.lower() == "tsv" else " ".join(row))

    lines.append("")
    lines.append(f"TOTAL_FILES: {len(files)}")
    if include_file_size:
        lines.append(f"TOTAL_BYTES: {total_bytes}")

    return "\n".join(lines)


def validate_catalog(root: Path, repos: list[dict]) -> Tuple[List[Tuple[str, Path]], List[str]]:
    """Validate catalog entries and return resolved repo paths."""
    errors: List[str] = []
    ok: List[Tuple[str, Path]] = []

    if not root.exists():
        errors.append(f'Catalog root does not exist: "{root}"')
        return ok, errors
    if not root.is_dir():
        errors.append(f'Catalog root is not a directory: "{root}"')
        return ok, errors

    seen_names: set[str] = set()
    seen_paths: set[Path] = set()

    for i, r in enumerate(repos):
        name = r.get("name")
        path = r.get("path")

        if not name or not isinstance(name, str):
            errors.append(f"repos[{i}].name is missing or not a string")
            continue
        if not path or not isinstance(path, str):
            errors.append(f"repos[{i}].path is missing or not a string (repo={name})")
            continue

        if name in seen_names:
            errors.append(f'Duplicate repo name "{name}"')
        seen_names.add(name)

        repo_abs = (root / path).expanduser().resolve()

        try:
            repo_abs.relative_to(root)
        except ValueError:
            errors.append(f'Repo "{name}" path escapes root: "{path}" -> "{repo_abs}" (root="{root}")')
            continue

        if repo_abs in seen_paths:
            errors.append(f'Duplicate repo path "{repo_abs}" (repo={name})')
        seen_paths.add(repo_abs)

        if not repo_abs.exists():
            errors.append(f'Repo folder not found for "{name}": "{repo_abs}"')
            continue
        if not repo_abs.is_dir():
            errors.append(f'Repo path is not a directory for "{name}": "{repo_abs}"')
            continue

        ok.append((name, repo_abs))

    return ok, errors
