"""Route extraction utilities for HTTP and gRPC endpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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

JAVA_METHOD_ANN_RE = re.compile(
    r"@(?P<verb>Get|Post|Put|Delete|Patch)Mapping"
    r"(?:\s*\((?P<args>[^)]*)\))?"
    r"\s*(?:public|private|protected)?"
    r"[\s\S]{0,240}?\b(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
JAVA_REQUEST_MAPPING_RE = re.compile(
    r"@RequestMapping"
    r"(?:\s*\((?P<args>[^)]*)\))?"
    r"\s*(?:public|private|protected)?"
    r"[\s\S]{0,240}?\b(?P<method>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
JAVA_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b")
JAVA_CLASS_DECL_RE = re.compile(
    r"(?P<ann>(?:@\w+(?:\([^)]*\))?\s*)*)"
    r"(?:public|private|protected)?\s*(?:final\s+)?(?:abstract\s+)?class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
PY_DECORATOR_RE = re.compile(r"@(?:app|router|bp)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]")
PY_ROUTE_RE = re.compile(r"@(?:app|router|bp)\.route\s*\(\s*['\"]([^'\"]+)['\"].*?methods\s*=\s*\[([^\]]+)\]")
JS_ROUTE_RE = re.compile(r"\b(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]")
GO_ROUTE_RE = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9_]*\.)?(GET|POST|PUT|DELETE|PATCH)\s*\(\s*\"([^\"]+)\"")
PROTO_PACKAGE_RE = re.compile(r"^\s*package\s+([A-Za-z0-9_.]+)\s*;", re.MULTILINE)
PROTO_SERVICE_RE = re.compile(r"\bservice\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")
PROTO_RPC_RE = re.compile(r"\brpc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
PROTO_HTTP_RE = re.compile(r"\b(get|post|put|delete|patch)\s*:\s*\"([^\"]+)\"", re.IGNORECASE)
PROTO_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
PROTO_COMMENT_LINE_RE = re.compile(r"//.*?$", re.MULTILINE)


def _has_ignored_dir(path_posix: str) -> bool:
    """Check if a path segment is under an ignored directory."""
    return any(seg in path_posix.lower().split("/") for seg in IGNORE_DIRS)


def _first_string_literal(args: Optional[str]) -> str:
    """Extract the first quoted string literal from an annotation arg list."""
    if not args:
        return ""
    m = re.search(r"\"([^\"]+)\"", args)
    return m.group(1) if m else ""


def _join_paths(prefix: str, path: str) -> str:
    """Join class and method paths while handling slashes."""
    if not prefix:
        return path
    if not path:
        return prefix
    if prefix.endswith("/") and path.startswith("/"):
        return prefix[:-1] + path
    if not prefix.endswith("/") and not path.startswith("/"):
        return prefix + "/" + path
    return prefix + path


def _sanitize_path_for_mermaid(path: str) -> str:
    """Rewrite route params to be Mermaid-friendly."""
    if not path:
        return path
    return re.sub(r"\{([^}]+)\}", r":\1", path)


def _java_class_name(text: str) -> str:
    """Extract a simple Java class name from source text."""
    m = JAVA_CLASS_RE.search(text)
    return m.group(1) if m else ""


def _java_class_prefix(annotations: str) -> str:
    """Extract a class-level @RequestMapping path prefix."""
    m = re.search(r"@RequestMapping(?:\s*\(([^)]*)\))?", annotations)
    if not m:
        return ""
    return _first_string_literal(m.group(1))


def _extract_java_classes(text: str) -> List[Tuple[str, str, str]]:
    """Extract Java class bodies along with @RequestMapping prefixes."""
    classes: List[Tuple[str, str, str]] = []
    idx = 0
    while True:
        m = JAVA_CLASS_DECL_RE.search(text, idx)
        if not m:
            break
        ann = m.group("ann") or ""
        name = m.group("name")
        prefix = _java_class_prefix(ann)
        brace_start = text.find("{", m.end())
        if brace_start == -1:
            idx = m.end()
            continue
        depth = 0
        end = None
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            break
        body = text[brace_start + 1 : end]
        classes.append((name, prefix, body))
        idx = end + 1
    return classes


def _java_routes(text: str, file_path: str) -> List[dict]:
    """Extract Spring routes from Java controllers."""
    routes: List[dict] = []
    classes = _extract_java_classes(text)
    if not classes:
        classes = [(_java_class_name(text), "", text)]

    for class_name, prefix, body in classes:
        for m in JAVA_METHOD_ANN_RE.finditer(body):
            verb = m.group("verb").upper()
            args = m.group("args")
            path = _join_paths(prefix, _first_string_literal(args))
            handler = m.group("method")
            routes.append(
                {
                    "method": verb,
                    "path": path,
                    "file": file_path,
                    "handler": f"{class_name}.{handler}" if class_name else handler,
                    "framework": "spring",
                }
            )

        for m in JAVA_REQUEST_MAPPING_RE.finditer(body):
            args = m.group("args") or ""
            method_name = m.group("method")
            methods = re.findall(r"RequestMethod\.(GET|POST|PUT|DELETE|PATCH)", args)
            if not methods:
                methods = ["ANY"]
            path = _join_paths(prefix, _first_string_literal(args))
            for verb in methods:
                routes.append(
                    {
                        "method": verb,
                        "path": path,
                        "file": file_path,
                        "handler": f"{class_name}.{method_name}" if class_name else method_name,
                        "framework": "spring",
                    }
                )
    return routes


def _python_routes(text: str, file_path: str) -> List[dict]:
    """Extract Flask/FastAPI-style routes from Python code."""
    routes: List[dict] = []
    for m in PY_DECORATOR_RE.finditer(text):
        routes.append(
            {
                "method": m.group(1).upper(),
                "path": m.group(2),
                "file": file_path,
                "handler": "",
                "framework": "python",
            }
        )
    for m in PY_ROUTE_RE.finditer(text):
        path = m.group(1)
        methods_raw = m.group(2)
        methods = re.findall(r"['\"](GET|POST|PUT|DELETE|PATCH)['\"]", methods_raw, re.IGNORECASE)
        if not methods:
            methods = ["ANY"]
        for verb in methods:
            routes.append(
                {
                    "method": verb.upper(),
                    "path": path,
                    "file": file_path,
                    "handler": "",
                    "framework": "python",
                }
            )
    return routes


def _js_routes(text: str, file_path: str) -> List[dict]:
    """Extract Express-style routes from JS/TS code."""
    routes: List[dict] = []
    for m in JS_ROUTE_RE.finditer(text):
        routes.append(
            {
                "method": m.group(1).upper(),
                "path": m.group(2),
                "file": file_path,
                "handler": "",
                "framework": "node",
            }
        )
    return routes


def _go_routes(text: str, file_path: str) -> List[dict]:
    """Extract Go HTTP routes from common router patterns."""
    routes: List[dict] = []
    for m in GO_ROUTE_RE.finditer(text):
        routes.append(
            {
                "method": m.group(1).upper(),
                "path": m.group(2),
                "file": file_path,
                "handler": "",
                "framework": "go",
            }
        )
    return routes


def _strip_proto_comments(text: str) -> str:
    """Remove proto comments to simplify parsing."""
    text = PROTO_COMMENT_BLOCK_RE.sub("", text)
    return PROTO_COMMENT_LINE_RE.sub("", text)


def _extract_service_blocks(text: str) -> List[Tuple[str, str]]:
    """Extract proto service blocks by matching braces."""
    blocks: List[Tuple[str, str]] = []
    idx = 0
    while True:
        m = PROTO_SERVICE_RE.search(text, idx)
        if not m:
            break
        name = m.group(1)
        brace_start = text.find("{", m.end() - 1)
        if brace_start == -1:
            break
        depth = 0
        end = None
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            break
        body = text[brace_start + 1 : end]
        blocks.append((name, body))
        idx = end + 1
    return blocks


def _extract_rpc_blocks(service_body: str) -> List[Tuple[str, str]]:
    """Extract rpc blocks inside a proto service body."""
    blocks: List[Tuple[str, str]] = []
    idx = 0
    while True:
        m = PROTO_RPC_RE.search(service_body, idx)
        if not m:
            break
        name = m.group(1)
        scan = m.end()
        while scan < len(service_body) and service_body[scan] not in "{;":
            scan += 1
        if scan >= len(service_body):
            idx = m.end()
            continue
        if service_body[scan] == ";":
            blocks.append((name, ""))
            idx = scan + 1
            continue
        brace_start = scan
        depth = 0
        end = None
        for i in range(brace_start, len(service_body)):
            if service_body[i] == "{":
                depth += 1
            elif service_body[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            idx = m.end()
            continue
        body = service_body[brace_start + 1 : end]
        blocks.append((name, body))
        idx = end + 1
    return blocks


def _proto_routes(text: str, file_path: str) -> List[dict]:
    """Extract gRPC and HTTP-mapped routes from proto files."""
    routes: List[dict] = []
    cleaned = _strip_proto_comments(text)
    pkg_match = PROTO_PACKAGE_RE.search(cleaned)
    package = pkg_match.group(1) if pkg_match else ""
    for service_name, service_body in _extract_service_blocks(cleaned):
        full_service = f"{package}.{service_name}" if package else service_name
        for rpc_name, rpc_body in _extract_rpc_blocks(service_body):
            http_methods = PROTO_HTTP_RE.findall(rpc_body)
            handler = f"{full_service}/{rpc_name}"
            if http_methods:
                for verb, path in http_methods:
                    routes.append(
                        {
                            "method": verb.upper(),
                            "path": path,
                            "file": file_path,
                            "handler": handler,
                            "framework": "grpc",
                        }
                    )
            else:
                routes.append(
                    {
                        "method": "GRPC",
                        "path": f"/{handler}",
                        "file": file_path,
                        "handler": handler,
                        "framework": "grpc",
                    }
                )
    return routes


def _is_text_candidate(p: Path) -> bool:
    """Decide whether a file is a text candidate for route scanning."""
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


def _read_file_limited(
    p: Path,
    *,
    max_bytes: int,
    head_lines: int = 240,
    tail_lines: int = 120,
) -> str:
    """Read head/tail content with a byte cap."""
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


def extract_routes(text: str, file_path: str) -> List[dict]:
    """Extract routes for a file based on extension."""
    routes: List[dict] = []
    if file_path.endswith((".java", ".kt", ".kts")):
        routes.extend(_java_routes(text, file_path))
    if file_path.endswith((".py",)):
        routes.extend(_python_routes(text, file_path))
    if file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
        routes.extend(_js_routes(text, file_path))
    if file_path.endswith((".go",)):
        routes.extend(_go_routes(text, file_path))
    if file_path.endswith((".proto",)):
        routes.extend(_proto_routes(text, file_path))
    return routes


def build_routes_profile(
    repo: Path,
    files: List[Path],
    *,
    out_path: Path,
    max_file_bytes: int,
    log: Optional[Callable[[str], None]] = None,
    verbose: bool = False,
) -> List[dict]:
    """Scan files and write a JSONL routes profile."""
    routes: List[dict] = []
    scanned = 0
    matched_files = 0

    for p in files:
        rp = relposix(repo, p)
        if _has_ignored_dir(rp):
            continue
        if not _is_text_candidate(p):
            continue
        content = _read_file_limited(p, max_bytes=max_file_bytes)
        if not content:
            continue
        content = redact(content)
        scanned += 1
        file_routes = extract_routes(content, rp)
        if file_routes:
            matched_files += 1
            routes.extend(file_routes)
            if log and verbose:
                log(f"[ROUTES] {rp} routes={len(file_routes)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in routes) + "\n",
        encoding="utf-8",
    )

    if log:
        log(f"[ROUTES] done files_scanned={scanned} files_with_routes={matched_files} total_routes={len(routes)}")

    return routes


def format_routes_text(
    routes: List[dict],
    *,
    limit: int = 2000,
    sanitize_for_mermaid: bool = False,
) -> str:
    """Render routes into a compact text block for prompts."""
    lines = ["ROUTES_PROFILE (method path | file | handler):", "-----"]
    for r in routes[:limit]:
        method = r.get("method") or "ANY"
        path = r.get("path") or "<unknown>"
        if sanitize_for_mermaid:
            path = _sanitize_path_for_mermaid(path)
        file_path = r.get("file") or "<unknown>"
        handler = r.get("handler") or ""
        if handler:
            lines.append(f"{method} {path} | {file_path} | {handler}")
        else:
            lines.append(f"{method} {path} | {file_path}")
    if len(routes) > limit:
        lines.append(f"... truncated, total_routes={len(routes)}")
    return "\n".join(lines)
