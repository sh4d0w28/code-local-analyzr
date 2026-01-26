"""Heuristic enrichment for profiles (datastore/dependency hints)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .repo_scan import (
    IGNORE_DIRS,
    IGNORE_EXTS,
    is_probably_binary,
    read_file_head_tail,
    relposix,
)


DATASTORE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PostgreSQL", re.compile(r"\bpostgres(?:ql)?\b|jdbc:postgresql", re.IGNORECASE)),
    ("MySQL", re.compile(r"\bmysql\b|jdbc:mysql", re.IGNORECASE)),
    ("MariaDB", re.compile(r"\bmariadb\b", re.IGNORECASE)),
    ("MongoDB", re.compile(r"\bmongodb\b|mongodb://", re.IGNORECASE)),
    ("Redis", re.compile(r"\bredis\b|redis://", re.IGNORECASE)),
    ("Elasticsearch", re.compile(r"\belasticsearch\b", re.IGNORECASE)),
    ("OpenSearch", re.compile(r"\bopensearch\b", re.IGNORECASE)),
    ("Cassandra", re.compile(r"\bcassandra\b", re.IGNORECASE)),
    ("DynamoDB", re.compile(r"\bdynamodb\b", re.IGNORECASE)),
    ("ClickHouse", re.compile(r"\bclickhouse\b", re.IGNORECASE)),
    ("BigQuery", re.compile(r"\bbigquery\b", re.IGNORECASE)),
    ("Snowflake", re.compile(r"\bsnowflake\b", re.IGNORECASE)),
    ("Redshift", re.compile(r"\bredshift\b", re.IGNORECASE)),
    ("Oracle", re.compile(r"\boracle\b|jdbc:oracle", re.IGNORECASE)),
    ("SQL Server", re.compile(r"\bsqlserver\b|\bmssql\b|jdbc:sqlserver", re.IGNORECASE)),
    ("SQLite", re.compile(r"\bsqlite\b|jdbc:sqlite", re.IGNORECASE)),
]

SERVICE_URL_RE = re.compile(r"\b([a-z][a-z0-9+.-]*):\/\/([^/\s:@]+)", re.IGNORECASE)
HOST_KV_RE = re.compile(
    r"\b(host|hostname|endpoint|address|url)\b\s*[:=]\s*([A-Za-z0-9_.:-]+)",
    re.IGNORECASE,
)

HOST_BLACKLIST = {"localhost", "127.0.0.1", "0.0.0.0"}

CONFIG_EXTS = {
    ".yml",
    ".yaml",
    ".toml",
    ".properties",
    ".conf",
    ".ini",
    ".env",
    ".json",
    ".xml",
}

CONFIG_NAME_HINTS = (
    "application",
    "config",
    "settings",
    "values",
    "docker-compose",
    "compose",
)


def _has_ignored_dir(path_posix: str) -> bool:
    return any(seg in path_posix.lower().split("/") for seg in IGNORE_DIRS)


def _is_config_candidate(p: Path) -> bool:
    name = p.name.lower()
    if p.suffix.lower() in IGNORE_EXTS:
        return False
    if p.suffix.lower() in CONFIG_EXTS:
        return True
    return any(hint in name for hint in CONFIG_NAME_HINTS)


def _clean_host(value: str) -> Optional[str]:
    host = value.strip().strip('"').strip("'")
    if not host:
        return None
    if "$" in host or "{" in host or "}" in host:
        return None
    if host in HOST_BLACKLIST:
        return None
    if host.startswith("http://") or host.startswith("https://"):
        host = host.split("://", 1)[1]
    if "/" in host:
        host = host.split("/", 1)[0]
    if ":" in host:
        host = host.split(":", 1)[0]
    if not host or host in HOST_BLACKLIST:
        return None
    return host


def _scan_text(
    text: str,
    *,
    file_path: str,
    datastores: Dict[str, Set[str]],
    deps: Dict[str, Set[str]],
) -> None:
    for name, rx in DATASTORE_PATTERNS:
        if rx.search(text):
            datastores.setdefault(name, set()).add(file_path)
    for m in SERVICE_URL_RE.finditer(text):
        host = _clean_host(m.group(2))
        if host:
            deps.setdefault(host, set()).add(file_path)
    for m in HOST_KV_RE.finditer(text):
        host = _clean_host(m.group(2))
        if host:
            deps.setdefault(host, set()).add(file_path)


def infer_hints(
    repo: Path,
    files: Iterable[Path],
    *,
    max_file_bytes: int = 200_000,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Infer datastore and dependency hints from config-like files."""
    datastores: Dict[str, Set[str]] = {}
    deps: Dict[str, Set[str]] = {}

    for p in files:
        rp = relposix(repo, p)
        if _has_ignored_dir(rp):
            continue
        if not _is_config_candidate(p):
            continue
        if is_probably_binary(p):
            continue
        try:
            text = read_file_head_tail(p, max_bytes=max_file_bytes)
        except Exception:
            continue
        if not text:
            continue
        _scan_text(text, file_path=rp, datastores=datastores, deps=deps)
        if log:
            log(f"[HINTS] {rp} datastores={len(datastores)} deps={len(deps)}")

    return datastores, deps


def _join_evidence(paths: Set[str], *, limit: int = 3) -> str:
    if not paths:
        return ""
    items = sorted(paths)
    if len(items) > limit:
        return ", ".join(items[:limit]) + f", +{len(items) - limit} more"
    return ", ".join(items)


def enrich_profile(
    profile: dict,
    *,
    datastores: Dict[str, Set[str]],
    dependencies: Dict[str, Set[str]],
) -> dict:
    """Merge inferred hints into the profile."""
    if not isinstance(profile, dict):
        return profile

    stores = profile.get("data_stores") if isinstance(profile.get("data_stores"), list) else []
    deps = profile.get("dependencies_outbound") if isinstance(profile.get("dependencies_outbound"), list) else []

    seen_store_types = {str(s.get("type")).strip().lower() for s in stores if isinstance(s, dict)}
    for dtype, files in datastores.items():
        key = dtype.lower()
        evidence = _join_evidence(files)
        details = f"Detected in config: {evidence}" if evidence else "Detected in config"
        if key in seen_store_types:
            for s in stores:
                if not isinstance(s, dict):
                    continue
                if str(s.get("type", "")).strip().lower() == key and not str(s.get("details", "")).strip():
                    s["details"] = details
            continue
        stores.append({"type": dtype, "details": details})
        seen_store_types.add(key)

    seen_dep_targets = {str(d.get("target")).strip().lower() for d in deps if isinstance(d, dict)}
    for target, files in dependencies.items():
        key = target.lower()
        if key in seen_dep_targets:
            continue
        evidence = _join_evidence(files)
        reason = f"Detected in config: {evidence}" if evidence else "Detected in config"
        deps.append({"target": target, "reason": reason})
        seen_dep_targets.add(key)

    profile["data_stores"] = stores
    profile["dependencies_outbound"] = deps
    return profile
