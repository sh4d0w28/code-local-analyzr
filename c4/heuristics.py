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

CODE_DEP_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("gRPC service", re.compile(r"\bgrpc\.(NewClient|Dial)\s*\(", re.IGNORECASE)),
]
CODE_DATASTORE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("PostgreSQL", re.compile(r"\bpgxpool\.New\b", re.IGNORECASE)),
]

SERVICE_URL_RE = re.compile(r"\b([a-z][a-z0-9+.-]*):\/\/([^/\s:@]+)", re.IGNORECASE)
HOST_KV_RE = re.compile(
    r"\b(host|hostname|endpoint|address|url)\b\s*[:=]\s*([A-Za-z0-9_.:-]+)",
    re.IGNORECASE,
)
KEY_VALUE_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*[:=]\s*(.+?)\s*$")
KEY_SUFFIXES = (".url", ".uri", ".host", ".hostname", ".endpoint", ".address", ".baseurl", ".base_url")

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

CODE_SCAN_EXTS = {".go"}
ENDPOINT_KEY_TOKENS = {"url", "uri", "host", "hostname", "endpoint", "address"}
DATASTORE_KEY_TOKENS = {
    "db",
    "database",
    "postgres",
    "postgresql",
    "mysql",
    "mariadb",
    "mongo",
    "mongodb",
    "redis",
    "cassandra",
    "dynamo",
    "dynamodb",
    "clickhouse",
    "bigquery",
    "snowflake",
    "redshift",
    "oracle",
    "sqlserver",
    "mssql",
    "sqlite",
}


def _has_ignored_dir(path_posix: str) -> bool:
    return any(seg in path_posix.lower().split("/") for seg in IGNORE_DIRS)


def _is_config_candidate(p: Path) -> bool:
    name = p.name.lower()
    if p.suffix.lower() in IGNORE_EXTS:
        return False
    if name.startswith(".env"):
        return True
    if p.suffix.lower() in CONFIG_EXTS:
        return True
    return any(hint in name for hint in CONFIG_NAME_HINTS)


def _clean_host(value: str, *, allow_localhost: bool = False) -> Optional[str]:
    host = value.strip().strip('"').strip("'")
    if not host:
        return None
    if "$" in host or "{" in host or "}" in host:
        return None
    if not allow_localhost and host in HOST_BLACKLIST:
        return None
    if host.startswith("http://") or host.startswith("https://"):
        host = host.split("://", 1)[1]
    if "/" in host:
        host = host.split("/", 1)[0]
    if ":" in host:
        host = host.split(":", 1)[0]
    if not host or (not allow_localhost and host in HOST_BLACKLIST):
        return None
    return host


def _key_to_name(key: str) -> Optional[str]:
    """Derive a human-friendly system name from a config key."""
    if not key:
        return None
    low = key.lower()
    for suffix in KEY_SUFFIXES:
        if low.endswith(suffix):
            key = key[: -len(suffix)]
            break
    name = key.replace(".", " ").replace("_", " ").replace("-", " ").strip()
    if not name:
        return None
    return name


def _merge_origin(current: Optional[str], new: str) -> str:
    if not current:
        return new
    if current == new:
        return current
    if current == "config+code" or new == "config+code":
        return "config+code"
    return "config+code"


def _origin_label(origin: Optional[str]) -> str:
    if origin == "code":
        return "code"
    if origin == "config+code":
        return "config+code"
    return "config"


def _key_tokens(key: str) -> List[str]:
    return [tok for tok in re.split(r"[._-]", key.lower()) if tok]


def _key_looks_like_endpoint(key: str) -> bool:
    return any(tok in ENDPOINT_KEY_TOKENS for tok in _key_tokens(key))


def _key_looks_like_datastore(key: str) -> bool:
    return any(tok in DATASTORE_KEY_TOKENS for tok in _key_tokens(key))


def _value_looks_like_datastore(value: str) -> bool:
    for _, rx in DATASTORE_PATTERNS:
        if rx.search(value):
            return True
    return False


def _parse_host_value(value: str, *, allow_localhost: bool) -> Optional[str]:
    val = value.strip().strip('"').strip("'")
    if not val:
        return None
    if "://" in val:
        m = SERVICE_URL_RE.search(val)
        if m:
            return _clean_host(m.group(2), allow_localhost=allow_localhost)
    host = val.split()[0].strip().strip(",")
    if "/" in host:
        host = host.split("/", 1)[0]
    if "?" in host:
        host = host.split("?", 1)[0]
    return _clean_host(host, allow_localhost=allow_localhost)


def _add_datastore(
    datastores: Dict[str, dict],
    name: str,
    file_path: str,
    *,
    origin: str,
) -> None:
    entry = datastores.setdefault(name, {"evidence": set(), "origin": origin})
    entry["evidence"].add(file_path)
    entry["origin"] = _merge_origin(entry.get("origin"), origin)


def _add_dependency(
    deps: Dict[str, dict],
    target: str,
    file_path: str,
    *,
    origin: str,
    key: Optional[str] = None,
    name: Optional[str] = None,
) -> None:
    entry = deps.setdefault(target, {"evidence": set(), "name": None, "origin": origin})
    evidence = f"{file_path} ({key})" if key else file_path
    entry["evidence"].add(evidence)
    if name and not entry.get("name"):
        entry["name"] = name
    entry["origin"] = _merge_origin(entry.get("origin"), origin)


def _scan_text(
    text: str,
    *,
    file_path: str,
    datastores: Dict[str, dict],
    deps: Dict[str, dict],
) -> None:
    # First pass: line-based key/value parsing to capture config keys as evidence.
    for line in text.splitlines():
        m = KEY_VALUE_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip()
        value = m.group(2).strip()
        if not key or not value:
            continue
        name = _key_to_name(key)
        allow_localhost = bool(name)
        skip_dep = _key_looks_like_datastore(key) or _value_looks_like_datastore(value)
        if _key_looks_like_endpoint(key) and not skip_dep:
            host = _parse_host_value(value, allow_localhost=allow_localhost)
            if host:
                _add_dependency(deps, host, file_path, origin="config", key=key, name=name)
                continue
        if skip_dep:
            continue
        url_match = SERVICE_URL_RE.search(value)
        if url_match:
            host = _clean_host(url_match.group(2), allow_localhost=allow_localhost)
            if host:
                _add_dependency(deps, host, file_path, origin="config", key=key, name=name)
                continue
        host_match = HOST_KV_RE.search(value)
        if host_match:
            host = _clean_host(host_match.group(2), allow_localhost=allow_localhost)
            if host:
                _add_dependency(deps, host, file_path, origin="config", key=key, name=name)

    # Second pass: broader text scan (keeps existing behavior).
    for name, rx in DATASTORE_PATTERNS:
        if rx.search(text):
            _add_datastore(datastores, name, file_path, origin="config")
    for m in SERVICE_URL_RE.finditer(text):
        host = _clean_host(m.group(2))
        if host:
            _add_dependency(deps, host, file_path, origin="config")
    for m in HOST_KV_RE.finditer(text):
        host = _clean_host(m.group(2))
        if host:
            _add_dependency(deps, host, file_path, origin="config")


def _scan_code_text(
    text: str,
    *,
    file_path: str,
    datastores: Dict[str, dict],
    deps: Dict[str, dict],
) -> None:
    for name, rx in CODE_DATASTORE_PATTERNS:
        if rx.search(text):
            _add_datastore(datastores, name, file_path, origin="code")
    for name, rx in CODE_DEP_PATTERNS:
        if rx.search(text):
            _add_dependency(deps, name, file_path, origin="code")


def infer_hints(
    repo: Path,
    files: Iterable[Path],
    *,
    max_file_bytes: int = 200_000,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Infer datastore and dependency hints from config-like files."""
    datastores: Dict[str, dict] = {}
    deps: Dict[str, dict] = {}

    for p in files:
        rp = relposix(repo, p)
        if _has_ignored_dir(rp):
            continue
        is_config = _is_config_candidate(p)
        is_code = p.suffix.lower() in CODE_SCAN_EXTS
        if not is_config and not is_code:
            continue
        if is_probably_binary(p):
            continue
        try:
            text = read_file_head_tail(p, max_bytes=max_file_bytes)
        except Exception:
            continue
        if not text:
            continue
        if is_config:
            _scan_text(text, file_path=rp, datastores=datastores, deps=deps)
        if is_code:
            _scan_code_text(text, file_path=rp, datastores=datastores, deps=deps)
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
    datastores: Dict[str, dict],
    dependencies: Dict[str, dict],
) -> dict:
    """Merge inferred hints into the profile."""
    if not isinstance(profile, dict):
        return profile

    stores = profile.get("data_stores") if isinstance(profile.get("data_stores"), list) else []
    deps = profile.get("dependencies_outbound") if isinstance(profile.get("dependencies_outbound"), list) else []

    seen_store_types = {str(s.get("type")).strip().lower() for s in stores if isinstance(s, dict)}
    for dtype, meta in datastores.items():
        key = dtype.lower()
        if isinstance(meta, dict):
            files = meta.get("evidence") or set()
            origin = meta.get("origin")
        else:
            files = meta or set()
            origin = None
        evidence = _join_evidence(files)
        origin_label = _origin_label(origin)
        details = f"Detected in {origin_label}: {evidence}" if evidence else f"Detected in {origin_label}"
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
    for target, meta in dependencies.items():
        if isinstance(meta, dict):
            files = meta.get("evidence") or set()
            name = meta.get("name")
            origin = meta.get("origin")
        else:
            files = meta or set()
            name = None
            origin = None
        dep_target = target
        if str(target).strip().lower() in HOST_BLACKLIST and name:
            dep_target = name
        key = str(dep_target).strip().lower()
        if key in seen_dep_targets:
            continue
        evidence = _join_evidence(files)
        origin_label = _origin_label(origin)
        reason = f"Detected in {origin_label}: {evidence}" if evidence else f"Detected in {origin_label}"
        dep_entry = {"target": dep_target, "reason": reason}
        if name:
            dep_entry["name"] = name
        deps.append(dep_entry)
        seen_dep_targets.add(key)

    profile["data_stores"] = stores
    profile["dependencies_outbound"] = deps
    return profile
