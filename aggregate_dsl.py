#!/usr/bin/env python3
"""Build a merged Structurizr DSL across repo outputs."""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

INFRA_ALIASES = {
    "kafka": "Kafka",
    "redis": "Redis",
    "postgresql": "PostgreSQL",
    "postgres": "PostgreSQL",
    "pgsql": "PostgreSQL",
    "psql": "PostgreSQL",
    "mysql": "MySQL",
    "mariadb": "MariaDB",
    "mongodb": "MongoDB",
    "mongo": "MongoDB",
    "cassandra": "Cassandra",
    "dynamodb": "DynamoDB",
    "dynamo": "DynamoDB",
    "elasticsearch": "Elasticsearch",
    "opensearch": "OpenSearch",
    "rabbitmq": "RabbitMQ",
    "amqp": "RabbitMQ",
    "nats": "NATS",
    "sqs": "SQS",
    "sns": "SNS",
    "memcached": "Memcached",
    "clickhouse": "ClickHouse",
    "bigquery": "BigQuery",
    "snowflake": "Snowflake",
    "redshift": "Redshift",
    "oracle": "Oracle",
    "sqlserver": "SQL Server",
    "mssql": "SQL Server",
    "cosmosdb": "Cosmos DB",
}

HOST_MERGE_TYPES = {
    "PostgreSQL",
    "MySQL",
    "MariaDB",
    "MongoDB",
    "Cassandra",
    "Oracle",
    "SQL Server",
}

INFRA_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b" + re.escape(alias) + r"\b", re.IGNORECASE), canonical)
    for alias, canonical in INFRA_ALIASES.items()
]

IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
JDBC_HOST_RE = re.compile(r"\bjdbc:[a-z0-9+.-]+://([^/\s:@]+)", re.IGNORECASE)
URI_HOST_RE = re.compile(r"\b[a-z][a-z0-9+.-]*://([^/\s:@]+)", re.IGNORECASE)
HOST_KV_RE = re.compile(r"\bhost\s*[:=]\s*([A-Za-z0-9_.-]+)")

HOST_BLACKLIST = {"localhost", "127.0.0.1", "0.0.0.0"}


@dataclass(frozen=True)
class InfraRef:
    """Represents a shared infrastructure reference (kind + optional host)."""
    kind: str
    host: Optional[str]


@dataclass
class InfraNode:
    """Represents a normalized infra node in the merged DSL."""
    kind: str
    host: Optional[str]
    var: str
    label: str
    desc: str


@dataclass
class SystemNode:
    """Represents a repo system in the merged DSL."""
    var: str
    label: str
    desc: str


def slugify(text: str) -> str:
    """Convert a label into a DSL-safe identifier."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    if not slug:
        slug = "id"
    if slug[0].isdigit():
        slug = "_" + slug
    return slug


def unique_var(base: str, used: Set[str]) -> str:
    """Generate a unique variable name from a base token."""
    var = base
    counter = 2
    while var in used:
        var = f"{base}_{counter}"
        counter += 1
    used.add(var)
    return var


def detect_infra_kinds(text: str) -> Set[str]:
    """Detect infrastructure kinds from freeform text."""
    found: Set[str] = set()
    if not text:
        return found
    for rx, canonical in INFRA_PATTERNS:
        if rx.search(text):
            found.add(canonical)
    return found


def clean_host(host: str) -> Optional[str]:
    """Normalize hosts and drop local or template-like values."""
    host = host.strip().strip('"').strip("'")
    if not host:
        return None
    if host in HOST_BLACKLIST:
        return None
    if "$" in host or "{" in host or "}" in host:
        return None
    return host


def extract_host(text: str) -> Optional[str]:
    """Extract a host from common URL/host patterns in text."""
    if not text:
        return None
    for rx in (IP_RE, JDBC_HOST_RE, URI_HOST_RE, HOST_KV_RE):
        m = rx.search(text)
        if m:
            host = clean_host(m.group(1))
            if host:
                return host
    return None


def collect_texts(profile: dict, dsl_text: str) -> List[str]:
    """Collect strings from profile sections and generated DSL."""
    texts: List[str] = []
    if dsl_text:
        texts.append(dsl_text)

    def _collect(value: object) -> None:
        """Recursively collect strings from nested structures."""
        if value is None:
            return
        if isinstance(value, str):
            if value.strip():
                texts.append(value)
            return
        if isinstance(value, list):
            for item in value:
                _collect(item)
            return
        if isinstance(value, dict):
            for item in value.values():
                _collect(item)

    for key in ("data_stores", "dependencies_outbound", "containers"):
        _collect(profile.get(key))

    return texts


def extract_infra_refs(texts: Iterable[str]) -> Set[InfraRef]:
    """Extract infra references with optional host hints."""
    # Only attach host info when a single infra kind is present to avoid mismatches.
    refs: Set[InfraRef] = set()
    for text in texts:
        kinds = detect_infra_kinds(text)
        if not kinds:
            continue
        host = extract_host(text) if len(kinds) == 1 else None
        for kind in kinds:
            host_for_kind = host if kind in HOST_MERGE_TYPES else None
            refs.add(InfraRef(kind=kind, host=host_for_kind))
    return refs


def load_json(path: Path) -> Optional[dict]:
    """Read a JSON file into a dict if possible."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def system_desc(profile: dict) -> str:
    """Build a short system description from profile metadata."""
    primary = profile.get("primary_language")
    if primary and isinstance(primary, str) and primary.lower() != "unknown":
        return f"{primary} repository"
    return "Repository"


def build_full_workspace(out_root: Path, out_path: Path) -> int:
    """Aggregate repo outputs into a system landscape DSL."""
    repos_dir = out_root / "repos"
    if not repos_dir.exists():
        return 0

    # Gather systems and inferred shared infra across all repo outputs.
    systems: List[SystemNode] = []
    repo_infra: Dict[str, Set[str]] = {}
    infra_meta: Dict[str, Tuple[str, Optional[str]]] = {}

    used_vars: Set[str] = set()
    used_labels: Set[str] = set()

    for repo_dir in sorted(repos_dir.iterdir(), key=lambda p: p.name.lower()):
        if not repo_dir.is_dir():
            continue

        profile = load_json(repo_dir / "repo-profile.json") or {}
        dsl_text = ""
        dsl_path = out_root / "dsl" / repo_dir.name / f"{repo_dir.name}.dsl"
        if not dsl_path.exists():
            dsl_path = repo_dir / "workspace.dsl"
        if dsl_path.exists():
            dsl_text = dsl_path.read_text(encoding="utf-8")

        label = str(profile.get("repo", {}).get("name") or repo_dir.name)
        if label in used_labels:
            label = f"{label} ({repo_dir.name})"
        used_labels.add(label)

        sys_var = unique_var(f"sys_{slugify(repo_dir.name)}", used_vars)
        systems.append(SystemNode(var=sys_var, label=label, desc=system_desc(profile)))

        texts = collect_texts(profile, dsl_text)
        infra_refs = extract_infra_refs(texts)
        infra_keys: Set[str] = set()
        for ref in infra_refs:
            key = f"{ref.kind}:{ref.host}" if ref.host else ref.kind
            infra_keys.add(key)
            infra_meta.setdefault(key, (ref.kind, ref.host))
        repo_infra[sys_var] = infra_keys

    if not systems:
        return 0

    infra_nodes: Dict[str, InfraNode] = {}
    for key in sorted(infra_meta.keys(), key=lambda k: k.lower()):
        kind, host = infra_meta[key]
        label = f"{kind}@{host}" if host else kind
        desc = f"{kind} at {host}" if host else f"Shared {kind}"
        var = unique_var(f"infra_{slugify(label)}", used_vars)
        infra_nodes[key] = InfraNode(kind=kind, host=host, var=var, label=label, desc=desc)

    lines: List[str] = []
    lines.append("workspace {")
    lines.append("  model {")

    for system in systems:
        lines.append(f'    {system.var} = softwareSystem("{system.label}", "{system.desc}")')

    for node in infra_nodes.values():
        lines.append(f'    {node.var} = dataStore("{node.label}", "{node.desc}")')

    for system in systems:
        for infra_key in sorted(repo_infra.get(system.var, set()), key=lambda k: k.lower()):
            infra_node = infra_nodes.get(infra_key)
            if not infra_node:
                continue
            lines.append(f'    relationship({system.var}, {infra_node.var}, "Uses")')

    lines.append("  }")
    lines.append("")
    lines.append("  views {")
    lines.append('    systemLandscapeView("AllSystems", "All repositories and shared infrastructure.") {')
    lines.append("      include *")
    lines.append("      autoLayout()")
    lines.append("    }")
    lines.append("  }")
    lines.append("}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(systems)


def main() -> int:
    """CLI entrypoint for building the merged DSL."""
    ap = argparse.ArgumentParser(description="Build a merged Structurizr DSL from repo outputs")
    ap.add_argument("--out", default="architecture-out", help="Output directory")
    ap.add_argument("--output", default=None, help="Path for merged workspace DSL")
    args = ap.parse_args()

    out_root = Path(args.out).resolve()
    out_path = Path(args.output).resolve() if args.output else (out_root / "dsl" / "workspace_full.dsl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = build_full_workspace(out_root, out_path)
    if count == 0:
        print(f'[WARN] No repositories found under "{out_root}"')
        return 2

    print(f"[OK] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
