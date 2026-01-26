"""Programmatic Structurizr DSL renderer from repo profiles."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple


_ALIAS_MAP = {
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


_LIB_IMPORT_RE = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*\\.)+[a-z]{2,}(?:/.*)?$",
    re.IGNORECASE,
)
_PKG_PREFIX_RE = re.compile(r"^(com|org|io|net)\\.[A-Za-z0-9_.-]+$")
_DOCKER_IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?::[A-Za-z0-9_.-]+)?$")
_DOCKER_BASE_NAMES = {
    "alpine",
    "busybox",
    "scratch",
    "debian",
    "ubuntu",
    "centos",
    "rockylinux",
    "amazonlinux",
    "distroless",
    "golang",
    "python",
    "node",
    "openjdk",
    "temurin",
    "corretto",
    "amazoncorretto",
    "maven",
    "gradle",
}


def _escape(text: str) -> str:
    """Escape text for DSL string literals."""
    text = str(text)
    text = text.replace("\\", "\\\\").replace("\"", "\\\"")
    text = " ".join(text.split())
    return text


def slugify(text: str) -> str:
    """Convert arbitrary text into a DSL-safe identifier."""
    text = str(text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not slug:
        slug = "id"
    if slug[0].isdigit():
        slug = f"_{slug}"
    return slug


def _is_library_target(target: str) -> bool:
    """Heuristic: treat package/module import paths as libraries, not systems."""
    target = target.strip()
    if not target:
        return True
    if "://" in target:
        return False
    if "/" in target:
        return True
    if _LIB_IMPORT_RE.match(target):
        return True
    if _PKG_PREFIX_RE.match(target):
        return True
    if _is_docker_image_target(target):
        return True
    return False


def _is_docker_image_target(target: str) -> bool:
    """Treat base Docker images/runtime tags as non-external dependencies."""
    target = target.strip().lower()
    if not target:
        return False
    if "/" in target:
        return False
    if not _DOCKER_IMAGE_RE.match(target):
        return False
    base = target.split(":", 1)[0]
    if base in _DOCKER_BASE_NAMES:
        return True
    if "distroless" in base or "debian" in base or "ubuntu" in base:
        return True
    return False


def unique_id(base: str, used: Set[str]) -> str:
    """Return a unique identifier based on a base token."""
    candidate = base
    counter = 2
    while candidate in used:
        candidate = f"{base}_{counter}"
        counter += 1
    used.add(candidate)
    return candidate


_GENERIC_STORE_TYPES = {
    "database",
    "db",
    "cache",
    "queue",
    "broker",
    "datastore",
    "storage",
    "search",
    "search_engine",
}


def _title_label(text: str) -> str:
    """Humanize labels from underscored tokens."""
    text = str(text).replace("_", " ").replace("-", " ").strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split())


def _infer_store_label(store_type: str, details: str) -> Tuple[str, Optional[str]]:
    """Infer a data store label and technology from type/details."""
    haystack = f"{store_type} {details}".lower()
    for alias, canonical in _ALIAS_MAP.items():
        if alias in haystack:
            return canonical, canonical
    raw = store_type.strip() or "datastore"
    label = _title_label(raw)
    tech = None if raw.lower() in _GENERIC_STORE_TYPES else label
    return label, tech


def _api_types(profile: dict) -> Set[str]:
    types: Set[str] = set()
    for api in profile.get("apis", []) or []:
        raw = str(api.get("type", "")).lower()
        if "http" in raw:
            types.add("http")
        if "grpc" in raw:
            types.add("grpc")
    if not types:
        for c in profile.get("containers", []) or []:
            for expose in c.get("exposes") or []:
                text = str(expose).lower()
                if "http" in text:
                    types.add("http")
                if "grpc" in text:
                    types.add("grpc")
    return types


def _container_candidates(containers: List[dict]) -> List[dict]:
    """Pick containers that likely expose APIs."""
    candidates: List[dict] = []
    for c in containers:
        exposes = c.get("exposes") or []
        if exposes:
            candidates.append(c)
            continue
        if str(c.get("type", "")).lower() == "service":
            candidates.append(c)
    return candidates or containers


def render_structurizr(profile: dict) -> str:
    """Render Structurizr DSL from a repo profile dict (no LLM)."""
    repo = profile.get("repo", {}) if isinstance(profile.get("repo"), dict) else {}
    repo_name = str(repo.get("name") or "system")
    system_id = slugify(repo_name)
    system_label = _escape(str(repo.get("name") or system_id))

    used_ids: Set[str] = set()
    used_ids.add(system_id)

    containers_raw = list(profile.get("containers") or [])
    containers_in: List[dict] = []
    for item in containers_raw:
        if isinstance(item, dict):
            containers_in.append(item)
        elif isinstance(item, str) and item.strip():
            containers_in.append({"name": item.strip()})
    container_defs: List[dict] = []
    container_ids_by_slug: Dict[str, str] = {}

    for c in containers_in:
        name = str(c.get("name") or "container")
        base = f"{system_id}_{slugify(name)}"
        cid = unique_id(base, used_ids)
        label = _escape(name)
        tech = _escape(str(c.get("tech") or "").strip())
        if tech.lower() == "unknown":
            tech = ""
        desc = _escape(str(c.get("responsibility") or "").strip())
        if desc.lower() == "unknown":
            desc = ""
        container_defs.append(
            {
                "id": cid,
                "label": label,
                "tech": tech,
                "desc": desc,
                "depends_on": list(c.get("depends_on") or []),
                "exposes": list(c.get("exposes") or []),
                "type": str(c.get("type") or ""),
                "name_slug": slugify(name),
            }
        )
        container_ids_by_slug[slugify(name)] = cid
        container_ids_by_slug[slugify(cid)] = cid

    store_defs: List[dict] = []
    store_ids_by_slug: Dict[str, str] = {}
    for store in profile.get("data_stores", []) or []:
        if isinstance(store, dict):
            store_type = str(store.get("type") or "").strip()
            details = _escape(str(store.get("details") or "").strip())
        elif isinstance(store, str):
            store_type = store.strip()
            details = ""
        else:
            continue
        label, tech = _infer_store_label(store_type, details)
        base = f"{system_id}_{slugify(label)}"
        if base in used_ids:
            continue
        sid = unique_id(base, used_ids)
        store_defs.append(
            {
                "id": sid,
                "label": _escape(label),
                "tech": _escape(tech or ""),
                "desc": details,
            }
        )
        store_ids_by_slug[slugify(label)] = sid
        store_ids_by_slug[slugify(sid)] = sid
        if store_type:
            store_ids_by_slug[slugify(store_type)] = sid

    build = profile.get("build_and_runtime", {}) or {}
    skip_targets: Set[str] = set()
    for key in ("build_tools", "runtime"):
        for item in build.get(key, []) or []:
            raw = str(item).strip()
            if not raw:
                continue
            skip_targets.add(raw.lower())
            skip_targets.add(slugify(raw))

    external_defs: List[dict] = []
    external_ids_by_slug: Dict[str, str] = {}
    for dep in profile.get("dependencies_outbound", []) or []:
        if isinstance(dep, dict):
            target = str(dep.get("target") or "").strip()
            reason = _escape(str(dep.get("reason") or "").strip())
        elif isinstance(dep, str):
            target = dep.strip()
            reason = ""
        else:
            continue
        if not target:
            continue
        if target.lower() in skip_targets or slugify(target) in skip_targets:
            continue
        if _is_library_target(target):
            continue
        target_slug = slugify(target)
        if target_slug in container_ids_by_slug or target_slug in store_ids_by_slug:
            continue
        base = target_slug
        if base == system_id or base in used_ids:
            base = f"ext_{base}"
        eid = unique_id(base, used_ids)
        external_defs.append(
            {
                "id": eid,
                "label": _escape(target),
                "desc": reason,
            }
        )
        external_ids_by_slug[slugify(target)] = eid
        external_ids_by_slug[slugify(eid)] = eid

    api_types = _api_types(profile)
    api_container_defs = _container_candidates(container_defs)
    api_container_ids = [c["id"] for c in api_container_defs if c.get("id")]

    lines: List[str] = []
    lines.append(f'{system_id} = softwareSystem "{system_label}" {{')
    for c in container_defs + store_defs:
        lines.append(f'  {c["id"]} = container "{c["label"]}" {{')
        if c.get("tech"):
            lines.append(f'    technology "{c["tech"]}"')
        if c.get("desc"):
            lines.append(f'    description "{c["desc"]}"')
        lines.append("  }")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    lines.append("}")
    lines.append("")

    if api_types:
        lines.append('CustomerPerson = person "Customer"')
        lines.append("")

    for ext in external_defs:
        lines.append(f'{ext["id"]} = softwareSystem "{ext["label"]}" {{')
        if ext.get("desc"):
            lines.append(f'  description "{ext["desc"]}"')
        lines.append("}")
        lines.append("")

    def add_rel(src: str, dst: str, label: str) -> None:
        if not src or not dst:
            return
        rel = f'{src} -> {dst} "{label}"' if label else f"{src} -> {dst}"
        rels.add(rel)

    rels: Set[str] = set()

    if api_types:
        targets = api_container_ids or [system_id]
        for tgt in targets:
            if "http" in api_types:
                add_rel("CustomerPerson", tgt, "Uses HTTP API")
            if "grpc" in api_types:
                add_rel("CustomerPerson", tgt, "Uses gRPC API")

    # container depends_on
    id_lookup: Dict[str, str] = {}
    id_lookup.update(container_ids_by_slug)
    id_lookup.update(store_ids_by_slug)
    id_lookup.update(external_ids_by_slug)

    for c in container_defs:
        src = c["id"]
        for dep in c.get("depends_on", []) or []:
            key = slugify(dep)
            dst = id_lookup.get(key)
            if dst:
                add_rel(src, dst, "Depends on")

    # outbound deps
    primary_src = api_container_ids[0] if api_container_ids else (container_defs[0]["id"] if container_defs else system_id)
    for ext in external_defs:
        label = ext.get("desc") or "Depends on"
        add_rel(primary_src, ext["id"], label)

    for rel in sorted(rels):
        lines.append(rel)

    return "\n".join(lines).strip() + "\n"
