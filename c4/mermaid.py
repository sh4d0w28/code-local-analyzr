"""Generate and sanitize Mermaid C4 diagrams from repo profiles."""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Tuple

from .ollama_client import ollama_chat
from .prompts import MERMAID_C4_SYSTEM

MERMAID_BLOCK_START_RE = re.compile(r"^\s*```mermaid\s*$")
MERMAID_BLOCK_END_RE = re.compile(r"^\s*```\s*$")
FLOWCHART_LABEL_RE = re.compile(r"\|([^|]+)\|")
C4_TITLE_RE = re.compile(r"^\s*title\s+")
BOUNDARY_ID_RE = re.compile(r"^\s*(System_Boundary|Container_Boundary|DeploymentNode)\(\s*([A-Za-z0-9_]+)\s*,")
BOUNDARY_LABEL_RE = re.compile(r"^\s*Container_Boundary\(\s*[A-Za-z0-9_]+\s*,\s*\"([^\"]+)\"")
SYSTEM_LABEL_RE = re.compile(r"^\s*System_Boundary\(\s*[A-Za-z0-9_]+\s*,\s*\"([^\"]+)\"")
SYSTEM_EXT_ID_RE = re.compile(r"^\s*System_Ext\(\s*([A-Za-z0-9_]+)\s*,")
CONTAINER_LINE_RE = re.compile(
    r"^\s*Container(?:Db|Queue|_Ext|Db_Ext|Queue_Ext)?\(\s*([A-Za-z0-9_]+)\s*,\s*\"([^\"]+)\""
)


def _sanitize_label_text(text: str) -> str:
    """Rewrite route params to avoid Mermaid label parsing issues."""
    return re.sub(r"\{([^}]+)\}", r":\1", text)


def _sanitize_flowchart_line(line: str) -> str:
    """Fix flowchart edge labels in a single line."""
    def _repl(match: re.Match) -> str:
        """Normalize route labels inside edge pipes."""
        label = _sanitize_label_text(match.group(1))
        return f"|{label}|"

    return FLOWCHART_LABEL_RE.sub(_repl, line)


def _sanitize_container_boundary_line(line: str, boundary_name: str) -> str:
    """Convert Container(...) { to the requested boundary type."""
    m = re.match(r"^(\s*)Container\(([^,)\s]+)\)\s*\{", line)
    if not m:
        return line
    indent, ident = m.groups()
    return f'{indent}{boundary_name}({ident}, "{ident}") {{'


def _slugify_id(text: str) -> str:
    """Make a Mermaid-safe identifier."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "system"


def _escape_label(text: str) -> str:
    """Escape Mermaid string labels."""
    return text.replace('"', '\\"')


def _split_rel_args(text: str) -> List[str]:
    """Split Rel(...) args without breaking on commas inside quotes."""
    out: List[str] = []
    buf: List[str] = []
    in_quote = False
    esc = False
    for ch in text:
        if esc:
            buf.append(ch)
            esc = False
            continue
        if ch == "\\":
            buf.append(ch)
            esc = True
            continue
        if ch == '"':
            buf.append(ch)
            in_quote = not in_quote
            continue
        if ch == "," and not in_quote:
            out.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf).strip())
    return out


def _rel_endpoints(line: str) -> Tuple[str, str]:
    """Extract the two endpoints from a Rel(...) line."""
    if "Rel(" not in line:
        return "", ""
    start = line.find("(")
    end = line.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return "", ""
    args = _split_rel_args(line[start + 1 : end])
    if len(args) < 2:
        return "", ""
    return args[0], args[1]


def _is_container_line(line: str) -> bool:
    """Check if a line declares a container node."""
    s = line.strip()
    return s.startswith(("Container(", "ContainerDb(", "ContainerQueue(", "Container_Ext(", "ContainerDb_Ext("))


def _is_component_line(line: str) -> bool:
    """Check if a line declares a component node."""
    s = line.strip()
    return s.startswith(("Component(", "ComponentDb(", "ComponentQueue(", "Component_Ext("))


def _parse_container_defs(lines: List[str]) -> List[Tuple[str, str]]:
    """Extract container ids and labels from existing lines."""
    defs: List[Tuple[str, str]] = []
    for line in lines:
        m = CONTAINER_LINE_RE.match(line)
        if not m:
            continue
        defs.append((m.group(1), m.group(2)))
    return defs


def _infer_store_node(store: dict, used_ids: set[str]) -> Tuple[str, str]:
    """Create a ContainerDb/ContainerQueue node for a datastore entry."""
    raw_type = str(store.get("type") or "").strip()
    details = str(store.get("details") or "").strip()
    combined = f"{raw_type} {details}".lower()

    label = ""
    kind = "db"
    if "postgres" in combined:
        label = "PostgreSQL"
    elif "mysql" in combined:
        label = "MySQL"
    elif "mariadb" in combined:
        label = "MariaDB"
    elif "mongodb" in combined or "mongo" in combined:
        label = "MongoDB"
    elif "redis" in combined:
        label = "Redis"
    elif "elastic" in combined:
        label = "Elasticsearch"
    elif "opensearch" in combined:
        label = "OpenSearch"
    elif "clickhouse" in combined:
        label = "ClickHouse"
    elif "cassandra" in combined:
        label = "Cassandra"
    elif "dynamo" in combined:
        label = "DynamoDB"
    elif "spanner" in combined:
        label = "Spanner"
    elif "kafka" in combined:
        label = "Kafka"
        kind = "queue"
    elif "rabbit" in combined or "amqp" in combined:
        label = "RabbitMQ"
        kind = "queue"
    elif "nats" in combined:
        label = "NATS"
        kind = "queue"
    elif "sqs" in combined:
        label = "SQS"
        kind = "queue"
    elif "sns" in combined:
        label = "SNS"
        kind = "queue"
    elif "queue" in combined or "broker" in combined:
        label = "Message Broker"
        kind = "queue"
    elif "cache" in combined:
        label = "Cache"
        kind = "cache"
    elif "database" in combined or "db" in raw_type.lower():
        label = "Database"

    if not label:
        label = raw_type or "Data Store"

    base_id = _slugify_id(label)
    store_id = base_id
    i = 2
    while store_id in used_ids:
        store_id = f"{base_id}_{i}"
        i += 1
    used_ids.add(store_id)

    tech = label
    description = details or raw_type or "Data store"
    label_esc = _escape_label(label)
    tech_esc = _escape_label(tech)
    desc_esc = _escape_label(description)

    if kind == "queue":
        line = f'ContainerQueue({store_id}, "{label_esc}", "{tech_esc}", "{desc_esc}")'
    else:
        line = f'ContainerDb({store_id}, "{label_esc}", "{tech_esc}", "{desc_esc}")'

    return store_id, line


def _normalize_c4_container(
    lines: List[str],
    *,
    system_label: str,
    profile_containers: List[dict],
    data_stores: List[dict],
    outbound_deps: List[dict],
) -> List[str]:
    """Rebuild C4Container view with proper boundaries and external deps."""
    # Adds stores and outbound systems inferred from the profile.
    boundary_ids: set[str] = set()
    external_lines: List[str] = []
    container_lines: List[str] = []
    rel_lines: List[str] = []
    title_lines: List[str] = []

    for line in lines:
        if not line.strip():
            continue
        if line.strip().startswith("C4Container"):
            title_lines.append(line)
            continue
        if C4_TITLE_RE.match(line):
            title_lines.append(line)
            continue

        m = BOUNDARY_ID_RE.match(line)
        if m:
            boundary_ids.add(m.group(2))
            continue

        if _is_container_line(line):
            container_lines.append(line.strip())
            continue

        if line.strip().startswith(("Person(", "System(", "System_Ext(", "Container_Ext(")):
            external_lines.append(line.strip())
            continue

        if line.strip().startswith("Rel("):
            rel_lines.append(line.strip())
            continue

    system_id = _slugify_id(system_label)

    container_defs = _parse_container_defs(container_lines)
    container_ids = {cid for cid, _ in container_defs}
    label_to_id = {label.lower(): cid for cid, label in container_defs}

    for c in profile_containers:
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        label_key = name.lower()
        cid = label_to_id.get(label_key) or _slugify_id(name)
        if cid not in container_ids:
            tech = str(c.get("tech") or "")
            resp = str(c.get("responsibility") or "")
            line = f'Container({cid}, "{_escape_label(name)}", "{_escape_label(tech)}", "{_escape_label(resp)}")'
            container_lines.append(line)
            container_defs.append((cid, name))
            container_ids.add(cid)
            label_to_id[label_key] = cid

    ext_ids = set()
    for line in external_lines:
        m = SYSTEM_EXT_ID_RE.match(line)
        if m:
            ext_ids.add(m.group(1))

    ext_map: dict[str, str] = {}
    for dep in outbound_deps:
        target = str(dep.get("target") or "").strip()
        if not target:
            continue
        ext_id = _slugify_id(target)
        if ext_id in ext_ids:
            ext_map[target.lower()] = ext_id
            continue
        reason = str(dep.get("reason") or "External system").strip()
        external_lines.append(f'System_Ext({ext_id}, "{_escape_label(target)}", "{_escape_label(reason)}")')
        ext_ids.add(ext_id)
        ext_map[target.lower()] = ext_id

    used_ids = set(container_ids) | ext_ids | {system_id}
    store_lines: List[str] = []
    store_ids: List[str] = []
    for store in data_stores:
        if not isinstance(store, dict):
            continue
        store_id, line = _infer_store_node(store, used_ids)
        store_lines.append(line)
        store_ids.append(store_id)

    rel_extra: List[str] = []
    # Keep stable ordering based on container definitions.
    container_list = [cid for cid, _ in container_defs] or list(container_ids)
    if container_list and store_ids:
        for cid in container_list:
            for sid in store_ids:
                rel_extra.append(f'Rel({cid}, {sid}, "uses")')

    if container_list and ext_map:
        depends_map: dict[str, set[str]] = {}
        for c in profile_containers:
            name = str(c.get("name") or "").strip()
            if not name:
                continue
            cid = label_to_id.get(name.lower()) or _slugify_id(name)
            deps = c.get("depends_on") or []
            dep_set = {str(d).strip().lower() for d in deps if isinstance(d, str)}
            depends_map[cid] = dep_set

        if depends_map:
            for cid, deps in depends_map.items():
                for dep_name, ext_id in ext_map.items():
                    if dep_name in deps:
                        rel_extra.append(f'Rel({cid}, {ext_id}, "uses")')
        else:
            for cid in container_list:
                for ext_id in ext_map.values():
                    rel_extra.append(f'Rel({cid}, {ext_id}, "uses")')
    out: List[str] = []
    out.extend(title_lines or ["C4Container"])
    if external_lines:
        out.extend(f"  {l}" for l in external_lines)
    out.append(f'  System_Boundary({system_id}, "{system_label}") {{')
    out.extend(f"    {l}" for l in container_lines)
    out.extend(f"    {l}" for l in store_lines)
    out.append("  }")

    for line in rel_lines:
        a, b = _rel_endpoints(line)
        if a in boundary_ids or b in boundary_ids:
            continue
        out.append(f"  {line}")
    for line in rel_extra:
        out.append(f"  {line}")

    return out


def _normalize_c4_component(lines: List[str], *, container_label: str) -> List[str]:
    """Rebuild C4Component view so components live under one container boundary."""
    boundary_ids: set[str] = set()
    component_lines: List[str] = []
    rel_lines: List[str] = []
    title_lines: List[str] = []

    for line in lines:
        if not line.strip():
            continue
        if line.strip().startswith("C4Component"):
            title_lines.append(line)
            continue
        if C4_TITLE_RE.match(line):
            title_lines.append(line)
            continue

        m = BOUNDARY_ID_RE.match(line)
        if m:
            boundary_ids.add(m.group(2))
            continue

        if _is_component_line(line):
            component_lines.append(line.strip())
            continue

        if line.strip().startswith("Rel("):
            rel_lines.append(line.strip())
            continue

    if not container_label:
        container_label = "Container"
    container_id = _slugify_id(container_label)
    out: List[str] = []
    out.extend(title_lines or ["C4Component"])
    out.append(f'  Container_Boundary({container_id}, "{container_label}") {{')
    out.extend(f"    {l}" for l in component_lines)
    out.append("  }")

    for line in rel_lines:
        a, b = _rel_endpoints(line)
        if a in boundary_ids or b in boundary_ids:
            continue
        out.append(f"  {line}")

    return out


def _sanitize_c4_block(
    lines: List[str],
    *,
    repo_name: str,
    container_name: str,
    profile_containers: List[dict],
    data_stores: List[dict],
    outbound_deps: List[dict],
) -> List[str]:
    """Sanitize a single Mermaid block based on its C4 type."""
    block_type = None
    for l in lines:
        if l.strip():
            block_type = l.strip().split()[0]
            break

    if block_type == "C4Container":
        system_label = repo_name or "System"
        label_match = next((SYSTEM_LABEL_RE.match(l) for l in lines if SYSTEM_LABEL_RE.match(l)), None)
        if label_match:
            system_label = label_match.group(1) or system_label
        return _normalize_c4_container(
            lines,
            system_label=system_label,
            profile_containers=profile_containers,
            data_stores=data_stores,
            outbound_deps=outbound_deps,
        )

    if block_type == "C4Component":
        label_match = next((BOUNDARY_LABEL_RE.match(l) for l in lines if BOUNDARY_LABEL_RE.match(l)), None)
        comp_label = label_match.group(1) if label_match else (container_name or repo_name or "Container")
        return _normalize_c4_component(lines, container_label=comp_label)

    if block_type in ("flowchart", "graph"):
        return [_sanitize_flowchart_line(line) for line in lines]

    return lines


def sanitize_mermaid_markdown(
    text: str,
    *,
    repo_name: str,
    container_name: str,
    profile_containers: List[dict],
    data_stores: List[dict],
    outbound_deps: List[dict],
) -> str:
    """Process a Mermaid markdown document and fix invalid C4 constructs."""
    lines = text.splitlines()
    out: List[str] = []
    in_block = False
    block_lines: List[str] = []

    for line in lines:
        if not in_block:
            out.append(line)
            if MERMAID_BLOCK_START_RE.match(line):
                in_block = True
                block_lines = []
            continue

        if MERMAID_BLOCK_END_RE.match(line):
            out.extend(
                _sanitize_c4_block(
                    block_lines,
                    repo_name=repo_name,
                    container_name=container_name,
                    profile_containers=profile_containers,
                    data_stores=data_stores,
                    outbound_deps=outbound_deps,
                )
            )
            out.append(line)
            in_block = False
            block_lines = []
            continue

        block_lines.append(line)

    if in_block:
        out.extend(
            _sanitize_c4_block(
                block_lines,
                repo_name=repo_name,
                container_name=container_name,
                profile_containers=profile_containers,
                data_stores=data_stores,
                outbound_deps=outbound_deps,
            )
        )

    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def generate_mermaid_c4(
    profile: dict,
    *,
    routes_text: Optional[str],
    ollama_base: str,
    model: str,
    timeout_s: int,
    num_predict: int,
    num_ctx: int,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    """Generate Mermaid C4 markdown and normalize it for valid rendering."""
    if not MERMAID_C4_SYSTEM:
        raise RuntimeError('Missing "mermaid_c4_system" prompt in config')
    payload = "REPO_PROFILE_JSON:\n" + json.dumps(profile, indent=2, ensure_ascii=False)
    if routes_text:
        payload += "\n\n" + routes_text
    raw = ollama_chat(
        ollama_base,
        model,
        MERMAID_C4_SYSTEM,
        payload,
        temperature=0.0,
        timeout_s=timeout_s,
        num_predict=num_predict,
        num_ctx=num_ctx,
        log=log,
        label="mermaid",
    )
    repo_name = profile.get("repo", {}).get("name", "System")
    containers = profile.get("containers") or []
    container_name = ""
    if isinstance(containers, list) and containers:
        first = containers[0]
        if isinstance(first, dict):
            container_name = str(first.get("name") or "")
    if not container_name:
        container_name = f"{repo_name} App"
    data_stores = profile.get("data_stores") or []
    outbound_deps = profile.get("dependencies_outbound") or []
    profile_containers = containers if isinstance(containers, list) else []
    return sanitize_mermaid_markdown(
        raw,
        repo_name=repo_name,
        container_name=container_name,
        profile_containers=profile_containers,
        data_stores=data_stores if isinstance(data_stores, list) else [],
        outbound_deps=outbound_deps if isinstance(outbound_deps, list) else [],
    )
