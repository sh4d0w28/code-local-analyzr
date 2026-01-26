"""Normalize repo profile JSON to keep schema stable and outputs clean."""

from __future__ import annotations

import re
from typing import Iterable, List, Optional


_HTTP_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}

_LIB_IMPORT_RE = re.compile(
    r"^(?:[a-z][a-z0-9+.-]*\.)+[a-z]{2,}(?:/.*)?$",
    re.IGNORECASE,
)
_PKG_PREFIX_RE = re.compile(r"^(com|org|io|net)\.[A-Za-z0-9_.-]+$")
_MAVEN_COORD_RE = re.compile(r"^[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+(:[A-Za-z0-9_.-]+)?$")

_MANIFEST_RE = re.compile(
    r"(pom\.xml|build\.gradle|build\.gradle\.kts|go\.mod|go\.sum|package\.json|"
    r"package-lock\.json|yarn\.lock|pnpm-lock\.yaml|requirements\.txt|pipfile|poetry\.lock)",
    re.IGNORECASE,
)

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

_ENTRYPOINT_KEEP_RE = re.compile(
    r"(\\|/)(cmd|bin)(\\|/)|(^|/)(main|app|server|index)\.(go|py|js|ts|java|kt|rb|php)$|"
    r"Application\.java$|App\.java$|Main\.java$|Dockerfile$|docker-compose.*$|Makefile$|Procfile$",
    re.IGNORECASE,
)

_ENTRYPOINT_SECONDARY_RE = re.compile(
    r"(Application|Main|Bootstrap|Server|Worker)\b|"
    r"(\\|/)(cmd|bin)(\\|/)|"
    r"(^|/)(main|app|server|index)\.",
    re.IGNORECASE,
)

_ENTRYPOINT_COMMAND_RE = re.compile(r"^\s*(java|node|python|go|dotnet)\b", re.IGNORECASE)

_ENTRYPOINT_DROP_RE = re.compile(
    r"(\\|/)(test|tests|spec)(\\|/)|Test\.(java|kt)$|_test\.go$|Controller\.java$|META-INF/MANIFEST\.MF$",
    re.IGNORECASE,
)

_EXTERNAL_HINT_RE = re.compile(
    r"\b(service|system|api|queue|broker|topic|gateway|proxy|db|database|redis|kafka|sqs|sns|s3|"
    r"elasticsearch|opensearch|mongo|mysql|postgres|oracle|sqlserver)\b",
    re.IGNORECASE,
)
_HOST_LIKE_RE = re.compile(
    r"^[A-Za-z0-9_.-]+\.(com|net|org|io|co|dev|internal|local|svc|cluster|cloud|corp|prod|staging|test)$",
    re.IGNORECASE,
)
_API_DETAILS_SIGNAL_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE)\b\s*/|/api/|/v\d+/|\\bgrpc\\b|openapi|swagger|asyncapi|\\.proto",
    re.IGNORECASE,
)

_OPEN_QUESTION_FIELD_RE = re.compile(r"question['\"]?\s*:\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_OPEN_QUESTION_DROP_RE = re.compile(
    r"provided text does not contain|schema does not include|no specific repository",
    re.IGNORECASE,
)


def _is_library_target(target: str) -> bool:
    target = target.strip()
    if not target:
        return True
    if "//" in target:
        return False
    if ":" in target and _HOST_LIKE_RE.search(target.split(":", 1)[0]):
        return False
    if _HOST_LIKE_RE.search(target):
        return False
    if "/" in target:
        return True
    if _LIB_IMPORT_RE.match(target):
        return True
    if _PKG_PREFIX_RE.match(target):
        return True
    if _MAVEN_COORD_RE.match(target):
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


def _normalize_list(values: Iterable[object]) -> List[str]:
    out: List[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def _filter_entrypoints(entrypoints: List[str]) -> List[str]:
    if not entrypoints:
        return []
    kept: List[str] = []
    for ep in entrypoints:
        if _ENTRYPOINT_DROP_RE.search(ep):
            continue
        if _ENTRYPOINT_KEEP_RE.search(ep):
            kept.append(ep)
    if kept:
        return kept
    secondary = [
        ep for ep in entrypoints if _ENTRYPOINT_SECONDARY_RE.search(ep) or _ENTRYPOINT_COMMAND_RE.search(ep)
    ]
    if secondary:
        return secondary
    # Fallback: drop only obvious test/controller/manifest noise.
    return [ep for ep in entrypoints if not _ENTRYPOINT_DROP_RE.search(ep)]


def _normalize_entrypoints(raw_entrypoints: List[object]) -> List[str]:
    if not raw_entrypoints:
        return []
    entries: List[str] = []
    for ep in raw_entrypoints:
        if isinstance(ep, dict):
            ep_type = str(ep.get("type") or "").strip()
            details = str(ep.get("details") or ep.get("path") or ep.get("entrypoint") or ep.get("file") or "").strip()
            if details:
                entries.append(f"{ep_type}: {details}".strip(": ").strip() if ep_type else details)
                continue
            reason = str(ep.get("reason") or "").strip()
            if reason:
                entries.append(f"{ep_type}: {reason}".strip(": ").strip() if ep_type else reason)
                continue
        elif ep is not None:
            s = str(ep).strip()
            if s:
                entries.append(s)
    return _filter_entrypoints(entries)


def _routes_to_details(routes: List[dict], *, limit: int = 20) -> str:
    items: List[str] = []
    seen = set()
    for r in routes:
        method = str(r.get("method") or "").upper()
        path = str(r.get("path") or "").strip()
        if not method or not path:
            continue
        if method not in _HTTP_METHODS:
            continue
        key = f"{method} {path}"
        if key in seen:
            continue
        seen.add(key)
        items.append(key)
        if len(items) >= limit:
            break
    return ", ".join(items)


def _normalize_path(path: str) -> str:
    path = path.strip()
    if not path:
        return ""
    if not path.startswith("/"):
        return "/" + path
    return path


def _base_path(path: str) -> str:
    path = _normalize_path(path)
    if not path or path == "/":
        return ""
    parts = [p for p in path.strip("/").split("/") if p]
    if not parts:
        return ""
    base_len = 1
    if parts[0] in {"api", "internal", "public"}:
        if len(parts) >= 3 and re.match(r"v\d+", parts[2], re.IGNORECASE):
            base_len = 3
        elif len(parts) >= 2:
            base_len = 2
        if len(parts) >= 3 and re.match(r"v\d+", parts[1], re.IGNORECASE):
            base_len = 3
    elif len(parts) >= 2 and re.match(r"v\d+", parts[1], re.IGNORECASE):
        base_len = 2
    return "/" + "/".join(parts[:base_len])


def _routes_to_summary(routes: List[dict], *, limit: int = 6) -> str:
    counts: dict[str, int] = {}
    for r in routes:
        method = str(r.get("method") or "").upper()
        if method and method not in _HTTP_METHODS and method != "GRPC":
            continue
        path = str(r.get("path") or "").strip()
        base = _base_path(path)
        if not base:
            continue
        counts[base] = counts.get(base, 0) + 1
    if not counts:
        return ""
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    parts = [f"{base} ({count})" for base, count in items[:limit]]
    if len(items) > limit:
        parts.append(f"+{len(items) - limit} more")
    return ", ".join(parts)


def _grpc_services_summary(routes: List[dict], *, limit: int = 6) -> str:
    services: dict[str, int] = {}
    for r in routes:
        method = str(r.get("method") or "").upper()
        if method != "GRPC":
            continue
        handler = str(r.get("handler") or "").strip()
        if not handler:
            continue
        service = handler.split("/", 1)[0]
        if not service:
            continue
        services[service] = services.get(service, 0) + 1
    if not services:
        return ""
    items = sorted(services.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    parts = [f"{svc} ({count})" for svc, count in items[:limit]]
    if len(items) > limit:
        parts.append(f"+{len(items) - limit} more")
    return ", ".join(parts)


def _details_look_like_paths(details: str) -> bool:
    if not details:
        return False
    return any(tok in details for tok in (".java", ".go", "/src/", "src/", ".proto", ".ts", ".js"))


def _api_details_actionable(details: str) -> bool:
    if not details:
        return False
    return bool(_API_DETAILS_SIGNAL_RE.search(details))


def _clean_open_question(q: str) -> Optional[str]:
    if not q:
        return None
    if _OPEN_QUESTION_DROP_RE.search(q):
        return None
    m = _OPEN_QUESTION_FIELD_RE.search(q)
    if m:
        return m.group(1).strip()
    return q.strip()


def normalize_profile(
    profile: dict,
    *,
    repo_name: str,
    repo_path: str,
    routes_entries: Optional[List[dict]] = None,
) -> dict:
    """Normalize profile fields to reduce noisy or schema-breaking output."""
    if not isinstance(profile, dict):
        return profile

    repo = profile.get("repo")
    if not isinstance(repo, dict):
        repo = {}
        profile["repo"] = repo
    repo["name"] = repo_name
    repo["path"] = repo_path

    build = profile.get("build_and_runtime") or {}
    if not isinstance(build, dict):
        build = {}
    build_tools = _normalize_list(build.get("build_tools", []))
    runtime = _normalize_list(build.get("runtime", []))
    build["build_tools"] = build_tools
    build["runtime"] = runtime
    profile["build_and_runtime"] = build

    entrypoints_raw = profile.get("entrypoints", [])
    entrypoints = _normalize_entrypoints(entrypoints_raw if isinstance(entrypoints_raw, list) else [])
    profile["entrypoints"] = entrypoints

    # Normalize APIs
    apis_raw = profile.get("apis", [])
    apis_out: List[dict] = []
    if isinstance(apis_raw, list):
        for a in apis_raw:
            if isinstance(a, dict):
                api_type = str(a.get("type") or "unknown")
                details = str(a.get("details") or "")
                apis_out.append({"type": api_type, "details": details})
            elif isinstance(a, str) and a.strip():
                apis_out.append({"type": "unknown", "details": a.strip()})
    if routes_entries:
        http_details = _routes_to_details(routes_entries)
        http_summary = _routes_to_summary(routes_entries)
        if http_summary:
            http_details = f"Routes grouped by base path: {http_summary}. Examples: {http_details}" if http_details else f"Routes grouped by base path: {http_summary}."
        if http_details:
            found_http = False
            for api in apis_out:
                if "http" in api.get("type", "").lower():
                    api["details"] = http_details
                    found_http = True
            if not found_http:
                apis_out.append({"type": "http", "details": http_details})
        grpc_summary = _grpc_services_summary(routes_entries)
        if grpc_summary:
            found_grpc = False
            for api in apis_out:
                if "grpc" in api.get("type", "").lower():
                    api["details"] = f"Services: {grpc_summary}"
                    found_grpc = True
            if not found_grpc:
                apis_out.append({"type": "grpc", "details": f"Services: {grpc_summary}"})
    for api in apis_out:
        details = api.get("details", "")
        if _details_look_like_paths(details) and not routes_entries:
            api["details"] = "unknown (routes not detected)"
        elif not routes_entries and not _api_details_actionable(details):
            api["details"] = "unknown (routes not detected)"
    apis_out = [
        a
        for a in apis_out
        if not (
            str(a.get("type", "")).strip().lower() == "unknown"
            and not str(a.get("details", "")).strip()
        )
    ]
    profile["apis"] = apis_out

    # Normalize data stores
    stores_raw = profile.get("data_stores", [])
    stores_out: List[dict] = []
    if isinstance(stores_raw, list):
        for s in stores_raw:
            if isinstance(s, dict):
                stores_out.append(
                    {
                        "type": str(s.get("type") or "unknown"),
                        "details": str(s.get("details") or ""),
                    }
                )
            elif isinstance(s, str) and s.strip():
                stores_out.append({"type": s.strip(), "details": ""})
    stores_out = [
        s
        for s in stores_out
        if not (str(s.get("type", "")).strip().lower() == "unknown" and not str(s.get("details", "")).strip())
    ]
    profile["data_stores"] = stores_out

    # Normalize containers
    containers_raw = profile.get("containers", [])
    containers_out: List[dict] = []
    if isinstance(containers_raw, list):
        for c in containers_raw:
            if isinstance(c, dict):
                containers_out.append(
                    {
                        "name": str(c.get("name") or "unknown"),
                        "type": str(c.get("type") or "unknown"),
                        "tech": str(c.get("tech") or "unknown"),
                        "responsibility": str(c.get("responsibility") or "unknown"),
                        "exposes": _normalize_list(c.get("exposes", [])),
                        "depends_on": _normalize_list(c.get("depends_on", [])),
                    }
                )
            elif isinstance(c, str) and c.strip():
                containers_out.append(
                    {
                        "name": c.strip(),
                        "type": "unknown",
                        "tech": "unknown",
                        "responsibility": "unknown",
                        "exposes": [],
                        "depends_on": [],
                    }
                )
    containers_out = [
        c
        for c in containers_out
        if not (
            str(c.get("name", "")).strip().lower() == "unknown"
            and str(c.get("type", "")).strip().lower() == "unknown"
            and str(c.get("tech", "")).strip().lower() == "unknown"
            and str(c.get("responsibility", "")).strip().lower() == "unknown"
            and not c.get("exposes")
            and not c.get("depends_on")
        )
    ]
    profile["containers"] = containers_out

    # Normalize outbound dependencies
    deps_raw = profile.get("dependencies_outbound", [])
    deps_out: List[dict] = []
    seen = set()
    skip_targets = {t.lower() for t in build_tools + runtime}

    if isinstance(deps_raw, list):
        for d in deps_raw:
            if isinstance(d, dict):
                target = str(d.get("target") or d.get("name") or "").strip()
                reason = str(d.get("reason") or "").strip()
            elif isinstance(d, str):
                target = d.strip()
                reason = ""
            else:
                continue
            if not target:
                continue
            target_l = target.lower()
            if target_l in skip_targets:
                continue
            if _is_library_target(target):
                continue
            if _is_docker_image_target(target):
                continue
            if _MANIFEST_RE.search(reason):
                continue
            if target_l.islower() and "-" in target_l and not _EXTERNAL_HINT_RE.search(target_l):
                continue
            if target_l in seen:
                continue
            seen.add(target_l)
            deps_out.append({"target": target, "reason": reason})
    profile["dependencies_outbound"] = deps_out

    # Normalize open questions
    oq_raw = _normalize_list(profile.get("open_questions", []))
    oq_clean: List[str] = []
    for item in oq_raw:
        cleaned = _clean_open_question(item)
        if cleaned:
            oq_clean.append(cleaned)
    profile["open_questions"] = list(dict.fromkeys(oq_clean))

    return profile
