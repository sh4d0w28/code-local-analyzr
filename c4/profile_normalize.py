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

_ENTRYPOINT_DROP_RE = re.compile(
    r"(\\|/)(test|tests|spec)(\\|/)|Test\.(java|kt)$|_test\.go$|Controller\.java$|META-INF/MANIFEST\.MF$",
    re.IGNORECASE,
)

_EXTERNAL_HINT_RE = re.compile(
    r"\b(service|system|api|queue|broker|topic|gateway|proxy|db|database|redis|kafka|sqs|sns|s3|"
    r"elasticsearch|opensearch|mongo|mysql|postgres|oracle|sqlserver)\b",
    re.IGNORECASE,
)
_API_DETAILS_SIGNAL_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE)\b\s*/|/api/|/v\d+/|\\bgrpc\\b|openapi|swagger|asyncapi|\\.proto",
    re.IGNORECASE,
)


def _is_library_target(target: str) -> bool:
    target = target.strip()
    if not target:
        return True
    if "//" in target:
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


def _details_look_like_paths(details: str) -> bool:
    if not details:
        return False
    return any(tok in details for tok in (".java", ".go", "/src/", "src/", ".proto", ".ts", ".js"))


def _api_details_actionable(details: str) -> bool:
    if not details:
        return False
    return bool(_API_DETAILS_SIGNAL_RE.search(details))


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
        if http_details:
            found_http = False
            for api in apis_out:
                if "http" in api.get("type", "").lower():
                    api["details"] = http_details
                    found_http = True
            if not found_http:
                apis_out.append({"type": "http", "details": http_details})
    for api in apis_out:
        details = api.get("details", "")
        if _details_look_like_paths(details) and not routes_entries:
            api["details"] = "unknown (routes not detected)"
        elif not routes_entries and not _api_details_actionable(details):
            api["details"] = "unknown (routes not detected)"
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
    oq = _normalize_list(profile.get("open_questions", []))
    profile["open_questions"] = list(dict.fromkeys(oq))

    return profile
