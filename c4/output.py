"""Render human-readable architecture summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional


def _pretty_list(items: List[str], *, fallback: str = "Not detected") -> str:
    cleaned = [str(x).strip() for x in items if x is not None and str(x).strip()]
    cleaned = [x for x in cleaned if x.lower() != "unknown"]
    return ", ".join(cleaned) if cleaned else fallback


def _format_bytes(num: int) -> str:
    if num < 0:
        num = 0
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.0f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_arch_md(out_path: Path, profile: dict) -> None:
    """Write a Markdown summary from a repo profile."""
    lines: List[str] = []
    repo = profile.get("repo")
    if isinstance(repo, dict):
        repo_name = repo.get("name", "Repository")
    else:
        repo_name = "Repository"
    lines.append(f"# {repo_name} Architecture\n")

    lines.append("## Overview\n")
    primary_lang = str(profile.get("primary_language") or "").strip()
    if not primary_lang or primary_lang.lower() == "unknown":
        primary_lang = "Not detected"
    lines.append(f"- **Primary language:** {primary_lang}\n")
    br = profile.get("build_and_runtime", {}) or {}
    build_tools = [str(x) for x in (br.get("build_tools", []) or []) if x is not None]
    runtime = [str(x) for x in (br.get("runtime", []) or []) if x is not None]
    lines.append(f"- **Build tools:** {_pretty_list(build_tools)}\n")
    lines.append(f"- **Runtime:** {_pretty_list(runtime)}\n")

    lines.append("\n## Entrypoints\n")
    entrypoints = profile.get("entrypoints", []) or []
    if not entrypoints:
        lines.append("- Not detected\n")
    else:
        for ep in entrypoints:
            lines.append(f"- {ep}\n")

    lines.append("\n## APIs\n")
    apis = profile.get("apis", []) or []
    if not apis:
        lines.append("- Not detected\n")
    else:
        for a in apis:
            if isinstance(a, dict):
                api_type = str(a.get("type") or "unknown")
                summary = a.get("summary") if isinstance(a.get("summary"), dict) else None
                routes_file = str(a.get("routes_file") or "").strip()
                details = str(a.get("details") or "").strip()
                if summary:
                    parts = []
                    total_routes = summary.get("total_routes")
                    if total_routes is not None:
                        parts.append(f"total_routes={int(total_routes)}")
                    if api_type.lower() == "http":
                        base_paths = summary.get("base_paths") or []
                        if base_paths:
                            bases = ", ".join(
                                [f"{b.get('path')} ({b.get('count')})" for b in base_paths if isinstance(b, dict)]
                            )
                            if bases:
                                parts.append(f"base_paths=[{bases}]")
                        examples = summary.get("examples") or []
                        if examples:
                            ex = ", ".join([str(x) for x in examples if x is not None])
                            if ex:
                                parts.append(f"examples=[{ex}]")
                    if api_type.lower() == "grpc":
                        services = summary.get("services") or []
                        if services:
                            sv = ", ".join(
                                [f"{s.get('service')} ({s.get('count')})" for s in services if isinstance(s, dict)]
                            )
                            if sv:
                                parts.append(f"services=[{sv}]")
                    if routes_file:
                        parts.append(f"routes_file={routes_file}")
                    details_out = "; ".join(parts) if parts else (details or "Not detected")
                else:
                    details_out = details or "Not detected"
                lines.append(f"- **{api_type}**: {details_out}\n")
            else:
                lines.append(f"- {a}\n")

    lines.append("\n## Data stores\n")
    stores = profile.get("data_stores", []) or []
    if not stores:
        lines.append("- Not detected\n")
    else:
        for s in stores:
            if isinstance(s, dict):
                s_type = str(s.get("type") or "unknown")
                details = str(s.get("details") or "").strip()
                if not details:
                    details = "Not detected"
                lines.append(f"- {s_type}: {details}\n")
            else:
                lines.append(f"- {s}\n")

    lines.append("\n## Containers\n")
    containers = profile.get("containers", []) or []
    if not containers:
        lines.append("- Not detected\n")
    else:
        for c in containers:
            if isinstance(c, dict):
                lines.append(f"### {c.get('name','unknown')}\n")
                lines.append(f"- Type: {c.get('type','unknown')}\n")
                lines.append(f"- Tech: {c.get('tech','unknown')}\n")
                lines.append(f"- Responsibility: {c.get('responsibility','unknown')}\n")
                exposes_raw = c.get("exposes", []) or []
                deps_raw = c.get("depends_on", []) or []
                exposes = [str(x) for x in exposes_raw if x is not None]
                deps = [str(x) for x in deps_raw if x is not None]
                lines.append(f"- Exposes: {_pretty_list(exposes)}\n")
                lines.append(f"- Depends on: {_pretty_list(deps, fallback='Not detected')}\n\n")
            else:
                lines.append(f"### {c}\n")
                lines.append("- Type: Not detected\n")
                lines.append("- Tech: Not detected\n")
                lines.append("- Responsibility: Not detected\n")
                lines.append("- Exposes: Not detected\n")
                lines.append("- Depends on: Not detected\n\n")

    lines.append("\n## Outbound dependencies\n")
    deps = profile.get("dependencies_outbound", []) or []
    if not deps:
        lines.append("- Not detected\n")
    else:
        for d in deps:
            if isinstance(d, dict):
                target = str(d.get("target") or "unknown")
                reason = str(d.get("reason") or "").strip()
                if not reason:
                    reason = "Not detected"
                lines.append(f"- {target}: {reason}\n")
            else:
                lines.append(f"- {d}\n")

    lines.append("\n## Open questions\n")
    oq = profile.get("open_questions", []) or []
    if not oq:
        lines.append("- None\n")
    else:
        for q in oq:
            lines.append(f"- {q}\n")

    coverage_path = out_path.parent / "coverage.json"
    coverage = _load_json(coverage_path) if coverage_path.exists() else None
    if coverage and isinstance(coverage.get("steps"), dict):
        lines.append("\n## Coverage & limits\n")
        for step_key, info in coverage["steps"].items():
            if not isinstance(info, dict):
                continue
            files_total = int(info.get("files_total") or 0)
            bytes_total = int(info.get("bytes_total") or 0)
            batches = int(info.get("batches") or 1)
            limited = bool(info.get("evidence_limited"))
            parse_failed = bool(info.get("parse_failed"))
            lines.append(
                f"- {step_key}: files={files_total}, bytes={_format_bytes(bytes_total)}, "
                f"batches={batches}, evidence_limited={str(limited).lower()}, parse_failed={str(parse_failed).lower()}\n"
            )

    out_path.write_text("".join(lines), encoding="utf-8")
