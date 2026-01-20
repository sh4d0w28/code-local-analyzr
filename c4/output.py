"""Render human-readable architecture summaries."""

from __future__ import annotations

from pathlib import Path
from typing import List


def write_arch_md(out_path: Path, profile: dict) -> None:
    """Write a Markdown summary from a repo profile."""
    lines: List[str] = []
    repo_name = profile.get("repo", {}).get("name", "Repository")
    lines.append(f"# {repo_name} Architecture\n")

    lines.append("## Overview\n")
    lines.append(f"- **Primary language:** {profile.get('primary_language', 'unknown')}\n")
    br = profile.get("build_and_runtime", {}) or {}
    lines.append(f"- **Build tools:** {', '.join(br.get('build_tools', []) or ['unknown'])}\n")
    lines.append(f"- **Runtime:** {', '.join(br.get('runtime', []) or ['unknown'])}\n")

    lines.append("\n## Entrypoints\n")
    for ep in profile.get("entrypoints", []) or ["unknown"]:
        lines.append(f"- {ep}\n")

    lines.append("\n## APIs\n")
    apis = profile.get("apis", []) or []
    if not apis:
        lines.append("- unknown\n")
    else:
        for a in apis:
            lines.append(f"- **{a.get('type','unknown')}**: {a.get('details','')}\n")

    lines.append("\n## Data stores\n")
    stores = profile.get("data_stores", []) or []
    if not stores:
        lines.append("- unknown\n")
    else:
        for s in stores:
            lines.append(f"- {s.get('type','unknown')}: {s.get('details','')}\n")

    lines.append("\n## Containers\n")
    containers = profile.get("containers", []) or []
    if not containers:
        lines.append("- unknown\n")
    else:
        for c in containers:
            lines.append(f"### {c.get('name','unknown')}\n")
            lines.append(f"- Type: {c.get('type','unknown')}\n")
            lines.append(f"- Tech: {c.get('tech','unknown')}\n")
            lines.append(f"- Responsibility: {c.get('responsibility','unknown')}\n")
            exposes = c.get("exposes", []) or []
            deps = c.get("depends_on", []) or []
            lines.append(f"- Exposes: {', '.join(exposes) if exposes else 'unknown'}\n")
            lines.append(f"- Depends on: {', '.join(deps) if deps else 'none/unknown'}\n\n")

    lines.append("\n## Outbound dependencies\n")
    deps = profile.get("dependencies_outbound", []) or []
    if not deps:
        lines.append("- unknown\n")
    else:
        for d in deps:
            lines.append(f"- {d.get('target','unknown')}: {d.get('reason','')}\n")

    lines.append("\n## Open questions\n")
    oq = profile.get("open_questions", []) or []
    if not oq:
        lines.append("- none\n")
    else:
        for q in oq:
            lines.append(f"- {q}\n")

    out_path.write_text("".join(lines), encoding="utf-8")
