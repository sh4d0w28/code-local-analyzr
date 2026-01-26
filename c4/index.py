"""Generate a top-level overview index for architecture outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_overview(out_root: Path, out_path: Path) -> int:
    """Build a top-level README linking to per-repo outputs."""
    repos_dir = out_root / "repos"
    if not repos_dir.exists():
        return 0

    repo_entries: List[str] = []
    for repo_dir in sorted(repos_dir.iterdir(), key=lambda p: p.name.lower()):
        if not repo_dir.is_dir():
            continue
        profile = _load_json(repo_dir / "repo-profile.json") or {}
        name = str(profile.get("repo", {}).get("name") or repo_dir.name)
        lang = str(profile.get("primary_language") or "unknown")
        arch_md = repo_dir / "ARCHITECTURE.md"
        coverage = repo_dir / "coverage.json"
        routes = repo_dir / "routes.jsonl"
        dsl = out_root / "dsl" / repo_dir.name / f"{repo_dir.name}.dsl"
        mermaid = out_root / "mermaid" / f"{repo_dir.name}_mermaid.md"

        parts = [f"- **{name}** ({lang})"]
        if arch_md.exists():
            parts.append(f"  - ARCH: {arch_md.relative_to(out_root)}")
        if dsl.exists():
            parts.append(f"  - DSL: {dsl.relative_to(out_root)}")
        if mermaid.exists():
            parts.append(f"  - Mermaid: {mermaid.relative_to(out_root)}")
        if routes.exists():
            parts.append(f"  - Routes: {routes.relative_to(out_root)}")
        if coverage.exists():
            parts.append(f"  - Coverage: {coverage.relative_to(out_root)}")
        repo_entries.append("\n".join(parts))

    lines: List[str] = []
    lines.append("# Architecture Outputs Index\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## How to read\n")
    lines.append("- Start with the system landscape DSL: `dsl/workspace_full.dsl`.\n")
    lines.append("- For each repo, review `repos/<name>/ARCHITECTURE.md` and the per-repo DSL/mermaid.\n")
    lines.append("- Use `routes.jsonl` for raw API evidence and `coverage.json` to see scan limits.\n")

    lines.append("\n## Repositories\n")
    if repo_entries:
        lines.append("\n".join(repo_entries))
        lines.append("")
    else:
        lines.append("- No repos found.\n")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return len(repo_entries)
