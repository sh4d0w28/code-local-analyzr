"""CLI runner for step-wise C4 extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml

from .config import get_paths_config, get_sources_config
from .file_catalog import build_file_catalog
from .json_tools import parse_or_repair_json
from .mermaid import generate_mermaid_c4
from .ollama_client import ollama_chat
from .output import write_arch_md
from .prompts import PROFILE_INIT_SYSTEM, PROFILE_UPDATE_SYSTEM, STRUCTURIZR_SYSTEM
from .repo_scan import (
    build_step_evidence,
    build_step_sources,
    list_repo_files,
    relposix,
    select_step_files,
    validate_catalog,
)
from .routes import build_routes_profile, format_routes_text
from .steps import STEPS


def main() -> int:
    """Run the CLI and orchestrate per-repo analysis."""
    ap = argparse.ArgumentParser(description="Iterative C4 generator using local Ollama (step-by-step evidence)")
    ap.add_argument("--catalog", required=True, help="Path to repos.yaml")
    ap.add_argument("--out", default="architecture-out", help="Output directory")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default=os.environ.get("MODEL", "deepseek-coder-v2:latest"), help="Ollama model name")
    ap.add_argument("--timeout", type=int, default=1200, help="HTTP timeout seconds")
    ap.add_argument("--num-ctx", type=int, default=32768, help="Context window tokens for requests")
    ap.add_argument("--num-predict", type=int, default=4096, help="Max tokens to generate per request")
    ap.add_argument("--max-step-bytes", type=int, default=1_800_000, help="Max evidence bytes per step")
    ap.add_argument("--max-file-bytes", type=int, default=220_000, help="Max bytes per single file excerpt")
    ap.add_argument("--max-snippets-per-file", type=int, default=14, help="Max snippet hits per file when using regex mode")
    ap.add_argument("--snippet-context-lines", type=int, default=14, help="Context lines around regex hit")
    ap.add_argument("--max-files-per-step", type=int, default=260, help="Hard cap of files per step (in addition to step.max_files)")
    ap.add_argument("--classify-files", action="store_true", help="Classify all text files with LLM and use catalog for step selection")
    ap.add_argument("--classify-max-file-bytes", type=int, default=120_000, help="Max bytes sent to classifier per file")
    ap.add_argument("--routes-profile", dest="routes_profile", action="store_true", default=True, help="Extract routes and include them in routing analysis (default: on)")
    ap.add_argument("--no-routes-profile", dest="routes_profile", action="store_false", help="Disable route extraction")
    ap.add_argument("--routes-max-file-bytes", type=int, default=120_000, help="Max bytes per file when extracting routes")
    ap.add_argument("--mermaid", action="store_true", help="Generate Mermaid C4 markdown alongside Structurizr DSL")
    ap.add_argument("--verbose", action="store_true", help="Log per-file selection and analysis details")
    ap.add_argument("--skip-aggregate", action="store_true", help="Skip building a merged workspace DSL across repos")
    args = ap.parse_args()

    catalog_path = Path(args.catalog).expanduser().resolve()
    if not catalog_path.exists():
        print(f'[ERROR] Catalog file not found: "{catalog_path}"', file=sys.stderr)
        return 2

    try:
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f'[ERROR] Failed to read YAML: "{catalog_path}": {e}', file=sys.stderr)
        return 2

    if not isinstance(catalog, dict) or "root" not in catalog or "repos" not in catalog:
        print('[ERROR] Catalog YAML must contain keys: "root" and "repos"', file=sys.stderr)
        return 2

    root = Path(str(catalog["root"])).expanduser().resolve()
    repos = catalog["repos"]
    if not isinstance(repos, list):
        print('[ERROR] "repos" must be a list', file=sys.stderr)
        return 2

    ok_repos, errors = validate_catalog(root, repos)
    if errors:
        print("[ERROR] Catalog validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 2

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    paths_cfg = get_paths_config()
    sources_cfg = get_sources_config()

    def _step_name(template: str, step_key: str) -> str:
        """Render per-step filenames from templates."""
        try:
            return template.format(step_key=step_key)
        except Exception as e:
            raise RuntimeError(f'Invalid filename template "{template}": {e}') from e

    steps_dir_name = str(paths_cfg.get("steps_dir_name", "steps"))
    evidence_tmpl = str(paths_cfg.get("evidence_filename_template", "{step_key}.evidence.txt"))
    sources_tmpl = str(paths_cfg.get("sources_filename_template", "{step_key}.sources.txt"))
    profile_snap_tmpl = str(paths_cfg.get("profile_snapshot_template", "{step_key}.profile.json"))
    profile_raw_tmpl = str(paths_cfg.get("profile_raw_template", "{step_key}.profile.raw.txt"))
    final_profile_name = str(paths_cfg.get("final_profile_filename", "repo-profile.json"))
    dsl_name = str(paths_cfg.get("workspace_filename", "workspace.dsl"))
    md_name = str(paths_cfg.get("architecture_md_filename", "ARCHITECTURE.md"))
    catalog_name = str(paths_cfg.get("file_catalog_filename", "file-catalog.jsonl"))
    routes_name = str(paths_cfg.get("routes_profile_filename", "routes.jsonl"))
    mermaid_name = str(paths_cfg.get("mermaid_filename", "workspace.mermaid.md"))

    include_size = bool(sources_cfg.get("include_file_size", True))
    include_mtime = bool(sources_cfg.get("include_mtime", False))
    sources_fmt = str(sources_cfg.get("format", "tsv"))

    def _normalize_profile(profile: dict, repo_name: str, repo_path: Path) -> dict:
        """Ensure repo.name/path fields are consistent with the catalog."""
        repo = profile.get("repo")
        if not isinstance(repo, dict):
            repo = {}
            profile["repo"] = repo
        repo["name"] = repo_name
        repo["path"] = str(repo_path)
        return profile

    for repo_name, repo_path in ok_repos:
        repo_out = out_root / "repos" / repo_name
        steps_out = repo_out / steps_dir_name
        steps_out.mkdir(parents=True, exist_ok=True)

        def _repo_log(msg: str) -> None:
            """Prefix log lines with the repo name."""
            print(f"[{repo_name}] {msg}")

        llm_log = _repo_log if args.verbose else None

        final_profile_path = repo_out / final_profile_name
        dsl_path = repo_out / dsl_name
        md_path = repo_out / md_name
        mermaid_path = repo_out / mermaid_name

        all_files = list_repo_files(repo_path)

        file_catalog = None
        if args.classify_files:
            catalog_path = repo_out / catalog_name

            file_catalog = build_file_catalog(
                repo_path,
                all_files,
                STEPS,
                out_path=catalog_path,
                ollama_base=args.ollama,
                model=args.model,
                timeout_s=args.timeout,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                max_file_bytes=args.classify_max_file_bytes,
                log=_repo_log,
                llm_log=llm_log,
                verbose=args.verbose,
            )

        routes_text = ""
        routes_text_mermaid = ""
        routes_entries = None
        if args.routes_profile:
            routes_path = repo_out / routes_name

            routes_entries = build_routes_profile(
                repo_path,
                all_files,
                out_path=routes_path,
                max_file_bytes=args.routes_max_file_bytes,
                log=_repo_log,
                verbose=args.verbose,
            )
            routes_text = format_routes_text(routes_entries)
            routes_text_mermaid = format_routes_text(routes_entries, sanitize_for_mermaid=True)

        profile: Optional[dict] = None

        # Iterate steps to refine the profile with additional evidence.
        for step in STEPS:
            if file_catalog is not None:
                step_files = []
                for p in all_files:
                    rp = relposix(repo_path, p)
                    entry = file_catalog.get(rp)
                    if entry and step.key in entry.get("categories", []):
                        step_files.append(p)
                step_files.sort(key=lambda x: relposix(repo_path, x))
                if step.max_files is not None:
                    step_files = step_files[: step.max_files]
            else:
                step_files = select_step_files(repo_path, all_files, step)
            if len(step_files) > args.max_files_per_step:
                step_files = step_files[: args.max_files_per_step]

            log = None
            if args.verbose:
                print(f"[{repo_name}] [STEP FILES] {step.key} ({len(step_files)} files)")
                for p in step_files:
                    print(f"[{repo_name}] [STEP FILE] {step.key} {relposix(repo_path, p)}")

                def _log(msg: str) -> None:
                    """Log step-specific messages with repo prefix."""
                    print(f"[{repo_name}] {msg}")

                log = _log

            evidence = build_step_evidence(
                repo_path,
                step_files,
                step,
                max_step_bytes=args.max_step_bytes,
                max_file_bytes=args.max_file_bytes,
                max_snippets_per_file=args.max_snippets_per_file,
                snippet_context_lines=args.snippet_context_lines,
                log=log,
            )
            if routes_text and step.key == "04_routing_api":
                evidence = evidence + "\n\n" + routes_text

            sources = build_step_sources(
                repo_path,
                step_files,
                step,
                include_file_size=include_size,
                include_mtime=include_mtime,
                fmt=sources_fmt,
            )
            sources_path = steps_out / _step_name(sources_tmpl, step.key)
            sources_path.write_text(sources, encoding="utf-8")

            evidence_path = steps_out / _step_name(evidence_tmpl, step.key)
            evidence_path.write_text(evidence, encoding="utf-8")

            if profile is None:
                print(f"[{repo_name}] [LLM INIT] {step.key}")
                user = (
                    f"repo.name={repo_name}\n"
                    f"repo.path={repo_path}\n\n"
                    f"EVIDENCE:\n{evidence}"
                )
                raw = ollama_chat(
                    args.ollama, args.model,
                    PROFILE_INIT_SYSTEM,
                    user,
                    temperature=0.0,
                    timeout_s=args.timeout,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    log=llm_log,
                    label=f"profile_init:{step.key}",
                )
                profile = parse_or_repair_json(
                    raw,
                    ollama_base=args.ollama,
                    model=args.model,
                    timeout_s=args.timeout,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    log=llm_log,
                    label=f"json_repair:{step.key}",
                )
                if not profile:
                    raw_path = steps_out / _step_name(profile_raw_tmpl, step.key)
                    raw_path.write_text(raw, encoding="utf-8")
                    print(f"[{repo_name}] [WARN] Could not parse init JSON; raw saved.", file=sys.stderr)
                    break
                _normalize_profile(profile, repo_name, repo_path)
            else:
                print(f"[{repo_name}] [LLM UPDATE] {step.key}")
                user = (
                    "CURRENT_PROFILE_JSON:\n"
                    + json.dumps(profile, indent=2, ensure_ascii=False)
                    + "\n\nNEW_EVIDENCE:\n"
                    + evidence
                )
                raw = ollama_chat(
                    args.ollama, args.model,
                    PROFILE_UPDATE_SYSTEM,
                    user,
                    temperature=0.0,
                    timeout_s=args.timeout,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    log=llm_log,
                    label=f"profile_update:{step.key}",
                )
                updated = parse_or_repair_json(
                    raw,
                    ollama_base=args.ollama,
                    model=args.model,
                    timeout_s=args.timeout,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    log=llm_log,
                    label=f"json_repair:{step.key}",
                )
                if not updated:
                    raw_path = steps_out / _step_name(profile_raw_tmpl, step.key)
                    raw_path.write_text(raw, encoding="utf-8")
                    print(f"[{repo_name}] [WARN] Could not parse update JSON; raw saved.", file=sys.stderr)
                else:
                    profile = updated
                    _normalize_profile(profile, repo_name, repo_path)

            # Persist a snapshot after each step for traceability.
            snap_path = steps_out / _step_name(profile_snap_tmpl, step.key)
            snap_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

        if profile is None:
            print(f"[{repo_name}] [ERROR] No profile produced.", file=sys.stderr)
            continue

        final_profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[{repo_name}] [LLM] Generate Structurizr DSL")
        dsl_user = "REPO_PROFILE_JSON:\n" + json.dumps(profile, indent=2, ensure_ascii=False)
        dsl = ollama_chat(
            args.ollama, args.model,
            STRUCTURIZR_SYSTEM,
            dsl_user,
            temperature=0.0,
            timeout_s=args.timeout,
            num_predict=args.num_predict,
            num_ctx=args.num_ctx,
            log=llm_log,
            label="structurizr",
        )
        dsl_path.write_text(dsl, encoding="utf-8")

        write_arch_md(md_path, profile)
        if args.mermaid:
            print(f"[{repo_name}] [LLM] Generate Mermaid C4")
            mermaid_md = generate_mermaid_c4(
                profile,
                routes_text=routes_text_mermaid or routes_text,
                ollama_base=args.ollama,
                model=args.model,
                timeout_s=args.timeout,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                log=llm_log,
            )
            mermaid_path.write_text(mermaid_md, encoding="utf-8")
        print(f"[{repo_name}] [OK] wrote {final_profile_path}, {dsl_path}, {md_path}")

    if not args.skip_aggregate:
        try:
            from aggregate_dsl import build_full_workspace

            aggregate_path = out_root / "workspace.full.dsl"
            count = build_full_workspace(out_root, aggregate_path)
            if count == 0:
                print("[WARN] No repositories found for aggregate workspace.", file=sys.stderr)
            else:
                print(f"[OK] wrote {aggregate_path}")
        except Exception as e:
            print(f"[WARN] Failed to build aggregate workspace: {e}", file=sys.stderr)

    return 0
