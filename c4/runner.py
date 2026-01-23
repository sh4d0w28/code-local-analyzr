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
from .file_catalog import build_file_catalog, load_file_catalog
from .json_tools import parse_or_repair_json
from .mermaid import generate_mermaid_c4
from .ollama_client import ollama_chat
from .output import write_arch_md
from .prompts import (
    PROFILE_INIT_SYSTEM,
    PROFILE_UPDATE_SYSTEM,
    STRUCTURIZR_REPAIR_SYSTEM,
    STRUCTURIZR_SYSTEM,
)
from .repo_scan import (
    build_step_evidence,
    build_step_sources,
    list_repo_files,
    relposix,
    select_step_files,
    validate_catalog,
)
from .routes import build_routes_profile, format_routes_text, load_routes_profile
from .steps import STEPS
from .views import extract_system_id_and_label, render_views


def _estimate_prompt_budget_bytes(
    num_ctx: int,
    num_predict: int,
    reserve_tokens: int,
    bytes_per_token: float,
) -> int:
    """Estimate prompt byte budget from context settings."""
    if bytes_per_token <= 0:
        return 0
    usable_tokens = num_ctx - num_predict - reserve_tokens
    if usable_tokens <= 0:
        return 0
    return max(0, int(usable_tokens * bytes_per_token))


def _evidence_budget_bytes(
    *,
    system_text: str,
    user_prefix: str,
    prompt_budget_bytes: int,
    hard_cap: int,
) -> int:
    """Compute evidence byte budget after accounting for prompt overhead."""
    system_bytes = len(system_text.encode("utf-8", errors="ignore"))
    prefix_bytes = len(user_prefix.encode("utf-8", errors="ignore"))
    available = prompt_budget_bytes - system_bytes - prefix_bytes
    if available < 0:
        available = 0
    return min(available, hard_cap)


def _truncate_text_bytes(text: str, max_bytes: int) -> str:
    """Trim text to a byte length, preserving UTF-8 safety."""
    if max_bytes <= 0:
        return ""
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) <= max_bytes:
        return text
    return raw[:max_bytes].decode("utf-8", errors="ignore")


def main() -> int:
    """Run the CLI and orchestrate per-repo analysis."""
    ap = argparse.ArgumentParser(description="Iterative C4 generator using local Ollama (step-by-step evidence)")
    ap.add_argument("--catalog", required=True, help="Path to repos.yaml")
    ap.add_argument("--out", default="architecture-out", help="Output directory")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default=os.environ.get("MODEL", "deepseek-coder-v2:latest"), help="Ollama model name")
    ap.add_argument(
        "--classify-files-model",
        default=os.environ.get("CLASSIFY_FILES_MODEL", "qwen2.5-coder:7b-instruct"),
        help="Ollama model name for file classification",
    )
    ap.add_argument("--timeout", type=int, default=7200, help="HTTP timeout seconds")
    ap.add_argument("--num-ctx", type=int, default=163840, help="Context window tokens for requests")
    ap.add_argument("--num-predict", type=int, default=32768, help="Max tokens to generate per request")
    ap.add_argument(
        "--respect-num-ctx",
        dest="respect_num_ctx",
        action="store_true",
        default=True,
        help="Scale evidence and classification payloads to stay within num_ctx (default: on)",
    )
    ap.add_argument(
        "--no-respect-num-ctx",
        dest="respect_num_ctx",
        action="store_false",
        help="Disable num_ctx-based prompt budgeting",
    )
    ap.add_argument(
        "--ctx-bytes-per-token",
        type=float,
        default=4.0,
        help="Estimated bytes per token for num_ctx budgeting",
    )
    ap.add_argument(
        "--ctx-reserve-tokens",
        type=int,
        default=512,
        help="Extra tokens reserved as headroom in addition to num_predict",
    )
    ap.add_argument("--max-step-bytes", type=int, default=6_000_000, help="Max evidence bytes per step")
    ap.add_argument("--max-file-bytes", type=int, default=800_000, help="Max bytes per single file excerpt")
    ap.add_argument("--max-snippets-per-file", type=int, default=14, help="Max snippet hits per file when using regex mode")
    ap.add_argument("--snippet-context-lines", type=int, default=14, help="Context lines around regex hit")
    ap.add_argument(
        "--chunk-large-files",
        action="store_true",
        help="Split large files into sequential chunks (size = --max-file-bytes) for evidence",
    )
    ap.add_argument("--max-files-per-step", type=int, default=800, help="Hard cap of files per step (in addition to step.max_files)")
    ap.add_argument("--classify-files", action="store_true", help="Classify all text files with LLM and use catalog for step selection")
    ap.add_argument("--classify-max-file-bytes", type=int, default=200_000, help="Max bytes sent to classifier per file")
    ap.add_argument("--routes-profile", dest="routes_profile", action="store_true", default=True, help="Extract routes and include them in routing analysis (default: on)")
    ap.add_argument("--no-routes-profile", dest="routes_profile", action="store_false", help="Disable route extraction")
    ap.add_argument("--routes-max-file-bytes", type=int, default=200_000, help="Max bytes per file when extracting routes")
    ap.add_argument("--mermaid", action="store_true", help="Generate Mermaid C4 markdown alongside Structurizr DSL")
    ap.add_argument("--verbose", action="store_true", help="Log per-file selection and analysis details")
    ap.add_argument("--skip-aggregate", action="store_true", help="Skip building a merged workspace DSL across repos")
    ap.add_argument(
        "--render-only",
        action="store_true",
        help="Generate DSL/mermaid from existing repo-profile.json without re-analyzing repositories",
    )
    args = ap.parse_args()

    prompt_budget_bytes = None
    if args.respect_num_ctx:
        prompt_budget_bytes = _estimate_prompt_budget_bytes(
            args.num_ctx,
            args.num_predict,
            args.ctx_reserve_tokens,
            args.ctx_bytes_per_token,
        )
        if prompt_budget_bytes <= 0:
            print(
                "[WARN] num_ctx too small for num_predict + ctx_reserve_tokens; "
                "ctx budgeting disabled.",
                file=sys.stderr,
            )
            prompt_budget_bytes = None

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

    def _view_name(template: str, system_id: str, repo_name: str) -> str:
        """Render the view filename from templates."""
        try:
            return template.format(system_id=system_id, repo_name=repo_name)
        except Exception as e:
            raise RuntimeError(f'Invalid view filename template "{template}": {e}') from e

    def _dsl_name(template: str, system_id: str, repo_name: str) -> str:
        """Render the DSL filename from templates."""
        try:
            return template.format(system_id=system_id, repo_name=repo_name)
        except Exception as e:
            raise RuntimeError(f'Invalid DSL filename template "{template}": {e}') from e

    steps_dir_name = str(paths_cfg.get("steps_dir_name", "steps"))
    evidence_tmpl = str(paths_cfg.get("evidence_filename_template", "{step_key}.evidence.txt"))
    sources_tmpl = str(paths_cfg.get("sources_filename_template", "{step_key}.sources.txt"))
    profile_snap_tmpl = str(paths_cfg.get("profile_snapshot_template", "{step_key}.profile.json"))
    profile_raw_tmpl = str(paths_cfg.get("profile_raw_template", "{step_key}.profile.raw.txt"))
    final_profile_name = str(paths_cfg.get("final_profile_filename", "repo-profile.json"))
    dsl_dir_name = str(paths_cfg.get("dsl_dir_name", "dsl"))
    dsl_tmpl = str(paths_cfg.get("dsl_filename_template", paths_cfg.get("workspace_filename", "workspace.dsl")))
    view_tmpl = str(paths_cfg.get("view_filename_template", "{repo_name}View.dsl"))
    workspace_full_name = str(paths_cfg.get("workspace_full_filename", "workspace_full.dsl"))
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

    def _load_json(path: Path) -> Optional[dict]:
        """Load a JSON file into a dict if possible."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    for repo_name, repo_path in ok_repos:
        repo_out = out_root / "repos" / repo_name
        steps_out = repo_out / steps_dir_name
        steps_out.mkdir(parents=True, exist_ok=True)

        def _repo_log(msg: str) -> None:
            """Prefix log lines with the repo name."""
            print(f"[{repo_name}] {msg}")

        llm_log = _repo_log if args.verbose else None

        final_profile_path = repo_out / final_profile_name
        dsl_root = out_root / dsl_dir_name / repo_name
        md_path = repo_out / md_name
        mermaid_path = repo_out / mermaid_name
        routes_text = ""
        routes_text_mermaid = ""
        routes_entries = None
        profile: Optional[dict] = None

        if args.render_only:
            if not final_profile_path.exists():
                print(
                    f"[{repo_name}] [ERROR] Missing repo-profile.json; rerun without --render-only.",
                    file=sys.stderr,
                )
                continue
            profile = _load_json(final_profile_path)
            if profile is None:
                print(
                    f"[{repo_name}] [ERROR] Failed to read repo-profile.json; rerun without --render-only.",
                    file=sys.stderr,
                )
                continue
            if args.routes_profile:
                routes_path = repo_out / routes_name
                if routes_path.exists():
                    routes_entries = load_routes_profile(routes_path)
                    routes_text = format_routes_text(routes_entries)
                    routes_text_mermaid = format_routes_text(routes_entries, sanitize_for_mermaid=True)
        else:
            all_files = list_repo_files(repo_path)

            file_catalog = None
            catalog_path = repo_out / catalog_name
            if args.classify_files:
                file_catalog = build_file_catalog(
                    repo_path,
                    all_files,
                    STEPS,
                    out_path=catalog_path,
                    ollama_base=args.ollama,
                    model=args.classify_files_model,
                    timeout_s=args.timeout,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    max_file_bytes=args.classify_max_file_bytes,
                    prompt_budget_bytes=prompt_budget_bytes,
                    log=_repo_log,
                    llm_log=llm_log,
                    verbose=args.verbose,
                )
            elif catalog_path.exists():
                file_catalog = load_file_catalog(catalog_path)
                if args.verbose:
                    _repo_log(f"[CLASSIFY] reuse existing catalog={catalog_path}")

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

                current_profile_json = None
                if profile is None:
                    system_prompt = PROFILE_INIT_SYSTEM
                    user_prefix = (
                        f"repo.name={repo_name}\n"
                        f"repo.path={repo_path}\n\n"
                        "EVIDENCE:\n"
                    )
                else:
                    current_profile_json = json.dumps(profile, indent=2, ensure_ascii=False)
                    system_prompt = PROFILE_UPDATE_SYSTEM
                    user_prefix = (
                        "CURRENT_PROFILE_JSON:\n"
                        + current_profile_json
                        + "\n\nNEW_EVIDENCE:\n"
                    )

                max_step_bytes = args.max_step_bytes
                max_file_bytes = args.max_file_bytes
                if prompt_budget_bytes is not None:
                    max_step_bytes = _evidence_budget_bytes(
                        system_text=system_prompt,
                        user_prefix=user_prefix,
                        prompt_budget_bytes=prompt_budget_bytes,
                        hard_cap=args.max_step_bytes,
                    )
                    if args.verbose and max_step_bytes < args.max_step_bytes:
                        _repo_log(f"[CTX] {step.key} evidence_budget={max_step_bytes} bytes")

                routes_blob = ""
                max_step_bytes_for_files = max_step_bytes
                if routes_text and step.key == "04_routing_api":
                    routes_blob = "\n\n" + routes_text
                    if prompt_budget_bytes is not None:
                        routes_blob_bytes = len(routes_blob.encode("utf-8", errors="ignore"))
                        if routes_blob_bytes > max_step_bytes:
                            marker = "\n[ROUTES_TRUNCATED]\n"
                            marker_bytes = len(marker.encode("utf-8", errors="ignore"))
                            overhead_bytes = len("\n\n".encode("utf-8"))
                            allowed = max_step_bytes - overhead_bytes - marker_bytes
                            if allowed <= 0:
                                routes_blob = _truncate_text_bytes(marker, max_step_bytes)
                            else:
                                trimmed = _truncate_text_bytes(routes_text, allowed)
                                routes_blob = "\n\n" + trimmed + marker
                            routes_blob_bytes = len(routes_blob.encode("utf-8", errors="ignore"))
                        max_step_bytes_for_files = max(0, max_step_bytes - routes_blob_bytes)

                if prompt_budget_bytes is not None:
                    max_file_bytes = min(max_file_bytes, max_step_bytes_for_files)

                evidence = build_step_evidence(
                    repo_path,
                    step_files,
                    step,
                    max_step_bytes=max_step_bytes_for_files,
                    max_file_bytes=max_file_bytes,
                    max_snippets_per_file=args.max_snippets_per_file,
                    snippet_context_lines=args.snippet_context_lines,
                    chunk_large_files=args.chunk_large_files,
                    log=log,
                )
                if routes_blob:
                    evidence = evidence + routes_blob

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
                    user = user_prefix + evidence
                    raw = ollama_chat(
                        args.ollama, args.model,
                        system_prompt,
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
                    if current_profile_json is None:
                        current_profile_json = json.dumps(profile, indent=2, ensure_ascii=False)
                        user_prefix = (
                            "CURRENT_PROFILE_JSON:\n"
                            + current_profile_json
                            + "\n\nNEW_EVIDENCE:\n"
                        )
                    user = user_prefix + evidence
                    raw = ollama_chat(
                        args.ollama, args.model,
                        system_prompt,
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

        if profile is None:
            print(f"[{repo_name}] [ERROR] No profile available for rendering.", file=sys.stderr)
            continue

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
        if STRUCTURIZR_REPAIR_SYSTEM:
            print(f"[{repo_name}] [LLM] Repair Structurizr DSL")
            dsl_repair_user = (
                "REPO_PROFILE_JSON:\n"
                + json.dumps(profile, indent=2, ensure_ascii=False)
                + "\n\nDSL_INPUT:\n"
                + dsl
            )
            repaired = ollama_chat(
                args.ollama, args.model,
                STRUCTURIZR_REPAIR_SYSTEM,
                dsl_repair_user,
                temperature=0.0,
                timeout_s=args.timeout,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                log=llm_log,
                label="structurizr_repair",
            )
            if repaired and repaired.strip():
                dsl = repaired
        system_id, system_label = extract_system_id_and_label(dsl, repo_name)
        dsl_root.mkdir(parents=True, exist_ok=True)
        dsl_path = dsl_root / _dsl_name(dsl_tmpl, system_id, repo_name)
        dsl_path.write_text(dsl, encoding="utf-8")

        view_path = dsl_root / _view_name(view_tmpl, system_id, repo_name)
        view_path.write_text(render_views(system_id, system_label), encoding="utf-8")

        if not args.render_only:
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
        print(f"[{repo_name}] [OK] wrote {final_profile_path}, {dsl_path}, {view_path}, {md_path}")

    if not args.skip_aggregate:
        try:
            from aggregate_dsl import build_full_workspace

            aggregate_dir = out_root / dsl_dir_name
            aggregate_dir.mkdir(parents=True, exist_ok=True)
            aggregate_path = aggregate_dir / workspace_full_name
            count = build_full_workspace(out_root, aggregate_path)
            if count == 0:
                print("[WARN] No repositories found for aggregate workspace.", file=sys.stderr)
            else:
                print(f"[OK] wrote {aggregate_path}")
        except Exception as e:
            print(f"[WARN] Failed to build aggregate workspace: {e}", file=sys.stderr)

    return 0
