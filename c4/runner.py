"""CLI runner for step-wise C4 extraction."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Optional

import yaml

from .config import get_paths_config, get_sources_config
from .file_catalog import build_file_catalog, load_file_catalog
from .json_tools import parse_or_repair_json
from .dsl_render import render_structurizr
from .heuristics import enrich_profile, infer_hints
from .index import build_overview
from .mermaid import generate_mermaid_c4
from .ollama_client import ollama_chat
from .output import write_arch_md
from .profile_normalize import normalize_profile
from .prompts import (
    PROFILE_INIT_SYSTEM,
    PROFILE_UPDATE_SYSTEM,
    STRUCTURIZR_REPAIR_SYSTEM,
    STRUCTURIZR_SYSTEM,
)
from .repo_scan import (
    build_step_evidence,
    build_step_sources,
    glob_match,
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


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds into a compact, human-readable string."""
    if seconds < 0:
        seconds = 0.0
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m{secs:04.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h{int(minutes):02d}m{secs:04.1f}s"


def _canon_item(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=True)
        except TypeError:
            return str(value).strip()
    return str(value).strip()


def _build_unique_items(values: list[object]) -> tuple[list[tuple[str, object]], set[str]]:
    items: list[tuple[str, object]] = []
    seen: set[str] = set()
    for item in values:
        key = _canon_item(item)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        items.append((key, item))
    return items, seen


def _merge_list_additive(
    prev_list: list[object],
    updated_list: list[object],
) -> tuple[list[object], int, int]:
    updated_items, updated_set = _build_unique_items(updated_list)
    prev_items, prev_set = _build_unique_items(prev_list)
    kept = len(prev_set - updated_set)
    added = len(updated_set - prev_set)
    merged = [item for _, item in updated_items]
    for key, item in prev_items:
        if key not in updated_set:
            merged.append(item)
    return merged, kept, added


def _merge_profile_additive(prev: dict, updated: dict) -> tuple[dict, dict]:
    """Merge update output with previous profile so new steps only add data."""
    if not isinstance(prev, dict) or not isinstance(updated, dict):
        return deepcopy(updated) if isinstance(updated, dict) else deepcopy(prev), {"added": {}, "kept": {}}
    merged = deepcopy(updated)
    info: dict[str, dict[str, int]] = {"added": {}, "kept": {}}

    for key, value in prev.items():
        if key not in merged:
            merged[key] = deepcopy(value)

    prev_repo = prev.get("repo") if isinstance(prev.get("repo"), dict) else None
    updated_repo = updated.get("repo") if isinstance(updated.get("repo"), dict) else None
    if prev_repo and not updated_repo:
        merged["repo"] = deepcopy(prev_repo)
    elif prev_repo and updated_repo:
        merged_repo = deepcopy(updated_repo)
        for key, value in prev_repo.items():
            if key not in merged_repo:
                merged_repo[key] = deepcopy(value)
        merged["repo"] = merged_repo

    if not str(merged.get("primary_language") or "").strip() and prev.get("primary_language"):
        merged["primary_language"] = prev.get("primary_language")
        info["kept"]["primary_language"] = 1

    prev_build = prev.get("build_and_runtime") if isinstance(prev.get("build_and_runtime"), dict) else {}
    updated_build = updated.get("build_and_runtime") if isinstance(updated.get("build_and_runtime"), dict) else {}
    merged_build = deepcopy(updated_build) if isinstance(updated_build, dict) else {}
    for key, value in prev_build.items():
        if key not in merged_build:
            merged_build[key] = deepcopy(value)
    for subkey in ("build_tools", "runtime"):
        prev_list = prev_build.get(subkey) if isinstance(prev_build.get(subkey), list) else []
        updated_list = updated_build.get(subkey) if isinstance(updated_build.get(subkey), list) else []
        merged_list, kept, added = _merge_list_additive(prev_list, updated_list)
        merged_build[subkey] = merged_list
        if kept:
            info["kept"][f"build_and_runtime.{subkey}"] = kept
        if added:
            info["added"][f"build_and_runtime.{subkey}"] = added
    merged["build_and_runtime"] = merged_build

    for key in ("entrypoints", "apis", "data_stores", "containers", "dependencies_outbound", "open_questions"):
        prev_list = prev.get(key) if isinstance(prev.get(key), list) else []
        updated_list = updated.get(key) if isinstance(updated.get(key), list) else []
        merged_list, kept, added = _merge_list_additive(prev_list, updated_list)
        merged[key] = merged_list
        if kept:
            info["kept"][key] = kept
        if added:
            info["added"][key] = added

    return merged, info


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    parts = [f"{key}+{value}" for key, value in sorted(counts.items())]
    return ", ".join(parts)


def _estimate_step_header_bytes(repo: Path, step: "Step") -> int:
    header = [
        f"REPO: {repo.name}",
        f"PATH: {repo}",
        f"STEP: {step.key} - {step.title}",
        "-----",
    ]
    return len("\n".join(header).encode("utf-8", errors="ignore"))


def _estimate_file_bytes(p: Path, *, max_file_bytes: int, overhead_bytes: int) -> int:
    try:
        size = p.stat().st_size
    except Exception:
        size = None
    if size is None or size <= 0:
        size = max_file_bytes
    return min(size, max_file_bytes) + overhead_bytes


def _split_files_for_budget(
    repo: Path,
    step: "Step",
    files: list[Path],
    *,
    max_step_bytes: int,
    max_file_bytes: int,
    overhead_bytes: int,
) -> list[list[Path]]:
    if not files:
        return []
    header_bytes = _estimate_step_header_bytes(repo, step)
    budget = max(0, max_step_bytes - header_bytes)
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_bytes = 0
    for p in files:
        est = _estimate_file_bytes(p, max_file_bytes=max_file_bytes, overhead_bytes=overhead_bytes)
        if current and (current_bytes + est) > budget:
            batches.append(current)
            current = [p]
            current_bytes = est
        else:
            current.append(p)
            current_bytes += est
    if current:
        batches.append(current)
    return batches


def _trim_routes_blob(
    routes_text: str,
    *,
    max_step_bytes: int,
    header_bytes: int,
) -> str:
    if not routes_text:
        return ""
    blob = "\n\n" + routes_text
    if max_step_bytes <= 0:
        return ""
    max_blob_bytes = max_step_bytes - header_bytes
    if max_blob_bytes <= 0:
        return "\n[ROUTES_TRUNCATED]\n"
    blob_bytes = len(blob.encode("utf-8", errors="ignore"))
    if blob_bytes <= max_blob_bytes:
        return blob
    marker = "\n[ROUTES_TRUNCATED]\n"
    marker_bytes = len(marker.encode("utf-8", errors="ignore"))
    overhead_bytes = len("\n\n".encode("utf-8"))
    allowed = max_blob_bytes - overhead_bytes - marker_bytes
    if allowed <= 0:
        return _truncate_text_bytes(marker, max_blob_bytes)
    trimmed = _truncate_text_bytes(routes_text, allowed)
    return "\n\n" + trimmed + marker


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
        "--analysis-mode",
        choices=("highmem", "lowmem"),
        default="highmem",
        help="Analysis mode: highmem=single-pass per step, lowmem=batch steps to fit small contexts",
    )
    ap.add_argument(
        "--lowmem-batch-max-files",
        type=int,
        default=0,
        help="Max files per lowmem batch (0=auto by byte budget)",
    )
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
    ap.add_argument(
        "--dsl-programmatic",
        action="store_true",
        help="Render Structurizr DSL from repo profiles without using the LLM",
    )
    args = ap.parse_args()

    run_log_path: Optional[Path] = None

    def _append_log(line: str) -> None:
        if run_log_path is None:
            return
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        with run_log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _log(msg: str, *, stderr: bool = False) -> None:
        stream = sys.stderr if stderr else sys.stdout
        print(msg, file=stream)
        _append_log(msg)

    prompt_budget_bytes = None
    if args.respect_num_ctx:
        prompt_budget_bytes = _estimate_prompt_budget_bytes(
            args.num_ctx,
            args.num_predict,
            args.ctx_reserve_tokens,
            args.ctx_bytes_per_token,
        )
        if prompt_budget_bytes <= 0:
            _log(
                "[WARN] num_ctx too small for num_predict + ctx_reserve_tokens; "
                "ctx budgeting disabled.",
                stderr=True,
            )
            prompt_budget_bytes = None

    catalog_path = Path(args.catalog).expanduser().resolve()
    if not catalog_path.exists():
        _log(f'[ERROR] Catalog file not found: "{catalog_path}"', stderr=True)
        return 2

    try:
        catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    except Exception as e:
        _log(f'[ERROR] Failed to read YAML: "{catalog_path}": {e}', stderr=True)
        return 2

    if not isinstance(catalog, dict) or "root" not in catalog or "repos" not in catalog:
        _log('[ERROR] Catalog YAML must contain keys: "root" and "repos"', stderr=True)
        return 2

    root = Path(str(catalog["root"])).expanduser().resolve()
    repos = catalog["repos"]
    if not isinstance(repos, list):
        _log('[ERROR] "repos" must be a list', stderr=True)
        return 2

    ok_repos, errors = validate_catalog(root, repos)
    if errors:
        _log("[ERROR] Catalog validation failed:", stderr=True)
        for e in errors:
            _log(f"  - {e}", stderr=True)
        return 2

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_log_path = out_root / "run.log"
    _append_log(f"[RUN] start {time.strftime('%Y-%m-%d %H:%M:%S')}")

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

    def _mermaid_name(template: str, system_id: str, repo_name: str) -> str:
        """Render the Mermaid filename from templates."""
        try:
            return template.format(system_id=system_id, repo_name=repo_name)
        except Exception as e:
            raise RuntimeError(f'Invalid mermaid filename template "{template}": {e}') from e

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
    mermaid_dir_name = str(paths_cfg.get("mermaid_dir_name", "mermaid"))
    mermaid_tmpl = str(paths_cfg.get("mermaid_filename_template", paths_cfg.get("mermaid_filename", "workspace.mermaid.md")))

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

        def _repo_log(msg: str, *, stderr: bool = False) -> None:
            """Prefix log lines with the repo name."""
            _log(f"[{repo_name}] {msg}", stderr=stderr)

        llm_log = _repo_log if args.verbose else None

        final_profile_path = repo_out / final_profile_name
        dsl_root = out_root / dsl_dir_name / repo_name
        md_path = repo_out / md_name
        mermaid_root = out_root / mermaid_dir_name
        routes_text = ""
        routes_text_mermaid = ""
        routes_entries = None
        profile: Optional[dict] = None
        step_durations: dict[str, float] = {}
        coverage_by_step: dict[str, dict] = {}
        selected_step_files: dict[str, list[Path]] = {}
        c4_duration = 0.0
        mermaid_duration = 0.0

        if args.render_only:
            if not final_profile_path.exists():
                _repo_log("[ERROR] Missing repo-profile.json; rerun without --render-only.", stderr=True)
                continue
            profile = _load_json(final_profile_path)
            if profile is None:
                _repo_log("[ERROR] Failed to read repo-profile.json; rerun without --render-only.", stderr=True)
                continue
            if args.routes_profile:
                routes_path = repo_out / routes_name
                if routes_path.exists():
                    routes_entries = load_routes_profile(routes_path)
                    routes_text = format_routes_text(routes_entries)
                    routes_text_mermaid = format_routes_text(routes_entries, sanitize_for_mermaid=True)
            profile = normalize_profile(
                profile,
                repo_name=repo_name,
                repo_path=str(repo_path),
                routes_entries=routes_entries or None,
            )
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
                step_started = time.perf_counter()
                profile_check_msg: Optional[str] = None
                merge_info: Optional[dict] = None

                apply_step_max = args.analysis_mode != "lowmem"
                if file_catalog is not None:
                    step_files = []
                    for p in all_files:
                        rp = relposix(repo_path, p)
                        entry = file_catalog.get(rp)
                        if entry and step.key in entry.get("categories", []):
                            step_files.append(p)
                    step_files.sort(key=lambda x: relposix(repo_path, x))
                    if apply_step_max and step.max_files is not None:
                        step_files = step_files[: step.max_files]
                else:
                    step_files = select_step_files(repo_path, all_files, step, apply_max_files=apply_step_max)
                if args.analysis_mode != "lowmem" and len(step_files) > args.max_files_per_step:
                    step_files = step_files[: args.max_files_per_step]
                if step.key == "03_entrypoints":
                    filtered = []
                    for p in step_files:
                        rp = relposix(repo_path, p)
                        if any(glob_match(rp, g) for g in step.globs):
                            filtered.append(p)
                    step_files = filtered
                selected_step_files[step.key] = list(step_files)
                total_bytes = 0
                for p in step_files:
                    try:
                        total_bytes += p.stat().st_size
                    except Exception:
                        continue

                log = None
                if args.verbose:
                    _repo_log(f"[STEP FILES] {step.key} ({len(step_files)} files)")
                    for p in step_files:
                        _repo_log(f"[STEP FILE] {step.key} {relposix(repo_path, p)}")

                    def _step_log(msg: str) -> None:
                        """Log step-specific messages with repo prefix."""
                        _repo_log(msg)

                    log = _step_log

                current_profile_json = None
                if args.analysis_mode == "lowmem":
                    lowmem_overhead_bytes = 220
                    step_aborted = False
                    profile_check_msg = None
                    batch_messages: list[str] = []
                    header_bytes = _estimate_step_header_bytes(repo_path, step)

                    max_step_bytes = args.max_step_bytes
                    max_file_bytes = args.max_file_bytes
                    if prompt_budget_bytes is not None:
                        max_step_bytes = _evidence_budget_bytes(
                            system_text=PROFILE_UPDATE_SYSTEM if profile is not None else PROFILE_INIT_SYSTEM,
                            user_prefix=(
                                "CURRENT_PROFILE_JSON:\n"
                                + (json.dumps(profile, indent=2, ensure_ascii=False) if profile else "")
                                + "\n\nNEW_EVIDENCE:\n"
                            )
                            if profile is not None
                            else (
                                f"repo.name={repo_name}\n"
                                f"repo.path={repo_path}\n\n"
                                "EVIDENCE:\n"
                            ),
                            prompt_budget_bytes=prompt_budget_bytes,
                            hard_cap=args.max_step_bytes,
                        )
                        if args.verbose and max_step_bytes < args.max_step_bytes:
                            _repo_log(f"[CTX] {step.key} evidence_budget={max_step_bytes} bytes")

                    if prompt_budget_bytes is not None:
                        max_file_bytes = min(max_file_bytes, max_step_bytes)
                    if max_step_bytes > 0:
                        lowmem_file_cap = max_step_bytes - header_bytes - lowmem_overhead_bytes
                        if lowmem_file_cap > 0:
                            max_file_bytes = min(max_file_bytes, lowmem_file_cap)

                    routes_blob = ""
                    if routes_text and step.key == "04_routing_api":
                        routes_blob = _trim_routes_blob(
                            routes_text,
                            max_step_bytes=max_step_bytes,
                            header_bytes=header_bytes,
                        )

                    batches: list[tuple[list[Path], str]] = []
                    if routes_blob:
                        batches.append(([], routes_blob))

                    if args.lowmem_batch_max_files and args.lowmem_batch_max_files > 0:
                        for i in range(0, len(step_files), args.lowmem_batch_max_files):
                            batches.append((step_files[i : i + args.lowmem_batch_max_files], ""))
                    else:
                        for files_batch in _split_files_for_budget(
                            repo_path,
                            step,
                            step_files,
                            max_step_bytes=max_step_bytes,
                            max_file_bytes=max_file_bytes,
                            overhead_bytes=lowmem_overhead_bytes,
                        ):
                            batches.append((files_batch, ""))

                    if not batches:
                        batches = [([], "")]
                    coverage = {
                        "files_total": len(step_files),
                        "bytes_total": total_bytes,
                        "batches": len(batches),
                        "evidence_limited": False,
                        "parse_failed": False,
                    }

                    evidence_index_lines = [
                        f"REPO: {repo_name}",
                        f"PATH: {repo_path}",
                        f"STEP: {step.key} - {step.title}",
                        "-----",
                        f"BATCHES: {len(batches)}",
                    ]
                    sources_index_lines = list(evidence_index_lines)

                    batch_total = len(batches)
                    for batch_idx, (batch_files, batch_routes_blob) in enumerate(batches, start=1):
                        batch_key = f"{step.key}.batch{batch_idx:02d}"
                        _repo_log(
                            f"[BATCH] {step.key} {batch_idx}/{batch_total} files={len(batch_files)} routes={'yes' if batch_routes_blob else 'no'}"
                        )

                        batch_max_step_bytes = max_step_bytes
                        batch_max_file_bytes = max_file_bytes
                        if prompt_budget_bytes is not None and profile is not None:
                            batch_max_step_bytes = _evidence_budget_bytes(
                                system_text=PROFILE_UPDATE_SYSTEM,
                                user_prefix=(
                                    "CURRENT_PROFILE_JSON:\n"
                                    + json.dumps(profile, indent=2, ensure_ascii=False)
                                    + "\n\nNEW_EVIDENCE:\n"
                                ),
                                prompt_budget_bytes=prompt_budget_bytes,
                                hard_cap=args.max_step_bytes,
                            )
                            batch_max_step_bytes = min(batch_max_step_bytes, max_step_bytes)
                            batch_max_file_bytes = min(batch_max_file_bytes, batch_max_step_bytes)
                            lowmem_file_cap = batch_max_step_bytes - header_bytes - lowmem_overhead_bytes
                            if lowmem_file_cap > 0:
                                batch_max_file_bytes = min(batch_max_file_bytes, lowmem_file_cap)

                        sources = build_step_sources(
                            repo_path,
                            batch_files,
                            step,
                            include_file_size=include_size,
                            include_mtime=include_mtime,
                            fmt=sources_fmt,
                        )
                        sources_path = steps_out / _step_name(sources_tmpl, batch_key)
                        sources_path.write_text(sources, encoding="utf-8")

                        evidence = build_step_evidence(
                            repo_path,
                            batch_files,
                            step,
                            max_step_bytes=batch_max_step_bytes,
                            max_file_bytes=batch_max_file_bytes,
                            max_snippets_per_file=args.max_snippets_per_file,
                            snippet_context_lines=args.snippet_context_lines,
                            chunk_large_files=args.chunk_large_files,
                            log=log,
                        )
                        if batch_routes_blob:
                            evidence = evidence + batch_routes_blob
                        if "[STEP_EVIDENCE_LIMIT_REACHED]" in evidence:
                            coverage["evidence_limited"] = True

                        evidence_path = steps_out / _step_name(evidence_tmpl, batch_key)
                        evidence_path.write_text(evidence, encoding="utf-8")

                        if profile is None:
                            _repo_log(f"[LLM INIT] {step.key} batch={batch_idx}/{batch_total}")
                            system_prompt = PROFILE_INIT_SYSTEM
                            user_prefix = (
                                f"repo.name={repo_name}\n"
                                f"repo.path={repo_path}\n\n"
                                "EVIDENCE:\n"
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
                                label=f"profile_init:{batch_key}",
                            )
                            profile = parse_or_repair_json(
                                raw,
                                ollama_base=args.ollama,
                                model=args.model,
                                timeout_s=args.timeout,
                                num_predict=args.num_predict,
                                num_ctx=args.num_ctx,
                                log=llm_log,
                                label=f"json_repair:{batch_key}",
                            )
                            if not profile:
                                raw_path = steps_out / _step_name(profile_raw_tmpl, batch_key)
                                raw_path.write_text(raw, encoding="utf-8")
                                _repo_log("[WARN] Could not parse init JSON; raw saved.", stderr=True)
                                profile_check_msg = "skipped (init parse failed)"
                                step_aborted = True
                                coverage["parse_failed"] = True
                                batch_messages.append(profile_check_msg)
                                break
                            _normalize_profile(profile, repo_name, repo_path)
                            profile = normalize_profile(
                                profile,
                                repo_name=repo_name,
                                repo_path=str(repo_path),
                                routes_entries=routes_entries or None,
                            )
                            profile_check_msg = "baseline (init)"
                        else:
                            _repo_log(f"[LLM UPDATE] {step.key} batch={batch_idx}/{batch_total}")
                            current_profile_json = json.dumps(profile, indent=2, ensure_ascii=False)
                            system_prompt = PROFILE_UPDATE_SYSTEM
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
                                label=f"profile_update:{batch_key}",
                            )
                            updated = parse_or_repair_json(
                                raw,
                                ollama_base=args.ollama,
                                model=args.model,
                                timeout_s=args.timeout,
                                num_predict=args.num_predict,
                                num_ctx=args.num_ctx,
                                log=llm_log,
                                label=f"json_repair:{batch_key}",
                            )
                            if not updated:
                                raw_path = steps_out / _step_name(profile_raw_tmpl, batch_key)
                                raw_path.write_text(raw, encoding="utf-8")
                                _repo_log("[WARN] Could not parse update JSON; raw saved.", stderr=True)
                                profile_check_msg = "skipped (update parse failed)"
                                coverage["parse_failed"] = True
                            else:
                                prev_profile = deepcopy(profile) if isinstance(profile, dict) else {}
                                profile, merge_info = _merge_profile_additive(prev_profile, updated)
                                _normalize_profile(profile, repo_name, repo_path)
                                profile = normalize_profile(
                                    profile,
                                    repo_name=repo_name,
                                    repo_path=str(repo_path),
                                    routes_entries=routes_entries or None,
                                )
                                added_msg = _format_counts(merge_info.get("added", {})) if merge_info else "none"
                                kept_msg = _format_counts(merge_info.get("kept", {})) if merge_info else "none"
                                profile_check_msg = f"added={added_msg} preserved={kept_msg}"

                        if profile is not None:
                            batch_snap_path = steps_out / _step_name(profile_snap_tmpl, batch_key)
                            batch_snap_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

                        if profile_check_msg:
                            batch_messages.append(f"batch={batch_idx} {profile_check_msg}")

                        evidence_index_lines.append(
                            f"BATCH_{batch_idx}: {batch_key} files={len(batch_files)} evidence={evidence_path.name}"
                        )
                        sources_index_lines.append(
                            f"BATCH_{batch_idx}: {batch_key} files={len(batch_files)} sources={sources_path.name}"
                        )

                    evidence_index_path = steps_out / _step_name(evidence_tmpl, step.key)
                    evidence_index_path.write_text("\n".join(evidence_index_lines), encoding="utf-8")
                    sources_index_path = steps_out / _step_name(sources_tmpl, step.key)
                    sources_index_path.write_text("\n".join(sources_index_lines), encoding="utf-8")

                    if not step_aborted and profile is not None:
                        snap_path = steps_out / _step_name(profile_snap_tmpl, step.key)
                        snap_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

                    step_elapsed = time.perf_counter() - step_started
                    step_durations[step.key] = step_elapsed
                    coverage_by_step[step.key] = coverage
                    _repo_log(f"[TIME] step={step.key} duration={_format_duration(step_elapsed)}")
                    if batch_messages:
                        _repo_log(f"[PROFILE] {step.key} batches={len(batch_messages)} last={batch_messages[-1]}")
                    if step_aborted:
                        break
                    continue

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
                coverage_by_step[step.key] = {
                    "files_total": len(step_files),
                    "bytes_total": total_bytes,
                    "batches": 1,
                    "evidence_limited": "[STEP_EVIDENCE_LIMIT_REACHED]" in evidence,
                    "parse_failed": False,
                }

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
                    _repo_log(f"[LLM INIT] {step.key}")
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
                        _repo_log("[WARN] Could not parse init JSON; raw saved.", stderr=True)
                        profile_check_msg = "skipped (init parse failed)"
                        coverage_by_step[step.key]["parse_failed"] = True
                        step_elapsed = time.perf_counter() - step_started
                        step_durations[step.key] = step_elapsed
                        _repo_log(f"[TIME] step={step.key} duration={_format_duration(step_elapsed)}")
                        if profile_check_msg:
                            _repo_log(f"[PROFILE] {step.key} {profile_check_msg}")
                        break
                    _normalize_profile(profile, repo_name, repo_path)
                    profile = normalize_profile(
                        profile,
                        repo_name=repo_name,
                        repo_path=str(repo_path),
                        routes_entries=routes_entries or None,
                    )
                    profile_check_msg = "baseline (init)"
                else:
                    _repo_log(f"[LLM UPDATE] {step.key}")
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
                        _repo_log("[WARN] Could not parse update JSON; raw saved.", stderr=True)
                        profile_check_msg = "skipped (update parse failed)"
                        coverage_by_step[step.key]["parse_failed"] = True
                    else:
                        prev_profile = deepcopy(profile) if isinstance(profile, dict) else {}
                        profile, merge_info = _merge_profile_additive(prev_profile, updated)
                        _normalize_profile(profile, repo_name, repo_path)
                        profile = normalize_profile(
                            profile,
                            repo_name=repo_name,
                            repo_path=str(repo_path),
                            routes_entries=routes_entries or None,
                        )
                        added_msg = _format_counts(merge_info.get("added", {})) if merge_info else "none"
                        kept_msg = _format_counts(merge_info.get("kept", {})) if merge_info else "none"
                        profile_check_msg = f"added={added_msg} preserved={kept_msg}"

                # Persist a snapshot after each step for traceability.
                snap_path = steps_out / _step_name(profile_snap_tmpl, step.key)
                snap_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
                step_elapsed = time.perf_counter() - step_started
                step_durations[step.key] = step_elapsed
                _repo_log(f"[TIME] step={step.key} duration={_format_duration(step_elapsed)}")
                if profile_check_msg:
                    _repo_log(f"[PROFILE] {step.key} {profile_check_msg}")

            if profile is None:
                _repo_log("[ERROR] No profile produced.", stderr=True)
                steps_total = sum(step_durations.values())
                _repo_log(
                    "[TIME] total steps={steps} c4=0ms mermaid=0ms total={total} (aborted)".format(
                        steps=_format_duration(steps_total),
                        total=_format_duration(steps_total),
                    )
                )
                continue

            # Heuristic enrichment for config-derived datastores and dependencies.
            hint_files: List[Path] = []
            for key in ("01_docs_infra", "03_entrypoints", "05_deps_datastores", "06_configs"):
                hint_files.extend(selected_step_files.get(key, []))
            datastores_hint, deps_hint = infer_hints(
                repo_path,
                hint_files,
                max_file_bytes=min(args.max_file_bytes, 200_000),
                log=_repo_log if args.verbose else None,
            )
            profile = enrich_profile(profile, datastores=datastores_hint, dependencies=deps_hint)
            profile = normalize_profile(
                profile,
                repo_name=repo_name,
                repo_path=str(repo_path),
                routes_entries=routes_entries or None,
            )

            final_profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

            coverage_path = repo_out / "coverage.json"
            coverage_path.write_text(
                json.dumps(
                    {
                        "repo": {"name": repo_name, "path": str(repo_path)},
                        "analysis_mode": args.analysis_mode,
                        "steps": coverage_by_step,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        if profile is None:
            _repo_log("[ERROR] No profile available for rendering.", stderr=True)
            continue

        c4_started = time.perf_counter()
        if args.dsl_programmatic:
            _repo_log("[DSL] Programmatic render")
            dsl = render_structurizr(profile)
        else:
            _repo_log("[LLM] Generate Structurizr DSL")
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
                _repo_log("[LLM] Repair Structurizr DSL")
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
        c4_duration = time.perf_counter() - c4_started
        _repo_log(f"[TIME] c4={_format_duration(c4_duration)}")

        if not args.render_only:
            write_arch_md(md_path, profile)
        if args.mermaid:
            mermaid_started = time.perf_counter()
            _repo_log("[LLM] Generate Mermaid C4")
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
            mermaid_root.mkdir(parents=True, exist_ok=True)
            mermaid_path = mermaid_root / _mermaid_name(mermaid_tmpl, system_id, repo_name)
            mermaid_path.write_text(mermaid_md, encoding="utf-8")
            mermaid_duration = time.perf_counter() - mermaid_started
            _repo_log(f"[TIME] mermaid={_format_duration(mermaid_duration)}")
        _repo_log(f"[OK] wrote {final_profile_path}, {dsl_path}, {view_path}, {md_path}")
        steps_total = sum(step_durations.values())
        steps_label = _format_duration(steps_total)
        if args.render_only:
            steps_label = f"{steps_label} (render-only)"
        mermaid_label = _format_duration(mermaid_duration)
        if not args.mermaid:
            mermaid_label = f"{mermaid_label} (skipped)"
        total_time = steps_total + c4_duration + mermaid_duration
        _repo_log(
            "[TIME] total steps={steps} c4={c4} mermaid={mermaid} total={total}".format(
                steps=steps_label,
                c4=_format_duration(c4_duration),
                mermaid=mermaid_label,
                total=_format_duration(total_time),
            )
        )

    if not args.skip_aggregate:
        try:
            from aggregate_dsl import build_full_workspace

            aggregate_dir = out_root / dsl_dir_name
            aggregate_dir.mkdir(parents=True, exist_ok=True)
            aggregate_path = aggregate_dir / workspace_full_name
            count = build_full_workspace(out_root, aggregate_path)
            if count == 0:
                _log("[WARN] No repositories found for aggregate workspace.", stderr=True)
            else:
                _log(f"[OK] wrote {aggregate_path}")
        except Exception as e:
            _log(f"[WARN] Failed to build aggregate workspace: {e}", stderr=True)

    try:
        overview_path = out_root / "README.md"
        count = build_overview(out_root, overview_path)
        if count:
            _log(f"[OK] wrote {overview_path}")
    except Exception as e:
        _log(f"[WARN] Failed to build overview index: {e}", stderr=True)

    return 0
