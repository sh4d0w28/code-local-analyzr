#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import yaml

# ============================================================
# Prompts
# ============================================================

REPO_PROFILE_SYSTEM = """You analyze a software repository using ONLY the provided signal-pack text.
Do not invent. If evidence is missing, use "unknown" and add an item in open_questions with what file(s) would confirm it.
Return STRICT JSON only. No markdown, no commentary.

Schema:
{
  "repo": {"name": string, "path": string},
  "primary_language": string,
  "build_and_runtime": {"build_tools": [string], "runtime": [string]},
  "entrypoints": [string],
  "apis": [{"type": "http"|"grpc"|"events"|"cli"|"unknown", "details": string}],
  "data_stores": [{"type": string, "details": string}],
  "dependencies_outbound": [{"target": string, "reason": string}],
  "containers": [
    {"name": string,
     "type": "service"|"worker"|"library"|"frontend"|"job"|"unknown",
     "tech": string,
     "responsibility": string,
     "exposes": [string],
     "depends_on": [string]}
  ],
  "open_questions": [string]
}

Rules:
- Put evidence references in details/reason fields as file paths (e.g., "Dockerfile", "k8s/deployment.yaml", "cmd/api/main.go").
- If you are not sure, write "unknown" (do not guess).
"""

STRUCTURIZR_SYSTEM = """Generate a Structurizr DSL workspace (C4 Context + Container) from the given repo profile JSON.
Use only what is in the JSON; do not add invented dependencies.
Output ONLY DSL text, no markdown.

Requirements:
- One workspace.
- One softwareSystem for this repo.
- One person "User" if any container exposes APIs (http/grpc).
- Add containers from repo_profile.containers.
- Add relationships using depends_on and dependencies_outbound.
- For unknowns, omit rather than guess.
"""

JSON_REPAIR_SYSTEM = """You are a strict JSON repair tool.
Convert the INPUT_TEXT into a single valid JSON object that conforms to the schema below.

Rules:
- Output JSON only (no markdown, no code fences, no commentary).
- Remove trailing commas and any comments.
- If the input seems incomplete, produce best-effort JSON; add an explanation into open_questions.

Schema:
{
  "repo": {"name": string, "path": string},
  "primary_language": string,
  "build_and_runtime": {"build_tools": [string], "runtime": [string]},
  "entrypoints": [string],
  "apis": [{"type": "http"|"grpc"|"events"|"cli"|"unknown", "details": string}],
  "data_stores": [{"type": string, "details": string}],
  "dependencies_outbound": [{"target": string, "reason": string}],
  "containers": [
    {"name": string,
     "type": "service"|"worker"|"library"|"frontend"|"job"|"unknown",
     "tech": string,
     "responsibility": string,
     "exposes": [string],
     "depends_on": [string]}
  ],
  "open_questions": [string]
}
"""

# ============================================================
# Signal selection & redaction
# ============================================================

DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "dist", "build", "vendor", ".next", ".turbo", "coverage",
    ".idea", ".vscode", "target", "bin", "obj", ".gradle", ".mvn", ".venv",
}

SIGNAL_GLOBS = [
    "README*", "docs/**", "doc/**", "architecture/**", "adr/**", "ADRs/**",
    "Dockerfile*", "docker-compose*.yml", "docker-compose*.yaml", "compose*.yml", "compose*.yaml",
    "k8s/**", "kubernetes/**", "deploy/**", "deployment/**", "manifests/**",
    "helm/**", "charts/**",
    ".github/workflows/**",
    "Makefile", "Taskfile.yml", "Taskfile.yaml",
    "go.mod", "go.sum", "go.work", "go.work.sum",
    "package.json", "pnpm-lock.yaml", "yarn.lock", "bun.lockb", "tsconfig*.json",
    "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle", "settings.gradle.kts", "gradle.properties",
    "requirements*.txt", "pyproject.toml", "poetry.lock", "Pipfile", "Pipfile.lock",
    "proto/**", "**/*.proto",
    "**/openapi*.yml", "**/openapi*.yaml", "**/swagger*.yml", "**/swagger*.yaml",
    "**/asyncapi*.yml", "**/asyncapi*.yaml",
]

ENTRYPOINT_NAME_HINTS = {
    # Go
    "main.go",
    # Node
    "index.js", "index.ts", "server.js", "server.ts", "app.js", "app.ts", "main.ts", "main.js",
    # Java
    "Application.java", "Main.java",
    # Python
    "app.py", "main.py", "manage.py",
}

ENTRYPOINT_PATH_HINTS = ["cmd/", "src/", "server/", "app/", "internal/", "services/", "api/"]

TEXT_EXT_ALLOWLIST = {
    ".go", ".js", ".ts", ".tsx", ".java", ".kt", ".py", ".rb", ".php",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".properties",
    ".proto", ".md", ".txt", ".sql", ".graphql", ".gql", ".xml", ".gradle",
}

PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 \-]*PRIVATE KEY-----.*?-----END [A-Z0-9 \-]*PRIVATE KEY-----",
    re.DOTALL,
)
AUTH_BEARER_RE = re.compile(r"(?i)(authorization:\s*bearer\s+)([A-Za-z0-9\-._~+/]+=*)")
SIMPLE_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|token|secret|password)\b\s*[:=]\s*([^\s'\"`]+)")

def redact(text: str) -> str:
    text = PRIVATE_KEY_BLOCK_RE.sub("[REDACTED_PRIVATE_KEY_BLOCK]", text)
    text = AUTH_BEARER_RE.sub(r"\1[REDACTED]", text)

    def _repl(m: re.Match) -> str:
        return f"{m.group(1)}=[REDACTED]"

    text = SIMPLE_SECRET_RE.sub(_repl, text)
    return text

def is_probably_binary(p: Path) -> bool:
    try:
        with p.open("rb") as f:
            chunk = f.read(4096)
        return b"\x00" in chunk
    except Exception:
        return True

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def relposix(base: Path, p: Path) -> str:
    return p.relative_to(base).as_posix()

def glob_match(path_posix: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path_posix, pattern)

# ============================================================
# Repo discovery
# ============================================================

def list_repo_files(repo: Path, ignore_dirs: set[str]) -> List[Path]:
    # Prefer tracked files to avoid vendor/build noise
    if (repo / ".git").exists():
        try:
            out = subprocess.check_output(
                ["git", "-C", str(repo), "ls-files", "-z"],
                stderr=subprocess.DEVNULL,
            )
            files: List[Path] = []
            for b in out.split(b"\x00"):
                if not b:
                    continue
                p = repo / b.decode("utf-8", errors="ignore")
                if p.is_file():
                    files.append(p)
            return files
        except Exception:
            pass

    files: List[Path] = []
    for root, dirs, filenames in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for fn in filenames:
            files.append(Path(root) / fn)
    return [p for p in files if p.is_file()]

# ============================================================
# Selection & packing
# ============================================================

@dataclass
class SelectionConfig:
    max_files: int = 180
    max_file_bytes: int = 220_000
    max_total_bytes: int = 2_800_000
    max_lines_head: int = 260
    max_lines_tail: int = 120

def file_score(repo: Path, p: Path) -> int:
    rp = relposix(repo, p)
    score = 0

    for g in SIGNAL_GLOBS:
        if glob_match(rp, g):
            score += 100
            break

    if p.name in ENTRYPOINT_NAME_HINTS:
        score += 60
    if any(h in rp for h in ENTRYPOINT_PATH_HINTS):
        score += 10

    low = rp.lower()
    if "openapi" in low or "swagger" in low or "asyncapi" in low:
        score += 80
    if low.endswith(".proto"):
        score += 80

    if any(x in low for x in ["/k8s/", "/kubernetes/", "/helm/", "/charts/", ".github/workflows"]):
        score += 70

    if "/test" in low or "/tests" in low:
        score -= 30

    return score

def should_consider(repo: Path, p: Path) -> bool:
    if p.suffix and p.suffix.lower() not in TEXT_EXT_ALLOWLIST:
        if p.name.startswith("Dockerfile") or p.name == "Makefile":
            return True
        return False
    if is_probably_binary(p):
        return False
    return True

def select_files(repo: Path, files: List[Path], cfg: SelectionConfig) -> List[Path]:
    candidates = [p for p in files if should_consider(repo, p)]
    scored = [(file_score(repo, p), p) for p in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    picked: List[Path] = []
    for s, p in scored:
        if s <= 0 and len(picked) >= int(cfg.max_files * 0.65):
            break
        picked.append(p)
        if len(picked) >= cfg.max_files:
            break

    # Ensure README presence if available
    readmes = [p for p in candidates if p.name.lower().startswith("readme")]
    for r in readmes[:2]:
        if r not in picked:
            picked.insert(0, r)

    # Ensure common infra files if present
    for must in ["Dockerfile", "docker-compose.yml", "docker-compose.yaml"]:
        for p in candidates:
            if p.name == must and p not in picked:
                picked.insert(0, p)

    # Deduplicate (keep order)
    seen = set()
    out: List[Path] = []
    for p in picked:
        rp = relposix(repo, p)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out

def read_text_sample(p: Path, cfg: SelectionConfig) -> str:
    raw = p.read_bytes()
    if len(raw) <= cfg.max_file_bytes:
        return raw.decode("utf-8", errors="ignore")

    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    head = lines[:cfg.max_lines_head]
    tail = lines[-cfg.max_lines_tail:] if len(lines) > (cfg.max_lines_head + cfg.max_lines_tail) else []
    sampled = "\n".join(head) + "\n\n[...TRUNCATED...]\n\n" + "\n".join(tail)
    return sampled

def build_signal_pack(repo: Path, selected: List[Path], cfg: SelectionConfig) -> Tuple[str, str]:
    parts: List[str] = []
    total = 0

    parts.append("\n".join([
        f"REPO: {repo.name}",
        f"PATH: {repo}",
        "NOTE: This is a SIGNAL PACK (selected high-signal files only).",
        "-----",
    ]))

    for p in selected:
        rel = relposix(repo, p)
        try:
            content = read_text_sample(p, cfg)
        except Exception:
            continue

        content = redact(content)
        b = content.encode("utf-8", errors="ignore")
        total += len(b)
        if total > cfg.max_total_bytes:
            parts.append("\n[PACK_LIMIT_REACHED]\n")
            break

        parts.append(f"\n===== FILE: {rel} =====\n{content}\n")

    pack_text = "\n".join(parts)
    pack_hash = sha256_bytes(pack_text.encode("utf-8", errors="ignore"))
    return pack_text, pack_hash

# ============================================================
# Ollama client
# ============================================================

def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.0,
    timeout_s: int = 900,
    num_predict: int = 4096,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
        "stream": False,
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(f"{base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

# ============================================================
# JSON sanitizing / repair
# ============================================================

CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)
LINE_COMMENT_RE = re.compile(r"(?m)^\s*//.*?$")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

def strip_code_fences(text: str) -> str:
    return CODE_FENCE_RE.sub("", text).strip()

def strip_json_comments(text: str) -> str:
    text = BLOCK_COMMENT_RE.sub("", text)
    text = LINE_COMMENT_RE.sub("", text)
    return text

def remove_trailing_commas(text: str) -> str:
    prev = None
    while prev != text:
        prev = text
        text = TRAILING_COMMA_RE.sub(r"\1", text)
    return text

def extract_first_json_value(text: str) -> Optional[str]:
    text = text.strip()
    start_obj = text.find("{")
    start_arr = text.find("[")
    if start_obj == -1 and start_arr == -1:
        return None

    if start_obj == -1 or (start_arr != -1 and start_arr < start_obj):
        start = start_arr
        open_c, close_c = "[", "]"
    else:
        start = start_obj
        open_c, close_c = "{", "}"

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None

def best_effort_json_text(raw: str) -> Optional[str]:
    s = strip_code_fences(raw)
    s = strip_json_comments(s).strip()
    extracted = extract_first_json_value(s)
    if extracted:
        s = extracted
    s = remove_trailing_commas(s)
    return s if s else None

def parse_or_repair_json(raw: str, *, ollama_base: str, model: str) -> Optional[dict]:
    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # sanitize parse
    cleaned = best_effort_json_text(raw)
    if cleaned:
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    # local LLM repair
    repaired = ollama_chat(
        ollama_base,
        model,
        JSON_REPAIR_SYSTEM,
        "INPUT_TEXT:\n" + raw,
        temperature=0.0,
        num_predict=4096,
    )
    repaired_clean = best_effort_json_text(repaired) or repaired
    try:
        return json.loads(repaired_clean)
    except Exception:
        return None

# ============================================================
# Catalog validation
# ============================================================

def validate_catalog(root: Path, repos: list[dict]) -> Tuple[List[Tuple[str, Path]], List[str]]:
    errors: List[str] = []
    ok: List[Tuple[str, Path]] = []

    if not root.exists():
        errors.append(f'Catalog root does not exist: "{root}"')
        return ok, errors
    if not root.is_dir():
        errors.append(f'Catalog root is not a directory: "{root}"')
        return ok, errors

    seen_names: set[str] = set()
    seen_paths: set[Path] = set()

    for i, r in enumerate(repos):
        name = r.get("name")
        path = r.get("path")

        if not name or not isinstance(name, str):
            errors.append(f"repos[{i}].name is missing or not a string")
            continue
        if not path or not isinstance(path, str):
            errors.append(f"repos[{i}].path is missing or not a string (repo={name})")
            continue

        if name in seen_names:
            errors.append(f'Duplicate repo name "{name}"')
        seen_names.add(name)

        repo_abs = (root / path).expanduser().resolve()

        # Safety: ensure repo_abs is under root
        try:
            repo_abs.relative_to(root)
        except ValueError:
            errors.append(f'Repo "{name}" path escapes root: "{path}" -> "{repo_abs}" (root="{root}")')
            continue

        if repo_abs in seen_paths:
            errors.append(f'Duplicate repo path "{repo_abs}" (repo={name})')
        seen_paths.add(repo_abs)

        if not repo_abs.exists():
            errors.append(f'Repo folder not found for "{name}": "{repo_abs}"')
            continue
        if not repo_abs.is_dir():
            errors.append(f'Repo path is not a directory for "{name}": "{repo_abs}"')
            continue

        ok.append((name, repo_abs))

    return ok, errors

# ============================================================
# Markdown summary (no extra LLM calls)
# ============================================================

def write_arch_md(out_path: Path, profile: dict) -> None:
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

# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Signal-pack C4 generator using local Ollama")
    ap.add_argument("--catalog", required=True, help="Path to repos.yaml")
    ap.add_argument("--out", default="architecture-out", help="Output directory")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default=os.environ.get("MODEL", "qwen2.5-coder:7b-instruct"), help="Ollama model name")
    ap.add_argument("--max-files", type=int, default=180)
    ap.add_argument("--max-pack-bytes", type=int, default=2_800_000)
    ap.add_argument("--max-file-bytes", type=int, default=220_000)
    ap.add_argument("--num-predict", type=int, default=4096, help="Max tokens to generate per LLM call")
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

    cfg = SelectionConfig(
        max_files=args.max_files,
        max_total_bytes=args.max_pack_bytes,
        max_file_bytes=args.max_file_bytes,
    )

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for name, repo_path in ok_repos:
        repo_out = out_root / "repos" / name
        repo_out.mkdir(parents=True, exist_ok=True)

        pack_path = repo_out / "signal-pack.txt"
        profile_path = repo_out / "repo-profile.json"
        dsl_path = repo_out / "workspace.dsl"
        md_path = repo_out / "ARCHITECTURE.md"
        cache_path = repo_out / ".cache.json"

        files = list_repo_files(repo_path, DEFAULT_IGNORE_DIRS)
        selected = select_files(repo_path, files, cfg)
        pack_text, pack_hash = build_signal_pack(repo_path, selected, cfg)

        # Cache by pack hash
        if cache_path.exists() and profile_path.exists() and dsl_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding="utf-8"))
                if cache.get("pack_hash") == pack_hash:
                    print(f"[CACHE] {name}")
                    continue
            except Exception:
                pass

        pack_path.write_text(pack_text, encoding="utf-8")

        # Pass A: repo profile JSON
        print(f"[LLM A] {name}")
        user_a = f"repo.name={name}\nrepo.path={repo_path}\n\nSIGNAL_PACK:\n{pack_text}"
        profile_raw = ollama_chat(
            args.ollama,
            args.model,
            REPO_PROFILE_SYSTEM,
            user_a,
            temperature=0.0,
            num_predict=args.num_predict,
        )

        profile = parse_or_repair_json(profile_raw, ollama_base=args.ollama, model=args.model)
        if not profile:
            (repo_out / "repo-profile.raw.txt").write_text(profile_raw, encoding="utf-8")
            print(f"[WARN] JSON parse failed for {name}; saved repo-profile.raw.txt", file=sys.stderr)
            continue

        profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

        # Pass B: Structurizr DSL
        print(f"[LLM B] {name}")
        user_b = "REPO_PROFILE_JSON:\n" + json.dumps(profile, indent=2, ensure_ascii=False)
        dsl = ollama_chat(
            args.ollama,
            args.model,
            STRUCTURIZR_SYSTEM,
            user_b,
            temperature=0.0,
            num_predict=args.num_predict,
        )
        dsl_path.write_text(dsl, encoding="utf-8")

        # Local markdown summary
        write_arch_md(md_path, profile)

        cache_path.write_text(json.dumps({"pack_hash": pack_hash}, indent=2), encoding="utf-8")
        print(f"[OK] {name} -> {repo_out}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
