# Local Architecture / C4 Generator (Ollama + DeepSeek-Coder-V2) — macOS (Apple Silicon)

This setup runs an LLM **locally** via Ollama and analyzes your repositories **incrementally (request-by-request)** so very large repos (multi‑GB) still work.

---

## 0 Requirements

- macOS **14 (Sonoma) or newer**.
- Apple Silicon (M-series) recommended.
- Enough disk space for models (can be tens to hundreds of GB).

---

## 1 Install Ollama (macOS)

### Option A — Ollama app (recommended)
1. Download the `ollama.dmg`.
2. Mount it and **drag Ollama to Applications**.
3. Start Ollama once; it will ensure `ollama` CLI is available in your PATH (may prompt to create a link in `/usr/local/bin`).

Verify:
```bash
ollama --version
```

### Option B — CLI only (if you prefer)
If you installed via package manager, make sure the server is running:
```bash
ollama serve
```

---

## 2 Pull DeepSeek-Coder-V2

```bash
ollama pull deepseek-coder-v2:latest
```

Sanity test:
```bash
ollama run deepseek-coder-v2:latest "Say OK"
```

---

## 3 Context length configuration (important)

Ollama’s default context length is 4096 tokens. You can override it globally via env var when serving, or per request using `num_ctx`.

### Global (server default)
```bash
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

### Per-request
This project’s script passes `num_ctx` in API calls:
```bash
--num-ctx 32768
```

Notes:
- Larger `num_ctx` requires more memory/VRAM.
- DeepSeek-Coder-V2 supports large context windows; still start with 32K and increase only if needed.

---

## 4 Create a repo catalog

Create `repos.yaml`:

```yaml
root: /ABS/PATH/TO/YOUR/WORKDIR
repos:
  - name: web-site-fe
    path: proj/site/fe
  - name: web-site-be
    path: proj/site/be
```

The tool validates that all folders exist before doing any LLM calls.

---

## 5 Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pyyaml httpx
```

---

## 6 Configure prompts and scan rules (optional)

All prompts, scan rules, step categories, and output filename templates live in:

```
source_of_truth.yaml
```

Edit this file to adjust what gets scanned and how outputs are named.
You can also override the config path with `C4_SPEC_PATH=/path/to/file.yaml`.

---

## 7 Run

Maxed-out run (deepseek-coder-v2, context length 163840 per `ollama show`, memaid):
```bash
MODEL=deepseek-coder-v2:latest \
python run_local_c4.py \
  --catalog repos.yaml \
  --out architecture-out \
  --num-ctx 163840 \
  --num-predict 32768 \
  --max-step-bytes 6000000 \
  --max-file-bytes 800000 \
  --max-files-per-step 800 \
  --timeout 7200 \
  --classify-files \
  --classify-max-file-bytes 200000 \
  --mermaid \
  --verbose
```

Full inventory mode (LLM categorizes every text file; slow but thorough).
Add these flags to any run above:
```bash
  --classify-files \
  --classify-max-file-bytes 200000
```

Optional add-ons:
```bash
  --mermaid
```

Fast run (Codex latest) with local imports:
```bash
PYTHONPATH=. MODEL=qwen2.5-coder:7b-instruct python run_local_c4.py --catalog repos.yaml --verbose
```

---

## 8 Analysis steps

The analyzer runs the following steps in order (configured in `source_of_truth.yaml`):
- `01_docs_infra`: docs, infra, CI/CD, and deployment descriptors to establish system context.
- `02_build_deps`: build files, manifests, and API specs to detect languages, tools, and dependencies.
- `03_entrypoints`: app bootstrap files to identify services, runtimes, and startup wiring.
- `04_routing_api`: routers/controllers/handlers to map API surface and protocol types.
- `05_deps_datastores`: clients/repositories/producers/consumers for databases, brokers, and outbound deps.
- `06_configs`: config/env templates for ports, hosts, and integration endpoints.

Add `--verbose` to log per-file selection and LLM prompt sizes.
Add `--classify-files` to build step inputs from an LLM file catalog instead of globs/keywords.
Classification logs progress as it scans.
Routes are extracted by default; use `--no-routes-profile` to disable.
Add `--mermaid` to generate a Mermaid C4 markdown file per repo.

---

## 9 Prompt flow

Here is the high-level prompt flow and what each step produces:
- Optional file classification: `FILE_CLASSIFY_SYSTEM` tags each text file using head/tail content -> `file-catalog.jsonl` (cached by size/mtime/model).
- Route extraction (regex, no LLM): routes are collected from code/proto -> `routes.jsonl` and appended to routing evidence.
- Step 01 init: `PROFILE_INIT_SYSTEM` + `01_docs_infra.evidence.txt` -> `01_docs_infra.profile.json`.
- Step 02 update: `PROFILE_UPDATE_SYSTEM` + `02_build_deps.evidence.txt` -> `02_build_deps.profile.json`.
- Step 03 update: `PROFILE_UPDATE_SYSTEM` + `03_entrypoints.evidence.txt` -> `03_entrypoints.profile.json`.
- Step 04 update: `PROFILE_UPDATE_SYSTEM` + `04_routing_api.evidence.txt` (+ routes profile) -> `04_routing_api.profile.json`.
- Step 05 update: `PROFILE_UPDATE_SYSTEM` + `05_deps_datastores.evidence.txt` -> `05_deps_datastores.profile.json`.
- Step 06 update: `PROFILE_UPDATE_SYSTEM` + `06_configs.evidence.txt` -> `06_configs.profile.json`.
- JSON repair: if any profile step fails to parse, `JSON_REPAIR_SYSTEM` is invoked and the raw LLM output is saved.
- Final outputs: `STRUCTURIZR_SYSTEM` -> `workspace.dsl`; optional `MERMAID_C4_SYSTEM` -> `workspace.mermaid.md` (sanitized post-process).

---

## 10 CLI reference

All flags (from `python run_local_c4.py --help`):
- `--catalog <path>`: Required. Path to `repos.yaml`.
- `--out <dir>`: Output directory (default: `architecture-out`).
- `--ollama <url>`: Ollama base URL (default: `http://localhost:11434`).
- `--model <name>`: Ollama model name (default: `MODEL` env or `deepseek-coder-v2:latest`).
- `--timeout <seconds>`: HTTP timeout per request (default: `1200`).
- `--num-ctx <tokens>`: Context window for LLM requests (default: `32768`).
- `--num-predict <tokens>`: Max tokens to generate per request (default: `4096`).
- `--max-step-bytes <bytes>`: Max total evidence bytes per step (default: `1800000`).
- `--max-file-bytes <bytes>`: Max bytes per file excerpt in evidence (default: `220000`).
- `--max-snippets-per-file <count>`: Max regex snippets per file (default: `14`).
- `--snippet-context-lines <count>`: Context lines around regex hits (default: `14`).
- `--max-files-per-step <count>`: Hard cap per step before evidence build (default: `260`).
- `--classify-files`: Use LLM file catalog to select step files (default: off).
- `--classify-max-file-bytes <bytes>`: Max bytes sent per file to classifier (default: `120000`).
- `--routes-profile`: Extract routes and append to routing evidence (default: on).
- `--no-routes-profile`: Disable route extraction.
- `--routes-max-file-bytes <bytes>`: Max bytes per file for route extraction (default: `120000`).
- `--mermaid`: Generate Mermaid C4 markdown (default: off).
- `--verbose`: Log file selection, progress, and prompt sizes (default: off).
- `--skip-aggregate`: Skip merged workspace generation (default: off).

Notes:
- Set `MODEL` env var to override the default model without passing `--model`.
- Use `C4_SPEC_PATH=/path/to/source_of_truth.yaml` to load a custom spec.

---

## 11 Outputs

For each repo:
```
architecture-out/
  repos/<repo-name>/
    file-catalog.jsonl
    routes.jsonl
    workspace.mermaid.md
    steps/
      01_docs_infra.evidence.txt
      01_docs_infra.sources.txt
      01_docs_infra.profile.json
      02_build_deps.evidence.txt
      02_build_deps.sources.txt
      02_build_deps.profile.json
      03_entrypoints.evidence.txt
      03_entrypoints.sources.txt
      03_entrypoints.profile.json
      04_routing_api.evidence.txt
      04_routing_api.sources.txt
      04_routing_api.profile.json
      05_deps_datastores.evidence.txt
      05_deps_datastores.sources.txt
      05_deps_datastores.profile.json
      06_configs.evidence.txt
      06_configs.sources.txt
      06_configs.profile.json
    repo-profile.json
    workspace.dsl
    ARCHITECTURE.md
  workspace.full.dsl
```

- `repo-profile.json`: normalized architecture facts for the repo
- `workspace.dsl`: Structurizr DSL (C4 Context + Container)
- `ARCHITECTURE.md`: human-readable summary
- `file-catalog.jsonl`: per-file LLM categories and summaries (when `--classify-files` is enabled)
- `routes.jsonl`: extracted routes (default unless `--no-routes-profile` is used)
- `workspace.mermaid.md`: Mermaid C4 markdown (when `--mermaid` is enabled)
- `workspace.full.dsl`: merged Structurizr DSL across all repos (shared infra de-duplicated)

You can also build the merged DSL without rerunning analysis:
```bash
python aggregate_dsl.py --out architecture-out
```

---

## 12 Performance tips for huge repos

- Increase analysis depth by raising step budgets:
  - `--max-step-bytes` (evidence per step)
  - `--max-files-per-step`
- Keep ignores aggressive (default ignores build outputs like `dist/`, `build/`, `target/`, `node_modules/` and compiled artifacts like `*.class`, `*.jar`).
- If a repo is a monorepo, list sub-repos/modules as separate entries in `repos.yaml`.
- File catalogs are cached by size/mtime; delete `file-catalog.jsonl` to force reclassification.

---

## 13 Troubleshooting

### “JSON parse failed … raw saved”
The script automatically tries to sanitize and repair JSON via a second local LLM call.
If it still fails:
- reduce step evidence size: `--max-step-bytes 800000`
- increase generation: `--num-predict 8192`

### “Ollama not reachable”
Ollama API base URL is typically:
- `http://localhost:11434/api`

If needed, start:
```bash
ollama serve
```
