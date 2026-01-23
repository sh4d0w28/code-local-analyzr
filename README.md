# Local Architecture / C4 Generator (Ollama + local LLMs) — macOS (Apple Silicon)

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

## 2 Pull a model

```bash
ollama pull codex-v2:latest
```

Sanity test:
```bash
ollama run codex-v2:latest "Say OK"
```

Replace `codex-v2:latest` with the model you installed if different.

---

## 3 Context length configuration (important)

Ollama’s default context length is 4096 tokens. This runner now defaults to a large `num_ctx` and budgets prompts to stay within it.
Make sure the server allows that context length, or lower `--num-ctx`.

### Global (server default)
```bash
OLLAMA_CONTEXT_LENGTH=163840 ollama serve
```

### Per-request
This project’s script passes `num_ctx` in API calls:
```bash
--num-ctx 163840
```

Notes:
- Larger `num_ctx` requires more memory/VRAM. If you hit OOM, reduce `--num-ctx` or `--num-predict`.
- Prompt budgeting (`--respect-num-ctx`) is enabled by default; use `--no-respect-num-ctx` to disable.

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

Copy/paste run (Codex v2, max defaults; prompt budgeting on):
```bash
MODEL=codex-v2:latest \
python run_local_c4.py \
  --catalog repos.yaml \
  --out architecture-out \
  --classify-files \
  --mermaid \
  --verbose
```

Make sure Ollama is running with `OLLAMA_CONTEXT_LENGTH=163840` (or higher), or lower `--num-ctx`.

If you want a lighter run, drop `--classify-files` and/or `--mermaid`.
If you need to override defaults, add flags like `--num-ctx`, `--num-predict`, or `--no-respect-num-ctx`.

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

Fast run (small model) with local imports:
```bash
PYTHONPATH=. MODEL=qwen2.5-coder:7b-instruct python run_local_c4.py --catalog repos.yaml --verbose
```

---

## 8 Keep the model loaded (optional)

If your runs pause long enough for Ollama to unload the model, you can pin it:

One-shot keep-alive (forever):
```bash
python keep_ollama_loaded.py --model codex-v2:latest --keep-alive -1
```

Or refresh every 4 minutes:
```bash
python keep_ollama_loaded.py --model codex-v2:latest --keep-alive 10m --interval 240
```

You can also set it at the server level:
```bash
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

---

## 9 Analysis steps

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

## 10 Prompt flow

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

## 11 CLI reference

All flags (from `python run_local_c4.py --help`):
- `--catalog <path>`: Required. Path to `repos.yaml`.
- `--out <dir>`: Output directory (default: `architecture-out`).
- `--ollama <url>`: Ollama base URL (default: `http://localhost:11434`).
- `--model <name>`: Ollama model name (default: `MODEL` env or `deepseek-coder-v2:latest`).
- `--timeout <seconds>`: HTTP timeout per request (default: `7200`).
- `--num-ctx <tokens>`: Context window for LLM requests (default: `163840`).
- `--num-predict <tokens>`: Max tokens to generate per request (default: `32768`).
- `--respect-num-ctx`: Scale evidence and classification payloads to stay within `num_ctx` (default: on).
- `--no-respect-num-ctx`: Disable `num_ctx` prompt budgeting.
- `--ctx-bytes-per-token <float>`: Estimated bytes per token for budgeting (default: `4.0`).
- `--ctx-reserve-tokens <tokens>`: Extra tokens reserved as headroom (default: `512`).
- `--max-step-bytes <bytes>`: Max total evidence bytes per step (default: `6000000`).
- `--max-file-bytes <bytes>`: Max bytes per file excerpt in evidence (default: `800000`).
- `--max-snippets-per-file <count>`: Max regex snippets per file (default: `14`).
- `--snippet-context-lines <count>`: Context lines around regex hits (default: `14`).
- `--chunk-large-files`: Split large files into sequential chunks for evidence (default: off).
- `--max-files-per-step <count>`: Hard cap per step before evidence build (default: `800`).
- `--classify-files`: Use LLM file catalog to select step files (default: off).
- `--classify-files-model <name>`: Ollama model name for file classification (default: `CLASSIFY_FILES_MODEL` env or `qwen2.5-coder:7b-instruct`).
- `--classify-max-file-bytes <bytes>`: Max bytes sent per file to classifier (default: `200000`).
- `--routes-profile`: Extract routes and append to routing evidence (default: on).
- `--no-routes-profile`: Disable route extraction.
- `--routes-max-file-bytes <bytes>`: Max bytes per file for route extraction (default: `200000`).
- `--mermaid`: Generate Mermaid C4 markdown (default: off).
- `--verbose`: Log file selection, progress, and prompt sizes (default: off).
- `--skip-aggregate`: Skip merged workspace generation (default: off).
- `--render-only`: Generate `workspace.dsl` and optional Mermaid output from existing `repo-profile.json` without re-analyzing repositories.

Notes:
- Set `MODEL` env var to override the default model without passing `--model`.
- Set `CLASSIFY_FILES_MODEL` env var to override the default classifier model without passing `--classify-files-model`.
- Use `C4_SPEC_PATH=/path/to/source_of_truth.yaml` to load a custom spec.

---

## 12 Outputs

For each repo:
```
architecture-out/
  dsl/
    <repo-name>/
      <repo-name>.dsl
      <repo-name>View.dsl
    workspace_full.dsl
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

## 13 Performance tips for huge repos

- Increase analysis depth by raising step budgets:
  - `--max-step-bytes` (evidence per step)
  - `--max-files-per-step`
- For very large files, use `--chunk-large-files` to include sequential pieces; chunk boundaries log with `--verbose`.
- Keep ignores aggressive (default ignores build outputs like `dist/`, `build/`, `target/`, `node_modules/` and compiled artifacts like `*.class`, `*.jar`).
- If a repo is a monorepo, list sub-repos/modules as separate entries in `repos.yaml`.
- File catalogs are cached by size/mtime; delete `file-catalog.jsonl` to force reclassification.

---

## 14 Troubleshooting

### “JSON parse failed … raw saved”
The script automatically tries to sanitize and repair JSON via a second local LLM call.
If it still fails:
- reduce step evidence size: `--max-step-bytes 800000`
- increase generation: `--num-predict 8192`

### “Ollama not reachable”
Ollama API base URL is typically:
- `http://localhost:11434`

If needed, start:
```bash
ollama serve
```
