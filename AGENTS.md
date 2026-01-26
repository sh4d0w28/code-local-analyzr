# AGENT Work Log

Purpose: keep a lightweight, append-only record of changes and progress in this repo.

## How to use
- Append newest updates at the top of the log.
- Keep entries short and factual.
- Prefer file references with paths.
- Avoid duplicating large diffs or code blocks.

## Current focus
- (fill in)

## Open questions
- (fill in)

## Recent changes (newest first)
- 2026-01-26 10:30 - Tightened entrypoint selection and filtering
  - Files: c4/runner.py, c4/profile_normalize.py
  - Notes: entrypoints step now filters by globs; command-like entrypoints kept, noisy class names dropped
- 2026-01-26 10:05 - Documented system logic in README
  - Files: README.md
  - Notes: added pipeline summary (selection, batching, merge, enrichment)
- 2026-01-26 09:55 - Added heuristics, coverage output, index generation, and improved normalization
  - Files: c4/heuristics.py, c4/profile_normalize.py, c4/output.py, c4/routes.py, aggregate_dsl.py, c4/index.py, c4/runner.py, README.md, c4/repo_scan.py
  - Notes: datastore/dependency hints, external/system relationships, skip generated files, coverage.json, architecture-out/README.md
- 2026-01-26 09:10 - Added lowmem batching mode and safer evidence truncation for C4 runs
  - Files: c4/runner.py, c4/repo_scan.py
  - Notes: new --analysis-mode lowmem/highmem, per-batch evidence/sources, content truncation
- YYYY-MM-DD HH:MM - (short summary)
  - Files: path/to/file.ext, path/to/other.ext
  - Notes: (optional)
