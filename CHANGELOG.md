# Changelog

## Unreleased
- Centralized prompts, scan rules, step categories, and output templates in `source_of_truth.yaml`.
- Added per-step `*.sources.txt` outputs listing input files and sizes.
- Expanded scan coverage for build/config files and CI/CD metadata; fixed entrypoint keyword matching.
- Added optional LLM file cataloging (`--classify-files`) to drive step inputs from per-file categories.
- Added `--classify-files-model` (and `CLASSIFY_FILES_MODEL`) to choose a separate classifier model.
- Added optional route extraction (`--routes-profile`) and Mermaid C4 output (`--mermaid`).
- Normalized repo path/name in profiles to the real catalog values.
- Added proto/gRPC route extraction and classification progress logging.
- Added per-request prompt size logging when `--verbose` is enabled.
- Routes are now extracted by default (use `--no-routes-profile` to disable).
- Added Mermaid post-processing to fix invalid C4 syntax and route label braces.
- Normalized Mermaid Container/Component diagrams to enforce system/container boundaries.
- Mermaid container diagrams now include data stores and outbound dependencies from the profile.
