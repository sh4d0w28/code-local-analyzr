"""Load and validate the YAML configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml

CONFIG_ENV_VAR = "C4_SPEC_PATH"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "source_of_truth.yaml"


def _load_config(path: Path) -> Dict[str, Any]:
    """Read the YAML config from disk and validate its top-level type."""
    if not path.exists():
        raise RuntimeError(f'Config file not found: "{path}"')
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f'Failed to read config "{path}": {e}') from e
    if not isinstance(data, dict):
        raise RuntimeError(f'Config file "{path}" must be a YAML mapping at top level')
    return data


def _resolve_config_path() -> Path:
    """Resolve the config path from env override or default."""
    override = os.environ.get(CONFIG_ENV_VAR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_CONFIG_PATH


_CONFIG_PATH = _resolve_config_path()
_CONFIG = _load_config(_CONFIG_PATH)


def get_config_path() -> Path:
    """Expose the resolved config path for diagnostics."""
    return _CONFIG_PATH


def _require_section(name: str, kind: type) -> Any:
    """Fetch a required config section and validate its type."""
    value = _CONFIG.get(name)
    if not isinstance(value, kind):
        raise RuntimeError(f'Config section "{name}" missing or not a {kind.__name__}')
    return value


def get_prompts() -> Dict[str, str]:
    """Return prompt templates from the config."""
    prompts = _require_section("prompts", dict)
    return prompts


def get_steps_config() -> List[Dict[str, Any]]:
    """Return step definitions from the config."""
    steps = _require_section("steps", list)
    return steps


def get_scan_rules() -> Dict[str, Any]:
    """Return scan rules from the config."""
    scan = _require_section("scan", dict)
    return scan


def get_paths_config() -> Dict[str, Any]:
    """Return output path templates from the config."""
    paths = _require_section("paths", dict)
    return paths


def get_sources_config() -> Dict[str, Any]:
    """Return optional sources config or an empty mapping."""
    sources = _CONFIG.get("sources")
    if sources is None:
        return {}
    if not isinstance(sources, dict):
        raise RuntimeError('Config section "sources" must be a mapping')
    return sources
