"""Render Structurizr view DSL from repo outputs."""

from __future__ import annotations

import re
from typing import Optional, Tuple


_ID_RX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ASSIGN_RX = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*softwareSystem\s+\"([^\"]+)\"",
    re.MULTILINE,
)
_ASSIGN_NO_LABEL_RX = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*softwareSystem\b",
    re.MULTILINE,
)
_BARE_RX = re.compile(
    r"^\s*softwareSystem\s+([A-Za-z_][A-Za-z0-9_]*)\s+\"([^\"]+)\"",
    re.MULTILINE,
)
_BARE_NO_LABEL_RX = re.compile(
    r"^\s*softwareSystem\s+([A-Za-z_][A-Za-z0-9_]*)\b",
    re.MULTILINE,
)
_NAME_ONLY_RX = re.compile(r"^\s*softwareSystem\s+\"([^\"]+)\"", re.MULTILINE)


def slugify(text: str) -> str:
    """Convert arbitrary text into a DSL-safe identifier."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    if not slug:
        slug = "system"
    if slug[0].isdigit():
        slug = f"_{slug}"
    return slug


def title_case_id(system_id: str) -> str:
    """Render a display-friendly PascalCase name from an identifier."""
    if "_" in system_id or "-" in system_id:
        parts = re.split(r"[_-]+", system_id)
        return "".join(p[:1].upper() + p[1:] for p in parts if p)
    return system_id[:1].upper() + system_id[1:]


def extract_system_id_and_label(dsl_text: str, fallback_name: str) -> Tuple[str, str]:
    """Extract system id/label from DSL text, with fallbacks."""
    system_id: Optional[str] = None
    label: Optional[str] = None

    m = _ASSIGN_RX.search(dsl_text)
    if m:
        system_id, label = m.group(1), m.group(2)
    if system_id is None:
        m = _BARE_RX.search(dsl_text)
        if m:
            system_id, label = m.group(1), m.group(2)
    if system_id is None:
        m = _ASSIGN_NO_LABEL_RX.search(dsl_text)
        if m:
            system_id = m.group(1)
    if system_id is None:
        m = _BARE_NO_LABEL_RX.search(dsl_text)
        if m:
            system_id = m.group(1)
    if label is None:
        m = _NAME_ONLY_RX.search(dsl_text)
        if m:
            label = m.group(1)

    if system_id is None:
        system_id = slugify(label or fallback_name)

    if label is None or not label.strip():
        label = fallback_name or system_id

    # Ensure the identifier is valid even if the DSL was malformed.
    if not _ID_RX.fullmatch(system_id):
        system_id = slugify(system_id)

    return system_id, label


def render_views(system_id: str, system_label: str) -> str:
    """Render a system context + container view DSL snippet."""
    view_base = title_case_id(system_id)
    context_name = f"{view_base}Context"
    container_name = f"{view_base}Containers"
    label = system_label or system_id

    return (
        f'systemcontext {system_id} "{context_name}" {{\n'
        "    include *\n"
        "    autoLayout\n"
        f'    description "The system context diagram for {label}."\n'
        "    properties {\n"
        "        structurizr.groups false\n"
        "    }\n"
        "}\n\n"
        f'container {system_id} "{container_name}" {{\n'
        "    include *\n"
        "    autoLayout\n"
        f'    description "The container diagram for {label}."\n'
        "}\n"
    )
