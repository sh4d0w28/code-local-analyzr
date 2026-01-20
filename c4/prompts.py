"""Load prompt templates from configuration."""

from .config import get_prompts

_PROMPTS = get_prompts()

PROFILE_INIT_SYSTEM = _PROMPTS["profile_init_system"]
PROFILE_UPDATE_SYSTEM = _PROMPTS["profile_update_system"]
STRUCTURIZR_SYSTEM = _PROMPTS["structurizr_system"]
JSON_REPAIR_SYSTEM = _PROMPTS["json_repair_system"]
FILE_CLASSIFY_SYSTEM = _PROMPTS.get("file_classify_system")
FILE_CLASSIFY_REPAIR_SYSTEM = _PROMPTS.get("file_classify_repair_system")
MERMAID_C4_SYSTEM = _PROMPTS.get("mermaid_c4_system")
