"""Ollama chat client helper."""

from __future__ import annotations

from typing import Callable, Optional

import httpx


def ollama_chat(
    base_url: str,
    model: str,
    system: str,
    user: str,
    *,
    temperature: float,
    timeout_s: int,
    num_predict: int,
    num_ctx: int,
    log: Optional[Callable[[str], None]] = None,
    label: str = "request",
) -> str:
    """Send a single chat request to Ollama and return the response text."""
    if log is not None:
        system_chars = len(system)
        user_chars = len(user)
        system_bytes = len(system.encode("utf-8", errors="ignore"))
        user_bytes = len(user.encode("utf-8", errors="ignore"))
        prompt_chars = system_chars + user_chars
        prompt_bytes = system_bytes + user_bytes
        log(
            f"[LLM] {label} model={model} prompt_chars={prompt_chars} "
            f"prompt_bytes={prompt_bytes} num_ctx={num_ctx} num_predict={num_predict}"
        )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
        "stream": False,
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(f"{base_url}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()["message"]["content"]
