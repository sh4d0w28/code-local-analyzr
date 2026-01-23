#!/usr/bin/env python3
"""Keep an Ollama model loaded by sending keep_alive requests."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Union

import httpx


def _parse_keep_alive(value: str) -> Union[int, str]:
    """Parse keep_alive to int when possible, otherwise keep as string."""
    v = value.strip()
    if not v:
        raise ValueError("keep_alive must not be empty")
    if v.lstrip("-").isdigit():
        return int(v)
    return v


def _build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the request payload for Ollama."""
    keep_alive = _parse_keep_alive(args.keep_alive)
    if args.endpoint == "generate":
        return {
            "model": args.model,
            "prompt": args.prompt,
            "keep_alive": keep_alive,
            "stream": False,
        }
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": args.message}],
        "keep_alive": keep_alive,
        "stream": False,
    }


def _send_once(client: httpx.Client, args: argparse.Namespace) -> Dict[str, Any]:
    """Send one keep_alive request."""
    payload = _build_payload(args)
    r = client.post(f"{args.ollama}/api/{args.endpoint}", json=payload)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser(description="Keep an Ollama model loaded")
    ap.add_argument("--ollama", default="http://localhost:11434", help="Ollama base URL")
    ap.add_argument("--model", default=os.environ.get("MODEL", "codex-v2:latest"), help="Model name")
    ap.add_argument(
        "--keep-alive",
        default="-1",
        help='Keep-alive duration (e.g. "10m", "3600", or -1 for forever)',
    )
    ap.add_argument(
        "--endpoint",
        choices=("generate", "chat"),
        default="generate",
        help="Ollama API endpoint to hit",
    )
    ap.add_argument("--prompt", default="", help="Prompt for /api/generate")
    ap.add_argument("--message", default="", help="Message for /api/chat")
    ap.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Repeat interval in seconds (0 = send once and exit)",
    )
    ap.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    args = ap.parse_args()

    if args.interval < 0:
        print("[ERROR] --interval must be >= 0", file=sys.stderr)
        return 2

    with httpx.Client(timeout=args.timeout) as client:
        try:
            if args.interval == 0:
                _send_once(client, args)
                print(
                    f"[OK] keep_alive={args.keep_alive} model={args.model} endpoint={args.endpoint}"
                )
                return 0

            while True:
                _send_once(client, args)
                print(
                    f"[OK] keep_alive={args.keep_alive} model={args.model} endpoint={args.endpoint}"
                )
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[STOP] interrupted")
            return 130
        except Exception as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
