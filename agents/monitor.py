"""
Simple model monitor agent

Features:
- Watches a models/ directory for new/updated .joblib or .keras files
- Validates the newest model by loading it (extend `validate_model` as needed)
- Optional LLM-based chooser (disabled by default). Enable with USE_LLM=1
- Graceful shutdown with Ctrl+C

Usage:
  python agents/monitor.py --dir data/models --interval 10

Env vars:
  OPENAI_API_KEY   (required only if USE_LLM=1)
  USE_LLM=1        (optional; defaults to 0)
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

from joblib import load as load_joblib
from tensorflow.keras.models import load_model as load_keras
import importlib
import contextlib

# --- Optional: OpenAI client if you want LLM decision making ---
USE_LLM = os.getenv("USE_LLM", "0") == "1"
if USE_LLM:
    try:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI()
    except Exception as e:  # pragma: no cover
        print(f"[monitor] Failed to initialize OpenAI client: {e}")
        USE_LLM = False


def list_models(model_dir: Path) -> List[Path]:
    return sorted(model_dir.glob("*.joblib")) + sorted(model_dir.glob("*.keras"))


def newest_model(models: List[Path]) -> Optional[Path]:
    if not models:
        return None
    return max(models, key=lambda p: p.stat().st_mtime)


def choose_with_llm(models: List[Path]) -> Optional[Path]:
    """Ask the LLM to pick one *exact* path string from the provided list.
    Falls back to newest if parsing fails.
    """
    if not models:
        return None
    if not USE_LLM:
        return newest_model(models)

    try:
        msg_models = "\n".join(str(p) for p in models)
        resp = _openai_client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that selects exactly one file path from a provided list.\n"
                        "Return only the chosen path, with no extra text."
                    ),
                },
                {"role": "user", "content": f"Choose a model to validate from:\n{msg_models}"},
            ],
        )
        content = resp.choices[0].message.content.strip()
        # Ensure the response is one of the provided paths
        for p in models:
            if content == str(p):
                return p
        # If the model replied with just a filename, try to resolve it
        for p in models:
            if content == p.name:
                return p
        print("[monitor] LLM response did not match any path; falling back to newest.")
        return newest_model(models)
    except Exception as e:
        print(f"[monitor] LLM selection failed: {e}; falling back to newest.")
        return newest_model(models)


def validate_model(path: Path) -> bool:
    """Load the model to ensure it is readable. Extend with real checks.

    For `.keras` models, we first try loading with compile=False (fast path for inference-only).
    If that fails due to custom layers/losses/metrics, we attempt to import them from a
    local `neural_network` module and pass them via `custom_objects`.
    """
    try:
        if path.suffix == ".joblib":
            _ = load_joblib(path)
            return True
        elif path.suffix == ".keras":
            # Fast path: don't require custom objects if we don't need to compile
            try:
                _ = load_keras(path, compile=False)
                return True
            except Exception as e_first:
                # Try again with custom_objects if available
                custom_objects = {}
                try:
                    nn = importlib.import_module("neural_network")
                    # Common names to look for; extend as needed
                    for name in ("PhysicsLoss", "CustomLoss", "CustomMetric", "PhysicsMetric"):
                        if hasattr(nn, name):
                            custom_objects[name] = getattr(nn, name)
                    # Some models save with a registered name like "Custom>PhysicsLoss"
                    if "PhysicsLoss" in custom_objects:
                        custom_objects["Custom>PhysicsLoss"] = custom_objects["PhysicsLoss"]
                except Exception:
                    # If we cannot import the module or objects, fall back to raising the original error
                    pass

                if custom_objects:
                    try:
                        _ = load_keras(path, custom_objects=custom_objects, compile=False)
                        return True
                    except Exception as e_second:
                        raise e_second from e_first
                else:
                    raise e_first
        else:
            print(f"[monitor] Unsupported file type: {path.suffix}")
            return False
    except Exception as e:
        print(f"[monitor] Failed to load {path}: {e}")
        return False


def run_loop(model_dir: Path, interval: int) -> None:
    print(f"[monitor] Watching {model_dir.resolve()} every {interval}s (USE_LLM={'1' if USE_LLM else '0'})")
    processed: Optional[Path] = None
    running = True

    def _handle_sigint(sig, frame):
        nonlocal running
        print("\n[monitor] Shutting down...")
        running = False

    signal.signal(signal.SIGINT, _handle_sigint)

    while running:
        models = list_models(model_dir)
        target = choose_with_llm(models)
        if target and target != processed:
            ok = validate_model(target)
            status = "OK" if ok else "FAIL"
            print(f"[monitor] Validated: {target} -> {status}")
            processed = target
        time.sleep(interval)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a directory of .joblib/.keras models.")
    parser.add_argument("--dir", dest="model_dir", default="models", help="Directory to watch (default: models)")
    parser.add_argument("--interval", dest="interval", type=int, default=10, help="Polling interval seconds (default: 10)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run_loop(Path(args.model_dir), args.interval)
