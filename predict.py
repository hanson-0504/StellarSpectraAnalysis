"""Unified prediction dispatcher.

Chooses Random-Forest or Neural-Network prediction based on available model
files ('.joblib' for RF, '.keras' for NN) or an explicit `kind` argument.
This module delegates to `predict_rf.py` and `predict_nn.py` if present.

Usage (programmatic):
    from predict import predict
    summary = predict(kind="auto", model_dir="data/models/", ...)

This function returns whatever the underlying predictor returns; if that
returns nothing, we provide a small summary dict for agents.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Lazy imports so the module can be imported even if only one backend exists
try:
    import predict_rf as _rf
except Exception:  # noqa: BLE001
    _rf = None  # type: ignore

try:
    import predict_nn as _nn
except Exception:  # noqa: BLE001
    _nn = None  # type: ignore


def _scan_models(model_dir: str) -> Tuple[List[Path], List[Path], float, float]:
    """Return (rf_files, nn_files, latest_rf_mtime, latest_nn_mtime)."""
    p = Path(model_dir)
    rf_files: List[Path] = sorted(p.glob("*.joblib"))
    nn_files: List[Path] = sorted(p.glob("*.keras"))
    latest = lambda xs: max((x.stat().st_mtime for x in xs), default=0.0)
    return rf_files, nn_files, latest(rf_files), latest(nn_files)


def _predict_rf(**kw: Any) -> Any:
    if _rf is None:
        raise ImportError("predict_rf.py is not available")
    # Prefer a clean run() wrapper if provided
    if hasattr(_rf, "run"):
        return _rf.run(**kw)
    # Fallback to older API
    if hasattr(_rf, "load_and_predict"):
        return _rf.load_and_predict(**kw)
    raise AttributeError("predict_rf has no run() or load_and_predict()")


def _predict_nn(**kw: Any) -> Any:
    if _nn is None:
        raise ImportError("predict_nn.py is not available")
    if hasattr(_nn, "run"):
        return _nn.run(**kw)
    if hasattr(_nn, "load_and_predict"):
        return _nn.load_and_predict(**kw)
    raise AttributeError("predict_nn has no run()/load_and_predict()/main()")


def predict(
    kind: str = "auto",
    *,
    fits_dir: Optional[str] = None,
    labels_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    config_path: str = "config.yaml",
) -> Any:
    """Top-level prediction entrypoint.

    Parameters
    ----------
    kind : {"auto", "rf", "nn"}
        Which predictor to use. "auto" inspects `model_dir` and prefers the
        model family whose newest file is most recent (falls back sensibly if
        only one family exists).
    fits_dir, labels_dir, model_dir, output_dir, config_path : str
        Usual IO arguments passed through to the underlying backend.

    Returns
    -------
    Any
        Whatever the backend returns. If it returns None, we synthesize a small
        summary dict for agent logging.
    """
    model_dir = model_dir or "data/models/"

    rf_files, nn_files, lrf, lnn = _scan_models(model_dir)
    if kind.lower() == "auto":
        if not rf_files and not nn_files:
            raise FileNotFoundError(f"No model files found in {model_dir} (.joblib or .keras)")
        # Prefer newer family; if only one exists, use it.
        chosen = "nn" if (nn_files and (lnn >= lrf or not rf_files)) else "rf"
    else:
        chosen = kind.lower()
        if chosen not in {"rf", "nn"}:
            raise ValueError("kind must be one of {'auto','rf','nn'}")

    logging.info(
        "[predict] kind=%s | rf=%d (latest=%s) nn=%d (latest=%s)",
        chosen, len(rf_files), lrf, len(nn_files), lnn,
    )

    # Optional hint env for downstream scripts (harmless if unused)
    os.environ["MODEL_KIND"] = chosen

    kw = dict(
        fits_dir=fits_dir,
        labels_dir=labels_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        config=config_path if "config" in (getattr(_rf, "run", None) or getattr(_nn, "run", None) or {}).__class__.__name__.lower() else config_path,
        # Backends accept either config or config_path; they ignore unknown kwargs.
        config_path=config_path,
    )

    try:
        result = _predict_nn(**kw) if chosen == "nn" else _predict_rf(**kw)
    except TypeError:
        # Some backends may not accept all kwargs; retry with a reduced set
        kw_fallback = dict(
            fits_dir=fits_dir,
            labels_dir=labels_dir,
            model_dir=model_dir,
            output_dir=output_dir,
            config_path=config_path,
        )
        result = _predict_nn(**kw_fallback) if chosen == "nn" else _predict_rf(**kw_fallback)

    if result is None:
        result = {
            "status": "ok",
            "model_kind": chosen,
            "n_rf": len(rf_files),
            "n_nn": len(nn_files),
            "model_dir": str(model_dir),
        }
    return result


if __name__ == "__main__":
    # Minimal CLI passthrough for quick checks
    import argparse

    parser = argparse.ArgumentParser(description="Unified prediction dispatcher")
    parser.add_argument("--kind", default="auto", choices=["auto", "rf", "nn"])
    parser.add_argument("--fits_dir", default=None)
    parser.add_argument("--labels_dir", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()

    out = predict(
        kind=args.kind,
        fits_dir=args.fits_dir,
        labels_dir=args.labels_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
    )
    print(out)