# agents/tools.py
import os
import time
import logging
from pathlib import Path

import yaml

# Import your project modules
import preprocess as pp
import train_model as tm
import predict as pr

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)

DIRS = CFG["directories"]

from joblib import load
from typing import Dict, List

def _dataset_profile():
    """Return (n_samples, n_features) from preprocessed flux if available; else (0,0)."""
    try:
        flux_path = os.path.join(DIRS["spectral"], "flux.joblib")
        X = load(flux_path)
        return int(getattr(X, "shape", [0, 0])[0]), int(getattr(X, "shape", [0, 0])[1])
    except Exception:
        return 0, 0

def _has_gpu():
    try:
        import tensorflow as tf
        return len(tf.config.list_physical_devices('GPU')) > 0
    except Exception:
        return False

# --- Model scanning helper ----------------------------------------------------

def _scan_models(model_dir: str) -> Dict[str, object]:
    """Scan model_dir for RF (.joblib) and NN (.keras) model files.
    Returns dict with lists and latest mtimes.
    """
    p = Path(model_dir)
    rf_files: List[Path] = sorted(p.glob("*.joblib"))
    nn_files: List[Path] = sorted(p.glob("*.keras"))
    latest_rf = max((f.stat().st_mtime for f in rf_files), default=0.0)
    latest_nn = max((f.stat().st_mtime for f in nn_files), default=0.0)
    return {
        "rf_files": rf_files,
        "nn_files": nn_files,
        "latest_rf": latest_rf,
        "latest_nn": latest_nn,
    }

def choose_model(strategy: str = "auto", n_threshold: int = 10000) -> str:
    """Heuristic selection between 'rf' and 'nn'.
    - If strategy is not 'auto', return it (normalized to 'rf'/'nn').
    - Else, prefer RF for small/medium tabular data; NN when very large and GPU present.
    """
    s = (strategy or "auto").lower()
    if s in {"rf", "random_forest"}:
        return "rf"
    if s in {"nn", "neural", "neural_network"}:
        return "nn"

    n_samples, n_features = _dataset_profile()
    gpu = _has_gpu()

    # Simple rules of thumb (tabular): RF shines on smaller N and when features>>samples; NN can win with very large N + GPU
    if n_samples >= n_threshold and gpu:
        return "nn"
    if n_features > 5 * max(1, n_samples):
        return "rf"
    return "rf"

def preprocess_spectra(fits_dir: str = None, labels_dir: str = None) -> dict:
    fits_dir = fits_dir or DIRS["spectral"]
    labels_dir = labels_dir or DIRS["labels"]
    logging.info(f"[tool] Preprocess: {fits_dir=} {labels_dir=}")
    # Directly call with keyword args
    pp.run(fits_dir=fits_dir, labels_dir=labels_dir, config_path="config.yaml")
    return {"status": "ok", "fits_dir": fits_dir, "labels_dir": labels_dir}

def train_models(strategy: str = "auto", large_threshold: int = 5000) -> dict:
    """Train either RF (train_model.py) or NN (train_nn_model.py) based on strategy.
    strategy: 'auto' | 'rf' | 'nn'
    """
    # If you want large_threshold directly:
    # model_choice = choose_model(strategy, n_threshold=large_threshold)
    model_choice = choose_model(strategy, n_threshold=large_threshold)
    logging.info(f"[tool] Train models: choice={model_choice}")
    if model_choice == "rf":
        try:
            tm.run(config_path="config.yaml", pipeline="auto", large_threshold=large_threshold)
        except AttributeError:
            tm.train_and_save_models()
        return {"status": "ok", "model": "rf"}
    else:
        try:
            import train_nn_model_v1 as tnn
        except ImportError:
            logging.error("train_nn_model.py not found; falling back to RandomForest training.")
            try:
                tm.run(config_path="config.yaml", pipeline="auto", large_threshold=large_threshold)
                return {"status": "ok", "model": "rf"}
            except AttributeError:
                tm.train_and_save_models()
                return {"status": "ok", "model": "rf"}

        # Try to call a friendly entrypoint
        if hasattr(tnn, "run"):
            tnn.run(config_path="config.yaml")
        elif hasattr(tnn, "train"):
            tnn.train()
        elif hasattr(tnn, "main"):
            tnn.main([])
        else:
            logging.error("No callable entrypoint found in train_nn_model.py; skipping NN training.")
            return {"status": "error", "model": "nn", "reason": "no entrypoint"}
        return {"status": "ok", "model": "nn"}

def predict_new(fits_dir: str = None, labels_dir: str = None, out_dir: str = None, model_dir: str = None) -> dict:
    fits_dir = fits_dir or DIRS["spectral"]
    labels_dir = labels_dir or DIRS["labels"]
    out_dir = out_dir or CFG["directories"]["output"]
    model_dir = model_dir or CFG["directories"].get("models", "data/models/")

    scan = _scan_models(model_dir)
    if not scan["rf_files"] and not scan["nn_files"]:
        logging.error(f"[tool] Predict: no model files found in {model_dir} (.joblib or .keras)")
        return {"status": "error", "reason": "no_models", "model_dir": model_dir}

    # Decide which family to use: prefer whichever exists; if both, prefer the newer set
    if scan["nn_files"] and (scan["latest_nn"] >= scan["latest_rf"] or not scan["rf_files"]):
        model_kind = "nn"
    else:
        model_kind = "rf"

    logging.info(
        f"[tool] Predict: choosing {model_kind} models | rf={len(scan['rf_files'])} (latest={scan['latest_rf']}) "
        f"nn={len(scan['nn_files'])} (latest={scan['latest_nn']})"
    )

    # Optional hint for downstream code (harmless if unused)
    os.environ["MODEL_KIND"] = model_kind

    # Use unified dispatcher in predict.py; it will handle RF/NN loading
    result = pr.predict(
        kind=model_kind,
        fits_dir=fits_dir,
        labels_dir=labels_dir,
        model_dir=model_dir,
        output_dir=out_dir,
        config_path="config.yaml",
    )
    return {"status": "ok", "out_dir": out_dir, "model_kind": model_kind, "result": result}

def evaluate_results() -> dict:
    # If you have an eval step, call it here. If not, compute quick RMSE/residual summaries from output files.
    # Stub keeps the agent pluggable.
    logging.info("[tool] Evaluate (stub)")
    return {"status": "ok"}

def mtime(p: str) -> float:
    try:
        return Path(p).stat().st_mtime
    except FileNotFoundError:
        return 0.0

def newest_in(pattern: str) -> float:
    # quick scanner for “is there anything new”
    candidates = Path(".").glob(pattern)
    return max((c.stat().st_mtime for c in candidates), default=0.0)