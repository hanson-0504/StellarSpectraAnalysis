# agents/rule_agent.py
import time
import logging
from pathlib import Path
import yaml

from .state import load_state, save_state
from .tools import preprocess_spectra, train_models, predict_new, evaluate_results, newest_in

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def decide_and_act(poll_seconds: int = 10, once: bool = False):
    with open("config.yaml", "r") as f:
        CFG = yaml.safe_load(f)
    dirs = CFG["directories"]
    patterns = CFG.get("file_patterns", {})

    state = load_state()

    while True:
        # Heuristics: run preprocess if new FITS/labels appear
        spec_new = newest_in(f"{dirs['spectral']}/{patterns.get('spectral','*.fits')}")
        lab_new  = newest_in(f"{dirs['labels']}/{patterns.get('labels','*.fits')}")

        # Run preprocess first if new raw data
        if spec_new > state["last_preprocess"] or lab_new > state["last_preprocess"]:
            logging.info("New raw data detected → preprocess")
            preprocess_spectra()
            state["last_preprocess"] = time.time()
            save_state(state)
            if once: break
            time.sleep(poll_seconds); 
            continue

        # Train if preprocessed flux newer than last train
        flux_new = newest_in(f"{dirs['spectral']}/{patterns.get('flux','*.joblib')}")
        if flux_new > state["last_train"]:
            logging.info("Flux updated → train models")
            train_models()
            state["last_train"] = time.time()
            save_state(state)
            if once: break
            time.sleep(poll_seconds); 
            continue

        # Predict if models exist and new spectra appear (or a new model)
        model_new = max(newest_in(f"{dirs['models']}/{p}") for p in CFG["file_patterns"]["model"])
        if max(model_new, spec_new) > state["last_predict"]:
            logging.info("New model or spectra → predict")
            predict_new()
            state["last_predict"] = time.time()
            save_state(state)
            if once: break
            time.sleep(poll_seconds); 
            continue

        # Evaluate optionally
        # logging.info("Evaluate results (optional)")
        # evaluate_results()

        if once:
            break

        time.sleep(poll_seconds)
