# train_nn_model_v2.py
import gc
import os
import time
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from torch_nn_random_search import train_neural_network
from utils import *


def train_and_save_models(args=None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    config = load_config(args.config)
    setup_env(config)

    # Directories from config with CLI overrides
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/labels_dir')
    model_dir  = Path(config['directories'].get('models',  'data/models/'))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Parameters to train
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))

    # Load flux and labels
    flux, wave = open_flux_file(spec_dir)
    labels_csv = Path(labels_dir) / 'labels.csv'

    for param in tqdm(param_names, desc='Training Models'):
        try:
            param_start = time.time()
            df_num, valid = load_numeric_labels(labels_csv, [param])
            rows = np.flatnonzero(valid)
            if rows.size == 0:
                raise ValueError(f"No valid rows for target {param}")
            y = df_num.loc[rows, param].to_numpy(dtype='float32')
            X = flux[rows, :]

            ckpt_path, nn_hps, val_rmse = train_and_save_torch(
                X, y, param_name=param, models_dir=str(model_dir),
                max_trials=15, seed=42
            )
            logging.info(f'Saved trained model for {param}: {ckpt_path}')
            logging.info(f'Best hyperparameters for {param}: {nn_hps}')
            logging.info(f'Validation RMSE for {param}: {val_rmse:.4f}')

            del X, y
            gc.collect()

            param_end = time.time()
            logging.info(f'{param} processed in {(param_end - param_start) / 60:.2f} min')
        except Exception as e:
            logging.error(f'Error processing {param}: {e}')
            continue

    end_time = time.time()
    print(f'Training completed in {(end_time - start_time) / 60:.2f} min')


def run(fits_dir=None, labels_dir=None, config_path='config.yaml'):
    args = SimpleNamespace(
        fits_dir=fits_dir,
        labels_dir=labels_dir,
        config=config_path,
    )
    return train_and_save_models(args)


def train_and_save_torch(X, y, param_name, models_dir="data/models/", max_trials=15, seed=42):
    """
    Trains a PyTorch MLP with random-search hparams and saves the best checkpoint.
    Returns (model_path, best_hps_dict, best_val_rmse).
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    model, best_hps, best_val_rmse = train_neural_network(
        X, y, param_name, max_trials=max_trials, seed=seed, verbose=True
    )

    # Package a simple checkpoint: state_dict + meta (input dim & hparams)
    # NOTE: train_neural_networks already restores best weights
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_name = f"{param_name}_torch_{stamp}.pt"
    ckpt_path = Path(models_dir) / ckpt_name
    
    in_dim = model.net[0].in_features
    
    torch.save({
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "hparams": best_hps,
        "param_name": param_name,
    }, ckpt_path)
    
    sidecar = str(ckpt_path).replace(".pt", ".json")
    with open(sidecar, "w") as f:
        json.dump({
            "param_name": param_name,
            "best_val_rmse": float(best_val_rmse),
            "hparams": best_hps,
            "checkpoint": ckpt_name,
        }, f, indent=2)

    print(f"[OK] Saved best model to {ckpt_path} (val_rmse={best_val_rmse:.5f})")
    return ckpt_path, best_hps, float(best_val_rmse)


def main():
    # Reuse parse_arguments() from utils to support --config, --fits_dir, --labels_dir
    args = parse_arguments()
    train_and_save_models(args)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
