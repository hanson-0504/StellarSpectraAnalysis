# train_nn_model_v1.py
import gc
import os
import time
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
from neural_network import train_neural_network
from utils import *


def train_and_save_models(args=None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    config = load_config(args.config)
    setup_env(config)

    # Load all the data
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir/')
    model_dir  = Path(config['directories'].get('models',  'data/model_dir/'))
    model_dir.mkdir(parents=True, exist_ok=True)
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))
    flux, wave = open_flux_file(spec_dir)
    labels_csv = Path(labels_dir) / "labels.csv"

    for param in tqdm(param_names, desc='Training Models'):
        try:
            param_start = time.time()
            df_num, valid = load_numeric_labels(labels_csv, [param])
            rows = np.flatnonzero(valid)
            if rows.size == 0:
                raise ValueError(f"No valid rows for target {param}")
            y = df_num.loc[rows, param].to_numpy(dtype='float32')
            X = flux[rows, :]

            nn_model, nn_hps, val_rmse = train_neural_network(X, y, param)
            nn_model.save(f"{model_dir}/{param}_model.keras")
            logging.info(f'\nSaved trained model for {param}')
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

if __name__ == "__main__":
    try:
        train_and_save_models()
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
