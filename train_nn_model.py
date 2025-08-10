import gc
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
from types import SimpleNamespace
from neural_network import train_neural_network
from utils import parse_arguments, load_config, setup_env, read_text_file


def train_and_save_models(args=None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    config = load_config(args.config)
    setup_env(config)

    # Load all the data
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    flux = load(os.path.join(spec_dir, "flux.joblib"))
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir/')
    model_dir = config['directories'].get('models', 'data/model_dir/')
    labels = pd.read_csv(os.path.join(labels_dir, "labels.csv"))
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))

    for param in tqdm(param_names, desc='Training Models'):
        # Skip if this parameter column is not present in labels
        if param not in labels.columns:
            logging.warning(f"Skipping {param}: column not found in labels.csv")
            continue
        try:
            param_start = time.time()
            y = labels[param].to_numpy()
            X = flux

            # Mask NaNs
            mask = ~np.isnan(y)
            y_masked, X_masked = y[mask], X[mask]
            del X, y, mask # Free memory
            gc.collect()

            nn_model, nn_hps, val_rmse = train_neural_network(X_masked, y_masked, param)
            nn_model.save(f"{model_dir}/{param}_model.keras")
            logging.info(f'\nSaved trained model for {param}')
            logging.info(f'Best hyperparameters for {param}: {nn_hps}')
            logging.info(f'Validation RMSE for {param}: {val_rmse}')

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
        logging.error(f"An error occurred during prediction: {e}")
