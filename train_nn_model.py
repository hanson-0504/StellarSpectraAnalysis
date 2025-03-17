import gc
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
from utils import setup_env
from neural_network import train_neural_network

def train_and_save_models():
    setup_env()
    start_time = time.time()

    flux = load("data/flux.joblib")
    labels = pd.read_csv("data/labels.csv")
    param_names = ['teff', 'logg', 'fe_h', 'ce_fe',
                   'ni_fe', 'co_fe', 'mn_fe', 'cr_fe',
                   'v_fe', 'tiii_fe', 'ti_fe', 'ca_fe', 
                   'k_fe', 's_fe', 'si_fe', 'al_fe', 
                   'mg_fe', 'na_fe', 'o_fe', 'n_fe', 
                   'ci_fe', 'c_fe']

    for param in tqdm(param_names, desc='Training Models'):
        try:
            param_start = time.time()
            y = labels[param].to_numpy()
            X = flux

            # Mask NaNs
            mask = ~np.isnan(y)
            y_masked, X_masked = y[mask], X[mask]
            del X, y, mask # Free memory
            gc.collect()

            nn_model, tuner = train_neural_network(X_masked, y_masked, param)

            nn_model.save(f"models/{param}_model.keras")
            logging.info(f'\nSaved trained model for {param}')

            param_end = time.time()
            logging.info(f'{param} processed in {(param_end - param_start) / 60:.2f} min')
        except Exception as e:
            logging.error(f'Error processing {param}: {e}')
            continue

    end_time = time.time()
    print(f'Training completed in {(end_time - start_time) / 60:.2f} min')


if __name__ == "__main__":
    train_and_save_models()
