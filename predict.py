import gc
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
from utils import setup_env
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_predict

def load_and_predict():
    setup_env()
    start_time = time.time()
    # load spectra
    flux = load("data/flux.joblib")
    labels = pd.read_csv("data/labels.csv")
    feh = labels['fe_h'].to_numpy()
    param_names = ['teff', 'logg', 'fe_h', 'ce_fe', 'ni_fe', 'co_fe', 'mn_fe', 'cr_fe', 'v_fe', 'tiii_fe', 'ti_fe', 'ca_fe', 'k_fe', 's_fe', 'si_fe', 'al_fe', 'mg_fe', 'na_fe', 'o_fe', 'n_fe', 'ci_fe', 'c_fe']

    errors = []

    for param in tqdm(param_names, desc='Predicting'):
        try:
            param_start = time.time()
            # load training model
            pipeline = load(f"models/{param}_model.joblib")
            # Prepare Data
            y = labels[param].to_numpy()
            X = flux

            mask = ~np.isnan(y) & ~np.isnan(feh)
            feh_masked, y_masked, X_masked = feh[mask], y[mask], X[mask]
            del mask, X, y  # Free memory
            gc.collect()

            # Predict
            predictions = cross_val_predict(pipeline, X_masked, y_masked, cv=5)
            errors.append(root_mean_squared_error(y_masked, predictions))

            # Save results
            pd.DataFrame({
                param:predictions,
                '[Fe/H]':feh_masked
            }).to_csv(f"results/{param}_predictions.csv", index=False)

            # Residuals
            residuals = y_masked - predictions
            pd.DataFrame({
                '[Fe/H]':feh_masked,
                'Residuals':residuals
            }).to_csv(f"residuals/{param}.csv", index=False)

            logging.info(f'Predictions and residuals for {param} saved successfully!')
            param_end = time.time()
            logging.info(f"{param} processed in {(param_end - param_start) / 60:.2f} min")

            del X_masked, y_masked, feh_masked, predictions, residuals
            gc.collect()

        except Exception as e:
            logging.error(f'Error processing {param}: {e}')
            continue

    pd.DataFrame({
        'Parameter':param_names,
        'RMSE':errors
    }).to_csv("results/predictions_errors.csv", index=False)

    end_time = time.time()
    print(f"Predictions completed in {(end_time - start_time) / 60:.2f}")

if __name__ == "__main__":
    load_and_predict()