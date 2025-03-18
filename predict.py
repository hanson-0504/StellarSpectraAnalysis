import gc
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_predict
from utils import setup_env, parse_arguments, read_text_file, load_config


def load_and_predict():
    start_time = time.time()
    args = parse_arguments()
    config = load_config(args.config)
    setup_env(config)

    # load spectra
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir/')
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    output_dir = args.output_dir or config['directories'].get('output', 'output/')
    model_dir = config['directories'].get('models', 'data/models/')
    flux = load(os.path.join(spec_dir, 'flux.joblib'))
    labels = pd.read_csv(os.path.join(labels_dir, 'labels.csv'))
    feh = labels['fe_h'].to_numpy()
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))
    errors = []

    for param in tqdm(param_names, desc='Predicting'):
        try:
            param_start = time.time()
            # load training model
            pipeline = load(os.path.join(model_dir, f"{param}_model.joblib"))
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
            }).to_csv(os.path.join(output_dir, f"results/{param}_predictions.csv"), index=False)

            # Residuals
            residuals = y_masked - predictions
            pd.DataFrame({
                '[Fe/H]':feh_masked,
                'Residuals':residuals
            }).to_csv(os.path.join(output_dir, f"residuals/{param}.csv"), index=False)

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
    }).to_csv(os.path.join(output_dir, "results/predictions_errors.csv"), index=False)

    end_time = time.time()
    print(f"Predictions completed in {(end_time - start_time) / 60:.2f}")


if __name__ == "__main__":
    load_and_predict()