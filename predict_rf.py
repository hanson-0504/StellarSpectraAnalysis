import gc
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
from types import SimpleNamespace
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_predict
from utils import setup_env, parse_arguments, read_text_file, load_config, open_flux_labels


def load_and_predict(args = None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    config = load_config(args.config)
    setup_env(config)

    # load spectra
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir/')
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    output_dir = args.output_dir or config['directories'].get('output', 'output/')
    model_dir = args.model_dir or config['directories'].get('models', 'data/models/')
    # Ensure output subfolders exist
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "residuals"), exist_ok=True)
    flux, labels, wave = open_flux_labels(spec_dir, labels_dir)

    # Robust [Fe/H] column lookup
    feh_col = None
    for c in labels.columns:
        cl = c.lower()
        if cl in ("fe_h", "[fe/h]", "feh", "feh_true", "fe_h_true"):
            feh_col = c
            break
    if feh_col is None:
        raise KeyError("Could not find a [Fe/H] column in labels.csv. Tried fe_h, [Fe/H], feh, feh_true, fe_h_true.")
    feh = labels[feh_col].to_numpy()
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))
    errors = []

    for param in tqdm(param_names, desc='Predicting'):
        # Skip if this parameter column is not present in labels
        if param not in labels.columns:
            logging.warning(f"Skipping {param}: column not found in labels.csv")
            continue
        try:
            param_start = time.time()
            # load training model
            model_path = os.path.join(model_dir, f"{param}_model.joblib")
            if not os.path.exists(model_path):
                logging.warning(f"Model file for {param} not found. Skipping...")
                continue
            pipeline = load(model_path)
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
    summary = {
        "n_parameters": len(errors),
        "rmse_by_param": dict(zip([p for p in param_names if p in labels.columns], errors)),
        "output_dir": output_dir,
        "duration_min": (end_time - start_time) / 60.0,
    }
    print(f"Predictions completed in {summary['duration_min']:.2f} min")
    return summary

def run(fits_dir=None, labels_dir=None, model_dir=None, output_dir=None, config_path='config.yaml'):
    args = SimpleNamespace(
        fits_dir=fits_dir,
        labels_dir=labels_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        config=config_path,
    )
    return load_and_predict(args)

if __name__ == "__main__":
    try:
        load_and_predict()
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
