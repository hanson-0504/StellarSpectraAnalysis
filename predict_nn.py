import gc
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
import tensorflow as tf
from types import SimpleNamespace
from tensorflow.keras.models import load_model
from neural_network import predict_with_nn
from utils import setup_env, parse_arguments, load_config, read_text_file, open_flux_labels


def load_and_predict(args = None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    setup_env(args.config)
    config = load_config(args.config)

    # Load all the data
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    flux = load(os.path.join(spec_dir, "flux.joblib"))
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir/')
    model_dir = config['directories'].get('models', 'data/model_dir/')
    output_dir = config['directories'].get('output', 'output/')
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "residuals"), exist_ok=True)

    labels = pd.read_csv(os.path.join(labels_dir, "labels.csv"))
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))

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

    errors = []

    for param in tqdm(param_names, desc='Predicting'):
        # Skip if this parameter column is not present in labels
        if param not in labels.columns:
            logging.warning(f"Skipping {param}: column not found in labels.csv")
            continue
        try:
            param_start = time.time()

            # Load model
            model_path = os.path.join(model_dir, f"{param}_model.keras")
            if not os.path.exists(model_path):
                logging.warning(f"Model file for {param} not found. Skipping...")
                continue
            model = load_model(model_path)
            y = labels[param].to_numpy()
            X = flux
            
            # Make Mask
            mask = ~np.isnan(y) & ~np.isnan(feh)
            feh_masked, y_masked, X_masked = feh[mask], y[mask], X[mask]
            del mask, X, y
            gc.collect()

            # Predict
            X_masked = tf.convert_to_tensor(X_masked, dtype=tf.float32)
            predictions = predict_with_nn(model, X_masked).flatten()
            logging.info(f'Prediction array shape = {predictions.shape}')
            residuals = y_masked - predictions
            rmse = np.sqrt(np.mean(residuals**2))
            logging.info(f'RMSE for {param}: {rmse}')
            errors.append(rmse)

            # Save Predictions and Residuals
            if param not in ['teff', 'logg', 'feh']:
                with open(os.path.join(output_dir, f"results/{param}_predictions_nn.csv"), 'w') as f:
                    pd.DataFrame({
                        param: predictions,
                        '[Fe/H]': feh_masked
                    }).to_csv(f, index=False)
                with open(os.path.join(output_dir, f"residuals/{param}_nn.csv"), 'w') as f:
                    pd.DataFrame({
                        '[Fe/H]':feh_masked,
                        'Residuals':residuals
                    }).to_csv(f, index=False)
            else:
                with open(os.path.join(output_dir, f"results/{param}_predictions_nn.csv"), 'w') as f:
                    pd.DataFrame({
                        'Prediction': predictions,
                        'Actual': y_masked
                    }).to_csv(f, index=False)
                with open(os.path.join(output_dir, f"residuals/{param}_nn.csv"), 'w') as f:
                    pd.DataFrame({
                        'Actual': y_masked,
                        'Residuals': residuals
                    }).to_csv(f, index=False)    
            logging.info(f'Predictions and residuals for {param} saved successfully!')
            param_end = time.time()
            logging.info(f"{param} processed in {(param_end-param_start)/60:.2f}")

            del X_masked, y_masked, feh_masked, predictions, residuals
            gc.collect()

        except Exception as e:
            logging.error(f'Error processing {param}: {e}')
            continue
    with open(os.path.join(output_dir, 'results/nn_rmse.csv'), 'w') as f:
        pd.DataFrame({
            'Parameter': param_names,
            'RMSE': errors
        }).to_csv(f, index=False)

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