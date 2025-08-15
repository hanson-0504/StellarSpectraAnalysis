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
from utils import *


def load_and_predict(args = None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    setup_env(args.config)
    config = load_config(args.config)

    # Load all the data
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir/')
    model_dir = config['directories'].get('models', 'data/model_dir/')
    output_dir = config['directories'].get('output', 'output/')
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "residuals"), exist_ok=True)

    flux, wave = open_flux_file(spec_dir)
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))
    labels_csv = Path(labels_dir) / "labels.csv"

    errors = []

    for param in tqdm(param_names, desc='Predicting'):
        try:
            param_start = time.time()

            # Load model
            model_path = os.path.join(model_dir, f"{param}_model.keras")
            if not os.path.exists(model_path):
                logging.warning(f"Model file for {param} not found. Skipping...")
                continue
            model = load_model(model_path)

            df__num, valid = load_numeric_labels(labels_csv, [param])
            rows = np.flatnonzero(valid)
            if rows.size == 0:
                raise ValueError(f"No valid rows for target {param}")
            y = df__num.loc[rows, param].to_numpy(dtype='float32')
            X = flux[rows, :]
            
            
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