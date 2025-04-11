import gc
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import load
import tensorflow as tf
from tf.keras.models import load_model
from neural_network import predict_with_nn
from utils import setup_env, parse_arguments, load_config, read_text_file


def load_and_predict():
    start_time = time.time()
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
    feh = labels['fe_h'].to_numpy()
    errors = []

    for param in tqdm(param_names, desc='Predicting'):
        try:
            param_start = time.time()
            # Load model
            model = load_model(os.path.join(model_dir, f"{param}_model.keras"))         
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
            errors.append(rmse)
            # Save Predictions
            pd.DataFrame({
                param:predictions,
                '[Fe/H]':feh_masked
            }).to_csv(f"results/{param}_predictions.csv", index=False)
            # Save Residuals
            pd.DataFrame({
                '[Fe/H]':feh_masked,
                'Residuals':residuals
            }).to_csv(f"residuals/{param}.csv", index=False)

            logging.info(f'Predictions and residuals for {param} saved successfully!')
            param_end = time.time()
            logging.info(f"{param} processed in {(param_end-param_start)/60:.2f}")

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
    print(f"Predictions completed in {(end_time - start_time)/60:.2f}")

if __name__ == "__main__":
    load_and_predict()