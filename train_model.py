import gc
import os
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import setup_env, parse_arguments, read_text_file, load_config


def tune_hyperparam(X, y, pipeline, param_grid):
    """Hyperparameter tuning to retrieve best pipeline for ML model.
    Tune on subset of samples, about 20% of total so that the tuning is not overfitting

    Args:
        X (ndarray): Holds the spectral data formatted for {name} stellar parameter, shape=(n_sample, n_features)
        y (ndarray): Holds the target data of {name} stellar parameter, shape=(n_sample)
        pipeline (object): Method for scaling, dimensionality reduction and rf estimation.
        param_grid (dict): Hyperparameter grid for tuning

    Returns:
        Pipline: Best pipeline for parameters for {name} target
        best parameters: Print out the best parameters for {name} target
    """
    try:
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            scoring='neg_root_mean_squared_error',
            cv=5
        )
        grid_search.fit(X, y)

        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f'Error during hyperparameter tuning: {e}')
        raise


# Train Model
def train_and_save_models():
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

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ipca', IncrementalPCA(n_components=100)),
        ('random_forest', RandomForestRegressor(random_state=42))
    ])
    param_grid = {
        'ipca__n_components': [50, 100, 200],  # Vary the number of IPCA components
        'random_forest__n_estimators': [50, 100, 200],  # Number of trees in the forest
        'random_forest__max_depth': [None, 10, 20],  # Tree depth
        'random_forest__min_samples_split': [2, 5, 10],  # Minimum samples to split a node
        'random_forest__min_samples_leaf': [1, 2, 4],  # Minimum samples in a leaf node
    }

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

            # Split data
            X_small, _, y_small, _ = train_test_split(
                X_masked, y_masked, train_size=0.1, random_state=42
            )
            del X_masked, y_masked # Free memory
            gc.collect()

            # Hyperparameter Tuning
            best_pipeline = tune_hyperparam(X_small, y_small, pipeline, param_grid)
            del X_small, y_small
            gc.collect()

            dump(best_pipeline, os.path.join(model_dir, f"{param}_model.joblib"))

            logging.info(f'Saved trained model for {param}')
            param_end = time.time()
            logging.info(f'{param} processed in {(param_end - param_start) / 60:.2f} min')
        except Exception as e:
            logging.error(f'Error processing {param}: {e}')
            continue
    end_time = time.time()
    print(f'Training completed in {(end_time - start_time) / 60:.2f} min')

if __name__ == "__main__":
    train_and_save_models()
