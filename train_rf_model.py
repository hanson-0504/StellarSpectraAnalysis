import gc
import os
import time
import logging
import numpy as np
from tqdm import tqdm
from joblib import dump
from types import SimpleNamespace
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.model_selection import train_test_split, GridSearchCV
from utils import setup_env, parse_arguments, read_text_file, load_config, open_flux_labels


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


def build_pipeline_and_grid(mode: str):
    """Return (pipeline, param_grid) according to dataset size mode.

    mode: 'small' | 'large'
    - small: full PCA (variance-based) + RF with a wider search
    - large: IncrementalPCA + RF with a narrower, faster search
    """
    if mode == 'small':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.99, svd_solver='full', whiten=False)),
            ('random_forest', RandomForestRegressor(random_state=42))
        ])
        param_grid = {
            'pca__n_components': [0.95, 0.99],
            'random_forest__n_estimators': [200, 400, 600],
            'random_forest__max_depth': [None, 15, 25],
            'random_forest__min_samples_split': [2, 5],
            'random_forest__min_samples_leaf': [1, 2],
        }
    else:  # large
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ipca', IncrementalPCA(n_components=100, batch_size=512)),
            ('random_forest', RandomForestRegressor(random_state=42, n_jobs=-1))
        ])
        param_grid = {
            'ipca__n_components': [50, 100, 150],
            'random_forest__n_estimators': [100, 200, 300],
            'random_forest__max_depth': [None, 12, 20],
            'random_forest__min_samples_split': [2, 5],
            'random_forest__min_samples_leaf': [1, 2],
        }
    return pipeline, param_grid


# Train Model
def train_and_save_models(args = None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    config = load_config(args.config)
    setup_env(args.config)

    # Load all the data
    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir/')
    model_dir = config['directories'].get('models', 'data/model_dir/')
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))
    flux, labels, wave = open_flux_labels(spec_dir, labels_dir)

    # Decide training mode based on dataset size (or user override)
    n_samples = flux.shape[0]
    default_threshold = 5000  # you can tweak in config if desired
    threshold = getattr(args, 'large_threshold', default_threshold)
    requested_mode = getattr(args, 'pipeline', 'auto')  # 'auto' | 'small' | 'large'
    if requested_mode == 'auto':
        mode = 'large' if n_samples >= threshold else 'small'
    else:
        mode = requested_mode
    logging.info(f"Training mode: {mode} (n_samples={n_samples}, threshold={threshold})")

    pipeline, param_grid = build_pipeline_and_grid(mode)

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

            # Select data for hyperparameter tuning
            if mode == 'large':
                # Subsample for speed on big datasets
                tune_size = min(max(1000, int(0.1 * n_samples)), 5000)
                X_small, _, y_small, _ = train_test_split(
                    X_masked, y_masked, train_size=min(0.8, tune_size / max(1, len(y_masked))), random_state=42
                )
            else:  # small
                # Use most of the data for tuning; GridSearchCV will CV-split internally
                X_small, _, y_small, _ = train_test_split(
                    X_masked, y_masked, train_size=0.8, random_state=42
                )

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

def run(fits_dir=None, labels_dir=None, config_path='config.yaml', pipeline='auto', large_threshold=5000):
    args = SimpleNamespace(
        fits_dir=fits_dir,
        labels_dir=labels_dir,
        config=config_path,
        pipeline=pipeline,
        large_threshold=large_threshold,
    )
    return train_and_save_models(args)

if __name__ == "__main__":
    try:
        train_and_save_models()
    except Exception as e:
        logging.error(f"An error occurred during prediction: {e}")
