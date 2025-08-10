import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import train_rf_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.ensemble import RandomForestRegressor
from train_rf_model import *

def test_tune_hyperparam_returns_pipeline():
    # Create dummy data
    X = np.random.rand(20, 10)  # 20 samples, 10 features
    y = np.random.rand(20)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ipca', IncrementalPCA(n_components=5)),
        ('random_forest', RandomForestRegressor())
    ])
    param_grid = {
        'ipca__n_components': [2],
        'random_forest__n_estimators': [10],
        'random_forest__max_depth': [3]
    }

    best_model = tune_hyperparam(X, y, pipeline, param_grid)
    assert hasattr(best_model, 'predict')
    
def test_train_and_save_models(tmp_path):
    # Create necessary subdirectories inside tmp_path
    spec_dir = tmp_path / "spectral_dir"
    labels_dir = tmp_path / "label_dir"
    model_dir = tmp_path / "models"
    spec_dir.mkdir()
    labels_dir.mkdir()
    model_dir.mkdir()

    # Create and save dummy flux data
    flux = np.random.rand(50, 7781)  # 50 samples, 7781 features
    from joblib import dump
    dump(flux, spec_dir / "flux.joblib")

    # Create and save dummy labels
    import pandas as pd
    labels_df = pd.DataFrame({
        "teff": np.random.uniform(4000, 6000, size=50),
        "logg": np.random.uniform(1.0, 5.0, size=50),
        "fe_h": np.random.uniform(-2.5, 0.5, size=50),
        "mg_fe": np.random.uniform(-0.5, 0.5, size=50),
    })
    labels_df.to_csv(labels_dir / "labels.csv", index=False)

    # Create label_names.txt
    with open(labels_dir / "label_names.txt", "w") as f:
        f.write("teff\nlogg\nfe_h\nmg_fe\n")

    # Write config.yaml pointing to those paths
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
directories:
  spectral: "{spec_dir}"
  labels: "{labels_dir}"
  models: "{model_dir}"
""")

    import types
    args = types.SimpleNamespace(
        config=str(config_path),
        fits_dir=None,
        labels_dir=None
    )
    train_rf_model.parse_arguments = lambda: args
    train_and_save_models()

    assert any(model_dir.glob("*.joblib")), "No models were save    # ...existing code..."
