import os
import yaml
import logging
import argparse
from pathlib import Path
import zarr
import pandas as pd
import numpy as np


def setup_env(config_source):
    """
    Ensures necessary directories exist and sets up logging based on config.

    Args:
        config_source (str | dict): Path to configuration file or a config dictionary.
    """
    # Load configuration
    if isinstance(config_source, str):
        config = load_config(config_source)
        if config is None:
            raise ValueError(f"Failed to load configuration from {config_source}")
    elif isinstance(config_source, dict):
        config = config_source
    else:
        raise TypeError("config_source must be ether file path (str) or a config dict")
    
    # Create directories dynamically from the config file
    directories = config.get("directories", {})
    for key, directory in config.get("directories", {}).items():
        if directory:
            os.makedirs(directory, exist_ok=True)

    # Ensure subdirectories inside output/
    output_dir = directories.get("output", "output/")
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "residuals"), exist_ok=True)

    # Set TensorFlow environment variable (if needed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Configure logging
    log_dir = directories.get("logs", "logs/")
    os.makedirs(log_dir, exist_ok=True)

    # Remove existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set up new logging
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(log_dir, "app.log"))
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def load_config(config_file):
    """
    Loads configuration from a YAML file.
    
    Args:
        config_file (str): Path to the YAML config file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Configuration loaded from {config_file}")
    return config


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process FITS spectra data")
    
    # Define command-line arguments
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--fits_dir", type=str, help="Path to FITS files (overrides config)")
    parser.add_argument("--labels_dir", type=str, help="Path to labels")
    parser.add_argument("--output_dir", type=str, help="Path to outputs")
    parser.add_argument("--max_spec", type=int, default=None, help="Maximum number of spectra to process")

    return parser.parse_args()


def read_text_file(filename):
    """Read text file"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines


def open_flux_file(spec_dir):
    z = zarr.open_group(Path(spec_dir) / "flux.zarr", mode='r')
    X = np.array(z["flux"][:])
    wave = np.array(z["wavelength"][:])
    return X, wave

def load_numeric_labels(labels_csv, cols):
    """
    load labels.csv and coerce specific columns to numeric.
    Treats '--' as missing.
    """
    df = pd.read_csv(labels_csv, na_values=["--"], dtype=str)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing label columns in {labels_csv}: {missing}")

    df_num = df.copy()
    for c in cols:
        df_num[c] = pd.to_numeric(df[c], errors="coerce")
    
    valid_mask = np.isfinite(df_num[cols].to_numpy(dtype="float64")).all(axis=1)
    return df_num, valid_mask
