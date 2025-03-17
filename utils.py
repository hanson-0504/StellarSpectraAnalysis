import os
import yaml
import logging
import argparse


def setup_env(config):
    """
    Ensures necessary directories exist and sets up logging based on config.
    
    Args:
        config (dict): Parsed configuration dictionary from YAML.
    """
    # Create directories dynamically from the config file
    directories = config.get("directories", {})
    for key, directory in directories.items():
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
    logger.setLevel(logging.DEBUG)

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

    return parser.parse_args()

def read_text_file(filename):
    """Read text file"""
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return lines