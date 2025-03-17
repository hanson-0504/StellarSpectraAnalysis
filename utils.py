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
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")

    # Ensure subdirectories inside output/
    output_dir = directories.get("output", "output/")
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "residuals"), exist_ok=True)
    
    logging.info("Ensured 'results' and 'residuals' directories exist under output/")

    # Set TensorFlow environment variable (if needed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Setup logging
    setup_logging(config)


def setup_logging(config):
    """
    Sets up logging based on the configuration file.

    Args:
        config (dict): Parsed configuration dictionary from YAML.
    """
    log_config = config.get("logging", {})

    log_level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_file = log_config.get("log_file", "logs/app.log")
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    log_to_console = log_config.get("console", True)

    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()

    # Prevent adding multiple handlers in case of reconfiguration
    if not logger.hasHandlers():
        # Define log handlers
        file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(console_handler)

        # Set log level
        logger.setLevel(log_level)

    logging.info("Logging setup complete.")


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