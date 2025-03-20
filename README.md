# Stellar Parameter and Chemical Abundances Estimation

This project applies **machine learning** to estimate stellar parameters ($T_\text{eff}$, $\log g$, [Fe/H]) and chemical abundances from spectral data. It uses **Random Forest Regression** with **IncrementalPCA** for dimensionality reduction. The input consists of stellar spectra (flux vs. wavelength), and the output includes estimates of the parameters and abundances.

## Overview

This project is designed to analyse stellar spectra and infer physical parameters and chemical abundances. The workflow consists of:

1. **Data Preprocessing**: Reads FITS files, extracts flux values, and handles missing values.
2. **Feature Engineering**: Normalises flux, applies Doppler corrections, and interpolates missing values.
3. **Machine Learning**: Uses **Random Forest Regression** with hyperparameter tuning.
4. **Model Tuning & Evaluation**: Trains separate models for each parameter, performs cross-validation, and evaluates performance using RMSE.
5. **Prediction**: Given a new spectrum, the trained models can estimate the stellar parameters.

## Project Structure

    ├── data/                 # Contains spectral data and labels 
    |   ├── spectral_dir/
    |   │   ├── flux.joblib       # Preprocessed flux values  
    |   |   ├── spectral fits files
    |
    │   ├── labels_dir/        # Stellar parameters and abundances  
    │   │   ├── labels for parameters (fits files)
    │   │   ├── parameter names (txt file)
    |
    ├── models/               # Stores trained models  
    │   ├── teff_model.joblib  (.keras for neural network model) 
    │   ├── fe_h_model.joblib  
    │   ├── (other element models)  
    │
    ├── config.yaml           # Configuration file (manages directories and files)
    ├── train_model.py        # Main script for training models  
    ├── predict.py            # Script for making predictions  
    ├── preprocess.py         # Preprocess spectra (normalisation, Doppler correction, etc.)
    ├── utils.py              # Helper functions (setup_env, read_text_file, etc)  
    ├── requirements.txt      # List of dependencies  
    └── README.md             # This file  

## Installation

### 1. Clone Repository

    git clone "[url_to_repository.git](https://github.com/hanson-0504/StellarSpectraAnalysis.git)"
    cd "StellarSpectraAnalysis"

### 2. Create a Virtual Environment & Install Dependencies

    conda create --name stell_ml python=3.12
    conda activate stell_ml
    pip install -r requirements.txt

## Usage

### 1. Preprocess Spectral Dataset

Before training models, the data needs preprocessing. This step reads FITS files, extracts the flux values, applies Doppler corrections, and normalises the flux.

    python preprocess.py --fits_dir <path_to_fits_files> --labels_dir <path_to_labels_files>

- --fits_dir: The directory containing the FITS files (default is data/sectral_dir).
- --labels_dir: The directory containing the labels for parameters (e.g., apogee_set.fits).

### 2. Train Models

After preprocessing the data, you can train the models to estimate the stellar parameters and chemical abundances. The training process uses Random Forest Regression and IncrementalPCA for dimensionality reduction. To train models, run:

    python train_model.py

This will train models for each stellar parameter and chemical abundance, then save the trained models in the models/ directory.

### 3. Make Predictions

Once the models are trained, you can use them to make predictions on new spectra. Provide the preprocessed spectral data and the trained models will estimate the stellar parameters and abundances.

To make predictions, run:

    python predict.py --fits_dir <path_to_its_files>

- --fits_dir: The directory containing the FITS files to predict on (default is data/).

The predictions will be saved in the results/ directory.

### 4. View Model Performance

To evaluate the performance of the trained models, RMSE (Root Mean Squared Error) values are calculated for each parameter. These are saved in results/predictions_errors.csv. You can review the model performance in this file.

### 5. View Residuals

For each parameter, the residuals (the difference between the true values and the predicted values) are calculated and saved in the residuals/ directory. These can help you analyze how well the models are performing on different spectra.
