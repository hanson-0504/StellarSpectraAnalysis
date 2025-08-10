# Stellar Parameter and Chemical Abundances Estimation

This project leverages machine learning to estimate stellar parameters ($T_\text{eff}$, $\log g$, [Fe/H]) and chemical abundances from stellar spectra. It integrates automated preprocessing, model training, and evaluation to facilitate efficient analysis of large spectral datasets.

## Features

- Estimates stellar parameters and chemical abundances from spectral data  
- Supports Random Forest Regression and neural network models  
- Automated preprocessing including normalization and Doppler correction  
- Dimensionality reduction with IncrementalPCA  
- Hyperparameter tuning and cross-validation  
- Comprehensive evaluation with RMSE and residual analysis  

## Installation

1. Clone the repository:  
   `git clone https://github.com/hanson-0504/StellarSpectraAnalysis.git`  
   `cd StellarSpectraAnalysis`

2. Create and activate a virtual environment:  
   `conda create --name stell_ml python=3.12`  
   `conda activate stell_ml`

3. Install dependencies:  
   `pip install -r requirements.txt`

## Usage

- **Preprocess spectra:**  
  `python preprocess.py --fits_dir <path_to_fits_files> --labels_dir <path_to_labels_files>`

- **Train models:**  
  `python train_model.py`

- **Make predictions:**  
  `python predict.py --fits_dir <path_to_fits_files>`

- **Evaluate models:**  
  Review RMSE and residuals saved in the results directory

## Project Structure

```
├── data/                  # Spectral data and labels
│   ├── spectral_dir/      # Preprocessed flux and FITS spectral files
│   └── labels_dir/        # Stellar parameters and chemical abundances
│
├── models/                # Trained models (Random Forest, neural nets)
│
├── results/               # Predictions, RMSE, residuals
│
├── config.yaml            # Configuration settings
├── preprocess.py          # Preprocessing scripts
├── train_model.py         # Model training scripts
├── predict.py             # Prediction scripts
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Data Preparation

- Download the APOGEE DR17 AllStar catalog from:  
  `https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits`

- Place the catalog in the `data/labels_dir/` directory.

- Optionally filter stars or elements using Astropy or pandas before training.

## Training & Prediction

1. **Preprocess the spectral data** to extract and normalize flux, apply Doppler corrections, and handle missing values.  
2. **Train models** for each stellar parameter and abundance using `train_model.py`.  
3. **Predict** parameters on new spectra with `predict.py` using the trained models.

## Results

- Trained models are saved in the `models/` directory.  
- Prediction outputs, RMSE metrics, and residuals are stored in the `results/` directory for performance analysis.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
