import os
import logging
import numpy as np
from scipy import constants
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from sklearn.model_selection import GridSearchCV


def doppler_shift(wave_obs, flux, v, wave_grid):
    """Shifts observed wavelength to rest frame and interpolates flux."""
    try:
        z = v / (constants.c / 1000)
        wave_shift = wave_obs / (1 + z)

        if np.any(np.isnan(wave_shift)) or np.any(np.isnan(flux)):
            raise ValueError("NaN values detected in wavelength or flux.")

        return np.interp(wave_grid, wave_shift, flux)

    except Exception as e:
        logging.error(f"Doppler shift failed: {e}", exc_info=True)
        raise


def interpolate_masked_values(spectrum, mask):
    spectrum = np.copy(spectrum)  # Ensure original data isn't modified
    mask = mask.astype(bool)# Ensure the mask is boolean

    if np.all(mask):
        raise ValueError("All values are masked. Cannot interpolate.")

    valid_indices = np.where(~mask)[0]
    valid_values = spectrum[~mask]
    masked_indices = np.where(mask)[0]

    interpolated_values = np.interp(masked_indices, valid_indices, valid_values)
    spectrum[mask] = interpolated_values

    return spectrum


def combine_tables(table1, table2):
    """Matches DESI spectra table with Apogee table."""
    try:
        if len(table1) == 0 or len(table2) == 0:
            raise ValueError("One of the input tables is empty.")

        table1_clean = table1[np.isfinite(table1['TARGET_RA']) & np.isfinite(table1['TARGET_DEC'])]
        table2_clean = table2[np.isfinite(table2['ra']) & np.isfinite(table2['dec'])]

        if len(table1_clean) == 0 or len(table2_clean) == 0:
            raise ValueError("No valid coordinates found after cleaning tables.")

        coords1 = SkyCoord(ra=table1_clean['TARGET_RA'], dec=table1_clean['TARGET_DEC'], unit=(u.deg, u.deg))
        coords2 = SkyCoord(ra=table2_clean['ra'], dec=table2_clean['dec'], unit=(u.deg, u.deg))

        idx1, idx2, sep2d, _ = coords1.search_around_sky(coords2, 1 * u.arcsec)

        if len(idx1) == 0:
            logging.warning("No matches found between tables.")

        unique_idx = np.unique(idx2, return_index=True)[1]
        idx1, idx2 = idx1[unique_idx], idx2[unique_idx]

        matched_table = Table()
        matched_table['RA_DESI'] = table1_clean['TARGET_RA'][idx2]
        matched_table['Dec_DESI'] = table1_clean['TARGET_DEC'][idx2]
        matched_table['vhelio'] = table2_clean['vhelio_avg'][idx1]
        matched_table['teff'] = table2_clean['teff'][idx1]
        matched_table['logg'] = table2_clean['logg'][idx1]
        matched_table['fe_h'] = table2_clean['fe_h'][idx1]

        for element in ['ce', 'ni', 'co', 'mn', 'cr', 'v', 'tiii', 'ti', 'ca', 'k', 's', 'si', 'al', 'mg', 'na', 'o', 'n', 'ci', 'c']:
            matched_table[f'{element}_fe'] = table2_clean[f'{element}_fe'][idx1]

        logging.info(f"Matched {len(matched_table)} spectra.")
        return matched_table

    except Exception as e:
        logging.error(f"Error in combine_tables(): {e}", exc_info=True)
        raise


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


def setup_env():
    """Ensure necessary directories exist and set up logging."""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('residuals', exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG,  # Logs everything from DEBUG and up
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("training.log"),  # Save logs to a file
            logging.StreamHandler()  # Also print logs to console
        ]
    )