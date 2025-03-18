import os
import glob
import time
import logging
import numpy as np
from tqdm import tqdm
from joblib import dump
from scipy import constants
from astropy.io import fits
from astropy import units as u
from astropy.table import Table, vstack
from scipy.ndimage import median_filter
from astropy.coordinates import SkyCoord
from utils import parse_arguments, load_config, setup_env, read_text_file


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


def combine_tables(table1, table2, parameters):
    """Matches DESI spectra table with Apogee table."""
    try:
        if len(table1) == 0 or len(table2) == 0:
            raise ValueError("One of the input tables is empty.")

        # Create case-insensitive column mappings for both tables
        table1_cols = {col.lower(): col for col in table1.colnames}
        table2_cols = {col.lower(): col for col in table2.colnames}

        # Standardized column names
        target_ra_col = table1_cols.get('target_ra', None)
        target_dec_col = table1_cols.get('target_dec', None)
        ra_col = table2_cols.get('ra', None)
        dec_col = table2_cols.get('dec', None)
        vhelio_col = table2_cols.get('vhelio_avg', None)

        # Check if necessary columns exist
        missing_cols = [c for c in ['target_ra', 'target_dec', 'ra', 'dec', 'vhelio_avg'] if c not in table1_cols and c not in table2_cols]
        if missing_cols:
            raise KeyError(f"Missing required columns: {missing_cols}. Available columns in table1: {table1.colnames}, table2: {table2.colnames}")

        # Clean data: ensure finite values
        table1_clean = table1[np.isfinite(table1[target_ra_col]) & np.isfinite(table1[target_dec_col])]
        table2_clean = table2[np.isfinite(table2[ra_col]) & np.isfinite(table2[dec_col])]

        if len(table1_clean) == 0 or len(table2_clean) == 0:
            raise ValueError("No valid coordinates found after cleaning tables.")

        # Convert to SkyCoord objects
        coords1 = SkyCoord(ra=table1_clean[target_ra_col], dec=table1_clean[target_dec_col], unit=(u.deg, u.deg))
        coords2 = SkyCoord(ra=table2_clean[ra_col], dec=table2_clean[dec_col], unit=(u.deg, u.deg))

        # Cross-match tables
        idx1, idx2, sep2d, _ = coords1.search_around_sky(coords2, 3 * u.arcsec)

        if len(idx1) == 0:
            logging.warning("No matches found between tables.")

        # Ensure unique matches
        unique_idx = np.unique(idx2, return_index=True)[1]
        idx1, idx2 = idx1[unique_idx], idx2[unique_idx]

        # Create matched table
        matched_table = Table()
        matched_table['RA'] = table1_clean[target_ra_col][idx2]
        matched_table['Dec'] = table1_clean[target_dec_col][idx2]
        matched_table['vhelio'] = table2_clean[vhelio_col][idx1]

        # Generalized parameter selection
        for param in parameters:
            param_col = table2_cols.get(param.lower(), None)
            if param_col:
                matched_table[param] = table2_clean[param_col][idx1]
            else:
                logging.warning(f"Parameter {param} not found in table2.")

        logging.info(f"Matched {len(matched_table)} spectra.")
        return matched_table

    except Exception as e:
        logging.error(f"Error in combine_tables(): {e}", exc_info=True)
        raise


def preprocess_spectra():
    start_time = time.time()
    args = parse_arguments()
    config = load_config(args.config)
    setup_env(config)

    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir')

    fits_files = glob.glob(os.path.join(spec_dir, "*.fits"))
    label_files = glob.glob(os.path.join(labels_dir, '*.fits'))
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))

    with fits.open(label_files[0], memap=True) as hdus:
        f = Table.read(hdus[1])
    available_columns = {col.lower(): col for col in f.colnames}
    # Standardize parameter names (case-insensitive)
    parameters = ['ra', 'dec', 'vhelio_avg'] + param_names
    normalized_parameters = [available_columns.get(param.lower(), param) for param in parameters]

    # Extract table data
    apogee_table = Table(data=f[normalized_parameters])
    del f
    # apogee_table.write("data/apogee_data.csv", format='ascii.csv', overwrite=True)

    num_spec = 0
    for file in fits_files:
        with fits.open(file) as hdu:
            num_spec += hdu['B_FLUX'].shape[0]
    logging.info(f"Number of spectra = {num_spec}")

    # initiate large arrays to contain flux and wave data for all spectra
    big_flux_array = np.zeros((num_spec, 7781)) # Number of stars x length of data arrays

    table_list = []
    for fits_file in tqdm(fits_files, desc="Processing FITS files"):
        wave = dict()
        flux = dict()
        mask = dict()

        with fits.open(fits_file) as hdus:
            spec_data = Table(hdus['FIBERMAP'].data)
            for camera in ['B', 'R', 'Z']:
                wave[camera] = hdus[f'{camera}_WAVELENGTH'].data
                mask[camera] = hdus[f'{camera}_MASK'].data
                flux[camera] = hdus[f'{camera}_FLUX'].data
        matched_tables = combine_tables(spec_data, apogee_table, param_names)
        if len(matched_tables) == 0:
            logging.error("No spectra matched. Check coordinate ranges and units.")
            return  # Exit gracefully instead of crashing
        table_list.append(matched_tables)

        # Eliminate the last 25 wavelengths from camera B
        wave1 = np.array(wave['B'])
        mask1 = np.array(mask['B'])
        flux1 = np.array(flux['B'])

        wave1 = wave1[:-25]
        mask1 = mask1[:, :-25]
        flux1 = flux1[:, :-25]

        # Eliminate the first 26 wavelengths from camera R
        # Eliminate the last 63 wavelengths from camera R
        wave2 = np.array(wave['R'])
        wave2 = wave2[26:]
        wave2 = wave2[:-63]

        mask2 = np.array(mask['R'])
        mask2 = mask2[:, 26:]
        mask2 = mask2[:, :-63]

        flux2 = np.array(flux['R'])
        flux2 = flux2[:, 26:]
        flux2 = flux2[:, :-63]

        # Eliminate the first 63 wavelengths from camera Z
        wave3 = np.array(wave['Z'])
        wave3 = wave3[63:]
        mask3 = np.array(mask['Z'])
        mask3 = mask3[:, 63:]
        flux3 = np.array(flux['Z'])
        flux3 = flux3[:, 63:]

        wave = np.concatenate([wave1, wave2, wave3])

        for ispec in range(num_spec):
            spec_v = matched_tables['vhelio'][ispec] # in km/s

            flux = np.concatenate([flux1[ispec], flux2[ispec], flux3[ispec]])
            mask = np.concatenate([mask1[ispec], mask2[ispec], mask3[ispec]])

            flux = np.ma.MaskedArray(data=flux, mask=mask)
            flux = interpolate_masked_values(flux, mask)

            # normalize flux
            window_size = 151
            moving_median = median_filter(flux, size=window_size)
            moving_median[moving_median == 0] = 1e-10
            normal_flux = flux / moving_median
            normal_flux = normal_flux

            flux_rest = doppler_shift(wave, normal_flux, spec_v, wave)
            # append data arrays
            big_flux_array[ispec] = flux_rest
    matched_tables = vstack(table_list)
    matched_tables.write(os.path.join(labels_dir, 'labels.csv'), format='ascii.csv', overwrite=True)

    # Create file to store data
    dump(big_flux_array, os.path.join(spec_dir, "flux.joblib"))
    end_time = time.time()
    logging.info(f"Preprocessing is complete in {(end_time-start_time)/60:.2f}")

if __name__ == "__main__":
    preprocess_spectra()