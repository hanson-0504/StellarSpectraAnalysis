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


def combine_tables(table1, table2, param_names):
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

        for param in param_names:
            matched_table[param] = table2_clean[param][idx1]

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

    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/')
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/')

    fits_files = glob.glob(os.path.join(spec_dir, "*.fits"))
    apogee_file = os.path.join(labels_dir, 'apogee_set.fits')
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))

    with fits.open(apogee_file) as hdus:
        f = Table.read(hdus[1])
    parameters = ['ra', 'dec', 'vhelio_avg'] + param_names
    apogee_table = Table(data=f[parameters])
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
        logging.info(f'\nNumber of spectra = {num_spec}')
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
    dump(big_flux_array, "data/flux.joblib")
    end_time = time.time()
    logging.info(f"Preprocessing is complete in {(end_time-start_time)/60:.2f}")

if __name__ == "__main__":
    preprocess_spectra()