import glob
import argparse
import numpy as np
from tqdm import tqdm
from joblib import dump
from astropy.io import fits
from astropy.table import Table, vstack
from scipy.ndimage import median_filter
from utils import doppler_shift, interpolate_masked_values, combine_tables


# Set up argparse
parser = argparse.ArgumentParser(description="Process FITS spectra data")
parser.add_argument('--fits_dir', type=str, default='data/', help='Path to FITS files')
parser.add_argument('--exclude_labels', type=str, default='apogee_set.fits', help='Pattern to exclude from the file list')
# You can add more arguments here as needed (e.g., for output directory, processing options, etc.)
args = parser.parse_args()

# Get the list of FITS files excluding the specified pattern
fitsdata = [
    f for f in glob.glob(f'{args.fits_dir}/*.fits') 
    if args.exclude_pattern not in f
]


def preprocess_spectra():
    apogee_file = "data/apogee_set.fits" # initiate loading of apogee file

    with fits.open(apogee_file) as hdus:
        f = Table.read(hdus[1])
    parameters = ['ra', 'dec', 'vhelio_avg', 'teff', 'logg', 'fe_h', 'ce_fe', 'ni_fe', 'co_fe', 'mn_fe', 'cr_fe', 'v_fe', 'tiii_fe', 'ti_fe', 'ca_fe', 'k_fe', 's_fe', 'si_fe', 'al_fe', 'mg_fe', 'na_fe', 'o_fe', 'n_fe', 'ci_fe', 'c_fe']
    apogee_table = Table(data=f[parameters])
    del f
    apogee_table.write("data/apogee_data.csv", format='ascii.csv', overwrite=True)

    num_spec = 0
    for file in fitsdata:
        with fits.open(file) as hdu:
            num_spec += hdu['B_FLUX'].shape[0]
    print(f"Number of spectra = {num_spec}")

    # initiate large arrays to contain flux and wave data for all spectra
    big_flux_array = np.zeros((num_spec, 7781)) # Number of stars x length of data arrays

    table_list = []
    for fits_file in tqdm(fitsdata, desc="Processing FITS files"):
        wave = dict()
        flux = dict()
        mask = dict()

        with fits.open(fits_file) as hdus:
            spec_data = Table(hdus['FIBERMAP'].data)
            for camera in ['B', 'R', 'Z']:
                wave[camera] = hdus[f'{camera}_WAVELENGTH'].data
                mask[camera] = hdus[f'{camera}_MASK'].data
                flux[camera] = hdus[f'{camera}_FLUX'].data
        matched_tables = combine_tables(spec_data, apogee_table)
        print(f'\nNumber of spectra = {num_spec}')
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
    matched_tables.write("data/labels.csv", format='ascii.csv', overwrite=True)

    # Create file to store data
    dump(big_flux_array, "data/flux.joblib")

if __name__ == "__main__":
    preprocess_spectra()