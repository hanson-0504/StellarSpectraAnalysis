import os
import zarr
import glob
import time
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump
from pathlib import Path
from scipy import constants
from astropy.io import fits
from astropy import units as u
from types import SimpleNamespace
from astropy.table import Table
from scipy.ndimage import median_filter
from astropy.coordinates import SkyCoord
from utils import parse_arguments, load_config, setup_env, read_text_file


def doppler_shift(wave_obs, flux, v, wave_grid):
    """Shifts observed wavelength to rest frame and interpolates flux."""
    try:
        z = v / (constants.c / 1000)  # Convert km/s to redshift
        wave_shift = wave_obs / (1 + z)  # Shift to rest frame

        # Debugging logs
        if wave_shift.size == 0:
            raise ValueError("wave_shift is empty before interpolation.")
        if flux.size == 0:
            raise ValueError("flux is empty before interpolation.")
        if wave_grid.size == 0:
            raise ValueError("wave_grid is empty.")

        return np.interp(wave_grid, wave_shift, flux)

    except Exception as e:
        logging.error(f"Doppler shift failed: {e}", exc_info=True)
        raise


def interpolate_masked_values(spectrum, mask):
    spectrum = np.copy(spectrum)# Ensure original data isn't modified
    mask = mask.astype(bool)# Ensure the mask is boolean

    if np.all(mask):
        raise ValueError("All values are masked. Cannot interpolate.")

    valid_indices = np.where(~mask)[0]
    valid_values = spectrum[~mask]
    masked_indices = np.where(mask)[0]

    interpolated_values = np.interp(masked_indices, valid_indices, valid_values)
    spectrum[mask] = interpolated_values

    return spectrum


def combine_tables(table1, table2, parameters, seplimit=3 * u.arcsec):
    """Matches DESI spectra (table1) with APOGEE table (table2) by RA/Dec.
    Returns:
        matched : astropy.Table  # rows from table1 matched to table2 with vhelio + requested params
        matched_indices : np.ndarray  # indices into ORIGINAL table1 rows (for slicing flux/mask arrays)
    """
    try:
        if len(table1) == 0 or len(table2) == 0:
            raise ValueError("One of the input tables is empty.")

        # Column resolution (case-insensitive)
        t1 = {c.lower(): c for c in table1.colnames}
        t2 = {c.lower(): c for c in table2.colnames}
        for need, colmap, name in [
            ('target_ra', t1, 'table1'),
            ('target_dec', t1, 'table1'),
            ('ra',        t2, 'table2'),
            ('dec',       t2, 'table2'),
            ('vhelio_avg',t2, 'table2'),
        ]:
            if need not in colmap:
                raise KeyError(f"Missing required column '{need}' in {name}. Got: {list(colmap.values())}")

        ra1, dec1 = t1['target_ra'], t1['target_dec']
        ra2, dec2, v2 = t2['ra'], t2['dec'], t2['vhelio_avg']

        # Clean both tables on finite RA/Dec; keep masks to map back to original indices
        finite1 = np.isfinite(table1[ra1]) & np.isfinite(table1[dec1])
        finite2 = np.isfinite(table2[ra2]) & np.isfinite(table2[dec2])

        if not np.any(finite1) or not np.any(finite2):
            raise ValueError("No valid coordinates after cleaning.")

        table1_clean = table1[finite1]
        table2_clean = table2[finite2]

        # Map from cleaned -> original row indices
        orig_idx1 = np.nonzero(finite1)[0]
        orig_idx2 = np.nonzero(finite2)[0]

        # Sky coords
        coords1 = SkyCoord(ra=table1_clean[ra1]*u.deg, dec=table1_clean[dec1]*u.deg)
        coords2 = SkyCoord(ra=table2_clean[ra2]*u.deg, dec=table2_clean[dec2]*u.deg)

        # Nearest neighbor from table1 -> table2, then filter by sep limit
        idx2_nearest, sep2d, _ = coords1.match_to_catalog_sky(coords2)
        keep = sep2d <= seplimit
        if not np.any(keep):
            logging.warning("No matches found within separation limit.")
            return Table(), np.array([])

        # If multiple table1 rows map to the same table2 row, keep first occurrence per table1 row (already unique).
        idx1_clean = np.nonzero(keep)[0]          # indices into table1_clean
        idx2_clean = idx2_nearest[keep]           # indices into table2_clean
        idx1_orig  = orig_idx1[idx1_clean]        # indices into original table1
        # idx2_orig = orig_idx2[idx2_clean]       # available if you ever need original table2 indices

        # Build matched table by slicing rows (safer than column-wise fancy indexing)
        matched = table1_clean[idx1_clean]        # brings along target_ra/target_dec, etc.
        matched.rename_column(ra1, 'ra')
        matched.rename_column(dec1, 'dec')

        # Add vhelio and requested parameters from table2_clean
        matched['vhelio'] = table2_clean[v2][idx2_clean]

        t2_cols = {c.lower(): c for c in table2_clean.colnames}
        for param in parameters:
            p = t2_cols.get(param.lower())
            if p is not None:
                matched[param] = table2_clean[p][idx2_clean]
            else:
                logging.warning(f"Parameter '{param}' not found in label table; skipping.")

        logging.info(f"Matched {len(matched)} spectra within {seplimit}.")
        return matched, idx1_orig  # return ORIGINAL table1 row indices for slicing flux/mask arrays

    except Exception as e:
        logging.error(f"Error in combine_tables(): {e}", exc_info=True)
        return Table(), np.array([])


def preprocess_spectra(args = None):
    start_time = time.time()
    if args is None:
        args = parse_arguments()
    setup_env(args.config)
    config = load_config(args.config)

    spec_dir = args.fits_dir or config['directories'].get('spectral', 'data/spectral_dir')
    labels_dir = args.labels_dir or config['directories'].get('labels', 'data/label_dir')

    fits_files = glob.glob(os.path.join(spec_dir, "*.fits"))
    if not fits_files:
        logging.error("No Fits files found!")
        return
    label_files = glob.glob(os.path.join(labels_dir, '*.fits'))
    if not label_files:
        logging.error(f"No label FITS files found in {labels_dir}")
        return
    param_names = read_text_file(os.path.join(labels_dir, 'label_names.txt'))

    with fits.open(label_files[0], memmap=True) as hdus:
        f = Table.read(hdus[1])
    available_columns = {col.lower(): col for col in f.colnames}
    # Standardize parameter names (case-insensitive)
    parameters = ['ra', 'dec', 'vhelio_avg'] + param_names
    keep = []
    for p in parameters:
        col = available_columns.get(p.lower())
        if col: keep.append(col)
        else:   logging.warning(f"Column '{p}' not in APOGEE file; skipping.")
    apogee_table = Table(data=f[keep])
    del f
    apogee_table.write("data/label_dir/apogee_data.csv", format='ascii.csv', overwrite=True)

    num_spec = 0
    for file in fits_files:
        with fits.open(file, memmap=True) as hdu:
            num_spec += hdu['B_FLUX'].shape[0]
    logging.info(f"Number of spectra = {num_spec}")

    max_spec = getattr(args, "max_spec", None)

    # Streaming setup
    batch_size = getattr(args, "batch_size", 256)
    dtype = np.float32
    zarr_path = Path(spec_dir) / "flux.zarr"
    # We'll create the Zarr store lazily after we know the final wavelength grid length
    zarr_array = None
    buffer = []  # list of np.ndarray rows to append in chunks

    labels_rows = []  # accumulate matched parameter rows aligned to written spectra

    table_list = []
    total_processed = 0

    for fits_file in tqdm(fits_files, desc="Processing FITS files"):
        wave = {}
        flux = {}
        mask = {}

        # Open with memory mapping to avoid loading full arrays
        with fits.open(fits_file, memmap=True) as hdus:
            spec_data = Table(hdus['FIBERMAP'].data)
            for camera in ['B', 'R', 'Z']:
                wave[camera] = hdus[f'{camera}_WAVELENGTH'].data
                mask[camera] = hdus[f'{camera}_MASK'].data
                flux[camera] = hdus[f'{camera}_FLUX'].data

        matched_tables, matched_indices = combine_tables(spec_data, apogee_table, param_names)
        num_spec = len(matched_tables)
        if num_spec == 0:
            logging.warning("No spectra matched in this file; skipping.")
            continue
        table_list.append(matched_tables)

        logging.info(f"Number of unmodified pixels: {len(wave['B'])} in camera B, {len(wave['R'])} in camera R, {len(wave['Z'])} in camera Z")
        logging.info(f"Total: {len(wave['B']) + len(wave['R']) + len(wave['Z'])}")

        # Apply camera-specific wavelength trimming (vectorized)
        wave1 = np.asarray(wave['B'], dtype=dtype)[:-25]
        mask1 = (np.asarray(mask['B'], dtype=bool)[matched_indices])[:, :-25]
        flux1 = np.asarray(flux['B'], dtype=dtype)[matched_indices][:, :-25]

        wave2 = np.asarray(wave['R'], dtype=dtype)[26: -63]
        mask2 = (np.asarray(mask['R'], dtype=bool)[matched_indices])[:, 26: -63]
        flux2 = np.asarray(flux['R'], dtype=dtype)[matched_indices][:, 26: -63]

        wave3 = np.asarray(wave['Z'], dtype=dtype)[63:]
        mask3 = (np.asarray(mask['Z'], dtype=bool)[matched_indices])[:, 63:]
        flux3 = np.asarray(flux['Z'], dtype=dtype)[matched_indices][:, 63:]

        wave_concat = np.concatenate([wave1, wave2, wave3])
        logging.info(f"Number of modified pixels: {wave_concat.shape}")

        # Lazily create Zarr array now that we know pixel count
        if zarr_array is None:
            n_pix = int(wave_concat.shape[0])
            # Fresh store each run
            if zarr_path.exists():
                import shutil
                shutil.rmtree(zarr_path)
            zgroup = zarr.open_group(zarr_path, mode="w")
            zarr_array = zgroup.create_array(
                "flux",
                shape=(0, n_pix),
                chunks=(batch_size, n_pix),
                dtype=dtype,
                overwrite=True,
            )
            zgroup.create_array("wavelength", data=wave_concat.astype(dtype), overwrite=True)
        # Process each matched spectrum
        for ispec in range(num_spec):
            if max_spec is not None and total_processed >= max_spec:
                if buffer:
                    zarr_array.append(np.stack(buffer, axis=0))
                    buffer.clear()
                break # break inner loop

            # Skip rows with NaNs in required labels to keep alignment exact
            row = {name: matched_tables[name][ispec] for name in matched_tables.colnames}
            # Identify label columns requested: vhelio plus the params in param_names
            label_values = [row.get('vhelio')] + [row.get(p) for p in param_names]
            if any([x is None or (hasattr(x, 'astype') and not np.isfinite(np.asarray(x).astype(float))) for x in label_values]):
                continue

            spec_v = float(matched_tables['vhelio'][ispec])  # km/s

            flux_concat = np.concatenate([
                flux1[ispec], flux2[ispec], flux3[ispec]
            ]).astype(dtype, copy=False)
            mask_concat = np.concatenate([
                mask1[ispec], mask2[ispec], mask3[ispec]
            ])

            # Interpolate masked values (works on plain ndarray)
            flux_filled = interpolate_masked_values(flux_concat, mask_concat)

            # Normalize via moving median; guard against zeros
            moving_median = median_filter(flux_filled, size=151)
            moving_median[moving_median == 0] = 1e-10
            normal_flux = (flux_filled / moving_median).astype(dtype, copy=False)

            # Doppler shift to rest frame on the same grid
            flux_rest = doppler_shift(wave_concat, normal_flux, spec_v, wave_concat).astype(dtype, copy=False)

            buffer.append(flux_rest)
            labels_rows.append({p: row.get(p) for p in ['ra','dec','vhelio'] + param_names})
            total_processed += 1

            # Flush buffer to Zarr when we hit batch_size
            if len(buffer) >= batch_size:
                zarr_array.append(np.stack(buffer, axis=0))
                buffer.clear()

        # If we hit max_spec inside the inner loop, break outer loop too
        if max_spec is not None and total_processed >= max_spec:
            break  # breaks out of fits_file loop

    # After processing all files, flush any remaining buffered rows
    if zarr_array is not None and buffer:
        zarr_array.append(np.stack(buffer, axis=0))
        buffer.clear()

    # Build labels DataFrame from collected rows and save
    if labels_rows:
        labels_df = pd.DataFrame.from_records(labels_rows)
        labels_df.to_csv(os.path.join(labels_dir, 'labels.csv'), index=False)
        logging.info(f"Wrote labels.csv with {len(labels_df)} rows aligned to flux.zarr")
    else:
        logging.warning("No labels to write; labels_rows is empty.")

    end_time = time.time()
    logging.info(f"Preprocessing is complete in {(end_time-start_time)/60:.2f} minutes. Total spectra written: {total_processed}")


def run(fits_dir=None, labels_dir=None, config_path="config.yaml", max_spec=None):
    """Programmatic entry point used by agents/tools.py.
    
    Examples
    --------
    >>> import preprocess as pp
    >>> pp.run(fits_dir="data/spectral_dir", labels_dir="data/label_dir", max_spec=500)
    # Outputs: <spec_dir>/flux.zarr (chunked float32) and <labels_dir>/labels.csv
    """
    args = SimpleNamespace(
        fits_dir=fits_dir,
        labels_dir=labels_dir,
        config=config_path,
        max_spec=max_spec,
    )
    return preprocess_spectra(args)

if __name__ == "__main__":
    try:
        preprocess_spectra()
    except Exception as e:
        logging.exception(f"Unexpected crash: {e}")