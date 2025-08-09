import os
import sys
import numpy as np
import subprocess
from astropy.io import fits
from astropy.table import Table

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess import *
def test_doppler():
    wave = np.array([100, 200, 300])
    flux = np.array([1, 0.5, 1.25])
    v = 20 # km/s
    
    result_flux = doppler_shift(wave, flux, v, wave)

    assert result_flux.shape == flux.shape
    assert isinstance(result_flux, np.ndarray)
    assert not np.any(np.isnan(result_flux))
    
def test_interpolation():
    flux = np.array([1, 0.5, np.nan, 1.25])
    mask = np.array([False, False, True, False])
    
    result_flux = interpolate_masked_values(flux, mask)
    
    assert isinstance(result_flux, np.ndarray)
    assert result_flux.shape == flux.shape
    assert not np.any(np.isnan(result_flux))
    
def test_combine_tables():
    table1 = Table({
        'target_ra': [100.0, 101.0],
        'target_dec': [20.0, 21.0]
    })
    table2 = Table({
        'ra': [100.0001, 105.0],
        'dec': [20.0001, 25.0],
        'vhelio_avg': [30.0, 40.0],
        'fe_h': [-0.1, 0.2],
        'mg_fe': [0.3, 0.6]
    })
    params = ['fe_h', 'mg_fe']
    matched_table, idx = combine_tables(table1, table2, params)
    
    assert len(matched_table) == 1
    assert 'vhelio' in matched_table.colnames
    assert 'fe_h' in matched_table.colnames
    assert np.isclose(matched_table['ra'][0], 100.0001)
    assert np.isclose(matched_table['fe_h'][0], -0.1)
    
def test_preprocess_spectra(tmp_path):
    """Test preprocess function with dummy directory structure."""

    fits_dir = tmp_path / "spectral_dir"
    label_dir = tmp_path / "label_dir"
    output_dir = tmp_path / "output"
    logs_dir = tmp_path / "logs"
    fits_dir.mkdir()
    label_dir.mkdir()
    output_dir.mkdir()
    logs_dir.mkdir()

    # Write a config.yaml pointing to those dirs
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
directories:
  spectral: "{fits_dir}"
  labels: "{label_dir}"
  output: "{output_dir}"
  logs: "{logs_dir}"
""")
    assert config_path.exists()
    
    spec_path = fits_dir / "dummy_spectrum.fits"
    label_path = label_dir / "dummy_labels.fits"
    
    create_dummy_spectrum_fits(spec_path)
    create_dummy_label_fits(label_path)
    
    params = "fe_h\nmg_fe"
    (label_dir / "label_names.txt").write_text(params)

    # Act: run preprocess with these dummy paths
    result = subprocess.run(
        ["python", "preprocess.py", "--config", str(config_path)],
        capture_output=True,
        text=True
    )
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert (fits_dir / "flux.joblib").exists()
    assert (label_dir / "labels.csv").exists()

def create_dummy_spectrum_fits(filepath):
    """Create a dummy FITS file with a simple spectrum."""
    n_spec = 2
    
    flux_b = np.random.rand(n_spec, 2751)
    flux_r = np.random.rand(n_spec, 2326)
    flux_z = np.random.rand(n_spec, 2881)
    wave_b = np.linspace(4000, 5000, 2751)
    wave_r = np.linspace(5000, 6000, 2326)
    wave_z = np.linspace(6000, 7000, 2881)
    mask_b = np.zeros_like(flux_b, dtype=np.uint8)
    mask_r = np.zeros_like(flux_r, dtype=np.uint8)
    mask_z = np.zeros_like(flux_z, dtype=np.uint8)

    primary = fits.PrimaryHDU()
    
    b_flux_hdu = fits.ImageHDU(flux_b, name='B_FLUX')
    b_wave_hdu = fits.ImageHDU(wave_b, name='B_WAVELENGTH')
    b_mask_hdu = fits.ImageHDU(mask_b, name='B_MASK')
    
    r_flux_hdu = fits.ImageHDU(flux_r, name='R_FLUX')
    r_wave_hdu = fits.ImageHDU(wave_r, name='R_WAVELENGTH')
    r_mask_hdu = fits.ImageHDU(mask_r, name='R_MASK')
    
    z_flux_hdu = fits.ImageHDU(flux_z, name='Z_FLUX')
    z_wave_hdu = fits.ImageHDU(wave_z, name='Z_WAVELENGTH')
    z_mask_hdu = fits.ImageHDU(mask_z, name='Z_MASK')
    
    # FIBERMAP with RA/DEC
    ra = [100.0, 101.0]
    dec = [20.0, 21.0]
    fibermap = fits.BinTableHDU.from_columns([
        fits.Column(name='TARGET_RA', array=ra, format='D'),
        fits.Column(name='TARGET_DEC', array=dec, format='D')
    ], name='FIBERMAP')

    hdul = fits.HDUList(
        [primary, b_flux_hdu, b_mask_hdu, b_wave_hdu, fibermap, 
         r_flux_hdu, r_mask_hdu, r_wave_hdu, z_flux_hdu, z_mask_hdu, z_wave_hdu]
    )
    hdul.writeto(filepath, overwrite=True)
    
def create_dummy_label_fits(filepath):
    """Create a dummy FITS file for labels."""
    ra = [100.0001, 105.0]
    dec = [20.0001, 25.0]
    vhelio = [30.0, 55.0]
    fe_h = [-.5, 0.0]
    mg_fe = [0.1, 0.2]
    
    label_table = fits.BinTableHDU.from_columns([
        fits.Column(name='RA', array=ra, format='D'),
        fits.Column(name='DEC', array=dec, format='D'),
        fits.Column(name='VHELIO_AVG', array=vhelio, format='D'),
        fits.Column(name='FE_H', array=fe_h, format='E'),
        fits.Column(name='MG_FE', array=mg_fe, format='E')
    ])
    label_table.writeto(filepath, overwrite=True)

    