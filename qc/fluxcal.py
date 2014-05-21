"""Functions relating to quality control of flux calibration."""

from ..manager import Manager
from ..dr import fluxcal2

import astropy.io.fits as pf
import numpy as np

import os
from glob import glob

def fluxcal_files(mngr):
    """
    Return two dictionaries of flux calibration files, one for each CCD.
    The keys are the paths to the combined calibration files, and each item
    is a list of individual files that contributed.

    The input can be a Manager object or a path, from which a Manager will
    be created. Or a list of managers, in which case the results are
    concatenated. A list of paths will not work.
    """
    if isinstance(mngr, str):
        mngr = Manager(mngr)
    if isinstance(mngr, Manager):
        mngr_list = [mngr]
    else:
        mngr_list = mngr
    result = []
    for mngr in mngr_list:
        for ccd in ['ccd_1', 'ccd_2']:
            result_ccd = {}
            groups = mngr.group_files_by(('date', 'field_id', 'ccd', 'name'),
                ndf_class='MFOBJECT', do_not_use=False,
                spectrophotometric=True, ccd=ccd)
            for group in groups.values():
                combined = os.path.join(
                    group[0].reduced_dir, 'TRANSFERcombined.fits')
                if os.path.exists(combined):
                    result_ccd[combined] = [f.reduced_path for f in group]
            result.append(result_ccd)
    return tuple(result)

def stability(mngr):
    """
    Return arrays of flux calibration stability, defined as the standard
    deviation of the end-to-end throughput as a function of wavelength.

    The return value is a dictionary containing arrays of 'wavelength', 
    'std_combined', 'std_individual', 'mean_individual' and 'mean_combined'.
    The input 'data_combined' and 'data_individual' are included.
    Finally, the mean and standard deviation of the normalisation,
    'mean_norm_comined', 'mean_norm_individual', 'std_norm_combined' and
    'std_norm_individual', are included.
    """
    result = {}
    file_pairs = []
    individual_file_pairs = []
    file_dicts = fluxcal_files(mngr)
    n_individual = 0
    for path_1 in file_dicts[0]:
        path_2 = path_1.replace('ccd_1', 'ccd_2')
        if path_2 in file_dicts[1]:
            file_pairs.append((path_1, path_2))
            filename_1_list = [
                os.path.basename(f) for f in file_dicts[0][path_1]]
            filename_2_list = [
                os.path.basename(f) for f in file_dicts[1][path_2]]
            for i_filename_1, filename_1 in enumerate(filename_1_list):
                filename_2 = filename_1[:5] + '2' + filename_1[6:]
                if filename_2 in filename_2_list:
                    i_filename_2 = filename_2_list.index(filename_2)
                    individual_file_pairs.append((
                        file_dicts[0][path_1][i_filename_1],
                        file_dicts[1][path_2][i_filename_2]))
    n_pix_1 = pf.getval(file_pairs[0][0], 'NAXIS1')
    n_pix_2 = pf.getval(file_pairs[0][1], 'NAXIS1')
    n_combined = len(file_pairs)
    n_individual = len(individual_file_pairs)
    data_combined = np.zeros((n_pix_1+n_pix_2, n_combined))
    data_individual = np.zeros((n_pix_1+n_pix_2, n_individual))
    i_individual = 0
    for i_combined, (path_1, path_2) in enumerate(file_pairs):
        data_combined[:n_pix_1, i_combined] = 1.0 / pf.getdata(path_1)
        data_combined[n_pix_1:, i_combined] = 1.0 / pf.getdata(path_2)
    for i_individual, (path_1, path_2) in enumerate(individual_file_pairs):
        data_individual[:n_pix_1, i_individual] = 1.0 / pf.getdata(
            path_1, 'FLUX_CALIBRATION')[-1, :]
        data_individual[n_pix_1:, i_individual] = 1.0 / pf.getdata(
            path_2, 'FLUX_CALIBRATION')[-1, :]
    common = (
        np.sum(np.isfinite(data_combined), axis=1) + 
        np.sum(np.isfinite(data_individual), axis=1)
        ) == (n_combined + n_individual)
    norm_combined = np.mean(
        data_combined[common, :], axis=0)
    norm_individual = np.mean(
        data_individual[common, :], axis=0)
    data_norm_combined = data_combined / norm_combined
    data_norm_individual = data_individual / norm_individual
    result['std_combined'] = np.std(data_norm_combined, axis=1)
    result['std_individual'] = np.std(data_norm_individual, axis=1)
    result['mean_combined'] = np.mean(data_norm_combined, axis=1)
    result['mean_individual'] = np.mean(data_norm_individual, axis=1)
    result['std_norm_combined'] = np.std(norm_combined)
    result['std_norm_individual'] = np.std(norm_individual)
    result['mean_norm_combined'] = np.mean(norm_combined)
    result['mean_norm_individual'] = np.mean(norm_individual)
    result['data_combined'] = data_combined
    result['data_individual'] = data_individual
    wavelength = np.zeros(n_pix_1 + n_pix_2)
    for i, path in enumerate(file_pairs[0]):
        header = pf.getheader(path)
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        crpix1 = header['CRPIX1']
        if i == 0:
            start = 0
            finish = n_pix_1
            n_pix = n_pix_1
        else:
            start = n_pix_1
            finish = n_pix_1 + n_pix_2
            n_pix = n_pix_2
        wavelength[start:finish] = crval1 + cdelt1 * (
            np.arange(n_pix) + 1 - crpix1)
    result['wavelength'] = wavelength
    return result


def stellar_colours(mngr):
    """
    Return stellar colours as measured by SAMI (via a template fit) compared
    to the SDSS imaging.
    """
    file_pair_list = list_star_files(mngr)
    model_catalogue = read_stellar_models()
    model_list = [fit_template(file_pair, model_catalogue)[-1]
                  for file_pair in file_pair_list]
    observed_colours = [measure_colour(model, model_catalogue['wavelength']) 
                        for model in model_list]
    return file_pair_list, observed_colours





def list_star_files(mngr):
    """Return a list of tuples of paths to star datacubes, blue and red."""
    if isinstance(mngr, str):
        mngr = Manager(mngr)
    if isinstance(mngr, Manager):
        mngr_list = [mngr]
    else:
        mngr_list = mngr
    result = []
    for mngr in mngr_list:
        blue_list = (
            glob(os.path.join(mngr.abs_root, 'cubed', '*', '*blue*.fits')) +
            glob(os.path.join(mngr.abs_root, 'cubed', '*', '*blue*.fits.gz')))
        for blue_path in blue_list:
            red_path = red_cube_path(blue_path)
            if os.path.exists(red_path):
                if (pf.getval(blue_path, 'NAME') == 
                        pf.getval(blue_path, 'STDNAME')):
                    result.append((blue_path, red_path))
    return result

def red_cube_path(blue_path):
    """Return the corresponding red cube path matched to a blue cube path."""
    start = blue_path.rfind('blue')
    red_path = blue_path[:start] + 'red' + blue_path[start+4:]
    return red_path

def fit_template(file_pair, model_catalogue):
    """Fit a stellar template to a pair of datacubes and infer the g-r."""
    flux, noise, wavelength = extract_stellar_spectrum(file_pair)
    flux, noise, count = fluxcal2.rebin_flux_noise(
        model_catalogue['wavelength'], wavelength, flux, noise)
    wavelength = model_catalogue['wavelength']
    good = np.isfinite(flux) & np.isfinite(noise)
    best_chisq = np.inf
    best_scale = np.nan
    best_model = np.nan
    for model_flux in model_catalogue['flux']:
        scale = (np.sum(flux[good] * model_flux[good] / noise[good]**2) / 
                 np.sum(model_flux[good]**2 / noise[good]**2))
        if not np.isfinite(scale):
            # This happens if the model flux is all zeros
            continue
        chisq = np.sum(((flux[good] - scale * model_flux[good]) / 
                        noise[good])**2)
        if chisq < best_chisq:
            best_chisq = chisq
            best_scale = scale
            best_model = model_flux
    return flux, noise, wavelength, best_chisq, best_scale, best_model

def extract_stellar_spectrum(file_pair):
    """Return the spectrum of a star, assumed to be at the centre."""
    # Replace the hard-coded numbers with something smarter
    x, y = np.meshgrid(0.5*(np.arange(50)-24.5), 0.5*(np.arange(50)-24.5))
    keep_x, keep_y = np.where(x**2 + y**2 < 2.0**2)
    flux_cube = np.vstack((pf.getdata(file_pair[0]), pf.getdata(file_pair[1])))
    variance_cube = np.vstack((pf.getdata(file_pair[0], 'VARIANCE'), 
                               pf.getdata(file_pair[1], 'VARIANCE')))
    flux = np.nansum(flux_cube[:, keep_x, keep_y], axis=1)
    # Doesn't include co-variance - Nic will provide code
    noise = np.sqrt(np.nansum(variance_cube[:, keep_x, keep_y], axis=1))
    # Fudge for co-variance
    noise *= 2.0
    wavelength = np.hstack((get_coords(pf.getheader(file_pair[0]), 3),
                            get_coords(pf.getheader(file_pair[1]), 3)))
    return flux, noise, wavelength

def get_coords(header, axis):
    """Return coordinates for a given axis from a header."""
    axis_str = str(axis)
    naxis = header['NAXIS' + axis_str]
    crpix = header['CRPIX' + axis_str]
    cdelt = header['CDELT' + axis_str]
    crval = header['CRVAL' + axis_str]
    coords = crval + cdelt * (np.arange(naxis) + 1.0 - crpix)
    return coords

def rebin_spectrum(old_flux, old_noise, old_wavelength, new_wavelength):
    """Rebin a spectrum onto a coarser wavelength scale."""
    # This is really crude and needs to be improved. No fractional pixels are
    # dealt with; the input pixels are just assigned to one or another output
    # pixel.
    delta = new_wavelength[1] - new_wavelength[0]
    n_pix = len(new_wavelength)
    new_flux = np.zeros(n_pix)
    new_noise = np.zeros(n_pix)
    for i_pix, wave_i in enumerate(new_wavelength):
        in_range = ((old_wavelength > (wave_i - 0.5*delta)) &
                    (old_wavelength < (wave_i + 0.5*delta)) &
                    np.isfinite(old_flux) & np.isfinite(old_noise))
        new_flux[i_pix] = np.mean(old_flux[in_range])
        new_noise[i_pix] = (np.sqrt(np.sum(old_noise[in_range]**2)) / 
                            np.sum(in_range))
    return new_flux, new_noise, new_wavelength

def read_stellar_models(path_root='./ck04models/'):
    """Return a dictionary of all the stellar models available."""
    file_list = glob(os.path.join(path_root, '*', '*.fits'))
    n_model = pf.getval(os.path.join(path_root, 'catalog.fits'), 'NAXIS2', 1)
    n_pix = pf.getval(file_list[0], 'NAXIS2', 1)
    models = {}
    models['wavelength'] = pf.getdata(file_list[0], 1)['WAVELENGTH']
    models['flux'] = np.zeros((n_model, n_pix))
    i_model = 0
    for filename in file_list:
        data = pf.getdata(filename, 1)
        for name in (n for n in data.columns.names if n != 'WAVELENGTH'):
            models['flux'][i_model, :] = data[name]
            i_model += 1
    # There's an extra, irregular pixel at 3640.0A. I don't know why.
    keep = ((models['wavelength'] >= 3000) & 
            (models['wavelength'] < 10000) & 
            (models['wavelength'] != 3640.0))
    models['wavelength'] = models['wavelength'][keep]
    models['flux'] = models['flux'][:, keep]
    return models

def measure_colour(flux, wavelength, sdss_dir='./sdss/'):
    """Return synthetic SDSS g-r colour for a given spectrum."""
    return (measure_band('g', flux, wavelength, sdss_dir=sdss_dir) -
            measure_band('r', flux, wavelength, sdss_dir=sdss_dir))

def read_filter(band, sdss_dir='./sdss/'):
    """Return filter response and wavelength for an SDSS filter."""
    data = np.loadtxt(os.path.join(sdss_dir, band + '.dat'))
    wavelength = data[:, 0]
    # Response through airmass 0.0
    response = data[:, 3]
    return response, wavelength

def measure_band(band, flux, wavelength, sdss_dir='./sdss/'):
    """Return the synthetic magnitude of a spectrum in an SDSS band."""
    filter_response, filter_wavelength = read_filter(band, sdss_dir=sdss_dir)
    filter_interpolated = np.interp(
        wavelength, filter_wavelength, filter_response)
    flux_band = (np.sum(wavelength * filter_interpolated * flux) / 
                 np.sum(filter_interpolated / wavelength))
    return -2.5 * np.log10(flux_band)

