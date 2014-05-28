"""Functions relating to quality control of flux calibration."""

from ..manager import Manager
from ..dr import fluxcal2
from ..utils import IFU

import astropy.io.fits as pf
import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.optimize import leastsq

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
    ccd_list = ['ccd_1', 'ccd_2']
    result = [{} for ccd in ccd_list]
    for mngr in mngr_list:
        for index, ccd in enumerate(ccd_list):
            groups = mngr.group_files_by(('date', 'field_id', 'ccd', 'name'),
                ndf_class='MFOBJECT', do_not_use=False,
                spectrophotometric=True, ccd=ccd)
            for group in groups.values():
                combined = os.path.join(
                    group[0].reduced_dir, 'TRANSFERcombined.fits')
                if os.path.exists(combined):
                    result[index][combined] = [f.reduced_path for f in group]
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


# def stellar_colours(mngr):
#     """
#     Return stellar colours as measured by SAMI (via a template fit) compared
#     to the SDSS imaging.
#     """
#     file_pair_list = list_star_files(mngr)
#     model_catalogue = read_stellar_models()
#     model_list = [fit_template(file_pair, model_catalogue)[-1]
#                   for file_pair in file_pair_list]
#     observed_colours = [measure_colour(model, model_catalogue['wavelength']) 
#                         for model in model_list]
#     return file_pair_list, observed_colours


def stellar_mags(mngr):
    """
    Return stellar magnitudes as measured by SAMI (via interpolation), for
    the datacubes and the input files.
    """
    file_pair_list, frame_pair_list_list = list_star_files(mngr)
    mag_cube = []
    mag_frame = []
    for file_pair, frame_pair_list in zip(
            file_pair_list, frame_pair_list_list):
        flux, noise, wavelength = extract_stellar_spectrum(file_pair)
        mag_cube.append(measure_mags(flux, noise, wavelength))
        mag_frame.append([])
        for frame_pair in frame_pair_list:
            flux, noise, wavelength = read_stellar_spectrum(frame_pair)
            mag_frame[-1].append(measure_mags(flux, noise, wavelength))
    return file_pair_list, frame_pair_list_list, mag_cube, mag_frame

def list_star_files(mngr):
    """
    Return a list of tuples of paths to star datacubes, blue and red,
    as well as a list of lists of tuples of paths to individual frames.
    """
    if isinstance(mngr, str):
        mngr = Manager(mngr)
    if isinstance(mngr, Manager):
        mngr_list = [mngr]
    else:
        mngr_list = mngr
    result = []
    frame = []
    for mngr in mngr_list:
        blue_list = (
            glob(os.path.join(mngr.abs_root, 'cubed', '*', '*blue*.fits')) +
            glob(os.path.join(mngr.abs_root, 'cubed', '*', '*blue*.fits.gz')))
            # [])
        for blue_path in blue_list:
            red_path = red_cube_path(blue_path)
            if os.path.exists(red_path):
                blue_header = pf.getheader(blue_path)
                if blue_header['NAME'] == blue_header['STDNAME']:
                    print blue_path
                    result.append((blue_path, red_path))
                    i = 0
                    frame.append([])
                    red_header = pf.getheader(red_path)
                    while True:
                        i += 1
                        try:
                            blue_filename = blue_header['RSS_FILE ' + str(i)]
                            red_filename = red_header['RSS_FILE ' + str(i)]
                        except KeyError:
                            break
                        blue_frame_path = glob(
                            mngr.abs_root+'/reduced/*/*/*/*/*/'+
                            blue_filename)[0]
                        red_frame_path = glob(
                            mngr.abs_root+'/reduced/*/*/*/*/*/'+
                            red_filename)[0]
                        frame[-1].append((blue_frame_path, red_frame_path))
    return result, frame

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
    flux_cube = np.vstack((pf.getdata(file_pair[0]), pf.getdata(file_pair[1])))
    variance_cube = np.vstack((pf.getdata(file_pair[0], 'VARIANCE'), 
                               pf.getdata(file_pair[1], 'VARIANCE')))
    noise_cube = np.sqrt(variance_cube)
    good = np.isfinite(flux_cube) & np.isfinite(variance_cube)
    image = np.nansum(flux_cube * good, 0) / np.sum(good, 0)
    noise_im = np.sqrt(np.nansum(variance_cube * good, 0)) / np.sum(good, 0)
    wavelength = np.hstack((get_coords(pf.getheader(file_pair[0]), 3),
                            get_coords(pf.getheader(file_pair[1]), 3)))
    psf_params = fit_moffat_to_image(image, noise_im)
    flux = np.zeros(len(wavelength))
    for i_pix, (image_slice, noise_slice) in enumerate(
            zip(flux_cube, noise_cube)):
        flux[i_pix] = scale_moffat_to_image(
            image_slice, noise_slice, psf_params)
    # FIX THIS!
    noise = np.ones(len(wavelength))
    return flux, noise, wavelength, psf_params

def extract_galaxy_spectrum(file_pair):
    """Return the spectrum of a galaxy, assumed to cover the IFU."""
    return sum_spectrum_from_cube(file_pair, 7.0)

def sum_spectrum_from_cube(file_pair, radius):
    """Return the summed spectrum from spaxels in the centre of a cube."""
    # Replace the hard-coded numbers with something smarter
    x, y = np.meshgrid(0.5*(np.arange(50)-24.5), 0.5*(np.arange(50)-24.5))
    keep_x, keep_y = np.where(x**2 + y**2 < radius**2)
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

def read_stellar_spectrum(file_pair):
    """Read and return the measured spectrum of a star from a single frame."""
    flux = []
    noise = []
    wavelength = []
    for path in file_pair:
        hdulist = pf.open(path)
        flux.append(hdulist['FLUX_CALIBRATION'].data[0, :])
        noise.append(hdulist['FLUX_CALIBRATION'].data[2, :])
        header = hdulist[0].header
        wavelength.append(get_coords(header, 1))
    flux = np.hstack(flux)
    noise = np.hstack(noise)
    wavelength = np.hstack(wavelength)
    return flux, noise, wavelength

def read_galaxy_spectrum(file_pair, name):
    """Read and return the summed spectrum of a galaxy from a single frame."""
    # Just sums over the fibres, which misses ~25% of the light
    flux = []
    noise = []
    wavelength = []
    for path in file_pair:
        ifu = IFU(path, name)
        flux.append(np.nansum(ifu.data, 0))
        noise.append(np.sqrt(np.nansum(ifu.var, 0)))
        wavelength.append(ifu.lambda_range)
    flux = np.hstack(flux)
    noise = np.hstack(noise)
    wavelength = np.hstack(wavelength)
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
    # Convert to SI units. Wavelength was in A, flux was in 1e-16 erg/s/cm^2/A
    wl_m = wavelength * 1e-10
    flux_wm3 = flux * 1e-16 * 1e-7 * (1e2)**2 * 1e10
    # AB magnitudes are zero for flux of 3631 Jy
    flux_zero = 3631.0 * 1.0e-26 * 2.99792458e8 / (wl_m**2)
    # Get the wavelength bin sizes - don't assume constant!
    delta_wl = np.hstack((
        wl_m[1] - wl_m[0],
        0.5 * (wl_m[2:] - wl_m[:-2]),
        wl_m[-1] - wl_m[-2]))
    flux_band = (np.sum(delta_wl * wl_m * filter_interpolated * flux_wm3) / 
                 np.sum(delta_wl * wl_m * filter_interpolated * flux_zero))
    return -2.5 * np.log10(flux_band)

def measure_mags(flux, noise, wavelength):
    """Do clipping and interpolation, then return g and r band mags."""
    good = clip_spectrum(flux, noise, wavelength)
    flux, noise, wavelength = interpolate_arms(
        flux, noise, wavelength, good)
    mag_g = measure_band('g', flux, wavelength)
    mag_r = measure_band('r', flux, wavelength)
    return (mag_g, mag_r)

def clip_spectrum(flux, noise, wavelength):
    """Return a "good" array, clipping mostly based on discrepant noise."""
    filter_width = 21
    limit = 0.35
    filtered_noise = median_filter(noise, filter_width)
    # Only clipping positive deviations - negative deviations are mostly due
    # to absorption lines so should be left in
    # noise_ratio = (noise - filtered_noise) / filtered_noise
    # Clipping both negative and positive values, even though this means
    # clipping out several absorption lines
    noise_ratio = np.abs((noise - filtered_noise) / filtered_noise)
    good = (np.isfinite(flux) &
            np.isfinite(noise) &
            (noise_ratio < limit))
    return good

def interpolate_arms(flux, noise, wavelength, good=None, n_pix_fit=300):
    """Interpolate between the red and blue arms."""
    # Establish basic facts about which pixels we should look at
    n_pix = len(wavelength)
    if good is None:
        good = np.arange(n_pix)
    middle = n_pix / 2
    good_blue = good & (np.arange(n_pix) < middle)
    good_red = good & (np.arange(n_pix) >= middle)
    wavelength_middle = 0.5 * (wavelength[middle-1] + wavelength[middle])
    delta_wave_blue = wavelength[1] - wavelength[0]
    delta_wave_red = wavelength[-1] - wavelength[-2]
    # Get the flux from the red end of the blue and the blue end of the red,
    # and fit a straight line between them
    index_blue = np.where(good_blue)[0][-n_pix_fit:]
    index_red = np.where(good_red)[0][:n_pix_fit]
    index_fit = np.hstack((index_blue, index_red))
    poly_params = np.polyfit(wavelength[index_fit], flux[index_fit], 1,
                             w=1.0/noise[index_fit]**2)
    wavelength_start = wavelength[middle-1] + delta_wave_blue
    n_pix_insert_blue = int(np.round(
        (wavelength_middle - wavelength_start) / delta_wave_blue))
    wavelength_end = wavelength[middle]
    n_pix_insert_red = int(np.round(
        (wavelength_end - wavelength_middle) / delta_wave_red))
    n_pix_insert = n_pix_insert_red + n_pix_insert_blue
    wavelength_insert = np.hstack((
        np.linspace(wavelength_start, wavelength_middle, n_pix_insert_blue,
                    endpoint=False),
        np.linspace(wavelength_middle, wavelength_end, n_pix_insert_red,
                    endpoint=False)))
    wavelength_out = np.hstack(
        (wavelength[:middle], wavelength_insert, wavelength[middle:]))
    flux_out = np.hstack(
        (flux[:middle], np.zeros(n_pix_insert), flux[middle:]))
    noise_out = np.hstack(
        (noise[:middle], np.zeros(n_pix_insert), noise[middle:]))
    insert = ((wavelength_out > wavelength[index_blue[-1]]) &
              (wavelength_out < wavelength[index_red[0]]))
    flux_out[insert] = np.polyval(poly_params, wavelength_out[insert])
    noise_out[insert] = np.nan
    interp = (~np.isfinite(flux_out) | ~np.isfinite(noise_out)) & ~insert
    flux_out[interp] = np.interp(
        wavelength_out[interp], wavelength_out[~interp], flux_out[~interp])
    noise_out[interp] = np.nan
    return flux_out, noise_out, wavelength_out

def collapse_cube(flux, noise, wavelength, good=None, n_band=1):
    """Collapse a cube into a 2-d image, or a series of images."""
    # Be careful about sending in mixed red+blue cubes - in this case
    # n_band should be even
    n_pix = len(wavelength)
    if good is None:
        good = np.arange(flux.size)
        good.shape = flux.shape
    flux_out = np.zeros((n_band, flux.shape[1], flux.shape[2]))
    noise_out = np.zeros((n_band, flux.shape[1], flux.shape[2]))
    wavelength_out = np.zeros(n_band)
    for i_band in xrange(n_band):
        start = i_band * n_pix / n_band
        finish = (i_band+1) * n_pix / n_band
        flux_out[i_band, :, :] = (
            np.nansum(flux[start:finish, :, :] * good[start:finish, :, :], 0) /
            np.sum(good[start:finish, :, :], 0))
        noise_out[i_band, :, :] = (
            np.sqrt(np.nansum(noise[start:finish, :, :]**2 * 
                              good[start:finish, :, :], 0)) /
            np.sum(good[start:finish, :, :], 0))
        wavelength_out[i_band] = (
            0.5 * (wavelength[start] + wavelength[finish-1]))
    flux_out = np.squeeze(flux_out)
    noise_out = np.squeeze(noise_out)
    wavelength_out = np.squeeze(wavelength_out)
    return flux_out, noise_out, wavelength_out

def fit_moffat_to_image(image, noise, elliptical=True):
    """Fit a Moffat profile to an image, optionally allowing ellipticity."""
    fit_pix = np.isfinite(image) & np.isfinite(noise)
    coords = np.meshgrid(np.arange(image.shape[0]), 
                         np.arange(image.shape[1]))
    x00 = 0.5 * (image.shape[0] - 1)
    y00 = 0.5 * (image.shape[1] - 1)
    alpha0 = 4.0
    beta0 = 4.0
    intensity0 = np.nansum(image)
    if elliptical:
        p0 = [alpha0, alpha0, 0.0, beta0, x00, y00, intensity0]
    else:
        p0 = [alpha0, beta0, x00, y00, intensity0]
    def fit_function(p):
        model = moffat_integrated(
            coords[0], coords[1], p, elliptical=elliptical, good=fit_pix)
        return ((model - image[fit_pix]) / noise[fit_pix])
    params = leastsq(fit_function, p0, full_output=True)[0]
    return params

def scale_moffat_to_image(image, noise, params, elliptical=True):
    """Scale a Moffat profile to fit the provided image."""
    fit_pix = np.isfinite(image) & np.isfinite(noise)
    if np.sum(fit_pix) == 0:
        return np.nan
    coords = np.meshgrid(np.arange(image.shape[0]), 
                         np.arange(image.shape[1]))
    params_norm = params.copy()
    params_norm[-1] = 1.0
    model_norm = moffat_integrated(
        coords[0], coords[1], params_norm, elliptical=elliptical, good=fit_pix)
    p0 = [np.nansum(image)]
    def fit_function(p):
        model = p[0] * model_norm
        return ((model - image[fit_pix]) / noise[fit_pix])
    intensity = leastsq(fit_function, p0, full_output=True)[0][0]
    return intensity

def moffat_integrated(x, y, params, elliptical=True, good=None, pix_size=1.0,
                      n_sub=10):
    """Return a Moffat profile, integrated over pixels."""
    if good is None:
        good = np.ones(x.size, bool)
        good.shape = x.shape
    n_pix = np.sum(good)
    x_flat = x[good]
    y_flat = y[good]
    delta = pix_size * (np.arange(float(n_sub)) / n_sub)
    delta -= np.mean(delta)
    x_sub = (np.outer(x_flat, np.ones(n_sub**2)) + 
             np.outer(np.ones(n_pix), np.outer(delta, np.ones(n_sub))))
    y_sub = (np.outer(y_flat, np.ones(n_sub**2)) + 
             np.outer(np.ones(n_pix), np.outer(np.ones(n_sub), delta)))
    if elliptical:
        moffat_sub = moffat_elliptical(x_sub, y_sub, *params)
    else:
        moffat_sub = moffat_circular(x_sub, y_sub, *params)
    moffat = np.mean(moffat_sub, 1)
    return moffat


def moffat_elliptical(x, y, alpha_x, alpha_y, rho, beta, x0, y0, intensity):
    """Return an elliptical Moffat profile."""
    norm = (beta - 1) / (np.pi * alpha_x * alpha_y * np.sqrt(1 - rho**2))
    norm = norm * intensity
    x_term = (x - x0) / alpha_x
    y_term = (y - y0) / alpha_y
    moffat = norm * (1 + (x_term**2 + y_term**2 - 2*rho*x_term*y_term) /
                         (1 - rho**2))**(-beta)
    return moffat

def moffat_circular(x, y, alpha, beta, x0, y0, intensity):
    """Return a circular Moffat profile."""
    norm = (beta - 1) / (np.pi * alpha**2)
    norm = norm * intensity
    x_term = (x - x0) / alpha
    y_term = (y - y0) / alpha
    moffat = norm * (1 + (x_term**2 + y_term**2))**(-beta)
    return moffat






