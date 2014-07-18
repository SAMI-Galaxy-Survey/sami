"""Quality control of cubing and datacubes."""

from .fluxcal import fit_moffat_to_image, get_coords
from ..utils.other import clip_spectrum

import numpy as np
import astropy.io.fits as pf

import itertools

def measure_position_at_wavelength(cube, variance, wavelength_array,
                                   central_wavelength, n_pix):
    """Return x, y position in pixels for cube at given wavelength."""
    n_wave, n_x, n_y = cube.shape
    good = np.zeros(cube.shape)
    for i_x, i_y in itertools.product(xrange(n_x), xrange(n_y)):
        good[:, i_x, i_y] = clip_spectrum(
            cube[:, i_x, i_y], np.sqrt(variance[:, i_x, i_y]),
            wavelength_array)
    in_range = (
        np.argsort(np.abs(wavelength_array - central_wavelength))[:n_pix])
    cube_cut = cube[in_range, :, :]
    variance_cut = variance[in_range, :, :]
    good_cut = good[in_range, :, :]
    image = np.nansum(cube_cut * good_cut, axis=0) / np.sum(good_cut, axis=0)
    noise = (np.sqrt(np.nansum(variance_cut * good_cut, axis=0)) / 
             np.sum(good_cut, axis=0))
    image[noise == 0] = np.nan
    noise[noise == 0] = np.nan
    psf_params, sigma_psf_params = fit_moffat_to_image(
        image, noise, elliptical=False)
    x, y = psf_params[2], psf_params[3]
    if sigma_psf_params is None:
        sigma_x = sigma_y = None
    else:
        sigma_x, sigma_y = sigma_psf_params[2], sigma_psf_params[3]
    return x, y, sigma_x, sigma_y

def measure_dar(file_pair, wavelength=(4200.0, 7100.0), n_pix=100):
    """Return the offset between red/blue wavelengths for a star."""
    hdulist = pf.open(file_pair[0])
    cube = hdulist[0].data
    variance = hdulist['VARIANCE'].data
    wavelength_array = get_coords(hdulist[0].header, 3)
    hdulist.close()
    x_0, y_0, sigma_x_0, sigma_y_0 = measure_position_at_wavelength(
        cube, variance, wavelength_array, wavelength[0], n_pix)
    hdulist = pf.open(file_pair[1])
    cube = hdulist[0].data
    variance = hdulist['VARIANCE'].data
    wavelength_array = get_coords(hdulist[0].header, 3)
    hdulist.close()
    x_1, y_1, sigma_x_1, sigma_y_1 = measure_position_at_wavelength(
        cube, variance, wavelength_array, wavelength[1], n_pix)
    delta = np.sqrt((x_0 - x_1)**2 + (y_0 - y_1)**2)
    angle = np.arctan2(y_1 - y_0, x_1 - x_0)
    if (sigma_x_0 is None or sigma_x_1 is None or
            sigma_y_0 is None or sigma_y_1 is None):
        sigma_delta = None
    else:
        sigma_delta = (
            (2*(x_0 - x_1))**2 * (sigma_x_0**2 + sigma_x_1**2) +
            (2*(y_0 - y_1))**2 * (sigma_y_0**2 + sigma_y_1**2)
            ) / (2 * delta)
    return delta, sigma_delta, angle

def measure_sn_continuum(path, radius=1.0):
    """
    Return the mean continuum S/N across the fibres within the radius.

    Radius is given in arcseconds. In each fibre the S/N is defined as the
    median value across all pixels."""
    hdulist = pf.open(path)
    flux = hdulist[0].data
    noise = np.sqrt(hdulist['VARIANCE'].data)
    n_x = flux.shape[2]
    n_y = flux.shape[1]
    x, y = np.meshgrid(0.5*(np.arange(n_x)-0.5*(n_x-1)),
                       0.5*(np.arange(n_y)-0.5*(n_y-1)))
    x_ind, y_ind = np.where((x**2 + y**2) <= radius**2)
    return np.nanmean(np.median(flux[:, y_ind, x_ind] /
                                noise[:, y_ind, x_ind], axis=0))
