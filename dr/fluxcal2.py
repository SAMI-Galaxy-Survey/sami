"""Flux calibration code that looks at the red and blue data together.

The allowed models are:

-- ref_centre_alpha_angle --

The centre position and alpha are fit for the reference wavelength,
and the positions and alpha values are then determined by the known
alpha dependence and the DAR, with the zenith distance and direction
also as free parameters.

-- ref_centre_alpha_angle_circ --

The same as ref_centre_alpha_angle, but with the Moffat function
constrained to be circular.

-- ref_centre_alpha_angle_circ_atm --

The same as ref_centre_alpha_angle_circ, but with atmospheric values
as free parameters too.
"""

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.filters import median_filter
from scipy.stats.stats import nanmean

from astropy import coordinates as coord
from astropy import units
from astropy.io import fits as pf

from .. import utils
from ..utils.ifu import IFU

HG_CHANGESET = utils.hg_changeset(__file__)

REFERENCE_WAVELENGTH = 5000.0

FIBRE_RADIUS = 0.798

def generate_subgrid(fibre_radius, n_inner=6, n_rings=10):
    """Generate a subgrid of points within a fibre."""
    radii = np.arange(0., n_rings) + 0.5
    rot_angle = 0.0
    radius = []
    theta = []
    for i_ring, radius_ring in enumerate(radii):
        n_points = np.round(n_inner * radius_ring)
        theta_ring = (np.linspace(0.0, 2.0*np.pi, n_points, endpoint=False) + 
                      rot_angle)
        radius = np.hstack((radius, np.ones(n_points) * radius_ring))
        theta = np.hstack((theta, theta_ring))
        rot_angle += theta_ring[1] / 2.0
    radius *= fibre_radius / n_rings
    xsub = radius * np.cos(theta)
    ysub = radius * np.sin(theta)
    return xsub, ysub

XSUB, YSUB = generate_subgrid(FIBRE_RADIUS)
N_SUB = len(XSUB)

def read_chunked_data(path_list, probename, n_drop=None, n_chunk=None):
    """Read flux from a list of files, chunk it and combine."""
    if isinstance(path_list, str):
        path_list = [path_list]
    for i_file, path in enumerate(path_list):
        ifu = IFU(path, probename, flag_name=False)
        data_i, variance_i, wavelength_i = chunk_data(ifu, n_drop=n_drop, 
                                                      n_chunk=n_chunk)
        if i_file == 0:
            data = data_i
            variance = variance_i
            wavelength = wavelength_i
        else:
            data = np.hstack((data, data_i))
            variance = np.hstack((variance, variance_i))
            wavelength = np.hstack((wavelength, wavelength_i))
    xfibre = ifu.xpos_rel
    yfibre = ifu.ypos_rel
    return data, variance, wavelength, xfibre, yfibre

def chunk_data(ifu, n_drop=None, n_chunk=None):
    """Condence a spectrum into a number of chunks."""
    n_pixel = ifu.naxis1
    n_fibre = len(ifu.data)
    if n_drop is None:
        n_drop = 24
    if n_chunk is None:
        n_chunk = round((n_pixel - 2*n_drop) / 100.0)
    chunk_size = round((n_pixel - 2*n_drop) / n_chunk)
    start = n_drop
    end = n_drop + n_chunk * chunk_size
    data = ifu.data[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    variance = ifu.var[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    wavelength = ifu.lambda_range[start:end].reshape(n_chunk, chunk_size)
    data = nanmean(data, axis=2)
    variance = (np.nansum(variance, axis=2) / 
                np.sum(np.isfinite(variance), axis=2)**2)
    wavelength = np.median(wavelength, axis=1)
    return data, variance, wavelength

def moffat_normalised(parameters, xfibre, yfibre, simple=False):
    """Return model Moffat flux for a single slice in wavelength."""
    if simple:
        xterm = (xfibre - parameters['xcen']) / parameters['alphax']
        yterm = (yfibre - parameters['ycen']) / parameters['alphay']
        alphax = parameters['alphax']
        alphay = parameters['alphay']
        beta = parameters['beta']
        rho = parameters['rho']
        moffat = (((beta - 1.0) / 
                   (np.pi * alphax * alphay * np.sqrt(1.0 - rho**2))) * 
                  (1.0 + ((xterm**2 + yterm**2 - 2.0 * rho * xterm * yterm) /
                          (1.0 - rho**2))) ** (-1.0 * beta))
        return moffat * np.pi * FIBRE_RADIUS**2
    else:
        n_fibre = len(xfibre)
        xfibre_sub = (np.outer(XSUB, np.ones(n_fibre)) + 
                      np.outer(np.ones(N_SUB), xfibre))
        yfibre_sub = (np.outer(YSUB, np.ones(n_fibre)) + 
                      np.outer(np.ones(N_SUB), yfibre))
        flux_sub = moffat_flux_slice(parameters, xfibre_sub, yfibre_sub, 
                                     simple=True)
        return np.mean(flux_sub, axis=0)

def moffat_flux(parameters_array, xfibre, yfibre):
    """Return n_fibre X n_wavelength array of Moffat function flux values."""
    n_slice = len(parameters_array)
    n_fibre = len(xfibre)
    flux = np.zeros((n_fibre, n_slice))
    for i_slice, parameters_slice in enumerate(parameters_array):
        fibre_psf = moffat_normalised(parameters_slice, xfibre, yfibre)
        flux[:, i_slice] = (parameters_slice['flux'] * fibre_psf + 
                            parameters_slice['background'])
    return flux

def model_flux(parameters_dict, xfibre, yfibre, wavelength, model_name):
    """Return n_fibre X n_wavelength array of model flux values."""
    parameters_array = parameters_dict_to_array(parameters_dict, wavelength,
                                                model_name)
    return moffat_flux(parameters_array, xfibre, yfibre)

def residual(parameters_vector, datatube, vartube, xfibre, yfibre,
             wavelength, model_name):
    """Return the residual in each fibre for the given model."""
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)
    model = model_flux(parameters_dict, xfibre, yfibre, wavelength, model_name)
    return np.ravel((model - datatube) / np.sqrt(vartube))

def fit_model_flux(datatube, vartube, xfibre, yfibre, wavelength, model_name):
    """Fit a model to the given datatube."""
    par_0_dict = first_guess_parameters(datatube, vartube, xfibre, yfibre, 
                                        wavelength, model_name)
    par_0_vector = parameters_dict_to_vector(par_0_dict, model_name)
    args = (datatube, vartube, xfibre, yfibre, wavelength, model_name)
    parameters_vector = leastsq(residual, par_0_vector, args=args)[0]
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)
    return parameters_dict

def first_guess_parameters(datatube, vartube, xfibre, yfibre, wavelength, 
                           model_name):
    """Return a first guess to the parameters that will be fitted."""
    par_0 = {}
    if model_name == 'ref_centre_alpha_angle':
        weighted_data = np.sum(datatube / vartube, axis=1)
        weighted_data /= np.sum(weighted_data)
        par_0['flux'] = np.nansum(datatube, axis=0)
        par_0['background'] = np.zeros(len(par_0['flux']))
        par_0['xcen_ref'] = np.sum(xfibre * weighted_data)
        par_0['ycen_ref'] = np.sum(yfibre * weighted_data)
        par_0['zenith_direction'] = np.pi / 4.0
        par_0['zenith_distance'] = np.pi / 8.0
        par_0['alphax_ref'] = 1.0
        par_0['alphay_ref'] = 1.0
        par_0['beta'] = 4.0
        par_0['rho'] = 0.0
    elif model_name == 'ref_centre_alpha_angle_circ':
        weighted_data = np.sum(datatube / vartube, axis=1)
        weighted_data /= np.sum(weighted_data)
        par_0['flux'] = np.nansum(datatube, axis=0)
        par_0['background'] = np.zeros(len(par_0['flux']))
        par_0['xcen_ref'] = np.sum(xfibre * weighted_data)
        par_0['ycen_ref'] = np.sum(yfibre * weighted_data)
        par_0['zenith_direction'] = np.pi / 4.0
        par_0['zenith_distance'] = np.pi / 8.0
        par_0['alpha_ref'] = 1.0
        par_0['beta'] = 4.0
    elif model_name == 'ref_centre_alpha_angle_circ_atm':
        weighted_data = np.sum(datatube / vartube, axis=1)
        weighted_data /= np.sum(weighted_data)
        par_0['flux'] = np.nansum(datatube, axis=0)
        par_0['background'] = np.zeros(len(par_0['flux']))
        par_0['temperature'] = 7.0
        par_0['pressure'] = 600.0
        par_0['vapour_pressure'] = 8.0
        par_0['xcen_ref'] = np.sum(xfibre * weighted_data)
        par_0['ycen_ref'] = np.sum(yfibre * weighted_data)
        par_0['zenith_direction'] = np.pi / 4.0
        par_0['zenith_distance'] = np.pi / 8.0
        par_0['alpha_ref'] = 1.0
        par_0['beta'] = 4.0
    else:
        raise KeyError('Unrecognised model name: ' + model_name)
    return par_0

def parameters_dict_to_vector(parameters_dict, model_name):
    """Convert a parameters dictionary to a vector."""
    if model_name == 'ref_centre_alpha_angle':
        parameters_vector = np.hstack(
            (parameters_dict['flux'],
             parameters_dict['background'],
             parameters_dict['xcen_ref'],
             parameters_dict['ycen_ref'],
             parameters_dict['zenith_direction'],
             parameters_dict['zenith_distance'],
             parameters_dict['alphax_ref'],
             parameters_dict['alphay_ref'],
             parameters_dict['beta'],
             parameters_dict['rho']))
    elif model_name == 'ref_centre_alpha_angle_circ':
        parameters_vector = np.hstack(
            (parameters_dict['flux'],
             parameters_dict['background'],
             parameters_dict['xcen_ref'],
             parameters_dict['ycen_ref'],
             parameters_dict['zenith_direction'],
             parameters_dict['zenith_distance'],
             parameters_dict['alpha_ref'],
             parameters_dict['beta']))
    elif model_name == 'ref_centre_alpha_angle_circ_atm':
        parameters_vector = np.hstack(
            (parameters_dict['flux'],
             parameters_dict['background'],
             parameters_dict['temperature'],
             parameters_dict['pressure'],
             parameters_dict['vapour_pressure'],
             parameters_dict['xcen_ref'],
             parameters_dict['ycen_ref'],
             parameters_dict['zenith_direction'],
             parameters_dict['zenith_distance'],
             parameters_dict['alpha_ref'],
             parameters_dict['beta']))
    else:
        raise KeyError('Unrecognised model name: ' + model_name)
    return parameters_vector

def parameters_vector_to_dict(parameters_vector, model_name):
    """Convert a parameters vector to a dictionary."""
    parameters_dict = {}
    if model_name == 'ref_centre_alpha_angle':
        n_slice = (len(parameters_vector) - 8) / 2
        parameters_dict['flux'] = parameters_vector[0:n_slice]
        parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
        parameters_dict['xcen_ref'] = parameters_vector[-8]
        parameters_dict['ycen_ref'] = parameters_vector[-7]
        parameters_dict['zenith_direction'] = parameters_vector[-6]
        parameters_dict['zenith_distance'] = parameters_vector[-5]
        parameters_dict['alphax_ref'] = parameters_vector[-4]
        parameters_dict['alphay_ref'] = parameters_vector[-3]
        parameters_dict['beta'] = parameters_vector[-2]
        parameters_dict['rho'] = parameters_vector[-1]
    elif model_name == 'ref_centre_alpha_angle_circ':
        n_slice = (len(parameters_vector) - 6) / 2
        parameters_dict['flux'] = parameters_vector[0:n_slice]
        parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
        parameters_dict['xcen_ref'] = parameters_vector[-6]
        parameters_dict['ycen_ref'] = parameters_vector[-5]
        parameters_dict['zenith_direction'] = parameters_vector[-4]
        parameters_dict['zenith_distance'] = parameters_vector[-3]
        parameters_dict['alpha_ref'] = parameters_vector[-2]
        parameters_dict['beta'] = parameters_vector[-1]
    elif model_name == 'ref_centre_alpha_angle_circ_atm':
        n_slice = (len(parameters_vector) - 9) / 2
        parameters_dict['flux'] = parameters_vector[0:n_slice]
        parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
        parameters_dict['temperature'] = parameters_vector[-9]
        parameters_dict['pressure'] = parameters_vector[-8]
        parameters_dict['vapour_pressure'] = parameters_vector[-7]
        parameters_dict['xcen_ref'] = parameters_vector[-6]
        parameters_dict['ycen_ref'] = parameters_vector[-5]
        parameters_dict['zenith_direction'] = parameters_vector[-4]
        parameters_dict['zenith_distance'] = parameters_vector[-3]
        parameters_dict['alpha_ref'] = parameters_vector[-2]
        parameters_dict['beta'] = parameters_vector[-1]
    else:
        raise KeyError('Unrecognised model name: ' + model_name)
    return parameters_dict

def parameters_dict_to_array(parameters_dict, wavelength, model_name):
    parameter_names = ('xcen ycen alphax alphay beta rho flux '
                       'background'.split())
    formats = ['float64'] * len(parameter_names)
    parameters_array = np.zeros(len(wavelength), 
                                dtype={'names':parameter_names, 
                                       'formats':formats})
    if model_name == 'ref_centre_alpha_angle':
        parameters_array['xcen'] = (
            parameters_dict['xcen_ref'] + 
            np.cos(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] + 
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['alphax'] = (
            alpha(wavelength, parameters_dict['alphax_ref']))
        parameters_array['alphay'] = (
            alpha(wavelength, parameters_dict['alphay_ref']))
        parameters_array['beta'] = parameters_dict['beta']
        parameters_array['rho'] = parameters_dict['rho']
        parameters_array['flux'] = parameters_dict['flux']
        parameters_array['background'] = parameters_dict['background']
    elif model_name == 'ref_centre_alpha_angle_circ':
        parameters_array['xcen'] = (
            parameters_dict['xcen_ref'] + 
            np.cos(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] + 
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['alphax'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['alphay'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['beta'] = parameters_dict['beta']
        parameters_array['rho'] = np.zeros(len(wavelength))
        parameters_array['flux'] = parameters_dict['flux']
        parameters_array['background'] = parameters_dict['background']
    elif model_name == 'ref_centre_alpha_angle_circ_atm':
        parameters_array['xcen'] = (
            parameters_dict['xcen_ref'] + 
            np.cos(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance'],
                temperature=parameters_dict['temperature'],
                pressure=parameters_dict['pressure'],
                vapour_pressure=parameters_dict['vapour_pressure']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] + 
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance'],
                temperature=parameters_dict['temperature'],
                pressure=parameters_dict['pressure'],
                vapour_pressure=parameters_dict['vapour_pressure']))
        parameters_array['alphax'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['alphay'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['beta'] = parameters_dict['beta']
        parameters_array['rho'] = np.zeros(len(wavelength))
        parameters_array['flux'] = parameters_dict['flux']
        parameters_array['background'] = parameters_dict['background']
    else:
        raise KeyError('Unrecognised model name: ' + model_name)
    return parameters_array

def alpha(wavelength, alpha_ref):
    """Return alpha at the specified wavelength(s)."""
    return alpha_ref * ((wavelength / REFERENCE_WAVELENGTH)**(-0.2))

def dar(wavelength, zenith_distance, temperature=None, pressure=None, 
        vapour_pressure=None):
    """Return the DAR offset in arcseconds at the specified wavelength(s)."""
    # Analytic expectations from Fillipenko (1982)
    n_observed = refractive_index(
        wavelength, temperature, pressure, vapour_pressure)
    n_reference = refractive_index(
        REFERENCE_WAVELENGTH, temperature, pressure, vapour_pressure)
    return 206265. * (n_observed - n_reference) * np.tan(zenith_distance)

def refractive_index(wavelength, temperature=None, pressure=None, 
                     vapour_pressure=None):
    """Return the refractive index at the specified wavelength(s)."""
    # Analytic expectations from Fillipenko (1982)
    if temperature is None:
        temperature = 7.
    if pressure is None:
        pressure = 600.
    if vapour_pressure is None:
        vapour_pressure = 8.
    # Convert wavelength from Angstroms to microns
    wl = wavelength * 1e-4
    seaLevelDry = 1e-6 * ( 64.328 + ( 29498.1 / ( 146. - ( 1 / wl**2. ) ) )
                           + 255.4 / ( 41. - ( 1. / wl**2. ) ) )
    altitudeCorrection = ( 
        ( pressure * ( 1. + (1.049 - 0.0157*temperature ) * 1e-6 * pressure ) )
        / ( 720.883 * ( 1. + 0.003661 * temperature ) ) )
    vapourCorrection = ( ( 0.0624 - 0.000680 / wl**2. )
                         / ( 1. + 0.003661 * temperature ) ) * vapour_pressure
    return seaLevelDry * altitudeCorrection * vapourCorrection + 1
