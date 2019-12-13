"""
Flux calibration code that looks at the red and blue data together.

This code replaces the original fluxcal module. The key functionality
(and the most difficult part to develop) is to infer the "total" observed
spectrum of a star based on the light in the fibres, i.e. to work out how
much light was lost between the fibres. This is done based on a model
that incorporates our understanding of how the atmosphere affects light,
both in terms of the PSF and the atmospheric refraction. A few different
models are available, with different amounts of freedom.

The allowed models are:

-- ref_centre_alpha_angle --

The centre position and alpha are fit for the reference wavelength,
and the positions and alpha values are then determined by the known
alpha dependence and the DAR, with the zenith distance and direction
also as free parameters.

-- ref_centre_alpha_angle_circ --

The same as ref_centre_alpha_angle, but with the Moffat function
constrained to be circular.

-- ref_centre_alpha_dist_circ --

The same as ref_centre_alpha_angle_circ, but with the zenith direction
fixed.

-- ref_centre_alpha_angle_circ_atm --

The same as ref_centre_alpha_angle_circ, but with atmospheric values
as free parameters too. Note, however, that the atmospheric parameters
are completely degenerate with each other and with ZD.

-- ref_centre_alpha_dist_circ_hdratm --

As ref_centre_alpha_dist_circ, but uses atmospheric values read from the
FITS header instead of the default values.

-- ref_centre_alpha_circ_hdratm --

Uses a circular Moffat function, fixed zenith distance and atmospheric
values from the FITS header.

Other than the functions for reading parameters in and out, the
functionality for doing the actual fitting is the same for all models,
so can be extended for further models quite straightforwardly.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import warnings

import numpy as np
from scipy.optimize import leastsq, curve_fit
from scipy.interpolate import LSQUnivariateSpline
from scipy.ndimage.filters import median_filter, gaussian_filter1d
from scipy.ndimage import zoom

from astropy import coordinates as coord
from astropy import units
from astropy import table
from astropy.io import fits as pf
from astropy.io import ascii
from astropy import __version__ as ASTROPY_VERSION
# extra astropy bits to calculate airmass
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.special import erfc
from scipy.stats import binned_statistic
from shutil import copyfile
 
# required for test plotting:
import pylab as py

from ..utils import hg_changeset
from ..utils.ifu import IFU
from ..utils.mc_adr import parallactic_angle, adr_r
from ..utils.other import saturated_partial_pressure_water
from ..config import millibar_to_mmHg
from ..utils.fluxcal2_io import read_model_parameters, save_extracted_flux
from .telluric2 import TelluricCorrectPrimary as telluric_correct_primary
from . import dust
#from ..manager import read_stellar_mags
#from ..qc.fluxcal import measure_band

try:
    from bottleneck import nansum, nanmean
except ImportError:
    from numpy import nansum, nanmean
    warnings.warn("Not Using bottleneck: Speed will be improved if you install bott    leneck")

# import of ppxf for fitting of secondary stds:
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
from ppxf.ppxf_util import log_rebin



# Get the astropy version as a tuple of integers
ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))

HG_CHANGESET = hg_changeset(__file__)

STANDARD_CATALOGUES = ('./standards/ESO/ESOstandards.dat',
                       './standards/Bessell/Bessellstandards.dat')
SSO_EXTINCTION_TABLE = './standards/extinsso.tab'

REFERENCE_WAVELENGTH = 5000.0

FIBRE_RADIUS = 0.798

TELLURIC_BANDS = np.array([[6850, 6960], 
                           [7130, 7360], 
                           [7560, 7770], 
                           [8100, 8360]])

def generate_subgrid(fibre_radius, n_inner=6, n_rings=10, wt_profile=False):
    """Generate a subgrid of points within a fibre."""
    radii = np.arange(0., n_rings) + 0.5
    rot_angle = 0.0
    radius = []
    theta = []
    for i_ring, radius_ring in enumerate(radii):
        n_points = np.int(np.round(n_inner * radius_ring))
        theta_ring = (np.linspace(0.0, 2.0*np.pi, n_points, endpoint=False) + 
                      rot_angle)
        radius = np.hstack((radius, np.ones(n_points) * radius_ring))
        theta = np.hstack((theta, theta_ring))
        rot_angle += theta_ring[1] / 2.0
    radius *= fibre_radius / n_rings
    xsub = radius * np.cos(theta)
    ysub = radius * np.sin(theta)
    # generate a weight for the points based on the radial profile.  In this case
    # we use an error function that goes to 0.5 at 0.8 of the radius of the fibre.
    # this is just experimental, no evidence it makes much improvement:
    if (wt_profile):
        wsub = 0.5*erfc((radius-fibre_radius*0.8)*4.0)
        wnorm = float(np.size(radius))/np.sum(wsub)
        wsub = wsub * wnorm
    else:
        # or unit weighting:
        wsub = np.ones(np.size(xsub)) 
    return xsub, ysub, wsub

XSUB, YSUB, WSUB= generate_subgrid(FIBRE_RADIUS)
N_SUB = len(XSUB)

def in_telluric_band(wavelength):
    """Return boolean array, True if in a telluric band."""
    retarray = np.zeros(np.shape(wavelength), dtype='bool')
    for band in TELLURIC_BANDS:
        retarray = retarray | ((wavelength >= band[0]) & 
                               (wavelength <= band[1]))
    return retarray

def read_chunked_data(path_list, probenum, n_drop=None, n_chunk=None,
                      sigma_clip=None):
    """Read flux from a list of files, chunk it and combine."""
    if isinstance(path_list, str):
        path_list = [path_list]
    for i_file, path in enumerate(path_list):
        ifu = IFU(path, probenum, flag_name=False)
        remove_atmosphere(ifu)
        data_i, variance_i, wavelength_i = chunk_data(
            ifu, n_drop=n_drop, n_chunk=n_chunk, sigma_clip=sigma_clip)
        if i_file == 0:
            data = data_i
            variance = variance_i
            wavelength = wavelength_i
        else:
            data = np.hstack((data, data_i))
            variance = np.hstack((variance, variance_i))
            wavelength = np.hstack((wavelength, wavelength_i))
    xfibre = ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos)))
    yfibre = ifu.ypos_rel
    # Only keep unbroken fibres
    good_fibre = (ifu.fib_type == 'P')
    chunked_data = {'data': data[good_fibre, :],
                    'variance': variance[good_fibre, :],
                    'wavelength': wavelength,
                    'xfibre': xfibre[good_fibre],
                    'yfibre': yfibre[good_fibre]}
    return chunked_data

def trim_chunked_data(chunked_data, n_trim):
    """Trim off the extreme blue end of the chunked data, because it's bad."""
    chunked_data['data'] = chunked_data['data'][:, n_trim:]
    chunked_data['variance'] = chunked_data['variance'][:, n_trim:]
    chunked_data['wavelength'] = chunked_data['wavelength'][n_trim:]
    return

def chunk_data(ifu, n_drop=None, n_chunk=None, sigma_clip=None):
    """Condence a spectrum into a number of chunks."""
    n_pixel = ifu.naxis1
    n_fibre = len(ifu.data)
    if n_drop is None:
        n_drop = 24
    if n_chunk is None:
        n_chunk = round((n_pixel - 2*n_drop) / 100.0)
    chunk_size = round((n_pixel - 2*n_drop) / n_chunk)
    if sigma_clip:
        good = np.isfinite(ifu.data)
        data_smooth = ifu.data.copy()
        data_smooth[~good] = np.median(ifu.data[good])
        data_smooth = median_filter(data_smooth, size=(1, 51))
        data_smooth[~good] = np.nan
        # Estimate of variance; don't trust 2dfdr values
        std_smooth = 1.4826 * np.median(np.abs(ifu.data[good] - 
                                               data_smooth[good]))
        data = ifu.data
        clip = abs(data - data_smooth) > (sigma_clip * std_smooth)
        data[clip] = data_smooth[clip]
    else:
        data = ifu.data

    # Convert to integer for future compatibility.
    n_chunk = np.int(np.floor(n_chunk))
    chunk_size = np.int(np.floor(chunk_size))

    start = n_drop
    end = n_drop + n_chunk * chunk_size

    data = data[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    variance = ifu.var[:, start:end].reshape(n_fibre, n_chunk, chunk_size)
    wavelength = ifu.lambda_range[start:end].reshape(n_chunk, chunk_size)
    data = nanmean(data, axis=2)
    variance = (np.nansum(variance, axis=2) / 
                np.sum(np.isfinite(variance), axis=2)**2)
    # Replace any remaining NaNs with 0.0; not ideal but should be very rare
    bad_data = ~np.isfinite(data)
    data[bad_data] = 0.0
    variance[bad_data] = np.inf
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
        wt_sub = (np.outer(WSUB, np.ones(n_fibre)))
        flux_sub = moffat_normalised(parameters, xfibre_sub, yfibre_sub, 
                                     simple=True)
        flux_sub = flux_sub * wt_sub
        
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
             wavelength, model_name, fixed_parameters=None, secondary=False):
    """Return the residual in each fibre for the given model."""
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)
    parameters_dict = insert_fixed_parameters(parameters_dict, 
                                              fixed_parameters)
    model = model_flux(parameters_dict, xfibre, yfibre, wavelength, model_name)
    # 2dfdr variance is just plain wrong for fibres with little or no flux!
    # Try replacing with something like sqrt(flux), but with a floor
    if (secondary):
        vartube = datatube.copy()
        cutoff = 0.05 * datatube.max()
        vartube[datatube < cutoff] = cutoff
    res = np.ravel((model - datatube) / np.sqrt(vartube))
    # Really crude way of putting bounds on the value of alpha
    if 'alpha_ref' in parameters_dict:
        if parameters_dict['alpha_ref'] < 0.5:
            res *= 1e10 * (0.5 - parameters_dict['alpha_ref'])
        elif parameters_dict['alpha_ref'] > 5.0:
            res *= 1e10 * (parameters_dict['alpha_ref'] - 5.0)
    if 'beta' in parameters_dict:
        if parameters_dict['beta'] <= 1.0:
            res *= 1e10*(1.01 - parameters_dict['beta'])
    return res

def fit_model_flux(datatube, vartube, xfibre, yfibre, wavelength, model_name,
                   fixed_parameters=None, secondary=False):
    """Fit a model to the given datatube."""
    par_0_dict = first_guess_parameters(datatube, vartube, xfibre, yfibre, 
                                        wavelength, model_name)
    par_0_vector = parameters_dict_to_vector(par_0_dict, model_name)
    args = (datatube, vartube, xfibre, yfibre, wavelength, model_name,
            fixed_parameters, secondary)
    parameters_vector = leastsq(residual, par_0_vector, args=args)[0]
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)
    return parameters_dict

def first_guess_parameters(datatube, vartube, xfibre, yfibre, wavelength, 
                           model_name):
    """Return a first guess to the parameters that will be fitted."""
    par_0 = {}
    #weighted_data = np.sum(datatube / vartube, axis=1)
    if (np.ndim(datatube)>1):
        weighted_data = np.nansum(datatube, axis=1)
        (nf, nc) = np.shape(datatube)
        print(nf,nc)
    else:
        weighted_data = np.copy(datatube)        
        (nf) = np.shape(datatube)
        nc = 1
        print(nf)
    weighted_data[weighted_data < 0] = 0.0
    weighted_data /= np.sum(weighted_data)
    if model_name == 'ref_centre_alpha_angle':
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
        par_0['flux'] = np.nansum(datatube, axis=0)
        par_0['background'] = np.zeros(len(par_0['flux']))
        par_0['xcen_ref'] = np.sum(xfibre * weighted_data)
        par_0['ycen_ref'] = np.sum(yfibre * weighted_data)
        par_0['zenith_direction'] = np.pi / 4.0
        par_0['zenith_distance'] = np.pi / 8.0
        par_0['alpha_ref'] = 1.0
        par_0['beta'] = 4.0
    elif (model_name == 'ref_centre_alpha_dist_circ' or
          model_name == 'ref_centre_alpha_dist_circ_hdratm'):
        par_0['flux'] = np.nansum(datatube, axis=0)
        if (nc > 1):
            par_0['background'] = np.zeros(len(par_0['flux']))
        else:
            par_0['background'] = np.zeros(1)
        par_0['xcen_ref'] = np.sum(xfibre * weighted_data)
        par_0['ycen_ref'] = np.sum(yfibre * weighted_data)
        par_0['zenith_distance'] = np.pi / 8.0
        par_0['alpha_ref'] = 1.0
        par_0['beta'] = 4.0
    elif model_name == 'ref_centre_alpha_angle_circ_atm':
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
    elif model_name == 'ref_centre_alpha_circ_hdratm':
        par_0['flux'] = np.nansum(datatube, axis=0)
        par_0['background'] = np.zeros(len(par_0['flux']))
        par_0['xcen_ref'] = np.sum(xfibre * weighted_data)
        par_0['ycen_ref'] = np.sum(yfibre * weighted_data)
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
    elif (model_name == 'ref_centre_alpha_dist_circ' or
          model_name == 'ref_centre_alpha_dist_circ_hdratm'):
        parameters_vector = np.hstack(
            (parameters_dict['flux'],
             parameters_dict['background'],
             parameters_dict['xcen_ref'],
             parameters_dict['ycen_ref'],
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
    elif model_name == 'ref_centre_alpha_circ_hdratm':
        parameters_vector = np.hstack(
            (parameters_dict['flux'],
             parameters_dict['background'],
             parameters_dict['xcen_ref'],
             parameters_dict['ycen_ref'],
             parameters_dict['alpha_ref'],
             parameters_dict['beta']))
    else:
        raise KeyError('Unrecognised model name: ' + model_name)
    return parameters_vector

def parameters_vector_to_dict(parameters_vector, model_name):
    """Convert a parameters vector to a dictionary."""
    parameters_dict = {}
    if model_name == 'ref_centre_alpha_angle':
        n_slice = np.int((len(parameters_vector) - 8) // 2)
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
        n_slice = np.int((len(parameters_vector) - 6) // 2)
        parameters_dict['flux'] = parameters_vector[0:n_slice]
        parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
        parameters_dict['xcen_ref'] = parameters_vector[-6]
        parameters_dict['ycen_ref'] = parameters_vector[-5]
        parameters_dict['zenith_direction'] = parameters_vector[-4]
        parameters_dict['zenith_distance'] = parameters_vector[-3]
        parameters_dict['alpha_ref'] = parameters_vector[-2]
        parameters_dict['beta'] = parameters_vector[-1]
    elif (model_name == 'ref_centre_alpha_dist_circ' or
          model_name == 'ref_centre_alpha_dist_circ_hdratm'):
        n_slice = np.int((len(parameters_vector) - 5) // 2)
        parameters_dict['flux'] = parameters_vector[0:n_slice]
        parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
        parameters_dict['xcen_ref'] = parameters_vector[-5]
        parameters_dict['ycen_ref'] = parameters_vector[-4]
        parameters_dict['zenith_distance'] = parameters_vector[-3]
        parameters_dict['alpha_ref'] = parameters_vector[-2]
        parameters_dict['beta'] = parameters_vector[-1]
    elif model_name == 'ref_centre_alpha_angle_circ_atm':
        n_slice = np.int((len(parameters_vector) - 9) // 2)
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
    elif model_name == 'ref_centre_alpha_circ_hdratm':
        n_slice = np.int((len(parameters_vector) - 4) // 2)
        parameters_dict['flux'] = parameters_vector[0:n_slice]
        parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
        parameters_dict['xcen_ref'] = parameters_vector[-4]
        parameters_dict['ycen_ref'] = parameters_vector[-3]
        parameters_dict['alpha_ref'] = parameters_vector[-2]
        parameters_dict['beta'] = parameters_vector[-1]
    else:
        raise KeyError('Unrecognised model name: ' + model_name)
    return parameters_dict

def parameters_dict_to_array(parameters_dict, wavelength, model_name):
    parameter_names = ('xcen ycen alphax alphay beta rho flux '
                       'background'.split())
    formats = ['float64'] * len(parameter_names)
    lw = np.size(wavelength)
    parameters_array = np.zeros(lw, 
                                dtype={'names':parameter_names, 
                                       'formats':formats})
    if model_name == 'ref_centre_alpha_angle':
        parameters_array['xcen'] = (
            parameters_dict['xcen_ref'] +
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] +
            np.cos(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['alphax'] = (
            alpha(wavelength, parameters_dict['alphax_ref']))
        parameters_array['alphay'] = (
            alpha(wavelength, parameters_dict['alphay_ref']))
        parameters_array['beta'] = parameters_dict['beta']
        parameters_array['rho'] = parameters_dict['rho']
        if len(parameters_dict['flux']) == len(parameters_array):
            parameters_array['flux'] = parameters_dict['flux']
        if len(parameters_dict['background']) == len(parameters_array):
            parameters_array['background'] = parameters_dict['background']
    elif (model_name == 'ref_centre_alpha_angle_circ' or
          model_name == 'ref_centre_alpha_dist_circ'):
        parameters_array['xcen'] = (
            parameters_dict['xcen_ref'] + 
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] + 
            np.cos(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['alphax'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['alphay'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['beta'] = parameters_dict['beta']
        parameters_array['rho'] = np.zeros(lw)
        if len(parameters_dict['flux']) == len(parameters_array):
            parameters_array['flux'] = parameters_dict['flux']
        if len(parameters_dict['background']) == len(parameters_array):
            parameters_array['background'] = parameters_dict['background']
    elif (model_name == 'ref_centre_alpha_angle_circ_atm' or
          model_name == 'ref_centre_alpha_dist_circ_hdratm' or
          model_name == 'ref_centre_alpha_circ_hdratm'):
        parameters_array['xcen'] = (
            parameters_dict['xcen_ref'] +
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance'],
                temperature=parameters_dict['temperature'],
                pressure=parameters_dict['pressure'],
                vapour_pressure=parameters_dict['vapour_pressure']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] +
            np.cos(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance'],
                temperature=parameters_dict['temperature'],
                pressure=parameters_dict['pressure'],
                vapour_pressure=parameters_dict['vapour_pressure']))
        parameters_array['alphax'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['alphay'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['beta'] = parameters_dict['beta']
        parameters_array['rho'] = np.zeros(lw)
        if len(parameters_dict['flux']) == len(parameters_array):
            parameters_array['flux'] = parameters_dict['flux']
        if len(parameters_dict['background']) == len(parameters_array):
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
    return (adr_r(wavelength, np.rad2deg(zenith_distance), 
                  air_pres=pressure, temperature=temperature, 
                  water_pres=vapour_pressure) - 
            adr_r(REFERENCE_WAVELENGTH, np.rad2deg(zenith_distance), 
                  air_pres=pressure, temperature=temperature, 
                  water_pres=vapour_pressure))

# def dar(wavelength, zenith_distance, temperature=None, pressure=None, 
#         vapour_pressure=None):
#     """Return the DAR offset in arcseconds at the specified wavelength(s)."""
#     # Analytic expectations from Fillipenko (1982)
#     n_observed = refractive_index(
#         wavelength, temperature, pressure, vapour_pressure)
#     n_reference = refractive_index(
#         REFERENCE_WAVELENGTH, temperature, pressure, vapour_pressure)
#     return 206265. * (n_observed - n_reference) * np.tan(zenith_distance)

# def refractive_index(wavelength, temperature=None, pressure=None, 
#                      vapour_pressure=None):
#     """Return the refractive index at the specified wavelength(s)."""
#     # Analytic expectations from Fillipenko (1982)
#     if temperature is None:
#         temperature = 7.
#     if pressure is None:
#         pressure = 600.
#     if vapour_pressure is None:
#         vapour_pressure = 8.
#     # Convert wavelength from Angstroms to microns
#     wl = wavelength * 1e-4
#     seaLevelDry = ( 64.328 + ( 29498.1 / ( 146. - ( 1 / wl**2. ) ) )
#                     + 255.4 / ( 41. - ( 1. / wl**2. ) ) )
#     altitudeCorrection = ( 
#         ( pressure * ( 1. + (1.049 - 0.0157*temperature ) * 1e-6 * pressure ) )
#         / ( 720.883 * ( 1. + 0.003661 * temperature ) ) )
#     vapourCorrection = ( ( 0.0624 - 0.000680 / wl**2. )
#                          / ( 1. + 0.003661 * temperature ) ) * vapour_pressure
#     return 1e-6 * (seaLevelDry * altitudeCorrection - vapourCorrection) + 1

def derive_transfer_function(path_list, max_sep_arcsec=60.0,
                             catalogues=STANDARD_CATALOGUES,
                             model_name='ref_centre_alpha_dist_circ_hdratm',
                             n_trim=0, smooth='spline',molecfit_available=False,
                             molecfit_dir='',speed='',tell_corr_primary=False):
    """Derive transfer function and save it in each FITS file."""
    # First work out which star we're looking at, and which hexabundle it's in
    star_match = match_standard_star(
        path_list[0], max_sep_arcsec=max_sep_arcsec, catalogues=catalogues)
    if star_match is None:
        raise ValueError('No standard star found in the data.')
    standard_data = read_standard_data(star_match)
    
    # Apply telluric correction to primary standards and write to new file, returning
    # the paths to those files.
    if molecfit_available & (speed == 'slow') & tell_corr_primary:
        path_list_tel = telluric_correct_primary(path_list,star_match['probenum'],
                                            molecfit_dir=molecfit_dir)
    else:
        path_list_tel = path_list

    # Read the observed data, in chunks
    chunked_data = read_chunked_data(path_list_tel, star_match['probenum'])
    trim_chunked_data(chunked_data, n_trim)
    # Fit the PSF
    fixed_parameters = set_fixed_parameters(
        path_list_tel, model_name, probenum=star_match['probenum'])
    psf_parameters = fit_model_flux(
        chunked_data['data'], 
        chunked_data['variance'],
        chunked_data['xfibre'],
        chunked_data['yfibre'],
        chunked_data['wavelength'],
        model_name,
        fixed_parameters=fixed_parameters)
    psf_parameters = insert_fixed_parameters(psf_parameters, fixed_parameters)
    good_psf = check_psf_parameters(psf_parameters, chunked_data)
    for path,path2 in zip(path_list_tel,path_list):
        ifu = IFU(path, star_match['probenum'], flag_name=False)
        remove_atmosphere(ifu)
        observed_flux, observed_background, sigma_flux, sigma_background = \
            extract_total_flux(ifu, psf_parameters, model_name)
        save_extracted_flux(path2, observed_flux, observed_background,
                            sigma_flux, sigma_background,
                            star_match, psf_parameters, model_name,
                            good_psf, HG_CHANGESET)
        transfer_function = take_ratio(
            standard_data['flux'],
            standard_data['wavelength'],
            observed_flux,
            sigma_flux,
            ifu.lambda_range,
            smooth=smooth,
            mf_av=molecfit_available,
            tell_corr_primary=tell_corr_primary)
        save_transfer_function(path2, transfer_function)
    return

def match_standard_star(filename, max_sep_arcsec=60.0, 
                        catalogues=STANDARD_CATALOGUES):
    """Return details of the standard star that was observed in this file."""
    fibre_table = pf.getdata(filename, 'FIBRES_IFU')
    primary_header = pf.getheader(filename)
    probenum_list = np.unique([fibre['PROBENUM'] for fibre in fibre_table
                               if 'SKY' not in fibre['PROBENAME']])
    for probenum in probenum_list:
        this_probe = ((fibre_table['PROBENUM'] == probenum) &
                      (fibre_table['TYPE'] == 'P'))
        ra = np.mean(fibre_table['FIB_MRA'][this_probe])
        dec = np.mean(fibre_table['FIB_MDEC'][this_probe])
        if primary_header['AXIS'] != 'REF':
            # The observation has been taken in an axis other than REF,
            # therefore the RA and DEC must be adjusted to get correct
            # coordinates.
            dec += primary_header['AXIS_X'] * 180/np.pi
            ra += primary_header['AXIS_Y'] * 180/np.pi
        star_match = match_star_coordinates(
            ra, dec, max_sep_arcsec=max_sep_arcsec, catalogues=catalogues)
        if star_match is not None:
            # Let's assume there will only ever be one match
            star_match['probenum'] = probenum
            return star_match
    # Uh-oh, should have found a star by now. Return None and let the outer
    # code deal with it.
    return

def match_star_coordinates(ra, dec, max_sep_arcsec=60.0, 
                           catalogues=STANDARD_CATALOGUES):
    """Return details of the star nearest to the supplied coordinates."""
    for index_path in catalogues:
        #index = np.loadtxt(index_path, dtype='S')
        converters = {'col6':[ascii.convert_numpy(np.str)]}
        index = table.Table.read(index_path, format='ascii.no_header'
                                 ,converters=converters)
        for star in index:
            RAstring = '%sh%sm%ss' % (star['col3'], star['col4'], star['col5'])
            Decstring = '%sd%sm%ss' % (star['col6'], star['col7'], star['col8'])

            ra_star = coord.Angle(RAstring, unit=units.hour) 
            dec_star = coord.Angle(Decstring, unit=units.degree)
            coords_star = coord.ICRS(ra_star, dec_star)

            ra_tgt = coord.Angle(ra, unit=units.degree)
            dec_tgt = coord.Angle(dec, unit=units.degree)

            coords = coord.ICRS(ra_tgt, dec_tgt)

            sep = coords.separation(coords_star).arcsec

            if sep < max_sep_arcsec:
                star_match = {
                    'path': os.path.join(os.path.dirname(index_path), star[0]),
                    'name': star[1],
                    'separation': sep
                    }
                return star_match
    # No matching star found. Let outer code deal with it.
    return

def extract_total_flux(ifu, psf_parameters, model_name, clip=None):
    """Extract the total flux, including light between fibres."""
    psf_parameters_array = parameters_dict_to_array(
        psf_parameters, ifu.lambda_range, model_name)
    n_pixel = len(psf_parameters_array)
    flux = np.zeros(n_pixel)
    background = np.zeros(n_pixel)
    sigma_flux = np.zeros(n_pixel)
    sigma_background = np.zeros(n_pixel)
    good_fibre = (ifu.fib_type == 'P')
    xfibre = ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos)))
    yfibre = ifu.ypos_rel
    for index, psf_parameters_slice in enumerate(psf_parameters_array):
        flux[index], background[index], sigma_flux[index], \
            sigma_background[index] = extract_flux_slice(
                ifu.data[good_fibre, index], ifu.var[good_fibre, index], 
                xfibre[good_fibre], yfibre[good_fibre],
                psf_parameters_slice)
    if clip is not None:
        # Clip out discrepant data. Wavelength slices are targeted based on
        # their overall deviation from the model, but within a slice only
        # positive features (like cosmic rays) are clipped.
        model_parameters = psf_parameters.copy()
        model_parameters['flux'] = flux
        model_parameters['background'] = background
        fit = model_flux(
            model_parameters, xfibre, yfibre, ifu.lambda_range, 
            model_name)
        rms = np.sqrt(np.nansum((ifu.data - fit)**2, axis=0))
        rms_smoothed = median_filter(rms, 41)
        bad = np.where(rms >= (clip * rms_smoothed))[0]
        for bad_pixel in bad:
            keep_bool = np.isfinite(ifu.data[:, bad_pixel])
            keep = np.where(keep_bool)[0]
            while rms[bad_pixel] >= (clip * rms_smoothed[bad_pixel]):
                if len(keep) < 0.5*len(xfibre):
                    # Over half the fibres have been clipped; time to give up
                    flux[bad_pixel] = np.nan
                    background[bad_pixel] = np.nan
                    sigma_flux[bad_pixel] = np.nan
                    sigma_background[bad_pixel] = np.nan
                    break
                # Clip out the fibre with the largest positive deviation
                worst_pixel = np.argmax(ifu.data[keep, bad_pixel] - 
                                        fit[keep, bad_pixel])
                keep_bool[keep[worst_pixel]] = False
                keep = np.where(keep_bool)[0]
                # Re-fit the model to the data
                flux[bad_pixel], background[bad_pixel], \
                    sigma_flux[bad_pixel], sigma_background[bad_pixel] = \
                    extract_flux_slice(
                        ifu.data[keep, bad_pixel], ifu.var[keep, bad_pixel],
                        xfibre[keep], yfibre[keep], 
                        psf_parameters_array[bad_pixel])
                # Re-calculate the deviation from the model
                model_parameters['flux'] = np.array(
                    [flux[bad_pixel]])
                model_parameters['background'] = np.array(
                    [background[bad_pixel]])
                fit_pixel = model_flux(
                    model_parameters, xfibre, yfibre, 
                    np.array([ifu.lambda_range[bad_pixel]]), model_name)[:,0]
                rms_pixel = np.sqrt(np.nansum((ifu.data[keep, bad_pixel] - 
                                               fit_pixel[keep])**2))
                rms[bad_pixel] = rms_pixel
    # Interpolate over any NaNs in the middle of the spectrum, but not 
    # at the ends
    interp_over = np.where(~np.isfinite(flux))[0]
    # Take out leading NaNs
    start = 0
    while len(interp_over) > 0 and interp_over[0] == start:
        start += 1
        interp_over = interp_over[1:]
    finish = n_pixel - 1
    while len(interp_over) > 0 and interp_over[-1] == finish:
        finish -= 1
        interp_over = interp_over[:-1]
    if len(interp_over) > 0:
        interp_from = np.where(np.isfinite(flux))[0]
        flux[interp_over] = np.interp(
            interp_over, interp_from, flux[interp_from])
        sigma_flux[interp_over] = np.interp(
            interp_over, interp_from , sigma_flux[interp_from])
        background[interp_over] = np.interp(
            interp_over, interp_from, background[interp_from])
        sigma_background[interp_over] = np.interp(
            interp_over, interp_from, sigma_background[interp_from])
    return flux, background, sigma_flux, sigma_background

def extract_flux_slice(data, variance, xpos, ypos, psf_parameters_slice):
    """Extract the flux from a single wavelength slice."""
    good_data = np.where(np.isfinite(data))[0]
    # Require at least half the fibres to perform a fit
    if len(good_data) > 30:
        data = data[good_data]
        variance = variance[good_data]
        noise = np.sqrt(variance)
        xpos = xpos[good_data]
        ypos = ypos[good_data]
        model = moffat_normalised(psf_parameters_slice, xpos, ypos)
        # args = (model, data, variance)
        # Initial guess for flux and background
        guess = [np.sum(data), 0.0]
        fitfunc = lambda x, flux, background: flux * model + background
        result, covar = curve_fit(
            fitfunc, np.arange(len(data)), data, guess, noise)
        flux_slice, background_slice = result
        reduced_chi2 = (
            ((fitfunc(None, flux_slice, background_slice) - data) / noise)**2 /
            (len(data) - 2)).sum()
        try:
            sigma_flux_slice, sigma_background_slice = np.sqrt(
                covar[np.arange(2), np.arange(2)] / reduced_chi2)
        except TypeError:
            # covar wasn't returned as an array
            flux_slice = np.nan
            background_slice = np.nan
            sigma_flux_slice = np.nan
            sigma_background_slice = np.nan
        # flux_slice, background_slice = leastsq(
        #     residual_slice, guess, args=args)[0]
    else:
        flux_slice = np.nan
        background_slice = np.nan
        sigma_flux_slice = np.nan
        sigma_background_slice = np.nan
    return flux_slice, background_slice, sigma_flux_slice, sigma_background_slice

def residual_slice(flux_background, model, data, variance):
    """Residual of the model flux in a single wavelength slice."""
    flux, background = flux_background
    # # For now, ignoring the variance - it has too many 2dfdr-induced mistakes
    # variance = data.copy()
    # cutoff = 0.05 * data.max()
    # variance[data < cutoff] = cutoff
    return ((background + flux * model) - data) / np.sqrt(variance)

def save_transfer_function(path, transfer_function):
    """Add the transfer function to a pre-existing FLUX_CALIBRATION HDU."""
    # Open the file to update
    hdulist = pf.open(path, 'update', do_not_scale_image_data=True)
    hdu = hdulist['FLUX_CALIBRATION']
    data = hdu.data
    if len(data) == 4:
        # No previous transfer function saved; append it to the data
        data = np.vstack((data, transfer_function))
    elif len(data) == 5:
        # Previous transfer function to overwrite
        data[-1, :] = transfer_function
    # Save the data back into the FITS file
    hdu.data = data
    hdulist.close()
    return

def read_standard_data(star):
    """Return the true wavelength and flux for a primary standard."""
    # First check how many header rows there are
    skiprows = 0
    with open(star['path']) as f_spec:
        finished = False
        while not finished:
            line = f_spec.readline()
            try:
                number = float(line.split()[0])
            except ValueError:
                skiprows += 1
                continue
            else:
                finished = True
    # Now actually read the data
    star_data = np.loadtxt(star['path'], dtype='d', skiprows=skiprows)
    wavelength = star_data[:, 0]
    flux = star_data[:, 1]
    source = os.path.basename(os.path.dirname(star['path']))
    if source == 'Bessell':
        flux /= 1e-16
    standard_data = {'wavelength': wavelength,
                     'flux': flux}
    return standard_data

def take_ratio(standard_flux, standard_wavelength, observed_flux, 
               sigma_flux, observed_wavelength, smooth='spline',
               mf_av=False, tell_corr_primary=False):
    """Return the ratio of two spectra, after rebinning."""
    # Rebin the observed spectrum onto the (coarser) scale of the standard
    observed_flux_rebinned, sigma_flux_rebinned, count_rebinned = \
        rebin_flux_noise(standard_wavelength, observed_wavelength,
                         observed_flux, sigma_flux)
    ratio = standard_flux / observed_flux_rebinned
    if smooth == 'gauss':
        ratio = smooth_ratio(ratio)
    elif smooth == 'chebyshev':
        ratio = fit_chebyshev(standard_wavelength, ratio, mf_av=mf_av,
                              tell_corr_primary=tell_corr_primary)
    elif smooth == 'spline':
        ratio = fit_spline(standard_wavelength, ratio, mf_av=mf_av,
                           tell_corr_primary=tell_corr_primary)
    # Put the ratio back onto the observed wavelength scale
    ratio = 1.0 / np.interp(observed_wavelength, standard_wavelength, 
                            1.0 / ratio)
    return ratio

def smooth_ratio(ratio, width=10.0):
    """Smooth a ratio (or anything else, really). Uses Gaussian kernel."""
    # Get best behaviour at edges if done in terms of transmission, rather
    # than transfer function
    inverse = 1.0 / ratio
    # Trim NaNs and infs from the ends (not any in the middle)
    useful = np.where(np.isfinite(inverse))[0]
    inverse_cut = inverse[useful[0]:useful[-1]+1]
    good = np.isfinite(inverse_cut)
    inverse_cut[~good] = np.interp(np.where(~good)[0], np.where(good)[0], 
                                   inverse_cut[good])
    # Extend the inverse ratio with a mirrored version. Not sure why this
    # can't be done in gaussian_filter1d.
    extra = int(np.round(3.0 * width))
    inverse_extended = np.hstack(
        (np.zeros(extra), inverse_cut, np.zeros(extra)))
    inverse_extended[:extra] = 2*inverse_cut[0] - inverse_cut[extra+1:1:-1]
    inverse_extended[-1*extra:] = (
        2*inverse_cut[-1] - inverse_cut[-1:-1*(extra+1):-1])
    # Do the actual smoothing
    inverse_smoothed = gaussian_filter1d(inverse_extended, width, 
                                         mode='nearest')
    # Cut off the extras
    inverse_smoothed = inverse_smoothed[extra:-1*extra]
    # Insert back into the previous array
    inverse[useful[0]:useful[-1]+1] = inverse_smoothed
    # Undo the inversion
    smoothed = 1.0 / inverse
    return smoothed

def fit_chebyshev(wavelength, ratio, deg=None, mf_av=False,tell_corr_primary=False):
    """Fit a Chebyshev polynomial, and return the fit."""
    # Do the fit in terms of 1.0 / ratio, because observed flux can go to 0.
    if mf_av & tell_corr_primary:
        good = np.where(np.isfinite(ratio))[0]
    else:
        good = np.where(np.isfinite(ratio) & ~(in_telluric_band(wavelength)))[0]
    if deg is None:
        # Choose default degree based on which arm this data is for.
        if wavelength[good[0]] >= 6000.0:
            deg = 3
        else:
            deg = 7
    coefficients = np.polynomial.chebyshev.chebfit(
        wavelength[good], (1.0 / ratio[good]), deg)
    fit = 1.0 / np.polynomial.chebyshev.chebval(wavelength, coefficients)
    # Mark with NaNs anything outside the fitted wavelength range
    fit[:good[0]] = np.nan
    fit[good[-1]+1:] = np.nan
    return fit

def fit_spline(wavelength, ratio, mf_av=False,tell_corr_primary=False):
    """Fit a smoothing spline to the data, and return the fit."""
    # Do the fit in terms of 1.0 / ratio, because it seems to give better
    # results.
    if mf_av & tell_corr_primary:
        good = np.where(np.isfinite(ratio))[0]
    else:
        good = np.where(np.isfinite(ratio) & ~(in_telluric_band(wavelength)))[0]
    knots = np.linspace(wavelength[good][0], wavelength[good][-1], 8)
    # Add extra knots around 5500A, where there's a sharp turn
    extra = knots[(knots > 5000) & (knots < 6000)]
    if len(extra) > 1:
        extra = 0.5 * (extra[1:] + extra[:-1])
        knots = np.sort(np.hstack((knots, extra)))
    # Remove any knots sitting in a telluric band, but only if not using tell_corr_primary:
    if (not tell_corr_primary):
        knots = knots[~in_telluric_band(knots)]
        
    spline = LSQUnivariateSpline(
        wavelength[good], 1.0/ratio[good], knots[1:-1])
    fit = 1.0 / spline(wavelength)
    # Mark with NaNs anything outside the fitted wavelength range
    #fit[:good[0]] = np.nan
    #fit[good[-1]+1:] = np.nan
    # don't flag as Nan.  indead allow the spline function to extrapolate
    # beyond the end points.  This means the that TF does not suddenly
    # stop at the end of the particular spectrum , which is good because the
    # wavelength coverage can vary between bundles and fibres.  The extrapolation is
    # low order and well behaved.

    return fit

def fit_spline_secondary(wavelength, ratio,sigma,nbins=40,nsig=5.0,verbose=False,medsize=51):
    """Fit a smoothing spline to the data, and return the fit."""
    # Do the fit in terms of 1.0 / ratio, because it seems to give better
    # results.

    nspec = np.size(ratio)

    # identify good pixels:
    good = np.where(np.isfinite(ratio))[0]
    ngood1 = np.size(ratio[good])
    
    # median filter the data and sigma (excluding NaNs):
    medspec = median_filter_nan_1d(ratio,medsize)
    medsig = median_filter_nan_1d(sigma,medsize)

    # identify pixels that are greater than nsig away from the median:
    good = np.where(np.isfinite(ratio) & (abs(medspec-ratio)<nsig*medsig))[0]
    ngood2 = np.size(ratio[good])

    if (verbose):
        print('number of pixels rejected:',ngood1-ngood2)
        idx = np.where((abs(medspec-ratio)<nsig*medsig) & np.isfinite(ratio))
        print(wavelength[idx])
    
    # specify knots.  As the data has already been flux calibrated using
    # the primary stds, the sharp features have been removed and the remaining
    # features are slowly varying.  Because of this, have a relatively small
    # number of knots: 
    knots = np.linspace(wavelength[good][0], wavelength[good][-1], 3)

    # do the fit
    spline = LSQUnivariateSpline(
        wavelength[good], 1.0/ratio[good], knots[1:-1])
    fit = 1.0 / spline(wavelength)

    if (verbose):
        print('knots: ',spline.get_knots())

    # Next we want to do a robust outlier rejection so that any bad pixels
    # (eg remaining CRs etc) do not cause problems.

    return fit

def rebin_flux_noise(target_wavelength, source_wavelength, source_flux,
                     source_noise, clip=True):
    """Rebin flux and noise onto a new wavelength grid."""
    # Interpolate over bad pixels
    good = np.isfinite(source_flux) & np.isfinite(source_noise)
    if clip:
        # Clip points that are a long way from a narrow median filter
        with warnings.catch_warnings():
            # We get lots of invalid value warnings arising because of divide by zero errors.
            warnings.filterwarnings('ignore', r'invalid value', RuntimeWarning)
            good = good & ((np.abs(source_flux - median_filter(source_flux, 7)) /
                            source_noise) < 10.0)
            # Also clip points where the noise value spikes (changes by more than
            # 35% relative to the baseline)
            filtered_noise = median_filter(source_noise, 21)
            good = good & ((np.abs(source_noise - filtered_noise) /
                            filtered_noise) < 0.35)
    interp_flux, interp_noise = interpolate_flux_noise(
        source_flux, source_noise, good)
    interp_good = np.isfinite(interp_flux)
    interp_flux = interp_flux[interp_good]
    interp_noise = interp_noise[interp_good]
    interp_wavelength = source_wavelength[interp_good]
    interp_variance = interp_noise ** 2
    # # Assume pixel size is fixed
    # target_delta_wave = target_wavelength[1] - target_wavelength[0]
    # interp_delta_wave = interp_wavelength[1] - interp_wavelength[0]
    # # Correct to start/end of pixels instead of centre points
    # target_wave_limits = target_wavelength - 0.5*target_delta_wave
    # interp_wave_limits = interp_wavelength - 0.5*interp_delta_wave
    # # The output pixel that each input pixel starts/ends in
    # start_pix = np.floor(
    #     (interp_wave_limits - target_wave_limits[0]) / 
    #     target_delta_wave).astype(int)
    # end_pix = np.floor(
    #     (interp_wave_limits + interp_delta_wave - target_wave_limits[0]) / 
    #     target_delta_wave).astype(int)
    #
    # Adjust each wavelength value to the start of its pixel, not the middle
    target_wave_limits = np.hstack(
        (target_wavelength[0] - 
         0.5 * (target_wavelength[1] - target_wavelength[0]),
         0.5 * (target_wavelength[:-1] + target_wavelength[1:])))
    interp_wave_limits = np.hstack(
        (interp_wavelength[0] - 
         0.5 * (interp_wavelength[1] - interp_wavelength[0]),
         0.5 * (interp_wavelength[:-1] + interp_wavelength[1:])))
    # Get the width of each wavelength pixel
    # target_delta_wave = np.hstack(
    #     (target_wave_limits[1:] - target_wave_limits[:-1],
    #      target_wave_limits[-1] - target_wave_limits[-2]))
    interp_delta_wave = np.hstack(
        (interp_wave_limits[1:] - interp_wave_limits[:-1],
         interp_wave_limits[-1] - interp_wave_limits[-2]))
    # The output pixel that each input pixel starts/ends in
    # Assumes standard star has complete wavelength coverage
    n_pix_out = np.size(target_wavelength)
    n_pix_in = np.size(interp_wavelength)
    start_pix = np.sum(
        np.outer(interp_wave_limits, np.ones(n_pix_out)) >
        np.outer(np.ones(n_pix_in), target_wave_limits),
        1) - 1
    end_pix = np.sum(
        np.outer(interp_wave_limits + interp_delta_wave, np.ones(n_pix_out)) >
        np.outer(np.ones(n_pix_in), target_wave_limits),
        1) - 1
    bins = np.arange(n_pix_out + 1)    
    # `complete` is True if the input pixel is entirely within one output pixel
    complete = (start_pix == end_pix)
    incomplete = ~complete
    # The fraction of the input pixel that falls in the start_pix output pixel
    # Only correct for incomplete pixels
    frac_low = (
        (target_wave_limits[end_pix] - interp_wave_limits) /
        interp_delta_wave)
    # Make output arrays
    flux_out = np.zeros(n_pix_out)
    variance_out = np.zeros(n_pix_out)
    count_out = np.zeros(n_pix_out)
    # Add the flux from the `complete` pixels
    flux_out += np.histogram(
        start_pix[complete], weights=interp_flux[complete], 
        bins=bins)[0]
    variance_out += np.histogram(
        start_pix[complete], weights=interp_variance[complete], 
        bins=bins)[0]
    count_out += np.histogram(
        start_pix[complete],
        bins=bins)[0]
    # Add the flux from the `incomplete` pixels
    flux_out += np.histogram(
        start_pix[incomplete], 
        weights=interp_flux[incomplete] * frac_low[incomplete],
        bins=bins)[0]
    variance_out += np.histogram(
        start_pix[incomplete], 
        weights=interp_variance[incomplete] * frac_low[incomplete]**2,
        bins=bins)[0]
    count_out += np.histogram(
        start_pix[incomplete],
        weights=frac_low[incomplete],
        bins=bins)[0]
    flux_out += np.histogram(
        end_pix[incomplete],
        weights=interp_flux[incomplete] * (1 - frac_low[incomplete]),
        bins=bins)[0]
    variance_out += np.histogram(
        end_pix[incomplete],
        weights=interp_variance[incomplete] * (1 - frac_low[incomplete])**2,
        bins=bins)[0]
    count_out += np.histogram(
        end_pix[incomplete],
        weights=(1 - frac_low[incomplete]),
        bins=bins)[0]
    with warnings.catch_warnings():
        # We get lots of invalid value warnings arising because of divide by zero errors.
        warnings.filterwarnings('ignore', r'invalid value', RuntimeWarning)
        flux_out /= count_out
        variance_out /= count_out ** 2
        zero_input = (count_out == 0)
        flux_out[zero_input] = np.nan
        variance_out[zero_input] = np.nan
        noise_out = np.sqrt(variance_out)
    return flux_out, noise_out, count_out

def interpolate_flux_noise(flux, noise, good):
    """Interpolate over bad pixels, fixing the noise for later rebinning."""
    bad = ~good
    interp_flux = flux.copy()
    interp_noise = noise.copy()
    interp_flux[bad] = np.interp(
        np.where(bad)[0], np.where(good)[0], flux[good])
    start_bad = np.where(good[:-1] & bad[1:])[0] + 1
    end_bad = np.where(bad[:-1] & good[1:])[0] + 1
    with warnings.catch_warnings():
        # We get lots of invalid value warnings arising because of divide by zero errors.
        warnings.filterwarnings('ignore', r'invalid value', RuntimeWarning)
        for begin, finish in zip(start_bad, end_bad):
            n_bad = finish - begin
            interp_noise[begin:finish] = np.sqrt(
                ((((1 + 0.5*n_bad)**2) - 1) / n_bad) *
                (noise[begin-1]**2 + noise[finish]**2))
    # Set any bad pixels at the start and end of the spectrum back to nan
    still_bad = ~np.isfinite(interp_noise)
    interp_flux[still_bad] = np.nan
    return interp_flux, interp_noise

def rebin_flux(target_wavelength, source_wavelength, source_flux):
    """Rebin a flux onto a new wavelength grid."""
    targetwl = target_wavelength
    originalwl = source_wavelength
    originaldata = source_flux[1:-1]
    # The following is copy-pasted from the original fluxcal.py
    originalbinlimits = ( originalwl[ :-1 ] + originalwl[ 1: ] ) / 2.
    okaytouse = np.isfinite( originaldata )

    originalweight = np.where(okaytouse, 1., 0.)
    originaldata = np.where(okaytouse, originaldata, 0.)

    originalflux = originaldata * np.diff( originalbinlimits )
    originalweight *= np.diff( originalbinlimits )

    nowlsteps = len( targetwl )
    rebinneddata   = np.zeros( nowlsteps )
    rebinnedweight = np.zeros( nowlsteps )

    binlimits = np.array( [ np.nan ] * (nowlsteps+1) )
    binlimits[ 0 ] = targetwl[ 0 ]
    binlimits[ 1:-1 ] = ( targetwl[ 1: ] + targetwl[ :-1 ] ) / 2.
    binlimits[ -1 ] = targetwl[ -1 ]
    binwidths = np.diff( binlimits )

    origbinindex = np.interp( binlimits, originalbinlimits, 
                              np.arange( originalbinlimits.shape[0] ),
                              left=np.nan, right=np.nan )

    fraccounted = np.zeros( originaldata.shape[0] )
    # use fraccounted to check what fraction of each orig pixel is counted,
    # and in this way check that flux is conserved.

    maximumindex = np.max( np.where( np.isfinite( origbinindex ) ) )

    for i, origindex in enumerate( origbinindex ):
        if np.isfinite( origindex ) :
            # deal with the lowest orig bin, which straddles the new lower limit
            lowlimit = int( origindex )
            lowfrac = 1. - ( origindex % 1 )
            indices = np.array( [ lowlimit] )
            weights = np.array( [ lowfrac ] )

            # deal with the orig bins that fall entirely within the new bin
            if np.isfinite( origbinindex[i+1] ):
                intermediate = np.arange( int( origindex )+1, \
                                      int(origbinindex[i+1]) )
            else :
                # XXX This is wrong: maximumindex is in the wrong scale
                #intermediate = np.arange( int( origindex )+1, \
                #                            maximumindex )
                # This may also be wrong, but at least it doesn't crash
                intermediate = np.arange(0)
            indices = np.hstack( ( indices, intermediate ) )
            weights = np.hstack( ( weights, np.ones( intermediate.shape ) ) )

            # deal with the highest orig bin, which straddles the new upper limit
            if np.isfinite( origbinindex[i+1] ):
                upplimit = int( origbinindex[i+1] )
                uppfrac = origbinindex[ i+1 ] % 1
                indices = np.hstack( ( indices, np.array( [ upplimit ] ) ) )
                weights = np.hstack( ( weights, np.array( [ uppfrac  ] ) ) )

            fraccounted[ indices ] += weights
            rebinneddata[ i ] = np.sum( weights * originalflux[indices] )
            rebinnedweight[i ]= np.sum( weights * originalweight[indices] )

    # now go back from total flux in each bin to flux per unit wavelength
    rebinneddata = rebinneddata / rebinnedweight 

    return rebinneddata


def calc_eff_airmass(header,return_zd=False):
    """Calculate the effective airmass using observatory location, coordinates 
    and time.  This makes use of various astropy functions.  The input is 
    a primary FITS header for a standard frame.  If return_zd = True, then
    return the effective ZD rather than airmass.
    """
    # this should really go into fluxcal, but there seems to be problems with
    # imports as this is also called from the ifu class that is within utils.
    # not sure why, but putting this in utils.other is a solution that seems to work.
    
    # get all the relevant header keywords:
    meanra = header['MEANRA']
    meandec = header['MEANDEC']
    utdate = header['UTDATE']
    utstart = header['UTSTART']
    utend = header['UTEND']
    lat_obs = header['LAT_OBS']
    long_obs = header['LONG_OBS']
    alt_obs = header['ALT_OBS']
    zdstart = header['ZDSTART']

    # define observatory location:
    obs_loc = EarthLocation(lat=lat_obs*u.deg, lon=long_obs*u.deg, height=alt_obs*u.m)

    # Convert to the correct time format:
    date_formatted = utdate.replace(':','-')
    time_start = date_formatted+' '+utstart
    # note that here we assume UT date start is the same as UT date end.  This works for
    # the AAT, given the time difference from UT at night, but will not for other observatories.
    time_end = date_formatted+' '+utend
    time1 = Time(time_start) 
    time2 = Time(time_end) 
    time_diff = time2-time1
    time_mid = time1 + time_diff/2.0

    # define coordinates using astropy coordinates object:
    coords = SkyCoord(meanra*u.deg,meandec*u.deg) 

    # calculate alt/az using astropy coordinate transformations:
    altazpos1 = coords.transform_to(AltAz(obstime=time1,location=obs_loc))   
    altazpos2 = coords.transform_to(AltAz(obstime=time2,location=obs_loc))
    altazpos_mid = coords.transform_to(AltAz(obstime=time_mid,location=obs_loc))   

    # get altitude and remove units put in by astropy.  We need the float(), as
    # even when divided by the units, we get back a dimensionless object, not an actual
    # float.
    alt1 = float(altazpos1.alt/u.deg)
    alt2 = float(altazpos2.alt/u.deg)
    alt_mid = float(altazpos_mid.alt/u.deg)
    
    # convert to ZD:
    zd1 = 90.0-alt1
    zd2 = 90.0-alt2
    zd_mid = 90.0-alt_mid

    # calc airmass at the start, end and midpoint:
    airmass1 = 1./ ( np.sin( ( alt1 + 244. / ( 165. + 47 * alt1**1.1 )
                    ) / 180. * np.pi ) )
    airmass2 = 1./ ( np.sin( ( alt2 + 244. / ( 165. + 47 * alt2**1.1 )
                    ) / 180. * np.pi ) )
    airmass_mid = 1./ ( np.sin( ( alt_mid + 244. / ( 165. + 47 * alt_mid**1.1 )
                    ) / 180. * np.pi ) )

    # get effective airmass by simpsons rule integration:
    airmass_eff = ( airmass1 + 4. * airmass_mid + airmass2 ) / 6.

    # if needed get effective ZD:
    if (return_zd):
        zd_eff = ( zd1 + 4. * zd_mid + zd2 ) / 6.
        
    #print('effective airmass:',airmass_eff)
    #print('ZD start:',zdstart)
    #print('ZD start (calculated):',zd1)
        
    # check that the ZD calculated actually agrees with the ZDSTART in the header
    d_zd = abs(zd1-zdstart)
    if (d_zd>0.1):
        print('WARNING: calculated ZD different from ZDSTART.  Difference:',d_zd)
        # if we have this problem, assume that the ZDSTART header keyword is correct
        # and that one or more of the other keywords has a problem.  Then set
        # the effective airmass to be based on ZDSTART:
        alt1 = 90.0-zdstart
        airmass_eff = 1./ ( np.sin( ( alt1 + 244. / ( 165. + 47 * alt1**1.1 )
                                ) / 180. * np.pi ) )
    if (return_zd):
        return zd_eff
    else:
        return airmass_eff


def remove_atmosphere(ifu):
    """Remove atmospheric extinction (not tellurics) from ifu data."""
    # Read extinction curve (in magnitudes)
    wavelength_extinction, extinction_mags = read_atmospheric_extinction()
    # Interpolate onto the data wavelength grid
    extinction_mags = np.interp(ifu.lambda_range, wavelength_extinction, 
                                extinction_mags)
    # Scale for observed airmass
    #airmass = calculate_airmass(ifu.zdstart, ifu.zdend)
    # no longer calculate airmass here,  instead take the value from the ifu
    # class that uses calc_eff_airmass()
    airmass = calc_eff_airmass(ifu.primary_header)
    extinction_mags *= airmass
    # Turn into multiplicative flux scaling
    extinction = 10.0 ** (-0.4 * extinction_mags)
    ifu.data /= extinction
    ifu.var /= (extinction**2)
    return

def remove_atmosphere_rss(hdulist):
    """Remove atmospheric extinction (not tellurics) from an RSS HDUList."""
    # Read extinction curve (in magnitudes)
    wavelength_extinction, extinction_mags = read_atmospheric_extinction()
    # Interpolate onto the data wavelength grid
    header = hdulist[0].header
    wavelength = (header['CRVAL1'] + 
        (np.arange(header['NAXIS1']) + 1 - header['CRPIX1']) * 
         header['CDELT1'])
    extinction_mags = np.interp(wavelength, wavelength_extinction, 
                                extinction_mags)
    # Scale for observed airmass
    #airmass = calculate_airmass(header['ZDSTART'], header['ZDEND'])
    # calculate effective airmass using header keywords, not just the
    # ZDSTART.  Note that the old code calculate_airmass() just sets
    # ZDSTART=ZDEND, which is not very good in some cases.
    airmass = calc_eff_airmass(header)
    extinction_mags *= airmass
    # Turn into multiplicative flux scaling
    extinction = 10.0 ** (-0.4 * extinction_mags)
    hdulist[0].data /= extinction
    hdulist['VARIANCE'].data /= extinction**2
    return
    

def calculate_airmass( zdstart, zdend ):
    # For now, ignoring zdend because it's wrong if the telescope slewed during
    # readout
    zdend = zdstart
    # Is this right? Doesn't it move non-linearly? [JTA]
    zdmid = ( zdstart + zdend ) / 2.
    amstart, ammid, amend = zd2am( np.array( [zdstart, zdmid, zdend] ) )
    airmass = ( amstart + 4. * ammid + amend ) / 6.
    # Simpson integration across 3 steps
    return airmass

def zd2am( zenithdistance ):
    # fitting formula from Pickering (2002)
    altitude = ( 90. - zenithdistance ) 
    airmass = 1./ ( np.sin( ( altitude + 244. / ( 165. + 47 * altitude**1.1 )
                            ) / 180. * np.pi ) )
    return airmass

def read_atmospheric_extinction(sso_extinction_table=SSO_EXTINCTION_TABLE):
    wl, ext = [], []
    #for entry in open( sso_extinction_table, 'r' ).xreadlines() :
    for entry in open( sso_extinction_table, 'r' ):
        line = entry.rstrip( '\n' )
        if not line.count( '*' ) and not line.count( '=' ):
            values = line.split()
            wl.append(  values[0] )
            ext.append( values[1] )    
    wl = np.array( wl ).astype( 'f' )
    ext= np.array( ext ).astype( 'f' )   
    return wl, ext

def combine_transfer_functions(path_list, path_out, use_all=False, sn_weight=True):
    """Read a set of transfer functions, combine them, and save to file."""
    # Make an empty array to hold all the individual transfer functions
    if use_all:
        path_list_good = path_list
    else:
        # Only keep paths where there is good flux calibration data
        path_list_good = []
        for path in path_list:
            try:
                good_psf = pf.getval(path, 'GOODPSF', 'FLUX_CALIBRATION')
            except KeyError:
                # There is no flux calibration data here
                continue
            if good_psf:
                path_list_good.append(path)
    n_file = len(path_list_good)
    n_pixel = pf.getval(path_list_good[0], 'NAXIS1')
    tf_array = np.zeros((n_file, n_pixel))
    med_sn = np.zeros(n_file)
    # Read the individual transfer functions
    for index, path in enumerate(path_list_good):
        tf_array[index, :] = pf.getdata(path, 'FLUX_CALIBRATION')[-1, :]
        # get the flux and sigma of spectrum and create a S/N spectrum:
        sn_spec = pf.getdata(path, 'FLUX_CALIBRATION')[0, :]/pf.getdata(path, 'FLUX_CALIBRATION')[2, :]
        # calculate a median S/N:
        med_sn[index] = np.nanmedian(sn_spec)
        # outout if needed:
        # print(index,path,med_sn[index],percent10,percent90)

    # print the S/N values so we have an idea of the range used:
    print('median S/N values for different std observations:',med_sn)
    # Make sure the overall scaling for each TF matches the others.
    # the scale is chosen to be the midpoint of the TF array (at index npix/2):
    scale = tf_array[:, n_pixel//2].copy()
    # dividing scale of each TF by the median scale.  This gives a value
    # to correct the TF to the median and align all the TFs vertically.
    scale /= np.median(scale)
    tf_array = (tf_array.T / scale).T
    # Combine them.
    # Here we take the mean, but we weight by (S/N)**2.  This assumes that
    # the variance is well propagated to the extracted std spectrum.
    # we could possibly place an upper limit on S/N, but looking at a
    # representative sample, the range seems to be median S/N~100 to 500.
    #  This is not a large dynamic range.
    # Using inverse because it's more stable when observed flux is low
    if (sn_weight):
        for i in range(n_file):
            tf_array[i,:] = med_sn[i]**2 * 1.0/tf_array[i,:]

        wtsum = nanmean(med_sn**2)
        tf_combined = 1.0 / (nanmean(tf_array, axis=0)/wtsum)

        
    else:
        tf_combined = 1.0 / nanmean(1.0 / tf_array, axis=0)

    save_combined_transfer_function(path_out, tf_combined, path_list_good)
    return

def save_combined_transfer_function(path_out, tf_combined, path_list):
    """Write the combined transfer function (and the individuals to file."""
    # Put the combined transfer function in the primary HDU
    primary_hdu = pf.PrimaryHDU(tf_combined)
    primary_hdu.header['HGFLXCAL'] = (HG_CHANGESET, 
                                      'Hg changeset ID for fluxcal code')
    # Copy the wavelength information into the new file
    header_input = pf.getheader(path_list[0])
    for key in ['CRVAL1', 'CDELT1', 'NAXIS1', 'CRPIX1', 'RO_GAIN']:
        primary_hdu.header[key] = header_input[key]
    zd = np.mean([pf.getval(path, 'ZDSTART') for path in path_list])
    primary_hdu.header['MEANZD'] = zd
    # Make an HDU list, which will also contain all the individual functions
    hdulist = pf.HDUList([primary_hdu])
    for index, path in enumerate(path_list):
        hdulist_tmp = pf.open(path)
        hdu = hdulist_tmp['FLUX_CALIBRATION']
        hdu.header['EXTVER'] = index + 1
        hdu.header['ORIGFILE'] = (
            os.path.basename(path), 'Originating file for this HDU')
        hdulist.append(hdu)
    if os.path.exists(path_out):
        os.remove(path_out)
    hdulist.writeto(path_out)
    return

def read_model(path):
    """Return the model encoded in a FITS file."""
    hdulist = pf.open(path)
    hdu = hdulist['FLUX_CALIBRATION']
    psf_parameters, model_name = read_model_parameters(hdu)
    ifu = IFU(path, hdu.header['PROBENUM'], flag_name=False)
    hdulist.close()
    xfibre = ifu.xpos_rel * np.cos(np.deg2rad(np.mean(ifu.ypos)))
    yfibre = ifu.ypos_rel
    wavelength = ifu.lambda_range
    model = model_flux(psf_parameters, xfibre, yfibre, wavelength, model_name)
    return model

def insert_fixed_parameters(parameters_dict, fixed_parameters):
    """Insert the fixed parameters into the parameters_dict."""
    if fixed_parameters is not None:
        for key, value in fixed_parameters.items():
            parameters_dict[key] = value
    return parameters_dict

def set_fixed_parameters(path_list, model_name, probenum=None):
    """Return fixed values for certain parameters."""
    fixed_parameters = {}
    if (model_name == 'ref_centre_alpha_dist_circ' or
        model_name == 'ref_centre_alpha_dist_circ_hdratm' or
        model_name == 'ref_centre_alpha_circ_hdratm'):
        header = pf.getheader(path_list[0])
        ifu = IFU(path_list[0], probenum, flag_name=False)
        ha_offset = ifu.xpos[0] - ifu.meanra  # The offset from the HA of the field centre
        ha_start = header['HASTART'] + ha_offset
        # The header includes HAEND, but this goes very wrong if the telescope
        # slews during readout. The equation below goes somewhat wrong if the
        # observation was paused, but somewhat wrong is better than very wrong.
        ha_end = ha_start + (ifu.exptime / 3600.0) * 15.0
        ha = 0.5 * (ha_start + ha_end)
        zenith_direction = np.deg2rad(parallactic_angle(
            ha, header['MEANDEC'], header['LAT_OBS']))
        fixed_parameters['zenith_direction'] = zenith_direction
    if (model_name == 'ref_centre_alpha_dist_circ_hdratm' or
        model_name == 'ref_centre_alpha_circ_hdratm'):
        fibre_table_header = pf.getheader(path_list[0], 'FIBRES_IFU')
        temperature = fibre_table_header['ATMTEMP']
        pressure = fibre_table_header['ATMPRES'] * millibar_to_mmHg
        vapour_pressure = (fibre_table_header['ATMRHUM'] * 
            saturated_partial_pressure_water(pressure, temperature))
        fixed_parameters['temperature'] = temperature
        fixed_parameters['pressure'] = pressure
        fixed_parameters['vapour_pressure'] = vapour_pressure
    if model_name == 'ref_centre_alpha_circ_hdratm':
        # Should take into account variation over course of observation
        # instead of just using the start value, which we do here:
        zd_eff = calc_eff_airmass(header,return_zd=True)
        fixed_parameters['zenith_distance'] = np.deg2rad(zd_eff)
        #fixed_parameters['zenith_distance'] = np.deg2rad(header['ZDSTART'])
    return fixed_parameters

def check_psf_parameters(psf_parameters, chunked_data):
    """Return True if the parameters look ok, False otherwise."""
    # Check if the star is comfortably inside the hexabundle
    if 'xcen_ref' in psf_parameters and 'ycen_ref' in psf_parameters:
        xcen_hexa = np.mean(chunked_data['xfibre'])
        ycen_hexa = np.mean(chunked_data['yfibre'])
        if np.sqrt((psf_parameters['xcen_ref'] - xcen_hexa)**2 + 
                   (psf_parameters['ycen_ref'] - ycen_hexa)**2) > 6.0:
            return False
    # Survived the checks
    return True

def primary_flux_calibrate(path_in, path_out, path_transfer_function):
    """Apply transfer function to reduced data."""
    hdulist = pf.open(path_in)
    remove_atmosphere_rss(hdulist)
    transfer_function = pf.getdata(path_transfer_function)
    hdulist[0].data *= transfer_function
    hdulist['VARIANCE'].data *= transfer_function**2
    hdulist[0].header['FCALFILE'] = (path_transfer_function,
                                     'Flux calibration file')
    hdulist[0].header['HGFLXCAL'] = (HG_CHANGESET, 
                                     'Hg changeset ID for fluxcal code')
    hdulist[0].header['BUNIT'] = ('10**(-16) erg /s /cm**2 /angstrom',
                                  'Units')
    hdulist.writeto(path_out)
    return

def median_filter_rotate(array, window):
    """Median filter with the array extended by rotating the end pieces."""
    good = np.where(np.isfinite(array))[0]
    array_ext = array[good[0]:good[-1]+1]
    end_values = (np.median(array_ext[:5*window]),
                  np.median(array_ext[-5*window:]))
    array_ext = np.hstack((2*end_values[0] - array_ext[window:0:-1],
                           array_ext,
                           2*end_values[1] - array_ext[-1:-1-window:-1]))
    result = np.zeros_like(array)
    result[good[0]:good[-1]+1] = (
        median_filter(array_ext, window)[window:-window])
    result[:good[0]] = array[:good[0]]
    result[good[-1]+1:] = array[good[-1]+1:]
    return result

def fit_sec_template_ppxf(path,doplot=False,verbose=False,tempfile='standards/kurucz_stds_raw_v5.fits',mdegree=8,lam1=3600.0,lam2=5700.0):
    """main routine that actually calls ppxf to do the fitting of the
    secondary flux calibration stars to model templates."""

    if (verbose):
        print('Fitting secondary stars to template using ppxf: %s' % str(path))

    # read spectrum of secondary star from the FLUX_CALIBRATION extension.
    # Put these into a temporary array, as we need to cut out Nans etc.
    lam_t,flux_t,sigma_t = read_flux_calibration_extension(path,gettel=False)

    # find the first and last good points in the spectrum as ppxf does not
    # like nans.  Also only fit within lam1 and lam2 wavelengths.  This is
    # particularly useful for ignoring the very ends, e.g. near dichroic.
    nlam = np.size(lam_t)
    for i in range(nlam):
        if (np.isfinite(flux_t[i]) & (lam_t[i]>lam1)):
            istart = i
            break
    for i in reversed(range(nlam)):
        if (np.isfinite(flux_t[i]) & (lam_t[i]<lam2)):
            iend = i
            break

    # check for other nans:
    nnan = 0
    for i in range(istart,iend):
        if (np.isnan(flux_t[i])):
            print(i,flux_t[i])
            nnan = nnan+1

    if (nnan > 0):
        print('WARNING: extra NaNs found in spectrum from FLUX_CALIBRATION extension: ',path)

    # TODO: remove any extra nans!
        
    if (verbose):
        print('first and last good pixels for secondary spectrum:',istart,iend)

    # specify good spectrum:
    lam = lam_t[istart:iend]
    flux = flux_t[istart:iend]
    sigma = sigma_t[istart:iend]
            
    # divide by median flux to avoid possible numerical issues with fitting:
    medflux = np.nanmedian(flux)
    flux = flux/medflux
    sigma = sigma/medflux

    # log rebin input spectrum using the ppxf log_rebin() function.
    # Returns the natural log (not log_10!!!) of wavelength values
    # and resampled spectrum.
    lamrange = np.array([lam[0],lam[-1]])
    logflux, loglam, velscale = log_rebin(lamrange,flux)
    logsigma = log_rebin(lamrange,sigma)[0]
    logsigma[np.isfinite(logsigma) == False] = np.nanmedian(logsigma)
    lam_gal = np.exp(loglam)
    if (verbose):
        print('Velocity scale after log rebinning: ',velscale)

    # read model templates:
    # TODO: robust checking that template file is present!
    temp_lam1, temp_kur1, model_feh, model_teff, model_g, model_mag = read_kurucz(tempfile,doplot=doplot,plot_iter=False,verbose=verbose)
    n_kur, n_lam_kur = np.shape(temp_kur1)
    
    # process templates to get them ready for fitting:
    # first reduce their resolution and sampling, as they are much higher resolution than our data
    # In principle we could do this differently for the red and blue arms, at least when getting
    # the TF, if not fitting the best fit template.
    lam_temp, templates_kur1 = resample_templates(temp_lam1,temp_kur1,verbose=verbose)

    # get range of wavelength:
    lamRange_temp = [np.min(lam_temp),np.max(lam_temp)]
        
    # now log rebinning of templates.  Do it for one to start with, so we know the number
    # of bins to use:
    tmp_tmp = log_rebin(lamRange_temp,templates_kur1[0,:],velscale=velscale)[0]
    templates_kur2 = np.zeros((n_kur,np.size(tmp_tmp)))
    # also define an array that will just contain the best few fitted templates:
    templates_kur3 = np.zeros((np.size(tmp_tmp),4))
    # now log rebin all templates:
    for i in range(n_kur):
        templates_kur2[i,:] = log_rebin(lamRange_temp,templates_kur1[i,:],velscale=velscale)[0] 

    # transpose templates:
    templates = np.transpose(templates_kur2)

    # define the number of templates:
    temp_n=n_kur

    # define the dv for ppxf.  The offset for the start of the
    # two spectra:
    c = 299792.458
    dv = np.log(lam_temp[0]/lam_gal[0])*c    # km/s
    if (verbose):
        print('dv in km/s: ',dv)
    
    # starting guess for velocity and dispersion:
    start = [0,50.0]

    # fit one template at a time.  Return and save the chisq
    # values (all of them) and the index for the best template:
    chisq = np.zeros(temp_n)
    chimin=1.0e10
    ibest = -9
    for i in range(temp_n):
        pp = ppxf(templates[:,i],logflux,logsigma, velscale, start,
              plot=False, moments=2, mdegree=mdegree,quiet=True,
              degree=-1, vsyst=dv, clean=False, lam=lam_gal)

        if (verbose):
            print(i,pp.chi2,model_teff[i],model_feh[i],model_g[i])
            
        chisq[i] = pp.chi2
        if (chisq[i] < chimin):
            ibest = i
            chimin = chisq[i]

    best_g = model_g[ibest]
    best_teff = model_teff[ibest]
    best_feh = model_feh[ibest]
    if (verbose):
        print('best template:',ibest,chimin,model_teff[ibest],model_feh[ibest],model_g[ibest])

    # now identify the templates closest in temperature and metallicity to the
    # best fit one.  We can then do a best fit allowing all of those templates
    # at once.  First decide on which gravity to use.  Just use the best fit model
    # for that, particularly as we only have 2 gravities.  Then loop through
    # each of the other models and find thoose that are close in teff or [Fe/H]
    for i in range(temp_n):
        if (model_g[i] == best_g):

            # metallicity (which is [Fe/H]) steps are in 0.5.  Teff steps are in 250K
            # we want to find the next best in Teff and [Fe/H]
            d_teff = abs(model_teff[i]-best_teff)
            d_feh = abs(model_feh[i]-best_feh)
            # check either side of best fit teff.  Find the one with the best chisq:
            chisq_best2 = 1.0e10
            if ((d_teff < 300) & (d_teff>200) & (d_feh<0.2)):
                if (chisq[i] < chisq_best2):
                    best_chisq2 = chisq[i]
                    best_teff2 = model_teff[i]
            # check either side of best fit [FeH].  Find the one with the best chisq:
            chisq_best2 = 1.0e10
            if ((d_feh < 0.6) & (d_feh>0.4) & (d_teff<200)):
                if (chisq[i] < chisq_best2):
                    best_chisq2 = chisq[i]
                    best_feh2 = model_feh[i]

    if (verbose):
        print(best_teff,best_teff2,best_feh,best_feh2)
        
    # assign the 4 best templates around the best fit to an array for the
    # final fit:
    n_best_temp = 0
    nn_best = np.zeros(4,dtype=int)
    for i in range(temp_n):
        good = False
        if (model_g[i] == best_g):
            if ((model_teff[i] == best_teff) & (model_feh[i] == best_feh)):
                good = True
            if ((model_teff[i] == best_teff) & (model_feh[i] == best_feh2)):
                good = True
            if ((model_teff[i] == best_teff2) & (model_feh[i] == best_feh)):
                good = True
            if ((model_teff[i] == best_teff2) & (model_feh[i] == best_feh2)):
                good = True

            if (good):
                templates_kur3[:,n_best_temp] = templates[:,i]
                nn_best[n_best_temp] = i
                n_best_temp = n_best_temp + 1
                if (verbose):
                    print(i,model_teff[i],model_feh[i])

    # redo the ppxf fitting with just the 4 best templates:
    pp = ppxf(templates_kur3,logflux,logsigma, velscale, start,
              plot=False, moments=2, mdegree=mdegree,quiet=True,
              degree=-1, vsyst=dv, clean=False, lam=lam_gal)

    if (verbose):
        print('chisq for optimal fit:',pp.chi2)
        for i in range(n_best_temp):
            ii = nn_best[i]
            print(i,ii,pp.weights[i],model_teff[ii],model_feh[ii],model_g[ii])

    # normalize weights to sum of 1:
    norm_weights = pp.weights/np.nansum(pp.weights)
            
    # now write the template numbers and weights to the FLUX_CALIBRATION
    # header:
    hdulist = pf.open(path,'update')
    header = hdulist['FLUX_CALIBRATION'].header
    header['TEMP1'] = (nn_best[0],'Best fit template to secondary std 1')
    header['TEMP2'] = (nn_best[1],'Best fit template to secondary std 2')
    header['TEMP3'] = (nn_best[2],'Best fit template to secondary std 3')
    header['TEMP4'] = (nn_best[3],'Best fit template to secondary std 4')
    header['TEMP1WT'] = (norm_weights[0],'Weight for best fit template to secondary std 1')
    header['TEMP2WT'] = (norm_weights[1],'Weight for best fit template to secondary std 2')
    header['TEMP3WT'] = (norm_weights[2],'Weight for best fit template to secondary std 3')
    header['TEMP4WT'] = (norm_weights[3],'Weight for best fit template to secondary std 4')
    header['TEMPVEL'] = (pp.sol[0],'Best fit template velocity (km/s)')
    header['TEMPSIG'] = (pp.sol[1],'Best fit template dispersion (km/s)')
    hdulist.flush()
    hdulist.close()
                
    return

def combine_template_weights(path_list,path_out,verbose=False):
    """look at all the weights from template fits to secondary std stars in
    separate frames in a given field and average the weights so we can derive
    a single optimal template for this star"""
    
    nframes = len(path_list)

    if (verbose):
        print('combining template weights from '+str(nframes)+' frames.')

    # set up arrays to contain weights and template numbers:
    weight_list = np.zeros((nframes,4))
    temp_list = np.zeros((nframes,4),dtype=int)
    temp_vel = np.zeros(nframes)
    temp_sig = np.zeros(nframes)
    
    # loop over individual frames and read in the weights from the headers:
    nf = 0
    for index, path in enumerate(path_list):

        weight_list[nf,0] = pf.getval(path,'TEMP1WT',extname='FLUX_CALIBRATION')
        weight_list[nf,1] = pf.getval(path,'TEMP2WT',extname='FLUX_CALIBRATION')
        weight_list[nf,2] = pf.getval(path,'TEMP3WT',extname='FLUX_CALIBRATION')
        weight_list[nf,3] = pf.getval(path,'TEMP4WT',extname='FLUX_CALIBRATION')
        temp_list[nf,0] = pf.getval(path,'TEMP1',extname='FLUX_CALIBRATION')
        temp_list[nf,1] = pf.getval(path,'TEMP2',extname='FLUX_CALIBRATION')
        temp_list[nf,2] = pf.getval(path,'TEMP3',extname='FLUX_CALIBRATION')
        temp_list[nf,3] = pf.getval(path,'TEMP4',extname='FLUX_CALIBRATION')
        temp_vel[nf] = pf.getval(path,'TEMPVEL',extname='FLUX_CALIBRATION')
        temp_sig[nf] = pf.getval(path,'TEMPSIG',extname='FLUX_CALIBRATION')

        nf = nf + 1

    # find the average velocity and sigma:
    vel = np.nanmean(temp_vel)
    vel_err = np.nanstd(temp_vel)/np.sqrt(float(nf))
    sig = np.nanmean(temp_sig)
    sig_err = np.nanstd(temp_sig)/np.sqrt(float(nf))
    if (verbose):
        print('mean template velocity:',vel,'+-',vel_err)
        print('mean template sigma:',sig,'+-',sig_err)

    # find all the unique templates:
    used_temp = np.unique(temp_list)
    nused_temp = np.size(used_temp)
    if (verbose):
        print('Unique templates used:',used_temp)

    # find the average weight for each unique template:
    used_weights = np.zeros(nused_temp)
    for nt in range(nused_temp):
            
        for i in range(nf):
            for j in range(4):
                if (temp_list[i,j] == used_temp[nt]):
                    used_weights[nt] = used_weights[nt] + weight_list[i,j]
        used_weights[nt] = used_weights[nt]/float(nf)
        if (verbose):
            print('template:',used_temp[nt],' weight: ',used_weights[nt])

    # finally, write the values to a binary table in the output file:

    # define the columns:
    col1 = pf.Column(name='templates', format='I', array=used_temp)
    col2 = pf.Column(name='weights', format='E', array=used_weights)
    cols = pf.ColDefs([col1, col2])
    hdutab = pf.BinTableHDU.from_columns(cols,name='TEMPLATE_LIST')
    # add best velocity and sigma to the header of the binary table:
    header = hdutab.header
    header['MTEMPVEL'] = (vel, 'Mean template velocity (km/s)')
    header['MTEMPSIG'] = (sig, 'Mean template dispersion (km/s)')
    hdutab.writeto(path_out,overwrite=True)
      
    return 

def derive_secondary_tf(path_list,path_list2,path_out,tempfile='standards/kurucz_stds_raw_v5.fits',verbose=False,doplot=False):
    """Use the best fit weights from template fits to secondary flux 
    calibration stars to derive a transfer function for each frame and
    write that to an extension in the data."""

    from ..manager import read_stellar_mags
    from ..qc.fluxcal import measure_band

    print('Deriving secondary transfer function')

    # Read in the templates, weights and kinematics of the best solution
    # template.
    hdu = pf.open(path_out,mode='update')
    datatab = hdu['TEMPLATE_LIST'].data
    temp_used = datatab['templates']
    temp_weights = datatab['weights']
    header_tab = hdu['TEMPLATE_LIST'].header
    vel = header_tab['MTEMPVEL']
    sig = header_tab['MTEMPSIG']

    if (verbose):
        print('templates and weights:',temp_used,temp_weights)
    
    # read in the templates:
    temp_lam1, temp_kur1, model_feh, model_teff, model_g, model_mag = read_kurucz(tempfile,doplot=False,plot_iter=False,verbose=verbose)

    # Reduce the template resolution and sampling, as they are much higher resolution than our data
    # (also done before fitting the templates):
    # we may want to adjust this to better match the red and blue arms.
    lam_temp, templates_kur1 = resample_templates(temp_lam1,temp_kur1,verbose=verbose)
    n_kur, n_lam_kur = np.shape(templates_kur1)

    # define array to hold the new optimal template:
    template_opt = np.zeros(n_lam_kur)
    
    # make the best template from a linear combination of the templates.
    for i in range(np.size(temp_used)):
        if (temp_weights[i] > 0.0):
            template_opt = template_opt + temp_weights[i] * templates_kur1[temp_used[i],:]
            
    # correct for velocity shift:
    lam_temp_opt = lam_temp*(1+vel/2.9979e5)

    # convolve best fit template for sigma:
    # TODO - this is not vital at this stage, and we should arguably do this separately
    # for the red and blue arms that have different resolution.
        
    # correct template for galactic extinction.  First need to get the the ra/dec of the
    # bundle for the star in question.  To do this, find the name of the star from the
    # FLUX_CALIBRATION header and then use the read_stellar_mags() function to get the
    # mags and the ra/dec:
    std_name = pf.getval(path_list[0],'STDNAME', 'FLUX_CALIBRATION')
    catalogue = read_stellar_mags()
    std_parameters = catalogue[std_name]
    if (verbose):
        print('getting star parameters for: ',std_name)
        print(std_parameters)
    std_ra = std_parameters['ra']
    std_dec = std_parameters['dec']
    theta, phi = dust.healpixAngularCoords(std_ra,std_dec)

    # now read the dust maps and find the E(B-V):
    for name, map_info in dust.MAPS_FILES.items():
        ebv = dust.EBV(name, theta, phi)
        if name == 'planck':
            correction_t = dust.MilkyWayDustCorrection(lam_temp_opt, ebv)

    if (verbose):
        print('std star: ',std_name,' coords:',std_ra,std_dec,theta,phi)
        print('Galactic extinction for std star, E(B-V): ',ebv)

    # and then actually correct the spectrum:
    template_opt_ext = template_opt/correction_t
        
    # derive template ab mags (relative).  This uses the SDSS bands and the measure_band
    # function in qc.fluxcal
    mag_temp = {}
    bands = 'ugriz'
    for band in bands:
        mag_temp[band] = measure_band(band,template_opt_ext,lam_temp_opt)
        if (verbose):
            print(band,mag_temp[band],std_parameters[band],mag_temp[band]-std_parameters[band])
    
    # normalize template based on ab mags.  Do this based on the g and r bands as these are
    # the bands contained within the SAMI spectral range:
    deltamag = ((mag_temp['g']-std_parameters['g'])+ (mag_temp['r']-std_parameters['r']))/2.0
    if (verbose):
        print('mean delta mag for g and r bands:',deltamag)

    fluxscale = 10**(-0.4*deltamag)

    template_opt_ext_scale = template_opt_ext/fluxscale

    # recalculate the mags after scaling the photometry.  This is really just a check that
    # the scaling was in the right direction.  We could just apply the offsets to all the
    # mags.
    mag_temp2 = {}
    for band in bands:
        mag_temp2[band] = measure_band(band,template_opt_ext_scale,lam_temp_opt)
        if (verbose):
            print(band,mag_temp2[band],std_parameters[band],mag_temp2[band]-std_parameters[band])
        
        
    # write template to TEMPLATES2combined.fits file as the best template for this
    # field/star.  For this we need the resampled and redshifted flux with extinction
    # and scaling added.  We can also write the nominal magnitudes that are re-scaled
    # to the combined g and r bands.  This will all go into an extension called
    # TEMPLATE_OPT.  This is written as a FITS binary table:
    col1 = pf.Column(name='wavelength', format='E', array=lam_temp_opt)
    col2 = pf.Column(name='flux', format='E', array=template_opt_ext_scale)
    cols = pf.ColDefs([col1, col2])
    hdutab = pf.BinTableHDU.from_columns(cols,name='TEMPLATE_OPT')
    header = hdutab.header
    for band in bands:
        header['TEMPMAG'+band.upper()] = (mag_temp2[band],band+' mag of optimal template (scaled)')
    
    hdu.append(hdutab)
    hdu.flush()
    hdu.close()

    # loop over all the individual frames, read the secondary std star fluxes and
    # find the TF:
    nfiles = 0
    for index, path1 in enumerate(path_list):

        path2 = path_list2[index]
        if (verbose):
            print('deriving TF for ',path1)
            print('and ',path2)

        # read the FLUX_CALIBRATION extension to get the star spectrum:
        lam_b,flux_b,sigma_b = read_flux_calibration_extension(path1,gettel=False)
        lam_r,flux_r,sigma_r, tel_r = read_flux_calibration_extension(path2,gettel=True)

        # if this is the first frame, set up arrays to hold different TFs:
        if (index == 0):
            tf_b = np.zeros((len(path_list),np.size(lam_b)))
            tf_r = np.zeros((len(path_list),np.size(lam_r)))

        # correct the red arm for telluric abs:
        flux_r = flux_r * tel_r

        # resample the template spectrum onto the SAMI wavelength scale.
        # TODO: need to check whether the template wavelength scale is air or vacuum.  Its probably
        # vacuum, so we should fix this up at an early stage, before we fit the velocity.  However
        # given that we fit the velocity anyway, to zeroth order the wavelength shift will be
        # taken out.
        #
        # this internal SAMI resampling code does not see to work in this case:
        #temp_flux_b = rebin_flux(lam_b,lam_temp_opt,template_opt_ext_scale)
        #temp_flux_r = rebin_flux(lam_r,lam_temp_opt,template_opt_ext_scale)
        #
        # currently use an external resampling routine and has been put into 
        temp_flux_b = spectres(lam_b,lam_temp_opt,template_opt_ext_scale)
        temp_flux_r = spectres(lam_r,lam_temp_opt,template_opt_ext_scale)

        # take the ratio:
        ratio_b = flux_b/temp_flux_b
        ratio_r = flux_r/temp_flux_r

        # get ratio errors:
        ratio_sig_b = sigma_b/temp_flux_b
        ratio_sig_r = (sigma_r*tel_r)/temp_flux_r

        # fit the ratios
        ratio_sp_b = fit_spline_secondary(lam_b,ratio_b,ratio_sig_b,verbose=verbose)
        ratio_sp_r = fit_spline_secondary(lam_r,ratio_r,ratio_sig_r,verbose=verbose)

        # store TFs in array for combining later:
        tf_b[index,:] = ratio_sp_b
        tf_r[index,:] = ratio_sp_r
        
        # generate some diagnostic plots if needed:
        if (doplot):
            fig1 = py.figure()
            ax1_1 = fig1.add_subplot(311)

            # first plot the template without extinction:
            ax1_1.plot(lam_temp_opt,template_opt_ext_scale*correction_t,'k',alpha=0.5)
            # then the template with extinction applied (both are scaled to mags):
            ax1_1.plot(lam_temp_opt,template_opt_ext_scale,'k')

            # now plot the SAMI data:
            ax1_1.plot(lam_b,flux_b,'b')
            ax1_1.plot(lam_r,flux_r,'r')
            ax1_1.plot(lam_b,flux_b+5.0*sigma_b,':',color='b')
            ax1_1.plot(lam_r,flux_r+5.0*sigma_r,':',color='r')
            ax1_1.plot(lam_b,flux_b-5.0*sigma_b,':',color='b')
            ax1_1.plot(lam_r,flux_r-5.0*sigma_r,':',color='r')

            # then the template on the SAMI wavelength scale:
            ax1_1.plot(lam_b,temp_flux_b,'c')
            ax1_1.plot(lam_r,temp_flux_r,'m')
            xmin = np.min(lam_b)-100.0
            xmax = np.max(lam_r)+100.0
            title = os.path.basename(path1)
            ax1_1.set(xlim=[xmin,xmax],xlabel='Wavelength (Ang.)',ylabel='Relative flux',title=title)

            # plot the ratios:
            ax1_2 = fig1.add_subplot(312)
            ax1_2.axhline(1.0,color='k')
            ax1_2.plot(lam_b,ratio_b,'b')
            ax1_2.plot(lam_r,ratio_r,'r')
            # don't plot these as they take some time to generate:
            #ax1_2.plot(lam_b,median_filter_nan_1d(ratio_b,51),'g')
            #ax1_2.plot(lam_r,median_filter_nan_1d(ratio_r,51),'g')
            # plot filtered stdev
            #ax1_2.plot(lam_b,median_filter_nan_1d(ratio_b,51)+ 5.0*median_filter_nan_1d(ratio_sig_b,51),':',color='g')
            #ax1_2.plot(lam_b,median_filter_nan_1d(ratio_b,51)- 5.0*median_filter_nan_1d(ratio_sig_b,51),':',color='g')
            #ax1_2.plot(lam_r,median_filter_nan_1d(ratio_r,51)+ 5.0*median_filter_nan_1d(ratio_sig_r,51),':',color='g')
            #ax1_2.plot(lam_r,median_filter_nan_1d(ratio_r,51)- 5.0*median_filter_nan_1d(ratio_sig_r,51),':',color='g')

            ax1_2.set(xlim=[xmin,xmax],xlabel='Wavelength (Ang.)',ylabel='SAMI/template')

            # plot the spline fits to the ratio:
            ax1_2.plot(lam_b,ratio_sp_b,'c')
            ax1_2.plot(lam_r,ratio_sp_r,'m')
        
            ax1_3 = fig1.add_subplot(313)
            ax1_3.axhline(1.0,color='k')
            ax1_3.plot(lam_b,ratio_sp_b,'c')
            ax1_3.plot(lam_r,ratio_sp_r,'m')
            ax1_3.set(xlim=[xmin,xmax],xlabel='Wavelength (Ang.)',ylabel='SAMI/template')

        # write the TF to the individual frame.  We will write this to a binary table
        # as it makes it easier to know what the format is after the fact (i.e. all the
        # columns have proper headings etc):
        hdu_b = pf.open(path1,mode='update')
        # remove old version of the FLUX_CALIBRATION2 extension:
        try:
            hdu_b.pop('FLUX_CALIBRATION2')
        except KeyError:
            print('no old FLUX_CALIBRATION2 extension found')

        col1 = pf.Column(name='wavelength', format='E', array=lam_b)
        col2 = pf.Column(name='flux', format='E', array=flux_b)
        col3 = pf.Column(name='template', format='E', array=temp_flux_b)
        col4 = pf.Column(name='transfer_fn', format='E', array=ratio_sp_b)
        cols = pf.ColDefs([col1, col2, col3, col4])
        hdutab = pf.BinTableHDU.from_columns(cols,name='FLUX_CALIBRATION2')
        hdu_b.append(hdutab)
        hdu_b.flush()
        hdu_b.close()

        # repeat for the red frame:
        hdu_r = pf.open(path2,mode='update')
        # remove old version of the FLUX_CALIBRATION2 extension:
        try:
            hdu_r.pop('FLUX_CALIBRATION2')
        except KeyError:
            print('no old FLUX_CALIBRATION2 extension found')

        col1 = pf.Column(name='wavelength', format='E', array=lam_r)
        col2 = pf.Column(name='flux', format='E', array=flux_r)
        col3 = pf.Column(name='template', format='E', array=temp_flux_r)
        col4 = pf.Column(name='transfer_fn', format='E', array=ratio_sp_r)
        cols = pf.ColDefs([col1, col2, col3, col4])
        hdutab = pf.BinTableHDU.from_columns(cols,name='FLUX_CALIBRATION2')
        hdu_r.append(hdutab)
        hdu_r.flush()
        hdu_r.close()
        

        nfiles = nfiles + 1
        
    # combine the different TFs together to get a mean TF for this field:
    # TODO: consider more robust combination.  e.g. scale before combine,
    # remove outliers etc.
    tf_mean_b = np.nanmean(tf_b,axis=0)
    tf_mean_r = np.nanmean(tf_r,axis=0)

    # copy the TRANSFER2combined.fits file to the CCD_2 directory.
    ccd2_path = os.path.dirname(path_list2[0])+'/'+os.path.basename(path_out)
    copyfile(path_out,ccd2_path)
    
    # write the tf_mean_b and tf_mean_r to the TRANSFER2combined.fits file.
    # write it to a binary table:
    # first blue:
    hdu = pf.open(path_out,mode='update')
    col1 = pf.Column(name='wavelength', format='E', array=lam_b)
    col2 = pf.Column(name='tf_average', format='E', array=tf_mean_b)
    cols = pf.ColDefs([col1, col2])
    hdutab = pf.BinTableHDU.from_columns(cols,name='TF_MEAN')
    hdu.append(hdutab)
    hdu.flush()
    hdu.close()
    
    # now for the red one:
    hdu = pf.open(ccd2_path,mode='update')
    col1 = pf.Column(name='wavelength', format='E', array=lam_r)
    col2 = pf.Column(name='tf_average', format='E', array=tf_mean_r)
    cols = pf.ColDefs([col1, col2])
    hdutab = pf.BinTableHDU.from_columns(cols,name='TF_MEAN')
    hdu.append(hdutab)
    hdu.flush()
    hdu.close()
    
    # plot the combined TFs:
    if (doplot):
        fig2 = py.figure()
        ax2_1 = fig2.add_subplot(111)
        ax2_1.axhline(1.0,color='k')
        for i in range(nfiles):
            ax2_1.plot(lam_b,tf_b[i,:],'c')
            ax2_1.plot(lam_r,tf_r[i,:],'m')
            
        ax2_1.plot(lam_b,tf_mean_b,'b')
        ax2_1.plot(lam_r,tf_mean_r,'r')
        ax2_1.set(xlim=[xmin,xmax],xlabel='Wavelength (Ang.)',ylabel='SAMI/template')

    return

def apply_secondary_tf(path1,path2,path_out1,path_out2,use_av_tf_sec=False,verbose=False,force=False):
    """Apply a previously measured secondary transfer function to the spectral
    data.  Optionally to use an average tranfer function.  force=True will force the
    correction to be made even if it has already been done."""

    # open files:
    hdulist1 = pf.open(path1,mode='update')
    hdulist2 = pf.open(path2,mode='update')

    # if force=True, then we force the correction, even though its already been done.
    if (not force):
        # check to see if done already.
        try:
            seccor = hdulist1[0].header['SECCOR']
            if (seccor):
                print('SECCOR keyword is True.  Not correcting ',path1)
                hdulist1.close()
                return
        except KeyError:
            seccor = False
    
        # also check if the red frame has been corrected already:
        try:
            seccor = hdulist2[0].header['SECCOR']
            if (seccor):
                print('SECCOR keyword is True.  Not correcting ',path2)
                hdulist2.close()
                return
        except KeyError:
            seccor = False

        
    # read TF from FLUX_CALIBRATION2 extension.
    tf_b = hdulist1['FLUX_CALIBRATION2'].data['transfer_fn']
    tf_r = hdulist2['FLUX_CALIBRATION2'].data['transfer_fn']

    # if needed, read average TF from TRANSFER2combined.fits file.
    if (use_av_tf_sec):
        hdulist_tf1 = pf.open(path_out1,mode='update')
        tf_av_b = hdulist_tf1['TF_MEAN'].data['tf_average']
        lam_av_b = hdulist_tf1['TF_MEAN'].data['wavelength']
        
        hdulist_tf2 = pf.open(path_out2,mode='update')
        tf_av_r = hdulist_tf2['TF_MEAN'].data['tf_average']
        lam_av_r = hdulist_tf2['TF_MEAN'].data['wavelength']

        hdulist_tf1.close()
        hdulist_tf2.close()
        
        # Need to think about how to scale the average - do we assume that the
        # global scaling is correct - we probably should, we assume that for the rescale_frames()
        # part.  If we are using the mean TF, then scale this so it has the same normalization
        # as the individual TFs.  That is, we are only using the average TF to define the shape
        # of the TF, not the normalization.
        ratio_b = tf_b/tf_av_b
        ratio_r = tf_r/tf_av_r

        # average the ratios between reasonable wavelength ranges, not using the very ends
        # that can move around a little more:
        idx_b = np.where((lam_av_b>4500.0) & (lam_av_b<5500.0))  
        idx_r = np.where((lam_av_r>6500.0) & (lam_av_r<7200.0))  
        scale_b = np.nanmean(ratio_b[idx_b]) 
        scale_r = np.nanmean(ratio_b[idx_r])

        # have a single scaling for blue and red:
        scale = (scale_b+scale_r)/2.0
        
        # generate
        tf_av_b = tf_av_b * scale
        tf_av_r = tf_av_r * scale
        
        if (verbose):
            print('scale_b: ',scale_b)
            print('scale_r: ',scale_r)
    
    # apply the TF to data and variance:
    hdulist1[0].data /= tf_b
    hdulist1['VARIANCE'].data /= tf_b**2

    hdulist2[0].data /= tf_r
    hdulist2['VARIANCE'].data /= tf_r**2
        
    # write FITS header keyword to say its done.
    hdulist1[0].header['SECCOR'] = (True,'Flag to indicate correction by secondary flux cal')
    hdulist2[0].header['SECCOR'] = (True,'Flag to indicate correction by secondary flux cal')

    hdulist1.flush()
    hdulist2.flush()
    hdulist1.close()
    hdulist2.close()
        
    return


def read_flux_calibration_extension(infile,gettel=False):
    """Read the FLUX_CALIBRATION extension for a science frame so we can
    get the spectrum for secondary calibration and any other things we need.  
    May not work with the format of a primary standard FLUX_CALIBRATION 
    extension - need to check"""
    
    # open the file:
    hdulist = pf.open(infile)
    primary_header=hdulist['PRIMARY'].header

    # read WCS:
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CDELT1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']
    
    # define wavelength array:
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1

    # get data from FLUX_CALIBRATION extension:
    fdata = hdulist['FLUX_CALIBRATION'].data
    fc_header=hdulist['FLUX_CALIBRATION'].header
        
    if (gettel):
        tel = fdata[5,:]
        
    flux = fdata[0, :]
    sigma = fdata[2, :]

    hdulist.close()
    
    if (gettel):
        return lam,flux,sigma,tel        
    else:
        return lam,flux,sigma

    
###########################################################################
# read in Krurucz from SDSS templates list.
#
def read_kurucz(infile,doplot=False,plot_iter=False,verbose=False):

    # open the file:
    hdulist = pf.open(infile)

    # get the model info and mags:
    table_data = hdulist[1].data
    
    # get individual model data cols:
    model_name=table_data.field('MODEL')
    model_feh =table_data.field('FEH')
    model_teff =table_data.field('TEFF')
    model_g =table_data.field('G')
    model_mag =table_data.field('MAG')
    
    # now read in the actual model spectra:
    model_flux = hdulist[0].data
    nmodel,nlam = np.shape(model_flux)
    if (verbose):
        print('number of template models:',nmodel)
        print('number of template wavelength bins:',nlam)

    # and get the wavelength array;
    primary_header=hdulist['PRIMARY'].header
    crval1=primary_header['CRVAL1']
    cdelt1=primary_header['CD1_1']
    crpix1=primary_header['CRPIX1']
    naxis1=primary_header['NAXIS1']
    x=np.arange(naxis1)+1
    L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    lam=L0+x*cdelt1

    # if plot requested, then do it:
    if (doplot):
        fig1 = py.figure()
        ax1 = fig1.add_subplot(211)
        ax2 = fig1.add_subplot(212)
        for i in range(nmodel):
            ax1.cla()
            ax2.cla()
            ax1.set(xlim=[3600,7000])
            ax2.set(xlim=[3600,7000])
            ax1.plot(lam,model_flux[i,:])
            label = '[Fe/H]='+str(model_feh[i])+' Teff='+str(model_teff[i])+' G='+str(model_g[i]) 
            ax1.text(0.5,1.1,label,verticalalignment='center',transform=ax1.transAxes)
            if (i<(nmodel-1)):
                label2 = '[Fe/H]='+str(model_feh[i+1])+' Teff='+str(model_teff[i+1])+' G='+str(model_g[i+1]) 
                ax2.plot(lam,model_flux[i,:]/model_flux[i+1,:])
                ax2.text(0.5,0.9,label+'/'+label2,verticalalignment='center',horizontalalignment='center',transform=ax2.transAxes)

            py.draw()
            if (plot_iter):
                yn=input('continue?')

    return lam, model_flux, model_feh, model_teff, model_g, model_mag


def resample_templates(temp_lam1,temp_kur1,nrebin=10,verbose=False):
    """Resample Kurucz high resolution model templates to power resolution to
    make the fitting faster/easier."""

    # factor to scale/zoom:
    zfact = 1.0/float(nrebin)
    
    # set up arrays:
    n_kur, n_lam_kur = np.shape(temp_kur1)
    templates_kur1 = np.zeros((n_kur,int(n_lam_kur/nrebin)))
    
    # convolve templates to lower resolution to better match data:
    lam_temp = zoom(temp_lam1,zfact)
    for i in range(n_kur):
        templates_kur1[i,:] = zoom(gaussian_filter1d(temp_kur1[i,:],float(nrebin)),zfact)

    if (verbose):
        print('template wavelength binsize:',lam_temp[1]-lam_temp[0])

    return lam_temp,templates_kur1

###############################################################################
# spectres routine from: https://github.com/ACCarnall/SpectRes
#
def spectres(new_spec_wavs, old_spec_wavs, spec_fluxes, spec_errs=None):

    """ 
    Function for resampling spectra (and optionally associated uncertainties) onto a new wavelength basis.
    Parameters
    ----------
    new_spec_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired for the spectrum or spectra.
    old_spec_wavs : numpy.ndarray
        1D array containing the current wavelength sampling of the spectrum or spectra.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths specified in old_spec_wavs, last dimension must correspond to the shape of old_spec_wavs.
        Extra dimensions before this may be used to include multiple spectra.
    spec_errs : numpy.ndarray (optional)
        Array of the same shape as spec_fluxes containing uncertainties associated with each spectral flux value.
    
    Returns
    -------
    resampled_fluxes : numpy.ndarray
        Array of resampled flux values, first dimension is the same length as new_spec_wavs, other dimensions are the same as spec_fluxes
    resampled_errs : numpy.ndarray
        Array of uncertainties associated with fluxes in resampled_fluxes. Only returned if spec_errs was specified.
    """

    # Generate arrays of left hand side positions and widths for the old and new bins
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_widths = np.zeros(old_spec_wavs.shape[0])
    spec_lhs = np.zeros(old_spec_wavs.shape[0])
    spec_lhs[0] = old_spec_wavs[0] - (old_spec_wavs[1] - old_spec_wavs[0])/2
    spec_widths[-1] = (old_spec_wavs[-1] - old_spec_wavs[-2])
    spec_lhs[1:] = (old_spec_wavs[1:] + old_spec_wavs[:-1])/2
    spec_widths[:-1] = spec_lhs[1:] - spec_lhs[:-1]

    filter_lhs = np.zeros(new_spec_wavs.shape[0]+1)
    filter_widths = np.zeros(new_spec_wavs.shape[0])
    filter_lhs[0] = new_spec_wavs[0] - (new_spec_wavs[1] - new_spec_wavs[0])/2
    filter_widths[-1] = (new_spec_wavs[-1] - new_spec_wavs[-2])
    filter_lhs[-1] = new_spec_wavs[-1] + (new_spec_wavs[-1] - new_spec_wavs[-2])/2
    filter_lhs[1:-1] = (new_spec_wavs[1:] + new_spec_wavs[:-1])/2
    filter_widths[:-1] = filter_lhs[1:-1] - filter_lhs[:-2]

    # Check that the range of wavelengths to be resampled_fluxes onto falls within the initial sampling region
    # SMC change, allow new wavelength ranges that are outside the range of the old ones.
    #if filter_lhs[0] < spec_lhs[0] or filter_lhs[-1] > spec_lhs[-1]:
    #    raise ValueError("spectres: The new wavelengths specified must fall within the range of the old wavelength values.")

    #Generate output arrays to be populated
    resampled_fluxes = np.zeros(spec_fluxes[...,0].shape + new_spec_wavs.shape)

    if spec_errs is not None:
        if spec_errs.shape != spec_fluxes.shape:
            raise ValueError("If specified, spec_errs must be the same shape as spec_fluxes.")
        else:
            resampled_fluxes_errs = np.copy(resampled_fluxes)

    start = 0
    stop = 0

    # Calculate the new spectral flux and uncertainty values, loop over the new bins
    for j in range(new_spec_wavs.shape[0]):

        # Find the first old bin which is partially covered by the new bin
        while spec_lhs[start+1] <= filter_lhs[j]:
            start += 1

        # Find the last old bin which is partially covered by the new bin
        while spec_lhs[stop+1] < filter_lhs[j+1]:
            stop += 1

        # if there is not an overlap, then set the value to NaN:
        # (SMC addition):
        if ((start == 0) & (stop == 0)):
            resampled_fluxes[...,j] = np.nan
        # If the new bin falls entirely within one old bin the are the same the new flux and new error are the same as for that bin
        elif stop == start:

            resampled_fluxes[...,j] = spec_fluxes[...,start]
            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = spec_errs[...,start]

        # Otherwise multiply the first and last old bin widths by P_ij, all the ones in between have P_ij = 1 
        else:

            start_factor = (spec_lhs[start+1] - filter_lhs[j])/(spec_lhs[start+1] - spec_lhs[start])
            end_factor = (filter_lhs[j+1] - spec_lhs[stop])/(spec_lhs[stop+1] - spec_lhs[stop])

            spec_widths[start] *= start_factor
            spec_widths[stop] *= end_factor

            # Populate the resampled_fluxes spectrum and uncertainty arrays
            resampled_fluxes[...,j] = np.sum(spec_widths[start:stop+1]*spec_fluxes[...,start:stop+1], axis=-1)/np.sum(spec_widths[start:stop+1])

            if spec_errs is not None:
                resampled_fluxes_errs[...,j] = np.sqrt(np.sum((spec_widths[start:stop+1]*spec_errs[...,start:stop+1])**2, axis=-1))/np.sum(spec_widths[start:stop+1])
            
            # Put back the old bin widths to their initial values for later use
            spec_widths[start] /= start_factor
            spec_widths[stop] /= end_factor


    # If errors were supplied return the resampled_fluxes spectrum and error arrays
    if spec_errs is not None:
        return resampled_fluxes, resampled_fluxes_errs

    # Otherwise just return the resampled_fluxes spectrum array
    else: 
        return resampled_fluxes


def median_filter_nan(im,filt):
    """Function to median filter an array with correction
    for NaN values."""
    
    V = im.copy()
    V[im!=im]=0
    print(np.shape(V))
    print(filt)
    VV = median_filter(V,size=filt)

    W = 0*im.copy()+1
    W[im!=im] = 0
    WW = median_filter(W,size=filt)

    im_med = VV/WW

    return im_med

def median_filter_nan_1d(spec,filt):
    """Function to median filter a 1D array with correction
    for NaN values."""

    n = np.size(spec)

    medspec = np.zeros(n)

    hsize = int(filt/2)
    for i in range(n):

        i1 = max(0,i-hsize)
        i2 = min(n-1,i+hsize)

        medspec[i] = np.nanmedian(spec[i1:i2])

    return medspec
