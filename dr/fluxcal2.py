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

-- ref_centre_alpha_dist_circ --

The same as ref_centre_alpha_angle_circ, but with the zenith direction
fixed.

-- ref_centre_alpha_angle_circ_atm --

The same as ref_centre_alpha_angle_circ, but with atmospheric values
as free parameters too. Note, however, that the atmospheric parameters
are completely degenerate with each other and with ZD.
"""

import os

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.filters import median_filter, gaussian_filter1d
from scipy.stats.stats import nanmean

from astropy import coordinates as coord
from astropy import units
from astropy.io import fits as pf
from astropy import __version__ as astropy_version

from .. import utils
from ..utils.ifu import IFU
from ..utils.mc_adr import parallactic_angle

HG_CHANGESET = utils.hg_changeset(__file__)

STANDARD_CATALOGUES = ('./standards/ESO/ESOstandards.dat',
                       './standards/Bessell/Bessellstandards.dat')
SSO_EXTINCTION_TABLE = './standards/extinsso.tab'

REFERENCE_WAVELENGTH = 5000.0

FIBRE_RADIUS = 0.798

TELLURIC_BANDS = np.array([[6850, 6960], 
                           [7130, 7360], 
                           [7560, 7770], 
                           [8100, 8360]])

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
    xfibre = ifu.xpos_rel
    yfibre = ifu.ypos_rel
    chunked_data = {'data': data,
                    'variance': variance,
                    'wavelength': wavelength,
                    'xfibre': xfibre,
                    'yfibre': yfibre}
    return chunked_data

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
        flux_sub = moffat_normalised(parameters, xfibre_sub, yfibre_sub, 
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
             wavelength, model_name, fixed_parameters=None):
    """Return the residual in each fibre for the given model."""
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)
    parameters_dict = insert_fixed_parameters(parameters_dict, 
                                              fixed_parameters)
    model = model_flux(parameters_dict, xfibre, yfibre, wavelength, model_name)
    # 2dfdr variance is just plain wrong for fibres with little or no flux!
    # Try replacing with something like sqrt(flux), but with a floor
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
    return res

def fit_model_flux(datatube, vartube, xfibre, yfibre, wavelength, model_name,
                   fixed_parameters=None):
    """Fit a model to the given datatube."""
    par_0_dict = first_guess_parameters(datatube, vartube, xfibre, yfibre, 
                                        wavelength, model_name)
    par_0_vector = parameters_dict_to_vector(par_0_dict, model_name)
    args = (datatube, vartube, xfibre, yfibre, wavelength, model_name,
            fixed_parameters)
    parameters_vector = leastsq(residual, par_0_vector, args=args)[0]
    parameters_dict = parameters_vector_to_dict(parameters_vector, model_name)
    return parameters_dict

def first_guess_parameters(datatube, vartube, xfibre, yfibre, wavelength, 
                           model_name):
    """Return a first guess to the parameters that will be fitted."""
    par_0 = {}
    #weighted_data = np.sum(datatube / vartube, axis=1)
    weighted_data = np.nansum(datatube, axis=1)
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
    elif model_name == 'ref_centre_alpha_dist_circ':
        par_0['flux'] = np.nansum(datatube, axis=0)
        par_0['background'] = np.zeros(len(par_0['flux']))
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
    elif model_name == 'ref_centre_alpha_dist_circ':
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
    elif model_name == 'ref_centre_alpha_dist_circ':
        n_slice = (len(parameters_vector) - 5) / 2
        parameters_dict['flux'] = parameters_vector[0:n_slice]
        parameters_dict['background'] = parameters_vector[n_slice:2*n_slice]
        parameters_dict['xcen_ref'] = parameters_vector[-5]
        parameters_dict['ycen_ref'] = parameters_vector[-4]
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
            parameters_dict['xcen_ref'] -
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] -
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
            parameters_dict['xcen_ref'] - 
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] - 
            np.cos(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance']))
        parameters_array['alphax'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['alphay'] = (
            alpha(wavelength, parameters_dict['alpha_ref']))
        parameters_array['beta'] = parameters_dict['beta']
        parameters_array['rho'] = np.zeros(len(wavelength))
        if len(parameters_dict['flux']) == len(parameters_array):
            parameters_array['flux'] = parameters_dict['flux']
        if len(parameters_dict['background']) == len(parameters_array):
            parameters_array['background'] = parameters_dict['background']
    elif model_name == 'ref_centre_alpha_angle_circ_atm':
        parameters_array['xcen'] = (
            parameters_dict['xcen_ref'] -
            np.sin(parameters_dict['zenith_direction']) * 
            dar(wavelength, parameters_dict['zenith_distance'],
                temperature=parameters_dict['temperature'],
                pressure=parameters_dict['pressure'],
                vapour_pressure=parameters_dict['vapour_pressure']))
        parameters_array['ycen'] = (
            parameters_dict['ycen_ref'] -
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
        parameters_array['rho'] = np.zeros(len(wavelength))
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
    seaLevelDry = ( 64.328 + ( 29498.1 / ( 146. - ( 1 / wl**2. ) ) )
                    + 255.4 / ( 41. - ( 1. / wl**2. ) ) )
    altitudeCorrection = ( 
        ( pressure * ( 1. + (1.049 - 0.0157*temperature ) * 1e-6 * pressure ) )
        / ( 720.883 * ( 1. + 0.003661 * temperature ) ) )
    vapourCorrection = ( ( 0.0624 - 0.000680 / wl**2. )
                         / ( 1. + 0.003661 * temperature ) ) * vapour_pressure
    return 1e-6 * (seaLevelDry * altitudeCorrection - vapourCorrection) + 1

def derive_transfer_function(path_list, max_sep_arcsec=60.0,
                             catalogues=STANDARD_CATALOGUES,
                             model_name='ref_centre_alpha_dist_circ'):
    """Derive transfer function and save it in each FITS file."""
    # First work out which star we're looking at, and which hexabundle it's in
    star_match = match_standard_star(
        path_list[0], max_sep_arcsec=max_sep_arcsec, catalogues=catalogues)
    if star_match is None:
        raise ValueError('No standard star found in the data.')
    standard_data = read_standard_data(star_match)
    # Read the observed data, in chunks
    chunked_data = read_chunked_data(path_list, star_match['probenum'])
    # Fit the PSF
    fixed_parameters = set_fixed_parameters(path_list, model_name)
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
    for path in path_list:
        ifu = IFU(path, star_match['probenum'], flag_name=False)
        observed_flux, observed_background = extract_total_flux(
            ifu, psf_parameters, model_name)
        save_extracted_flux(path, observed_flux, observed_background,
                            star_match, psf_parameters, model_name,
                            good_psf)
        transfer_function = take_ratio(
            standard_data['flux'], 
            standard_data['wavelength'], 
            observed_flux, 
            ifu.lambda_range)
        save_transfer_function(path, transfer_function)
    return

def match_standard_star(filename, max_sep_arcsec=60.0, 
                        catalogues=STANDARD_CATALOGUES):
    """Return details of the standard star that was observed in this file."""
    fibre_table = pf.getdata(filename, 'FIBRES_IFU')
    probenum_list = np.unique([fibre['PROBENUM'] for fibre in fibre_table
                               if 'SKY' not in fibre['PROBENAME']])
    for probenum in probenum_list:
        this_probe = (fibre_table['PROBENUM'] == probenum)
        ra = np.mean(fibre_table['FIB_MRA'][this_probe])
        dec = np.mean(fibre_table['FIB_MDEC'][this_probe])
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
        index = np.loadtxt(index_path, dtype='S')
        for star in index:
            RAstring = '%sh%sm%ss' % ( star[2], star[3], star[4] )
            Decstring= '%sd%sm%ss' % ( star[5], star[6], star[7] )
            coords_star = coord.ICRSCoordinates( RAstring, Decstring )
            ra_star = coords_star.ra.degrees
            dec_star= coords_star.dec.degrees
            ### BUG IN ASTROPY.COORDINATES ###
            if astropy_version == '0.2.0' and star[5] == '-' and dec_star > 0:
                dec_star *= -1.0
                print 'Upgrade your version of astropy!!!!'
                print 'Version 0.2.0 has a major bug in coordinates!!!!'
            sep = coord.angles.AngularSeparation(
                ra, dec, ra_star, dec_star, units.degree).arcsecs
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
    for index, psf_parameters_slice in enumerate(psf_parameters_array):
        flux[index], background[index] = extract_flux_slice(
            ifu.data[:, index], ifu.var[:, index], ifu.xpos_rel, ifu.ypos_rel,
            psf_parameters_slice)
    if clip is not None:
        # Clip out discrepant data. Wavelength slices are targeted based on
        # their overall deviation from the model, but within a slice only
        # positive features (like cosmic rays) are clipped.
        model_parameters = psf_parameters.copy()
        model_parameters['flux'] = flux
        model_parameters['background'] = background
        fit = model_flux(
            model_parameters, ifu.xpos_rel, ifu.ypos_rel, ifu.lambda_range, 
            model_name)
        rms = np.sqrt(np.nansum((ifu.data - fit)**2, axis=0))
        rms_smoothed = median_filter(rms, 41)
        bad = np.where(rms >= (clip * rms_smoothed))[0]
        for bad_pixel in bad:
            keep_bool = np.isfinite(ifu.data[:, bad_pixel])
            keep = np.where(keep_bool)[0]
            while rms[bad_pixel] >= (clip * rms_smoothed[bad_pixel]):
                # Clip out the fibre with the largest positive deviation
                worst_pixel = np.argmax(ifu.data[keep, bad_pixel] - 
                                        fit[keep, bad_pixel])
                keep_bool[keep[worst_pixel]] = False
                keep = np.where(keep_bool)[0]
                # Re-fit the model to the data
                flux[bad_pixel], background[bad_pixel] = extract_flux_slice(
                    ifu.data[keep, bad_pixel], ifu.var[keep, bad_pixel],
                    ifu.xpos_rel[keep], ifu.ypos_rel[keep], 
                    psf_parameters_array[bad_pixel])
                # Re-calculate the deviation from the model
                model_parameters['flux'] = np.array(
                    [flux[bad_pixel]])
                model_parameters['background'] = np.array(
                    [background[bad_pixel]])
                fit_pixel = model_flux(
                    model_parameters, ifu.xpos_rel, ifu.ypos_rel, 
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
        background[interp_over] = np.interp(
            interp_over, interp_from, background[interp_from])
    return flux, background

def extract_flux_slice(data, variance, xpos, ypos, psf_parameters_slice):
    """Extract the flux from a single wavelength slice."""
    good_data = np.where(np.isfinite(data))[0]
    # Require at least half the fibres to perform a fit
    if len(good_data) > 30:
        data = data[good_data]
        variance = variance[good_data]
        xpos = xpos[good_data]
        ypos = ypos[good_data]
        model = moffat_normalised(psf_parameters_slice, xpos, ypos)
        args = (model, data, variance)
        # Initial guess for flux and background
        guess = [np.sum(data), 0.0]
        flux_slice, background_slice = leastsq(
            residual_slice, guess, args=args)[0]
    else:
        flux_slice = np.nan
        background_slice = np.nan
    return flux_slice, background_slice

def residual_slice(flux_background, model, data, variance):
    """Residual of the model flux in a single wavelength slice."""
    flux, background = flux_background
    # For now, ignoring the variance - it has too many 2dfdr-induced mistakes
    variance = data.copy()
    cutoff = 0.05 * data.max()
    variance[data < cutoff] = cutoff
    return ((background + flux * model) - data) / np.sqrt(variance)
    #return ((background + flux * model) - data) / np.sqrt(variance)

def save_extracted_flux(path, observed_flux, observed_background,
                        star_match, psf_parameters, model_name,
                        good_psf):
    """Add the extracted flux to the specified FITS file."""
    # Turn the data into a single array
    data = np.vstack((observed_flux, observed_background))
    # Make the new HDU
    hdu_name = 'FLUX_CALIBRATION'
    new_hdu = pf.ImageHDU(data, name=hdu_name)
    # Add info to the header
    header_item_list = [
        ('PROBENUM', star_match['probenum'], 'Number of the probe containing '
                                             'the star'),
        ('STDNAME', star_match['name'], 'Name of standard star'),
        ('HGFLXCAL', HG_CHANGESET, 'Hg changeset ID for fluxcal code'),
        ('MODEL', model_name, 'Name of model used in PSF fit'),
        ('GOODPSF', good_psf, 'Whether the PSF fit has good parameters')]
    if 'path' in star_match:
        header_item_list.append(
            ('STDFILE', star_match['path'], 'Filename of standard spectrum'))
    if 'separation' in star_match:
        header_item_list.append(
            ('STDOFF', star_match['separation'], 'Offset (arcsec) to standard '
                                                 'star coordinates'))
    for key, value in psf_parameters.items():
        header_item_list.append((header_translate(key), value, 
                                 'PSF model parameter'))
    for key, value, comment in header_item_list:
        try:
            new_hdu.header[key] = (value, comment)
        except ValueError:
            # Probably tried to save an array, just ditch it
            pass
    # Update the file
    hdulist = pf.open(path, 'update', do_not_scale_image_data=True)
    # Check if there's already an extracted flux, and delete if so
    try:
        existing_index = hdulist.index_of(hdu_name)
    except KeyError:
        pass
    else:
        del hdulist[existing_index]
    hdulist.append(new_hdu)
    hdulist.close()
    del hdulist
    return

def header_translate(key):
    """Translate parameter names to be suitable for headers."""
    name_dict = {'xcen_ref': 'XCENREF',
                 'ycen_ref': 'YCENREF',
                 'zenith_direction': 'ZENDIR',
                 'zenith_distance': 'ZENDIST',
                 'flux': 'FLUX',
                 'beta': 'BETA',
                 'background': 'BCKGRND',
                 'alpha_ref': 'ALPHAREF'}
    try:
        header_name = name_dict[key]
    except KeyError:
        header_name = key[:8]
    return header_name

def header_translate_inverse(header_name):
    """Translate parameter names back from headers."""
    name_dict = {'XCENREF': 'xcen_ref',
                 'YCENREF': 'ycen_ref',
                 'ZENDIR': 'zenith_direction',
                 'ZENDIST': 'zenith_distance',
                 'FLUX': 'flux',
                 'BETA': 'beta',
                 'BCKGRND': 'background',
                 'ALPHAREF': 'alpha_ref'}
    return name_dict[header_name]

def save_transfer_function(path, transfer_function):
    """Add the transfer function to a pre-existing FLUX_CALIBRATION HDU."""
    # Open the file to update
    hdulist = pf.open(path, 'update', do_not_scale_image_data=True)
    hdu = hdulist['FLUX_CALIBRATION']
    data = hdu.data
    if len(data) == 2:
        # No previous transfer function saved; append it to the data
        data = np.vstack((data, transfer_function))
    elif len(data) == 3:
        # Previous transfer function to overwrite
        data[2, :] = transfer_function
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
               observed_wavelength, smooth=True):
    """Return the ratio of two spectra, after rebinning."""
    # Rebin the observed spectrum onto the (coarser) scale of the standard
    observed_flux_rebinned = rebin_flux(
        standard_wavelength, observed_wavelength, observed_flux)
    ratio = standard_flux / observed_flux_rebinned
    if smooth:
        ratio = fit_chebyshev(standard_wavelength, ratio)
        # ratio = smooth_ratio(ratio)
    # Put the ratio back onto the observed wavelength scale
    ratio = np.interp(observed_wavelength, standard_wavelength, ratio)
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

def fit_chebyshev(wavelength, ratio, deg=7):
    """Fit a Chebyshev polynomial, and return the fit."""
    good = np.isfinite(ratio) & ~(in_telluric_band(wavelength))
    coefficients = np.polynomial.chebyshev.chebfit(
        wavelength[good], ratio[good], deg)
    fit = np.polynomial.chebyshev.chebval(wavelength, coefficients)
    return fit

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
            rebinneddata[ i ] = np.sum( weights * originalflux[ :, indices ] )
            rebinnedweight[i ]= np.sum( weights * originalweight[:,indices ] )

    # now go back from total flux in each bin to flux per unit wavelength
    rebinneddata = rebinneddata / rebinnedweight 

    return rebinneddata

def remove_atmosphere(ifu):
    """Remove atmospheric extinction (not tellurics) from ifu data."""
    # Read extinction curve (in magnitudes)
    wavelength_extinction, extinction_mags = read_atmospheric_extinction()
    # Interpolate onto the data wavelength grid
    extinction_mags = np.interp(ifu.lambda_range, wavelength_extinction, 
                                extinction_mags)
    # Scale for observed airmass
    airmass = calculate_airmass(ifu.zdstart, ifu.zdend)
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
    airmass = calculate_airmass(header['ZDSTART'], header['ZDEND'])
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
    for entry in open( sso_extinction_table, 'r' ).xreadlines() :
        line = entry.rstrip( '\n' )
        if not line.count( '*' ) and not line.count( '=' ):
            values = line.split()
            wl.append(  values[0] )
            ext.append( values[1] )    
    wl = np.array( wl ).astype( 'f' )
    ext= np.array( ext ).astype( 'f' )   
    return wl, ext

def combine_transfer_functions(path_list, path_out):
    """Read a set of transfer functions, combine them, and save to file."""
    # Make an empty array to hold all the individual transfer functions
    n_file = len(path_list)
    n_pixel = pf.getval(path_list[0], 'NAXIS1')
    tf_array = np.zeros((n_file, n_pixel))
    # Read the individual transfer functions
    for index, path in enumerate(path_list):
        tf_array[index, :] = pf.getdata(path, 'FLUX_CALIBRATION')[-1, :]
    # Make sure the overall scaling for each TF matches the others
    scale = tf_array[:, n_pixel//2].copy()
    scale /= np.median(scale)
    tf_array = (tf_array.T / scale).T
    # Combine them.
    # For now, just take the mean. Maybe implement Ned's weighted combination
    # later, preferably when proper variance propagation is in place.
    tf_combined = nanmean(tf_array, axis=0)
    save_combined_transfer_function(path_out, tf_combined, path_list)
    return

def save_combined_transfer_function(path_out, tf_combined, path_list):
    """Write the combined transfer function (and the individuals to file."""
    # Put the combined transfer function in the primary HDU
    primary_hdu = pf.PrimaryHDU(tf_combined)
    primary_hdu.header['HGFLXCAL'] = (HG_CHANGESET, 
                                      'Hg changeset ID for fluxcal code')
    # Make an HDU list, which will also contain all the individual functions
    hdulist = pf.HDUList([primary_hdu])
    for index, path in enumerate(path_list):
        hdulist_tmp = pf.open(path)
        hdu = hdulist_tmp['FLUX_CALIBRATION']
        hdu.header['EXTVER'] = index + 1
        hdu.header['ORIGFILE'] = (path, 'Originating file for this HDU')
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
    xfibre = ifu.xpos_rel
    yfibre = ifu.ypos_rel
    wavelength = ifu.lambda_range
    model = model_flux(psf_parameters, xfibre, yfibre, wavelength, model_name)
    return model

def read_model_parameters(hdu):
    """Return the PSF model parameters in a header, with the model name."""
    psf_parameters = {}
    model_name = None
    for key, value in hdu.header.items():
        if key == 'MODEL':
            model_name = value
        else:
            try:
                psf_parameters[header_translate_inverse(key)] = value
            except KeyError:
                continue
    psf_parameters['flux'] = hdu.data[0, :]
    psf_parameters['background'] = hdu.data[1, :]
    return psf_parameters, model_name

def insert_fixed_parameters(parameters_dict, fixed_parameters):
    """Insert the fixed parameters into the parameters_dict."""
    if fixed_parameters is not None:
        for key, value in fixed_parameters.items():
            parameters_dict[key] = value
    return parameters_dict

def set_fixed_parameters(path_list, model_name):
    """Return fixed values for certain parameters."""
    if model_name == 'ref_centre_alpha_dist_circ':
        header = pf.getheader(path_list[0])
        zenith_direction = np.deg2rad(parallactic_angle(
            header['HASTART'], header['ZDSTART'], header['LAT_OBS']))
        fixed_parameters = {'zenith_direction': zenith_direction}
    else:
        fixed_parameters = {}
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
    hdulist.writeto(path_out)
    return




