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
as free parameters too. Note, however, that the atmospheric parameters
are completely degenerate with each other and with ZD.
"""

import os

import numpy as np
from scipy.optimize import leastsq
from scipy.ndimage.filters import median_filter
from scipy.stats.stats import nanmean

from astropy import coordinates as coord
from astropy import units
from astropy.io import fits as pf
from astropy import __version__ as astropy_version

from .. import utils
from ..utils.ifu import IFU

HG_CHANGESET = utils.hg_changeset(__file__)

STANDARD_CATALOGUES = ('./standards/ESO/ESOstandards.dat',
                       './standards/Bessell/Bessellstandards.dat')

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

def read_chunked_data(path_list, probenum, n_drop=None, n_chunk=None):
    """Read flux from a list of files, chunk it and combine."""
    if isinstance(path_list, str):
        path_list = [path_list]
    for i_file, path in enumerate(path_list):
        ifu = IFU(path, probenum, flag_name=False)
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
    chunked_data = {'data': data,
                    'variance': variance,
                    'wavelength': wavelength,
                    'xfibre': xfibre,
                    'yfibre': yfibre}
    return chunked_data

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
        if len(parameters_dict['flux']) == len(parameters_array):
            parameters_array['flux'] = parameters_dict['flux']
        if len(parameters_dict['background']) == len(parameters_array):
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
        if len(parameters_dict['flux']) == len(parameters_array):
            parameters_array['flux'] = parameters_dict['flux']
        if len(parameters_dict['background']) == len(parameters_array):
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

def derive_transfer_function(path_list, max_sep_arcsec=30.0,
                             catalogues=STANDARD_CATALOGUES,
                             model_name='ref_centre_alpha_angle_circ'):
    """Derive transfer function and save it in each FITS file."""
    # First work out which star we're looking at, and which hexabundle it's in
    star_match = match_standard_star(
        path_list[0], max_sep_arcsec=max_sep_arcsec, catalogues=catalogues)
    if star_match is None:
        raise ValueError('No standard star found in the data.')
    #################standard_data = read_standard_data(star_match)
    # Read the observed data, in chunks
    chunked_data = read_chunked_data(path_list, star_match['probenum'])
    # Fit the PSF
    psf_parameters = fit_model_flux(
        chunked_data['data'], 
        chunked_data['variance'],
        chunked_data['xfibre'],
        chunked_data['yfibre'],
        chunked_data['wavelength'],
        model_name)
    for path in path_list:
        ifu = IFU(path, star_match['probenum'], flag_name=False)
        observed_flux, observed_background = extract_total_flux(
            ifu, psf_parameters, model_name)
        save_extracted_flux(path, observed_flux, observed_background,
                            star_match)
    return

def match_standard_star(filename, max_sep_arcsec=30.0, 
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

def match_star_coordinates(ra, dec, max_sep_arcsec=30.0, 
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

def extract_total_flux(ifu, psf_parameters, model_name):
    """Extract the total flux, including light between fibres."""
    psf_parameters_array = parameters_dict_to_array(
        psf_parameters, ifu.lambda_range, model_name)
    n_pixel = len(psf_parameters_array)
    flux = np.zeros(n_pixel)
    background = np.zeros(n_pixel)
    for index, psf_parameters_slice in enumerate(psf_parameters_array):
        data = ifu.data[:, index]
        variance = ifu.var[:, index]
        xpos = ifu.xpos_rel
        ypos = ifu.ypos_rel
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
        flux[index] = flux_slice
        background[index] = background_slice
    return flux, background

def residual_slice(flux_background, model, data, variance):
    """Residual of the model flux in a single wavelength slice."""
    flux, background = flux_background
    # For now, ignoring the variance - it has too many 2dfdr-induced mistakes
    return ((background + flux * model) - data)
    #return ((background + flux * model) - data) / np.sqrt(variance)

def save_extracted_flux(path, observed_flux, observed_background,
                        star_match):
    """Add the extracted flux to the specified FITS file."""
    # Turn the data into a single array
    data = np.vstack((observed_flux, observed_background))
    # Make the new HDU
    hdu_name = 'EXTRACTED_FLUX'
    new_hdu = pf.ImageHDU(data, name=hdu_name)
    # Add info to the header
    header_item_list = [
        ('PROBENUM', star_match['probenum'], 'Number of the probe containing '
                                             'the star'),
        ('STDNAME', star_match['name'], 'Name of standard star'),
        ('STDFILE', star_match['path'], 'Filename of standard spectrum'),
        ('STDOFF', star_match['separation'], 'Offset (arcsec) to standard '
                                             'star coordinates'),
        ('HGFLXCAL', HG_CHANGESET, 'Hg changeset ID for fluxcal code')]
    for key, value, comment in header_item_list:
        new_hdu.header[key] = (value, comment)
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





