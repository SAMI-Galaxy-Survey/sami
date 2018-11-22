"""
Modul for cubing with Gaussian Process Prior
Functions that are specific for the SAMI survey data, e.g. read of fiber data, are labeled 
with prename 'sami_'
The main class is DataFuse3D which loops over chunks in wavelengths and fuse otical fibre data 
given PSF parameters, fibre positions, and data variance 
21 March 2018, Seb Haan
"""

from __future__ import print_function
import sys
import os
import time
import numpy as np
import matplotlib.pylab as plt
from scipy import fftpack, interpolate, linalg, signal, optimize
from scipy.ndimage.interpolation import shift as imshift
from astropy.io import fits as pf
import astropy.wcs as pw
from astropy.modeling import models as astromodels
from astropy.modeling import fitting as astrofitting
from glob import glob
# Sami specific modules:
import sami.utils as utils
from sami.general import wcs as wcs
from sami.utils.mc_adr import adr_r, DARCorrector, parallactic_angle, zenith_distance
# The new cubesolve modul:
from sami.gpcubing import gpcubesolve_r2 as cs
from sami.gpcubing.settings_sami import _Nfib, _Rfib_arcsec, _plate_scale, _fov_arcsec

import code

cubing_method = 'GP_prior'
wavelength_ref = 5000.0

np.random.seed(3121)

class CubeError(Exception):
    """
    General exception class for stuff in here
    """

    def __init__(self, msg):
        self.msg = msg

    def __unicode__(self):
        return self.msg


class VerboseMessager(object):
    """
    Class for flushed verbose debugging
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def __call__(self, msg):
        if self.verbose:
            print(msg)
            sys.stdout.flush()


def profile_time(func, *args, **kwargs):
    """
    Measures a function's execution time.
    :param func: function to time
    :param args: ordered arguments to pass to func
    :param kwargs: keyword arguments to pass to func
    """
    t0 = time.time()
    result = func(*args, **kwargs)
    t1 = time.time()
   # print "time to run:  {}:  {:.3f} sec".format(func.__name__, t1 - t0)
    return result

def rebin(A, rb):
    """
    Rebins each axis of an array by a factor rb.  Right now rb must
    evenly divide each axis of the array, though in future more general
    rebinning patterns may be supported.
    :param A: array to rebin
    :param rb: rebinning factor (positive integer)
    """
    # Sanity checks
    if not isinstance(rb, int) or rb <= 0:
        raise CubeError("rb must be positive integer in rebin()")
    for i, dim in enumerate(A.shape):
        if dim % rb != 0:
            raise CubeError("rb doesn't evenly divide axis {} of array"
                            .format(i))
        new_shape = list(A.shape[:i])
        new_shape.extend([dim/rb, rb])
        new_shape.extend(list(A.shape[i+1:]))
        A = A.reshape(new_shape).sum(axis=i+1)
    return A

def gaussian_psf(dx, dy, sigma):
    """
    Round Gaussian PSF model, normalized to unit flux.
    :param dx: distance from PSF center along x-axis, in pixels
    :param dy: distance from PSF center along y-axis, in pixels
    :param sigma: standard deviation of Gaussian in pixels
    """
    return np.exp(-0.5*(dx**2 + dy**2)/sigma**2)/(2*np.pi*sigma)

def moffat_psf(dx, dy, alpha, beta):
    """
    Round Moffat PSF model, normalized to unit flux.
    :param dx: distance from PSF center along x-axis, in pixels
    :param dy: distance from PSF center along y-axis, in pixels
    :param alpha: core width parameter of Moffat in pixels
    :param beta: wings slope parameter of Moffat (dimensionless)
    """
    norm = (beta - 1.0)/(np.pi * alpha**2)
    return norm * (1.0 + (dx**2 + dy**2)/alpha**2)**(-beta)

def moffat_ellip_psf(dx, dy, alpha_x, alpha_y, rho, beta):
    """
    Round Moffat PSF model, normalized to unit flux, not used currently.
    :param dx: distance from PSF center along x-axis, in pixels
    :param dy: distance from PSF center along y-axis, in pixels
    :param rho: ellpticity?
    :param alpha_x: core width parameter of Moffat in x-direction in pixels
    :param alpha_y: core width parameter of Moffat in y-direction in pixels
    :param beta: wings slope parameter of Moffat (dimensionless)
    """
    norm = (beta - 1) / (np.pi * alpha_x * alpha_y * np.sqrt(1 - rho**2))
    return norm * (1.0 + ((dx/alpha_x)**2 + (dy/alpha_y)**2 - 2*rho*dx/alpha_x*dy/alpha_y) / (1 - rho**2))**(-beta)

def moffat2fwhm(alpha, beta):
    """ Calculation of FWHM from Moffat Parameters
    param alpha: width parameter of Moffat
    param beta: slope parameter of Moffat, same shape as alpha
    """
    return 2* alpha * np.sqrt(np.power(2.,1/beta) - 1)

def sig2alpha(beta):
    """
    Given the beta parameter for a Moffat profile, solves for the ratio
    of alpha to the standard deviation of a Gaussian with the same FWHM.
    """
    return np.sqrt(2*np.log(2)/(2**(1.0/beta) - 1.0))


def fit_source_fwhm(data, cen, func = 'gaussian'):
    """
    Fits the FWHM of a round source with its peak at zero.
    :param data: numpy 2D array with data
    :param center: initial guesses for center of peak given by (x,y) index of data array
    :param func:  string variable for 2D model: either 'moffat' or 'gaussian'
    """
    cen = np.asarray(cen)
    y,x = np.mgrid[:data.shape[0], :data.shape[1]]
    fit_p = astrofitting.LevMarLSQFitter()
    amp0 = data[cen[0], cen[1]] # initial estimates for center
    if func == 'gaussian':
        p_init = astromodels.Gaussian2D(amplitude = amp0, x_mean = cen[0], y_mean = cen[1], x_stddev = 2., y_stddev = 2., theta = 0.)
        params = fit_p(p_init, x, y, data)
        fwhm = 2.3548 * np.mean([params.x_stddev.value, params.y_stddev.value])
        if fit_p.fit_info['param_cov'] is not None:
            error =  np.sqrt(np.diagonal(fit_p.fit_info['param_cov']))
            fwhm_err = 2.3548 * np.mean([error[3], error[4]])
        else: fwhm_err = np.nan 
        xcen= params.x_mean.value
        ycen= params.y_mean.value

    elif func == 'moffat':
        p_init = astromodels.Moffat2D(amplitude = amp0, x_0 = cen[0], y_0 = cen[1], gamma = 2, alpha = 3.7)
        params = fit_p(p_init, x, y, data)
        width =  params.gamma.value 
        slope = params.alpha.value 
        if slope > 0.: 
            fwhm = 2* width * np.sqrt(np.power(2.,1/slope) - 1)
        else:
            fwhm = np.nan
        if fit_p.fit_info['param_cov'] is not None:
            error =  np.sqrt(np.diagonal(fit_p.fit_info['param_cov']))
            ewidth, eslope = error[3], error[4]
            fwhm_err = fwhm / width * ewidth + 2 * width * eslope* 0.346574 * 2**(1/slope)/(np.sqrt(2**(2/slope)-1) * slope**2)
        else: fwhm_err = np.nan 
        xcen= params.x_0.value
        ycen= params.y_0.value

    else: 
        raise ValueError("Model name not possible, use for func either 'gaussian' or 'moffat'!")   
    
    return fwhm, fwhm_err, xcen, ycen


def sami_header_translate_inverse(header_name):
    """Translate parameter names back from headers."""
    name_dict = {'XCENREF': 'xcen_ref',
                 'YCENREF': 'ycen_ref',
                 'ZENDIR': 'zenith_direction',
                 'ZENDIST': 'zenith_distance',
                 'FLUX': 'flux',
                 'BETA': 'beta',
                 'BCKGRND': 'background',
                 'ALPHAREF': 'alpha_ref',
                 'TEMP': 'temperature',
                 'PRESSURE': 'pressure',
                 'VAPPRESS': 'vapour_pressure'}
    return name_dict[header_name]

def sami_read_model_parameters(hdu):
    """Return the PSF model parameters in a header, with the model name."""
    psf_parameters = {}
    model_name = None
    for key, value in hdu.header.items():
        if key == 'MODEL':
            model_name = value
        else:
            try:
                psf_parameters[sami_header_translate_inverse(key)] = value
            except KeyError:
                continue
    psf_parameters['flux'] = hdu.data[0, :]
    psf_parameters['background'] = hdu.data[1, :]
    return psf_parameters, model_name

def sami_alpha(wavelength, alpha_ref, wavelength_ref):
    """Return alpha at the specified wavelength(s)."""
    return alpha_ref * ((wavelength / wavelength_ref)**(-0.2))


def sami_dar(wavelength, ref_wavelength, zenith_distance, temperature=None, pressure=None,
        vapour_pressure=None):
    """Return the DAR offset in arcseconds at the specified wavelength(s)."""
    return (adr_r(wavelength, np.rad2deg(zenith_distance), 
                  air_pres=pressure, temperature=temperature, 
                  water_pres=vapour_pressure) - 
            adr_r(ref_wavelength, np.rad2deg(zenith_distance), 
                  air_pres=pressure, temperature=temperature, 
                  water_pres=vapour_pressure))

def sami_get_object_names(infile):
    """Get the object names observed in the file infile."""

    # Open up the file and pull out list of observed objects.
    try:
        table=pf.getdata(infile, 'FIBRES_IFU')
    except KeyError:
        table=pf.getdata(infile, 'MORE.FIBRES_IFU')
    names=table.field('NAME')

    # Find the set of unique values in names
    names_unique=list(set(names))

    # Pick out the object names, rejecting SKY and empty strings
    object_names_unique = [s for s in names_unique if ((s.startswith('SKY')==False)
                            and (s.startswith('Sky')==False)) and len(s)>0]

    return object_names_unique

def sami_get_probe_all(files, name, verbose=True):
    """Obtain a list of probe names, and write it to the cube header."""

    probes = [sami_get_probe_single(fl, name, verbose=verbose) for fl in files]
    probes = np.unique(probes)
    probes_string = np.array2string(probes,separator=',')[1:-1]

    return probes_string

def sami_get_probe_single(infile, object_name, verbose=True):
    """ This should read in the RSS files and return the probe number the object was observed in"""

    # First find the IFU the object was returned in
    hdulist=pf.open(infile)
    fibre_table=hdulist['FIBRES_IFU'].data
    
    mask_name=fibre_table.field('NAME')==object_name

    table_short=fibre_table[mask_name]

    # Now find the ifu the galaxy was observed with in this file
    ifu_array=table_short.field('PROBENUM')

    # The ifu
    ifu=ifu_array[0]

    if verbose==True:
        print("Object", object_name, "was observed in IFU", ifu, "in file", infile)

    hdulist.close()

    # Return the probe number
    return ifu


def sami_simulate_data(filename_out, s2n = 10, alpha_ref = 2, beta = 4.7, psf_sigma = 0.2, dar_amp = 1., sim_option = 1):
    """ 
    Creates a simulated data cube with fibre fluxes and variances from a model sky scene, either unform source or point source 
    The Number of exposures, fibres, wavelength range, and offset position reflect the typical SAMI data
    The simulation takes into account multiple exposures with different atmospheric PSF and DAR.
    :param filename_out: string, path + filename of output file
    :param s2n: Signal to noise of data, default 10
    :param alpha_ref: Moffat alpha parameter at reference wavelength of 5000Angstrom, default 2. 
                    Note that alpha will change as function of wavelength.
    :param beta: Moffat power index parameter, default 4.7
    :param psf_sigma: PSF width varies given normal distribution (sigma) across exposures (7 for sami), default = 0.2
    :param dar_sigma: Variation of spatial differential athmospheric refraction (DAR) along wavelength axis
    :param sim_option: choose sky model: 1 - point spread function
                                         2 - uniform point source with 2 arcsec radius
    """
    x = np.arange(2000)+1
    L0 = 6209.78171133
    cdelt1 = 0.569362891961
    zenith_distance = 45.
    wavelength = L0 + x*cdelt1
    alpha = sami_alpha(wavelength, alpha_ref, wavelength_ref)
    dar = sami_dar(wavelength, ref_wavelength, zenith_distance = zenith_distance)
    
    flux_array = np.zeros(7, 61, 2000)
    var_array = np.zeros_like(flux_array)

    seeing_pixel_std = seeing_fwhm_arcsec * fwhm_2_std / pixscale
    # Use RBFs for both kernel and PSF, to make computations easy.
    # Remember that the response matrix needs the convolution kernel
    # that takes the (finite-resolution) inferred scene into sky PSF.
    if seeing_pixel_std < kernel_scale:
        raise CubeError("seeing_pixel_std = {} < kernel_scale = {}"
                        .format(seeing_pixel_std, kernel_scale))
    conv_std = np.sqrt(seeing_pixel_std**2 - kernel_scale**2)
    def conv_kernel(dx, dy):
        dxp, dyp = dx/conv_std, dy/conv_std
        return np.exp(-0.5*(dxp**2 + dyp**2))/(2*np.pi*conv_std)
    response = ResponseMatrix(conv_kernel, Lpix, pixscale,_Nexp=_Nexp)
    view = SliceView(response)
    # Generate a fake scene
    def scene_psf(dx, dy):
        dxp, dyp = dx/seeing_pixel_std, dy/seeing_pixel_std
        return np.exp(-0.5*(dxp**2 + dyp**2))/(2*np.pi*seeing_pixel_std)
    kx, ky = np.meshgrid(np.arange(Lpix) + 0.5, np.arange(Lpix) + 0.5)
    view.scene += scene_psf(kx - 0.5*Lpix, ky - 0.5*Lpix)



def readfits_samidata(fitsfile, identifier, flag_name = True):
    """
    Manual reading in SAMI fits files to extract calibrated fiber data, psf_parameters, positions 
    and their corresponding uncertanties for all 13 hexabundles (61 fibres + 2 psf)
    Selects only particular object or probe with identifier.
    Only still included for references.
    :param fitsfile: Name of fitsfile
    :param identifier: Object Identifier name
    :param flag_name: if True takes object name, if  not true selecting on probe (IFU) number. Default True
    """ 
    hdulist = pf.open(fitsfile)
    primary_header = hdulist['PRIMARY'].header
    fibre_table_header = hdulist['FIBRES_IFU'].header
    fibre_table = hdulist['FIBRES_IFU'].data
    naxis1 = primary_header['NAXIS1']
    offsets_table = hdulist['ALIGNMENT'].data
    # Field centre values (not bundle values!)
    meanra = primary_header['MEANRA']
    meandec = primary_header['MEANDEC']

    # Data and Variance
    data_in = hdulist[0].data
    variance_in = hdulist['VARIANCE'].data
    
    # Wavelength range
    x=np.arange(naxis1)+1
    crval1 = primary_header['CRVAL1']
    crpix1 = primary_header['CRPIX1']
    cdelt1 = primary_header['CDELT1']
    L0=crval1-crpix1*cdelt1
    lambda_range = L0+x*cdelt1
    wavelength = lambda_range
    print('L0', L0)
    print('cdelt1', cdelt1)
    
    # PSF parameter: 
    hdu = hdulist['FLUX_CALIBRATION']
    psf_parameters, model_name = sami_read_model_parameters(hdu)
    psf_alpha_ref = psf_parameters['alpha_ref']
    psf_beta = psf_parameters['beta']
    # Calculate PSF as function of wavelength
    psf_alpha = sami_alpha(wavelength, psf_alpha_ref, wavelength_ref)

    #DAR correction:
    zendist = hdu.header['ZENDIST']
    pressure = hdu.header['PRESSURE']
    vappress  =  hdu.header['VAPPRESS']
    temp = hdu.header['TEMP']
    print(hdu.header)
    dar_cor = sami_dar(wavelength, wavelength_ref, zendist, temperature=temp, pressure=pressure, vapour_pressure=vappress)
    

    if flag_name == True:
        # selecting on object name
        name = identifier
        msk0 = fibre_table.field('NAME') == identifier # First mask on name.
        table_find = fibre_table[msk0] 
        # Find the IFU name from the find table.
        ifu = np.unique(table_find.field('PROBENUM'))[0]
        
    else:
        # Flag is not true so we're selecting on probe (IFU) number.
        ifu=identifier
        msk0=fibre_table.field('PROBENUM') == identifier # First mask on probe number.
        table_find=fibre_table[msk0]

        # Pick out the place in the table with object names, rejecting SKY and empty strings.
        object_names_nonsky = [s for s in table_find.field('NAME') if s.startswith('SKY')==False and s.startswith('Sky')==False and len(s)>0]
        #print np.shape(object_names_nonsky)
        name=list(set(object_names_nonsky))[0]
        
    mask=np.logical_and(fibre_table.field('PROBENUM')==ifu, fibre_table.field('NAME')==name)
    table_new=fibre_table[mask]

    # indices of the corresponding spectra (SPEC_ID counts from 1, image counts from 0)
    ind=table_new.field('SPEC_ID')-1    
    exptime = primary_header['EXPOSED']
    data=data_in[ind,:]  /exptime  
    variance=variance_in[ind,:] /(exptime**2)

    # Calculate position of Fibers with DAR corrections
    offsets_new = offsets_table[offsets_table.field('PROBENUM')==ifu]
    xpos_rel = table_new.field('XPOS') 
    ypos_rel = table_new.field('YPOS') 
    xarc2micron = xpos_rel.mean()/table_new.field('FIBPOS_X').mean()
    yarc2micron = ypos_rel.mean()/table_new.field('FIBPOS_Y').mean()
    x_shift_full = np.median(offsets_new['X_REFMED'] + offsets_table['X_SHIFT'])
    y_shift_full = np.median(offsets_new['Y_REFMED'] - offsets_table['Y_SHIFT'])
    xfibre=(table_new.field('XPOS') - x_shift_full * _plate_scale / 1000.)
    yfibre=(table_new.field('YPOS') - y_shift_full * _plate_scale / 1000.)
           
    nfiber = table_new.field('FIBNUM')
    # Probe Name
    hexabundle_name = table_new.field('PROBENAME')
    # Fibre designation.
    fib_type = table_new.field('TYPE')
    name_tab = table_new.field('NAME')
   # print('FIBPOS_X, X-SHIFT:', np.mean(table_new.field('FIBPOS_X')), np.mean(offsets_new.field('X_SHIFT')))
   # print('FIBPOS_Y, Y-SHIFT:', np.mean(table_new.field('FIBPOS_Y')), np.mean(offsets_new.field('X_SHIFT')))
   # print('arc2micron = ', arc2micron)
  #  print('offsets x:', np.mean(offsets_table['X_REFMED']), ' pm ', np.std(offsets_table['X_REFMED']))
   # print('offsets y:', np.mean(offsets_table['Y_REFMED']), ' pm ', np.std(offsets_table['Y_REFMED'])) 
    return data, variance, psf_alpha, psf_beta,  xfibre, yfibre, dar_cor # , xpos , ypos  #, name_tag


def sami_listnames(fitsfile):
    """ Returns list with all target names excluding the Sky observations
    :param fitsfile: Path to fitsfile
    """
    hdulist = pf.open(fitsfile)
    fibre_table = hdulist['FIBRES_IFU'].data
    name_tab = fibre_table.field('NAME')
    object_names_nonsky = [s for s in fibre_table.field('NAME') if s.startswith('SKY')==False and s.startswith('Sky')==False and len(s)>0]
    return list(set(object_names_nonsky))


def sami_dar_correct(ifu_list, xfibre_all, yfibre_all, method='simple'):
    """Update the fiber positions as a function of wavelength to reflect DAR correction.
     Similar to pipeline function dar_correct
    :param ifu_list: list of names
    :param xfibre_all: array with fibre x positions
    :param yfibre_all: array with fibre y positions
    :param method: specify method for DARCorrector, default = 'simple'
    """

    n_obs, n_fibres, n_slices = xfibre_all.shape

    # Set up the differential atmospheric refraction correction. Each frame
    # requires it's own correction, so we create a list of DARCorrectors.
    dar_correctors = []
                      
    for obs in ifu_list:
        darcorr = DARCorrector(method=method)
    
        ha_offset = obs.ra - obs.meanra  # The offset from the HA of the field centre

        ha_start = obs.primary_header['HASTART'] + ha_offset
        # The header includes HAEND, but this goes very wrong if the telescope
        # slews during readout. The equation below goes somewhat wrong if the
        # observation was paused, but somewhat wrong is better than very wrong.
        ha_end = ha_start + (obs.exptime / 3600.0) * 15.0
    
        if hasattr(obs, 'atmosphere'):
            # Take the atmospheric parameters from the file, as measured
            # during telluric correction
            darcorr.temperature = obs.atmosphere['temperature']
            darcorr.air_pres = obs.atmosphere['pressure']
            darcorr.water_pres = obs.atmosphere['vapour_pressure']
            darcorr.zenith_distance = np.rad2deg(obs.atmosphere['zenith_distance'])
        else:
            # Get the atmospheric parameters from the fibre table header
            darcorr.temperature = obs.fibre_table_header['ATMTEMP'] 
            darcorr.air_pres = obs.fibre_table_header['ATMPRES'] * millibar_to_mmHg
            #                     (factor converts from millibars to mm of Hg)
            darcorr.water_pres = \
                utils.saturated_partial_pressure_water(darcorr.air_pres, darcorr.temperature) * \
                obs.fibre_table_header['ATMRHUM']
            darcorr.zenith_distance = \
                integrate.quad(lambda ha: zenith_distance(obs.dec, ha),
                               ha_start, ha_end)[0] / (ha_end - ha_start)

        darcorr.hour_angle = 0.5 * (ha_start + ha_end)
    
        darcorr.declination = obs.dec

        dar_correctors.append(darcorr)

        del darcorr # Cleanup since this is meaningless outside the loop.


    wavelength_array = ifu_list[0].lambda_range
    xdar_all = np.zeros_like(xfibre_all)
    ydar_all = np.zeros_like(yfibre_all)
    
    # Iterate over wavelength slices
    for l in range(n_slices):
        
        # Iterate over observations
        for i_obs in range(n_obs):
            # Determine differential atmospheric refraction correction for this slice
            dar_correctors[i_obs].update_for_wavelength(wavelength_array[l])
            
            # Parallactic angle is direction to zenith measured north through east.
            # Must move light away from the zenith to correct for DAR.
            dar_x = dar_correctors[i_obs].dar_east * 1000.0 / _plate_scale 
            dar_y = dar_correctors[i_obs].dar_north * 1000.0 / _plate_scale 

            #xfibre_all[i_obs,:,l] = (xfibre_all[i_obs,:,l] + dar_x) 
            #yfibre_all[i_obs,:,l] = (yfibre_all[i_obs,:,l] + dar_y) 
            xdar_all[i_obs,:,l] = dar_x
            ydar_all[i_obs,:,l] = dar_y

    return xfibre_all, yfibre_all, xdar_all * _plate_scale / 1000., ydar_all * _plate_scale / 1000.
            


def sami_combine(files, identifier, do_dar_correct = True, offsets='file', clip_throughput=True):
    """Calculation of fibre positions, fluxes, variance, and psf parameters.
    For reading in fibre specific data and DAR corrections the SAMI class utils.IFU is used.
    PSF parameters are loaded and calculated as function of wavelength
    :param path: Pathname
    :param files: list of file names
    :param identifer: object name
    :param do_dar_correct: include DAR correction, default = True
    :param offsets: take pre-measured offsets from file
    :param clip_throughput: Clip out fibres that have suspicious throughput values, default = True
    :return fibre positions, fluxes,  variance, psf parameters, and DAR corrections
    """
    n_files = len(files)
    ifu_list = []
    for filename in files:
        ifu_list.append(utils.IFU(filename, identifier, flag_name=True))

    diagnostic_info = {}

    # Number of observations/files
    n_obs = len(ifu_list)
    
    # The number of wavelength slices
    n_slices = np.shape(ifu_list[0].data)[1]
    
    # Number of fibres
    n_fibres = ifu_list[0].data.shape[0]
    
    # Empty lists for positions and data.
    xfibre_all = np.empty((n_obs, n_fibres))
    yfibre_all = np.empty((n_obs, n_fibres))
    data_all= np.empty((n_obs, n_fibres, n_slices))
    var_all = np.empty((n_obs, n_fibres, n_slices))
    psf_alpha_all = np.empty((n_obs, n_slices))
    psf_beta_all = np.empty((n_obs, n_slices))
    ifus_all= np.empty(n_obs)   

    for j in range(n_obs):

        # Get the data.
        galaxy_data=ifu_list[j]
        
        # Smooth the spectra and median.
        data_smoothed=np.zeros_like(galaxy_data.data)
        for p in range(np.shape(galaxy_data.data)[0]):
            data_smoothed[p,:]=utils.smooth(galaxy_data.data[p,:], 10) #default hanning

        naxis1 = galaxy_data.naxis1
        # READ-IN PSF PARAMETERS
        # Wavelength range
        wavelength = galaxy_data.lambda_range
        # PSF parameter: 
        psf_alpha_ref = galaxy_data.atmosphere['alpha_ref']
        psf_beta = galaxy_data.atmosphere['beta']
        # Calculate PSF as function of wavelength
        psf_alpha = sami_alpha(wavelength, psf_alpha_ref, wavelength_ref)

        # Collapse the smoothed data over a large wavelength range to get continuum data
        data_med=np.nanmedian(data_smoothed[:,300:1800], axis=1)

        # Pick out only good fibres (i.e. those allocated as P)
        # Assumes currently 61 fibres
        goodfibres=np.where(galaxy_data.fib_type=='P')
        x_good=galaxy_data.x_microns[goodfibres]
        y_good=galaxy_data.y_microns[goodfibres]
        data_good=data_med[goodfibres]
    
        # First try to get rid of the bias level in the data, by subtracting a median
        data_bias=np.median(data_good)
        if data_bias<0.0:
            data_good=data_good-data_bias
            
        # Mask out any "cold" spaxels - defined as negative, due to poor
        # throughtput calibration from CR taking out 5577.
        msk_notcold=np.where(data_good>0.0)
    
        # Apply the mask to x,y,data
        x_good=x_good[msk_notcold]
        y_good=y_good[msk_notcold]
        data_good=data_good[msk_notcold]

        # set bad fibres to zero flux and variance high, so statistically disregard later
        mask_fibre = np.zeros(galaxy_data.data.shape[0],dtype=bool)
        mask_fibre2 = np.zeros(galaxy_data.data.shape[0],dtype=bool)
        mask_fibre[msk_notcold] = True
        mask_fibre2[goodfibres] = True
        galaxy_data.data[~mask_fibre, :] = 0.
        galaxy_data.var[~mask_fibre, :] = 1e9
        galaxy_data.data[~mask_fibre2, :] = 0.
        galaxy_data.var[~mask_fibre2, :] = 1e9

        # Change the offsets method if necessary
        if offsets == 'file' and not hasattr(galaxy_data, 'x_refmed'):
            print('Offsets have not been pre-measured! Fitting them now.')
            offsets = 'fit'

        if (offsets == 'fit'):
            print('Fitting PSF parameters again...')
            # Fit parameter estimates from a crude centre of mass
            com_distr=utils.comxyz(x_good,y_good,data_good)
        
            # First guess sigma
            sigx=100.0
        
            # Peak height guess could be closest fibre to com position.
            dist=(x_good-com_distr[0])**2+(y_good-com_distr[1])**2 # distance between com and all fibres.
        
            # First guess Gaussian parameters.
            p0=[data_good[np.sum(np.where(dist==np.min(dist)))], com_distr[0], com_distr[1], sigx, sigx, 45.0, 0.0]
       
            gf1=fitting.TwoDGaussFitter(p0, x_good, y_good, data_good)
            gf1.fit()
            
            # Adjust the micron positions of the fibres - for use in making final cubes.
            xm=galaxy_data.x_microns-gf1.p[1]
            ym=galaxy_data.y_microns-gf1.p[2]

        elif (offsets == 'file'):
            # Use pre-measured offsets saved in the file itself
            x_shift_full = galaxy_data.x_refmed + galaxy_data.x_shift
            y_shift_full = galaxy_data.y_refmed - galaxy_data.y_shift
            xm=galaxy_data.x_microns-x_shift_full
            ym=galaxy_data.y_microns-y_shift_full
            
        else:
            # only useful for test purposes. 
            xm=galaxy_data.x_microns - np.mean(galaxy_data.x_microns)
            ym=galaxy_data.y_microns - np.mean(galaxy_data.y_microns)

        if clip_throughput:
            # Clip out fibres that have suspicious throughput values
            bad_throughput = ((galaxy_data.fibre_throughputs < 0.5) |
                              (galaxy_data.fibre_throughputs > 1.5))
            galaxy_data.data[bad_throughput, :] = 0. 
            galaxy_data.var[bad_throughput, :] = 1e9 
    
        xfibre_all[j, :] = xm
        yfibre_all[j, :] = ym

        data_all[j, :, :] = galaxy_data.data
        var_all[j, :, :] = galaxy_data.var

        # could do some mroe cleaning of input data, but not sure if this is right:
        # data_all[data_all<0.] = 0.
        # var_all[data_all<0.] = 1e9
        # data_all[np.isnan(data_all)] = 0.
        # var_all[np.isnan(data_all)] = 1e9

        psf_alpha_all[j,:] = psf_alpha
        psf_beta_all[j,:] = (psf_beta * np.ones(psf_alpha.shape).T).T

        ifus_all[j] = galaxy_data.ifu

    # Scale these up to have a wavelength axis as well
    xfibre_all = xfibre_all.reshape(n_obs, n_fibres, 1).repeat(n_slices,2)
    yfibre_all = yfibre_all.reshape(n_obs, n_fibres, 1).repeat(n_slices,2)
    

    # DAR Correction
    #
    #     The correction for differential atmospheric refraction as a function
    #     of wavelength must be applied for each file/observation independently.
    #     The DAR correction is handled by another function in this module, which
    #     updates the fibre positions in the response matrix calculation.
    
    if do_dar_correct:
        xfibre_all, yfibre_all, xdar, ydar = sami_dar_correct(ifu_list, xfibre_all, yfibre_all)
    else:
        xdar, ydar = xfibre_all * 0., xfibre_all * 0.


    return data_all, var_all, psf_alpha_all, psf_beta_all, xfibre_all * _plate_scale / 1000., yfibre_all * _plate_scale / 1000., xdar, ydar


        
def sami_plotfibre(data, xfibre, yfibre, name, path_out, show = False):
    """ Plots fibre offset positions for SAMI
    :param data: optical fibre data 
    :param xfibre: fibre position in x
    :param yfibre: fibre position in y
    :param name: specific name for savinng plot file
    :param path_out: output directory name
    :param show: if True, show interactive matplotlib plot
    """
     # Check if the object directory already exists or not.
    directory = path_out
    try:
        os.makedirs(directory)
    except OSError:
        print("Directory Exists", directory)
        print("Writing files to the existing directory")
    else:
        print("Making directory", directory)
    # Scatterplot of data for each fibre
    plt.clf()
    #plt.scatter(xfibre.flatten(),yfibre.flatten(),c=np.nansum(data, axis=2).flatten(),s=20*1.6**2, alpha=0.5)
    #plt.scatter(xfibre[0,:],yfibre[0,:],c=np.nansum(data[0,:,:], axis=1), s=600*1.6**2, alpha=1.0)
    plt.scatter(xfibre[:,0,0],yfibre[:,0,0],c=np.nansum(data[:,0,:], axis=1), s=20*1.6**2, alpha=1.0)
    # plt.scatter(np.mean(xfibre[:,:,0], axis=1),np.mean(yfibre[:,:,0], axis=1),c=np.nansum(data[:,0,:], axis=1),s=20*1.6**2, alpha=0.5)
    cbar = plt.colorbar()
    cbar.set_label('Flux', rotation=270)
    plt.axes().set_aspect('equal')
    plt.draw()
    plt.savefig(path_out + 'offsets_' + name +'.png')
    if show:
        plt.show()
        


class DataFuse3D():
    """
    General Class for building cube from data points given by fiber flux measurements(uncertainities), 
    PSF parameters(uncertainties), and coordinates of data(uncertainties) 
    as function of wavelength and number of independent observations.
    """
    def __init__(self, data, psf_params, coord, data_sigma = None , psf_params_sigma = None, 
        coord_sigma = None, name = None, dar_cor = None, pixscale = 0.25, avgpsf = False, gcovar = False, gpmethod = 'squared_exp', _Nexp = 7,marginalize=False):
        """
        :param data: callable accepting ndarrays of shape (N_obs, N_datapoints, N_wavelength)
        :param psf_params: Moffat parameters (alpha,beta) for each wavelength and observation. 
                           The shape is (2, N_obs, N_wavelength) 
        :param coord: x,y coordinates of data points. Shape (2, N_obs, N_datapoints)
        :param data_sigma: Uncertainty of data in terms of standard deviation; shape same as data. 
        :param psf_params_sigma: Standdard Deviation of Moffat parameters (alpha,beta) 
                                for each wavelength and observation. Same shape as psf_params
        :param coord_sigma: uncertainty of coordinates of data points in terms of standard deviation.
                       Shape same as coord.
        :param name: Name or identifier of object. String, Default = None
        :param dar_cor: DAR correction in arcsec. Shape (N_obs, N_wavelength). Default = None
        :param pixscale: pixel size in arcsec
        :param avgpsf: use average PSF acroos different exposures rather than individual.
        :param gcovar: caluclate covariance, currently only possible for small cubes since no compression
        :param gpmethod: specify GP kernel used: 'squared_exp' (Default)
                                            'sparse', 
                                            'wide', 
                                            'moffat'
        """

        # Checking input values and dimensions
        data = np.asarray(data)
        if data.ndim == 3:
            self.data = data
        else:
            raise ValueError("data array has not 3 dimensions!")
        psf_params = np.asarray(psf_params)
        if psf_params.ndim == 3:
            if psf_params.shape[0] == 2:
                self.psf_params = psf_params
            else:
                raise ValueError("psf_params array must be shape (2, N_wavelength, N_obs)!")
        else: 
            raise ValueError("psf_params array has not 3 dimensions!")
        coord = np.asarray(coord)
        if coord.ndim == 4:
            self.coord = coord
        else:
            raise ValueError("Coordinate points of data must be shape (2, N_data, N_obs, N_Wavelength)!")
        if data_sigma is not None:
            data_sigma = np.asarray(data_sigma)
            if data_sigma.shape == data.shape:
                self.data_sigma = data_sigma
            else:
                raise ValueError("data_sigma array has not same shape as data array!")
        else: 
            self.data_sigma = np.zeros_like(self.data) + np.mean(data) * 1e-9
        if psf_params_sigma is not None:
            psf_params_sigma = np.asarray(psf_params_sigma)
            if psf_params_sigma.shape == psf_params.shape:
                self.psf_params_sigma = psf_params_sigma
            else:
                raise ValueError("psf_params_sigma array has not same shape as psf_params array!")
        else: 
            self.psf_params_sigma = np.zeros_like(psf_params) + np.mean(psf_params) * 1e-9
        if coord_sigma is not None:
            coord_sigma = np.asarray(coord_sigma)
            if coord_sigma.shape == coord.shape:
                self.coord_sigma = coord_sigma
            else:
                raise ValueError("coord_sigma array has not same shape as coord array!")
        else: 
            self.coord_sigma = np.zeros_like(coord) + np.mean(coord) * 1e-9
        if name is not None:
            self.name = name
        else:
            self.name = 'NONAME'
        if dar_cor is not None:
            dar_cor = np.asarray(dar_cor)
            if dar_cor.shape == coord.shape:
                self.dar_cor = dar_cor
            else:
                print("DAR correction input has wrong shape, must be same as coordinate array")
        else:
            self.dar_cor = None
        self.pixscale = pixscale
        self.avgpsf = avgpsf
        self.gcovar = gcovar
        self.gpmethod = gpmethod
        self._Nexp = _Nexp

        # calculate FWHM
        fwhm = moffat2fwhm(psf_params[0], psf_params[1])

        # set gamma to half of fwhm as function of wavelength, which has been shown to maximise likleihood
        # eventually include S/N ratio as well.
        self.gamma = 0.5 * np.nanmean(fwhm, axis=0)
        #print("Average Hyperparameter gamma:", self.gamma.mean())

        if not marginalize:
            self.gamma = 0.5*np.nanmean(fwhm,axis=0)
            print("Average Hyperparameter gamma:",self.gamma.mean())
        else:
            print("Marginalizing over hyperparameter gamma")


    def fuseimage(self, wavenumber, wavenumber2 = None, Lpix = 50, plot = False, 
                  show = False, covar = False, nresponse = 32, marginalize = False):
        """Function for transformation of fiber to image 
        : param wavenumber: spectral axis number, or lower end for binnning if wavenumber2 > 1
        : param wavenumber: upper end of spectral axis for binning, if None. no binning over data
        : param Lpix: Number of pixels in x/y direction
        : param plot: Boolean indicator if final image shoudl be plotted, default None
        : param show: If True the plot will be shown for each image interactively
        : param covar: Boolean indicator if full covariance matrix should be stored or not, Default = False; 
                    check if enough RAM availbale for full covariance. 
                    Default method is to store only matrix components to reconstruct covariance matrix later.
        : param nresponse: number of steps along spectral axis to repeat calculation of response matrix; 
             nresponse should be a number = 2^n (2^n <= spectral axis), Default = 32
        : param marginalize: if True, marginalize gamma and scale factor out of posterior cube prediction
        """
        if wavenumber2 is not None:
            fibflux = np.sum(self.data[:, :, wavenumber:wavenumber2], axis = 2).flatten()
            fibfluxerr = np.sqrt(np.sum(self.data_sigma[:, :, wavenumber:wavenumber2] ** 2, axis = 2)).flatten()
            alpha = np.nanmean(self.psf_params[0,:,wavenumber:wavenumber2], axis = 1).flatten() # shape 7
            beta = np.nanmean(self.psf_params[1,:,wavenumber:wavenumber2], axis = 1).flatten() # shape 7
            coord = np.nanmean(self.coord[:,:,:,wavenumber:wavenumber2], axis = 3)
            dar_cor = np.nanmean(self.dar_cor[:,:,:,wavenumber:wavenumber2], axis = 3)
            gamma = np.nanmean(self.gamma[wavenumber:wavenumber2])
        else:
            fibflux = self.data[:, :, wavenumber].flatten()  # shape 61 * 7
            fibfluxerr = self.data_sigma[:, :, wavenumber].flatten() # shape 7
            alpha = self.psf_params[0,:,wavenumber].flatten() # shape 7
            beta = self.psf_params[1,:,wavenumber].flatten() # shape 7
            coord = self.coord[:,:,:,wavenumber]
            dar_cor = self.dar_cor[:,:,:,wavenumber]
            gamma = self.gamma[wavenumber]

        #Make sure data contains only finite elements:
        fibflux[~np.isfinite(fibflux)] = 0.
        fibfluxerr[(~np.isfinite(fibfluxerr)) | (~np.isfinite(fibflux))] = 1e9
        # Calculate response matrix only every nresponse step:
        kl_approx = False
        if wavenumber % nresponse == 0:
            self.response = cs.ResponseMatrix(coord, dar_cor, cs.moffat_psf, Lpix, self.pixscale, fft=True, 
                                              avgpsf=self.avgpsf,_Nexp = self._Nexp)
            model = cs.GPModel(self.response, fibflux, fibfluxerr, calcresponse = True, gpmethod = self.gpmethod)
        else: 
            model = cs.GPModel(self.response, fibflux, fibfluxerr, calcresponse = False, gpmethod = self.gpmethod)
            model._A = self.resp0
            model._K_gv = self.K0
            model._AK_gv = self.AK0
            model._AKA_gv = self.AKA0

        if self.avgpsf:
            #print('mean/std alpha , beta:', alpha.mean(), alpha.std(), beta.mean(), beta.std())
            if marginalize:
                logL = model.logL_margsimple(hp = [alpha.mean(), beta.mean(), gamma], verbose=True) 
                #logL = model.logL_marg(hp = [alpha.mean(), beta.mean(), gamma], verbose=True)
            else:
                logL = model.logL(hp = [alpha.mean(), beta.mean(), gamma]) 
        else:
            if marginalize:
                logL = model.logL_margsimple(hp = [alpha, beta, gamma], verbose=True)
                #logL = model.logL_marg(hp = [alpha, beta, gamma], verbose=True)
            else:
                logL = model.logL(hp = [alpha, beta, gamma])
       # print("Mean FWHM of seeing:", 2*alpha.mean()  * np.sqrt(np.power(2.,1/beta.mean()) - 1))
       # print("Std FWHM of seeing:", np.std(2*alpha  * np.sqrt(np.power(2.,1/beta) - 1)))
       # model._Kxx = model._gpkernel(model.D2, gamma)
       # RS: if we're marginalizing then we already have posterior predictions
        if not marginalize:
            model.predict(gcovar = self.gcovar)
        if not self.gcovar:
            model.covariance = 0.

        if plot:
            plt.figure(1)
            plt.subplot(121, aspect='equal')
            plt.title('Fibre Data')
            plt.scatter(self.coord[1].flatten(), self.coord[0].flatten(), c=fibflux, s=20*1.6**2, alpha=0.6)
            plt.subplot(122, aspect='equal')
            plt.title('Reconstructed Scene')
            plt.imshow(model.scene, origin="lower")
            plt.colorbar()
            plt.subplots_adjust(hspace=0.3)
            plt.savefig('image_' + self.name + '_ew' + wavenumber +'.png')
            if show: plt.show()    

        return model.scene, model.variance, model._A, model.covariance, model._K_gv, model._AK_gv, model._AKA_gv

    
    def fusecube(self, Lpix = 50, binsize = 1, nresponse = 32,  silent = True, marginalize=True):
        """ Build cube by looping over all wavelength slides
        : param Lpix: Number of pixels in x/y direction
        : param binsize: number of wavelength slices to combine in one bin; 
                        must be a number of 2^n.  Default = 2.  
        : param nresponse: number of steps along spectral axis to repeat calculation of response matrix; 
             nresponse should be a number = 2^n (2^n <= spectral axis), Default = 32
        : param silent: if True, print out each number of spectral slice or spectral bin processed, default = False
        """
        if self.data.shape[2] % binsize != 0:
            print("Warning: Ratio of Wavelength Size by Binsize not Integer. Default to binsize = 1!")
            binsize = 1
        if nresponse % binsize != 0:
            raise ValueError("nresponse must be a multiple of binsize!")
        Nbins = self.data.shape[2] / binsize
        #size_of_grid = Lpix + 10 # takes currently same number of final grid size as processing occurs
        if self.gcovar:
            # Check if its feasible to store covariance cube, need to be changed in future to sparse format (e.g. rtree)
            n_covar = Nbins * Lpix**4
            if n_covar > 5e8:
                print('Warning: Large covariance array size!')
            if n_covar > 2e9:
                print('WARNING! COVARIANCE ARRAY VERY LARGE! ONLY CONTINUE IF ENOUGH RAM/MEMORY AVAILABLE!')
        
        # Define cube size
        Lpix,Nbins = int(Lpix),int(Nbins)
        flux_cube=np.zeros((Lpix, Lpix, Nbins)) * np.nan
        var_cube=np.zeros((Lpix, Lpix, Nbins)) * np.nan
        resp_cube=np.zeros((int(self._Nexp * _Nfib), Lpix*Lpix,int(Nbins * binsize/nresponse))) * np.nan
        if self.gcovar:
            covar_cube = np.zeros((Lpix**2, Lpix**2, Nbins)) * np.nan
        else:
            covar_cube = 0.
        self.wlow_vec = np.linspace(0, Nbins-1, Nbins, dtype=int) * binsize
        self.wup_vec = np.linspace(1, Nbins, Nbins, dtype=int) * binsize
        
        #define coordinate array to correct for that each wavelength slice is reduced by mean value
        yar,xar = np.mgrid[1:flux_cube.shape[0]+1, 1:flux_cube.shape[1]+1]
        r = np.sqrt((xar - flux_cube.shape[0]/2)**2 + (yar - flux_cube.shape[1]/2)**2)
        self.RA_cen = self.coord[0].mean()
        self.DEC_cen  = self.coord[1].mean()

        # We assume here that the input data is in units of surface brightness,
        # where the number in each input RSS pixel is the surface brightness in
        # units of the fibre area (e.g., ergs/s/cm^2/fibre). We want the surface
        # brightness in units of the output pixel area.
        fibre_area_pix = np.pi * _Rfib_arcsec**2 / self.pixscale**2
        self.data /= fibre_area_pix
        self.data_sigma /= fibre_area_pix
        # array to store binnned FHWM of seeing PSF
        self.fwhm_seeing = np.zeros(Nbins)

        # loop over wavelength, treat each bin independent from adjacent wavelength bins!
        # replace evtl loop with map function and enable multiprocessing option
        # map(self.fuseimage, range(Nbins))
        #print("Looping over wavelength range...")
        for l in range(Nbins):
            print("wavelength slide:", self.wlow_vec[l], ' - ', self.wup_vec[l])
            if binsize > 1:
                data_l, var_l, resp_l, covar_l, K_l, AK_l, AKA_l = self.fuseimage(self.wlow_vec[l], wavenumber2 = self.wup_vec[l], 
                                                                                  Lpix = Lpix, plot = False, covar = False, 
                                                                                  nresponse = nresponse, marginalize=marginalize)
            elif binsize == 1:
                data_l, var_l, resp_l, covar_l, K_l, AK_l, AKA_l = self.fuseimage(self.wlow_vec[l], Lpix = Lpix, plot = False, 
                                                                                  covar = False, nresponse = nresponse, marginalize=marginalize)
            ### optionally shift image in cube for additonal correction, comment out:
            #xdar_l = (self.coord[0, :, :, self.wlow_vec[l]:self.wup_vec[l]].mean() - self.RA_cen) / self.pixscale
            #ydar_l = (self.coord[1, :, :, self.wlow_vec[l]:self.wup_vec[l]].mean() - self.DEC_cen) / self.pixscale
            #flux_cube[:, :, l] = imshift(data_l, (xdar_l,ydar_l), order=1)
            #var_cube[:, :, l] = imshift(var_l, (xdar_l,ydar_l), order=1)
            # or uncomment for no additional correction:
            flux_cube[:, :, l] = data_l 
            var_cube[:, :, l] = var_l 
            if l * binsize % nresponse == 0:
                #code.interact(local=dict(globals(),**locals()))
                resp_cube[:,:,l*binsize//nresponse] = resp_l
                self.resp0 = resp_l # use repsone matrix for runs till next interval
                self.K0 = K_l
                self.AK0 = AK_l
                self.AKA0 = AKA_l
            if self.gcovar:
                covar_cube[:,:,l] =  covar_l #imshift(var_l, (xdar_l,ydar_l), order=1)

            self.fwhm_seeing[l] = np.nanmean(self.gamma[self.wlow_vec[l]:self.wup_vec[l]]) * 2.


        #OPtional: Accept data values with at least signal-to-noise ratio of >=1, otherwise set to zero
        # flux_cube[flux_cube/np.sqrt(var_cube) < 1.] = 0.
        # Exlude range outside fiber fundle: size = 14.7 arces = 15.22arcsec/mm
        # Set values outside of FoV radius to NaN:
        rmax = (_fov_arcsec + 2 * _Rfib_arcsec)/2. / self.pixscale
        flux_cube[r > rmax, :] = np.nan
        var_cube[r > rmax, :] = np.nan

        return flux_cube, var_cube, resp_cube, covar_cube




def sami_write_file(filenames_in, identifier, data_cube, var_cube, path_out = 'cubes/', filename_out = None, overwrite = True, covar_mode = None, pixscale = 0.5):
    """ Writes cube and variance data in fits file, similar to SAMI pipeline
    :param path_in: path to optical fibre data for header
    :param filenames_in: filename of optical fibre data for header
    :param identifier: String of identifier or object ID
    :param data_cube: flux data  to be written to fits file
    :param var_cube:  flux variance data to be written to fits file
    :param path_out: output path, by default will create a new directory
    :param filename_out: filename for output fits file
    :param overwrite: if True overwrites any exixting cube data
    :param covar_mode: not enable yet, default None
    :param pixscale: pixel size in arcsec
    """

    # First get SAMI specific parameters:
    files = filenames_in
    name = identifier
    ifu_list = []
    for filename in files:
        ifu_list.append(utils.IFU(filename, name, flag_name=True))
    size_of_grid = data_cube.shape[0]
    output_pix_size_arcsec = pixscale
    n_files = len(files)

    # Check if the object directory already exists or not.
    directory = path_out
    try:
        os.makedirs(directory)
    except OSError:
        print("Directory Exists", directory)
        print("Writing files to the existing directory")
    else:
        print("Making directory", directory)

    # Filename to write to
    if filename_out is not None:
        outfile_name_full=os.path.join(directory, filename_out)
    else: 
        arm = ifu_list[0].spectrograph_arm            
        outfile_name=str(name)+'_'+str(arm)+'_'+str(n_files)+'.fits'
        outfile_name_full=os.path.join(directory, outfile_name)

    # Check if the filename already exists
    if os.path.exists(outfile_name_full):
        if overwrite:
            os.remove(outfile_name_full)
        else:
            print('Output file already exists:')
            print(outfile_name_full)
            print('Skipping this object')
            return False
    if overwrite & (filename_out == None):
    # Check for old cubes that used a different number of files
        for path in glob(os.path.join(
                directory, str(name)+'_'+str(arm)+'_*'+suffix+'.fits')):
            os.remove(path)

    if ifu_list[0].gratid == '580V':
        band = 'g'
    elif ifu_list[0].gratid == '1000R':
        band = 'r'
    elif not nominal:
        # Need an identified band if we're going to do full WCS matching.
        # Should try to work something out, like in scale_cube_pair_to_mag()
        raise ValueError('Could not identify band. Exiting')
    else:
        # When nominal is True, wcs_solve doesn't even look at band
        band = None

    # Equate Positional WCS
    WCS_pos, WCS_flag=wcs.wcs_solve(ifu_list[0], data_cube, name, band, size_of_grid, output_pix_size_arcsec, plot=True, nominal=True)
    
    # First get some info from one of the headers.
    list1=pf.open(files[0])
    hdr=list1[0].header

    hdr_new = sami_create_primary_header(ifu_list, name, files, WCS_pos, WCS_flag)

    # Define the units for the datacube
    hdr_new['BUNIT'] = ('10**(-16) erg /s /cm**2 /angstrom /pixel', 
                        'Units')

    hdr_new['CBINGMET'] = (cubing_method, 'Method adopted for cubing')

    hdr_new['IFUPROBE'] = (sami_get_probe_all(files, name, verbose=False),
                           'Id number(s) of the SAMI IFU probe(s)')

    # Create HDUs for each cube.
    
    list_of_hdus = []

    # @NOTE: PyFITS writes axes to FITS files in the reverse of the sense
    # of the axes in Numpy/Python. So a numpy array with dimensions
    # (5,10,20) will produce a FITS cube with x-dimension 20,
    # y-dimension 10, and the cube (wavelength) dimension 5.  --AGreen
    list_of_hdus.append(pf.PrimaryHDU(np.transpose(data_cube, (2,1,0)), hdr_new))
    list_of_hdus.append(pf.ImageHDU(np.transpose(var_cube, (2,1,0)), name='VARIANCE'))
    # list_of_hdus.append(pf.ImageHDU(np.transpose(weight_cube, (2,1,0)), name='WEIGHT'))

    # Covariance cube not yet included, instead use response matrix to recover covariance
    if covar_mode is not None:
        hdu4 = pf.ImageHDU(np.transpose(covariance_cube,(4,3,2,1,0)),name='COVAR')
        hdu4.header['COVARMOD'] = (covar_mode, 'Covariance mode')
        if covar_mode == 'optimal':
            hdu4.header['COVAR_N'] = (len(covar_locs), 'Number of covariance locations')
            for i in range(len(covar_locs)):
                hdu4.header['HIERARCH COVARLOC_'+str(i+1)] = covar_locs[i]
        list_of_hdus.append(hdu4)


    list_of_hdus.append(sami_create_qc_hdu(files, name))
    
    # Put individual HDUs into a HDU list
    hdulist = pf.HDUList(list_of_hdus)

    # Write the file
    print("Writing", outfile_name_full)
    hdulist.writeto(outfile_name_full)

    # Close the open file
    list1.close()



def sami_create_primary_header(ifu_list,name,files,WCS_pos,WCS_flag):
    """Create a primary header to attach to each cube from the RSS file headers
        See SAMI pipeline
    """

    hdr = ifu_list[0].primary_header
    fbr_hdr = ifu_list[0].fibre_table_header

    # Create the wcs. Note 2dfdr uses non-standard CTYPE and CUNIT, so these
    # are not copied
    wcs_new=pw.WCS(naxis=3)
    wcs_new.wcs.crpix = [WCS_pos["CRPIX1"], WCS_pos["CRPIX2"], hdr['CRPIX1']]
    wcs_new.wcs.cdelt = np.array([WCS_pos["CDELT1"], WCS_pos["CDELT2"], hdr['CDELT1']])
    wcs_new.wcs.crval = [WCS_pos["CRVAL1"], WCS_pos["CRVAL2"], hdr['CRVAL1']]
    wcs_new.wcs.ctype = [WCS_pos["CTYPE1"], WCS_pos["CTYPE2"], "AWAV"]
    wcs_new.wcs.equinox = 2000
    wcs_new.wcs.radesys = 'FK5'
            
    # Create a header
    hdr_new=wcs_new.to_header(relax=True)
    #hdr_new.update('WCS_SRC',WCS_flag,'WCS Source')
    hdr_new['WCS_SRC'] = (WCS_flag, 'WCS Source')

    # Putting in the units by hand, because otherwise astropy converts
    # 'Angstrom' to 'm'. Note 2dfdr uses 'Angstroms', which is non-standard.
    hdr_new['CUNIT1'] = WCS_pos['CUNIT1']
    hdr_new['CUNIT2'] = WCS_pos['CUNIT2']
    hdr_new['CUNIT3'] = 'Angstrom'
            
    # Add the name to the header
    hdr_new['NAME'] = (name, 'Object ID')
    # Need to implement a database specific-specific OBSTYPE keyword to indicate galaxies
    # *NB* This is different to the OBSTYPE keyword already in the header below
    
    # Determine total exposure time and add to header
    total_exptime = 0.
    for ifu in ifu_list:
        total_exptime+=ifu.primary_header['EXPOSED']
    hdr_new['TOTALEXP'] = (total_exptime, 'Total exposure (seconds)')

    # Add the mercurial changeset ID to the header
    # hdr_new['HGCUBING'] = (HG_CHANGESET, 'Hg changeset ID for cubing code')
    # Need to implement a global version number for the database
    
    # Put the RSS files into the header
    for num in range(len(files)):
        rss_key='HIERARCH RSS_FILE '+str(num+1)
        rss_string='Input RSS file '+str(num+1)
        hdr_new[rss_key] = (os.path.basename(files[num]), rss_string)

    # Extract header keywords of interest from the metadata table, check for consistency
    # then append to the main header

    primary_header_keyword_list = ['DCT_DATE','DCT_VER','DETECXE','DETECXS','DETECYE','DETECYS',
                                   'DETECTOR','XPIXSIZE','YPIXSIZE','METHOD','SPEED','READAMP','RO_GAIN',
                                   'RO_NOISE','ORIGIN','TELESCOP','ALT_OBS','LAT_OBS','LONG_OBS',
                                   'RCT_VER','RCT_DATE','INSTRUME','SPECTID',
                                   'GRATID','GRATTILT','GRATLPMM','ORDER','TDFCTVER','TDFCTDAT','DICHROIC',
                                   'OBSTYPE','TOPEND','AXIS','AXIS_X','AXIS_Y','TRACKING','TDFDRVER']

    primary_header_conditional_keyword_list = ['HGCOORDS','COORDROT','COORDREV']

    for keyword in primary_header_keyword_list:
        if keyword in ifu.primary_header.keys():
            val = []
            for ifu in ifu_list: val.append(ifu.primary_header[keyword])
            if len(set(val)) == 1: 
                hdr_new.append(hdr.cards[keyword])
            else:
                print('Non-unique value for keyword:',keyword)

    # Extract the couple of relevant keywords from the fibre table header and again
    # check for consistency of keyword values

    fibre_header_keyword_list = ['PLATEID','LABEL']

    for keyword in fibre_header_keyword_list:
        val = []
        for ifu in ifu_list: val.append(ifu.fibre_table_header[keyword])
        if len(set(val)) == 1:
            hdr_new.append(ifu_list[0].fibre_table_header.cards[keyword])
        else:
            print('Non-unique value for keyword:', keyword)

    # Append HISTORY from the initial RSS file header, assuming HISTORY is
    # common for all RSS frames.

    hdr_new.append(hdr.cards['SCRUNCH'])
    hist_ind = np.where(np.array(hdr.keys()) == 'HISTORY')[0]
    for i in hist_ind: 
        hdr_new.append(hdr.cards[i])

    # Add catalogue RA & DEC to header
    hdr_new.set('CATARA', ifu_list[0].obj_ra[0], after='CRVAL3')
    hdr_new.set('CATADEC', ifu_list[0].obj_dec[0], after='CATARA')

    # Additional header items from random extensions
    # Each key in `additional` is a FITS extension name
    # Each value is a list of FITS header keywords to copy from that extension
    additional = {'FLUX_CALIBRATION': ('STDNAME',)}
    for extname, key_list in additional.items():
        try:
            add_hdr_list = [pf.getheader(f, extname) for f in files]
        except KeyError:
            print('Extension not found:', extname)
            continue
        for key in key_list:
            val = []
            try:
                for add_hdr in add_hdr_list:
                    val.append(add_hdr[key])
            except KeyError:
                print('Keyword not found:', key, 'in extension', extname)
                continue
            if len(set(val)) == 1:
                hdr_new.append(add_hdr.cards[key])
            else:
                print('Non-unique value for keyword:', key, 'in extension', extension)

    return hdr_new

def sami_create_qc_hdu(file_list, name):
    """Create and return an HDU of QC information.
        See SAMI pipeline
    """
    # The name of the object is passed, so that information specific to that
    # object can be included, but at the moment it is not used.
    qc_keys = (
        'SKYMDCOF',
        'SKYMDLIF',
        'SKYMDCOA',
        'SKYMDLIA',
        'SKYMNCOF',
        'SKYMNLIF',
        'SKYMNCOA',
        'SKYMNLIA',
        'TRANSMIS',
        'FWHM')
    rel_transp = []
    qc_data = {key: [] for key in qc_keys}
    for path in file_list:
        hdulist = pf.open(path)
        try:
            rel_transp.append(
                1.0 / hdulist['FLUX_CALIBRATION'].header['RESCALE'])
        except KeyError:
            rel_transp.append(-9999)
        if 'QC' in hdulist:
            qc_header = hdulist['QC'].header
            for key in qc_keys:
                if key in qc_header:
                    qc_data[key].append(qc_header[key])
                else:
                    qc_data[key].append(-9999)
        else:
            for key in qc_keys:
                qc_data[key].append(-9999)
        hdulist.close()
    filename_list = [os.path.basename(f) for f in file_list]
    columns = [
        pf.Column(name='filename', format='20A', array=filename_list),
        pf.Column(name='rel_transp', format='E', array=rel_transp),
        ]
    for key in qc_keys:
        columns.append(pf.Column(name=key, format='E', array=qc_data[key]))
    hdu = pf.BinTableHDU.from_columns(columns, name='QC')
    return hdu

def read_cube(fullname):
    """ Loads data and variance saved in cube fits file
    :param fullname: path and filename of cube fits file to read in
    """
    hdulist = pf.open(fullname)
    data = np.transpose(hdulist[0].data, (2,1,0))
    variance = np.transpose(hdulist[1].data, (2,1,0))

    return data, variance


def write_response_cube(identifier, resp_cube, var_fibre, gamma, pixscale, Lpix, gpmethod, marginalize, 
                        path_out, filename_out, overwrite = True, _Nexp = 7):
    """ Writes response matrix, fibre data variance, and gama in fits file to calculate covariance later    
    :param identifier: String of identifier or object ID
    :param resp_cube: response matrix/cube A
    :param var_fibre:  flux variance if fibre data
    :param gamma:  GP hyperparameter gamma vector along spectral axis
    :param pixscale: pixel scale of cube in arcsec
    :param Lpix: number of pixels in x,y axis of cube 
    :param gpmethod: specify GP kernel used: 'squared_exp'
                                            'sparse', 
                                            'wide', 
                                            'moffat'
    :param path_out: output path, by default will create a new directory
    :param filename_out: filename for output fits file
    :param overwrite: if True overwrites any exixting cube data
    """
    name = identifier
    # Check if the object directory already exists or not.
    directory = path_out
    try:
        os.makedirs(directory)
    except OSError:
        print("Directory Exists", directory)
        print("Writing files to the existing directory")
    else:
        print("Making directory", directory)

    # Filename to write to
    outfile_name_full=os.path.join(directory, filename_out)

    # Check if the filename already exists
    if os.path.exists(outfile_name_full):
        if overwrite:
            os.remove(outfile_name_full)
        else:
            print('Output file already exists:')
            print(outfile_name_full)
            print('Skipping this object')
            return False

    # create header
    wcs_new = pw.WCS(naxis=3)
    hdr_new = wcs_new.to_header(relax=True)

    if marginalize:
        gpmarginalized = 'yes'
    else:
        gpmarginalized = 'no'

    # Putting in the units by hand, because otherwise astropy converts
    # 'Angstrom' to 'm'. Note 2dfdr uses 'Angstroms', which is non-standard.
    hdr_new['CUNIT3'] = 'Angstrom'       
    # Add the name to the header
    hdr_new['NAME'] = (name, 'Object ID')
    hdr_new['PIXSCALE'] = pixscale
    hdr_new['NEXP'] = _Nexp
    hdr_new['NFIB'] = _Nfib
    hdr_new['LPIX'] = Lpix
    hdr_new['GPMETHOD'] = gpmethod
    hdr_new['GPMARGINAL'] = gpmarginalized

    # @NOTE: PyFITS writes axes to FITS files in the reverse of the sense
    # of the axes in Numpy/Python.
    list_of_hdus = []
    list_of_hdus.append(pf.PrimaryHDU(np.transpose(resp_cube, (2,1,0)), hdr_new))
    list_of_hdus.append(pf.ImageHDU(np.transpose(var_fibre, (2,1,0)), name='VARIANCE'))
    list_of_hdus.append(pf.ImageHDU(gamma, name='GAMMA'))
    
    # Put individual HDUs into a HDU list
    hdulist = pf.HDUList(list_of_hdus)

    # Write the file
    print("Writing", outfile_name_full)
    hdulist.writeto(outfile_name_full)
    print("--------------------------------------------------------------")


def read_response_cube(path, filename):
    """ Read fits file and returns response matrix/cube, fibre data variance, and gamma 
    :param path: input path
    :param filename: filename 
    """
    hdulist = pf.open(fullname)
    resp = np.transpose(hdulist[0].data, (2,1,0))
    variance = np.transpose(hdulist[1].data, (2,1,0))
    gamma = hdulist[2].data
    hdr = hdulist[0].header

    return resp, var, gamma
    

def test_dar_solution(cubedata, cubevar, cubecovar, data, var, path_out, plot = True, show = False, pixelarcsec = 0.35, seeing = 0., gcovar = False):
    """ Plot central position, FWHM and flux as function of wavevelength to test for effect of DAR misalignment.
     FWHM is fitted with 2D Moffat and Gaussian.
    : param cubedata: Data cube to test for DAR correction, shape = [Npixel, Npixel, Nwave]
    : param fibredata: data before cubing as function of wavelength. Is used to normalize using total flux.
    : param npix: used to cut out central box around center peak flux with plus/minus npixels
    : param plot: if True saves an image
    : param pixelarcsec: size of one pixel in acrsec
    : param seeing: array of PSF seeing in arcsec, must have same length in wavelength as cubedata
    : param gcovar: boolean, plot radial covariance for differt wavlenegth slices, default = False
    """

    print('Evaluation and plotting of DAR and FWHM...')
    # First include only data larger than 2 sigma
    s2n_cube = cubedata/np.sqrt(cubevar)
    #cubedata[(s2n_cube < 2) | np.isnan(s2n_cube)] = 0.
    #cubevar[(s2n_cube < 2) | np.isnan(s2n_cube)] = np.nan
    nslides = cubedata.shape[2]
    npix = cubedata.shape[0]
    flux = np.zeros((9,nslides))
    flux_total = np.zeros(nslides)
    fluxvar = np.zeros((9,nslides))
    fluxvar_total =  np.zeros(nslides)
    fwhm, fwhm2 = np.zeros(nslides), np.zeros(nslides)
    fwhm_err, fwhm2_err = np.zeros(nslides), np.zeros(nslides)
    cen_pos = np.zeros((2,nslides))
    #define pixel area to measure flux measurements, first define center:
    idx0, idy0 = np.where(cubedata[:,:,nslides / 2] == np.nanmax(cubedata[:,:,nslides / 2]))
    idx0, idy0 = idx0.sum(), idy0.sum()
    # print('idx0, idy0', idx0, idy0)
    rmax = np.round((14.7/2. - 3.2) / pixelarcsec).astype(int)
    # print('rmax:', rmax)
    diff = (cubedata.shape[0]/2 - rmax)
    # print('diff:', diff)
    # Loop over wavelengths
    for i in range(nslides):
        if np.isfinite(cubedata[:,:,i]).any() & (np.nanmean(cubevar[:,:,i]) < 25. * np.nanmean(cubevar)):
            n = 0
            for j in range(3):
                for k in range(3):
                    flux[n,i] = cubedata[idx0 -1 + j, idy0 -1 +k, i]
                    fluxvar[n,i] = cubevar[idx0 -1 + j, idy0 -1 +k, i]
                    n += 1
            flux_total[i] = np.nansum(cubedata[idx0 - rmax : idx0 + rmax ,idy0 - rmax : idy0 + rmax, i])
            fluxvar_total[i] = np.nansum(cubevar[idx0 - rmax : idx0 + rmax ,idy0 - rmax : idy0 + rmax, i])
            if flux_total[i]/np.sqrt(fluxvar_total[i]) < 2.:
                flux_total[i] = np.nan
                flux[:,i] = np.nan
            # Fit reconstructed FWHM
            try:
                fwhm[i],fwhm_err[i], cen_pos[0,i], cen_pos[1,i] = fit_source_fwhm(cubedata[idx0 - rmax : idx0 + rmax ,idy0 - rmax : idy0 + rmax, i], (idx0-diff, idy0-diff), func = 'gaussian')
                fwhm2[i], fwhm2_err[i], _, _= fit_source_fwhm(cubedata[idx0 - rmax : idx0 + rmax ,idy0 - rmax : idy0 + rmax, i], (idx0-diff, idy0-diff), func = 'moffat')
            except: 
                fwhm[i], fwhm2[i] = 0., 0.
                fwhm_err[i], fwhm2_err[i] = 0., 0.
        else:
            flux[:,i] = 0.
            flux_total[i] = 1.
            #fluxvar[:,i] = np.nan
            #fluxvar_total[i] = np.nan
            fwhm[i], fwhm2[i] = 0., 0.
            fwhm_err[i], fwhm2_err[i] = 0., 0. 
            cen_pos[:,i] = [0.,0.]



    # Calculate Signal-to-Noise ratios of input data and cube data
    cubedata = cubedata.reshape(cubedata.shape[0] * cubedata.shape[1], cubedata.shape[2])
    cubevar = cubevar.reshape(cubevar.shape[0] * cubevar.shape[1], cubevar.shape[2])
    cubedata[np.isnan(cubedata)] = 0
    cubevar[np.isnan(cubevar)] = 1e9
    cubedata[(cubedata < np.nanmean(cubedata)) | (cubevar>1.)] = 0.
    cubevar[(cubedata < np.nanmean(cubedata)) | (cubevar>1.)] = 1e9
    s2n_cube = np.nansum(cubedata, axis=0) / np.sqrt(np.nansum(cubevar, axis=0))
    s2n_cube[s2n_cube == 0] = np.nan

    data_fibre = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
    var_fibre = var.reshape(data.shape[0] * data.shape[1], data.shape[2])
    data_fibre[np.isnan(data_fibre)] = 0
    var_fibre[np.isnan(var_fibre)] = 1e9
    data_fibre[(data_fibre < np.nanmean(data_fibre)) | (var_fibre>1.) ] = 0
    var_fibre[(data_fibre < np.nanmean(data_fibre)) | (var_fibre>1.) ] = 1e9
    nbins = len(s2n_cube)
    s2n_fibre = np.zeros(nbins)
    binsize = data_fibre.shape[1]/nbins
    for i in range(nbins):
        s2n_fibre[i] = np.nansum(data_fibre[:, i*binsize:(i+1)*binsize]) / np.sqrt(np.nansum(var_fibre[:, i*binsize:(i+1)*binsize]))

    s2n_fibre[s2n_fibre == 0] = np.nan
    fwhm[np.isnan(flux_total)] = np.nan
    fwhm2[np.isnan(flux_total)] = np.nan
    if plot:
        plt.figure(figsize=(11,8))
        norm = np.nanmean(flux_total[flux_total > 0])
        ax2 = plt.subplot(321)
        #ax2.plot([1,nslides],[0. , 2*(flux0/flux_total).max() ], '.', alpha=0.)
        ax2.plot([1,nslides],[0. , flux.max() * 1.2], '.', alpha=0.)
        fluxa = np.zeros_like(flux)
        #fluxa[np.isnan(fluxa)] = 0
        for i in range(9):
            if np.nanmedian(flux[i,:]) > 0.:
                fluxa[i] = flux[i,:]/flux_total/np.nanmedian(flux[i,:])* norm
            else: 
                fluxa[i] = np.nan
            ax2.plot(np.linspace(1,nslides, nslides), fluxa[i], color='k')
        #ax2.set_ylim(np.nanmin(fluxa[fluxa > 0.]) - 0.1, np.nanmax(fluxa) * 1.1)
        #ax2.set_ylim(np.nanmean(fluxa[fluxa > 0.]) - 5*np.nanstd(fluxa[fluxa > 0.]), np.nanmean(fluxa[fluxa > 0.]) + 5*np.nanstd(fluxa[fluxa > 0.]))
        ax2.set_ylim(0.3, 1.9)
        ax2.set_ylabel('Flux / Total Flux')
        ax4 = plt.subplot(322)
        flux_mean = np.nanmean(fluxa, axis=0)
        flux_std = np.nanstd(fluxa, axis=0)
        flux_mean[np.isnan(flux_total) | (flux_mean <= 0)] = np.nan
        flux_std[np.isnan(flux_mean)] = np.nan
        print("Average Std of flux ratio between 9 pixel:", np.nanmedian(flux_std))
        ax4.plot([1,nslides],[0. , flux.max() * 1.2], '.', alpha=0.)
        ax4.plot(np.linspace(1,nslides, nslides),flux_mean, linestyle = '--', color='blue')
        ax4.fill_between(np.linspace(1,nslides, nslides), flux_mean-flux_std, flux_mean+flux_std, color = 'cyan')
        #ax4.set_ylim(np.nanmin(fluxa[fluxa > 0]) - 0.1, np.nanmax(fluxa) * 1.1)
        #ax4.set_ylim(np.nanmean(fluxa[fluxa > 0.]) - 5*np.nanstd(fluxa[fluxa > 0.]), np.nanmean(fluxa[fluxa > 0.]) + 5*np.nanstd(fluxa[fluxa > 0.]))
        ax4.set_ylim(0.3, 1.9)
        ax4.set_ylabel('Mean(Std) of Flux/Total Flux')
        ax1 = plt.subplot(323)
        roff = np.sqrt((cen_pos[0,:] - idx0 + diff - 1) **2 + (cen_pos[1,:]- idy0 + diff - 1) **2)
        roff[flux_total == 1.] = 0.
        roff[np.isnan(flux_mean)] = np.nan
        ax1.plot([1,nslides],[0. , roff.max() * 1.2*pixelarcsec ], '.', alpha=0.)
        ax1.plot(np.linspace(1,nslides, nslides), roff * pixelarcsec)
        #ax1.set_ylim(np.nanmean(roff* pixelarcsec) - 2* np.nanstd(roff* pixelarcsec), np.nanmean(roff* pixelarcsec) + 2* np.nanstd(roff* pixelarcsec))
        #ax1.set_ylim(np.nanmin(roff[roff>0]* pixelarcsec) - 0.1, np.nanmax(roff* pixelarcsec) *1.1)
        ax1.set_ylim(0., 1.)
        ax1.set_ylabel('Offset [arcsec]')
        ax3 = plt.subplot(324)
        fwhm[np.isnan(flux_mean)] = np.nan
        fwhm2[np.isnan(flux_mean)] = np.nan
        ax3.plot([1,nslides],[0. , np.nanmax(fwhm) * 2.2*pixelarcsec ], '.', alpha=0.)
        ax3.plot(np.linspace(1,nslides, nslides), fwhm * pixelarcsec, color = 'b', label = 'Gaussian')
        ax3.plot(np.linspace(1,nslides, nslides), fwhm2 * pixelarcsec, color = 'r', label = 'Moffat')
        print("Average FWHM Gaussian, Moffat, Seeing:", np.nanmean(fwhm)* pixelarcsec, np.nanmean(fwhm2)* pixelarcsec, np.nanmean(seeing))
        try:
            ax3.plot(np.linspace(1,nslides, nslides), seeing, color = 'k', linestyle='--', label = 'PSF seeing')
        except:
            print('Fail to plot PSF seeing.')
        #ax3.set_ylim(np.nanmin(fwhm[fwhm > 0]) * pixelarcsec - 0.5, np.nanmax(fwhm) * pixelarcsec * 1.5)
        #ax3.set_ylim(np.nanmean(fwhm* pixelarcsec) - 5* np.nanstd(fwhm* pixelarcsec), np.nanmean(fwhm* pixelarcsec) + 5* np.nanstd(fwhm* pixelarcsec) + 1)
        ax3.set_ylim(1., 4.)
        plt.legend()
        ax3.set_ylabel('FWHM [arcsec]')
        ax5 = plt.subplot(325)
        s2n_fibre[np.isnan(flux_mean)] = np.nan
        ax5.plot([1,nslides],[0. , s2n_fibre.max() * 1.2], '.', alpha=0.)
        ax5.plot(np.linspace(1,nslides, nslides), s2n_fibre)
        ax5.set_ylim(np.nanmean(s2n_fibre) - 5 * np.nanstd(s2n_fibre), np.nanmean(s2n_fibre) + 5 * np.nanstd(s2n_fibre) )
        ax5.set_ylabel('Signal/Noise Fibre')
        ax5.set_xlabel('Wavelength Slice')
        ax6 = plt.subplot(326)
        s2n_cube[np.isnan(flux_mean)] = np.nan
        ax6.plot([1,nslides],[0. , s2n_cube.max() * 1.2], '.', alpha=0.)
        ax6.plot(np.linspace(1,nslides, nslides), s2n_cube)
        ax6.set_ylim(np.nanmean(s2n_cube) - 5 * np.nanstd(s2n_cube), np.nanmean(s2n_cube) + 5 * np.nanstd(s2n_cube) )
        ax6.set_ylabel('Signal/Noise Cube')
        ax6.set_xlabel('Wavelength Slice')
        plt.savefig(path_out + 'DAR_test3.png')
        if show: plt.show()

        if gcovar:
            import matplotlib.cm as cm
            colors = cm.jet(np.linspace(0, 1, cubecovar.shape[2]))
            plt.figure(2)
            plt.clf()
            Nbin_half = np.round(nslides/2).astype(int)
            Npix_half = np.round(npix/2).astype(int)
            for i in range(nslides):
                #plt.plot(np.linspace(-25,24,50)*0.32,cubecovar[1025,1000:1050,i], c=colors[i])
                plt.plot(np.linspace(-Npix_half,Npix_half-1,npix)*pixelarcsec,cubecovar[Nbin_half*npix + Npix_half,Nbin_half * npix:(Nbin_half + 1) * npix,i], c=colors[i])
            plt.xlabel('Radius [arcsec]')
            plt.ylabel('covariance')
            plt.savefig('covariance.png')
            if show: plt.show()

        # plt.figure(3)
        # plt.clf()
        # plt.plot(pixelarcsec*fwhm[fwhm>0.5]/seeing[fwhm>0.5], s2n_cube[fwhm>0.5]/np.nanmean(s2n_cube),'o',color='b',label='Cube')
        # plt.plot(pixelarcsec*fwhm[fwhm>0.5]/seeing[fwhm>0.5], s2n_fibre[fwhm>0.5]/np.nanmean(s2n_fibre),'o',color='r',label='Fibre')
        # plt.xlabel('reconstructed FWHM / Seeing')
        # plt.ylabel('Normalized S/N')
        # plt.legend(numpoints=1)
        # plt.savefig(path_out + 's2n_fwhm.png')
        # if show: plt.show()

    return flux, flux_total, fwhm, fwhm2, fwhm_err, fwhm2_err, s2n_cube, s2n_fibre


















