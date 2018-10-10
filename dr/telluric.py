"""
Code for correcting for telluric absorption in SAMI data.

Most of the heavy lifting is done by the fluxcal2 module, which fits the
data to find the spectrum of the standard star. derive_transfer_function()
fits a straight line to the data (under the assumption that that is a
decent description of the stellar spectrum) and infers the telluric
absorption from that. By default the secondary standard, in the galaxy
field, is used. As an alternative the primary (spectrophotometric)
standard can be used - this is useful for the pilot data where the
secondary standards were too faint to be much use.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import warnings

from .fluxcal2 import read_chunked_data, set_fixed_parameters, fit_model_flux
from .fluxcal2 import insert_fixed_parameters, check_psf_parameters
from .fluxcal2 import extract_total_flux, save_extracted_flux, trim_chunked_data

from .. import utils
from ..utils.ifu import IFU
from ..utils.other import clip_spectrum

import astropy.io.fits as pf
import numpy as np
from scipy.ndimage.filters import median_filter
import scipy.optimize as optimize
import re

HG_CHANGESET = utils.hg_changeset(__file__)

# KEY:      SS = Secondary Standard, PS = Primary Standard

def derive_transfer_function(frame_list, PS_spec_file=None, use_PS=False,
                             scale_PS_by_airmass=False, 
                             model_name='ref_centre_alpha_dist_circ_hdratm',
                             n_trim=0, use_probe=None, hdu_name='FLUX_CALIBRATION'):
    """
    Finds the telluric correction factor to multiply object data by. The factor 
    as a function of wavelength is saved into the red frame under the extension 
    "FLUX_CALIBRATION" and has values that are mostly 1s apart from the telluric 
    regions.
    """
    
    # frame_list = list = two element list of strings that give the path and file names for the location of the secondary standard. First element is the blue frame and the second is the red frame.
    # PS_spec_file = str = path and file name of Primary Standard's "TRANSFERcombined.fits" file.
    # use_PS = bool = switch to use Primary Standard or Secondary Standard for the telluric transfer function. Default is to use the SS (use_PS=False), but if working with Pilot data, then the user might want to change this to "use_PS=True" such that the PS is used and scaled to the SS optical depth. This might become default, but requires testing before so.
    # n_trim = int = trim this many chunks off the blue end (used for pilot data only)
    
    # Always re-extract the secondary standard
    # NB: when use_PS is True and scale_PS_by_airmass is False, we don't
    # actually need to extract it, but we do still need to create the
    # extension and copy atmospheric parameters across
    extract_secondary_standard(frame_list, model_name=model_name, n_trim=n_trim, use_probe=use_probe, hdu_name=hdu_name)

    # if user defines, use a scaled primary standard telluric correction
    if use_PS:
        # get primary standard transfer function
        PS_transfer_function, PS_sigma_transfer, linear_fit, PS_wave_axis = primary_standard_transfer_function(PS_spec_file)

        if scale_PS_by_airmass:
            # Use the theoretical scaling based on airmass
            scale = ((1.0 / np.cos(np.deg2rad(pf.getval(frame_list[1], 'ZDSTART')))) / 
                     (1.0 / np.cos(np.deg2rad(pf.getval(PS_spec_file, 'MEANZD')))))
        else:
            # find least squares fit on scalar
            A = 1.1
            best_scalar = optimize.leastsq(residual,A,args=(SS_transfer_function,PS_transfer_function,PS_wave_axis),full_output=1)
            scale = best_scalar[0][0]

        PS_transfer_function_scaled = PS_transfer_function ** scale
        PS_sigma_factor_scaled = PS_sigma_transfer * scale * PS_transfer_function**(scale-1)

        transfer_function = PS_transfer_function_scaled
        sigma_transfer = PS_sigma_transfer
    
    else:
        # Get data
        hdulist = pf.open(frame_list[1])
        hdu = hdulist[hdu_name]
       
        # Load in SS flux data
        SS_flux_data_raw = hdu.data[0, :]
        SS_sigma_flux = hdu.data[2, :]
        # Might put in an interpolation over NaNs; for now just taking a straight copy
        SS_flux_data = SS_flux_data_raw.copy()
        header = hdulist[0].header
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        naxis1 = header['NAXIS1']
        crpix1 = header['CRPIX1']
        SS_wave_axis = crval1 + cdelt1 * (np.arange(naxis1) + 1 - crpix1)

        # Done with the file for now; will re-open in update mode later
        hdulist.close()
        
        # create transfer function for secondary standard
        SS_transfer_function, SS_sigma_transfer, linear_fit = create_transfer_function(SS_flux_data,SS_sigma_flux,SS_wave_axis,naxis1)
    
        transfer_function = SS_transfer_function
        sigma_transfer = SS_sigma_transfer

    # Require that all corrections are > 1, as expected for absorption
    transfer_function = np.maximum(transfer_function, 1.0)
    
    model_flux = linear_fit / transfer_function

    # Update the files to include telluric correction factor
    n_pix = len(transfer_function)
    data_1 = np.vstack((np.zeros(n_pix), np.ones(n_pix), np.zeros(n_pix)))
    data_2 = np.vstack((model_flux, transfer_function, sigma_transfer))
    for path, data_new in zip(frame_list, (data_1, data_2)):
        hdulist = pf.open(path, 'update', do_not_scale_image_data=True)
        hdulist[0].header['HGTELLUR'] = (HG_CHANGESET,'Hg changeset ID for telluric code')
        hdu = hdulist[hdu_name]
        # Arrange the data into a single array
        data = np.vstack((hdu.data[:4, :], data_new))
        # Save the data back into the FITS file
        hdu.data = data
        hdulist.close()

    return

def residual(A, SS_transfer_function, PS_transfer_function, PS_wave_axis):
    
    transfer_function_residual = 1./SS_transfer_function - 1./(PS_transfer_function ** A)

    return transfer_function_residual

def primary_standard_transfer_function(PS_spec_file):
    
    # import data
    PS_spec_data = pf.open(PS_spec_file)
    
    # build wavelength axis
    header = PS_spec_data[0].header
    crval1 = header['CRVAL1']
    cdelt1 = header['CDELT1']
    naxis1 = header['NAXIS1']
    crpix1 = header['CRPIX1']
    PS_wave_axis = crval1 + cdelt1 * (np.arange(naxis1) + 1 - crpix1)
    
    # extract PSS spectra and create a median spectrum that has be shape corrected
    PS_spec_list = []
    PS_noise_list = []
    for i in range(len(PS_spec_data)):
        if i == 0:
            pass
        else:
            shape = PS_spec_data[i].data[-1]
            spectrum = PS_spec_data[i].data[0]
            noise = PS_spec_data[i].data[2]
            PS_spec_corrected = spectrum*shape
            PS_spec_noise_corrected = noise*shape
            PS_spec_list.append(PS_spec_corrected)
            PS_noise_list.append(PS_spec_noise_corrected)
    PS_spec_array = np.asarray(PS_spec_list)
    PS_noise_array = np.asarray(PS_noise_list)
    PS_spec_median = np.median(PS_spec_array,axis=0)
    # This is very approximate. A better approach would be to use a
    # sigma-clipped mean, which has better noise properties
    PS_spec_noise = 1.25 * np.median(PS_noise_array,axis=0) / np.sqrt(len(PS_noise_list))
    
    # get transfer function for primary standard
    PS_transfer_function, PS_sigma_factor, linear_fit = create_transfer_function(PS_spec_median,PS_spec_noise,PS_wave_axis,naxis1)
    
    return PS_transfer_function, PS_sigma_factor, linear_fit, PS_wave_axis

def create_transfer_function(standard_spectrum,sigma,wave_axis,naxis1):

    # Select clean regions (no Halpha, no tellurics), to fit a straight line to
    clean_limits = [[6600, 6850],
                    [6970, 7130],
                    [7450, 7560],
                    [7770, 8100]]
    in_clean = np.zeros(naxis1, dtype=bool)
    for clean_limits_single in clean_limits:
        in_clean[(wave_axis >= clean_limits_single[0]) &
                (wave_axis <= clean_limits_single[1])] = True
    in_clean[~(np.isfinite(standard_spectrum))] = False
    wave_axis_cut = wave_axis[in_clean]
    standard_spectrum_cut = standard_spectrum[in_clean]
    sigma_cut = sigma[in_clean]
    # Clip out bad pixels
    # standard_spectrum_cut = median_filter(standard_spectrum_cut, 5)
    good = clip_spectrum(standard_spectrum_cut, sigma_cut, wave_axis_cut)
    standard_spectrum_cut = standard_spectrum_cut[good]
    sigma_cut = sigma_cut[good]
    wave_axis_cut = wave_axis_cut[good]
                
    # Fit linear slope to wavelength cut data
    p = np.polyfit(wave_axis_cut, standard_spectrum_cut, 1)
        
    #fit = np.polyval(p, wav_lin)
    fit = np.polyval(p, wave_axis)
        
    # Extract telluric features from original data
    telluric_limits = [[6850, 6960],
                        [7130, 7360],
                        [7560, 7770],
                        [8100, 8360]]
    # This is a copy-paste of earlier code - put into a subroutine
    in_telluric = np.zeros(naxis1, dtype=bool)
    for telluric_limits_single in telluric_limits:
        in_telluric[(wave_axis >= telluric_limits_single[0]) &
                    (wave_axis <= telluric_limits_single[1])] = True
        # If there are only a few non-telluric pixels at the end of the
        # spectrum, mark them as telluric anyway, in case the primary flux
        # calibration screwed them up.
        minimum_end_pixels = 50
        n_blue_end = np.sum(wave_axis < telluric_limits_single[0])
        if n_blue_end > 0 and n_blue_end < minimum_end_pixels:
            in_telluric[wave_axis < telluric_limits_single[0]] = True
        n_red_end = np.sum(wave_axis > telluric_limits_single[1])
        if n_red_end > 0 and n_red_end < minimum_end_pixels:
            in_telluric[wave_axis > telluric_limits_single[1]] = True
    standard_spectrum_telluric = fit.copy()
    standard_spectrum_telluric[in_telluric] = standard_spectrum[in_telluric]
                                   
    # Create the normalisation factor to apply to object data
    standard_spectrum_telluric_factor = fit / standard_spectrum_telluric
    # Require that we actually have a correction factor everywhere
    standard_spectrum_telluric_factor[~np.isfinite(standard_spectrum_telluric_factor)] = 1.0

    # Calculate the uncertainty in the correction
    sigma_factor = (sigma / standard_spectrum) * standard_spectrum_telluric_factor
    sigma_factor[~in_telluric] = 0.0
    
    # rename to "transfer_function"
    transfer_function = standard_spectrum_telluric_factor
    
    return transfer_function, sigma_factor, fit

def extract_secondary_standard(path_list,model_name='ref_centre_alpha_dist_circ_hdratm',n_trim=0,use_probe=None,hdu_name='FLUX_CALIBRATION'):
    """Identify and extract the secondary standard in a reduced RSS file."""
    
    # First check which hexabundle we need to look at
    star_match = identify_secondary_standard(path_list[0], use_probe=use_probe)
    # Read the observed data, in chunks
    chunked_data = read_chunked_data(path_list, star_match['probenum'], 
                                     sigma_clip=5)
    trim_chunked_data(chunked_data, n_trim)
    # Fit the PSF
    fixed_parameters = set_fixed_parameters(
        path_list, model_name, probenum=star_match['probenum'])
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
        observed_flux, observed_background, sigma_flux, sigma_background = \
            extract_total_flux(ifu, psf_parameters, model_name, clip=5.0)
        save_extracted_flux(path, observed_flux, observed_background,
                            sigma_flux, sigma_background,
                            star_match, psf_parameters, model_name,
                            good_psf, HG_CHANGESET, hdu_name=hdu_name)
    return

def identify_secondary_standard(path, use_probe=None):
    """Identify the secondary standard star in the given file."""
    fibre_table = pf.getdata(path, 'FIBRES_IFU')
    if use_probe is None:
        unique_names = np.unique(fibre_table['NAME'])
        for name in unique_names:
            if is_star(name):
                break
        else:
            raise ValueError('No star identified in file: ' + path)
        probenum = fibre_table['PROBENUM'][fibre_table['NAME'] == name]
        probenum = probenum[0]
    else:
        probenum = use_probe
        name = fibre_table['NAME'][(fibre_table['PROBENUM'] == probenum) & (fibre_table['TYPE'] == 'P')]
        name = name[0]
    star_match = {'name': name, 'probenum': probenum}
    return star_match
    
def is_star(name):
    """Return True if the name provided is for a star"""
    pilot_star = '([0-9]{15,})'
    gama_star = '(1000[0-9]{4})'
    abell_star = '(Abell[0-9]+_SS[0-9]+)'
    cluster_star = '(((999)|(888))[0-9]{9})'
    fornax_star1 = r'((?<!\-)\b(\d\d|\d)\b)'
    fornax_star2 = r'(100[0-9]{3})'
    fornax_star3 = r'(99[0-0]{3})'
    star_re = '|'.join((pilot_star, gama_star, abell_star, cluster_star,fornax_star1,fornax_star2,fornax_star3))
    return bool(re.match(star_re, name))

def apply_correction(path_in, path_out):
    """Apply an already-derived correction to the file."""
    hdulist = pf.open(path_in)
    telluric_function = hdulist['FLUX_CALIBRATION'].data[-2, :]
    sigma_factor = hdulist['FLUX_CALIBRATION'].data[-1, :]
    uncorrected_flux = hdulist[0].data.copy()
    hdulist[0].data *= telluric_function
    with warnings.catch_warnings():
        # We get lots of invalid value warnings arising because of divide by zero errors.
        warnings.filterwarnings('ignore', r'invalid value', RuntimeWarning)
        warnings.filterwarnings('ignore', r'divide by zero', RuntimeWarning)
        hdulist['VARIANCE'].data = hdulist[0].data**2 * (
            (sigma_factor / telluric_function)**2 +
            hdulist['VARIANCE'].data / uncorrected_flux**2)
    hdulist[0].header['HGTELLUR'] = (HG_CHANGESET, 
                                     'Hg changeset ID for telluric code')
    hdulist.writeto(path_out)
    return

### END OF FILE ###
