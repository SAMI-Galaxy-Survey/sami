"""Code for correcting for telluric absorption in SAMI data."""

from .fluxcal2 import read_chunked_data, set_fixed_parameters, fit_model_flux
from .fluxcal2 import insert_fixed_parameters, check_psf_parameters
from .fluxcal2 import extract_total_flux, save_extracted_flux, trim_chunked_data

from .. import utils
from ..utils.ifu import IFU

import astropy.io.fits as pf
import numpy as np
from scipy.ndimage.filters import median_filter
import scipy.optimize as optimize
import re

HG_CHANGESET = utils.hg_changeset(__file__)

# KEY:      SS = Secondary Standard, PS = Primary Standard

def derive_transfer_function(frame_list, PS_spec_file=None, use_PS=False, 
                             n_trim=0):
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
    extract_secondary_standard(frame_list, n_trim=n_trim)

    # Get data
    hdulist = pf.open(frame_list[1])
    hdu_name = 'FLUX_CALIBRATION'
    hdu = hdulist[hdu_name]
   
    # Load in SS flux data
    SS_flux_data_raw = hdu.data[0, :]
    SS_flux_background = hdu.data[1, :]
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
    SS_transfer_function = create_transfer_function(SS_flux_data,SS_wave_axis,naxis1)
    
    # if user defines, use a scaled primary standard telluric correction
    if use_PS:
        # get primary standard transfer function
        PS_transfer_function, PS_wave_axis = primary_standard_transfer_function(PS_spec_file)
        
        # find least squares fit on scalar
        A = 1.1
        best_scalar = optimize.leastsq(residual,A,args=(SS_transfer_function,PS_transfer_function,PS_wave_axis),full_output=1)
        
        PS_transfer_function_scaled = PS_transfer_function.copy() ** best_scalar[0][0]

        #py.figure()
        #py.plot(SS_transfer_function,'b')
        #py.plot(PS_transfer_function_scaled,'r')
        
        transfer_function = PS_transfer_function_scaled
    
    else:
        transfer_function = SS_transfer_function

    # Update the file to include telluric correction factor
    hdulist = pf.open(frame_list[1], 'update', do_not_scale_image_data=True)
    hdulist[0].header['HGTELLUR'] = (HG_CHANGESET,'Hg changeset ID for telluric code')
    hdu_name = 'FLUX_CALIBRATION'
    hdu = hdulist[hdu_name]
    data = hdu.data
    if len(data) == 2:
        # No previous telluric fit saved; append it to the data
        data = np.vstack((data, transfer_function))
    elif len(data) == 3:
        # Previous telluric fit to overwrite
        data[2, :] = transfer_function
    # Save the data back into the FITS file
    hdu.data = data
    hdulist.close()
    
    return

def residual(A, SS_transfer_function, PS_transfer_function, PS_wave_axis):
    
    transfer_function_residual = SS_transfer_function - PS_transfer_function ** A

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
    for i in xrange(len(PS_spec_data)):
        if i == 0:
            pass
        else:
            shape = PS_spec_data[i].data[2]
            spectrum = PS_spec_data[i].data[0]
            PS_spec_corrected = spectrum*shape
            PS_spec_list.append(PS_spec_corrected)
    PS_spec_array = np.asarray(PS_spec_list)
    PS_spec_median = np.median(PS_spec_array,axis=0)
    
    # get transfer function for primary standard
    PS_transfer_function = create_transfer_function(PS_spec_median,PS_wave_axis,naxis1)
    
    return PS_transfer_function, PS_wave_axis

def create_transfer_function(standard_spectrum,wave_axis,naxis1):

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
    # Mild smoothing so that one bad pixel doesn't screw up the linear fit
    standard_spectrum_cut = median_filter(standard_spectrum_cut, 5)
                
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
    # Require that all corrections are > 1, as expected for absorption
    standard_spectrum_telluric_factor = np.maximum(standard_spectrum_telluric_factor, 1.0)
    # Require that we actually have a correction factor everywhere
    standard_spectrum_telluric_factor[~np.isfinite(standard_spectrum_telluric_factor)] = 1.0
    
    # rename to "transfer_function"
    transfer_function = standard_spectrum_telluric_factor
    
    return transfer_function

def extract_secondary_standard(path_list,model_name='ref_centre_alpha_dist_circ_hdratm',n_trim=0):
    """Identify and extract the secondary standard in a reduced RSS file."""
    
    # First check which hexabundle we need to look at
    star_match = identify_secondary_standard(path_list[0])
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
        observed_flux, observed_background = extract_total_flux(
            ifu, psf_parameters, model_name, clip=5.0)
        save_extracted_flux(path, observed_flux, observed_background,
                            star_match, psf_parameters, model_name,
                            good_psf, HG_CHANGESET)
    return

def identify_secondary_standard(path):
    """Identify the secondary standard star in the given file."""
    fibre_table = pf.getdata(path, 'FIBRES_IFU')
    unique_names = np.unique(fibre_table['NAME'])
    pilot_star = '([0-9]{15,})'
    gama_star = '(1000[0-9]{4})'
    abell_star = '(Abell[0-9]+_SS[0-9]+)'
    cluster_star = '(((999)|(888))[0-9]{9})'
    star_re = '|'.join((pilot_star, gama_star, abell_star, cluster_star))
    for name in unique_names:
        if re.match(star_re, name):
            break
    else:
        raise ValueError('No star identified in file: ' + path)
    probenum = fibre_table['PROBENUM'][fibre_table['NAME'] == name]
    probenum = probenum[0]
    star_match = {'name': name, 'probenum': probenum}
    return star_match

def apply_correction(path_in, path_out):
    """Apply an already-derived correction to the file."""
    hdulist = pf.open(path_in)
    telluric_function = hdulist['FLUX_CALIBRATION'].data[2, :]
    hdulist[0].data *= telluric_function
    hdulist['VARIANCE'].data *= telluric_function**2
    hdulist[0].header['HGTELLUR'] = (HG_CHANGESET, 
                                     'Hg changeset ID for telluric code')
    hdulist.writeto(path_out)
    return

### END OF FILE ###
