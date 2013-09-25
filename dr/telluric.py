"""Code for correcting for telluric absorption in SAMI data."""

from .fluxcal2 import read_chunked_data, set_fixed_parameters, fit_model_flux
from .fluxcal2 import insert_fixed_parameters, check_psf_parameters
from .fluxcal2 import extract_total_flux, save_extracted_flux

from .. import utils
from ..utils.ifu import IFU

import astropy.io.fits as pf
import numpy as np
import re

HG_CHANGESET = utils.hg_changeset(__file__)

# KEY:      SS = Secondary Standard

def correction_linear_fit(frame_list):
    """
    Finds the telluric correction factor to multiply object data by. The factor 
    as a function of wavelength is saved into the red frame under the extension 
    "FLUX_CALIBRATION" and has values that are mostly 1s apart from the telluric 
    regions.
    """
    
    # Always re-extract the secondary standard
    extract_secondary_standard(frame_list)

    # Get data
    hdulist = pf.open(frame_list[1])
    hdu_name = 'FLUX_CALIBRATION'
    hdu = hdulist[hdu_name]
    # # Check if there's already "FLUX_CALIBRATION" extension, and execute SS flux extraction if not
    # try:
    #     existing_index = hdulist.index_of(hdu_name)
    # except KeyError:
    #     # extract observed_flux and observed_background from secondary standard and save as FLUX_CALIBRATION extension in *fcal.fits file.
    #     extract_secondary_standard(frame_list)
    # else:
    #     pass
    # hdulist.close()

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

    # Select clean regions (no Halpha, no tellurics), to fit a straight line to
    clean_limits = [[6600, 6850],
                    [6970, 7130],
                    [7450, 7560],
                    [7770, 8100]]
    in_clean = np.zeros(naxis1, dtype=bool)
    for clean_limits_single in clean_limits:
        in_clean[(SS_wave_axis >= clean_limits_single[0]) & 
                 (SS_wave_axis <= clean_limits_single[1])] = True
    SS_wave_axis_cut = SS_wave_axis[in_clean]
    SS_flux_data_cut = SS_flux_data[in_clean]
            
    # Fit linear slope to wavelength cut data
    p = np.polyfit(SS_wave_axis_cut, SS_flux_data_cut, 1)
    # Sam's code defines a new wav_lin - why? Just use SS_wave_axis?
    #wav_lin = np.arange(np.min(SS_wave_axis), np.max(SS_wave_axis)+CDELT1, CDELT1)
    #fit = np.polyval(p, wav_lin)
    fit = np.polyval(p, SS_wave_axis)

    # Extract telluric features from original data
    telluric_limits = [[6850, 6960],
                       [7130, 7360],
                       [7560, 7770],
                       [8100, 8360]]
    # This is a copy-paste of earlier code - put into a subroutine
    in_telluric = np.zeros(naxis1, dtype=bool)
    for telluric_limits_single in telluric_limits:
        in_telluric[(SS_wave_axis >= telluric_limits_single[0]) & 
                    (SS_wave_axis <= telluric_limits_single[1])] = True
        # If there are only a few non-telluric pixels at the end of the
        # spectrum, mark them as telluric anyway, in case the primary flux
        # calibration screwed them up.
        minimum_end_pixels = 50
        n_blue_end = np.sum(SS_wave_axis < telluric_limits_single[0])
        if n_blue_end > 0 and n_blue_end < minimum_end_pixels:
            in_telluric[SS_wave_axis < telluric_limits_single[0]] = True
        n_red_end = np.sum(SS_wave_axis > telluric_limits_single[1])
        if n_red_end > 0 and n_red_end < minimum_end_pixels:
            in_telluric[SS_wave_axis > telluric_limits_single[1]] = True
    SS_flux_data_telluric = fit.copy()
    SS_flux_data_telluric[in_telluric] = SS_flux_data[in_telluric]

    # Create the normalisation factor to apply to object data
    SS_flux_data_telluric_factor = fit / SS_flux_data_telluric
    # Require that all corrections are > 1, as expected for absorption
    SS_flux_data_telluric_factor = np.maximum(
        SS_flux_data_telluric_factor, 1.0)
    # Require that we actually have a correction factor everywhere
    SS_flux_data_telluric_factor[
        ~np.isfinite(SS_flux_data_telluric_factor)] = 1.0

    # Update the file to include telluric correction factor
    hdulist = pf.open(frame_list[1], 'update', do_not_scale_image_data=True)
    hdulist[0].header['HGTELLUR'] = (HG_CHANGESET, 
                                     'Hg changeset ID for telluric code')
    hdu_name = 'FLUX_CALIBRATION'
    hdu = hdulist[hdu_name]
    data = hdu.data
    if len(data) == 2:
        # No previous telluric fit saved; append it to the data
        data = np.vstack((data, SS_flux_data_telluric_factor))
    elif len(data) == 3:
        # Previous telluric fit to overwrite
        data[2, :] = SS_flux_data_telluric_factor
    # Save the data back into the FITS file
    hdu.data = data
    hdulist.close()
    return

def extract_secondary_standard(path_list, 
                               model_name='ref_centre_alpha_dist_circ'):
    """Identify and extract the secondary standard in a reduced RSS file."""
    
    # First check which hexabundle we need to look at
    star_match = identify_secondary_standard(path_list[0])
    # Read the observed data, in chunks
    chunked_data = read_chunked_data(path_list, star_match['probenum'], 
                                     sigma_clip=5)
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
            ifu, psf_parameters, model_name, clip=5.0)
        save_extracted_flux(path, observed_flux, observed_background,
                            star_match, psf_parameters, model_name,
                            good_psf)
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
