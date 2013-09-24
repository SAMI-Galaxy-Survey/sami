from fluxcal2 import read_chunked_data, set_fixed_parameters, fit_model_flux
from fluxcal2 import insert_fixed_parameters, check_psf_parameters
from fluxcal2 import extract_total_flux, save_extracted_flux

from ..utils.ifu import IFU

import astropy.io.fits as pf
import numpy as np
import pylab as py
import re

# KEY:      SS = Secondary Standard

def correction_linear_fit(frame_list):
    """
    Finds the telluric correction factor to multiply object data by. The factor 
    as a function of wavelength is saved into the red frame under the extension 
    "FLUX_CALIBRATION" and has values that are mostly 1s apart from the telluric 
    regions.
    """
    
    # Get data
    hdulist = pf.open(frame_list[1])
    hdu_name = 'FLUX_CALIBRATION'
    # Check if there's already "FLUX_CALIBRATION" extension, and execute SS flux extraction if not
    try:
        existing_index = hdulist.index_of(hdu_name)
    except KeyError:
        # extract observed_flux and observed_background from secondary standard and save as FLUX_CALIBRATION extension in *fcal.fits file.
        extract_secondary_standard(frame_list)
    else:
        pass
    hdulist.close()

    # Load in SS flux data
    SS_flux_data_raw = pf.open(frame_list[1])['FLUX_CALIBRATION'].data[0,:]
    SS_flux_background = pf.open(frame_list[1])['FLUX_CALIBRATION'].data[1,:]
    # Replace NANs with nearest real value (maybe not correct method, but better than zeros)
    SS_flux_data = pf.open(frame_list[1])['FLUX_CALIBRATION'].data[0,:]
    ind = np.where(~np.isnan(SS_flux_data))[0]
    first, last = ind[0], ind[-1]
    SS_flux_data[:first] = SS_flux_data[first]
    SS_flux_data[last + 1:] = SS_flux_data[last]
    SS_flux_data[SS_flux_data < 0] = 0. # replacing negative fluxes with 0.0
    # Wavelength
    CRVAL1 = pf.open(frame_list[1])[0].header['CRVAL1']
    CDELT1 = pf.open(frame_list[1])[0].header['CDELT1']
    Nwave  = pf.open(frame_list[1])[0].header['NAXIS1']
    CRVAL1 = CRVAL1 - ((Nwave-1)/2)*CDELT1
    SS_wave_axis = CRVAL1 + CDELT1*np.arange(Nwave)
            
    # H_alpha + Tellurics + short wavelength oddities exclusions
    SS_wave_axis_cen_1 = SS_wave_axis[(SS_wave_axis >= 6600) & (SS_wave_axis <= 6850)]
    SS_wave_axis_cen_2 = SS_wave_axis[(SS_wave_axis >= 6970) & (SS_wave_axis <= 7140)]
    SS_wave_axis_cen_3 = SS_wave_axis[(SS_wave_axis >= 7450) & (SS_wave_axis <= 7560)]
    SS_wave_axis_cen_4 = SS_wave_axis[(SS_wave_axis >= 7770) & (SS_wave_axis <= 8100)]
    
    # Concatenate good wavelength cuts
    SS_wave_axis_cut = np.asarray(np.concatenate((SS_wave_axis_cen_1,SS_wave_axis_cen_2,SS_wave_axis_cen_3,SS_wave_axis_cen_4)))
    
    # Get flux data for resepctive wavelength cuts
    SS_flux_data_cut = []
    for wav in SS_wave_axis:
        if wav in SS_wave_axis_cut:
            SS_flux_data_cut.append(SS_flux_data[np.where(SS_wave_axis == wav)])

    # Ensure data is in array format
    SS_flux_data_cut = np.asarray(SS_flux_data_cut)
    SS_wave_axis_cut = np.asarray(SS_wave_axis_cut)

    # Fit linear slope to wavelength cut data
    p=np.polyfit(SS_wave_axis_cut, SS_flux_data_cut, 1)
    wav_lin=np.arange(np.min(SS_wave_axis), np.max(SS_wave_axis)+CDELT1, CDELT1)
    fit=np.polyval(p, wav_lin)

    # Extract telluric features from original data
    SS_wave_axis_tell_1 = SS_wave_axis[(SS_wave_axis >= 6850) & (SS_wave_axis <= 6960)]
    SS_wave_axis_tell_2 = SS_wave_axis[(SS_wave_axis >= 7130) & (SS_wave_axis <= 7360)]
    SS_wave_axis_tell_3 = SS_wave_axis[(SS_wave_axis >= 7560) & (SS_wave_axis <= 7770)]
    SS_wave_axis_tell_4 = SS_wave_axis[(SS_wave_axis >= 8100) & (SS_wave_axis <= 8360)]
    SS_wave_axis_tell = np.asarray(np.concatenate((SS_wave_axis_tell_1,SS_wave_axis_tell_2,SS_wave_axis_tell_3,SS_wave_axis_tell_4)))

    # Use original flux for telluric regions, but linear fit flux for non-telluric
    SS_wave_axis_telluric = []
    SS_flux_data_telluric = []
    for wav in SS_wave_axis:
        if wav in SS_wave_axis_tell:
            SS_flux_data_telluric.append(SS_flux_data[np.where(SS_wave_axis == wav)][0])
        else:
            SS_flux_data_linear = np.polyval(p,wav)
            SS_flux_data_telluric.append(SS_flux_data_linear[0])
        SS_wave_axis_telluric.append(wav)

    # Make sure data is in array format
    SS_wave_axis_telluric = np.asarray(SS_wave_axis_telluric)
    SS_flux_data_telluric = np.asarray(SS_flux_data_telluric)

    # Create the normalisation factor to apply to object data
    SS_wave_axis_telluric_factor = np.asarray(SS_wave_axis_telluric)
    SS_flux_data_telluric_factor = fit / SS_flux_data_telluric

    # Update the file to include telluric correction factor
    hdulist = pf.open(frame_list[1], 'update', do_not_scale_image_data=True)
    hdu_name = 'FLUX_CALIBRATION'
    hdu_header = pf.open(frame_list[1])['FLUX_CALIBRATION'].header
    # Check if there's already "FLUX_CALIBRATION" extension, and delete if so
    try:
        existing_index = hdulist.index_of(hdu_name)
    except KeyError:
        pass
    else:
        del hdulist[existing_index]
    # Compile data
    data = np.vstack((SS_flux_data_raw,SS_flux_background,SS_flux_data_telluric_factor))
    # Make the new HDU copying the header information from the original file
    new_hdu = pf.ImageHDU(data, name=hdu_name)
    new_hdu.header = hdu_header
    hdulist.append(new_hdu)
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

### END OF FILE ###