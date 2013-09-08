from fluxcal2 import read_chunked_data, set_fixed_parameters, fit_model_flux
from fluxcal2 import insert_fixed_parameters, check_psf_parameters
from fluxcal2 import extract_total_flux, save_extracted_flux

from ..utils.ifu import IFU

import astropy.io.fits as pf
import numpy as np

import re

def extract_secondary_standard(path_list, 
                               model_name='ref_centre_alpha_dist_circ'):
    """Identify and extract the secondary standard in a reduced RSS file."""
    # First check which hexabundle we need to look at
    star_match = identify_secondary_standard(path_list[0])
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
        ### DO SOMETHING WITH THE OBSERVED FLUX HERE ###
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
