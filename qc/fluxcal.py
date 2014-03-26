"""Functions relating to quality control of flux calibration."""

from ..manager import Manager

import astropy.io.fits as pf
import numpy as np

import os.path

def fluxcal_files(mngr):
    """
    Return two dictionaries of flux calibration files, one for each CCD.
    The keys are the paths to the combined calibration files, and each item
    is a list of individual files that contributed.

    The input can be a Manager object or a path, from which a Manager will
    be created.
    """
    if isinstance(mngr, str):
        mngr = Manager(mngr)
    result = []
    for ccd in ['ccd_1', 'ccd_2']:
        result_ccd = {}
        groups = mngr.group_files_by(('date', 'field_id', 'ccd', 'name'),
            ndf_class='MFOBJECT', do_not_use=False,
            spectrophotometric=True, ccd=ccd)
        for group in groups.values():
            combined = os.path.join(
                group[0].reduced_dir, 'TRANSFERcombined.fits')
            if os.path.exists(combined):
                result_ccd[combined] = [f.reduced_path for f in group]
        result.append(result_ccd)
    return tuple(result)

def stability(mngr):
    """
    Return arrays of flux calibration stability, defined as the standard
    deviation of the end-to-end throughput as a function of wavelength.

    Two dictionaries are returned, one for each CCD. Each dictionary
    contains 'wavelength', 'std_combined', 'std_individual', 
    'mean_individual' and 'mean_combined'.
    """
    result = []
    for files_dict in fluxcal_files(mngr):
        n_pix = pf.getval(files_dict.keys()[0], 'NAXIS1')
        n_combined = len(files_dict)
        n_individual = sum(len(l) for l in files_dict.values())
        data_combined = np.zeros((n_combined, n_pix))
        data_individual = np.zeros((n_individual, n_pix))
        i_individual = 0
        for i_combined, path_combined in enumerate(files_dict.keys()):
            hdulist = pf.open(path_combined)
            data_combined[i_combined, :] = 1.0 / hdulist[0].data
            for hdu in hdulist[1:]:
                data_individual[i_individual, :] = 1.0 / hdu.data[2, :]
                i_individual += 1
        std_combined = np.std(data_combined, axis=0)
        std_individual = np.std(data_individual, axis=0)
        mean_combined = np.mean(data_combined, axis=0)
        mean_individual = np.mean(data_individual, axis=0)
        header = hdulist[0].header
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        crpix1 = header['CRPIX1']
        wavelength = crval1 + cdelt1 * (np.arange(n_pix) + 1 - crpix1)
        result.append({'wavelength': wavelength,
                       'std_combined': std_combined,
                       'std_individual': std_individual,
                       'mean_combined': mean_combined,
                       'mean_individual': mean_individual})
    return tuple(result)
