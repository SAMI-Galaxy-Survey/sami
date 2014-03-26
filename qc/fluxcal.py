"""Functions relating to quality control of flux calibration."""

from ..manager import Manager

import astropy.io.fits as pf

import os.path

def fluxcal_files(mngr):
    """
    Returns two dictionaries of flux calibration files, one for each CCD.
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
        for group in groups.items():
            combined = os.path.join(
                group[0].reduced_dir, 'TRANSFERcombined.fits')
            if os.path.exists(combined):
                result_ccd[combined] = group
        result.append(result_ccd)
    return tuple(result)
    