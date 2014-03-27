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

    The return value is a dictionary containing arrays of 'wavelength', 
    'std_combined', 'std_individual', 'mean_individual' and 'mean_combined'.
    The input 'data_combined' and 'data_individual' are included.
    Finally, the mean and standard deviation of the normalisation,
    'mean_norm_comined', 'mean_norm_individual', 'std_norm_combined' and
    'std_norm_individual', are included.
    """
    result = {}
    file_pairs = []
    individual_file_pairs = []
    file_dicts = fluxcal_files(mngr)
    n_individual = 0
    for path_1 in file_dicts[0]:
        path_2 = path_1.replace('ccd_1', 'ccd_2')
        if path_2 in file_dicts[1]:
            file_pairs.append((path_1, path_2))
            filename_1_list = [
                os.path.basename(f) for f in file_dicts[0][path_1]]
            filename_2_list = [
                os.path.basename(f) for f in file_dicts[1][path_2]]
            for i_filename_1, filename_1 in enumerate(filename_1_list):
                filename_2 = filename_1[:5] + '2' + filename_1[6:]
                if filename_2 in filename_2_list:
                    i_filename_2 = filename_2_list.index(filename_2)
                    individual_file_pairs.append((
                        file_dicts[0][path_1][i_filename_1],
                        file_dicts[1][path_2][i_filename_2]))
    n_pix_1 = pf.getval(file_pairs[0][0], 'NAXIS1')
    n_pix_2 = pf.getval(file_pairs[0][1], 'NAXIS1')
    n_combined = len(file_pairs)
    n_individual = len(individual_file_pairs)
    data_combined = np.zeros((n_pix_1+n_pix_2, n_combined))
    data_individual = np.zeros((n_pix_1+n_pix_2, n_individual))
    i_individual = 0
    for i_combined, (path_1, path_2) in enumerate(file_pairs):
        data_combined[:n_pix_1, i_combined] = 1.0 / pf.getdata(path_1)
        data_combined[n_pix_1:, i_combined] = 1.0 / pf.getdata(path_2)
    for i_individual, (path_1, path_2) in enumerate(individual_file_pairs):
        data_individual[:n_pix_1, i_individual] = 1.0 / pf.getdata(
            path_1, 'FLUX_CALIBRATION')[2, :]
        data_individual[n_pix_1:, i_individual] = 1.0 / pf.getdata(
            path_2, 'FLUX_CALIBRATION')[2, :]
    common = (
        np.sum(np.isfinite(data_combined), axis=1) + 
        np.sum(np.isfinite(data_individual), axis=1)
        ) == (n_combined + n_individual)
    norm_combined = np.mean(
        data_combined[common, :], axis=0)
    norm_individual = np.mean(
        data_individual[common, :], axis=0)
    data_norm_combined = data_combined / norm_combined
    data_norm_individual = data_individual / norm_individual
    result['std_combined'] = np.std(data_norm_combined, axis=1)
    result['std_individual'] = np.std(data_norm_individual, axis=1)
    result['mean_combined'] = np.mean(data_norm_combined, axis=1)
    result['mean_individual'] = np.mean(data_norm_individual, axis=1)
    result['std_norm_combined'] = np.std(norm_combined)
    result['std_norm_individual'] = np.std(norm_individual)
    result['mean_norm_combined'] = np.mean(norm_combined)
    result['mean_norm_individual'] = np.mean(norm_individual)
    result['data_combined'] = data_combined
    result['data_individual'] = data_individual
    wavelength = np.zeros(n_pix_1 + n_pix_2)
    for i, path in enumerate(file_pairs[0]):
        header = pf.getheader(path)
        crval1 = header['CRVAL1']
        cdelt1 = header['CDELT1']
        crpix1 = header['CRPIX1']
        if i == 0:
            start = 0
            finish = n_pix_1
            n_pix = n_pix_1
        else:
            start = n_pix_1
            finish = n_pix_1 + n_pix_2
            n_pix = n_pix_2
        wavelength[start:finish] = crval1 + cdelt1 * (
            np.arange(n_pix) + 1 - crpix1)
    result['wavelength'] = wavelength
    return result

