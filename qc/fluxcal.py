from __future__ import division
"""
Functions relating to quality control of flux calibration.

A lot of this functionality is now directly used by the data reduction
pipeline, particularly stellar_mags_frame_pair and stellar_mags_cube_pair
for measuring the magnitude of an observed star. get_sdss_stellar_mags is
used for downloading the correct magnitudes from SDSS. Occasionally,
calculate_mean_throughput and save_mean_throughput need to be run to
create a mean throughput file against which observations can be compared,
to check what the relative throughput on a given night is. Observations
from nights with very bad throughput (i.e. cloud) can then be rejected.
"""

from ..dr import fluxcal2
from ..dr.telluric import identify_secondary_standard
from ..utils import IFU
from ..utils.other import clip_spectrum

import astropy.io.fits as pf
import numpy as np
from scipy.ndimage.filters import median_filter
from scipy.optimize import leastsq

import os
from glob import glob
import multiprocessing

BANDS = 'ugriz'

def throughput(path, combined=True):
    """
    Return the instrument throughput, as measured from a standard star.

    `combined' should be set to True if the provided file is a
    TRANSFERcombined.fits file, False if it's an individual observation.

    The returned spectrum is a fractional throughput for the telescope,
    fibres and spectrograph (expected atmosphere was already removed).
    """
    h = 6.626E-27 #erg.s
    c = 3E18  #speed of light in A/s [1m/s=1E10A/s]
    area = 0.847*(np.pi*(389.3/2)**2)  #=100818cm^2 Telescope collecting area
    hdulist = pf.open(path)
    # data is the value of transfer function corrected for expected atm for
    # each pixel, i.e. accounting for airmass
    if combined:
        data = hdulist[0].data
    else:
        data = hdulist['FLUX_CALIBRATION'].data[-1, :]
    hdr = hdulist[0].header
    # Wavelength range
    wavelength = get_coords(hdr, 1)
    delta_wl = hdr['CDELT1']
    gain = hdr['RO_GAIN']  # photons/e-
    # hc/lambda h=planck const c=speed of light in A/s ergs.s.A/A.s=ergs
    hcl = h * c / wavelength
    thput = (1/data) * (1E16) * gain * hcl / (area*delta_wl)
    return thput

def save_mean_throughput(path_out, mean_thput, thput, filename_list,
                         good_list, detector, date_start=None,
                         date_finish=None):
    """
    Save the mean instrument throughput to a FITS file.

    Also included in the FITS file are all the individual throughputs that
    went into the mean.

    Inputs:
    `path_out' - path to save to
    `mean_throughput' - 1d array
    `throughput' - n_spec X n_pix array of individual measurements
    `filename_list' - n_spec list of filenames
    `good_list' - n_spec boolean list of files included in the mean
    `detector' - name of the CCD, e.g. E2V2A

    Optional:
    `date_start' - first date on which this CCD was used
    `date_finish' - last date on which this CCD was used
    
    Dates should be recorded as fractional years, e.g. 2014.461
    If not provided then a semi-infinite or infinite range is allowed.
    """
    filename_length = np.max([len(filename) for filename in filename_list])
    hdulist = pf.HDUList(
        [pf.PrimaryHDU(mean_thput),
         pf.ImageHDU(thput, name='INPUT_DATA'),
         pf.BinTableHDU.from_columns(
             [pf.Column(name='filename', format='{}A'.format(filename_length),
                        array=filename_list),
              pf.Column(name='used', format='L', array=good_list)],
              name='INPUTS')])
    header = hdulist[0].header
    header['DETECTOR'] = (detector, 'Detector name')
    if date_start is not None:
        header['DATESTRT'] = (
            date_start, 'Do not use for data before this date')
    if date_finish is not None:
        header['DATEFNSH'] = (
            date_finish, 'Do not use for data after this date')
    hdulist.writeto(path_out)
    return

def calculate_mean_throughput(path_out, mngr_list, detector, date_start=None,
                              date_finish=None, reject=0.2):
    """
    Calculate and save the mean throughput for a given CCD.

    Inputs:
    `path_out' - path to save to
    `mngr_list' - a list of Manager objects from which to draw the data
    `detector' - name of the CCD, e.g. E2V2A

    Optional:
    `date_start' - first date on which this CCD was used
    `date_finish' - last date on which this CCD was used
    `reject' - fractional deviation from median throughput at which to
               reject a spectrum

    Dates should be recorded as fractional years, e.g. 2014.461
    If not provided then a semi-infinite or infinite range is allowed.

    The following Python commands were used to make the most recent set of
    mean throughput files (Feb 2016):

    import sami
    from glob import glob
    root_list = glob('/import/opus1/jallen/20*')
    mngr_list = [sami.manager.Manager(root) for root in root_list]
    sami.qc.fluxcal.calculate_mean_throughput('mean_throughput_E2V2.fits', mngr_list, 'E2V2')
    sami.qc.fluxcal.calculate_mean_throughput('mean_throughput_E2V2A.fits', mngr_list, 'E2V2A')
    sami.qc.fluxcal.calculate_mean_throughput('mean_throughput_E2V3_dirty.fits', mngr_list, 'E2V3', date_finish=2014.208)
    sami.qc.fluxcal.calculate_mean_throughput('mean_throughput_E2V3_clean.fits', mngr_list, 'E2V3', date_start=2014.208)
    sami.qc.fluxcal.calculate_mean_throughput('mean_throughput_E2V3A.fits', mngr_list, 'E2V3A')
    """
    thput_list = []
    filename_list = []
    for mngr in mngr_list:
        for fits in mngr.files(ndf_class='MFOBJECT', spectrophotometric=True,
                               do_not_use=False):
            if ((fits.epoch < date_start and date_start is not None) or 
                (fits.epoch > date_finish and date_finish is not None)):
                continue
            if pf.getval(fits.reduced_path, 'DETECTOR') != detector:
                continue
            try:
                thput = throughput(fits.reduced_path, combined=False)
            except KeyError:
                # Couldn't find the data for some reason
                continue
            thput_list.append(thput)
            filename_list.append(fits.filename)
    thput = np.array(thput_list)
    deviation = 1.0 - np.median(thput / np.median(thput, axis=0), axis=1)
    good = np.abs(deviation) < reject
    mean_thput = np.mean(thput[good, :], axis=0)
    save_mean_throughput(path_out, mean_thput, thput, filename_list, good,
                         detector, date_start=date_start,
                         date_finish=date_finish)

def fluxcal_files(mngr):
    """
    Return two dictionaries of flux calibration files, one for each CCD.
    The keys are the paths to the combined calibration files, and each item
    is a list of individual files that contributed.

    The input can be a Manager object or 
    a list of managers, in which case the results are
    concatenated.
    """
    if isinstance(mngr, list):
        mngr_list = mngr
    else:
        mngr_list = [mngr]
    ccd_list = ['ccd_1', 'ccd_2']
    result = [{} for ccd in ccd_list]
    for mngr in mngr_list:
        for index, ccd in enumerate(ccd_list):
            groups = mngr.group_files_by(('date', 'field_id', 'ccd', 'name'),
                ndf_class='MFOBJECT', do_not_use=False,
                spectrophotometric=True, ccd=ccd)
            for group in groups.values():
                combined = os.path.join(
                    group[0].reduced_dir, 'TRANSFERcombined.fits')
                if os.path.exists(combined):
                    result[index][combined] = [f.reduced_path for f in group]
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
            path_1, 'FLUX_CALIBRATION')[-1, :]
        data_individual[n_pix_1:, i_individual] = 1.0 / pf.getdata(
            path_2, 'FLUX_CALIBRATION')[-1, :]
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


# def stellar_colours(mngr):
#     """
#     Return stellar colours as measured by SAMI (via a template fit) compared
#     to the SDSS imaging.
#     """
#     file_pair_list = list_star_files(mngr)
#     model_catalogue = read_stellar_models()
#     model_list = [fit_template(file_pair, model_catalogue)[-1]
#                   for file_pair in file_pair_list]
#     observed_colours = [measure_colour(model, model_catalogue['wavelength']) 
#                         for model in model_list]
#     return file_pair_list, observed_colours
def get_sdss_stellar_mags(mngr, catalogue=None, automatic=False):
    """Get magnitudes for stars from SDSS, possibly with a little help from the user.

    If automatic=True, then it will try to download the data directly, and will
    return it as a string.

    If automatic=False, or the automatic downloading fails, then it will prompt the
    user to go and download the file directly, in which case it simply returns True.

    If the stars are already in the catalogue, then it returns False.
    """
    name_list, coords_list = get_missing_stars(mngr, catalogue=catalogue)
    if not name_list:
        # Already had all the stars required
        print('All magnitudes already in the catalogue.')
        return False

    url = 'https://skyserver.sdss.org/dr10/en/tools/crossid/crossid.asp'

    query_list = ""
    query_list += "name ra dec\n"
    for name, coords in zip(name_list, coords_list):
        query_list += "{}, {}, {}\n".format(name, coords[0], coords[1])

    sql = """SELECT
   p.objID, p.ra, p.dec,
   dbo.fPhotoTypeN(p.type) as type,
   p.psfMag_u, p.psfMagErr_u, p.psfMag_g, p.psfMagErr_g, p.psfMag_r,
   p.psfMagErr_r, p.psfMag_i, p.psfMagErr_i, p.psfMag_z, p.psfMagErr_z
FROM #x x, #upload u, PhotoTag p
WHERE u.up_id = x.up_id and x.objID=p.objID
ORDER BY x.up_id"""

    if automatic:
        try:
            import requests
        except ImportError:
            print("Install the python 'requests' library to allow automatic downloading.")
            automatic = False
    if automatic:
        post_data = {'searchType': "photo",
            'photoScope': "nearPrim",
            'photoUpType': "ra-dec",
            'radius': "0.5",
            'firstcol': "1",
            'paste': query_list,
		    'uquery': sql,
	        'format': "csv" }
        r = requests.post(url, data=post_data, files={'upload':("", "")})
        if r.status_code != 200:
            raise IOError("Error connectiong to SDSS servers. Try with 'automatic=False'")
        return r.text
    else:
        print('Go to:')
        print()
        print(url)
        print()
        print('Copy-paste the following into the upload list box:')
        print()
        print(query_list)
        print()
        print('And the following into the SQL query box:')
        print()
        print(sql)
        print()
        print('Change output format to CSV, then hit submit.')
        return True

def get_missing_stars(mngr, catalogue=None):
    """Return lists of observed stars missing from the catalogue."""
    name_list = []
    coords_list = []
    for fits_list in mngr.group_files_by(
            'field_id', ndf_class='MFOBJECT', reduced=True).values():
        fits = fits_list[0]
        path = fits.reduced_path
        try:
            star = identify_secondary_standard(path)
        except ValueError:
            # A frame didn't have a recognised star. Just skip it.
            continue
        if catalogue and star['name'] in catalogue:
            continue
        fibres = pf.getdata(path, 'FIBRES_IFU')
        fibre = fibres[fibres['NAME'] == star['name']][0]
        name_list.append(star['name'])
        coords_list.append((fibre['GRP_MRA'], fibre['GRP_MDEC']))
    return name_list, coords_list

def get_sdss_galaxy_mags(galaxy_file_pair_list):
    """Get magnitudes for galaxies from SDSS, with a little help from the user."""
    name_list = []
    coords_list = []
    for file_pair in galaxy_file_pair_list:
        header = pf.getheader(file_pair[0])
        name = header['NAME']
        if name not in name_list:
            name_list.append(name)
            coords_list.append((header['CATARA'], header['CATADEC']))
    print('Go to:')
    print()
    print('http://cas.sdss.org/dr7/en/tools/crossid/crossid.asp')
    print()
    print('Copy-paste the following into the upload list box:')
    print()
    for name, coords in zip(name_list, coords_list):
        print(name, coords[0], coords[1])
    print()
    print('And the following into the SQL query box:')
    print()
    print("""SELECT 
   p.objID, p.ra, p.dec,
   dbo.fPhotoTypeN(p.type) as type,
   p.modelMag_u, p.modelMagErr_u, p.modelMag_g, p.modelMagErr_g, p.modelMag_r,
   p.modelMagErr_r, p.modelMag_i, p.modelMagErr_i, p.modelMag_z, p.modelMagErr_z 
FROM #x x, #upload u, PhotoTag p
WHERE u.up_id = x.up_id and x.objID=p.objID 
ORDER BY x.up_id""")
    print()
    print('Change output format to CSV, then hit submit.')
    print('Put the result somewhere safe.')
    return

def stellar_mags_cubes(file_pair_list, n_cpu=None, verbose=True):
    """
    Return stellar magnitudes as measured by SAMI (via interpolation),
    for the datacubes.
    """
    if n_cpu == 1:
        _map = map
    else:
        pool = multiprocessing.Pool(n_cpu)
        _map = pool.map
    if verbose:
        print('Measuring magnitudes for all files')
    mag_cube = _map(stellar_mags_cube_pair, file_pair_list)
    if n_cpu != 1:
        pool.close()
    return mag_cube

def read_stellar_mags_cubes(file_pair_list, verbose=True):
    """
    Return pre-measured stellar magnitudes from SAMI datacubes.
    """
    mag_cube = []
    for file_pair in file_pair_list:
        if verbose:
            print('Reading magnitudes from', os.path.basename(file_pair[0]))
        hdulist = pf.open(file_pair[0])
        magg = hdulist[0].header['MAGG']
        # Early versions wrote MAGR to the wrong header, so have to check both
        try:
            magr = hdulist[0].header['MAGR']
        except KeyError:
            magr = hdulist[1].header['MAGR']
        mag_cube.append((magg, magr))
    return mag_cube

def read_stellar_mags_frames(frame_pair_list_list, bands=('g', 'r'),
                             verbose=True):
    """
    Return stellar magnitudes as measured by SAMI (via interpolation),
    for the input files.
    """
    mag_frame = []
    for frame_pair_list in frame_pair_list_list:
        mag_frame.append([])
        for frame_pair in frame_pair_list:
            if verbose:
                print('Reading magnitudes from', os.path.basename(frame_pair[0]))
            flux, noise, wavelength = read_stellar_spectrum(frame_pair)
            mags = measure_mags(flux, noise, wavelength)
            mag_frame[-1].append((mags[bands[0]], mags[bands[1]]))
    return mag_frame

# def stellar_mags(mngr, n_cpu=1):
#     """
#     Return stellar magnitudes as measured by SAMI (via interpolation), for
#     the datacubes and the input files.
#     """
#     if n_cpu == 1:
#         _map = map
#     else:
#         pool = multiprocessing.Pool(n_cpu)
#         _map = pool.map
#     file_pair_list, frame_pair_list_list = list_star_files(mngr)
#     mag_cube = _map(stellar_mags_cube_pair, file_pair_list)
#     mag_frame = []
#     for file_pair, frame_pair_list in zip(
#             file_pair_list, frame_pair_list_list):
#         # NOTE: measure_mags() return value has changed, so the following
#         # code will no longer work
#         mag_frame.append([])
#         for frame_pair in frame_pair_list:
#             flux, noise, wavelength = read_stellar_spectrum(frame_pair)
#             mag_frame[-1].append(measure_mags(flux, noise, wavelength))
#     if n_cpu != 1:
#         pool.close()
#     return file_pair_list, frame_pair_list_list, mag_cube, mag_frame
    
def stellar_mags_cube_pair(file_pair, sum_cubes=False, save=False):
    """Return unscaled stellar mags for a single pair of datacubes."""
    if sum_cubes:
        flux, noise, wavelength = (
            extract_galaxy_spectrum(file_pair))
    else:
        flux, noise, back, noise_back, wavelength, psf_params, sigma_params = (
            extract_stellar_spectrum(file_pair, background=True,
                                     variable_psf=True))
    try:
        old_scale = pf.getval(file_pair[0], 'RESCALE')
    except KeyError:
        old_scale = 1
    flux /= old_scale
    noise /= old_scale
    mags = measure_mags(flux, noise, wavelength)
    if save:
        for path in file_pair:
            hdulist = pf.open(path, 'update')
            for band in BANDS:
                if np.isfinite(mags[band]):
                    mag = mags[band]
                else:
                    mag = -99999
                hdulist[0].header['MAG'+band.upper()] = (
                    mag, band+' mag before scaling')    
            if not sum_cubes:
                # CDELT1 gives the pixel scale in degrees; convert to arcsec
                alpha = (3600.0 * np.abs(hdulist[0].header['CDELT1'])
                         * psf_params[0])
                beta = psf_params[1]
                fwhm = alpha * 2.0 * np.sqrt(2.0**(1.0/beta) - 1.0)
                hdulist[0].header['PSFALPHA'] = (
                    alpha, 'PSF parameter: alpha')
                hdulist[0].header['PSFBETA'] = (
                    beta, 'PSF parameter: beta')
                hdulist[0].header['PSFFWHM'] = (
                    fwhm, 'FWHM (arcsec) of PSF')
            hdulist.flush()
            hdulist.close()
    return mags

def stellar_mags_scatter_cube_pair(file_pair, min_relative_flux=0.5, save=False):
    """Return the scatter in stellar colours within a star datacube pair."""
    hdulist_pair = [pf.open(path, 'update') for path in file_pair]
    flux = np.vstack(
        [hdulist[0].data for hdulist in hdulist_pair])
    noise = np.sqrt(np.vstack(
        [hdulist['VARIANCE'].data for hdulist in hdulist_pair]))
    wavelength = np.hstack(
        [get_coords(hdulist[0].header, 3) for hdulist in hdulist_pair])
    smoothed_flux = flux.copy()
    smoothed_flux[~np.isfinite(smoothed_flux)] = 0.0
    smoothed_flux = median_filter(smoothed_flux, (201, 1, 1))
    image = np.sum(smoothed_flux, 0)
    keep = (image >= (min_relative_flux * np.max(image)))
    flux = flux[:, keep]
    noise = noise[:, keep]
    mags = []
    for flux_i, noise_i in zip(flux.T, noise.T):
        mags_i = measure_mags(flux_i, noise_i, wavelength)
        mags.append([mags_i['g'], mags_i['r']])
    mags = np.array(mags)
    colour = mags[:, 0] - mags[:, 1]
    scatter = np.std(colour)
    if save:
        for hdulist in hdulist_pair:
            hdulist[0].header['COLORSTD'] = (
                scatter, 'Scatter in g-r within cubes')
            hdulist.flush()
    for hdulist in hdulist_pair:
        hdulist.close()
    return scatter

def stellar_mags_frame_pair(file_pair, save=False):
    """Return stellar mags for a single pair of flux calibrated RSS files."""
    # Unlike in stellar_mags_cube_pair the old scaling is not taken into
    # account, because it is never applied to the FLUX_CALIBRATION HDU
    # from which these measurements are made.
    flux, noise, wavelength = read_stellar_spectrum(file_pair)
    mags = measure_mags(flux, noise, wavelength)
    if save:
        for path in file_pair:
            hdulist = pf.open(path, 'update')
            for band in BANDS:
                if np.isfinite(mags[band]):
                    mag = mags[band]
                else:
                    mag = -99999
                hdulist['FLUX_CALIBRATION'].header['MAG'+band.upper()] = (
                    mag, band+' mag before scaling')    
            hdulist.flush()
            hdulist.close()
    return mags

def list_star_files(mngr, gzip=True, verbose=True):
    """
    Return a list of tuples of paths to star datacubes, blue and red,
    as well as a list of lists of tuples of paths to individual frames.
    """
    if isinstance(mngr, list):
        mngr_list = mngr
    else:
        mngr_list = [mngr]
    result = []
    frame = []
    for mngr in mngr_list:
        if gzip:
            pattern = os.path.join(
                mngr.abs_root, 'cubed', '*', '*blue*.fits.gz')
        else:
            pattern = os.path.join(
                mngr.abs_root, 'cubed', '*', '*blue*.fits')
        blue_list = glob(pattern)
        for blue_path in blue_list:
            red_path = red_cube_path(blue_path)
            if os.path.exists(red_path):
                blue_header = pf.getheader(blue_path)
                if blue_header['NAME'] == blue_header['STDNAME']:
                    if verbose:
                        print('Found star file:', os.path.basename(blue_path))
                    result.append((blue_path, red_path))
                    i = 0
                    frame.append([])
                    while True:
                        i += 1
                        try:
                            blue_filename = blue_header['RSS_FILE ' + str(i)]
                        except KeyError:
                            break
                        red_filename = (blue_filename[:5] + '2' + 
                                        blue_filename[6:10] + 'sci.fits')
                        try:
                            blue_frame_path = glob(
                                mngr.abs_root+'/reduced/*/*/*/*/*/'+
                                blue_filename)[0]
                            red_frame_path = glob(
                                mngr.abs_root+'/reduced/*/*/*/*/*/'+
                                red_filename)[0]
                            frame[-1].append((blue_frame_path, red_frame_path))
                        except KeyError:
                            break
    return result, frame
    
def list_galaxy_files(mngr, gzip=True, verbose=True):
    """
    Return a list of tuples of paths to galaxy datacubes, blue and red,
    as well as a list of lists of tuples of paths to individual frames.
    """
    if isinstance(mngr, list):
        mngr_list = mngr
    else:
        mngr_list = [mngr]
    result = []
    frame = []
    for mngr in mngr_list:
        if gzip:
            pattern = os.path.join(
                mngr.abs_root, 'cubed', '*', '*blue*.fits.gz')
        else:
            pattern = os.path.join(
                mngr.abs_root, 'cubed', '*', '*blue*.fits')
        blue_list = glob(pattern)
        for blue_path in blue_list:
            red_path = red_cube_path(blue_path)
            if os.path.exists(red_path):
                blue_header = pf.getheader(blue_path)
                if blue_header['NAME'] != blue_header['STDNAME']:
                    if verbose:
                        print('Found galaxy file:', os.path.basename(blue_path))
                    result.append((blue_path, red_path))
                    i = 0
                    frame.append([])
                    while True:
                        i += 1
                        try:
                            blue_filename = blue_header['RSS_FILE ' + str(i)]
                        except KeyError:
                            break
                        red_filename = (blue_filename[:5] + '2' + 
                                        blue_filename[6:10] + 'sci.fits')
                        try:
                            blue_frame_path = glob(
                                mngr.abs_root+'/reduced/*/*/*/*/*/'+
                                blue_filename)[0]
                            red_frame_path = glob(
                                mngr.abs_root+'/reduced/*/*/*/*/*/'+
                                red_filename)[0]
                            frame[-1].append((blue_frame_path, red_frame_path))
                        except KeyError:
                            break
    return result, frame

def red_cube_path(blue_path):
    """Return the corresponding red cube path matched to a blue cube path."""
    start = blue_path.rfind('blue')
    red_path = blue_path[:start] + 'red' + blue_path[start+4:]
    return red_path

def fit_template(file_pair, model_catalogue):
    """Fit a stellar template to a pair of datacubes and infer the g-r."""
    flux, noise, wavelength = extract_stellar_spectrum(file_pair)
    flux, noise, count = fluxcal2.rebin_flux_noise(
        model_catalogue['wavelength'], wavelength, flux, noise)
    wavelength = model_catalogue['wavelength']
    good = np.isfinite(flux) & np.isfinite(noise)
    best_chisq = np.inf
    best_scale = np.nan
    best_model = np.nan
    for model_flux in model_catalogue['flux']:
        scale = (np.sum(flux[good] * model_flux[good] / noise[good]**2) / 
                 np.sum(model_flux[good]**2 / noise[good]**2))
        if not np.isfinite(scale):
            # This happens if the model flux is all zeros
            continue
        chisq = np.sum(((flux[good] - scale * model_flux[good]) / 
                        noise[good])**2)
        if chisq < best_chisq:
            best_chisq = chisq
            best_scale = scale
            best_model = model_flux
    return flux, noise, wavelength, best_chisq, best_scale, best_model

class IFUDuck(object):
    def __init__(self, file_pair):
        hdulist_0 = pf.open(file_pair[0])
        hdulist_1 = pf.open(file_pair[1])
        self.data = np.vstack((hdulist_0[0].data, hdulist_1[0].data))
        self.var = np.vstack((hdulist_0['VARIANCE'].data,
                              hdulist_1['VARIANCE'].data))
        self.dim = self.data.shape
        self.data.shape = (self.dim[0], self.dim[1]*self.dim[2])
        self.data = self.data.T
        self.var.shape = (self.dim[0], self.dim[1]*self.dim[2])
        self.var = self.var.T
        self.lambda_range = np.hstack((get_coords(hdulist_0[0].header, 3),
                                       get_coords(hdulist_1[0].header, 3)))
        self.naxis1 = len(self.lambda_range)
        hdulist_0.close()
        hdulist_1.close()
        del hdulist_0
        del hdulist_1

def extract_stellar_spectrum(file_pair, variable_psf=False, elliptical=False,
                             background=False):
    """Return the spectrum of a star, assumed to be at the centre."""
    if variable_psf:
        flux_chunked, variance_chunked, wavelength_chunked = (
            fluxcal2.chunk_data(IFUDuck(file_pair)))
        flux_chunked = flux_chunked.T
        variance_chunked = variance_chunked.T
        n_pix = int(np.sqrt(flux_chunked.shape[1]))
        n_wave = flux_chunked.shape[0]
        flux_chunked.shape = (n_wave, n_pix, n_pix)
        variance_chunked.shape = (n_wave, n_pix, n_pix)
        noise_chunked = np.sqrt(variance_chunked)
        psf_params, sigma_params = fit_moffat_to_chunks(
            flux_chunked, variance_chunked, wavelength_chunked,
            elliptical=elliptical)
        hdulist_pair = [pf.open(path) for path in file_pair]
        flux_cube = np.vstack([
            hdulist[0].data for hdulist in hdulist_pair])
        noise_cube = np.sqrt(np.vstack([
            hdulist['VARIANCE'].data for hdulist in hdulist_pair]))
        wavelength = np.hstack([
            get_coords(hdulist[0].header, 3) for hdulist in hdulist_pair])
    else:
        hdulist_pair = [pf.open(path) for path in file_pair]
        flux_cube = np.vstack([
            hdulist[0].data for hdulist in hdulist_pair])
        noise_cube = np.sqrt(np.vstack([
            hdulist['VARIANCE'].data for hdulist in hdulist_pair]))
        wavelength = np.hstack([
            get_coords(hdulist[0].header, 3) for hdulist in hdulist_pair])
        flux_image, noise_image, wavelength_image = collapse_cube(
            flux_cube, noise_cube, wavelength)
        psf_params, sigma_params = fit_moffat_to_image(
            flux_image, noise_image, elliptical=elliptical,
            background=background)
    flux = np.zeros(len(wavelength))
    noise = np.zeros(len(wavelength))
    if background:
        back = np.zeros(len(wavelength))
        noise_back = np.zeros(len(wavelength))
    print('Getting slice-by-slice flux from', os.path.basename(file_pair[0]))
    for i_pix, (image_slice, noise_slice, wavelength_slice) in enumerate(zip(
            flux_cube, noise_cube, wavelength)):
        if variable_psf:
            psf_params_i = psf_params_at_slice(
                psf_params, i_pix, wavelength_slice, copy_intensity=False)
            if background:
                psf_params_i = np.hstack((psf_params_i, 0.0))
        else:
            psf_params_i = psf_params
        result = scale_moffat_to_image(
            image_slice, noise_slice, psf_params_i, elliptical=elliptical,
            background=background)
        if background:
            flux[i_pix], noise[i_pix], back[i_pix], noise_back[i_pix] = result
        else:
            flux[i_pix], noise[i_pix] = result
    if background:
        return (flux, noise, back, noise_back, wavelength, psf_params,
                sigma_params)
    else:
        return flux, noise, wavelength, psf_params, sigma_params

def extract_galaxy_spectrum(file_pair):
    """Return the spectrum of a galaxy, assumed to cover the IFU."""
    return sum_spectrum_from_cube(file_pair, 7.0)

def sum_spectrum_from_cube(file_pair, radius):
    """Return the summed spectrum from spaxels in the centre of a cube."""
    # Replace the hard-coded numbers with something smarter
    hdulist_pair = [pf.open(path) for path in file_pair]
    n_pix = hdulist_pair[0][0].header['NAXIS1']
    # Pixel scale in arcseconds
    pix_scale = 3600.0 * np.abs(hdulist_pair[0][0].header['CDELT1'])
    # Construct coordinates along a single axis
    x_tmp = pix_scale * (np.arange(n_pix) - (n_pix - 1)/2.0)
    # Expand that to a grid of coordinate values
    x, y = np.meshgrid(x_tmp, x_tmp)
    weight_map = np.nansum(np.array((hdulist_pair[0]['WEIGHT'].data,
                                     hdulist_pair[1]['WEIGHT'].data)), (0, 1))
    cen_x = np.nansum(x * weight_map) / np.nansum(weight_map)
    cen_y = np.nansum(y * weight_map) / np.nansum(weight_map)
    keep_x, keep_y = np.where((x - cen_x)**2 + (y - cen_y)**2 < radius**2)
    flux_cube = np.vstack((hdulist_pair[0][0].data, hdulist_pair[1][0].data))
    variance_cube = np.vstack((hdulist_pair[0]['VARIANCE'].data,
                               hdulist_pair[1]['VARIANCE'].data))
    # flux_cube = np.vstack((pf.getdata(file_pair[0]), pf.getdata(file_pair[1])))
    # variance_cube = np.vstack((pf.getdata(file_pair[0], 'VARIANCE'), 
    #                            pf.getdata(file_pair[1], 'VARIANCE')))
    flux = np.nansum(flux_cube[:, keep_x, keep_y], axis=1)
    # Doesn't include co-variance - Nic will provide code
    noise = np.sqrt(np.nansum(variance_cube[:, keep_x, keep_y], axis=1))
    # Fudge for co-variance
    noise *= 2.0
    wavelength = np.hstack((get_coords(hdulist_pair[0][0].header, 3),
                            get_coords(hdulist_pair[1][0].header, 3)))
    # wavelength = np.hstack((get_coords(pf.getheader(file_pair[0]), 3),
    #                         get_coords(pf.getheader(file_pair[1]), 3)))
    return flux, noise, wavelength

def read_stellar_spectrum(file_pair):
    """Read and return the measured spectrum of a star from a single frame."""
    flux = []
    noise = []
    wavelength = []
    for path in file_pair:
        hdulist = pf.open(path)
        flux.append(hdulist['FLUX_CALIBRATION'].data[0, :])
        noise.append(hdulist['FLUX_CALIBRATION'].data[2, :])
        header = hdulist[0].header
        wavelength.append(get_coords(header, 1))
    flux = np.hstack(flux)
    noise = np.hstack(noise)
    wavelength = np.hstack(wavelength)
    return flux, noise, wavelength

def read_galaxy_spectrum(file_pair, name):
    """Read and return the summed spectrum of a galaxy from a single frame."""
    # Just sums over the fibres, which misses ~25% of the light
    flux = []
    noise = []
    wavelength = []
    for path in file_pair:
        ifu = IFU(path, name)
        flux.append(np.nansum(ifu.data, 0))
        noise.append(np.sqrt(np.nansum(ifu.var, 0)))
        wavelength.append(ifu.lambda_range)
    flux = np.hstack(flux)
    noise = np.hstack(noise)
    wavelength = np.hstack(wavelength)
    return flux, noise, wavelength

def get_coords(header, axis):
    """Return coordinates for a given axis from a header."""
    axis_str = str(axis)
    naxis = header['NAXIS' + axis_str]
    crpix = header['CRPIX' + axis_str]
    cdelt = header['CDELT' + axis_str]
    crval = header['CRVAL' + axis_str]
    coords = crval + cdelt * (np.arange(naxis) + 1.0 - crpix)
    return coords

def rebin_spectrum(old_flux, old_noise, old_wavelength, new_wavelength):
    """Rebin a spectrum onto a coarser wavelength scale."""
    # This is really crude and needs to be improved. No fractional pixels are
    # dealt with; the input pixels are just assigned to one or another output
    # pixel.
    delta = new_wavelength[1] - new_wavelength[0]
    n_pix = len(new_wavelength)
    new_flux = np.zeros(n_pix)
    new_noise = np.zeros(n_pix)
    for i_pix, wave_i in enumerate(new_wavelength):
        in_range = ((old_wavelength > (wave_i - 0.5*delta)) &
                    (old_wavelength < (wave_i + 0.5*delta)) &
                    np.isfinite(old_flux) & np.isfinite(old_noise))
        new_flux[i_pix] = np.mean(old_flux[in_range])
        new_noise[i_pix] = (np.sqrt(np.sum(old_noise[in_range]**2)) / 
                            np.sum(in_range))
    return new_flux, new_noise, new_wavelength

def read_stellar_models(path_root='./ck04models/'):
    """Return a dictionary of all the stellar models available."""
    file_list = glob(os.path.join(path_root, '*', '*.fits'))
    n_model = pf.getval(os.path.join(path_root, 'catalog.fits'), 'NAXIS2', 1)
    n_pix = pf.getval(file_list[0], 'NAXIS2', 1)
    models = {}
    models['wavelength'] = pf.getdata(file_list[0], 1)['WAVELENGTH']
    models['flux'] = np.zeros((n_model, n_pix))
    i_model = 0
    for filename in file_list:
        data = pf.getdata(filename, 1)
        for name in (n for n in data.columns.names if n != 'WAVELENGTH'):
            models['flux'][i_model, :] = data[name]
            i_model += 1
    # There's an extra, irregular pixel at 3640.0A. I don't know why.
    keep = ((models['wavelength'] >= 3000) & 
            (models['wavelength'] < 10000) & 
            (models['wavelength'] != 3640.0))
    models['wavelength'] = models['wavelength'][keep]
    models['flux'] = models['flux'][:, keep]
    return models

def measure_colour(flux, wavelength, sdss_dir='./sdss/'):
    """Return synthetic SDSS g-r colour for a given spectrum."""
    return (measure_band('g', flux, wavelength, sdss_dir=sdss_dir) -
            measure_band('r', flux, wavelength, sdss_dir=sdss_dir))

def read_filter(band, sdss_dir='./sdss/'):
    """Return filter response and wavelength for an SDSS filter."""
    data = np.loadtxt(os.path.join(sdss_dir, band + '.dat'))
    wavelength = data[:, 0]
    # Response through airmass 0.0
    response = data[:, 3]
    return response, wavelength

def measure_band(band, flux, wavelength, sdss_dir='./sdss/'):
    """Return the synthetic magnitude of a spectrum in an SDSS band."""
    filter_response, filter_wavelength = read_filter(band, sdss_dir=sdss_dir)
    filter_interpolated = np.interp(
        wavelength, filter_wavelength, filter_response)
    # Convert to SI units. Wavelength was in A, flux was in 1e-16 erg/s/cm^2/A
    wl_m = wavelength * 1e-10
    flux_wm3 = flux * 1e-16 * 1e-7 * (1e2)**2 * 1e10
    # AB magnitudes are zero for flux of 3631 Jy
    flux_zero = 3631.0 * 1.0e-26 * 2.99792458e8 / (wl_m**2)
    # Get the wavelength bin sizes - don't assume constant!
    delta_wl = np.hstack((
        wl_m[1] - wl_m[0],
        0.5 * (wl_m[2:] - wl_m[:-2]),
        wl_m[-1] - wl_m[-2]))
    flux_band = (np.sum(delta_wl * wl_m * filter_interpolated * flux_wm3) / 
                 np.sum(delta_wl * wl_m * filter_interpolated * flux_zero))
    return -2.5 * np.log10(flux_band)

def measure_mags(flux, noise, wavelength):
    """Do clipping and interpolation, then return g and r band mags."""
    good = clip_spectrum(flux, noise, wavelength, limit_flux=20.0)
    flux, noise, wavelength = interpolate_arms(
        flux, noise, wavelength, good)
    mags = {}
    for band in BANDS:
        mags[band] = measure_band(band, flux, wavelength)
    return mags

def interpolate_arms(flux, noise, wavelength, good=None, n_pix_fit=300):
    """Interpolate between the red and blue arms."""
    # This function currently assumes that:
    #  * There are an even number of pixels, half of which come from each arm
    #  * There is a gap in the wavelength coverage between the two arms
    #  * There are at least 300 pixels in each arm to fit to
    # TODO: Clean up to remove these assumptions, particularly the second one
    # Establish basic facts about which pixels we should look at
    n_pix = len(wavelength)
    if good is None:
        good = np.arange(n_pix)
    middle = np.int(n_pix // 2)
    good_blue = good & (np.arange(n_pix) < middle)
    good_red = good & (np.arange(n_pix) >= middle)
    wavelength_middle = 0.5 * (wavelength[middle-1] + wavelength[middle])
    delta_wave_blue = wavelength[1] - wavelength[0]
    delta_wave_red = wavelength[-1] - wavelength[-2]
    # Get the flux from the red end of the blue and the blue end of the red,
    # and fit a straight line between them
    index_blue = np.where(good_blue)[0][-n_pix_fit:]
    index_red = np.where(good_red)[0][:n_pix_fit]
    index_fit = np.hstack((index_blue, index_red))
    poly_params = np.polyfit(wavelength[index_fit], flux[index_fit], 1,
                             w=1.0/noise[index_fit]**2)
    wavelength_start = wavelength[middle-1] + delta_wave_blue
    n_pix_insert_blue = int(np.round(
        (wavelength_middle - wavelength_start) / delta_wave_blue))
    wavelength_end = wavelength[middle]
    n_pix_insert_red = int(np.round(
        (wavelength_end - wavelength_middle) / delta_wave_red))
    n_pix_insert = n_pix_insert_red + n_pix_insert_blue
    wavelength_insert = np.hstack((
        np.linspace(wavelength_start, wavelength_middle, n_pix_insert_blue,
                    endpoint=False),
        np.linspace(wavelength_middle, wavelength_end, n_pix_insert_red,
                    endpoint=False)))
    wavelength_out = np.hstack(
        (wavelength[:middle], wavelength_insert, wavelength[middle:]))
    flux_out = np.hstack(
        (flux[:middle], np.zeros(n_pix_insert), flux[middle:]))
    noise_out = np.hstack(
        (noise[:middle], np.zeros(n_pix_insert), noise[middle:]))
    insert = ((wavelength_out > wavelength[index_blue[-1]]) &
              (wavelength_out < wavelength[index_red[0]]))
    flux_out[insert] = np.polyval(poly_params, wavelength_out[insert])
    noise_out[insert] = np.nan
    good_extended = np.hstack(
        (good[:middle], np.zeros(n_pix_insert, bool), good[middle:]))
    interp = ~good_extended & ~insert
    flux_out[interp] = np.interp(
        wavelength_out[interp], wavelength_out[~interp], flux_out[~interp])
    noise_out[interp] = np.nan
    return flux_out, noise_out, wavelength_out

def collapse_cube(flux, noise, wavelength, good=None, n_band=1):
    """Collapse a cube into a 2-d image, or a series of images."""
    # Be careful about sending in mixed red+blue cubes - in this case
    # n_band should be even
    n_pix = len(wavelength)
    if good is None:
        good = np.arange(flux.size)
        good.shape = flux.shape
    flux_out = np.zeros((n_band, flux.shape[1], flux.shape[2]))
    noise_out = np.zeros((n_band, flux.shape[1], flux.shape[2]))
    wavelength_out = np.zeros(n_band)
    for i_band in range(n_band):
        start = i_band * n_pix / n_band
        finish = (i_band+1) * n_pix / n_band
        flux_out[i_band, :, :] = (
            np.nansum(flux[start:finish, :, :] * good[start:finish, :, :], 0) /
            np.sum(good[start:finish, :, :], 0))
        noise_out[i_band, :, :] = (
            np.sqrt(np.nansum(noise[start:finish, :, :]**2 * 
                              good[start:finish, :, :], 0)) /
            np.sum(good[start:finish, :, :], 0))
        wavelength_out[i_band] = (
            0.5 * (wavelength[start] + wavelength[finish-1]))
    flux_out = np.squeeze(flux_out)
    noise_out = np.squeeze(noise_out)
    wavelength_out = np.squeeze(wavelength_out)
    return flux_out, noise_out, wavelength_out

def fit_moffat_to_image(image, noise, elliptical=False, background=False):
    """Fit a Moffat profile to an image, optionally allowing ellipticity."""
    fit_pix = np.isfinite(image) & np.isfinite(noise)
    coords = np.meshgrid(np.arange(image.shape[0]), 
                         np.arange(image.shape[1]))
    x00 = 0.5 * (image.shape[0] - 1)
    y00 = 0.5 * (image.shape[1] - 1)
    alpha0 = 4.0
    beta0 = 4.0
    intensity0 = np.nansum(image)
    if elliptical:
        p0 = [alpha0, alpha0, 0.0, beta0, x00, y00, intensity0]
    else:
        p0 = [alpha0, beta0, x00, y00, intensity0]
    if background:
        p0.append(0.0)
    def fit_function(p):
        model = moffat_integrated(
            coords[0], coords[1], p, elliptical=elliptical, 
            background=background, good=fit_pix)
        return ((model - image[fit_pix]) / noise[fit_pix])
    result = leastsq(fit_function, p0, full_output=True)
    params = result[0]
    if result[1] is None:
        sigma = None
    else:
        reduced_chi2 = np.sum(fit_function(params)**2 / (np.sum(fit_pix) - 1))
        n_params = len(params)
        sigma = np.sqrt(result[1][np.arange(n_params), np.arange(n_params)] /
                        reduced_chi2)
    return params, sigma
    
def fit_moffat_to_chunks(flux, noise, wavelength, elliptical=False):
    """Fit a Moffat profile to a chunked datacube"""
    fit_pix = np.isfinite(flux) & np.isfinite(noise)
    coords = np.meshgrid(
        np.arange(flux.shape[1]), 
                         np.arange(flux.shape[2]))
    x00 = 0.5 * (flux.shape[1] - 1)
    y00 = 0.5 * (flux.shape[2] - 1)
    alpha0 = 4.0
    beta0 = 4.0
    intensity0 = [np.nansum(image) for image in flux]
    if elliptical:
        p0 = [alpha0, alpha0, 0.0, beta0, x00, y00]
    else:
        p0 = [alpha0, beta0, x00, y00]
    p0.extend(intensity0)
    p0 = np.array(p0)
    def fit_function(p):
        model = np.array([moffat_integrated(
            coords[0], coords[1],
            psf_params_at_slice(p, i_slice, wavelength_slice,
                                elliptical=elliptical),
            elliptical=elliptical)
            for i_slice, wavelength_slice
            in enumerate(wavelength)])
        model.shape = flux.shape
        return ((model[fit_pix] - flux[fit_pix]) / noise[fit_pix])
    result = leastsq(fit_function, p0, full_output=True)
    params = result[0]
    if result[1] is None:
        sigma = None
    else:
        reduced_chi2 = np.sum(fit_function(params)**2 / (np.sum(fit_pix) - 1))
        n_params = len(params)
        sigma = np.sqrt(result[1][np.arange(n_params), np.arange(n_params)] /
                        reduced_chi2)
    return params, sigma
    
def psf_params_at_slice(params, i_slice, wavelength, elliptical=False,
                        copy_intensity=True):
    """Return the PSF parameters at a particular wavelength."""
    reference_wavelength = 5000.0
    factor = (wavelength / reference_wavelength) ** (-0.2)
    if elliptical:
        n_basic = 6
    else:
        n_basic = 4
    params_slice = np.zeros(n_basic + 1)
    params_slice[:n_basic] = params[:n_basic].copy()
    if copy_intensity:
        params_slice[-1] = params[n_basic + i_slice]
    else:
        params_slice[-1] = 1.0
    if elliptical:
        params_slice[0] = params[0] * factor
        params_slice[1] = params[1] * factor
    else:
        params_slice[0] = params[0] * factor
    return params_slice

def scale_moffat_to_image(image, noise, params, elliptical=False,
                          background=False):
    """Scale a Moffat profile to fit the provided image."""
    fit_pix = np.isfinite(image) & np.isfinite(noise)
    if np.sum(fit_pix) == 0:
        if background:
            return np.nan, np.nan, np.nan, np.nan
        else:
            return np.nan, np.nan
    coords = np.meshgrid(np.arange(image.shape[0]), 
                         np.arange(image.shape[1]))
    params_norm = params.copy()
    if background:
        # Set flux to 1 and background to 0
        params_norm[-2] = 1.0
        params_norm[-1] = 0.0
    else:
        # Set flux to 1
        params_norm[-1] = 1.0
    model_norm = moffat_integrated(
        coords[0], coords[1], params_norm, elliptical=elliptical, 
        background=background, good=fit_pix)
    p0 = [np.nansum(image)]
    if background:
        p0.append(0.0)
    def fit_function(p):
        model = p[0] * model_norm
        if background:
            model += p[1]
        return ((model - image[fit_pix]) / noise[fit_pix])
    result = leastsq(fit_function, p0, full_output=True)
    intensity = result[0][0]
    reduced_chi2 = np.sum(fit_function(result[0])**2 / (np.sum(fit_pix) - 1))
    if result[1] is None:
        sigma = np.nan
    else:
        sigma = np.sqrt(result[1][0, 0] / reduced_chi2)
    if background:
        back = result[0][1]
        if result[1] is None:
            sigma_back = np.nan
        else:
            sigma_back = np.sqrt(result[1][1, 1] / reduced_chi2)
        return intensity, sigma, back, sigma_back
    else:
        return intensity, sigma

def moffat_integrated(x, y, params, elliptical=False, background=False,
                      good=None, pix_size=1.0, n_sub=10):
    """Return a Moffat profile, integrated over pixels."""
    if good is None:
        good = np.ones(x.size, bool)
        good.shape = x.shape
    n_pix = np.sum(good)
    x_flat = x[good]
    y_flat = y[good]
    delta = pix_size * (np.arange(float(n_sub)) / n_sub)
    delta -= np.mean(delta)
    x_sub = (np.outer(x_flat, np.ones(n_sub**2)) + 
             np.outer(np.ones(n_pix), np.outer(delta, np.ones(n_sub))))
    y_sub = (np.outer(y_flat, np.ones(n_sub**2)) + 
             np.outer(np.ones(n_pix), np.outer(np.ones(n_sub), delta)))
    if background:
        params_sub = params[:-1]
    else:
        params_sub = params
    if elliptical:
        moffat_sub = moffat_elliptical(x_sub, y_sub, *params_sub)
    else:
        moffat_sub = moffat_circular(x_sub, y_sub, *params_sub)
    moffat = np.mean(moffat_sub, 1)
    if background:
        moffat += params[-1]
    return moffat


def moffat_elliptical(x, y, alpha_x, alpha_y, rho, beta, x0, y0, intensity):
    """Return an elliptical Moffat profile."""
    norm = (beta - 1) / (np.pi * alpha_x * alpha_y * np.sqrt(1 - rho**2))
    norm = norm * intensity
    x_term = (x - x0) / alpha_x
    y_term = (y - y0) / alpha_y
    moffat = norm * (1 + (x_term**2 + y_term**2 - 2*rho*x_term*y_term) /
                         (1 - rho**2))**(-beta)
    return moffat

def moffat_circular(x, y, alpha, beta, x0, y0, intensity):
    """Return a circular Moffat profile."""
    norm = (beta - 1) / (np.pi * alpha**2)
    norm = norm * intensity
    x_term = (x - x0) / alpha
    y_term = (y - y0) / alpha
    moffat = norm * (1 + (x_term**2 + y_term**2))**(-beta)
    return moffat






