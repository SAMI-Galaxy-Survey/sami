from . import fluxcal2

import matplotlib.pyplot as plt
import astropy.io.fits as pf
import numpy as np

import os
import itertools

def check_flx(fits_list):
    """Plot the results of flux calibration."""
    # Tell the user what to do
    message = """For each individual file, check that the observed ratio has no
obvious artefacts, and the fitted ratio follows the observed
ratio. For the combined plot, check that all the input ratios
have the same approximate shape. Some variation in
nomalisation is ok."""
    print message
    # Set up colors
    color_cycle = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y'])
    # First find a CCD 1 file and a CCD 2 file
    fits_1 = None
    fits_2 = None
    for fits in fits_list:
        if fits.ccd == 'ccd_1':
            fits_1 = fits
        elif fits.ccd == 'ccd_2':
            fits_2 = fits
        if fits_1 is not None and fits_2 is not None:
            break
    if fits_1 is None or fits_2 is None:
        raise ValueError('fits_list must contain matched CCD 1/2 files')
    # Find the combined files that relate to these files
    # We assume that they all belong to the same group
    path_combined_1 = os.path.join(fits_1.reduced_dir,
                                   'TRANSFERcombined.fits')
    hdulist_combined_1 = pf.open(path_combined_1)
    path_combined_2 = os.path.join(fits_2.reduced_dir,
                                   'TRANSFERcombined.fits')
    hdulist_combined_2 = pf.open(path_combined_2)
    # Load the spectrum of the standard star
    standard_star = fluxcal2.read_standard_data(
        {'path': hdulist_combined_1[1].header['STDFILE']})
    # Construct wavelength arrays
    header_1 = pf.getheader(fits_1.reduced_path)
    wavelength_1 = header_1['CRVAL1'] + header_1['CDELT1'] * (
        1 + np.arange(header_1['NAXIS1']) - header_1['CRPIX1'])
    header_2 = pf.getheader(fits_2.reduced_path)
    wavelength_2 = header_2['CRVAL1'] + header_2['CDELT1'] * (
        1 + np.arange(header_2['NAXIS1']) - header_2['CRPIX1'])
    # Make a plot showing the combined transfer function and each individual one
    fig_combined = plt.figure('Combined data', figsize=(16., 6.))
    for hdu_1 in hdulist_combined_1[1:]:
        filename = os.path.basename(hdu_1.header['ORIGFILE'])
        hdu_2 = match_fcal_hdu(hdulist_combined_2, hdu_1)
        color = next(color_cycle)
        plt.plot(wavelength_1, 1.0 / hdu_1.data[2, :], c=color, label=filename)
        plt.plot(wavelength_2, 1.0 / hdu_2.data[2, :], c=color)
    plt.plot(wavelength_1, 1.0 / hdulist_combined_1[0].data, c='k', linewidth=3,
             label='Combined')
    plt.plot(wavelength_2, 1.0 / hdulist_combined_2[0].data, c='k', linewidth=3)
    plt.legend(loc='best')
    # Make a plot for each input file, showing the data and the fit
    for hdu_1 in hdulist_combined_1[1:]:
        filename = os.path.basename(hdu_1.header['ORIGFILE'])
        hdu_2 = match_fcal_hdu(hdulist_combined_2, hdu_1)
        fig_single = plt.figure(filename, figsize=(16., 6.))
        observed_ratio_1 = (
            fluxcal2.rebin_flux(standard_star['wavelength'], wavelength_1, 
                                hdu_1.data[0, :]) / standard_star['flux'])
        plt.plot(standard_star['wavelength'], observed_ratio_1, 
                 label='Observed ratio', c='b')
        plt.plot(wavelength_1, 1.0 / hdu_1.data[2, :], 
                 label='Fitted ratio', c='g')
        observed_ratio_2 = (
            fluxcal2.rebin_flux(standard_star['wavelength'], wavelength_2, 
                                hdu_2.data[0, :]) / standard_star['flux'])
        plt.plot(standard_star['wavelength'], observed_ratio_2, c='b')
        plt.plot(wavelength_2, 1.0 / hdu_2.data[2, :], c='g')
        plt.legend(loc='best')
    print "When you're ready to move on..."
    return


def check_tel(fits_list):
    """Plot the results of telluric correction."""
    pass

def check_cube(filename_list):
    """Plot the results of cubing."""
    pass



def match_fcal_hdu(hdulist, hdu):
    """Return the HDU from the HDUList that matches the given HDU."""
    filename = os.path.basename(hdu.header['ORIGFILE'])
    hdu_match = None
    for hdu_match in hdulist[1:]:
        filename_match = os.path.basename(hdu_match.header['ORIGFILE'])
        if (filename_match.startswith(filename[:5]) and 
            filename_match[6:10] == filename[6:10]):
            return hdu_match
    raise ValueError('fits_list must contain matched CCD 1/2 files')
