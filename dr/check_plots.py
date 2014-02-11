from . import fluxcal2

import matplotlib.pyplot as plt
import astropy.io.fits as pf
import numpy as np

import os
import itertools
from glob import glob

def check_bia(combined_path):
    """Plot a combined bias calibration frame."""
    message = """Check that the combined bias frame has no more artefacts than
normal."""
    check_combined(combined_path, message)
    return

def check_drk(combined_path):
    """Plot a combined dark calibration frame."""
    message = """Check that the combined dark frame has no more artefacts than
normal."""
    check_combined(combined_path, message)
    return

def check_lfl(combined_path):
    """Plot a combined long-slit flat calibration frame."""
    message = """Check that the combined long-slit flat has no more artefacts
than normal."""
    check_combined(combined_path, message)
    return

def check_combined(combined_path, message):
    """Plot a combined calibration frame of some sort."""
    print message
    # Read the data
    image = pf.getdata(combined_path)
    sorted_image = np.ravel(image)
    sorted_image = np.sort(sorted_image[np.isfinite(sorted_image)])
    one_per_cent = len(sorted_image) / 100
    vmin = sorted_image[one_per_cent]
    vmax = sorted_image[-one_per_cent]
    fig = plt.figure('Combined calibration', figsize=(6., 10.))
    plt.imshow(image, vmin=vmin, vmax=vmax, cmap='GnBu')
    plt.colorbar()
    print "When you're ready to move on..."
    return

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
    message = """Check that each plotted absorption spectrum has the correct
shape for telluric absorption."""
    print message
    for fits in fits_list:
        plt.figure(fits.filename)
        header = pf.getheader(fits.fluxcal_path)
        wavelength = header['CRVAL1'] + header['CDELT1'] * (
            1 + np.arange(header['NAXIS1']) - header['CRPIX1'])
        spectrum = (
            1.0 / pf.getdata(fits.fluxcal_path, 'FLUX_CALIBRATION')[-1, :])
        plt.plot(wavelength, spectrum)
        plt.ylim((0, 1.1))
    print "When you're ready to move on..."
    return

def check_cub(fits_list):
    """Plot the results of cubing."""
    message = """Check that the galaxies appear in the centre in each arm, that
they look galaxy-like, and the spectra have no obvious artefacts."""
    print message
    # The positions of points to plot
    position_dict = {'Centre': (24.5, 24.5),
                     'North': (24.5, 34.5),
                     'East': (14.5, 24.5),
                     'South': (24.5, 14.5),
                     'West': (34.5, 24.5)}
    # Construct the list of object names
    # We're assuming that a single field was sent
    fibre_table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
    object_name_list = np.unique(
        fibre_table[fibre_table['TYPE'] == 'P']['NAME'])
    field_id = fits_list[0].field_id
    root = os.path.join(fits_list[0].raw_dir, '../../../cubed')
    for object_name in object_name_list:
        # Find the datacubes
        path_blue = glob(root+'/'+object_name+'/*blue*'+field_id+'*.fits')[0]
        path_red = glob(root+'/'+object_name+'/*red*'+field_id+'*.fits')[0]
        # Load the data
        hdulist_blue = pf.open(path_blue)
        data_blue = hdulist_blue[0].data
        header_blue = hdulist_blue[0].header
        hdulist_blue.close()
        hdulist_red = pf.open(path_red)
        data_red = hdulist_red[0].data
        header_red = hdulist_red[0].header
        hdulist_red.close()
        # Set up the figure
        fig = plt.figure(object_name, figsize=(16., 12.))
        # Show collapsed images of the object
        trim = 100
        ax_blue = fig.add_subplot(221)
        plt.imshow(np.nansum(data_blue[trim:-1*trim, :, :], axis=0), 
                   origin='lower', interpolation='nearest', cmap='GnBu')
        ax_blue.set_title('Blue arm')
        ax_red = fig.add_subplot(222)
        plt.imshow(np.nansum(data_red[trim:-1*trim, :, :], axis=0), 
                   origin='lower', interpolation='nearest', cmap='GnBu')
        ax_red.set_title('Red arm')
        # Show the central spectrum and a few others
        color_cycle = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
        wavelength_blue = header_blue['CRVAL3'] + header_blue['CDELT3'] * (
            1 + np.arange(header_blue['NAXIS3']) - header_blue['CRPIX3'])
        wavelength_red = header_red['CRVAL3'] + header_red['CDELT3'] * (
            1 + np.arange(header_red['NAXIS3']) - header_red['CRPIX3'])
        ax_spec = fig.add_subplot(212)
        for name, coords in position_dict.items():
            flux_blue = np.nansum(np.nansum(
                data_blue[:, int(coords[0]):int(coords[0])+2, 
                          int(coords[1]):int(coords[1])+2], 
                axis=2), axis=1) / 4.0
            flux_red = np.nansum(np.nansum(
                data_red[:, int(coords[0]):int(coords[0])+2, 
                         int(coords[1]):int(coords[1])+2], 
                axis=2), axis=1) / 4.0
            color = next(color_cycle)
            plt.plot(wavelength_blue, flux_blue, c=color, label=name)
            plt.plot(wavelength_red, flux_red, c=color)
            plt.legend()
            # Also add location markers to the images
            ax_blue.scatter(coords[0], coords[1], marker='x', c=color)
            ax_red.scatter(coords[0], coords[1], marker='x', c=color)
    print "When you're ready to move on..."
    return



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
