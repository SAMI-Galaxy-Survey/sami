from . import fluxcal2
from ..utils import IFU

import matplotlib.pyplot as plt
from matplotlib import cm
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
    in_telluric = fluxcal2.in_telluric_band(standard_star['wavelength'])
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
        plt.plot(wavelength_1, 1.0 / hdu_1.data[-1, :], c=color, label=filename)
        plt.plot(wavelength_2, 1.0 / hdu_2.data[-1, :], c=color)
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
            fluxcal2.rebin_flux_noise(
                standard_star['wavelength'], wavelength_1, 
                hdu_1.data[0, :], hdu_1.data[2, :])[0] /
            standard_star['flux'])
        observed_ratio_masked_1 = observed_ratio_1.copy()
        observed_ratio_masked_1[in_telluric] = np.nan
        plt.plot(standard_star['wavelength'], observed_ratio_1, c='g', 
                 label='Observed ratio')
        plt.plot(standard_star['wavelength'], observed_ratio_masked_1, c='b',
                 label='Observed ratio (masked)')
        plt.plot(wavelength_1, 1.0 / hdu_1.data[-1, :], c='r', 
                 label='Fitted ratio')
        observed_ratio_2 = (
            fluxcal2.rebin_flux_noise(
                standard_star['wavelength'], wavelength_2, 
                hdu_2.data[0, :], hdu_2.data[2, :])[0] / 
            standard_star['flux'])
        observed_ratio_masked_2 = observed_ratio_2.copy()
        observed_ratio_masked_2[in_telluric] = np.nan
        plt.plot(standard_star['wavelength'], observed_ratio_2, c='g')
        plt.plot(standard_star['wavelength'], observed_ratio_masked_2, c='b')
        plt.plot(wavelength_2, 1.0 / hdu_2.data[-1, :], c='r')
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
            1.0 / pf.getdata(fits.fluxcal_path, 'FLUX_CALIBRATION')[-2, :])
        plt.plot(wavelength, spectrum)
        plt.ylim((0, 1.1))
    print "When you're ready to move on..."
    return

def check_ali(fits_list):
    """Plot the results of alignment."""
    message = """Check that any bad fits have been rejected."""
    print message
    data = []
    x_rms = []
    y_rms = []
    n_sigma = []
    for fits in fits_list:
        path_list = (fits.telluric_path, fits.fluxcal_path, fits.reduced_path)
        for path in path_list:
            try:
                data_i = pf.getdata(path, 'ALIGNMENT')
                order = np.argsort(
                    IFU(path, probenum, flag_name=False).name
                    for probenum in data_i['PROBENUM'])
                date_i = data_i[order]
                header = pf.getheader(path, 'ALIGNMENT')
            except (KeyError, IOError):
                pass
            else:
                break
        else:
            continue
        data.append(data_i)
        x_rms.append(header['X_RMS'])
        y_rms.append(header['Y_RMS'])
        n_sigma.append(header['SIGMA'])
    data = np.array(data)
    x_rms = np.array(x_rms)
    y_rms = np.array(y_rms)
    scale = 200.0
    radius_plot = 130000.0
    radius_field = 125000.0
    radius_fibre = scale * 105.0 / 2.0
    x_centroid_plot = (
        (data['X_CEN'] - data['X_REFMED']) * scale + data['X_REFMED'])
    y_centroid_plot = (
        (data['Y_CEN'] - data['Y_REFMED']) * scale + data['Y_REFMED'])
    x_fit_plot = data['X_SHIFT'] * scale + data['X_REFMED']
    y_fit_plot = -data['Y_SHIFT'] * scale + data['Y_REFMED']
    n_ifu = data.shape[1]
    fig = plt.figure(fits_list[0].field_id, figsize=(10., 10.))
    axes = fig.add_subplot(111, aspect='equal')
    plt.xlim((-radius_plot, radius_plot))
    plt.ylim((-radius_plot, radius_plot))
    axes.add_patch(plt.Circle((0, 0), radius_field, fill=False, lw=0.5))
    for ifu, x_cen, y_cen, x_fit, y_fit, good in zip(
            data[0], x_centroid_plot.T, y_centroid_plot.T, x_fit_plot.T, 
            y_fit_plot.T, data['GOOD'].T):
        good = good.astype(bool)
        color = cm.winter(ifu['PROBENUM'] / float(n_ifu - 1))
        fibre = plt.Circle(
            (ifu['X_REFMED'], ifu['Y_REFMED']), 
            radius_fibre, 
            fill=False, 
            ls='dashed', 
            color=color)
        axes.add_patch(fibre)
        plt.scatter(x_cen[good], y_cen[good], color='k')
        plt.scatter(x_cen[~good], y_cen[~good], color='r')
        for index in np.where(~good)[0]:
            plt.plot((x_fit[index], x_cen[index]), 
                     (y_fit[index], y_cen[index]), ':', color=color)
        plt.plot(x_fit, y_fit, color=color, lw=2.0)
        plt.annotate(
            'IFS'+str(ifu['PROBENUM']), 
            xy=(ifu['X_REFMED'], ifu['Y_REFMED']+radius_fibre), 
            xycoords='data', 
            xytext=None, 
            textcoords='data', 
            arrowprops=None, 
            color=color)
    plt.title('RMS: ' + ', '.join('{:.1f}'.format(rms) 
                                  for rms in np.sqrt((x_rms**2 + y_rms**2)))
              + '\nSigma clip: ' + ', '.join('{:.2f}'.format(n) 
                                             for n in n_sigma))
    print "When you're ready to move on..."
    
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
        glob_blue = root+'/'+object_name+'/*blue*'+field_id+'*.fits'
        path_list_blue = glob(glob_blue) + glob(glob_blue + '.gz')
        if path_list_blue:
            path_blue = path_list_blue[0]
            blue_available = True
        else:
            blue_available = False
        glob_red = root+'/'+object_name+'/*red*'+field_id+'*.fits'
        path_list_red = glob(glob_red) + glob(glob_red + '.gz')
        if path_list_red:
            path_red = path_list_red[0]
            red_available = True
        else:
            red_available = False
        if not blue_available and not red_available:
            print 'No data found for object', object_name
        # Load the data
        if blue_available:
            hdulist_blue = pf.open(path_blue)
            data_blue = hdulist_blue[0].data
            header_blue = hdulist_blue[0].header
            hdulist_blue.close()
        if red_available:
            hdulist_red = pf.open(path_red)
            data_red = hdulist_red[0].data
            header_red = hdulist_red[0].header
            hdulist_red.close()
        # Set up the figure
        fig = plt.figure(object_name, figsize=(16., 12.))
        # Show collapsed images of the object
        trim = 100
        if blue_available:
            ax_blue = fig.add_subplot(221)
            plt.imshow(np.nansum(data_blue[trim:-1*trim, :, :], axis=0), 
                       origin='lower', interpolation='nearest', cmap='GnBu')
            ax_blue.set_title('Blue arm')
            wavelength_blue = header_blue['CRVAL3'] + header_blue['CDELT3'] * (
                1 + np.arange(header_blue['NAXIS3']) - header_blue['CRPIX3'])
        if red_available:
            ax_red = fig.add_subplot(222)
            plt.imshow(np.nansum(data_red[trim:-1*trim, :, :], axis=0), 
                       origin='lower', interpolation='nearest', cmap='GnBu')
            ax_red.set_title('Red arm')
            wavelength_red = header_red['CRVAL3'] + header_red['CDELT3'] * (
                1 + np.arange(header_red['NAXIS3']) - header_red['CRPIX3'])
        # Show the central spectrum and a few others
        color_cycle = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
        ax_spec = fig.add_subplot(212)
        for name, coords in position_dict.items():
            color = next(color_cycle)
            if blue_available:
                flux_blue = np.nansum(np.nansum(
                    data_blue[:, int(coords[0]):int(coords[0])+2, 
                              int(coords[1]):int(coords[1])+2], 
                    axis=2), axis=1) / 4.0
                plt.plot(wavelength_blue, flux_blue, c=color, label=name)
                # Add location marker to the blue image
                ax_blue.scatter(coords[0], coords[1], marker='x', c=color)
            if red_available:
                flux_red = np.nansum(np.nansum(
                    data_red[:, int(coords[0]):int(coords[0])+2, 
                             int(coords[1]):int(coords[1])+2], 
                    axis=2), axis=1) / 4.0
                plt.plot(wavelength_red, flux_red, c=color)
                # Add location marker to the red image
                ax_red.scatter(coords[0], coords[1], marker='x', c=color)
            plt.legend()
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
