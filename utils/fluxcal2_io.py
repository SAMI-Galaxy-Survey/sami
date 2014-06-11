"""
This module exists to prevent a circular import between ifu.py and
fluxcal2.py
"""

import numpy as np
import astropy.io.fits as pf

def read_model_parameters(hdu):
    """Return the PSF model parameters in a header, with the model name."""
    psf_parameters = {}
    model_name = None
    for key, value in hdu.header.items():
        if key == 'MODEL':
            model_name = value
        else:
            try:
                psf_parameters[header_translate_inverse(key)] = value
            except KeyError:
                continue
    psf_parameters['flux'] = hdu.data[0, :]
    psf_parameters['background'] = hdu.data[1, :]
    return psf_parameters, model_name

def header_translate_inverse(header_name):
    """Translate parameter names back from headers."""
    name_dict = {'XCENREF': 'xcen_ref',
                 'YCENREF': 'ycen_ref',
                 'ZENDIR': 'zenith_direction',
                 'ZENDIST': 'zenith_distance',
                 'FLUX': 'flux',
                 'BETA': 'beta',
                 'BCKGRND': 'background',
                 'ALPHAREF': 'alpha_ref',
                 'TEMP': 'temperature',
                 'PRESSURE': 'pressure',
                 'VAPPRESS': 'vapour_pressure'}
    return name_dict[header_name]

def save_extracted_flux(path, observed_flux, observed_background,
                        sigma_flux, sigma_background,
                        star_match, psf_parameters, model_name,
                        good_psf, hg_changeset, hdu_name='FLUX_CALIBRATION'):
    """Add the extracted flux to the specified FITS file."""
    # Turn the data into a single array
    data = np.vstack((observed_flux, observed_background, 
                      sigma_flux, sigma_background))
    # Make the new HDU
    new_hdu = pf.ImageHDU(data, name=hdu_name)
    # Add info to the header
    header_item_list = [
        ('PROBENUM', star_match['probenum'], 'Number of the probe containing '
                                             'the star'),
        ('STDNAME', star_match['name'], 'Name of standard star'),
        ('HGFLXCAL', hg_changeset, 'Hg changeset ID for fluxcal code'),
        ('MODEL', model_name, 'Name of model used in PSF fit'),
        ('GOODPSF', good_psf, 'Whether the PSF fit has good parameters')]
    if 'path' in star_match:
        header_item_list.append(
            ('STDFILE', star_match['path'], 'Filename of standard spectrum'))
    if 'separation' in star_match:
        header_item_list.append(
            ('STDOFF', star_match['separation'], 'Offset (arcsec) to standard '
                                                 'star coordinates'))
    for key, value in psf_parameters.items():
        header_item_list.append((header_translate(key), value, 
                                 'PSF model parameter'))
    for key, value, comment in header_item_list:
        try:
            new_hdu.header[key] = (value, comment)
        except ValueError:
            # Probably tried to save an array, just ditch it
            pass
    # Update the file
    hdulist = pf.open(path, 'update', do_not_scale_image_data=True)
    # Check if there's already an extracted flux, and delete if so
    try:
        existing_index = hdulist.index_of(hdu_name)
    except KeyError:
        pass
    else:
        del hdulist[existing_index]
    hdulist.append(new_hdu)
    hdulist.close()
    del hdulist
    return

def header_translate(key):
    """Translate parameter names to be suitable for headers."""
    name_dict = {'xcen_ref': 'XCENREF',
                 'ycen_ref': 'YCENREF',
                 'zenith_direction': 'ZENDIR',
                 'zenith_distance': 'ZENDIST',
                 'flux': 'FLUX',
                 'beta': 'BETA',
                 'background': 'BCKGRND',
                 'alpha_ref': 'ALPHAREF',
                 'temperature': 'TEMP',
                 'pressure': 'PRESSURE',
                 'vapour_pressure': 'VAPPRESS'}
    try:
        header_name = name_dict[key]
    except KeyError:
        header_name = key[:8]
    return header_name

