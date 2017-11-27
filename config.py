from __future__ import print_function
"""
Useful quantities to be used throughout the sami package.

Note that actual use of these values varies: some modules correctly use this
module, but others define these values themselves, so do not assume that a
change made here will be correctly propagated everywhere.
"""

import os
import re
import shutil
import warnings
warnings.simplefilter('always', RuntimeWarning)

import astropy.units as u
import astropy.coordinates as coords
from astropy import __version__ as ASTROPY_VERSION



try: # Catch to maintain compatibility with both python 2.7 and 3.x
    import configparser
except ImportError:
    import ConfigParser as configparser

# This script contains constants that are used in other SAMI packages.

ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))

# ----------------------------------------------------------------------------------------

# Approximate plate scale
plate_scale=15.22

# Diameter of individual SAMI fibres in arcseconds
fibre_diameter_arcsec = 1.6

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# For Andy's dome windscreen position script.
# Distance between the polar and declination axes
polar_declination_dist=0.0625 # dome radii

# Distance between the declination axis & dome center on meridian
declination_dome_dist=0.0982  # dome radii

# Latitude of SSO (based on AAT Zenith Position, may not be most accurate)
latitude=coords.Angle(-31.3275, unit=u.degree)

# astropy version catch to be backwards compatible
if ASTROPY_VERSION[0] == 0 and ASTROPY_VERSION[1] <= 2:
    latitude_radians = latitude.radians
    latitude_degrees = latitude.degrees
else:
    latitude_radians = latitude.radian
    latitude_degrees = latitude.degree
# ----------------------------------------------------------------------------------------

# Pressure conversion factor from millibars to mm of Hg 
millibar_to_mmHg = 0.750061683

# Set the test data directory, assumed to be at the same level as the sami package.
test_data_dir = os.path.dirname(os.path.realpath(__file__)) + '/../test_data/'


# +---+------------------------------------------------------------------------+
# | 2.| SAMI configuration file, using ``configparser`` module.                |
# +---+------------------------------------------------------------------------+

# Setup catalogue and imaging files.

def __setup_sami_config():
    # Catalogue files.
    sami_config = configparser.ConfigParser()

    # Attempt to read the existing configuration from the user.
    __CONFIG_SECTIONS__ = [
        'Catalogue0000', 'Catalogue0001', 'Catalogue0002']
    sami_config_file = os.path.join(os.environ['HOME'], '.sami-package/sami_default.config')
    default_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                       'sami_default.config')

    if os.path.isfile(sami_config_file):
        sami_config.read(sami_config_file)

        default_config = configparser.ConfigParser()
        default_config.read(default_config_file)

        print(sami_config.sections(), default_config.sections())
        print(sami_config_file, default_config_file)

        if sami_config.sections() != default_config.sections():
            warn_message = (
                ('The default configuration file ({}) and the user configuration'
                 + ' file ({}) have different structure.\n').format(
                 default_config_file, sami_config_file)
                + 'If this is not intended please exit and inspect the files before'
                + ' proceeding with the reduction.')
            warnings.warn(warn_message, RuntimeWarning)
    else: # If no configuration file is available:
        try:
            os.makedirs(os.path.join(os.environ['HOME'], '.sami-package'))
            shutil.copy(default_config_file, sami_config_file)
            print('Successfully created file {}'.format(sami_config_file),
                  ('You can edit this file to set the path of catalogue and image'
                   + ' files.'))
            sami_config.read(sami_config_file)
        except:
            print('Failed to create file {}'.format(sami_config_file),
                  'Using {}'.format(default_config_file))
            sami_config.read(default_config_file)

    return sami_config



def __setup_catalogues_and_imaging(input_config):

    __catalg_ids__ = filter(lambda x: re.match('Catalogue[0-9]{,4}', x),
                            input_config.sections())
    __CATALOGUES__ = [input_config.get(cat_id, 'src_name') for cat_id in __catalg_ids__]
    __CAT_FILENM__ = [
        os.path.join(input_config.get(cat_id, 'cat_path'),
                     input_config.get(cat_id, 'cat_name')) for cat_id in __catalg_ids__]
    __CATALOGUES__ = dict(zip(__CATALOGUES__, __CAT_FILENM__))

   
    __PHOTOMETRY__ = [input_config.get(cat_id, 'src_name') for cat_id in __catalg_ids__]
    __PHOT_FILEN__ = [
        os.path.join(input_config.get(cat_id, 'img_path'),
                     input_config.get(cat_id, 'img_name')) for cat_id in __catalg_ids__]
    __PHOTOMETRY__ = dict(zip(__PHOTOMETRY__, __PHOT_FILEN__))

    __PHOTEXTNUM__ = [input_config.get(cat_id, 'src_name') for cat_id in __catalg_ids__]
    __PHOT_EXTNU__ = [int(input_config.get(cat_id, 'ext_numb')) for cat_id in __catalg_ids__]
    __PHOTEXTNUM__ = dict(zip(__PHOTEXTNUM__, __PHOT_EXTNU__))

    return __CATALOGUES__, __PHOTOMETRY__, __PHOTEXTNUM__


# Setup the relevant variables.
sami_config = __setup_sami_config()

__CATALOGUES__, __PHOTOMETRY__, __PHOTEXTNUM__ = __setup_catalogues_and_imaging(sami_config)
