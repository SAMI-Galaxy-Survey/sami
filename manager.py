"""
Code for organising and reducing SAMI data.

Instructions on how to use this module are given in the docstring for the
Manager class. The following describes some of the under-the-hood details.

This module contains two classes: Manager and FITSFile. The Manager stores
information about an observing run, including a list of all raw files. Each
FITSFile object stores information about a particular raw file. Note that
neither object stores whether or not a file has been reduced; this is checked
on the fly when necessary.

When a Manager object is initiated, it makes an empty list to store the raw
files. It will then inspect given directories to find raw files, with names of
like 01jan10001.fits. It will reject duplicate filenames. Each valid filename
is used to initialise a FITSFile object, which is added to the Manager's file
list. The file itself is also moved into a suitable location in the output
directory structure.

Each FITSFile object stores basic information about the file, such as the path
to the raw file and to the reduced file. The plate and field IDs are
determined automatically from the FITS headers. A check is made to see if the
telescope was pointing at the field listed in the MORE.FIBRES_IFU extension.
If not, the user is asked to give a name for the pointing, which will
generally be the name of whatever object was being observed. This name is then
added to an "extra" list in the Manager, so that subsequent observations at
the same position will be automatically recognised.

The Manager also keeps lists of the different dark frame exposure lengths (as
both string and float), as well as a list of directories that have been
recently reduced, and hence should be visually checked.

2dfdr is controlled via the tdfdr module. Almost all data reduction steps are
run in parallel, creating a Pool as it is needed.

As individual files are reduced, entries are added to the checklist of
directories to visually inspect. There are some functions for loading up 2dfdr
in the relevant directories, but the user must select and plot the individual
files themself. This whole system is a bit clunky and needs overhauling.

There are a few generators for useful items, most notably Manager.files. This
iterates through all entries in the internal file list and yields those that
satisfy a wide range of optional parameters.

The Manager class can be run in demo mode, in which no actual data reduction
is done. Instead, the pre-calculated results are simply copied into the output
directories. This is useful for demonstrating how to use the Manager without
waiting for the actual data reduction to happen.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from typing import List, Tuple, Dict, Sequence

import shutil
import os
import re
import multiprocessing

import warnings
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict
from getpass import getpass
from time import sleep
from glob import glob
from pydoc import pager
import itertools
import traceback
import datetime

import six
from six.moves import input

# Set up logging
from . import slogging

log = slogging.getLogger(__name__)
log.setLevel(slogging.WARNING)
# log.enable_console_logging()

import astropy.coordinates as coord
from astropy import units
import astropy.io.fits as pf
from astropy import __version__ as ASTROPY_VERSION
import numpy as np

try:
    import pysftp

    PYSFTP_AVAILABLE = True
except ImportError:
    PYSFTP_AVAILABLE = False
try:
    from mock import patch

    PATCH_AVAILABLE = True
except ImportError:
    PATCH_AVAILABLE = False

MF_BIN_DIR = '/suphys/nscott/molecfit_install/bin' # directory for molecfit binary files
#MF_BIN_DIR = '/Users/scroom/code/molecfit/bin/' # directory for molecfit binary files
if not os.path.exists(os.path.join(MF_BIN_DIR,'molecfit')):
	warnings.warn('molecfit not found. Disabling improved telluric subtraction')
	MOLECFIT_AVAILABLE = False
else:
	MOLECFIT_AVAILABLE = True

from .utils.other import find_fibre_table, gzip
from .utils import IFU
from .general.cubing import dithered_cubes_from_rss_list, get_object_names
from .general.cubing import dithered_cube_from_rss_wrapper
from .general.cubing import scale_cube_pair, scale_cube_pair_to_mag
from .general.align_micron import find_dither
from .dr import fluxcal2, telluric, check_plots, tdfdr, dust, binning
from .dr.throughput import make_clipped_thput_files
from .qc.fluxcal import stellar_mags_cube_pair, stellar_mags_frame_pair
from .qc.fluxcal import throughput, get_sdss_stellar_mags, identify_secondary_standard
from .qc.sky import sky_residuals
from .qc.arc import bad_fibres
from .dr.fflat import correct_bad_fibres
from .dr.twilight_wavecal import wavecorr_frame, wavecorr_av, apply_wavecorr

# Temporary edit. Prevent bottleneck 1.0.0 being used.
try:
    import bottleneck.__version__ as BOTTLENECK_VERSION
except ImportError:
    BOTTLENECK_VERSION = ''

if BOTTLENECK_VERSION == '1.0.0':
    raise ImportError('Bottleneck {} has a Blue Whale sized bug. Please update your library NOW')

# Get the astropy version as a tuple of integers
ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))
if ASTROPY_VERSION[:2] == (0, 2):
    ICRS = coord.ICRSCoordinates
    warnings.warn('Support for astropy {} is being phased out. Please update your software!'.format(ASTROPY_VERSION))
elif ASTROPY_VERSION[:2] == (0, 3):
    ICRS = coord.ICRS
    warnings.warn('Support for astropy {} is being phased out. Please update your software!'.format(ASTROPY_VERSION))
else:
    def ICRS(*args, **kwargs):
        return coord.SkyCoord(*args, frame='icrs', **kwargs)


IDX_FILES_SLOW = {'580V': 'sami580V_v1_7.idx',
                  '1500V': 'sami1500V_v1_5.idx',
                  '1000R': 'sami1000R_v1_7.idx'}

IDX_FILES_FAST = {'580V': 'sami580V.idx',
                  '1500V': 'sami1500V.idx',
                  '1000R': 'sami1000R.idx'}
IDX_FILES = {'fast': IDX_FILES_FAST,
             'slow': IDX_FILES_SLOW}

GRATLPMM = {'580V': 582.0,
            '1500V': 1500.0,
            '1000R': 1001.0}

CATALOG_PATH = "./gama_catalogues/"

# This list is used for identifying field numbers in the pilot data.
PILOT_FIELD_LIST = [
    {'plate_id': 'run_6_star_P1', 'field_no': 1,
     'coords': '18h01m54.38s -22d46m49.1s'},
    {'plate_id': 'run_6_star_P1', 'field_no': 2,
     'coords': '21h12m25.06s +04d14m59.6s'},
    {'plate_id': 'run_6_P1', 'field_no': 1,
     'coords': '00h41m35.46s -09d40m29.9s'},
    {'plate_id': 'run_6_P1', 'field_no': 2,
     'coords': '01h13m02.16s +00d26m42.2s'},
    {'plate_id': 'run_6_P1', 'field_no': 3,
     'coords': '21h58m30.77s -08d09m23.9s'},
    {'plate_id': 'run_6_P2', 'field_no': 2,
     'coords': '01h16m01.24s +00d03m23.4s'},
    {'plate_id': 'run_6_P2', 'field_no': 3,
     'coords': '21h55m37.75s -07d40m58.3s'},
    {'plate_id': 'run_6_P3', 'field_no': 2,
     'coords': '01h16m19.66s +00d17m46.9s'},
    {'plate_id': 'run_6_P3', 'field_no': 3,
     'coords': '21h56m37.34s -07d32m16.2s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 1,
     'coords': '20h04m08.32s +07d16m40.6s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 2,
     'coords': '23h14m36.57s +12d45m20.6s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 3,
     'coords': '02h11m46.77s -08d56m09.0s'},
    {'plate_id': 'run_7_star_P1', 'field_no': 4,
     'coords': '05h32m00.40s -00d17m56.9s'},
    {'plate_id': 'run_7_P1', 'field_no': 1,
     'coords': '21h58m27.59s -07d43m50.7s'},
    {'plate_id': 'run_7_P1', 'field_no': 2,
     'coords': '00h40m12.73s -09d31m47.5s'},
    {'plate_id': 'run_7_P2', 'field_no': 1,
     'coords': '21h56m27.49s -07d12m02.4s'},
    {'plate_id': 'run_7_P2', 'field_no': 2,
     'coords': '00h40m33.40s -09d04m21.6s'},
    {'plate_id': 'run_7_P3', 'field_no': 1,
     'coords': '21h56m27.86s -07d46m17.1s'},
    {'plate_id': 'run_7_P3', 'field_no': 2,
     'coords': '00h41m25.78s -09d17m14.4s'},
    {'plate_id': 'run_7_P4', 'field_no': 1,
     'coords': '21h57m48.55s -07d23m40.6s'},
    {'plate_id': 'run_7_P4', 'field_no': 2,
     'coords': '00h42m34.09s -09d12m08.1s'}]

# Things that should be visually checked
# Priorities: Lower numbers (more negative) should be done first
# Each key ('TLM', 'ARC',...) matches to a check method named 
# check_tlm, check_arc,...
CHECK_DATA = {
    'BIA': {'name': 'Bias',
            'ndf_class': 'BIAS',
            'spectrophotometric': None,
            'priority': -3,
            'group_by': ('ccd', 'date')},
    'DRK': {'name': 'Dark',
            'ndf_class': 'DARK',
            'spectrophotometric': None,
            'priority': -2,
            'group_by': ('ccd', 'exposure_str', 'date')},
    'LFL': {'name': 'Long-slit flat',
            'ndf_class': 'LFLAT',
            'spectrophotometric': None,
            'priority': -1,
            'group_by': ('ccd', 'date')},
    'TLM': {'name': 'Tramline map',
            'ndf_class': 'MFFFF',
            'spectrophotometric': None,
            'priority': 0,
            'group_by': ('date', 'ccd', 'field_id')},
    'ARC': {'name': 'Arc reduction',
            'ndf_class': 'MFARC',
            'spectrophotometric': None,
            'priority': 1,
            'group_by': ('date', 'ccd', 'field_id')},
    'FLT': {'name': 'Flat field',
            'ndf_class': 'MFFFF',
            'spectrophotometric': None,
            'priority': 2,
            'group_by': ('date', 'ccd', 'field_id')},
    'SKY': {'name': 'Twilight sky',
            'ndf_class': 'MFSKY',
            'spectrophotometric': None,
            'priority': 3,
            'group_by': ('date', 'ccd', 'field_id')},
    'OBJ': {'name': 'Object frame',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': None,
            'priority': 4,
            'group_by': ('date', 'ccd', 'field_id', 'name')},
    'FLX': {'name': 'Flux calibration',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': True,
            'priority': 5,
            'group_by': ('date', 'field_id', 'name')},
    'TEL': {'name': 'Telluric correction',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': None,
            'priority': 6,
            'group_by': ('date', 'ccd', 'field_id')},
    'ALI': {'name': 'Alignment',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': False,
            'priority': 7,
            'group_by': ('field_id',)},
    'CUB': {'name': 'Cubes',
            'ndf_class': 'MFOBJECT',
            'spectrophotometric': False,
            'priority': 8,
            'group_by': ('field_id',)}}
# Extra priority for checking re-reductions
PRIORITY_RECENT = 100

STELLAR_MAGS_FILES = [
    ('standards/secondary/APMCC_0917_STARS.txt', 'ATLAS',
     (0.076, 0.059, 0.041, 0.030, 0.023)),
    ('standards/secondary/Abell_3880_STARS.txt', 'ATLAS',
     (0.064, 0.050, 0.034, 0.025, 0.019)),
    ('standards/secondary/Abell_4038_STARS.txt', 'ATLAS',
     (0.081, 0.063, 0.044, 0.033, 0.024)),
    ('standards/secondary/EDCC_0442_STARS.txt', 'ATLAS',
     (0.071, 0.052, 0.038, 0.029, 0.020)),
    ('standards/secondary/Abell_0085.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    ('standards/secondary/Abell_0119.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    ('standards/secondary/Abell_0168.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    ('standards/secondary/Abell_2399.fstarcat.txt', 'SDSS_cluster',
     (0.0, 0.0, 0.0, 0.0, 0.0)),
    ('standards/secondary/sdss_stellar_mags.csv', 'SDSS_GAMA',
     (0.0, 0.0, 0.0, 0.0, 0.0))]


def stellar_mags_files():
    """Yield details of each stellar magnitudes file that can be found."""
    for mags_file in STELLAR_MAGS_FILES:
        # The pre-determined ones listed above
        yield mags_file
    for path in glob('standards/secondary/sdss_stellar_mags_*.csv'):
        # Extra files that have been downloaded by the user
        yield (path, 'SDSS_GAMA', (0.0, 0.0, 0.0, 0.0, 0.0))


class Manager:
    """Object for organising and reducing SAMI data.

    Initial setup
    =============

    You start a new manager by creating an object, and telling it where
    to put its data, e.g.:

    >>> import sami
    >>> mngr = sami.manager.Manager('130305_130317')

    The directory name you give it should normally match the dates of the
    observing run that you will be reducing.

    IMPORTANT: When starting a new manager, the directory you give it should be 
    non-existent or empty. Inside that directory it will create "raw" and
    "reduced" directories, and all the subdirectories described on the SAMI wiki
    (http://sami-survey.org/wiki/staging-disk-file-structure). You do not need
    to create the subdirectories yourself.

    Note that the manager carries out many I/O tasks that will fail if you
    change directories during your python session. If you do need to change
    directory, make sure you change back before running any further manager
    tasks. This limitation may be addressed in a future update, if enough people
    want it. You may be able to avoid it by providing an absolute path when you
    create the manager, but no guarantee is made.

    By default, the manager will perform full science-grade reductions, which
    can be quite slow. If you want quick-look reductions only (e.g. if you are
    at the telescope, or for testing purposes), then set the 'fast' keyword:

    >>> mngr = sami.manager.Manager('130305_130317', fast=True)

    At this point the manager is not aware of any actual data - skip to
    "Importing data" and carry on from there.

    Deriving the tram-line maps from the twilight sky frames (blue arm only)
    ========================================================================

    The keyword `use_twilight_tlm_blue` instructs the manager to use the
    twilight sky frames to derive the tram-line maps (default value is `False`).
    For the blue arm, using tram-line maps derived from the twilight sky frames
    reduces the noise at the blue end of the spectra.

    When this keyword is set to `True`, two sets of tram-line maps are derived:
    one contains the tram-line maps from each twilight sky frame, and one
    contains the tram-line maps derived from each dome flat frame. The tram-line
    maps derived from the dome flats are used: a) for the red arm in *any* case 
    b) for the blue arm if no twilight frame was available to derive the
    tram-line maps.

    >>> mngr = sami.manager.Manager('130305_130317',use_twilight_tlm_blue=True)

    The reductions will search for a twilight from the same plate to use as
    a TLM file.  If one from the same plate cannot be found, a twilight from another
    plate (or another night) will be used in preference to a dome flat.  The current
    default is use_twilight_tlm_blue=False until full testing has been completed.

    For the on site data reduction, it might be advisable to use `False`
    (default), because this requires less time.
    
    Improving the blue arm wavelength calibration using twilight sky frames
    =======================================================================
    
    The keyword 'improve_blue_wavecorr' instructs the manager to determine
    an improved blue arm wavelength solution from the twilight sky frames.
    
    When this keyword is set to true, the reduced twilight sky frame spectra
    are compared to a high-resolution solar spectrum (supplied as part of the
    SAMI package) to determine residual wavelength shifts. An overall
    fibre-to-fibre wavelength offset is derived by averaging over all twilight
    sky frames in a run. This average offset is stored in a calibration file in
    the root directory of the run. These shifts are then applied to all arc frames. 

    Applying telluric correction to primary standards before flux calibration
    =========================================================================

    The keyword 'telluric_correct_primary' instructs the manager to use molecfit
    to telluric correct the primary standard stars before determining the flux
    calibration transfer function. This is only applied if molecfit is installed.

    This keyword should only be set if the reference spectra for the primary
    standard stars have themselves been telluric corrected. If they have not this
    will result in highly unphysical transfer functions.

    Continuing a previous session
    =============================

    If you quit python and want to get back to where you were, just restart
    the manager on the same directory (after importing sami):

    >>> mngr = sami.manager.Manager('130305_130317')

    It will search through the subdirectories and restore its previous
    state. By default it will restore previously-assigned object names that
    were stored in the headers. To re-assign all names:

    >>> mngr = sami.manager.Manager('data_directory', trust_header=False)

    As before, set the 'fast' keyword if you want quick-look reductions.

    Importing data
    ==============

    After creating the manager, you can import data into it:

    >>> mngr.import_dir('path/to/raw/data')

    It will copy the data into the data directory you defined earlier,
    putting it into a neat directory structure. You will typically need to
    import all of your data before you start to reduce anything, to ensure
    you have all the bias, dark and lflat frames.

    When importing data, the manager will do its best to work out what the
    telescope was pointing at in each frame. Sometimes it wont be able to and
    will ask you for the object name to go with a particular file. Depending on
    the file, you should give an actual object name - e.g. HR7950 or NGC2701 -
    or a more general description - e.g. SNAFU or blank_sky. If the telescope
    was in fact pointing at the field specified by the configuration .csv file -
    i.e. the survey field rather than some extra object - then enter main. It
    will also ask you which of these objects should be used as
    spectrophotometric standards for flux calibration; simply enter y or n as
    appropriate.

    Importing at the AAT
    ====================

    If you're at the AAT and connected to the network there, you can import
    raw data directly from the data directories there:

    >>> mngr.import_aat()

    The first time you call this function you will be prompted for a username
    and password, which are saved for future times. By default the data from
    the latest date is copied; if you want to copy earlier data you can
    specify the date:

    >>> mngr.import_aat(date='140219')

    Only new files are copied over, so you can use this function to update
    your manager during the night. An ethernet connection is strongly
    recommended, as the data transfer is rather slow over wifi.

    Reducing bias, dark and lflat frames
    ====================================

    The standard procedure is to reduced all bias frames and combine them,
    then reduce all darks and combine them, then reduce all long-slit flats
    and combine them. To do this:

    >>> mngr.reduce_bias()
    >>> mngr.combine_bias()
    >>> mngr.reduce_dark()
    >>> mngr.combine_dark()
    >>> mngr.reduce_lflat()
    >>> mngr.combine_lflat()

    The manager will put in symbolic links as it goes, to ensure the
    combined files are available wherever they need to be.

    If you later import more of the above frames, you'll need to re-run the
    above commands to update the combined frames.

    In these (and later) commands, by default nothing happens if the
    reduced file already exists. To override this behaviour, set the
    overwrite keyword, e.g.

    >>> mngr.reduce_bias(overwrite=True)

    If you later import more data that goes into new directories, you'll
    need to update the links:

    >>> mngr.link_bias()
    >>> mngr.link_dark()
    >>> mngr.link_lflat()

    Reducing fibre flat, arc and offset sky frames
    ==============================================

    2dfdr works by creating a tramline map from the fibre flat fields, then
    reducing the arc frames, then re-reducing the fibre flat fields. After
    this the offset skies (twilights) can also be reduced.

    >>> mngr.make_tlm()
    >>> mngr.reduce_arc()
    >>> mngr.reduce_fflat()
    >>> mngr.reduce_sky()

    In any of these commands, keywords can be used to restrict the files
    that get reduced, e.g.

    >>> mngr.reduce_arc(ccd='ccd_1', date='130306',
                        field_id='Y13SAR1_P002_09T004')

    will only reduce arc frames for the blue CCD that were taken on the
    6th March 2013 for the field ID Y13SAR1_P002_09T004. Allowed keywords
    and examples are:
    
        date            '130306'
        plate_id        'Y13SAR1_P002_09T004_12T006'
        plate_id_short  'Y13SAR1_P002'
        field_no        0
        field_id        'Y13SAR1_P002_09T004'
        ccd             'ccd_1'
        exposure_str    '10'
        min_exposure    5.0
        max_exposure    20.0
        reduced_dir     ('130305_130317/reduced/130305/'
                         'Y13SAR1_P001_09T012_15T001/Y13SAR1_P001_09T012/'
                         'calibrators/ccd_1')

    Note the reduced_dir is a single long string, in this example split
    over several lines.

    Reducing object frames
    ======================

    Once all calibration files have been reduced, object frames can be
    reduced in a similar manner:

    >>> mngr.reduce_object()

    This function takes the same keywords as described above for fibre
    flats etc.

    Flux calibration
    ================

    The flux calibration requires a series of steps. First, a transfer
    function is derived for each observation of a standard star:

    >>> mngr.derive_transfer_function()

    Next, transfer functions from related observations are combined:

    >>> mngr.combine_transfer_function()

    The combined transfer functions can then be applied to the data to
    flux calibrate it:

    >>> mngr.flux_calibrate()

    Finally, the telluric correction is derived and applied with a
    single command:

    >>> mngr.telluric_correct()

    As before, keywords can be set to restrict which files are processed,
    and overwrite=True can be set to force re-processing of files that
    have already been done.

    A option is to flux calibrate from the secondary stars.  This can 
    have several advantages.  For example, it means that any residual 
    extinction variations can be removed on a frame-by-frame basis.  It can
    also provide better calibration when the pirmary standards have too much
    scattered light in the blue.  This can cause PSF fitting problems and so
    lead to some systematics at trhe far blue end (can be 10-20%).  To fit
    using the secondaries we need to match the star to a model (Kurucz 
    theoretical models) and we do this using ppxf to find the best model. 
    The model used is a linear combination of the 4 templates closest in
    Teff and [Fe/H] to the observed star.  The best template is estimated
    from all the frames in the field.  This is then use to estimate
    the transfer function for inditivual frames.  There is also an option
    to average the TF across all the frames used.  The model star spectrum
    is also scaled to the SDSS photometry so that application of the TF 
    also does the equivalent of the scale_frames() proceedure, so this
    should not need to be done if fluxcal_secondary() is used.

    >>> mngr.fluxcal_secondary()

    of if averaging:

    >>> mngr.fluxcal_secondary(use_av_tf_sec=True)  (default)

    >>> mngr.fluxcal_secondary(use_av_tf_sec=False)  (or not)

    Scaling frames
    ==============

    To take into account variations in atmospheric transmission over the
    course of a night (or between nights), the flux level in each frame is
    scaled to set the standard star flux to the catalogue level. First the
    manager needs to find out what that catalogue level is:

    >>> mngr.get_stellar_photometry()

    This will check the stars in your observing run against the inbuilt
    catalogue. If any are missing you will be prompted to download the
    relevant data from the SDSS website.

    Once the stellar photometry is complete, you can continue with scaling
    each frame:

    >>> mngr.scale_frames()
    
    Cubing
    ======

    Before the frames can be combined their relative offsets must be
    measured:

    >>> mngr.measure_offsets()

    The individual calibrated spectra can then be turned into datacubes:

    >>> mngr.cube()

    A final rescaling is done, to make sure everything is on the correct
    flux scale:

    >>> mngr.scale_cubes()

    If you have healpy installed and the relevant dust maps downloaded, you
    can record the E(B-V) and attenuation curves in the datacubes:

    >>> mngr.record_dust()

    It's safe to leave this step out if you don't have the necessary
    components.

    Finally, the cubes can be gzipped to save space/bandwidth. You might
    want to leave this until after the output checking (see below), to
    improve read times.

    >>> mngr.gzip_cubes()

    Checking outputs
    ================

    As the reductions are done, the manager keeps a record of reduced
    files that need to be plotted to check that the outputs are ok. These 
    are grouped in sets of related files. To print the checks that need to 
    be done:

    >>> mngr.print_checks()

    Separate lists are returned for checks that have never been done, and
    those that haven't been done since the last re-reduction. You can
    specify one or the other by passing 'ever' or 'recent' as an argument
    to print_checks().

    To perform the next check:

    >>> mngr.check_next_group()

    The manager will either load the 2dfdr GUI and give you a list of
    files to check, or will make some plots in python. Check the things it
    tells you to, keeping a note of any files that need to be disabled or
    examined further (if you don't know what to do about a file, ask a
    friendly member of the data reduction working group).

    If 2dfdr was loaded, no more commands can be entered until you close it. 
    When you do so, or immediately for python-based plots, you will be asked
    whether the files can be removed from the checklist. Enter 'y'
    if you have checked all the files, 'n' otherwise.

    You can also perform a particular check by specifying its index in the
    list:

    >>> mngr.check_group(3)

    The index numbers are given when mngr.print_checks() is called.

    Checking the outputs is a crucial part of data reduction, so please
    make sure you do it thoroughly, and ask for assistance if you're not
    sure about anything.

    Disabling files
    ===============

    If there are any problems with some files, either because an issue is
    noted in the log or because they wont reduce properly, you can disable
    them, preventing them from being used in any later reductions:

    >>> mngr.disable_files(['06mar10003', '06mar20003.fits', '06mar10047'])

    If you only have one file you want to disable, you still need the
    square brackets. The filenames can be with or without the extension (.fits)
    but must be without the directory. You can disable lots of files at a time
    using the files generator:

    >>> mngr.disable_files(mngr.files(
                date='130306', field_id='Y13SAR1_P002_09T004'))

    This allows the same keywords as described earlier, as well as:

        ndf_class           'MFFFF'
        reduced             False
        tlm_created         False
        flux_calibrated     True
        telluric_corrected  True
        name                'LTT2179'

    For example, specifying the first three of these options as given
    would disable all fibre flat fields that had not yet been reduced and
    had not yet had tramline maps created. Specifying the last three
    would disable all observations of LTT2179 that had already been flux
    calibrated and telluric corrected.

    To re-enable files:

    >>> mngr.enable_files(['06mar10003', '06mar20003', '06mar10047'])

    This function follows exactly the same syntax as disable_files.

    Summarising results
    ===================

    At any time you can print out a summary of the object frames observed,
    including some basic quality control metrics:

    >>> mngr.qc_summary()

    The QC values are not filled in until certain steps of the data
    reduction have been done; you need to get as far as mngr.scale_frames()
    to see everything. While observing, keep an eye on the seeing and the
    transmission. If the seeing is above 3" or the transmission is below
    about 0.7, the data are unlikely to be of much use.

    Changing object names and spectrophotometric flags
    ==================================================

    If you want to change the object names for one or more files, or change
    whether they should be used as spectrophotometric standards, use the
    following commands:

    >>> mngr.update_name(['06mar10003', '06mar20003'], 'new_name')
    >>> mngr.update_spectrophotometric(['06mar10003', '06mar20003'], True)

    In the above example, the given files are set to have the name
    'new_name' and they are listed as spectrophotometric standards. The
    options for spectrophotometric flags must be entered as True or
    False (without quote marks, with capital letter). You can use the
    same file generator syntax as for disabling/enabling files
    (above), so for example if you realise that on importing some
    files you entered LTT2197 instead of LTT2179 you can correct all
    affected files at once:

    >>> mngr.update_name(mngr.files(name='LTT2197'), 'LTT2179')

    Changing speed/accuracy of the reductions
    =========================================

    If you want to switch between fast and slow (rough vs accurate) reductions:

    >>> mngr.change_speed()

    Or to ensure you end up with a particular speed, specify 'fast' or 'slow':

    >>> mngr.change_speed('slow')

    Reducing everything in one go
    =============================

    You can perform all of the above reduction tasks with a single command:

    >>> mngr.reduce_all()

    You should only do this for re-reducing data that you have previously
    checked and are confident that nothing will go wrong, and after
    disabling all unwanted files. Otherwise, you can easily have a tramline
    map go haywire (for example) and wreck everything that follows.

    Parallel processing
    ===================

    You can make use of multi-core machines by setting the number of CPUs to
    use when the manager is made, e.g.:

    >>> mngr = sami.manager.Manager('130305_130317', n_cpu=4)

    Note that you cannot run multiple instances of 2dfdr in the same
    directory, so you wont always be able to use all your cores. To keep
    track of where 2dfdr is running, the manager makes empty directories
    called '2dfdrLockDir'. These will normally be cleaned up when 2dfdr
    completes, but after a bad crash they may be left behind and block
    any following reductions. In this case, you can force their removal:

    >>> mngr.remove_directory_locks()

    Including data from other runs
    ==============================

    If a field has been observed over more than one run, the manager will
    need to be made aware of the pre-existing data to make combined
    datacubes. Note that this is only necessary for the final data
    reduction, so observers do not need to worry about this.

    To combine the data, first create a manager for each run (you may
    already have done this):

    >>> mngr = sami.manager.Manager('2014_04_24-2014_05_04')
    >>> mngr_old = sami.manager.Manager('2014_05_23-2014_06_01')

    Then create the link:

    >>> mngr.link_manager(mngr_old)

    Now `mngr` will include files from `mngr_old` when necessary, i.e. for
    these steps:

    >>> mngr.measure_offsets()
    >>> mngr.cube()
    >>> mngr.scale_cubes()
    >>> mngr.bin_cubes()

    For all previous steps the two managers still act independently, so
    you need to follow through up to scale_frames() for each manager
    individually.

    Other functions
    ===============

    The other functions defined probably aren't useful to you.
    """

    # Task list provides the list of standard reduction tasks in the necessary
    # order. This is used by `reduce_all`, and also by each reduction step to provide instructions on the next step to run.
    task_list = (
        ('reduce_bias', True),
        ('combine_bias', False),
        ('reduce_dark', True),
        ('combine_dark', False),
        ('reduce_lflat', True),
        ('combine_lflat', False),
        ('make_tlm', True),
        ('reduce_arc', True),
        ('reduce_fflat', True),
        ('reduce_sky', True),
        ('reduce_object', True),
        ('derive_transfer_function', True),
        ('combine_transfer_function', True),
        ('flux_calibrate', True),
        ('telluric_correct', True),
        ('fluxcal_secondary',True),
        ('scale_frames', True),
        ('measure_offsets', True),
        ('cube', True),
        #('scale_cubes', True),
        ('bin_cubes', True),
        ('record_dust', True),
        ('bin_aperture_spectra', True),
        ('gzip_cubes', True)
    )

    def __init__(self, root, copy_files=False, move_files=False, fast=False,
                 gratlpmm=GRATLPMM, n_cpu=1,demo_data_source='demo',
                 use_twilight_tlm_blue=False, use_twilight_flat_blue=False,
                 improve_blue_wavecorr=False, telluric_correct_primary=False, debug=False):
        if fast:
            self.speed = 'fast'
        else:
            self.speed = 'slow'
        self.idx_files = IDX_FILES[self.speed]
        # define the internal flag that allows twilights to be used for
        # making tramline maps:
        self.use_twilight_tlm_blue = use_twilight_tlm_blue
        # define the internal flag that allows twilights to be used for
        # fibre flat fielding:
        self.use_twilight_flat_blue = use_twilight_flat_blue
        # define the internal flag that specifies the improved twlight wavelength
        # calibration step should be applied
        self.improve_blue_wavecorr = improve_blue_wavecorr
        # Internal flag to set telluric correction for primary standards
        self.telluric_correct_primary = telluric_correct_primary
        self.gratlpmm = gratlpmm
        self.n_cpu = n_cpu
        self.root = root
        self.abs_root = os.path.abspath(root)
        self.tmp_dir = os.path.join(self.abs_root, 'tmp')
        # Match objects within 1'
        if ASTROPY_VERSION[0] == 0 and ASTROPY_VERSION[1] == 2:
            self.matching_radius = coord.AngularSeparation(
                0.0, 0.0, 0.0, 1.0, units.arcmin)
        else:
            self.matching_radius = coord.Angle('0:1:0 degrees')
        self.file_list = []
        self.extra_list = []
        self.dark_exposure_str_list = []
        self.dark_exposure_list = []
        self.linked_managers = []
        self.cwd = os.getcwd()
        if 'IMP_SCRATCH' in os.environ:
            self.imp_scratch = os.environ['IMP_SCRATCH']
        else:
            self.imp_scratch = None
        self.scratch_dir = None
        self.min_exposure_for_throughput = 900.0
        self.min_exposure_for_sky_wave = 900.0
        self.aat_username = None
        self.aat_password = None
        self.inspect_root(copy_files, move_files)
        if self.find_directory_locks():
            print('Warning: directory locks in place!')
            print('If this is because you killed a crashed manager, clean them')
            print('up using mngr.remove_directory_locks()')

        if use_twilight_tlm_blue:
            print('Using twilight frames to derive TLM and profile map')
        else:
            print('NOT using twilight frames to derive TLM and profile map')

        if use_twilight_flat_blue:
            print('Using twilight frames for fibre flat field')
        else:
            print('NOT using twilight frames for fibre flat field')

        if improve_blue_wavecorr:
            print('Applying additional twilight-based wavelength calibration step')
        else:
            print('NOT applying additional twilight-based wavelength calibration step')

        if telluric_correct_primary:
            print('Applying telluric correction to primary standard stars before flux calibration')
            print('WARNING: Only do this if the reference spectra for the primary standards have good telluric correction')
        else:
            print('NOT applying telluric correction to primary standard stars')

        self._debug = False
        self.debug = debug

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            raise ValueError("debug must be set to a boolean value.")
        if not value == self._debug:
            if value:
                log.setLevel(slogging.DEBUG)
                tdfdr.log.setLevel(slogging.DEBUG)
            else:
                log.setLevel(slogging.WARNING)
                tdfdr.log.setLevel(slogging.WARNING)
            self._debug = value

    def next_step(self, step, print_message=False):
        task_name_list = list(map(lambda x: x[0], self.task_list))
        current_index = task_name_list.index(step)
        if current_index + 1 < len(task_name_list):
            next_step = task_name_list[current_index + 1]
        else:
            # nothing left
            next_step = None
        if print_message:
            print("'{}' step complete. Next step is '{}'".format(step, next_step))
        return next_step

    def __repr__(self):
        return "SAMIManagerInstance at {}".format(self.root)

    def map(self, function, input_list):
        """Map inputs to a function, using built-in map or multiprocessing."""
        if not input_list:
            # input_list is empty. I expected the map functions to deal with
            # this issue, but in one case it hung on aatmacb, so let's be
            # absolutely sure to avoid the issue
            return []
        # if asyncio.iscoroutinefunction(function):
        #
        #     result_list = []
        #
        #     # loop = asyncio.new_event_loop()
        #     loop = asyncio.get_event_loop()
        #     # Break up the overall job into chunks that are n_cpu in size:
        #     for i in range(0, len(input_list), self.n_cpu):
        #         print("{} jobs total, running {} to {} in parallel".format(len(input_list), i, min(i+self.n_cpu, len(input_list))))
        #         # Create an awaitable object which can be used as a future.
        #         # This is the job that will be run in parallel.
        #         @asyncio.coroutine
        #         def job():
        #             tasks = [function(item) for item in input_list[i:i+self.n_cpu]]
        #             # for completed in asyncio.as_completed(tasks):  # print in the order they finish
        #             #     await completed
        #             #     # print(completed.result())
        #             sub_results = yield from asyncio.gather(*tasks, loop=loop)
        #             result_list.extend(sub_results)
        #
        #         loop.run_until_complete(job())
        #     # loop.close()
        #
        #     return np.array(result_list)
        #
        # else:
        # Fall back to using multiprocessing for non-coroutine functions
        if self.n_cpu == 1:
            result_list = list(map(function, input_list))
        else:
            pool = multiprocessing.Pool(self.n_cpu)
            result_list = pool.map(function, input_list, chunksize=1)
            pool.close()
            pool.join()
        return result_list

    def inspect_root(self, copy_files, move_files, trust_header=True):
        """Add details of existing files to internal lists."""
        files_to_add = []
        for dirname, subdirname_list, filename_list in os.walk(os.path.join(self.abs_root, "raw")):
            for filename in filename_list:
                if self.file_filter(filename):
                    full_path = os.path.join(dirname, filename)
                    files_to_add.append(full_path)

        assert len(set(files_to_add)) == len(files_to_add), "Some files would be duplicated on manager startup."

        if self.n_cpu == 1:
            fits_list = list(map(FITSFile, files_to_add))
        else:
            pool = multiprocessing.Pool(self.n_cpu)
            fits_list = pool.map(FITSFile, files_to_add, chunksize=20)
            pool.close()
            pool.join()

        for fits in fits_list:
            self.import_file(fits,
                             trust_header=trust_header,
                             copy_files=copy_files,
                             move_files=move_files)

    def file_filter(self, filename):
        """Return True if the file should be added."""
        # Match filenames of the form 01jan10001.fits
        return (re.match(r'[0-3][0-9]'
                         r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
                         r'[1-2][0-9]{4}\.(fit|fits|FIT|FITS)$',
                         filename)
                and (self.fits_file(filename) is None))

    def import_file(self, source,
                    trust_header=True, copy_files=True, move_files=False):
        """Add details of a file to the manager"""
        if not isinstance(source, FITSFile):
            # source_path = os.path.join(dirname, filename)
            # Initialize an instance of the FITSFile:
            filename = os.path.basename(source)
            fits = FITSFile(source)
        else:
            filename = source.filename
            fits = source
        if fits.copy:
            # print 'this is a copy, do not import:',dirname,filename
            # This is a copy of a file, don't add it to the list
            return
        if fits.ndf_class not in [
            'BIAS', 'DARK', 'LFLAT', 'MFFFF', 'MFARC', 'MFSKY',
            'MFOBJECT']:
            print('Unrecognised NDF_CLASS for {}: {}'.format(
                filename, fits.ndf_class))
            print('Skipping this file')
            return
        if fits.ndf_class == 'DARK':
            if fits.exposure_str not in self.dark_exposure_str_list:
                self.dark_exposure_str_list.append(fits.exposure_str)
                self.dark_exposure_list.append(fits.exposure)
        self.set_raw_path(fits)
        if os.path.abspath(fits.source_path) != os.path.abspath(fits.raw_path):
            if copy_files:
                print('Copying file:', filename)
                self.update_copy(fits.source_path, fits.raw_path)
            if move_files:
                print('Moving file: ', filename)
                self.move(fits.source_path, fits.raw_path)
            if not copy_files and not move_files:
                print('Warning! Adding', filename, 'in unexpected location')
                fits.raw_path = fits.source_path
        else:
            print('Adding file: ', filename)
        self.set_name(fits, trust_header=trust_header)
        fits.set_check_data()
        self.set_reduced_path(fits)
        if not fits.do_not_use:
            fits.make_reduced_link()
        if fits.grating in self.gratlpmm:
            try:
                fits.add_header_item('GRATLPMM', self.gratlpmm[fits.grating])
            except IOError:
                pass
        if fits.grating not in self.idx_files:
            # Without an idx file we would have no way to reduce this file
            self.disable_files([fits])
        self.file_list.append(fits)
        return

    def set_raw_path(self, fits):
        """Set the raw path for a FITS file."""
        if fits.ndf_class == 'BIAS':
            rel_path = os.path.join('bias', fits.ccd, fits.date)
        elif fits.ndf_class == 'DARK':
            rel_path = os.path.join('dark', fits.ccd, fits.exposure_str,
                                    fits.date)
        elif fits.ndf_class == 'LFLAT':
            rel_path = os.path.join('lflat', fits.ccd, fits.date)
        else:
            rel_path = os.path.join(fits.date, fits.ccd)
        fits.raw_dir = os.path.join(self.abs_root, 'raw', rel_path)
        fits.raw_path = os.path.join(fits.raw_dir, fits.filename)
        return

    def update_copy(self, source_path, dest_path):
        """Copy the file, unless a more recent version exists."""
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        elif os.path.exists(dest_path):
            if os.path.getmtime(source_path) <= os.path.getmtime(dest_path):
                # File has already been copied and no update to be done
                return
        shutil.copy2(source_path, dest_path)
        return

    def move(self, source_path, dest_path):
        """Move the file."""
        dest_dir = os.path.dirname(dest_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.move(source_path, dest_path)
        return

    def move_reduced_files(self, filename_root, old_reduced_dir, reduced_dir):
        """Move all reduced files connected to the given root."""
        for filename in os.listdir(old_reduced_dir):
            if filename.startswith(filename_root):
                self.move(os.path.join(old_reduced_dir, filename),
                          os.path.join(reduced_dir, filename))
        # If there is nothing useful left in the old directory, delete it.
        if not self.check_reduced_dir_contents(old_reduced_dir):
            # There's nothing useful in the old directory, so move any
            # remaining files to the new directory and then delete it
            for filename in os.listdir(old_reduced_dir):
                self.move(os.path.join(old_reduced_dir, filename),
                          os.path.join(reduced_dir, filename))
            os.removedirs(old_reduced_dir)
        return

    def set_name(self, fits, trust_header=True):
        """Set the object name for a FITS file."""
        fits.name = None
        fits.spectrophotometric = None
        if fits.ndf_class != 'MFOBJECT':
            # Don't try to set a name for calibration files
            return
        # Check if there's already a name in the header
        try:
            name_header = pf.getval(fits.raw_path, 'MNGRNAME')
        except KeyError:
            name_header = None
        try:
            spectrophotometric_header = pf.getval(fits.raw_path, 'MNGRSPMS')
        except KeyError:
            spectrophotometric_header = None
        # Check if the telescope was pointing in the right direction
        fits_coords = ICRS(
            ra=fits.coords['ra'],
            dec=fits.coords['dec'],
            unit=fits.coords['unit'])
        fits_cfg_coords = ICRS(
            ra=fits.cfg_coords['ra'],
            dec=fits.cfg_coords['dec'],
            unit=fits.cfg_coords['unit'])
        if fits_coords.separation(fits_cfg_coords) < self.matching_radius:
            # Yes it was
            name_coords = 'main'
            spectrophotometric_coords = False
        else:
            # No it wasn't
            name_coords = None
            spectrophotometric_coords = None
        # See if it matches any previous fields
        name_extra = None
        spectrophotometric_extra = None
        for extra in self.extra_list:
            if (fits_coords.separation(extra['coords']) < self.matching_radius):
                # Yes it does
                name_extra = extra['name']
                spectrophotometric_extra = extra['spectrophotometric']
                break
        # Now choose the best name
        if name_header and trust_header:
            best_name = name_header
        elif name_coords:
            best_name = name_coords
        elif name_extra:
            best_name = name_extra
        else:
            # As a last resort, ask the user
            best_name = None
            while best_name is None:
                try:
                    best_name = input('Enter object name for file ' +
                                      fits.filename + '\n > ')
                except ValueError as error:
                    print(error)
        # If there are any remaining bad characters (from an earlier version of
        # the manager), just quietly replace them with underscores
        best_name = re.sub(r'[\\\[\]*/?<>|;:&,.$ ]', '_', best_name)
        fits.update_name(best_name)
        # Now choose the best spectrophotometric flag
        if spectrophotometric_header is not None and trust_header:
            fits.update_spectrophotometric(spectrophotometric_header)
        elif spectrophotometric_coords is not None:
            fits.update_spectrophotometric(spectrophotometric_coords)
        elif spectrophotometric_extra is not None:
            fits.update_spectrophotometric(spectrophotometric_extra)
        else:
            # Ask the user whether this is a spectrophotometric standard
            yn = input('Is ' + fits.name + ' in file ' + fits.filename +
                       ' a spectrophotometric standard? (y/n)\n > ')
            spectrophotometric_input = (yn.lower()[0] == 'y')
            fits.update_spectrophotometric(spectrophotometric_input)
        # If the field was new and it's not a "main", add it to the list
        if name_extra is None and name_coords is None:
            self.extra_list.append(
                {'name': fits.name,
                 'coords': fits_coords,
                 'spectrophotometric': fits.spectrophotometric,
                 'fitsfile': fits})
        return

    def update_name(self, file_iterable, name):
        """Change the object name for a set of FITSFile objects."""
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            # Update the name
            try:
                fits.update_name(name)
            except ValueError as error:
                print(error)
                return
            # Update the extra list if necessary
            for extra in self.extra_list:
                if extra['fitsfile'] is fits:
                    extra['name'] = name
            # Update the path for the reduced files
            if fits.do_not_use is False:
                old_reduced_dir = fits.reduced_dir
                self.set_reduced_path(fits)
                if fits.reduced_dir != old_reduced_dir:
                    # The path has changed, so move all the reduced files
                    self.move_reduced_files(fits.filename_root, old_reduced_dir,
                                            fits.reduced_dir)
        return

    def update_spectrophotometric(self, file_iterable, spectrophotometric):
        """Change the spectrophotometric flag for FITSFile objects."""
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            # Update the flag
            fits.update_spectrophotometric(spectrophotometric)
            # Update the extra list if necessary
            for extra in self.extra_list:
                if extra['fitsfile'] is fits:
                    extra['spectrophotometric'] = spectrophotometric
        return

    def set_reduced_path(self, fits):
        """Set the reduced path for a FITS file."""
        if fits.ndf_class == 'BIAS':
            rel_path = os.path.join('bias', fits.ccd, fits.date)
        elif fits.ndf_class == 'DARK':
            rel_path = os.path.join('dark', fits.ccd, fits.exposure_str,
                                    fits.date)
        elif fits.ndf_class == 'LFLAT':
            rel_path = os.path.join('lflat', fits.ccd, fits.date)
        elif fits.ndf_class in ['MFFFF', 'MFARC', 'MFSKY']:
            rel_path = os.path.join(fits.date, fits.plate_id, fits.field_id,
                                    'calibrators', fits.ccd)
        else:
            rel_path = os.path.join(fits.date, fits.plate_id, fits.field_id,
                                    fits.name, fits.ccd)
        fits.reduced_dir = os.path.join(self.abs_root, 'reduced', rel_path)
        fits.reduced_link = os.path.join(fits.reduced_dir, fits.filename)
        fits.reduced_path = os.path.join(fits.reduced_dir,
                                         fits.reduced_filename)
        # set the tlm_path for MFSKY frames that can be used as a TLM:
        if fits.ndf_class == 'MFSKY':
            fits.tlm_path = os.path.join(fits.reduced_dir, fits.tlm_filename)
        if fits.ndf_class == 'MFFFF':
            fits.tlm_path = os.path.join(fits.reduced_dir, fits.tlm_filename)
        elif fits.ndf_class == 'MFOBJECT':
            fits.fluxcal_path = os.path.join(fits.reduced_dir,
                                             fits.fluxcal_filename)
            fits.telluric_path = os.path.join(fits.reduced_dir,
                                              fits.telluric_filename)
        return

    def import_dir(self, source_dir, trust_header=True):
        """Import the contents of a directory and all subdirectories."""
        for dirname, subdirname_list, filename_list in os.walk(source_dir):
            for filename in filename_list:
                if self.file_filter(filename):
                    tmp_path = os.path.join(self.tmp_dir, filename)
                    self.update_copy(os.path.join(dirname, filename),
                                     tmp_path)
                    self.import_file(
                        os.path.join(self.tmp_dir, filename),
                        trust_header=trust_header,
                        copy_files=False, move_files=True)
                    if os.path.exists(tmp_path):
                        # The import was abandoned; delete the temporary copy
                        os.remove(tmp_path)
        if os.path.exists(self.tmp_dir) and len(os.listdir(self.tmp_dir)) == 0:
            os.rmdir(self.tmp_dir)
        return

    def import_aat(self, username=None, password=None, date=None,
                   server='aatlxa', path='/data_lxy/aatobs/OptDet_data'):
        """Import from the AAT data disks."""
        if os.path.exists(path):
            # Assume we are on a machine at the AAT which has direct access to
            # the data directories
            if date is None:
                date_options = [s for s in os.listdir(path)
                                if (re.match(r'\d{6}', s) and
                                    os.path.isdir(os.path.join(path, s)))]
                date = sorted(date_options)[-1]
            self.import_dir(os.path.join(path, date))
            return

        # Otherwise, it is necessary to SCP!
        with self.connection(server=server, username=username,
                             password=password) as srv:
            if srv is None:
                return
            if date is None:
                date_options = [s for s in srv.listdir(path)
                                if re.match(r'\d{6}', s)]
                date = sorted(date_options)[-1]
            if not os.path.exists(self.tmp_dir):
                os.makedirs(self.tmp_dir)
            for ccd in ['ccd_1', 'ccd_2']:
                dirname = os.path.join(path, date, ccd)
                filename_list = sorted(srv.listdir(dirname))
                for filename in filename_list:
                    if self.file_filter(filename):
                        srv.get(os.path.join(dirname, filename),
                                localpath=os.path.join(self.tmp_dir, filename))
                        self.import_file(
                            os.path.join(self.tmp_dir, filename),
                            trust_header=False, copy_files=False,
                            move_files=True)
        if os.path.exists(self.tmp_dir) and len(os.listdir(self.tmp_dir)) == 0:
            os.rmdir(self.tmp_dir)
        return

    def fits_file(self, filename, include_linked_managers=False):
        """Return the FITSFile object that corresponds to the given filename."""
        filename_options = [filename, filename + '.fit', filename + '.fits',
                            filename + '.FIT', filename + '.FITS']
        if include_linked_managers:
            # Include files from linked managers too
            file_list = itertools.chain(
                self.file_list,
                *[mngr.file_list for mngr in self.linked_managers])
        else:
            file_list = self.file_list
        for fits in file_list:
            if fits.filename in filename_options:
                return fits
        return None

    def check_reduced_dir_contents(self, reduced_dir):
        """Return True if any FITSFile objects point to reduced_dir."""
        for fits in self.file_list:
            if (fits.do_not_use is False and
                    os.path.samefile(fits.reduced_dir, reduced_dir)):
                # There is still something in this directory
                return True
        # Failed to find anything
        return False

    def disable_files(self, file_iterable):
        """Disable (delete links to) files in provided list (or iterable)."""
        if isinstance(file_iterable, str):
            raise ValueError("disable_files must be passed a list of files, e.g., ['07mar10032.fits']")
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            fits.update_do_not_use(True)
            # Delete the reduced directory if it's now empty
            try:
                os.removedirs(fits.reduced_dir)
            except OSError:
                # It wasn't empty - no harm done
                pass
        return

    def enable_files(self, file_iterable):
        """Enable files in provided list (or iterable)."""
        if isinstance(file_iterable, str):
            raise ValueError("enable_files must be passed a list of files, e.g., ['07mar10032.fits']")
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            fits.update_do_not_use(False)
        return

    def link_manager(self, manager):
        """Include data from specified manager when cubing."""
        if manager not in self.linked_managers:
            self.linked_managers.append(manager)
        else:
            print('Already including that manager!')
        return

    def unlink_manager(self, manager):
        """Remove specified manager from list to include when cubing."""
        if manager in self.linked_managers:
            self.linked_managers.remove(manager)
        else:
            print('Manager not in linked list!')
        return

    def bias_combined_filename(self):
        """Return the filename for BIAScombined.fits"""
        return 'BIAScombined.fits'

    def dark_combined_filename(self, exposure_str):
        """Return the filename for DARKcombined.fits"""
        return 'DARKcombined' + exposure_str + '.fits'

    def lflat_combined_filename(self):
        """Return the filename for LFLATcombined.fits"""
        return 'LFLATcombined.fits'

    def bias_combined_path(self, ccd):
        """Return the path for BIAScombined.fits"""
        return os.path.join(self.abs_root, 'reduced', 'bias', ccd,
                            self.bias_combined_filename())

    def dark_combined_path(self, ccd, exposure_str):
        """Return the path for DARKcombined.fits"""
        return os.path.join(self.abs_root, 'reduced', 'dark', ccd, exposure_str,
                            self.dark_combined_filename(exposure_str))

    def lflat_combined_path(self, ccd):
        """Return the path for LFLATcombined.fits"""
        return os.path.join(self.abs_root, 'reduced', 'lflat', ccd,
                            self.lflat_combined_filename())

    def reduce_calibrator(self, calibrator_type, overwrite=False, check=None,
                          **kwargs):
        """Reduce all biases, darks of lflats."""
        self.check_calibrator_type(calibrator_type)
        file_iterable = self.files(ndf_class=calibrator_type.upper(),
                                   do_not_use=False, **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite, check=check)
        return reduced_files

    def combine_calibrator(self, calibrator_type, overwrite=False):
        """Produce and link necessary XXXXcombined.fits files."""
        for ccd, exposure_str, filename, path in self.combined_filenames_paths(
                calibrator_type, do_not_use=False):
            if overwrite and os.path.exists(path):
                # Delete the existing file
                os.remove(path)
            if not os.path.exists(path):
                self.run_2dfdr_combine(
                    self.files(ccd=ccd, exposure_str=exposure_str,
                               ndf_class=calibrator_type.upper(),
                               reduced=True, do_not_use=False),
                    path)
                dirname = os.path.dirname(path)
        self.link_calibrator(calibrator_type, overwrite)
        return

    def link_calibrator(self, calibrator_type, overwrite=False):
        """Make necessary symbolic links for XXXXcombined.fits files."""
        if calibrator_type.lower() == 'bias':
            dir_type_list = ['dark', 'lflat', 'calibrators', 'object',
                             'spectrophotometric']
        elif calibrator_type.lower() == 'dark':
            dir_type_list = ['lflat', 'calibrators', 'object',
                             'spectrophotometric']
        elif calibrator_type.lower() == 'lflat':
            dir_type_list = ['calibrators', 'object', 'spectrophotometric']
        for ccd, exposure_str, filename, path in self.combined_filenames_paths(
                calibrator_type, do_not_use=False):
            for dir_type in dir_type_list:
                for link_dir in self.reduced_dirs(dir_type, ccd=ccd,
                                                  do_not_use=False):
                    link_path = os.path.join(link_dir, filename)
                    if overwrite and (os.path.exists(link_path) or
                                      os.path.islink(link_path)):
                        os.remove(link_path)
                    if (not os.path.exists(link_path)) and os.path.exists(path):
                        os.symlink(os.path.relpath(path, link_dir),
                                   link_path)
        return

    def check_calibrator_type(self, calibrator_type):
        """Raise an exception if that's not a real calibrator type."""
        if calibrator_type.lower() not in ['bias', 'dark', 'lflat']:
            raise ValueError(
                'calibrator type must be "bias", "dark" or "lflat"')
        return

    def reduce_bias(self, overwrite=False, **kwargs):
        """Reduce all bias frames."""
        self.reduce_calibrator(
            'bias', overwrite=overwrite, check='BIA', **kwargs)
        self.next_step('reduce_bias', print_message=True)
        return

    def combine_bias(self, overwrite=False):
        """Produce and link necessary BIAScombined.fits files."""
        self.combine_calibrator('bias', overwrite=overwrite)
        self.next_step('combine_bias', print_message=True)
        return

    def link_bias(self, overwrite=False):
        """Make necessary symbolic links for BIAScombined.fits files."""
        self.link_calibrator('bias', overwrite=overwrite)
        return

    def reduce_dark(self, overwrite=False, **kwargs):
        """Reduce all dark frames."""
        self.reduce_calibrator(
            'dark', overwrite=overwrite, check='DRK', **kwargs)
        self.next_step('reduce_dark', print_message=True)
        return

    def combine_dark(self, overwrite=False):
        """Produce and link necessary DARKcombinedXXXX.fits files."""
        self.combine_calibrator('dark', overwrite=overwrite)
        self.next_step('combine_dark', print_message=True)
        return

    def link_dark(self, overwrite=False):
        """Make necessary symbolic links for DARKcombinedXXXX.fits files."""
        self.link_calibrator('dark', overwrite=overwrite)
        return

    def reduce_lflat(self, overwrite=False, **kwargs):
        """Reduce all lflat frames."""
        self.reduce_calibrator(
            'lflat', overwrite=overwrite, check='LFL', **kwargs)
        self.next_step('reduce_lflat', print_message=True)
        return

    def combine_lflat(self, overwrite=False):
        """Produce and link necessary LFLATcombined.fits files."""
        self.combine_calibrator('lflat', overwrite=overwrite)
        self.next_step('combine_lflat', print_message=True)
        return

    def link_lflat(self, overwrite=False):
        """Make necessary symbolic links for LFLATcombined.fits files."""
        self.link_calibrator('lflat', overwrite=overwrite)
        return

    def make_tlm(self, overwrite=False, leave_reduced=False, **kwargs):
        """Make TLMs from all files matching given criteria.
        If the use_twilight_tlm_blue keyword is set to True in the manager
        (when the manager is initialized), then we will also
        attempt to get a tramline map from twilight frames.  This is done
        by copying them to a different file that has class MFFFF using the
        copy_as function."""

        # check if ccd keyword argument is set, as we need to account for the
        # fact that it is also set for the twilight reductions (ccd1 only).
        do_twilight = True
        if ('ccd' in kwargs):
            if (kwargs['ccd'] == 'ccd_1'):
                do_twilight = True

            else:
                do_twilight = False
            # make a copy of the keywords, but remove the ccd flag for the
            # reduction of twilights, as it is set again in the call below.
            kwargs_copy = dict(kwargs)
            del kwargs_copy['ccd']
        else:
            kwargs_copy = dict(kwargs)

        if (self.use_twilight_tlm_blue and do_twilight):
            fits_twilight_list = []
            print('Processing twilight frames to get TLM')
            # for each twilight frame use the copy_as() function to
            # make a copy with file type MFFFF.  The copied files are
            # placed in the list fits_twilight_list and then can be
            # processed as normal MFFFF files.
            # for fits in self.files(ndf_class='MFSKY',do_not_use=False,**kwargs):
            for fits in self.files(ndf_class='MFSKY', do_not_use=False, ccd='ccd_1', **kwargs_copy):
                fits_twilight_list.append(self.copy_as(fits, 'MFFFF', overwrite=overwrite))

            # use the iterable file reducer to loop over the copied twilight list and
            # reduce them as MFFFF files to make TLMs.
            self.reduce_file_iterable(fits_twilight_list, overwrite=overwrite, tlm=True, leave_reduced=leave_reduced,
                                      check='TLM')

        # now we will process the normal MFFFF files
        # this currently only allows TLMs to be made from MFFFF files
        file_iterable = self.files(ndf_class='MFFFF', do_not_use=False,
                                   **kwargs)

        self.reduce_file_iterable(
            file_iterable, overwrite=overwrite, tlm=True,
            leave_reduced=leave_reduced, check='TLM')
        self.next_step('make_tlm', print_message=True)
        return

    def reduce_arc(self, overwrite=False, **kwargs):
        """Reduce all arc frames matching given criteria."""
        file_iterable = self.files(ndf_class='MFARC', do_not_use=False,
                                   **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite, check='ARC')
        for fits in reduced_files:
            bad_fibres(fits.reduced_path, save=True)
        self.next_step('reduce_arc', print_message=True)
        return

    def reduce_fflat(self, overwrite=False, twilight_only=False, **kwargs):
        """Reduce all fibre flat frames matching given criteria."""

        # check if ccd keyword argument is set, as we need to account for the
        # fact that it is also set for the twilight reductions (ccd1 only).
        do_twilight = True
        if ('ccd' in kwargs):
            if (kwargs['ccd'] == 'ccd_1'):
                do_twilight = True

            else:
                do_twilight = False
            # make a copy of the keywords, but remove the ccd flag for the
            # reduction of twilights, as it is set again in the call below.
            kwargs_copy = dict(kwargs)
            del kwargs_copy['ccd']
        else:
            kwargs_copy = dict(kwargs)

        if (self.use_twilight_flat_blue and do_twilight):
            fits_twilight_list = []
            print('Processing twilight frames to get fibre flat field')
            # The twilights should already have been copied as MFFFF
            # at the make_tlm stage, but we can do this again here without
            # any penalty (easier to sue the same code).  So for   
            # each twilight frame use the copy_as() function to
            # make a copy with file type MFFFF.  The copied files are
            # placed in the list fits_twilight_list and then can be
            # processed as normal MFFFF files.
            # for fits in self.files(ndf_class='MFSKY',do_not_use=False,**kwargs):
            for fits in self.files(ndf_class='MFSKY', do_not_use=False, ccd='ccd_1', **kwargs_copy):
                fits_twilight_list.append(self.copy_as(fits, 'MFFFF', overwrite=overwrite))

            # use the iterable file reducer to loop over the copied twilight list and
            # reduce them as MFFFF files:
            reduced_twilights = self.reduce_file_iterable(fits_twilight_list, overwrite=overwrite, check='FLT')

            # Identify bad fibres and replace with an average over all other twilights
            if len(reduced_twilights) >= 3:
                path_list = [os.path.join(fits.reduced_dir, fits.copy_reduced_filename) for fits in reduced_twilights]
                correct_bad_fibres(path_list)

        # now we will process the normal MFFFF files
        if (not twilight_only):
            file_iterable = self.files(ndf_class='MFFFF', do_not_use=False,
                                       **kwargs)
            self.reduce_file_iterable(
                file_iterable, overwrite=overwrite, check='FLT')

        self.next_step('reduce_fflat', print_message=True)
        return

    def reduce_sky(self, overwrite=False, fake_skies=True, **kwargs):
        """Reduce all offset sky frames matching given criteria."""
        groups = self.group_files_by(
            ('field_id', 'plate_id', 'date', 'ccd'),
            ndf_class='MFSKY', do_not_use=False, **kwargs)
        file_list = []
        for files in groups.values():
            file_list.extend(files)
        self.reduce_file_iterable(
            file_list, overwrite=overwrite, check='SKY')
        
        # Average the throughput values in each group
        for files in groups.values():
            path_list = [fits.reduced_path for fits in files]
            make_clipped_thput_files(
                path_list, overwrite=overwrite, edit_all=True, median=True)
                
        # Send all the sky frames to the improved wavecal routine then
        # apply correction to all the blue arcs
        if  self.improve_blue_wavecorr:
            file_list_tw = []
            for f in file_list:
                if f.ccd == 'ccd_1':
                    file_list_tw.append(f)
            input_list = zip(file_list_tw,[overwrite]*len(file_list_tw))
            self.map(wavecorr_frame,input_list)
            wavecorr_av(file_list_tw,self.root)
            
            kwargs_tmp = kwargs.copy()
            if 'ccd' in kwargs_tmp:
                del kwargs_tmp['ccd']

            arc_file_iterable = self.files(ndf_class='MFARC', ccd = 'ccd_1',
                                    do_not_use=False, **kwargs_tmp)
            
            arc_paths = [fits.reduced_path for fits in arc_file_iterable]
            for arc_path in arc_paths:
                apply_wavecorr(arc_path,self.root)           
            
        if fake_skies:
            no_sky_list = self.fields_without_skies(**kwargs)
            # Certain parameters will already have been set so don't need
            # to be passed (and passing will cause duplicate kwargs)
            for key in ('field_id', 'plate_id', 'date', 'ccd'):
                if key in kwargs:
                    del kwargs[key]
            fits_sky_list = []
            for (field_id, plate_id, date, ccd) in no_sky_list:
                # This field has no MFSKY files, so copy the dome flats
                for fits in self.files(
                        ndf_class='MFFFF', do_not_use=False, lamp='Dome',
                        field_id=field_id, plate_id=plate_id, date=date,
                        ccd=ccd, **kwargs):
                    fits_sky_list.append(
                        self.copy_as(fits, 'MFSKY', overwrite=overwrite))
            # Now reduce the fake sky files from all fields
            self.reduce_file_iterable(fits_sky_list, overwrite=overwrite)
        self.next_step('reduce_sky', print_message=True)
        return

    def fields_without_skies(self, **kwargs):
        """Return a list of fields that have a dome flat but not a sky."""
        keys = ('field_id', 'plate_id', 'date', 'ccd')
        field_id_list_dome = self.group_files_by(
            keys, ndf_class='MFFFF', do_not_use=False,
            lamp='Dome', **kwargs).keys()
        field_id_list_sky = self.group_files_by(
            keys, ndf_class='MFSKY', do_not_use=False,
            **kwargs).keys()
        no_sky = [field for field in field_id_list_dome
                  if field not in field_id_list_sky]
        return no_sky

    def copy_as(self, fits, ndf_class, overwrite=False):
        """Copy a fits file and change its class. Return a new FITSFile."""
        old_num = int(fits.filename[6:10])
        new_num = old_num + 1000 * (9 - (old_num // 1000))
        new_filename = (
                fits.filename[:6] + '{:04d}'.format(int(new_num)) + fits.filename[10:])
        new_path = os.path.join(fits.reduced_dir, new_filename)
        if os.path.exists(new_path) and overwrite:
            os.remove(new_path)
        if not os.path.exists(new_path):
            # Make the actual copy
            shutil.copy2(fits.raw_path, new_path)
            # Open up the file and change its NDF_CLASS
            hdulist = pf.open(new_path, 'update')
            hdulist['STRUCT.MORE.NDF_CLASS'].data['NAME'][0] = ndf_class
            hdulist[0].header['MNGRCOPY'] = (
                True, 'True if this is a copy created by a Manager')
            hdulist.flush()
            hdulist.close()

        new_fits = FITSFile(new_path)

        #        print('new_fits:',new_fits
        # Add paths to the new FITSFile instance.
        # Don't use Manager.set_reduced_path because the raw location is
        # unusual
        new_fits.raw_dir = fits.reduced_dir
        new_fits.raw_path = new_path
        new_fits.reduced_dir = fits.reduced_dir
        new_fits.reduced_link = new_path
        new_fits.reduced_path = os.path.join(fits.reduced_dir, new_fits.reduced_filename)
        # as this file has not been imported normally, we need to also set the check_data:
        new_fits.set_check_data()
        # if the new class is MFFFF, then add tlm_path to the FITSfile instance as this
        # is also usually done by set_reduced_path.
        if ndf_class == 'MFFFF':
            new_fits.tlm_path = os.path.join(new_fits.reduced_dir, new_fits.tlm_filename)
            # Do we also set the lamp? Probably not.

        return new_fits

    def copy_path(self, path):
        """Return the path for a copy of the specified file."""
        directory = os.path.dirname(path)
        old_filename = os.path.basename(path)
        old_num = int(old_filename[6:10])
        new_num = old_num + 1000 * (9 - (old_num // 1000))
        new_filename = (
                old_filename[:6] + '{:04d}'.format(int(new_num)) + old_filename[10:])
        new_path = os.path.join(directory, new_filename)
        return new_path

    def reduce_object(self, overwrite=False, recalculate_throughput=True,
                      sky_residual_limit=0.025, **kwargs):
        """Reduce all object frames matching given criteria."""
        # Reduce long exposures first, to make sure that any required
        # throughput measurements are available
        file_iterable_long = self.files(
            ndf_class='MFOBJECT', do_not_use=False,
            min_exposure=self.min_exposure_for_throughput, **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable_long, overwrite=overwrite, check='OBJ')
        # Check how good the sky subtraction was
        for fits in reduced_files:
            self.qc_sky(fits)
        if sky_residual_limit is not None:
            # Switch to sky line throughputs if the sky residuals are bad
            fits_list = self.files_with_bad_dome_throughput(
                reduced_files, sky_residual_limit=sky_residual_limit)
            # Only keep them if they actually have a sky line to use
            fits_list = [fits for fits in fits_list if fits.has_sky_lines()]
            self.reduce_file_iterable(
                fits_list, throughput_method='skylines',
                overwrite=True, check='OBJ')
            bad_fields = set([fits.field_id for fits in fits_list])
        else:
            bad_fields = []
        if recalculate_throughput:
            # Average over individual throughputs measured from sky lines
            extra_files = self.correct_bad_throughput(
                overwrite=overwrite, **kwargs)
            for fits in extra_files:
                if fits not in reduced_files:
                    reduced_files.append(fits)
        # Now reduce the short exposures, which might need the long
        # exposure reduced above
        if 'max_exposure' in kwargs:
            upper_limit = (1.*kwargs['max_exposure'] - 
                           np.finfo(1.*kwargs['max_exposure']).epsneg)
            del kwargs['max_exposure']
        else:
            upper_limit = (self.min_exposure_for_throughput -
                       np.finfo(self.min_exposure_for_throughput).epsneg)
        file_iterable_short = self.files(
            ndf_class='MFOBJECT', do_not_use=False,
            max_exposure=upper_limit, **kwargs)
        file_iterable_sky_lines = []
        file_iterable_default = []
        for fits in file_iterable_short:
            if fits.field_id in bad_fields:
                file_iterable_sky_lines.append(fits)
            else:
                file_iterable_default.append(fits)
        # Although 'skylines' is requested, these will be throughput calibrated
        # by matching to long exposures, because they will be recognised as
        # short
        reduced_files.extend(self.reduce_file_iterable(
            file_iterable_sky_lines, overwrite=overwrite,
            throughput_method='skylines', check='OBJ'))
        # These will be throughput calibrated using dome flats
        reduced_files.extend(self.reduce_file_iterable(
            file_iterable_default, overwrite=overwrite, check='OBJ'))
        # Check how good the sky subtraction was
        for fits in reduced_files:
            self.qc_sky(fits)
        self.next_step('reduce_object', print_message=True)
        return

    def reduce_file_iterable(self, file_iterable, throughput_method='default',
                             overwrite=False, tlm=False, leave_reduced=True,
                             check=None):
        """Reduce all files in the iterable."""
        # First establish the 2dfdr options for all files that need reducing
        # Would be more memory-efficient to construct a generator

        input_list = []  # type: List[Tuple[FITSFile, str, Sequence]]
        for fits in file_iterable:
            if (overwrite or
                    not os.path.exists(self.target_path(fits, tlm=tlm))):
                tdfdr_options = tuple(self.tdfdr_options(fits, throughput_method=throughput_method, tlm=tlm))
                input_list.append(
                    (fits, self.idx_files[fits.grating], tdfdr_options))
        reduced_files = [item[0] for item in input_list]

        # Send the items out for reducing. Keep track of which ones were done.
        while input_list:
            print(len(input_list), 'files remaining.')
            finished = np.array(self.map(
                run_2dfdr_single_wrapper, input_list))

            # Mark finished files as requiring checks
            if check:
                for fin, reduction_tuple in zip(finished, input_list):
                    fits = reduction_tuple[0]
                    if fin:
                        update_checks(check, [fits], False)

            input_list = [item for i, item in enumerate(input_list)
                          if not finished[i]]
        # Delete unwanted reduced files
        for fits in reduced_files:
            if (fits.ndf_class == 'MFFFF' and tlm and not leave_reduced and os.path.exists(fits.reduced_path)):
                os.remove(fits.reduced_path)
        # Return a list of fits objects that were reduced
        return reduced_files

    def target_path(self, fits, tlm=False):
        """Return the path of the file we want 2dfdr to produce."""
        if fits.ndf_class == 'MFFFF' and tlm:
            target = fits.tlm_path
        else:
            target = fits.reduced_path
        return target

    def files_with_bad_dome_throughput(self, fits_list,
                                       sky_residual_limit=0.025):
        """Return list of fields with bad residuals that used dome flats."""
        # Get a list of all throughput files used
        thput_file_list = np.unique([
            fits.reduce_options().get('THPUT_FILENAME', '')
            for fits in fits_list])
        # Keep only the dome flats
        thput_file_list = [
            filename for filename in thput_file_list if
            filename and not self.fits_file(filename) and
            not filename.startswith('thput')]
        file_list = []
        for fits in self.files(
                ndf_class='MFOBJECT', do_not_use=False, reduced=True):
            try:
                residual = pf.getval(fits.reduced_path, 'SKYMNCOF', 'QC')
            except KeyError:
                # The QC measurement hasn't been done
                continue
            file_list.append(
                (fits,
                 fits.reduce_options().get('THPUT_FILENAME', ''),
                 residual))
        bad_files = []
        for thput_file in thput_file_list:
            # Check all files, not just the ones that were just reduced
            matching_files = [
                (fits, sky_residual) for
                (fits, thput_file_match, sky_residual) in file_list
                if thput_file_match == thput_file]
            mean_sky_residual = np.mean([
                sky_residual
                for (fits, sky_residual) in matching_files
                if fits.exposure >= self.min_exposure_for_throughput])
            if mean_sky_residual >= sky_residual_limit:
                bad_files.extend([fits for (fits, _) in matching_files])
        return bad_files

    def correct_bad_throughput(self, overwrite=False, **kwargs):
        """Create thput files with bad values replaced by mean over field."""
        rereduce = []
        # used_flats = self.fields_without_skies(**kwargs)
        for group in self.group_files_by(
                ('date', 'field_id', 'ccd'), ndf_class='MFOBJECT',
                do_not_use=False,
                min_exposure=self.min_exposure_for_throughput, reduced=True,
                **kwargs).values():
            # done = False
            # for (field_id_done, plate_id_done, date_done,
            #         ccd_done) in used_flats:
            #     if (date == date_done and field_id == field_id_done and
            #             ccd == ccd_done):
            #         # This field has been taken care of using dome flats
            #         done = True
            #         break
            # if done:
            #     continue
            # Only keep files that used sky lines for throughput calibration
            group = [fits for fits in group
                     if fits.reduce_options()['TPMETH'] in
                     ('SKYFLUX(MED)', 'SKYFLUX(COR)')]
            if len(group) <= 1:
                # Can't do anything if there's only one file available, or none
                continue
            path_list = [fits.reduced_path for fits in group]
            edited_list = make_clipped_thput_files(
                path_list, overwrite=overwrite, edit_all=True)
            for fits, edited in zip(group, edited_list):
                if edited:
                    rereduce.append(fits)
        reduced_files = self.reduce_file_iterable(
            rereduce, throughput_method='external', overwrite=True)
        return reduced_files

    def derive_transfer_function(self,
                                 overwrite=False, model_name='ref_centre_alpha_dist_circ_hdratm',
                                 smooth='spline', **kwargs):
        """Derive flux calibration transfer functions and save them."""
        inputs_list = []
        for fits in self.files(ndf_class='MFOBJECT', do_not_use=False,
                               spectrophotometric=True, ccd='ccd_1', **kwargs):
            if not overwrite:
                hdulist = pf.open(fits.reduced_path)
                try:
                    hdu = hdulist['FLUX_CALIBRATION']
                except KeyError:
                    # Hasn't been done yet. Close the file and carry on.
                    hdulist.close()
                else:
                    # Has been done. Close the file and skip to the next one.
                    hdulist.close()
                    continue
            fits_2 = self.other_arm(fits)
            path_pair = (fits.reduced_path, fits_2.reduced_path)
            log.info(path_pair)
            if fits.epoch < 2013.0:
                # SAMI v1 had awful throughput at blue end of blue, need to
                # trim that data
                n_trim = 3
            else:
                n_trim = 0
            inputs_list.append({'path_pair': path_pair, 'n_trim': n_trim,
                                'model_name': model_name, 'smooth': smooth,
                                'speed':self.speed,'tellcorprim':self.telluric_correct_primary})

        self.map(derive_transfer_function_pair, inputs_list)
        self.next_step('derive_transfer_function', print_message=True)
        return

    def combine_transfer_function(self, overwrite=False, **kwargs):
        """Combine and save transfer functions from multiple files."""

        # First sort the spectrophotometric files into date/field/CCD/name
        # groups. Grouping by name is not strictly necessary and could be
        # removed, which would cause results from different stars to be
        # combined.
        #groups = self.group_files_by(('date', 'field_id', 'ccd', 'name'),
        #                             ndf_class='MFOBJECT', do_not_use=False,
        #                             spectrophotometric=True, **kwargs)
        # revise grouping of standards, so that we average over all the
        # standard star observations in the run.  Only group based on ccd.
        # (SMC 17/10/2019)
        groups = self.group_files_by(('ccd'),
                                     ndf_class='MFOBJECT', do_not_use=False,
                                     spectrophotometric=True, **kwargs)



        

        # Now combine the files within each group
        for fits_list in groups.values():
            path_list = [fits.reduced_path for fits in fits_list]
            path_out = os.path.join(os.path.dirname(path_list[0]),
                                    'TRANSFERcombined.fits')
            if overwrite or not os.path.exists(path_out):
                print('Combining files to create', path_out)
                fluxcal2.combine_transfer_functions(path_list, path_out)
                # Run the QC throughput measurement
                self.qc_throughput_spectrum(path_out)

                # Since we've now changed the transfer functions, mark these as needing checks.
                update_checks('FLX', fits_list, False)

            # Copy the file into all required directories
            paths_with_copies = [os.path.dirname(path_list[0])]
            for path in path_list:
                if os.path.dirname(path) not in paths_with_copies:
                    path_copy = os.path.join(os.path.dirname(path),
                                             'TRANSFERcombined.fits')
                    if overwrite or not os.path.exists(path_copy):
                        print('Copying combined file to', path_copy)
                        shutil.copy2(path_out, path_copy)

                    paths_with_copies.append(os.path.dirname(path_copy))
        self.next_step('combine_transfer_function', print_message=True)
        return

    def flux_calibrate(self, overwrite=False, **kwargs):
        """Apply flux calibration to object frames."""
        for fits in self.files(ndf_class='MFOBJECT', do_not_use=False,
                               spectrophotometric=False, **kwargs):
            fits_spectrophotometric = self.matchmaker(fits, 'fcal')
            if fits_spectrophotometric is None:
                # Try again with less strict criteria
                fits_spectrophotometric = self.matchmaker(fits, 'fcal_loose')
                if fits_spectrophotometric is None:
                    raise MatchException('No matching flux calibrator found ' +
                                         'for ' + fits.filename)
            if overwrite or not os.path.exists(fits.fluxcal_path):
                print('Flux calibrating file:', fits.reduced_path)
                if os.path.exists(fits.fluxcal_path):
                    os.remove(fits.fluxcal_path)
                path_transfer_fn = os.path.join(
                    fits_spectrophotometric.reduced_dir,
                    'TRANSFERcombined.fits')
                fluxcal2.primary_flux_calibrate(
                    fits.reduced_path,
                    fits.fluxcal_path,
                    path_transfer_fn)
        self.next_step('flux_calibrate', print_message=True)
        return

    def telluric_correct(self, overwrite=False, model_name=None, name='main',
                         **kwargs):
        """Apply telluric correction to object frames."""
        # First make the list of file pairs to correct
        inputs_list = []
        for fits_2 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd='ccd_2',
                                 name=name, **kwargs):
            if os.path.exists(fits_2.telluric_path) and not overwrite:
                # Already been done; skip to the next file
                continue
            fits_1 = self.other_arm(fits_2)
            if fits_2.epoch < 2013.0:
                # SAMI v1 had awful throughput at blue end of blue, need to
                # trim that data.
                n_trim = 3
                # Also get telluric shape from primary standard
                use_PS = True
                fits_spectrophotometric = self.matchmaker(fits_2, 'fcal')
                if fits_spectrophotometric is None:
                    # Try again with less strict criteria
                    fits_spectrophotometric = self.matchmaker(
                        fits_2, 'fcal_loose')
                    if fits_spectrophotometric is None:
                        raise MatchException('No matching flux calibrator ' +
                                             'found for ' + fits_2.filename)
                PS_spec_file = os.path.join(
                    fits_spectrophotometric.reduced_dir,
                    'TRANSFERcombined.fits')
                # For September 2012, secondary stars were often not in the
                # hexabundle at all, so use the theoretical airmass scaling
                if fits_2.epoch < 2012.75:
                    scale_PS_by_airmass = True
                else:
                    scale_PS_by_airmass = False
                # Also constrain the zenith distance in fitting the star
                if model_name is None:
                    model_name_out = 'ref_centre_alpha_circ_hdratm'
                else:
                    model_name_out = model_name
            else:
                # These days everything is hunkydory
                n_trim = 0
                use_PS = False
                PS_spec_file = None
                scale_PS_by_airmass = False
                if model_name is None:
#                    model_name_out = 'ref_centre_alpha_dist_circ_hdratm'
                    # in some case the fit of the model does not do a good job of
                    # getting the zenith distance.  A more reliable fit is
                    # obtained when we instead use the ZD based on the atmosphere
                    # fully, not just the direction:
                    model_name_out = 'ref_centre_alpha_circ_hdratm'
                else:
                    model_name_out = model_name
            inputs_list.append({
                'fits_1': fits_1,
                'fits_2': fits_2,
                'n_trim': n_trim,
                'use_PS': use_PS,
                'scale_PS_by_airmass': scale_PS_by_airmass,
                'PS_spec_file': PS_spec_file,
                'model_name': model_name_out,
                'speed':self.speed})
        # Now send this list to as many cores as we are using
        # Limit this to 10, because of semaphore issues I don't understand
        old_n_cpu = self.n_cpu
        if old_n_cpu > 10:
            self.n_cpu = 10

        done_list = self.map(telluric_correct_pair, inputs_list)

        # Mark files as needing visual checks:
        for item in inputs_list:
            update_checks('TEL', [item["fits_2"]], False)

        self.n_cpu = old_n_cpu
        for inputs in [inputs for inputs, done in
                       zip(inputs_list, done_list) if done]:
            # Copy the FWHM measurement to the QC header
            self.qc_seeing(inputs['fits_1'])
            self.qc_seeing(inputs['fits_2'])
        self.next_step('telluric_correct', print_message=True)
        return

    def _get_missing_stars(self, catalogue=None):
        """Return lists of observed stars missing from the catalogue."""
        name_list = []
        coords_list = []
        for fits_list in self.group_files_by(
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

    def get_stellar_photometry(self, refresh=False, automatic=True):
        """Get photometry of stars, with help from the user."""
        if refresh:
            catalogue = None
        else:
            catalogue = read_stellar_mags()

        name_list, coords_list = self._get_missing_stars(catalogue=catalogue)
        new = get_sdss_stellar_mags(name_list, coords_list, catalogue=catalogue, automatic=automatic)
        # Note: with automatic=True, get_sdss_stellar_mags will try to download
        # the data and return it as a string
        if isinstance(new, bool) and not new:
            # No new magnitudes were downloaded
            return
        idx = 1
        path_out = 'standards/secondary/sdss_stellar_mags_{}.csv'.format(idx)
        while os.path.exists(path_out):
            idx += 1
            path_out = (
                'standards/secondary/sdss_stellar_mags_{}.csv'.format(idx))
        if isinstance(new, bool) and new:
            # get_sdss_stellar_mags could not do an automatic retrieval.
            path_in = input('Enter the path to the downloaded file:\n')
            shutil.move(path_in, path_out)
        else:
            with open(path_out, 'w') as f:
                f.write(new)
        return

    def fluxcal_secondary(self, overwrite=False, use_av_tf_sec = True,force=False, **kwargs):
        """derive a flux calibration for individual frames based on the secondary std stars.
        This is done with fits to stellar models using ppxf to get the correct stellar model.
        If force=True, we will apply the correction even if the SECCOR keyword is set to 
        True."""

        # Generate a list of frames to do fit the stellar models to.
        # these are only for ccd_1 (not enough features in the red arm to
        # make it useful).  The fits are done one frame at a time for all
        # object frames and the best templates are written to the header
        # of the frame.  Also write a header keyword to signify that the
        # secondary correction has been done - keyword is SECCOR.
        inputs_list = []
        for fits_1 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd='ccd_1',
                                 name='main',**kwargs):
            if (not overwrite and 'SECCOR' in
                    pf.getheader(fits_1.telluric_path)):
                # Already been done; skip to the next file
                continue
            inputs_list.append((fits_1.telluric_path))
            
        # fit each of the frames indivdually using ppxf, store the results
        # (the best template(s) and possibly weights) in the headers.
        # call to actually do the fitting:
        self.map(fit_sec_template, inputs_list)
 
        # group the data by field and/or std star (probably field).  Average
        # The best templates or weights to determine the best model for the
        # star in each field.
        groups = self.group_files_by(('date', 'field_id', 'ccd'),
                                     ndf_class='MFOBJECT', do_not_use=False,
                                     ccd='ccd_1',
                                     spectrophotometric=False, **kwargs)

        for fits_list in groups.values():
            #fits_1 = self.other_arm(fits_2)
            #inputs_list.append((fits_1.telluric_path, fits_2.telluric_path))
            # get the path list for all the ccd_1 frames in this group:
            path_list = [fits.telluric_path for fits in fits_list]
            # also get the equivalent list for the ccd_2 frames:
            path_list2 = [self.other_arm(fits).telluric_path for fits in fits_list]
            path_out = os.path.join(os.path.dirname(path_list[0]),
                                    'TRANSFER2combined.fits')
            path_out2 = os.path.join(os.path.dirname(path_list2[0]),
                                    'TRANSFER2combined.fits')
            if overwrite or not os.path.exists(path_out):
                print('combined template weight into', path_out)
                # now actually call the routine to combine the weights:
                fluxcal2.combine_template_weights(path_list, path_out)
   
            # for each frame (red and blue) use the best template (gal extinction corrected)
            # to derive a transfer function.  Write the transfer function to the data frame
            # as a separate extension - FLUX_CALIBRATION2.  Also grouped by field, average
            # the indivdual secondary calibrations to derive one per field.  This may be
            # optional depending on how good invididual fits are.  Write the combined secondary
            # TF to a separate file for each field.
            fluxcal2.derive_secondary_tf(path_list,path_list2,path_out)

            # by group now correct the spectra by applying the TF.  This can be done on a
            # frame by frame basis, or by field.
            for index, path1 in enumerate(path_list):
                path2 = path_list2[index]
                fluxcal2.apply_secondary_tf(path1,path2,path_out,path_out2,use_av_tf_sec=use_av_tf_sec,force=force)
        
        # possibly set some QC stuff here...?

        
        self.next_step('fluxcal_secondary', print_message=True)
        
        return
        
    
    def scale_frames(self, overwrite=False, **kwargs):
        """Scale individual RSS frames to the secondary standard flux."""
        # First make the list of file pairs to scale
        inputs_list = []
        for fits_2 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd='ccd_2',
                                 telluric_corrected=True, name='main',
                                 **kwargs):
            if (not overwrite and 'RESCALE' in
                    pf.getheader(fits_2.telluric_path, 'FLUX_CALIBRATION')):
                # Already been done; skip to the next file
                continue
            fits_1 = self.other_arm(fits_2)
            inputs_list.append((fits_1.telluric_path, fits_2.telluric_path))
        self.map(scale_frame_pair, inputs_list)
        # Measure the relative atmospheric transmission
        for (path_1, path_2) in inputs_list:
            self.qc_throughput_frame(path_1)
            self.qc_throughput_frame(path_2)
        self.next_step('scale_frames', print_message=True)
        return

    def measure_offsets(self, overwrite=False, min_exposure=599.0, name='main',
                        ccd='both', **kwargs):
        """Measure the offsets between dithered observations."""
        if ccd == 'both':
            ccd_measure = 'ccd_2'
            copy_to_other_arm = True
        else:
            ccd_measure = ccd
            copy_to_other_arm = False
        groups = self.group_files_by(
            'field_id', ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, min_exposure=min_exposure, name=name, ccd=ccd_measure,
            include_linked_managers=True, **kwargs)
        complete_groups = []
        for key, fits_list in groups.items():
            fits_list_other_arm = [self.other_arm(fits, include_linked_managers=True)
                                   for fits in fits_list]
            if overwrite:
                complete_groups.append(
                    (key, fits_list, copy_to_other_arm, fits_list_other_arm))
                continue
            for fits in fits_list:
                # This loop checks each fits file and adds the group to the
                # complete list if *any* of them are missing the ALIGNMENT HDU
                try:
                    pf.getheader(best_path(fits), 'ALIGNMENT')
                except KeyError:
                    # No previous measurement, so we need to do this group
                    complete_groups.append(
                        (key, fits_list, copy_to_other_arm,
                         fits_list_other_arm))

                    # Also mark this group as requiring visual checks:
                    update_checks('ALI', fits_list, False)
                    break

        self.map(measure_offsets_group, complete_groups)

        self.next_step('measure_offsets', print_message=True)
        return

    def cube(self, overwrite=False, min_exposure=599.0, name='main',
             star_only=False, drop_factor=None, tag='', update_tol=0.02,
             size_of_grid=50, output_pix_size_arcsec=0.5,
             min_transmission=0.333, max_seeing=4.0, min_frames=6, **kwargs):
        """Make datacubes from the given RSS files."""
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers=True, **kwargs)
        # Add in the root path as well, so that cubing puts things in the 
        # right place
        cubed_root = os.path.join(self.root, 'cubed')
        inputs_list = []

        failed_qc_file = os.path.join(self.root, 'failed_qc_fields.txt')
        with open(failed_qc_file, "w+") as infile:
            failed_fields = [line.rstrip() for line in infile]

            for (field_id, ccd), fits_list in groups.items():
                good_fits_list = self.qc_for_cubing(
                    fits_list, min_transmission=min_transmission,
                    max_seeing=max_seeing, min_exposure=min_exposure)
                path_list = [best_path(fits) for fits in good_fits_list]
                if len(path_list) < min_frames:
                    # Not enough good frames to bother making the cubes
                    objects = ''
                    if field_id not in failed_fields:
                        failed_fields.append(field_id)
                elif star_only:
                    objects = [pf.getval(path_list[0], 'STDNAME', 'FLUX_CALIBRATION')]
                else:
                    objects = get_object_names(path_list[0])
                    if field_id in failed_fields:
                        failed_fields.remove(field_id)
                if drop_factor is None:
                    if fits_list[0].epoch < 2013.0:
                        # Large pitch of pilot data requires a larger drop size
                        drop_factor = 0.75
                    else:
                        drop_factor = 0.5

                for name in objects:
                    inputs_list.append(
                        (field_id, ccd, path_list, name, cubed_root, drop_factor,
                         tag, update_tol, size_of_grid, output_pix_size_arcsec,
                         overwrite))

        with open(failed_qc_file, "w") as outfile:
            failed_fields = [field + '\n' for field in failed_fields]
            outfile.writelines(failed_fields)

        # Send the cubing tasks off to multiple CPUs
        cubed_list = self.map(cube_object, inputs_list)

        # Mark cubes as not checked. Only mark the first file in each input set
        for inputs, cubed in zip(inputs_list, cubed_list):
            if cubed:
                # Select the first fits file from this run (not linked runs)
                path_list = inputs[2]  # From inputs_list above.
                for path in path_list:
                    fits = self.fits_file(os.path.basename(path)[:10])
                    if fits:
                        break
                if fits:
                    update_checks('CUB', [fits], False)
        self.next_step('cube', print_message=True)
        return

    def qc_for_cubing(self, fits_list, min_transmission=0.333, max_seeing=4.0,
                      min_exposure=599.0):
        """Return a list of fits files from the inputs that pass basic QC."""
        good_fits_list = []
        for fits in fits_list:
            # Check that the file pair meets basic QC requirements.
            # We check both files and use the worst case, so that
            # either both are used or neither.
            transmission = np.inf
            seeing = 0.0
            fits_pair = (
                fits, self.other_arm(fits, include_linked_managers=True))
            for fits_test in fits_pair:
                try:
                    transmission = np.minimum(
                        transmission,
                        pf.getval(best_path(fits_test), 'TRANSMIS', 'QC'))
                except KeyError:
                    # Either QC HDU doesn't exist or TRANSMIS isn't there
                    pass
                try:
                    seeing = np.maximum(
                        seeing,
                        pf.getval(best_path(fits_test), 'FWHM', 'QC'))
                except KeyError:
                    # Either QC HDU doesn't exist or FWHM isn't there
                    pass
            if (transmission >= min_transmission
                    and seeing <= max_seeing
                    and fits.exposure >= min_exposure):
                good_fits_list.append(fits)
        return good_fits_list

    def scale_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                    min_transmission=0.333, max_seeing=4.0, tag=None,
                    **kwargs):
        """Scale datacubes based on the stellar g magnitudes."""
        groups = self.group_files_by(
            'field_id', ccd='ccd_1', ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers=True, **kwargs)
        input_list = []
        for (field_id,), fits_list in groups.items():
            table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
            objects = table['NAME'][table['TYPE'] == 'P']
            objects = np.unique(objects).tolist()
            objects = [obj.strip() for obj in objects]  # Stripping whitespace from object names
            for name in objects:
                if telluric.is_star(name):
                    break
            else:
                print('No star found in field, skipping: ' + field_id)
                continue
            star = name
            objects.remove(star)
            star_path_pair = [
                self.cubed_path(star, arm, fits_list, field_id,
                                exists=True, min_exposure=min_exposure,
                                min_transmission=min_transmission,
                                max_seeing=max_seeing, tag=tag)
                for arm in ('blue', 'red')]
            if star_path_pair[0] is None or star_path_pair[1] is None:
                continue
            if not overwrite:
                # Need to check if the scaling has already been done
                try:
                    [pf.getval(path, 'RESCALE') for path in star_path_pair]
                except KeyError:
                    pass
                else:
                    continue
            object_path_pair_list = [
                [self.cubed_path(name, arm, fits_list, field_id,
                                 exists=True, min_exposure=min_exposure,
                                 min_transmission=min_transmission,
                                 max_seeing=max_seeing, tag=tag)
                 for arm in ('blue', 'red')]
                for name in objects]
            object_path_pair_list = [
                pair for pair in object_path_pair_list if None not in pair]
            input_list.append((star_path_pair, object_path_pair_list, star))
        self.map(scale_cubes_field, input_list)
        self.next_step('scale_cubes', print_message=True)
        return

    def bin_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                  min_transmission=0.333, max_seeing=4.0, tag=None, **kwargs):
        """Apply default binning schemes to datacubes."""
        path_pair_list = []
        groups = self.group_files_by(
            'field_id', ccd='ccd_1', ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers=True, **kwargs)
        for (field_id,), fits_list in groups.items():
            table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
            objects = table['NAME'][table['TYPE'] == 'P']
            objects = np.unique(objects).tolist()
            objects = [obj.strip() for obj in objects]  # Strip whitespace from object names
            for name in objects:
                path_pair = [
                    self.cubed_path(name, arm, fits_list, field_id,
                                    exists=True, min_exposure=min_exposure,
                                    min_transmission=min_transmission,
                                    max_seeing=max_seeing, tag=tag)
                    for arm in ('blue', 'red')]
                if path_pair[0] and path_pair[1]:
                    skip = False
                    if not overwrite:
                        hdulist = pf.open(path_pair[0])
                        for hdu in hdulist:
                            if hdu.name.startswith('BINNED_FLUX'):
                                skip = True
                                break
                    if not skip:
                        path_pair_list.append(path_pair)
        self.map(bin_cubes_pair, path_pair_list)
        self.next_step('bin_cubes', print_message=True)
        return

    def bin_aperture_spectra(self, overwrite=False, min_exposure=599.0, name='main',
                             min_transmission=0.333, max_seeing=4.0, tag=None, **kwargs):
        """Create aperture spectra."""
        print('Producing aperture spectra')
        path_pair_list = []
        groups = self.group_files_by(
            'field_id', ccd='ccd_1', ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers=True, **kwargs)
        for (field_id,), fits_list in groups.items():
            table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
            objects = table['NAME'][table['TYPE'] == 'P']
            objects = np.unique(objects)
            for name in objects:
                path_pair = [
                    self.cubed_path(name.strip(), arm, fits_list, field_id,
                                    exists=True, min_exposure=min_exposure,
                                    min_transmission=min_transmission,
                                    max_seeing=max_seeing, tag=tag)
                    for arm in ('blue', 'red')]
                if path_pair[0] and path_pair[1]:
                    path_pair_list.append(path_pair)

        inputs_list = []
        for path_pair in path_pair_list:
            inputs_list.append(overwrite)
        self.map(aperture_spectra_pair, path_pair_list)
        self.next_step('bin_aperture_spectra', print_message=True)

        return

    def record_dust(self, overwrite=False, min_exposure=599.0, name='main',
                    min_transmission=0.333, max_seeing=4.0, tag=None, **kwargs):
        """Record information about dust in the output datacubes."""
        groups = self.group_files_by(
            'field_id', ccd='ccd_1', ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers = True, **kwargs)
        for (field_id,), fits_list in groups.items():
            table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
            objects = table['NAME'][table['TYPE'] == 'P']
            objects = np.unique(objects).tolist()
            objects = [obj.strip() for obj in objects]
            for name in objects:
                for arm in ('blue', 'red'):
                    path = self.cubed_path(
                        name, arm, fits_list, field_id,
                        exists=True, min_exposure=min_exposure,
                        min_transmission=min_transmission,
                        max_seeing=max_seeing, tag=tag)
                    if path:
                        dust.dustCorrectSAMICube(path, overwrite=overwrite)

        self.next_step('record_dust',print_message=True)
        return

    def gzip_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                   star_only=False, min_transmission=0.333, max_seeing=4.0,
                   tag=None, **kwargs):
        """Gzip the final datacubes."""
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, include_linked_managers = True, **kwargs)
        input_list = []
        for (field_id, ccd), fits_list in groups.items():
            if ccd == 'ccd_1':
                arm = 'blue'
            else:
                arm = 'red'
            if star_only:
                objects = [pf.getval(fits_list[0].fcal_path, 'STDNAME',
                                     'FLUX_CALIBRATION')]
            else:
                table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
                objects = table['NAME'][table['TYPE'] == 'P']
                objects = np.unique(objects).tolist()
                objects = [obj.strip() for obj in objects]
            for obj in objects:
                input_path = self.cubed_path(
                    obj, arm, fits_list, field_id,
                    exists=True, min_exposure=min_exposure,
                    min_transmission=min_transmission,
                    max_seeing=max_seeing, tag=tag)
                if input_path:
                    if input_path.endswith('.gz'):
                        # Already gzipped, and no non-gzipped version exists
                        continue
                    output_path = input_path + '.gz'
                    if os.path.exists(output_path) and overwrite:
                        os.remove(output_path)
                    if not os.path.exists(output_path):
                        input_list.append(input_path)
        self.map(gzip_wrapper, input_list)
        self.next_step('gzip_cubes', print_message=True)
        return

    def reduce_all(self, start=None, finish=None, overwrite=False, **kwargs):
        """Reduce everything, in order. Don't use unless you're sure."""

        task_list = self.task_list

        # Check for valid inputs:
        if start is None:
            start = task_list[0][0]
        if finish is None:
            finish = task_list[-1][0]

        task_name_list = list(map(lambda x: x[0], task_list))
        if start not in task_name_list:
            raise ValueError("Invalid start step! Must be one of: {}".format(", ".join(task_name_list)))
        if finish not in task_name_list:
            raise ValueError("Invalid finish step! Must be one of: {}".format(", ".join(task_name_list)))

        started = False
        for task, include_kwargs in task_list:
            if not started and task != start:
                # Haven't yet reached the first task to do
                continue
            started = True
            method = getattr(self, task)
            print("Starting reduction step '{}'".format(task))
            if include_kwargs:
                method(overwrite, **kwargs)
            else:
                method(overwrite)
            if task == finish:
                # Do not do any further tasks
                break
        # self.reduce_bias(overwrite, **kwargs)
        # self.combine_bias(overwrite)
        # self.reduce_dark(overwrite, **kwargs)
        # self.combine_dark(overwrite)
        # self.reduce_lflat(overwrite, **kwargs)
        # self.combine_lflat(overwrite)
        # self.make_tlm(overwrite, **kwargs)
        # self.reduce_arc(overwrite, **kwargs)
        # self.reduce_fflat(overwrite, **kwargs)
        # self.reduce_sky(overwrite, **kwargs)
        # self.reduce_object(overwrite, **kwargs)
        # self.derive_transfer_function(overwrite, **kwargs)
        # self.combine_transfer_function(overwrite, **kwargs)
        # self.flux_calibrate(overwrite, **kwargs)
        # self.telluric_correct(overwrite, **kwargs)
        # self.scale_frames(overwrite, **kwargs)
        # self.measure_offsets(overwrite, **kwargs)
        # self.cube(overwrite, **kwargs)
        # self.scale_cubes(overwrite, **kwargs)
        # self.bin_cubes(overwrite, **kwargs)
        return

    def ensure_qc_hdu(self, path, name='QC'):
        """Ensure that the file has a QC HDU."""
        hdulist = pf.open(path, 'update')
        try:
            hdulist[name]
        except KeyError:
            # QC HDU doesn't exist, so make one
            hdu = pf.ImageHDU(name=name)
            hdulist.append(hdu)
            hdulist.flush()
        hdulist.close()
        return

    def qc_seeing(self, fits):
        """Copy the FWHM over the QC header."""
        self.ensure_qc_hdu(fits.telluric_path)
        hdulist = pf.open(fits.telluric_path, 'update')
        source_header = hdulist['FLUX_CALIBRATION'].header
        header = hdulist['QC'].header
        header['FWHM'] = source_header['FWHM'], source_header.comments['FWHM']
        hdulist.flush()
        hdulist.close()

    def qc_sky(self, fits):
        """Run QC check on sky subtraction accuracy and save results."""
        results = sky_residuals(fits.reduced_path)
        for key, value in results.items():
            if not np.isfinite(value):
                results[key] = -9999
        self.ensure_qc_hdu(fits.reduced_path)
        hdulist = pf.open(fits.reduced_path, 'update')
        header = hdulist['QC'].header
        header['SKYMDCOF'] = (
            results['med_frac_skyres_cont'],
            'Median continuum fractional sky residual')
        header['SKYMDLIF'] = (
            results['med_frac_skyres_line'],
            'Median line fractional sky residual')
        header['SKYMDCOA'] = (
            results['med_skyflux_cont'],
            'Median continuum absolute sky residual')
        header['SKYMDLIA'] = (
            results['med_skyflux_line'],
            'Median line absolute sky residual')
        header['SKYMNCOF'] = (
            results['mean_frac_skyres_cont'],
            'Mean continuum fractional sky residual')
        header['SKYMNLIF'] = (
            results['mean_frac_skyres_line'],
            'Mean line fractional sky residual')
        header['SKYMNCOA'] = (
            results['mean_skyflux_cont'],
            'Mean continuum absolute sky residual')
        header['SKYMNLIA'] = (
            results['mean_skyflux_line'],
            'Mean line absolute sky residual')
        hdulist.flush()
        hdulist.close()
        return

    def qc_throughput_spectrum(self, path):
        """Save the throughput function for a TRANSFERcombined file."""
        absolute_throughput = throughput(path)
        # Check the CCD and date for this file
        file_input = pf.getval(path, 'ORIGFILE', 1)
        path_input = os.path.join(
            self.fits_file(file_input[:10]).reduced_dir, file_input)
        detector = pf.getval(path_input, 'DETECTOR')
        epoch = pf.getval(path_input, 'EPOCH')
        # Load mean throughput function for that CCD
        path_list = (glob('standards/throughput/mean_throughput_' +
                          detector + '.fits') +
                     glob('standards/throughput/mean_throughput_' +
                          detector + '_*.fits'))
        for path_mean in path_list:
            hdulist_mean = pf.open(path_mean)
            header = hdulist_mean[0].header
            if (('DATESTRT' not in header or
                 epoch >= header['DATESTRT']) and
                    ('DATEFNSH' not in header or
                     epoch <= header['DATEFNSH'])):
                # This file is suitable for use
                found_mean = True
                mean_throughput = hdulist_mean[0].data
                hdulist_mean.close()
                break
            hdulist_mean.close()
        else:
            print('Warning: No mean throughput file found for QC checks.')
            found_mean = False
        if found_mean:
            relative_throughput = absolute_throughput / mean_throughput
            data = np.vstack((absolute_throughput, relative_throughput))
            median_relative_throughput = np.nanmedian(relative_throughput)
            if not np.isfinite(median_relative_throughput):
                median_relative_throughput = -1.0
        else:
            data = absolute_throughput
        hdulist = pf.open(path, 'update')
        hdulist.append(pf.ImageHDU(data, name='THROUGHPUT'))
        if found_mean:
            hdulist['THROUGHPUT'].header['MEDRELTH'] = (
                median_relative_throughput, 'Median relative throughput')
            hdulist['THROUGHPUT'].header['PATHMEAN'] = (
                path_mean, 'File used to define mean throughput')
        hdulist.flush()
        hdulist.close()
        return

    def qc_throughput_frame(self, path):
        """Calculate and save the relative throughput for an object frame."""
        try:
            median_relative_throughput = (
                pf.getval(pf.getval(path, 'FCALFILE'),
                          'MEDRELTH', 'THROUGHPUT'))
        except KeyError:
            # Not all the data is available
            print("Warning: 'combine_transfer_function' required to calculate transmission.")
            return
        try:
            median_relative_throughput /= (
                pf.getval(path, 'RESCALE', 'FLUX_CALIBRATION'))
        except KeyError:
            # Not all the data is available
            print('Warning: Flux calibration required to calculate transmission.')
            return

        if not np.isfinite(median_relative_throughput):
            median_relative_throughput = -1.0
        self.ensure_qc_hdu(path)
        hdulist = pf.open(path, 'update')
        header = hdulist['QC'].header
        header['TRANSMIS'] = (
            median_relative_throughput, 'Relative transmission')
        hdulist.flush()
        hdulist.close()
        return

    def qc_summary(self, min_exposure=599.0, ccd='ccd_1', **kwargs):
        """Print a summary of the QC information available."""
        text = 'QC summary table\n'
        text += '=' * 75 + '\n'
        text += 'Use space bar and cursor keys to move up and down; q to quit.\n'
        text += 'If one file in a pair is disabled it is marked with a +\n'
        text += 'If both are disabled it is marked with a *\n'
        text += '\n'

        #
        #  Summarize shared calibrations
        #
        text += "Summary of shared calibrations\n"
        text += "-" * 75 + "\n"

        # Get grouped lists and restructure to be a dict of dicts.
        by_ndf_class = defaultdict(dict)
        for k, v in self.group_files_by(["ndf_class", "date", "ccd"]).items():
            if k[2] == ccd:
                by_ndf_class[k[0]].update({k[1]: v})

        # Print info about basic cals
        for cal_type in ("BIAS", "DARK", "LFLAT"):
            if cal_type in by_ndf_class:
                text += "{} Frames:\n".format(cal_type)
                total_cals = 0
                for date in sorted(by_ndf_class[cal_type].keys()):
                    n_cals = len(by_ndf_class[cal_type][date])
                    text += "  {}: {} frames\n".format(date, n_cals)
                    total_cals += n_cals
                text += "  TOTAL {}s: {} frames\n".format(cal_type, total_cals)

        # Gather info about flux standards
        text += "Flux standards\n"
        flux_standards = defaultdict(dict)
        for k, v in self.group_files_by(["date", "name", "spectrophotometric", "ccd"]).items():
            if k[3] == ccd and k[2]:
                flux_standards[k[0]].update({k[1]: v})

        # Print info about flux standards
        total_all_stds = 0
        for date in flux_standards:
            text += "  {}:\n".format(date)
            total_cals = 0
            for std_name in sorted(flux_standards[date].keys()):
                n_cals = len(flux_standards[date][std_name])
                text += "    {}: {} frames\n".format(std_name, n_cals)
                total_cals += n_cals
            text += "    Total: {} frames\n".format(total_cals)
            total_all_stds += total_cals
        text += "  TOTAL Flux Standards: {} frames\n".format(total_all_stds)
        text += "\n"

        # Summarize field observations
        for (field_id,), fits_list in self.group_files_by(
                'field_id', ndf_class='MFOBJECT', min_exposure=min_exposure,
                ccd=ccd, **kwargs).items():
            text += '+' * 75 + '\n'
            text += field_id + '\n'
            text += '-' * 75 + '\n'
            text += 'File        Exposure  FWHM (")  Transmission  Sky residual\n'
            for fits in sorted(fits_list, key=lambda f: f.filename):
                fwhm = '       -'
                transmission = '           -'
                sky_residual = '           -'
                try:
                    header = pf.getheader(best_path(fits), 'QC')
                except (IOError, KeyError):
                    pass
                else:
                    if 'FWHM' in header:
                        fwhm = '{:8.2f}'.format(header['FWHM'])
                    if 'TRANSMIS' in header:
                        transmission = '{:12.3f}'.format(header['TRANSMIS'])
                    if 'SKYMDCOF' in header:
                        sky_residual = '{:12.3f}'.format(header['SKYMDCOF'])
                fits_2 = self.other_arm(fits)
                if fits.do_not_use and fits_2.do_not_use:
                    disabled_flag = '*'
                elif fits.do_not_use or fits_2.do_not_use:
                    disabled_flag = '+'
                else:
                    disabled_flag = ' '
                text += '{} {}{} {:8d}  {}  {}  {}\n'.format(
                    fits.filename[:5], fits.filename[6:10], disabled_flag,
                    int(fits.exposure), fwhm, transmission, sky_residual)
            text += '+' * 75 + '\n'
            text += '\n'
        pager(text)
        return

    def tdfdr_options(self, fits, throughput_method='default', tlm=False):
        """Set the 2dfdr reduction options for this file."""
        options = []

        # Define what the best choice is for a TLM:
        if (self.use_twilight_tlm_blue and (fits.ccd == 'ccd_1') and
            (fits.plate_id_short is not 'Y14SAR4_P007')):
            best_tlm = 'tlmap_mfsky'
        else:
            best_tlm = 'tlmap'

        # Define what the best choice is for a FFLAT, in particular
        # if we are going to use a twilight flat:
        if (self.use_twilight_flat_blue  and (fits.ccd == 'ccd_1')):
            best_fflat = 'fflat_mfsky'
        else:
            best_fflat = 'fflat'

        # only do skyscrunch for longer exposures (both CCDs):
        if fits.exposure >= self.min_exposure_for_sky_wave:
                # Adjust wavelength calibration of red frames using sky lines
            options.extend(['-SKYSCRUNCH', '1'])
        else:
            options.extend(['-SKYSCRUNCH', '0'])
                
        # add options for just CCD_2:
        if fits.ccd == 'ccd_2':
            # Turn off bias and dark subtraction
            if fits.detector == 'E2V3':
                options.extend(['-USEBIASIM', '0', '-USEDARKIM', '0'])
            elif fits.detector == 'E2V3A':
                options.extend(['-USEBIASIM', '0'])

        # turn off bias and dark for new CCD. These are named
        # E2V2A (blue) and E2V3A (red).  The old ones are E2V2 (blue
        # and E2V3 (red).
        if fits.detector == 'E2V2A':
            options.extend(['-USEBIASIM', '0', '-USEDARKIM', '0'])

        if fits.ndf_class == 'BIAS':
            files_to_match = []
        elif fits.ndf_class == 'DARK':
            files_to_match = ['bias']
        elif fits.ndf_class == 'LFLAT':
            files_to_match = ['bias', 'dark']
        elif fits.ndf_class == 'MFFFF' and tlm:
            files_to_match = ['bias', 'dark', 'lflat']
        elif fits.ndf_class == 'MFARC':
            files_to_match = ['bias', 'dark', 'lflat', best_tlm]
            # Arc frames can't use optimal extraction because 2dfdr screws up
            # and marks entire columns as bad when it gets too many saturated
            # pixels
            options.extend(['-EXTR_OPERATION', 'GAUSS'])
        elif fits.ndf_class == 'MFFFF' and not tlm:
            # new version of 2dfdr aaorun (2dfdr 7.0) also needs to pass the TLMAP_FILENAME argument
            # when reducing flat fields.  As a result we need to add this to the arguments.
            # We need to be careful (see discussion below) that it is the right filename,
            # i.e. just the tlm for the frame being reduced (SMC 14/06/2018).
            options.extend(['-TLMAP_FILENAME',fits.tlm_filename])
            # if not reduced for a TLM, then assume it has already been done, and set
            # flag to not repeat TLM.  The only reason to do this is save time in the
            # reductions.  Otherwise the result should be the same:
            options.extend(['-DO_TLMAP','0'])
            if fits.lamp == 'Flap':
                # Flap flats should use their own tramline maps, not those
                # generated by dome flats.  Do we want this to happen, even
                # if the best TLM could be from a twilight frame?  For now
                # leave it as this, but it may be that the twilight tlm (if
                # available) is better, at least in regard to the measurement
                # of the fibre profile widths.
                # files_to_match = ['bias', 'dark', 'lflat', 'tlmap_flap',
                #                  'wavel']
                # 2dfdr always remakes a TLM for an MFFFF, so don't set the
                # tlmap for these anyway:
                files_to_match = ['bias', 'dark', 'lflat', 'wavel']
            else:
                # if this is an MFFFF then always assure that we are using the
                # TLM that came from that file.  The main reason for this is that
                # if we pass a different TLM file, then a new TLM will be generated
                # anyway, but overwritten into the filename that is passed (e.g.
                # a twilight TLM could be overwritten by a dome flat TLM): 
                #files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel']
                files_to_match = ['bias', 'dark', 'lflat','wavel']

        elif fits.ndf_class == 'MFSKY':
            files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                              best_fflat]
        elif fits.ndf_class == 'MFOBJECT':
            if throughput_method == 'default':
                files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                  best_fflat, 'thput']
                options.extend(['-TPMETH', 'OFFSKY'])
            elif throughput_method == 'external':
                files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                  best_fflat]
                options.extend(['-TPMETH', 'OFFSKY'])
                options.extend(['-THPUT_FILENAME',
                                'thput_' + fits.reduced_filename])
            elif throughput_method == 'skylines':
                if (fits.exposure >= self.min_exposure_for_throughput and
                        fits.has_sky_lines()):
                    files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                      best_fflat]
                    options.extend(['-TPMETH', 'SKYFLUX(MED)'])
                else:
                    files_to_match = ['bias', 'dark', 'lflat', best_tlm, 'wavel',
                                      best_fflat, 'thput_object']
                    options.extend(['-TPMETH', 'OFFSKY'])
        else:
            raise ValueError('Unrecognised NDF_CLASS: ' + fits.ndf_class)
        # Remove unnecessary files from files_to_match
        if 'bias' in files_to_match and '-USEBIASIM' in options:
            if options[options.index('-USEBIASIM') + 1] == '0':
                files_to_match.remove('bias')
        if 'dark' in files_to_match and '-USEDARKIM' in options:
            if options[options.index('-USEDARKIM') + 1] == '0':
                files_to_match.remove('dark')
        if 'lflat' in files_to_match and '-USEFLATIM' in options:
            if options[options.index('-USEFLATIM') + 1] == '0':
                files_to_match.remove('lflat')
        # Disable bias/dark/lflat if they're not being used
        # If you don't, 2dfdr might barf
        if 'bias' not in files_to_match and '-USEBIASIM' not in options:
            options.extend(['-USEBIASIM', '0'])
        if 'dark' not in files_to_match and '-USEDARKIM' not in options:
            options.extend(['-USEDARKIM', '0'])
        if 'lflat' not in files_to_match and '-USEFLATIM' not in options:
            options.extend(['-USEFLATIM', '0'])
        for match_class in files_to_match:
            # this is the main call to the matching routine:
            filename_match = self.match_link(fits, match_class)
            if filename_match is None:
                # What to do if no match was found
                if match_class == 'bias':
                    print('Warning: Bias frame not found. '
                          'Turning off bias subtraction for ' + fits.filename)
                    options.extend(['-USEBIASIM', '0'])
                    continue
                elif match_class == 'dark':
                    print('Warning: Dark frame not found. '
                          'Turning off dark subtraction for ' + fits.filename)
                    options.extend(['-USEDARKIM', '0'])
                    continue
                elif match_class == 'lflat':
                    print('Warning: LFlat frame not found. '
                          'Turning off LFlat division for ' + fits.filename)
                    options.extend(['-USEFLATIM', '0'])
                    continue
                elif match_class == 'thput':
                    # Try to find a fake MFSKY made from a dome flat
                    filename_match = self.match_link(fits, 'thput_fflat')
                    if filename_match is None:
                        if (fits.exposure < self.min_exposure_for_throughput or
                                not fits.has_sky_lines()):
                            # Try to find a suitable object frame instead
                            filename_match = self.match_link(
                                fits, 'thput_object')
                            # Really run out of options here
                            if filename_match is None:
                                # Still nothing
                                print('Warning: Offsky (or substitute) frame '
                                      'not found. Turning off throughput '
                                      'calibration for ' + fits.filename)
                                options.extend(['-THRUPUT', '0'])
                                continue
                        else:
                            # This is a long exposure, so use the sky lines
                            options[options.index('-TPMETH') + 1] = (
                                'SKYFLUX(MED)')
                elif match_class == best_tlm:
                    # If we are using twilights, then go through the 3 different
                    # twilight options first.  If they are not found, then default
                    # back to the normal tlmap route.
                    found = 0
                    if (self.use_twilight_tlm_blue and (fits.ccd == 'ccd_1') and 
                        (fits.plate_id_short is not 'Y14SAR4_P007')):
                        filename_match = self.match_link(fits, 'tlmap_mfsky')
                        if filename_match is None:
                            filename_match = self.match_link(fits, 'tlmap_mfsky_loose')
                            if filename_match is None:
                                filename_match = self.match_link(fits, 'tlmap_mfsky_any')
                                if filename_match is None:
                                    print('Warning: no matching twilight frames found for TLM.'
                                          'Will default to using flat field frames instead'
                                          'for ' + fits.filename)
                                else:
                                    print('Warning: No matching twilight found for TLM.'
                                          'Using a twilight frame from a different night'
                                          'for ' + fits.filename)
                                    found = 1
                            else:
                                print('Warning: No matching twilight found for TLM.'
                                      'Using a twilight frame from the same night'
                                      'for ' + fits.filename)
                                found = 1
                        else:
                            print('Found matching twilight for TLM '
                                  'for ' + fits.filename)
                            found = 1

                    # if we haven't already found a matching TLM above (i.e. if found = 0), then
                    # go through the options with the flats:
                    if (found == 0):
                        # Try with normal TLM from flat:
                        filename_match = self.match_link(fits, 'tlmap')
                        if filename_match is None:
                            # Try with looser criteria
                            filename_match = self.match_link(fits, 'tlmap_loose')
                            if filename_match is None:
                                # Try using a flap flat instead
                                filename_match = self.match_link(fits, 'tlmap_flap')
                                if filename_match is None:
                                    # Try with looser criteria
                                    filename_match = self.match_link(
                                        fits, 'tlmap_flap_loose')
                                    if filename_match is None:
                                        # Still nothing. Raise an exception
                                        raise MatchException(
                                            'No matching tlmap found for ' +
                                            fits.filename)
                                    else:
                                        print('Warning: No good flat found for TLM. '
                                              'Using flap flat from different field '
                                              'for ' + fits.filename)
                                else:
                                    print('Warning: No dome flat found for TLM. '
                                          'Using flap flat instead for ' + fits.filename)
                            else:
                                print('Warning: No matching flat found for TLM. '
                                      'Using flat from different field for ' +
                                      fits.filename)
                        else:
                            print('Warning: No matching twilight found for TLM. '
                                  'Using a dome flat instead ' +
                                  fits.filename)

                elif match_class == best_fflat:
                    # If we are using twilights, then go through the 3 different
                    # twilight options first.  If they are not found, then default
                    # back to the normal fflat route (this is a copy of the version
                    # for the TLM above - with minor changes):
                    found = 0
                    if (self.use_twilight_flat_blue  and (fits.ccd == 'ccd_1')):
                        filename_match = self.match_link(fits, 'fflat_mfsky')
                        if filename_match is None:
                            filename_match = self.match_link(fits, 'fflat_mfsky_loose')
                            if filename_match is None:
                                filename_match = self.match_link(fits, 'fflat_mfsky_any')
                                if filename_match is None:
                                    print('Warning: no matching twilight frames found for FFLAT.'
                                          'Will default to using flat field frames instead'
                                          'for ' + fits.filename)
                                else:
                                    print('Warning: No matching twilight found for FFLAT.'
                                          'Using a twilight frame from a different night'
                                          'for ' + fits.filename)
                                    found = 1
                            else:
                                print('Warning: No matching twilight found for FFLAT.'
                                      'Using a twilight frame from the same night'
                                      'for ' + fits.filename)
                                found = 1
                        else:
                            print('Found matching twilight for FFLAT '
                                  'for ' + fits.filename)
                            found = 1

                    # if we haven't already found a matching FFLAT above (i.e. if found = 0), then
                    # go through the options with the flats:
                    if (found == 0):
                        # Try with normal FFLAT from flat:
                        filename_match = self.match_link(fits, 'fflat')
                        if filename_match is None:
                            # Try with looser criteria
                            filename_match = self.match_link(fits, 'fflat_loose')
                            if filename_match is None:
                                # Try using a flap flat instead
                                filename_match = self.match_link(fits, 'fflat_flap')
                                if filename_match is None:
                                    # Try with looser criteria
                                    filename_match = self.match_link(
                                        fits, 'fflat_flap_loose')
                                    if filename_match is None:
                                        # Still nothing. Raise an exception
                                        raise MatchException(
                                            'No matching tlmap found for ' +
                                            fits.filename)
                                    else:
                                        print('Warning: No good flat found for FFLAT. '
                                              'Using flap flat from different field '
                                              'for ' + fits.filename)
                                else:
                                    print('Warning: No dome flat found for FFLAT. '
                                          'Using flap flat instead for ' + fits.filename)
                            else:
                                print('Warning: No matching flat found for FFLAT. '
                                      'Using flat from different field for ' +
                                      fits.filename)
                        else:
                            print('Warning: No matching twilight found for FFLAT. '
                                  'Using a dome flat instead ' +
                                  fits.filename)

                ## elif match_class == 'fflat':
                ##     # Try with looser criteria
                ##     filename_match = self.match_link(fits, 'fflat_loose')
                ##     if filename_match is None:
                ##         # Try using a flap flat instead
                ##         filename_match = self.match_link(fits, 'fflat_flap')
                ##         if filename_match is None:
                ##             # Try with looser criteria
                ##             filename_match = self.match_link(
                ##                 fits, 'fflat_flap_loose')
                ##             if filename_match is None:
                ##                 # Still nothing. Raise an exception
                ##                 raise MatchException(
                ##                     'No matching fflat found for ' + 
                ##                     fits.filename)
                ##             else:
                ##                 print ('Warning: No good flat found for '
                ##                     'flat fielding. '
                ##                     'Using flap flat from different field '
                ##                     'for ' + fits.filename)
                ##         else:
                ##             print ('Warning: No dome flat found for flat '
                ##                 'fielding. '
                ##                 'Using flap flat instead for ' + fits.filename)
                ##     else:
                ##         print ('Warning: No matching flat found for flat '
                ##             'fielding. '
                ##             'Using flat from different field for ' + 
                ##             fits.filename)
                elif match_class == 'wavel':
                    # Try with looser criteria
                    filename_match = self.match_link(fits, 'wavel_loose')
                    if filename_match is None:
                        # Still nothing. Raise an exception
                        raise MatchException('No matching wavel found for ' +
                                             fits.filename)
                    else:
                        print('Warning: No good arc found for wavelength '
                              'solution. Using arc from different field '
                              'for ' + fits.filename)
                else:
                    # Anything else missing is fatal
                    raise MatchException('No matching ' + match_class +
                                         ' found for ' + fits.filename)
            if filename_match is not None:
                # Note we can't use else for the above line, because
                # filename_match might have changed
                # Make sure that 2dfdr gets the correct option names
                # We have added the tlmap_mfsky option here.
                if match_class == 'tlmap_flap':
                    match_class = 'tlmap'
                elif match_class == 'tlmap_mfsky':
                    match_class = 'tlmap'
                elif match_class == 'tlmap_mfsky_loose':
                    match_class = 'tlmap'
                elif match_class == 'tlmap_mfsky_any':
                    match_class = 'tlmap'
                elif match_class == 'thput_object':
                    match_class = 'thput'
                elif match_class == 'fflat_flap':
                    match_class = 'fflat'
                elif match_class == 'fflat_loose':
                    match_class = 'fflat'
                elif match_class == 'fflat_mfsky':
                    match_class = 'fflat'
                elif match_class == 'fflat_mfsky_loose':
                    match_class = 'fflat'
                elif match_class == 'fflat_mfsky_any':
                    match_class = 'fflat'

                options.extend(['-' + match_class.upper() + '_FILENAME',
                                filename_match])
        return options

    def determine_tlm_shift(self,fits,twilight_fits,flat_fits):

        twilight_fits = os.path.join(fits.reduced_dir,twilight_fits)
        flat_fits = os.path.join(fits.reduced_dir,flat_fits)
        
        twilight_tlm = pf.getdata(twilight_fits,'PRIMARY')
        flat_tlm = pf.getdata(flat_fits,'PRIMARY')
        
        tlm_offset = np.mean(twilight_tlm-flat_tlm)
        return tlm_offset

    def run_2dfdr_combine(self, file_iterable, output_path):
        """Use 2dfdr to combine the specified FITS files."""
        file_iterable, file_iterable_copy = itertools.tee(file_iterable)
        input_path_list = [fits.reduced_path for fits in file_iterable]
        if not input_path_list:
            print('No reduced files found to combine!')
            return
        # Following line uses the last FITS file, assuming all are the same CCD
        grating = next(file_iterable_copy).grating
        idx_file = self.idx_files[grating]
        print('Combining files to create', output_path)
        tdfdr.run_2dfdr_combine(input_path_list, output_path, idx_file)
        return

    def files(self, ndf_class=None, date=None, plate_id=None,
              plate_id_short=None, field_no=None, field_id=None,
              ccd=None, exposure_str=None, do_not_use=None,
              min_exposure=None, max_exposure=None,
              reduced_dir=None, reduced=None, copy_reduced=None,
              tlm_created=None, flux_calibrated=None, telluric_corrected=None,
              spectrophotometric=None, name=None, lamp=None, min_fluxlev=None,
              max_fluxlev=None,
              central_wavelength=None, include_linked_managers=False):
        """Generator for FITS files that satisfy requirements."""
        if include_linked_managers:
            # Include files from linked managers too
            file_list = itertools.chain(
                self.file_list,
                *[mngr.file_list for mngr in self.linked_managers])
        else:
            file_list = self.file_list  # type: List[FITSFile]
        for fits in file_list:
            if fits.ndf_class is None:
                continue
            if ((ndf_class is None or fits.ndf_class in ndf_class) and
                    (date is None or fits.date in date) and
                    (plate_id is None or fits.plate_id in plate_id) and
                    (plate_id_short is None or
                     fits.plate_id_short in plate_id_short) and
                    (field_no is None or fits.field_no == field_no) and
                    (field_id is None or fits.field_id in field_id) and
                    (ccd is None or fits.ccd in ccd) and
                    (exposure_str is None or
                     fits.exposure_str in exposure_str) and
                    (do_not_use is None or fits.do_not_use == do_not_use) and
                    (min_exposure is None or fits.exposure >= min_exposure) and
                    (max_exposure is None or fits.exposure <= max_exposure) and
                    (min_fluxlev is None or fits.fluxlev >= min_fluxlev) and  # add flux level limits
                    (max_fluxlev is None or fits.fluxlev <= max_fluxlev) and
                    (reduced_dir is None or
                     os.path.realpath(reduced_dir) ==
                     os.path.realpath(fits.reduced_dir)) and
                    (reduced is None or
                     (reduced and os.path.exists(fits.reduced_path)) or
                     (not reduced and
                      not os.path.exists(fits.reduced_path))) and
                    (copy_reduced is None or
                     (copy_reduced and os.path.exists(
                         self.copy_path(fits.reduced_path))) or
                     (not copy_reduced and not os.path.exists(
                         self.copy_path(fits.reduced_path)))) and
                    (tlm_created is None or
                     (tlm_created and hasattr(fits, 'tlm_path') and
                      os.path.exists(fits.tlm_path)) or
                     (not tlm_created and hasattr(fits, 'tlm_path') and
                      not os.path.exists(fits.tlm_path))) and
                    (flux_calibrated is None or
                     (flux_calibrated and hasattr(fits, 'fluxcal_path') and
                      os.path.exists(fits.fluxcal_path)) or
                     (not flux_calibrated and hasattr(fits, 'fluxcal_path') and
                      not os.path.exists(fits.fluxcal_path))) and
                    (telluric_corrected is None or
                     (telluric_corrected and hasattr(fits, 'telluric_path') and
                      os.path.exists(fits.telluric_path)) or
                     (not telluric_corrected and
                      hasattr(fits, 'telluric_path') and
                      not os.path.exists(fits.telluric_path))) and
                    (spectrophotometric is None or
                     (hasattr(fits, 'spectrophotometric') and
                      (fits.spectrophotometric == spectrophotometric))) and
                    (name is None or
                     (fits.name is not None and fits.name in name)) and
                    (lamp is None or fits.lamp == lamp) and
                    (central_wavelength is None or
                     fits.central_wavelength == central_wavelength)):
                yield fits
        return

    def group_files_by(self, keys, require_this_manager=True, **kwargs):
        """Return a dictionary of FITSFile objects grouped by the keys."""
        if isinstance(keys, six.string_types):
            keys = [keys]
        groups = defaultdict(list)
        for fits in self.files(**kwargs):
            combined_key = []
            for key in keys:
                combined_key.append(getattr(fits, key))
            combined_key = tuple(combined_key)
            groups[combined_key].append(fits)
        if require_this_manager:
            # Check that at least one of the files from each group has come
            # from this manager
            for combined_key, fits_list in list(groups.items()):
                for fits in fits_list:
                    if fits in self.file_list:
                        break
                else:
                    # None of the files are from this manager
                    del groups[combined_key]
        return groups

    def ccds(self, do_not_use=False):
        """Generator for ccd names in the data."""
        ccd_list = []
        for fits in self.files(do_not_use=do_not_use):
            if fits.ccd not in ccd_list:
                ccd_list.append(fits.ccd)
                yield fits.ccd
        return

    def reduced_dirs(self, dir_type=None, **kwargs):
        """Generator for reduced directories containing particular files."""
        reduced_dir_list = []
        if dir_type is None:
            ndf_class = None
            spectrophotometric = None
        else:
            ndf_class = {'bias': 'BIAS',
                         'dark': 'DARK',
                         'lflat': 'LFLAT',
                         'calibrators': ['MFFFF', 'MFARC', 'MFSKY'],
                         'object': 'MFOBJECT',
                         'mffff': 'MFFFF',
                         'mfarc': 'MFARC',
                         'mfsky': 'MFSKY',
                         'mfobject': 'MFOBJECT',
                         'spectrophotometric': 'MFOBJECT'}[dir_type.lower()]
            if dir_type == 'spectrophotometric':
                spectrophotometric = True
            elif ndf_class == 'MFOBJECT':
                spectrophotometric = False
            else:
                spectrophotometric = None
        for fits in self.files(ndf_class=ndf_class,
                               spectrophotometric=spectrophotometric,
                               **kwargs):
            if fits.reduced_dir not in reduced_dir_list:
                reduced_dir_list.append(fits.reduced_dir)
                yield fits.reduced_dir
        return

    def dark_exposure_strs(self, ccd, do_not_use=False):
        """Generator for dark exposure strings for a given ccd name."""
        exposure_str_list = []
        for fits in self.files(ndf_class='DARK', ccd=ccd,
                               do_not_use=do_not_use):
            if fits.exposure_str not in exposure_str_list:
                exposure_str_list.append(fits.exposure_str)
                yield fits.exposure_str
        return

    def combined_filenames_paths(self, calibrator_type, do_not_use=False):
        """Generator for filename and path of XXXXcombined.fits files."""
        self.check_calibrator_type(calibrator_type)
        for ccd in self.ccds(do_not_use=do_not_use):
            if calibrator_type.lower() == 'bias':
                yield (ccd,
                       None,
                       self.bias_combined_filename(),
                       self.bias_combined_path(ccd))
            elif calibrator_type.lower() == 'dark':
                for exposure_str in self.dark_exposure_strs(
                        ccd, do_not_use=do_not_use):
                    yield (ccd,
                           exposure_str,
                           self.dark_combined_filename(exposure_str),
                           self.dark_combined_path(ccd, exposure_str))
            elif calibrator_type.lower() == 'lflat':
                yield (ccd,
                       None,
                       self.lflat_combined_filename(),
                       self.lflat_combined_path(ccd))
        return

    def other_arm(self, fits, include_linked_managers=False):
        """Return the FITSFile from the other arm of the spectrograph."""
        if fits.ccd == 'ccd_1':
            other_number = '2'
        elif fits.ccd == 'ccd_2':
            other_number = '1'
        else:
            raise ValueError('Unrecognised CCD: ' + fits.ccd)
        other_filename = fits.filename[:5] + other_number + fits.filename[6:]
        other_fits = self.fits_file(
            other_filename, include_linked_managers=include_linked_managers)
        return other_fits

    def cubed_path(self, name, arm, fits_list, field_id, gzipped=False,
                   exists=False, tag=None, **kwargs):
        """Return the path to the cubed file."""
        n_file = len(self.qc_for_cubing(fits_list, **kwargs))
        path = os.path.join(
            self.abs_root, 'cubed', name,
            name + '_' + arm + '_' + str(n_file) + '_' + field_id)
        if tag:
            path += '_' + tag
        path += '.fits'
        if gzipped:
            path = path + '.gz'
        if exists:
            if not os.path.exists(path):
                path = self.cubed_path(name, arm, fits_list, field_id,
                                       gzipped=(not gzipped), exists=False,
                                       tag=tag, **kwargs)
                if not os.path.exists(path):
                    return None
        return path

    def matchmaker(self, fits, match_class):
        """Return the file that should be used to help reduce the FITS file.

        match_class is one of the following:
        tlmap_mfsky      -- Find a tramline map from twilight flat fields
        tlmap_mfsky_loose-- Find a tramline map from any twilight flat field on a night
        tlmap_mfsky_any  -- Find a tramline map from any twilight flat field in a manager set 
        tlmap            -- Find a tramline map from the dome lamp
        tlmap_loose      -- As tlmap, but with less strict criteria
        tlmap_flap       -- As tlmap, but from the flap lamp
        tlmap_flap_loose -- As tlmap_flap, but with less strict criteria
        wavel            -- Find a reduced arc file
        wavel_loose      -- As wavel, but with less strict criteria
        fflat_mfsky      -- Find a reduced fibre flat field from a twilight flat
        fflat_mfsky_loose-- Find a reduced fibre flat field from any twilight flat field on a night
        fflat_mksky_any  -- Find a reduced fibre flat field from any twilight flat field in a manager set 
        fflat            -- Find a reduced fibre flat field from the dome lamp
        fflat_loose      -- As fflat, but with less strict criteria
        fflat_flap       -- As fflat, but from the flap lamp
        fflat_flap_loose -- As fflat_flap, but with less strict criteria
        thput            -- Find a reduced offset sky (twilight) file
        thput_fflat      -- Find a dome flat that's had a copy made as MFSKY
        thput_sky        -- As thput, but find long-exposure object file
        bias             -- Find a combined bias frame
        dark             -- Find a combined dark frame
        lflat            -- Find a combined long-slit flat frame
        fcal             -- Find a reduced spectrophotometric standard star
        fcal_loose       -- As fcal, but with less strict criteria

        The return type depends on what is asked for:
        tlmap, wavel, fflat, thput, fcal and related 
                                -- A FITS file object
        bias, dark, lflat       -- The path to the combined file
        """
        fits_match = None
        # The following are the things that could potentially be matched
        date = None
        plate_id = None
        field_id = None
        ccd = None
        exposure_str = None
        min_exposure = None
        max_exposure = None
        reduced_dir = None
        reduced = None
        copy_reduced = None
        tlm_created = None
        flux_calibrated = None
        telluric_corrected = None
        spectrophotometric = None
        lamp = None
        central_wavelength = None
        # extra match criteria that is the amount of flux in the
        # frame, based on the FLXU90P value (9-95th percentile value
        # of raw frame).  This is for twilights used as flats for
        # TLMs.  If a frame is a twilight, then this paramater is
        # set on initialization of the FITSFile object.  Then we
        # have easy access to the value.
        min_fluxlev = None
        max_fluxlev = None
        # Define some functions for figures of merit
        time_difference = lambda fits, fits_test: (
            abs(fits_test.epoch - fits.epoch))
        recent_reduction = lambda fits, fits_test: (
                -1.0 * os.stat(fits_test.reduced_path).st_mtime)
        copy_recent_reduction = lambda fits, fits_test: (
                -1.0 * os.stat(self.copy_path(fits_test.reduced_path)).st_mtime)
        # merit function that returns the best fluxlev value.  As the
        # general f-o-m selects objects if the f-o-m is LESS than other values
        # we should just multiple fluxlev by -1:
        flux_level = lambda fits, fits_test: (
                -1.0 * fits_test.fluxlev)

        def time_difference_min_exposure(min_exposure):
            def retfunc(fits, fits_test):
                if fits_test.exposure <= min_exposure:
                    return np.inf
                else:
                    return time_difference(fits, fits_test)

            return retfunc

        def determine_tlm_shift_fits(twilight_fits,flat_fits):

            twilight_tlm = pf.getdata(twilight_fits.tlm_path,'PRIMARY')
            flat_tlm = pf.getdata(flat_fits.tlm_path,'PRIMARY')

            tlm_offset = np.mean(twilight_tlm-flat_tlm)
            return tlm_offset

        def flux_level_shift(fits,fits_test):
            fits_comp = self.matchmaker(fits,'tlmap')
            shift = determine_tlm_shift_fits(fits_test,fits_comp)
            if np.abs(shift) >= 1:
                    return np.inf
            else:
                return flux_level(fits, fits_test)

        # Determine what actually needs to be matched, depending on match_class
        #
        # this case is where we want to use a twilight sky frame to derive the
        # tramline maps, rather than a flat field, as the flat can often have too
        # little flux in the far blue to do a good job.  The order of matching for
        # the twilights should be:
        # 1) The brightest twilight frame of the same field (needs to be brighter than
        #    some nominal level, say FLUX90P>500) - tlmap_mfsky.
        # 2) The brightest twilight frame from the same night (same constraint on
        #    brightness) - tlmap_mfsky_loose.
        # 3) The brightest twilight frame from a different night (same constraint on
        #    brightness) - tlmap_mfsky_any.
        if match_class.lower() == 'tlmap_mfsky':
            # allow MFSKY to be used:
            ndf_class = 'MFSKY'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            tlm_created = True
            fom = flux_level
        elif match_class.lower() == 'tlmap_mfsky_loose':
            # this is the case where we take the brightest twilight on the same
            # night, irrespective of whether its from the same plate.
            ndf_class = 'MFSKY'
            date = fits.date
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            tlm_created = True
            fom = flux_level_shift
        elif match_class.lower() == 'tlmap_mfsky_any':
            # in this case find the best (brightest) twilight frame from anywhere
            # during the run.
            ndf_class = 'MFSKY'
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            tlm_created = True
            fom = flux_level_shift
        elif match_class.lower() == 'tlmap':
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'tlmap_loose':
            # Find a tramline map with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'tlmap_flap':
            # Find a tramline map from a flap flat
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'tlmap_flap_loose':
            # Tramline map from flap flat with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            tlm_created = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'wavel':
            # Find a reduced arc field
            ndf_class = 'MFARC'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            fom = time_difference
        elif match_class.lower() == 'wavel_loose':
            # Find a reduced arc field, with looser criteria
            ndf_class = 'MFARC'
            ccd = fits.ccd
            reduced = True
            fom = time_difference
        # options for using twilight frame as flibre flat:
        elif match_class.lower() == 'fflat_mfsky':
            # allow MFSKY to be used:
            ndf_class = 'MFSKY'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            copy_reduced = True
            fom = flux_level
        elif match_class.lower() == 'fflat_mfsky_loose':
            # this is the case where we take the brightest twilight on the same
            # night, irrespective of whether its from the same plate.
            ndf_class = 'MFSKY'
            date = fits.date
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            copy_reduced = True
            fom = flux_level_shift
        elif match_class.lower() == 'fflat_mfsky_any':
            # in this case find the best (brightest) twilight frame from anywhere
            # during the run.
            ndf_class = 'MFSKY'
            min_fluxlev = 1000.0
            max_fluxlev = 40000.0  # use a max_fluxlev to reduce the chance of saturated twilights
            ccd = fits.ccd
            copy_reduced = True
            fom = flux_level_shift
        elif match_class.lower() == 'fflat':
            # Find a reduced fibre flat field from the dome lamp
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'fflat_loose':
            # Find a reduced fibre flat field with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            reduced = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'fflat_flap':
            # Find a reduced flap fibre flat field
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'fflat_flap_loose':
            # Fibre flat field from flap lamp with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            reduced = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'thput':
            # Find a reduced offset sky field
            ndf_class = 'MFSKY'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            fom = recent_reduction
        elif match_class.lower() == 'thput_fflat':
            # Find a dome flat that's had a fake sky copy made
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            copy_reduced = True
            fom = copy_recent_reduction
        elif match_class.lower() == 'thput_object':
            # Find a reduced object field to take the throughput from
            ndf_class = 'MFOBJECT'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            fom = time_difference_min_exposure(
                self.min_exposure_for_throughput)
        elif match_class.lower() == 'fcal':
            # Find a spectrophotometric standard star
            ndf_class = 'MFOBJECT'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            spectrophotometric = True
            central_wavelength = fits.central_wavelength
            fom = time_difference
        elif match_class.lower() == 'fcal_loose':
            # Spectrophotometric with less strict criteria
            ndf_class = 'MFOBJECT'
            ccd = fits.ccd
            reduced = True
            spectrophotometric = True
            central_wavelength = fits.central_wavelength
            fom = time_difference
        elif match_class.lower() == 'bias':
            # Just return the standard BIAScombined filename
            filename = self.bias_combined_filename()
            if os.path.exists(os.path.join(fits.reduced_dir, filename)):
                return filename
            else:
                return None
        elif match_class.lower() == 'dark':
            # This works a bit differently. Return the filename of the
            # combined dark frame with the closest exposure time.
            best_fom = np.inf
            exposure_str_match = None
            for exposure_str in self.dark_exposure_strs(ccd=fits.ccd):
                test_fom = abs(float(exposure_str) - fits.exposure)
                if test_fom < best_fom:
                    exposure_str_match = exposure_str
                    best_fom = test_fom
            if exposure_str_match is None:
                return None
            filename = self.dark_combined_filename(exposure_str_match)
            if os.path.exists(os.path.join(fits.reduced_dir, filename)):
                return filename
            else:
                return None
        elif match_class.lower() == 'lflat':
            # Just return the standard LFLATcombined filename
            filename = self.lflat_combined_filename()
            if os.path.exists(os.path.join(fits.reduced_dir, filename)):
                return filename
            else:
                return None
        else:
            raise ValueError('Unrecognised match_class')
        # Perform the match
        best_fom = np.inf
        for fits_test in self.files(
                ndf_class=ndf_class,
                date=date,
                plate_id=plate_id,
                field_id=field_id,
                ccd=ccd,
                exposure_str=exposure_str,
                min_exposure=min_exposure,
                max_exposure=max_exposure,
                reduced_dir=reduced_dir,
                reduced=reduced,
                copy_reduced=copy_reduced,
                tlm_created=tlm_created,
                flux_calibrated=flux_calibrated,
                telluric_corrected=telluric_corrected,
                spectrophotometric=spectrophotometric,
                lamp=lamp,
                min_fluxlev=min_fluxlev,
                max_fluxlev=max_fluxlev,
                do_not_use=False,
        ):
            test_fom = fom(fits, fits_test)
            if test_fom < best_fom:
                fits_match = fits_test
                best_fom = test_fom
        #        exit()
        if (best_fom == np.inf) & (('tlmap_mfsky' in match_class.lower()) | ('fflat_mfsky' in match_class.lower())):
            return None
        return fits_match

    def match_link(self, fits, match_class):
        """Match and make a link to a file, and return the filename."""
        #        print 'started match_link: ',match_class,fits.filename
        fits_match = self.matchmaker(fits, match_class)

        if fits_match is None:
            # No match was found, send the lack of match onwards
            return None
        if match_class.lower() in ['bias', 'dark', 'lflat']:
            # matchmaker returns a filename in these cases; send it straight on
            filename = fits_match
        elif match_class.lower().startswith('tlmap'):
            filename = fits_match.tlm_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
            # add im file as this is needed for tlm offset estimate:
            imfilename = fits_match.im_filename
        elif match_class.lower() == 'thput':
            thput_filename = 'thput_' + fits_match.reduced_filename
            thput_path = os.path.join(fits_match.reduced_dir, thput_filename)
            if os.path.exists(thput_path):
                filename = thput_filename
            else:
                filename = fits_match.reduced_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
        elif match_class.lower() == 'thput_fflat':
            filename = self.copy_path(fits_match.reduced_filename)
            raw_filename = self.copy_path(fits_match.filename)
            raw_dir = fits_match.reduced_dir
        elif match_class.lower().startswith('fflat_mfsky'):
            # case of using twilight frame for fibre flat.  In this case
            # we need to use the copy_reduced_filename, that is the one
            # with the leading 9 in the file name:
            filename = fits_match.copy_reduced_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
        else:
            filename = fits_match.reduced_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
        # These are the cases where we do want to make a link
        require_link = [
            'tlmap_mfsky', 'tlmap_mfsky_loose', 'tlmap_mfsky_any',
            'tlmap', 'tlmap_loose', 'tlmap_flap', 'tlmap_flap_loose',
            'fflat', 'fflat_loose', 'fflat_flap', 'fflat_flap_loose',
            'wavel', 'wavel_loose', 'thput', 'thput_fflat', 'thput_object',
            'tlmap_mfsky', 'fflat_mfsky', 'fflat_mfsky_loose', 'fflat_mfsky_any']
        if match_class.lower() in require_link:
            link_path = os.path.join(fits.reduced_dir, filename)
            source_path = os.path.join(fits_match.reduced_dir, filename)
            raw_link_path = os.path.join(fits.reduced_dir, raw_filename)
            raw_source_path = os.path.join(raw_dir, raw_filename)
            # If the link path is occupied by a link, delete it
            # Leave actual files in place
            if os.path.islink(link_path):
                os.remove(link_path)
            if os.path.islink(raw_link_path):
                os.remove(raw_link_path)
            # Make a link, unless the file is already there
            if not os.path.exists(link_path):
                os.symlink(os.path.relpath(source_path, fits.reduced_dir),
                           link_path)
            if not os.path.exists(raw_link_path):
                os.symlink(os.path.relpath(raw_source_path, fits.reduced_dir),
                           raw_link_path)
            # add links to im files if we are looking for a TLM:
            if match_class.lower().startswith('tlmap'):
                im_link_path = os.path.join(fits.reduced_dir, imfilename)
                im_source_path = os.path.join(fits_match.reduced_dir, imfilename)
                if os.path.islink(im_link_path):
                    os.remove(im_link_path)
                if not os.path.exists(im_link_path):
                    os.symlink(os.path.relpath(im_source_path, fits.reduced_dir),
                               im_link_path)
        return filename

    def change_speed(self, speed=None):
        """Switch between fast and slow reductions."""
        if speed is None:
            if self.speed == 'fast':
                speed = 'slow'
            else:
                speed = 'fast'
        if speed not in ('fast', 'slow'):
            raise ValueError("Speed must be 'fast' or 'slow'.")
        self.speed = speed
        self.idx_files = IDX_FILES[self.speed]
        return

    @contextmanager
    def connection(self, server='aatlxa', username=None, password=None):
        """Make a secure connection to a remote server."""
        if not PYSFTP_AVAILABLE:
            print("You must install the pysftp package to do that!")
        if username is None:
            if self.aat_username is None:
                username = input('Enter AAT username: ')
                self.aat_username = username
            else:
                username = self.aat_username
        if password is None:
            if self.aat_password is None:
                password = getpass('Enter AAT password: ')
                self.aat_password = password
            else:
                password = self.aat_password
        try:
            srv = pysftp.Connection(server, username=username, password=password)
        except pysftp.paramiko.AuthenticationException:
            print('Authentication failed! Check username and password.')
            self.aat_username = None
            self.aat_password = None
            yield None
        else:
            try:
                yield srv
            finally:
                srv.close()

    def load_2dfdr_gui(self, fits_or_dirname):
        """Load the 2dfdr GUI in the required directory."""
        if isinstance(fits_or_dirname, FITSFile):
            # A FITS file has been provided, so go to its directory
            dirname = fits_or_dirname.reduced_dir
            idx_file = self.idx_files[fits_or_dirname.grating]
        else:
            # A directory name has been provided
            dirname = fits_or_dirname
            # Let the GUI sort out what idx file to use
            # TODO: Look in the directory for a suitable fits file to work out
            # the idx file
            idx_file = None
        tdfdr.load_gui(dirname, idx_file=idx_file)
        return

    def find_directory_locks(self, lock_name='2dfdrLockDir'):
        """Return a list of directory locks that currently exist."""
        lock_list = []
        for dirname, subdirname_list, _ in os.walk(self.abs_root):
            if lock_name in subdirname_list:
                lock_list.append(os.path.join(dirname, lock_name))
        return lock_list

    def remove_directory_locks(self, lock_name='2dfdrLockDir'):
        """Remove all 2dfdr locks from directories."""
        for path in self.find_directory_locks(lock_name=lock_name):
            os.rmdir(path)
        return

    def list_checks(self, recent_ever='both', *args, **kwargs):
        """Return a list of checks that need to be done."""
        if 'do_not_use' not in kwargs:
            kwargs['do_not_use'] = False
        # Each element in the list will be a tuple, where
        # element[0] = key from below
        # element[1] = list of fits objects to be checked
        if recent_ever == 'both':
            complete_list = []
            complete_list.extend(self.list_checks('ever', *args, **kwargs))
            # Should ditch the duplicate checks, but will work anyway
            complete_list.extend(self.list_checks('recent', *args, **kwargs))
            return complete_list
        # The keys for the following defaultdict will be tuples, where
        # key[0] = 'TLM' (or similar)
        # key[1] = tuple according to CHECK_DATA group_by
        # key[2] = 'recent' or 'ever'
        check_dict = defaultdict(list)
        for fits in self.files(*args, **kwargs):
            if recent_ever == 'ever':
                items = fits.check_ever.items()
            elif recent_ever == 'recent':
                items = fits.check_recent.items()
            else:
                raise KeyError(
                    'recent_ever must be "both", "ever" or "recent"')

            # Iterate over checks which have been explicitly been marked as `False`
            for key in [key for key, value in items if value is False]:
                check_dict_key = []
                for attribute_to_group_by in CHECK_DATA[key]['group_by']:
                    check_dict_key.append(getattr(fits, attribute_to_group_by))
                check_dict_key = (key, tuple(check_dict_key), recent_ever)
                check_dict[check_dict_key].append(fits)
        # Now change the dictionary into a sorted list
        key_func = lambda item: CHECK_DATA[item[0][0]]['priority']
        check_list = sorted(check_dict.items(), key=key_func)
        return check_list

    def print_checks(self, *args, **kwargs):
        """Print the list of checks to be done."""
        check_list = self.list_checks(*args, **kwargs)
        for index, (key, fits_list) in enumerate(check_list):
            check_data = CHECK_DATA[key[0]]
            print('{}: {}'.format(index, check_data['name']))
            if key[2] == 'ever':
                print('Never been checked')
            else:
                print('Not checked since last re-reduction')
            for group_by_key, group_by_value in zip(
                    check_data['group_by'], key[1]):
                print('   {}: {}'.format(group_by_key, group_by_value))
            for fits in fits_list:
                print('      {}'.format(fits.filename))
        return

    def check_next_group(self, *args, **kwargs):
        """Perform required checks on the highest priority group."""
        if len(self.list_checks(*args, **kwargs)) == 0:
            print("Yay! no more checks to do.")
            return
        self.check_group(0, *args, **kwargs)

    def check_group(self, index, *args, **kwargs):
        """Perform required checks on the specified group."""
        try:
            key, fits_list = self.list_checks(*args, **kwargs)[index]
        except IndexError:
            print("Check group '{}' does not exist.\n"
                  + "Try mngr.print_checks() for a list of "
                  + "available checks.").format(index)
            return
        check_method = getattr(self, 'check_' + key[0].lower())
        check_method(fits_list)
        print('Have you finished checking all the files? (y/n)')
        print('If yes, the check will be removed from the list.')
        y_n = input(' > ') + "n"
        finished = (y_n.lower()[0] == 'y')
        if finished:
            print('Removing this test from the list.')
            for fits in fits_list:
                fits.update_checks(key[0], True)
        else:
            print('Leaving this test in the list.')
        print('\nIf any files need to be disabled, use commands like:')
        print(">>> mngr.disable_files(['" + fits_list[0].filename + "'])")
        print('To add comments to a specifc file, use commands like:')
        print(">>> mngr.add_comment(['" + fits_list[0].filename + "'])")
        return

    def check_2dfdr(self, fits_list, message, filename_type='reduced_filename'):
        """Use 2dfdr to perform a check of some sort."""
        print('Use 2dfdr to plot the following files.')
        print('You may need to click on the triangles to see reduced files.')
        print('If the files are not listed, use the plot commands in the 2dfdr menu.')
        for fits in fits_list:
            print('   ' + getattr(fits, filename_type))
        print(message)
        self.load_2dfdr_gui(fits_list[0])
        return

    def check_bia(self, fits_list):
        """Check a set of bias frames."""
        # Check the individual bias frames, and then the combined file
        message = 'Check that the bias frames have no more artefacts than normal.'
        self.check_2dfdr(fits_list, message)
        combined_path = self.bias_combined_path(fits_list[0].ccd)
        if os.path.exists(combined_path):
            check_plots.check_bia(combined_path)
        return

    def check_drk(self, fits_list):
        """Check a set of dark frames."""
        # Check the individual dark frames, and then the combined file
        message = 'Check that the dark frames are free from any stray light.'
        self.check_2dfdr(fits_list, message)
        combined_path = self.dark_combined_path(fits_list[0].ccd,
                                                fits_list[0].exposure_str)
        if os.path.exists(combined_path):
            check_plots.check_drk(combined_path)
        return

    def check_lfl(self, fits_list):
        """Check a set of long-slit flats."""
        # Check the individual long-slit flats, and then the combined file
        message = 'Check that the long-slit flats have smooth illumination.'
        self.check_2dfdr(fits_list, message)
        combined_path = self.lflat_combined_path(fits_list[0].ccd)
        if os.path.exists(combined_path):
            check_plots.check_lfl(combined_path)
        return

    def check_tlm(self, fits_list):
        """Check a set of tramline maps."""
        message = 'Zoom in to check that the red fitted tramlines go through the data.'
        filename_type = 'tlm_filename'
        self.check_2dfdr(fits_list, message, filename_type)
        return

    def check_arc(self, fits_list):
        """Check a set of arc frames."""
        message = 'Zoom in to check that the arc lines are vertically aligned.'
        self.check_2dfdr(fits_list, message)
        return

    def check_flt(self, fits_list):
        """Check a set of flat field frames."""
        message = 'Check that the output varies smoothly with wavelength.'
        self.check_2dfdr(fits_list, message)
        return

    def check_sky(self, fits_list):
        """Check a set of offset sky (twilight) frames."""
        message = 'Check that the spectra are mostly smooth, and any features are aligned.'
        self.check_2dfdr(fits_list, message)
        return

    def check_obj(self, fits_list):
        """Check a set of reduced object frames."""
        message = 'Check that there is flux in each hexabundle, with no bad artefacts.'
        self.check_2dfdr(fits_list, message)
        return

    def check_flx(self, fits_list):
        """Check a set of spectrophotometric frames."""
        check_plots.check_flx(fits_list)
        return

    def check_tel(self, fits_list):
        """Check a set of telluric corrections."""
        check_plots.check_tel(fits_list)
        return

    def check_ali(self, fits_list):
        """Check the alignment of a set of object frames."""
        check_plots.check_ali(fits_list)
        return

    def check_cub(self, fits_list):
        """Check a set of final datacubes."""
        check_plots.check_cub(fits_list)
        return

    def _add_comment_to_file(self, fits_file_name, user_comment):
        """Add a comment to the FITS file corresponding to the name (with path)
        ``fits_file_name``.
        """

        try:
            hdulist = pf.open(fits_file_name, 'update',
                              do_not_scale_image_data=True)
            hdulist[0].header['COMMENT'] = user_comment
            hdulist.close()
        except IOError:
            return

    def add_comment(self, fits_list):
        """Add a comment to the FITS header of the files in ``fits_list``."""

        # Separate file names from vanilla names.
        # Run one thing on vanilla names
        # Run the other thing on file names.

        user_comment = input('Please enter a comment (type n to abort):\n')

        # If ``user_comment`` is equal to ``'n'``, skip updating the FITS
        # headers and jump to the ``return`` statement.
        if user_comment != 'n':

            time_stamp = 'Comment added by SAMI Observer on '
            time_stamp += '{:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
            user_comment += ' (' + time_stamp + ')'

            fits_file_list, FITSFile_list = [], []
            for fits_file in fits_list:
                if os.path.isfile(fits_file):
                    fits_file_list.append(fits_file)
                elif isinstance(self.fits_file(fits_file), FITSFile):
                    FITSFile_list.append(self.fits_file(fits_file))
                else:
                    error_message = "'{}' must be a valid file name".format(
                        fits_file)
                    error_message += "Please use the full path for combined "
                    error_message += "products, and simple filenames for raw "
                    error_message += "files."

                    raise ValueError(error_message)

            # Add the comments to each FITSFile instance.
            list(map(FITSFile.add_header_item,
                     FITSFile_list,
                     ['COMMENT' for _ in FITSFile_list],
                     [user_comment for _ in FITSFile_list]))

            # Add the comments to each instance of pyfits.
            list(map(Manager._add_comment_to_file,
                     [self for _ in fits_file_list],
                     fits_file_list,
                     [user_comment for _ in fits_file_list]))

            comments_file = os.path.join(self.root, 'observer_comments.txt')
            with open(comments_file, "a") as infile:
                comments_list = [
                    '{}: '.format(fits_file) \
                    + user_comment + '\n'
                    for fits_file in FITSFile_list]
                comments_list += [
                    '{}: '.format(fits_file) \
                    + user_comment + '\n'
                    for fits_file in fits_file_list]
                infile.writelines(comments_list)

        return


class FITSFile:
    """Holds information about a FITS file to be copied."""

    def __init__(self, input_path):
        self.input_path = input_path
        self.source_path = os.path.realpath(input_path)
        self.filename = os.path.basename(self.source_path)
        self.filename_root = self.filename[:self.filename.rfind('.')]
        try:
            self.hdulist = pf.open(self.source_path)
        except IOError:
            self.ndf_class = None
            return
        self.header = self.hdulist[0].header
        self.set_ndf_class()
        self.set_reduced_filename()
        self.set_copy_reduced_filename()
        self.set_date()
        if self.ndf_class and self.ndf_class not in ['BIAS', 'DARK', 'LFLAT']:
            self.set_fibres_extno()
        else:
            self.fibres_extno = None
        self.set_coords()
        if self.ndf_class and self.ndf_class not in ['BIAS', 'DARK', 'LFLAT']:
            self.set_plate_id()
            self.set_plate_id_short()
            self.set_field_no()
            self.set_field_id()
        else:
            self.plate_id = None
            self.plate_id_short = None
            self.field_no = None
            self.field_id = None
        self.set_ccd()
        self.set_detector()
        self.set_grating()
        self.set_exposure()
        self.set_epoch()
        # define the fluxlev property that is the 5-95th percentile range for the
        # raw frame.  This is a reasonable metric to use for assessing whether
        # twilight frames have sufficient flux:
        if self.ndf_class == 'MFSKY':
            flux = self.hdulist[0].data
            p05 = np.nanpercentile(flux, 5.0)
            p95 = np.nanpercentile(flux, 95.0)
            self.fluxlev = p95 - p05
            log.debug('%s: 5th,95th flux percentile: %s, %s, range:%s', self.filename, p05, p95, self.fluxlev)

        self.set_lamp()
        self.set_central_wavelength()
        self.set_do_not_use()
        self.set_coords_flags()
        self.set_copy()
        self.hdulist.close()
        del self.hdulist

    def __repr__(self):
        return "FITSFile {}, type: {}".format(self.filename, self.ndf_class)

    def set_ndf_class(self):
        """Save the NDF_CLASS of an AAT fits file."""
        for hdu in self.hdulist:
            if ('EXTNAME' in hdu.header.keys() and
                    (hdu.header['EXTNAME'] == 'STRUCT.MORE.NDF_CLASS' or
                     hdu.header['EXTNAME'] == 'NDF_CLASS')):
                # It has a class
                self.ndf_class = hdu.data['NAME'][0]
                # Change DFLAT to LFLAT
                if self.ndf_class == 'DFLAT':
                    self.overwrite_ndf_class('LFLAT')
                # Ask the user if SFLAT should be changed to MFSKY
                if self.ndf_class == 'SFLAT':
                    print('NDF_CLASS of SFLAT (OFFSET FLAT) found for ' +
                          self.filename)
                    print('Change to MFSKY (OFFSET SKY)? (y/n)')
                    y_n = input(' > ')
                    if y_n.lower()[0] == 'y':
                        self.overwrite_ndf_class('MFSKY')
                break
        else:
            self.ndf_class = None

    def set_reduced_filename(self):
        """Set the filename for the reduced file."""
        self.reduced_filename = self.filename_root + 'red.fits'
        # also set the intermediate im.fits filename as this can
        # be used for some processing steps, for example cross-matching
        # to get the best TLM offset:
        self.im_filename = self.filename_root + 'im.fits'
        if self.ndf_class == 'MFFFF':
            self.tlm_filename = self.filename_root + 'tlm.fits'
        # If the object is an MFSKY, then set the name of the
        # tlm_filename to be the copy of the MFSKY that is reduced
        # as a MFFFF:
        elif self.ndf_class == 'MFSKY':
            old_num = int(self.filename_root[6:10])
            new_num = old_num + 1000 * (9 - (old_num // 1000))
            new_filename_root = (self.filename_root[:6] + '{:04d}'.format(int(new_num)) + self.filename_root[10:])
            self.tlm_filename = new_filename_root + 'tlm.fits'
            # If the file is a copy, then we'll also need to set the copy name as
            # the im_filename, as this is the one that will be looked for when
            # doing the TLM offset measurements:
            self.im_filename = new_filename_root + 'im.fits'

        elif self.ndf_class == 'MFOBJECT':
            self.fluxcal_filename = self.filename_root + 'fcal.fits'
            self.telluric_filename = self.filename_root + 'sci.fits'
        return

    def set_copy_reduced_filename(self):
        """Set the filename for the reduced version of a copied file.
        This will be numbered 06mar19001red.fits rather than  06mar10001red.fits"""
        self.copy_reduced_filename = self.filename_root + 'red.fits'
        old_num = int(self.filename_root[6:10])
        new_num = old_num + 1000 * (9 - (old_num // 1000))
        new_filename_root = (self.filename_root[:6] + '{:04d}'.format(int(new_num)) + self.filename_root[10:])
        self.copy_reduced_filename = new_filename_root + 'red.fits'

        return

    def set_date(self):
        """Save the observation date as a 6-digit string yymmdd."""
        try:
            file_orig = self.header['FILEORIG']
            finish = file_orig.rfind('/', 0, file_orig.rfind('/'))
            self.date = file_orig[finish - 6:finish]
        except KeyError:
            self.date = None
        return

    def set_fibres_extno(self):
        """Save the extension number for the fibre table."""
        self.fibres_extno = find_fibre_table(self.hdulist)

    def set_plate_id(self):
        """Save the plate ID."""
        try:
            # First look in the primary header
            self.plate_id = self.header['PLATEID']
        except KeyError:
            # Check in the fibre table instead
            header = self.hdulist[self.fibres_extno].header
            self.plate_id = header['PLATEID']
            match = re.match(r'(^run_[0-9]+_)(P[0-9]+$)', self.plate_id)
            comment = 'Plate ID (from config file)'
            if match and 'star plate' in header['LABEL']:
                # This is a star field; adjust the plate ID accordingly
                self.plate_id = match.group(1) + 'star_' + match.group(2)
                comment = 'Plate ID (edited by manager)'
            # Also save it in the primary header, for future reference
            try:
                self.add_header_item('PLATEID', self.plate_id, comment,
                                     source=True)
            except IOError:
                # This probably means we don't have write access to the
                # source file. Ideally we would still edit the copied file,
                # but that doesn't actually exist yet.
                pass
        if self.plate_id == '':
            self.plate_id = 'none'
        return

    def set_plate_id_short(self):
        """Save the shortened plate ID."""
        finish = self.plate_id.find('_', self.plate_id.find('_') + 1)
        first_sections = self.plate_id[:finish]
        if self.plate_id == 'none':
            self.plate_id_short = 'none'
        elif (re.match(r'^Y[0-9]{2}S(A|B)R[0-9]+_P[0-9]+$', first_sections) or
              re.match(r'^A[0-9]+_P[0-9]+$', first_sections)):
            self.plate_id_short = first_sections
        else:
            self.plate_id_short = self.plate_id
        return

    def set_field_no(self):
        """Save the field number."""
        if int(self.date) < 130101:
            # SAMIv1. Can only get the field number by cross-checking the
            # config file RA and Dec.
            for pilot_field in PILOT_FIELD_LIST:
                cfg_coords = ICRS(
                    ra=self.cfg_coords['ra'],
                    dec=self.cfg_coords['dec'],
                    unit=self.cfg_coords['unit'])
                if (cfg_coords.separation(
                        ICRS(pilot_field['coords'])).arcsecs < 1.0
                        and self.plate_id == pilot_field['plate_id']):
                    self.field_no = pilot_field['field_no']
                    break
            else:
                raise RuntimeError('Field is from pilot data but cannot find'
                                   ' it in list of known pilot fields: ' +
                                   self.filename)
        else:
            # SAMIv2. Should be in the fibre table header somewhere
            header = self.hdulist[self.fibres_extno].header
            # First, see if it's included in the LABEL keyword
            match = re.search(r'(field )([0-9]+)', header['LABEL'])
            if match:
                # Yes, it is
                self.field_no = int(match.group(2))
            else:
                # The field number should be included in the filename of
                # the config file.
                match = re.search(r'(.*_f)([0-9]+)', header['FILENAME'])
                if match:
                    self.field_no = int(match.group(2))
                else:
                    # Nothing found. Default to 0.
                    self.field_no = 0
        return

    def set_field_id(self):
        """Save the field ID."""
        if self.plate_id == 'none':
            self.field_id = 'none'
        else:
            # First check if the LABEL keyword is of the correct form
            expr = (r'(.*?)( - )(Run [0-9]+ .*? plate [0-9]+)'
                    r'( - )(field [0-9]+)')
            header = self.hdulist[self.fibres_extno].header
            match = re.match(expr, header['LABEL'])
            if match:
                # It is, so just copy the field ID directly
                self.field_id = match.group(1)
            elif (self.plate_id.startswith('run_') or
                  re.match(r'[0-9]+S[0-9]+', self.plate_id)):
                # Pilot and commissioning data. No field ID in the plate ID, so
                # append one.
                self.field_id = self.plate_id + '_F' + str(self.field_no)
            elif (re.match(r'^Y[0-9]{2}S(A|B)R[0-9]+_P[0-9]+$',
                           self.plate_id_short) or
                  re.match(r'^A[0-9]+_P[0-9]+$',
                           self.plate_id_short)):
                # Main survey or early cluster data. Field ID is stored within the 
                # plate ID.
                start = len(self.plate_id_short)
                for i in range(self.field_no):
                    start = self.plate_id.find('_', start) + 1
                finish = self.plate_id.find('_', start)
                if finish == -1:
                    field_id = self.plate_id[start:]
                else:
                    field_id = self.plate_id[start:finish]
                self.field_id = self.plate_id_short + '_' + field_id
            elif re.match(r'^A[0-9]+T[0-9]+_A[0-9]+T[0-9]+$', self.plate_id):
                # Cluster data. Field ID is one segment of the plate ID.
                start = 0
                for i in range(self.field_no - 1):
                    start = self.plate_id.find('_', start) + 1
                finish = self.plate_id.find('_', start)
                if finish == -1:
                    field_id = self.plate_id[start:]
                else:
                    field_id = self.plate_id[start:finish]
                self.field_id = field_id
            else:
                # Unrecognised form for the plate ID.
                self.field_id = self.plate_id + '_F' + str(self.field_no)
        return

    def set_coords(self):
        """Save the RA/Dec and config RA/Dec."""
        if self.ndf_class and self.ndf_class not in ['BIAS', 'DARK', 'LFLAT']:
            header = self.hdulist[self.fibres_extno].header
            self.cfg_coords = {'ra': header['CFGCENRA'],
                               'dec': header['CFGCENDE'],
                               'unit': (units.radian, units.radian)}
            if self.ndf_class == 'MFOBJECT':
                self.coords = {'ra': header['CENRA'],
                               'dec': header['CENDEC'],
                               'unit': (units.radian, units.radian)}
            else:
                self.coords = None
        else:
            self.cfg_coords = None
        return

    def set_detector(self):
        """Set the specific detector name, e.g. E2V2A etc.  This is different from
        the ccd name as ccd is just whether ccd_1 (blue) or ccd_2 (red).  We need
        to know which detector as some reduction steps can be different, e.g. treatment
        of bias and dark frames."""
        if self.ndf_class:
            detector_id = self.header['DETECTOR']
            self.detector = detector_id
        else:
            self.detector = None
        return

    def set_ccd(self):
        """Set the CCD name."""
        if self.ndf_class:
            spect_id = self.header['SPECTID']
            if spect_id == 'BL':
                self.ccd = 'ccd_1'
            elif spect_id == 'RD':
                self.ccd = 'ccd_2'
            else:
                self.ccd = 'unknown_ccd'
        else:
            self.ccd = None
        return

    def set_grating(self):
        """Set the grating name."""
        if self.ndf_class:
            self.grating = self.header['GRATID']
        else:
            self.grating = None
        return

    def set_exposure(self):
        """Set the exposure time."""
        if self.ndf_class:
            self.exposure = self.header['EXPOSED']
            self.exposure_str = '{:d}'.format(int(np.round(self.exposure)))
        else:
            self.exposure = None
            self.exposure_str = None
        return

    def set_epoch(self):
        """Set the observation epoch."""
        if self.ndf_class:
            self.epoch = self.header['EPOCH']
        else:
            self.epoch = None
        return

    def set_lamp(self):
        """Set which lamp was on, if any."""
        if self.ndf_class == 'MFARC':
            lamp = self.header['LAMPNAME']
            if lamp == '':
                # No lamp set. Most likely that the CuAr lamp was on but the
                # control software screwed up. Patch the headers assuming this
                # is the case, but warn the user.
                lamp = 'CuAr'
                hdulist_write = pf.open(self.source_path, 'update')
                hdulist_write[0].header['LAMPNAME'] = lamp
                hdulist_write[0].header['OBJECT'] = 'ARC - ' + lamp
                hdulist_write.flush()
                hdulist_write.close()
                print('No arc lamp specified for ' + self.filename)
                print('Updating LAMPNAME and OBJECT keywords assuming a ' +
                      lamp + ' lamp')
            self.lamp = lamp
        elif self.ndf_class == 'MFFFF':
            if self.exposure >= 17.5:
                # Assume longer exposures are dome flats
                self.lamp = 'Dome'
            else:
                self.lamp = 'Flap'
        else:
            self.lamp = None
        return

    def set_central_wavelength(self):
        """Set what the requested central wavelength of the observation was."""
        if self.ndf_class:
            self.central_wavelength = self.header['LAMBDCR']
        else:
            self.central_wavelength = None
        return

    def set_do_not_use(self):
        """Set whether or not to use this file."""
        try:
            self.do_not_use = self.header['DONOTUSE']
        except KeyError:
            # By default, don't use fast readout files
            self.do_not_use = (self.header['SPEED'] != 'NORMAL')
        return

    def set_coords_flags(self):
        """Set whether coordinate corrections have been done."""
        try:
            self.coord_rot = self.header['COORDROT']
        except KeyError:
            self.coord_rot = None
        try:
            self.coord_rev = self.header['COORDREV']
        except KeyError:
            self.coord_rev = None
        return

    def set_copy(self):
        """Set whether this is a copy of a file."""
        try:
            self.copy = self.header['MNGRCOPY']
        except KeyError:
            self.copy = False
        return

    def relevant_check(self, check):
        """Return True if a visual check is relevant for this file."""
        return (self.ndf_class == check['ndf_class'] and
                (check['spectrophotometric'] is None or
                 check['spectrophotometric'] == self.spectrophotometric))

    def set_check_data(self):
        """Set whether the relevant checks have been done."""
        self.check_ever = {}
        self.check_recent = {}
        for key in [key for key, check in CHECK_DATA.items()
                    if self.relevant_check(check)]:
            try:
                check_done_ever = self.header['MNCH' + key]
            except KeyError:
                check_done_ever = None
            self.check_ever[key] = check_done_ever
            try:
                check_done_recent = self.header['MNCH' + key + 'R']
            except KeyError:
                check_done_recent = None
            self.check_recent[key] = check_done_recent
        return

    def make_reduced_link(self):
        """Make the link in the reduced directory."""
        if not os.path.exists(self.reduced_dir):
            os.makedirs(self.reduced_dir)
        if not os.path.exists(self.reduced_link):
            os.symlink(os.path.relpath(self.raw_path, self.reduced_dir),
                       self.reduced_link)
        return

    def reduce_options(self):
        """Return a dictionary of options used to reduce the file."""
        if not os.path.exists(self.reduced_path):
            return None
        return dict(pf.getdata(self.reduced_path, 'REDUCTION_ARGS'))

    def update_name(self, name):
        """Change the object name assigned to this file."""
        if self.name != name:
            if re.match(r'.*[\\\[\]*/?<>|;:&,.$ ].*', name):
                raise ValueError(r'Invalid character in name; '
                                 r'do not use any of []\/*?<>|;:&,.$ or space')
            # Update the FITS header
            self.add_header_item('MNGRNAME', name,
                                 'Object name set by SAMI manager')
            # Update the object
            self.name = name
        return

    def update_spectrophotometric(self, spectrophotometric):
        """Change the spectrophotometric flag assigned to this file."""
        if self.spectrophotometric != spectrophotometric:
            # Update the FITS header
            self.add_header_item('MNGRSPMS', spectrophotometric,
                                 'Flag set if a spectrophotometric star')
            # Update the object
            self.spectrophotometric = spectrophotometric
        return

    def update_do_not_use(self, do_not_use):
        """Change the do_not_use flag assigned to this file."""
        if self.do_not_use != do_not_use:
            # Update the FITS header
            self.add_header_item('DONOTUSE', do_not_use,
                                 'Do Not Use flag for SAMI manager')
            # Update the object
            self.do_not_use = do_not_use
            # Update the file system
            if do_not_use:
                if os.path.exists(self.reduced_link):
                    os.remove(self.reduced_link)
            else:
                self.make_reduced_link()
        return

    def update_checks(self, key, value, force=False):
        """Update both check flags for this key for this file."""
        self.update_check_recent(key, value)
        # Don't set the "ever" check to False unless forced, or there
        # is no value set yet
        if value or force or self.check_ever[key] is None:
            self.update_check_ever(key, value)
        return

    def update_check_recent(self, key, value):
        """Change one of the check flags assigned to this file."""
        if self.check_recent[key] != value:
            # Update the FITS header
            self.add_header_item('MNCH' + key + 'R', value,
                                 CHECK_DATA[key]['name'] +
                                 ' checked since re-reduction')
            # Update the object
            self.check_recent[key] = value
        return

    def update_check_ever(self, key, value):
        """Change one of the check flags assigned to this file."""
        if self.check_ever[key] != value:
            # Update the FITS header
            self.add_header_item('MNCH' + key, value,
                                 CHECK_DATA[key]['name'] +
                                 ' checked ever')
            # Update the object
            self.check_ever[key] = value
        return

    def add_header_item(self, key, value, comment=None, source=False):
        """Add a header item to the FITS file."""
        if comment is None:
            value_comment = value
        else:
            value_comment = (value, comment)
        if source:
            path = self.source_path
        else:
            path = self.raw_path
        # old_header = pf.getheader(path)
        old_header = self.header
        # Only update if necessary
        if (key not in self.header or
                self.header[key] != value or
                type(self.header[key]) != type(value) or
                (comment is not None and self.header.comments[key] != comment)):
            with pf.open(path, 'update', do_not_scale_image_data=True) as hdulist:
                hdulist[0].header[key] = value_comment
                self.header = hdulist[0].header
        return

    def overwrite_ndf_class(self, new_ndf_class):
        """Change the NDF_CLASS value in the FITS file and in the object."""
        hdulist_write = pf.open(self.source_path, 'update')
        for hdu_name in ('STRUCT.MORE.NDF_CLASS', 'NDF_CLASS'):
            try:
                hdu = hdulist_write[hdu_name]
                break
            except KeyError:
                pass
        else:
            # No relevant extension found
            raise KeyError('No NDF_CLASS extension found in file')
        hdu.data['NAME'][0] = new_ndf_class
        hdulist_write.flush()
        hdulist_write.close()
        self.ndf_class = new_ndf_class
        return

    def has_sky_lines(self):
        """Return True if there are sky lines in the wavelength range."""
        # Coverage taken from http://ftp.aao.gov.au/2df/aaomega/aaomega_gratings.html
        coverage_dict = {
            '1500V': 750,
            '580V': 2100,
            '1000R': 1100,
        }
        coverage = coverage_dict[self.grating]
        wavelength_range = (
            self.header['LAMBDCR'] - 0.5 * coverage,
            self.header['LAMBDCR'] + 0.5 * coverage
        )
        # Highly incomplete list! May give incorrect results for high-res
        # red gratings
        useful_lines = (5577.338, 6300.309, 6553.626, 6949.066, 7401.862,
                        7889.680, 8382.402, 8867.605, 9337.874, 9799.827, 9972.357)
        for line in useful_lines:
            if wavelength_range[0] < line < wavelength_range[1]:
                # This sky line is within the observed wavelength range
                return True
        # No sky lines were within the observed wavelength range
        return False


def update_checks(key, file_iterable, value, force=False):
    """Set flags for whether the files have been manually checked."""
    for fits in file_iterable:
        fits.update_checks(key, value, force)
    return


def safe_for_multiprocessing(function):
    @wraps(function)
    def safe_function(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except KeyboardInterrupt:
            print("Handling KeyboardInterrupt in worker process")
            print("You many need to press Ctrl-C multiple times")
            result = None
        return result

    return safe_function


@safe_for_multiprocessing
def derive_transfer_function_pair(inputs):
    """Derive transfer function for a pair of fits files."""
    path_pair = inputs['path_pair']
    n_trim = inputs['n_trim']
    model_name = inputs['model_name']
    smooth = inputs['smooth']
    print('Deriving transfer function for ' +
          os.path.basename(path_pair[0]) + ' and ' +
          os.path.basename(path_pair[1]))
    try:
        fluxcal2.derive_transfer_function(
            path_pair, n_trim=n_trim, model_name=model_name, smooth=smooth,
            molecfit_available = MOLECFIT_AVAILABLE, molecfit_dir = MF_BIN_DIR,
            speed=inputs['speed'],tell_corr_primary=inputs['tellcorprim'])
    except ValueError:
        print('Warning: No star found in dataframe, skipping ' +
              os.path.basename(path_pair[0]))
        return
    good_psf = pf.getval(path_pair[0], 'GOODPSF',
                         'FLUX_CALIBRATION')
    if not good_psf:
        print('Warning: Bad PSF fit in ' + os.path.basename(path_pair[0]) +
              '; will skip this one in combining.')
    return


@safe_for_multiprocessing
def telluric_correct_pair(inputs):
    """Telluric correct a pair of fits files."""
    fits_1 = inputs['fits_1']
    fits_2 = inputs['fits_2']
    n_trim = inputs['n_trim']
    use_PS = inputs['use_PS']
    scale_PS_by_airmass = inputs['scale_PS_by_airmass']
    PS_spec_file = inputs['PS_spec_file']
    model_name = inputs['model_name']
    if fits_1 is None or not os.path.exists(fits_1.fluxcal_path):
        print('Matching blue arm not found for ' + fits_2.filename +
              '; skipping this file.')
        return
    path_pair = (fits_1.fluxcal_path, fits_2.fluxcal_path)
    print('Deriving telluric correction for ' + fits_1.filename +
          ' and ' + fits_2.filename)
    try:
        telluric.derive_transfer_function(
            path_pair, PS_spec_file=PS_spec_file, use_PS=use_PS, n_trim=n_trim,
            scale_PS_by_airmass=scale_PS_by_airmass, model_name=model_name,
            molecfit_available = MOLECFIT_AVAILABLE, molecfit_dir = MF_BIN_DIR,speed=inputs['speed'])
    except ValueError as err:
        if err.args[0].startswith('No star identified in file:'):
            # No standard star found; probably a star field
            print(err.args[0])
            print('Skipping telluric correction for file:', fits_2.filename)
            return False
        else:
            # Some other, unexpected error. Re-raise it.
            raise err
    for fits in (fits_1, fits_2):
        print('Telluric correcting file:', fits.filename)
        if os.path.exists(fits.telluric_path):
            os.remove(fits.telluric_path)
        telluric.apply_correction(fits.fluxcal_path,
                                  fits.telluric_path)
    return True


@safe_for_multiprocessing
def measure_offsets_group(group):
    """Measure offsets between a set of dithered observations."""
    field, fits_list, copy_to_other_arm, fits_list_other_arm = group
    print('Measuring offsets for field ID: {}'.format(field[0]))
    path_list = [best_path(fits) for fits in fits_list]
    print('These are the files:')
    for path in path_list:
        print('  ', os.path.basename(path))
    if len(path_list) < 2:
        # Can't measure offsets for a single file
        print('Only one file so no offsets to measure!')
        return
    find_dither(path_list, path_list[0], centroid=True,
                remove_files=True, do_dar_correct=True)
    if copy_to_other_arm:
        for fits, fits_other_arm in zip(fits_list, fits_list_other_arm):
            hdulist_this_arm = pf.open(best_path(fits))
            hdulist_other_arm = pf.open(best_path(fits_other_arm), 'update')
            try:
                del hdulist_other_arm['ALIGNMENT']
            except KeyError:
                # Nothing to delete; no worries
                pass
            hdulist_other_arm.append(hdulist_this_arm['ALIGNMENT'])
            hdulist_other_arm.flush()
            hdulist_other_arm.close()
            hdulist_this_arm.close()
    return


@safe_for_multiprocessing
def cube_group(group):
    """Cube a set of RSS files."""
    field, fits_list, root, overwrite, star_only = group
    print('Cubing field ID: {}, CCD: {}'.format(field[0], field[1]))
    path_list = [best_path(fits) for fits in fits_list]
    print('These are the files:')
    for path in path_list:
        print('  ', os.path.basename(path))
    if star_only:
        objects = [pf.getval(path_list[0], 'STDNAME', 'FLUX_CALIBRATION')]
    else:
        objects = 'all'
    if fits_list[0].epoch < 2013.0:
        # Large pitch of pilot data requires a larger drop size
        drop_factor = 0.75
    else:
        drop_factor = 0.5
    dithered_cubes_from_rss_list(
        path_list, suffix='_' + field[0], size_of_grid=50, write=True,
        nominal=True, root=root, overwrite=overwrite, do_dar_correct=True,
        objects=objects, clip=True, drop_factor=drop_factor)
    return


@safe_for_multiprocessing
def cube_object(inputs):
    """Cube a single object in a set of RSS files."""
    (field_id, ccd, path_list, name, cubed_root, drop_factor, tag,
     update_tol, size_of_grid, output_pix_size_arcsec, overwrite) = inputs
    print('Cubing {} in field ID: {}, CCD: {}'.format(name, field_id, ccd))
    print('{} files available'.format(len(path_list)))
    suffix = '_' + field_id
    if tag:
        suffix += '_' + tag
    return dithered_cube_from_rss_wrapper(
        path_list, name, suffix=suffix, write=True, nominal=True,
        root=cubed_root, overwrite=overwrite, do_dar_correct=True, clip=True,
        drop_factor=drop_factor, update_tol=update_tol,
        size_of_grid=size_of_grid,
        output_pix_size_arcsec=output_pix_size_arcsec)


def best_path(fits):
    """Return the best (most calibrated) path for the given file."""
    if os.path.exists(fits.telluric_path):
        path = fits.telluric_path
    elif os.path.exists(fits.fluxcal_path):
        path = fits.fluxcal_path
    else:
        path = fits.reduced_path
    return path


@safe_for_multiprocessing
def run_2dfdr_single_wrapper(group):
    """Run 2dfdr on a single file."""
    fits, idx_file, options = \
        group
    try:
        tdfdr.run_2dfdr_single(fits, idx_file, options=options)
    except tdfdr.LockException:
        message = ('Postponing ' + fits.filename +
                   ' while other process has directory lock.')
        print(message)
        return False
    return True


@safe_for_multiprocessing
def scale_cubes_field(group):
    """Scale a field to the correct magnitude."""
    star_path_pair, object_path_pair_list, star = group
    print('Scaling field with star', star)
    stellar_mags_cube_pair(star_path_pair, save=True)
    # Copy the PSF data to the galaxy datacubes
    star_header = pf.getheader(star_path_pair[0])
    for object_path_pair in object_path_pair_list:
        for object_path in object_path_pair:
            hdulist_write = pf.open(object_path, 'update')
            for key in ('PSFFWHM', 'PSFALPHA', 'PSFBETA'):
                try:
                    hdulist_write[0].header[key] = star_header[key]
                except KeyError:
                    pass
            hdulist_write.flush()
            hdulist_write.close()
    # Previously tried reading the catalogue once and passing it, but for
    # unknown reasons that was corrupting the data when run on aatmacb.
    found = assign_true_mag(star_path_pair, star, catalogue=None)
    if found:
        scale = scale_cube_pair_to_mag(star_path_pair, 3)
        for object_path_pair in object_path_pair_list:
            scale_cube_pair(object_path_pair, scale)
    else:
        print('No photometric data found for', star)
    return


@safe_for_multiprocessing
def fit_sec_template(path):
    """Fit theoretical templates to secondary calibration stars that have been
    selected to be halo F-stars.  This uses ppxf and save the best template and
    weight to the fits header."""

    # call the main template fitting routine for the given file:
    fluxcal2.fit_sec_template_ppxf(path)
    
    
    return

@safe_for_multiprocessing
def scale_frame_pair(path_pair):
    """Scale a pair of RSS frames to the correct magnitude."""
    print('Scaling RSS files to give star correct magnitude: %s' %
          str((os.path.basename(path_pair[0]), os.path.basename(path_pair[1]))))
    stellar_mags_frame_pair(path_pair, save=True)
    star = pf.getval(path_pair[0], 'STDNAME', 'FLUX_CALIBRATION')
    # Previously tried reading the catalogue once and passing it, but for
    # unknown reasons that was corrupting the data when run on aatmacb.
    found = assign_true_mag(path_pair, star, catalogue=None,
                            hdu='FLUX_CALIBRATION')
    if found:
        scale_cube_pair_to_mag(path_pair, 1, hdu='FLUX_CALIBRATION')
    else:
        print('No photometric data found for', star)
    return


@safe_for_multiprocessing
def bin_cubes_pair(path_pair):
    """Bin a pair of datacubes using each of the default schemes."""
    # TODO: Allow the user to specify name/kwargs pairs. Will require
    # coordination with Manager.bin_cubes() [JTA 23/9/2015]
    path_blue, path_red = path_pair
    print('Binning datacubes:')
    print(os.path.basename(path_blue), os.path.basename(path_red))
    binning_settings = (
        ('adaptive', {'mode': 'adaptive'}),
        ('annular', {'mode': 'prescriptive', 'sectors': 1}),
        ('sectors', {'mode': 'prescriptive'}))
    for name, kwargs in binning_settings:
        binning.bin_cube_pair(path_blue, path_red, name=name, **kwargs)
    return


@safe_for_multiprocessing
def aperture_spectra_pair(path_pair, overwrite=False):
    """Create aperture spectra for a pair of data cubes using default apertures."""
    path_blue, path_red = path_pair
    global CATALOG_PATH
    try:
        print('Processing: ' + path_blue + ', ' + path_red)
        binning.aperture_spectra_pair(path_blue, path_red, CATALOG_PATH, overwrite)
    except Exception as e:
        print("ERROR on pair %s, %s:\n %s" % (path_blue, path_red, e))
        traceback.print_exc()
    return


@safe_for_multiprocessing
def gzip_wrapper(path):
    """Gzip a single file."""
    print('Gzipping file: ' + path)
    gzip(path)
    return


# @safe_for_multiprocessing
# def test_function(variable):
#     import time
#     print("starting", variable)
#     start_time = time.time()
#     current_time = time.time()
#     while current_time < (start_time + 5):
#         print("waiting...")
#         time.sleep(1); current_time = time.time()
#     print("finishing", variable)

def assign_true_mag(path_pair, name, catalogue=None, hdu=0):
    """Find the magnitudes in a catalogue and save them to the header."""
    if catalogue is None:
        catalogue = read_stellar_mags()
        # in some cases the catalogue keys can end up at bytes, rather than as strings.
        # this is due to a numpy bug that is fixed in later versions.  It is certainly
        # fixed by numpy vesion 1.14.2.
        # this bug can lead to a failure of matching in the lines below (matched against "name").
    if name not in catalogue:
        return False
    line = catalogue[name]
    for path in path_pair:
        hdulist = pf.open(path, 'update')
        for band in 'ugriz':
            hdulist[hdu].header['CATMAG' + band.upper()] = (
                line[band], band + ' mag from catalogue')
        hdulist.flush()
        hdulist.close()
    return True

def read_stellar_mags():
    """Read stellar magnitudes from the various catalogues available.  Note that
    for python 3 some versions of numpy will have a problem with loadtxt() not
    converting the strings from byte to str.  This is fixed in later versions of
    numpy, so make sure to update your numpy."""
    data_dict = {}
    for (path, catalogue_type, extinction) in stellar_mags_files():
        if catalogue_type == 'ATLAS':
            names = ('PHOT_ID', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                     'sigma', 'radius')
            formats = ('U20', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                       'f8', 'f8')
            skiprows = 2
            delimiter = None
            name_func = lambda d: d['PHOT_ID']
        elif catalogue_type == 'SDSS_cluster':
            names = ('obj_id', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                     'priority')
            formats = ('U30', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i')
            skiprows = 0
            delimiter = None
            name_func = lambda d: '888' + d['obj_id'][-9:]
        elif catalogue_type == 'SDSS_GAMA':
            names = ('name', 'obj_id', 'ra', 'dec', 'type', 'u', 'sig_u',
                     'g', 'sig_g', 'r', 'sig_r', 'i', 'sig_i', 'z', 'sig_z')
            formats = ('U20', 'U30', 'f8', 'f8', 'U10', 'f8', 'f8',
                       'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')
            skiprows = 1
            delimiter = ','
            name_func = lambda d: d['name']
        data = np.loadtxt(path, skiprows=skiprows, delimiter=delimiter,
                          dtype={'names': names, 'formats': formats})
        if data.shape == ():
            # Ensure data is iterable in case of only a single line
            data.shape = (1,)
        data['u'] += extinction[0]
        data['g'] += extinction[1]
        data['r'] += extinction[2]
        data['i'] += extinction[3]
        data['z'] += extinction[4]
        new_data_dict = {name_func(line): line for line in data}
        data_dict.update(new_data_dict)
    return data_dict


class MatchException(Exception):
    """Exception raised when no matching calibrator is found."""
    pass
