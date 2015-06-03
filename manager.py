"""Code for organising and reducing SAMI data.

Instructions on how to use this module are given in the docstring for
the  Manager class. The following describes some of the under-the-hood
details.

This module contains two classes: Manager and FITSFile. The Manager
stores information about an observing run, including a list of all raw
files. Each FITSFile object stores information about a particular raw
file. Note that neither object stores whether or not a file has been
reduced; this is checked on the fly when necessary.

When a Manager object is initiated, it makes an empty list to store
the raw files. It will then inspect given directories to find raw
files, with names of like 01jan10001.fits. It will reject duplicate
filenames. Each valid filename is used to initialise a FITSFile
object, which is added to the Manager's file list. The file itself is
also moved into a suitable location in the output directory structure.

Each FITSFile object stores basic information about the file, such as
the path to the raw file and to the reduced file. The plate and field
IDs are determined automatically from the FITS headers. A check is
made to see if the telescope was pointing at the field listed in the
MORE.FIBRES_IFU extension. If not, the user is asked to give a name
for the pointing, which will generally be the name of whatever object
was being observed. This name is then added to an "extra" list in the
Manager, so that subsequent observations at the same position will be
automatically recognised.

The Manager also keeps lists of the different dark frame exposure
lengths (as  both string and float), as well as a list of directories
that have been recently reduced, and hence should be visually checked.

There are three different methods for calling 2dfdr in different
modes:

Manager.run_2dfdr_single calls 2dfdr via the aaorun functionality to
reduce a single file. This is generally called from
Manager.reduce_file, which first uses Manager.matchmaker to work out
which calibration files should be used, and makes symbolic links to
them in the target file's directory.

Manager.run_2dfdr_auto calls 2dfdr in AutoScript mode and tells it to
auto-reduce everything in a specified directory. This is no longer used.

Manager.run_2dfdr_combine also uses AutoScript mode, this time to
combine a given list of files. This is used for making combined bias,
dark or long-slit flat frames.

Functionality for flux calibration and cubing are provided via
functions from other sami modules.

As individual files are reduced, entries are added to the checklist of
directories to visually inspect. There are some functions for loading
up 2dfdr in the relevant directories, but the user must select and
plot the individual files themself. This whole system is a bit clunky
and needs overhauling.

There are a few generators for useful items, most notably
Manager.files. This iterates through all entries in the internal file
list and yields those that satisfy a wide range of optional
parameters. """

import shutil
import os
import re
import subprocess
import multiprocessing
import signal
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict
from getpass import getpass
from time import sleep
from glob import glob
from pydoc import pager

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

from .utils.other import find_fibre_table, gzip
from .utils import IFU
from .general.cubing import dithered_cubes_from_rss_list, get_object_names
from .general.cubing import dithered_cube_from_rss_wrapper
from .general.cubing import scale_cube_pair, scale_cube_pair_to_mag
from .general.align_micron import find_dither
from .dr import fluxcal2, telluric, check_plots, tdfdr, dust, binning
from .dr.throughput import make_clipped_thput_files
from .qc.fluxcal import stellar_mags_cube_pair, stellar_mags_frame_pair
from .qc.fluxcal import throughput, get_sdss_stellar_mags
from .qc.sky import sky_residuals
from .qc.arc import bad_fibres

# Get the astropy version as a tuple of integers
ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))
if ASTROPY_VERSION[:2] == (0, 2):
    ICRS = coord.ICRSCoordinates
elif ASTROPY_VERSION[:2] == (0, 3):
    ICRS = coord.ICRS
else:
    def ICRS(*args, **kwargs):
        return coord.SkyCoord(*args, frame='icrs', **kwargs)

IDX_FILES_SLOW = {'1': 'sami580V_v1_2.idx',
                  '2': 'sami1000R_v1_2.idx',
                  'ccd_1': 'sami580V_v1_2.idx',
                  'ccd_2': 'sami1000R_v1_2.idx'}
IDX_FILES_FAST = {'1': 'sami580V.idx',
                  '2': 'sami1000R.idx',
                  'ccd_1': 'sami580V.idx',
                  'ccd_2': 'sami1000R.idx'}
IDX_FILES = {'fast': IDX_FILES_FAST,
             'slow': IDX_FILES_SLOW}

GRATLPMM = {'ccd_1': 582.0,
            'ccd_2': 1001.0}

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

    Other functions
    ===============

    The other functions defined probably aren't useful to you.
    """

    def __init__(self, root, copy_files=False, move_files=False, fast=False,
                 gratlpmm=GRATLPMM, n_cpu=1, demo=False,
                 demo_data_source='demo'):
        if fast:
            self.speed = 'fast'
        else:
            self.speed = 'slow'
        self.idx_files = IDX_FILES[self.speed]
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
        if demo:
            if PATCH_AVAILABLE:
                print 'WARNING: Manager is in demo mode.'
                print 'No actual data reduction will take place!'
            else:
                print 'You must install the mock module to use the demo mode.'
                print 'Continuing in normal mode.'
                demo = False
        self.demo = demo
        self.demo_data_source = demo_data_source

    def map(self, function, input_list):
        """Map inputs to a function, using built-in map or multiprocessing."""
        if not input_list:
            # input_list is empty. I expected the map functions to deal with
            # this issue, but in one case it hung on aatmacb, so let's be
            # absolutely sure to avoid the issue
            return []
        if self.n_cpu == 1:
            result_list = map(function, input_list)
        else:
            pool = multiprocessing.Pool(self.n_cpu)
            result_list = pool.map(function, input_list, chunksize=1)
            pool.close()
            pool.join()
        return result_list

    def inspect_root(self, copy_files, move_files, trust_header=True):
        """Add details of existing files to internal lists."""
        for dirname, subdirname_list, filename_list in os.walk(self.abs_root):
            for filename in filename_list:
                if self.file_filter(filename):
                    self.import_file(dirname, filename,
                                     trust_header=trust_header,
                                     copy_files=copy_files,
                                     move_files=move_files)
        return

    def file_filter(self, filename):
        """Return True if the file should be added."""
        # Match filenames of the form 01jan10001.fits
        return (re.match(r'[0-3][0-9]'
                         r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
                         r'[1-2][0-9]{4}\.(fit|fits|FIT|FITS)$',
                         filename)
                and (self.fits_file(filename) is None))

    def import_file(self, dirname, filename,
                    trust_header=True, copy_files=True, move_files=False):
        """Add details of a file to the manager"""
        source_path = os.path.join(dirname, filename)
        fits = FITSFile(source_path)
        if fits.copy:
            # This is a copy of a file, don't add it to the list
            return
        if fits.ndf_class == 'DARK':
            if fits.exposure_str not in self.dark_exposure_str_list:
                self.dark_exposure_str_list.append(fits.exposure_str)
                self.dark_exposure_list.append(fits.exposure)
        self.set_raw_path(fits)
        if os.path.abspath(fits.source_path) != os.path.abspath(fits.raw_path):
            if copy_files:
                print 'Copying file:', filename
                self.update_copy(fits.source_path, fits.raw_path)
            if move_files:
                print 'Moving file: ', filename
                self.move(fits.source_path, fits.raw_path)
            if not copy_files and not move_files:
                print 'Warning! Adding', filename, 'in unexpected location'
                fits.raw_path = fits.source_path
        else:
            print 'Adding file: ', filename
        self.set_name(fits, trust_header=trust_header)
        fits.set_check_data()
        self.set_reduced_path(fits)
        if not fits.do_not_use:
            fits.make_reduced_link()
        fits.add_header_item('GRATLPMM', self.gratlpmm[fits.ccd])
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
                    best_name = raw_input('Enter object name for file ' +
                                          fits.filename + '\n > ')
                except ValueError as error:
                    print error
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
            yn = raw_input('Is ' + fits.name + ' in file ' + fits.filename +
                           ' a spectrophotometric standard? (y/n)\n > ')
            spectrophotometric_input = (yn.lower()[0] == 'y')
            fits.update_spectrophotometric(spectrophotometric_input)
        # If the field was new and it's not a "main", add it to the list
        if name_extra is None and name_coords is None:
            self.extra_list.append(
                {'name':fits.name,
                 'coords':fits_coords,
                 'spectrophotometric':fits.spectrophotometric,
                 'fitsfile':fits})
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
                print error
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
                    self.update_copy(os.path.join(dirname, filename),
                                     os.path.join(self.tmp_dir, filename))
                    self.import_file(self.tmp_dir, filename,
                                     trust_header=trust_header,
                                     copy_files=False, move_files=True)
        if os.path.exists(self.tmp_dir) and len(os.listdir(self.tmp_dir)) == 0:
            os.rmdir(self.tmp_dir)
        return

    def import_aat(self, username=None, password=None, date=None,
                   server='aatlxa', path='/data_lxy/aatobs/OptDet_data'):
        """Import from the AAT data disks."""
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
                        self.import_file(self.tmp_dir, filename,
                                         trust_header=False, copy_files=False,
                                         move_files=True)
        if os.path.exists(self.tmp_dir) and len(os.listdir(self.tmp_dir)) == 0:
            os.rmdir(self.tmp_dir)
        return

    def fits_file(self, filename):
        """Return the FITSFile object that corresponds to the given filename."""
        filename_options = [filename, filename+'.fit', filename+'.fits',
                            filename+'.FIT', filename+'.FITS']
        for fits in self.file_list:
            if fits.filename in filename_options:
                return fits

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
        for fits in file_iterable:
            if isinstance(fits, str):
                fits = self.fits_file(fits)
            fits.update_do_not_use(False)
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

    def reduce_calibrator(self, calibrator_type, overwrite=False, **kwargs):
        """Reduce all biases, darks of lflats."""
        self.check_calibrator_type(calibrator_type)
        file_iterable = self.files(ndf_class=calibrator_type.upper(), 
                                   do_not_use=False, **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite)
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
                    if overwrite and os.path.exists(link_path):
                        os.remove(link_path)
                    if not os.path.exists(link_path):
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
        reduced_files = self.reduce_calibrator(
            'bias', overwrite=overwrite, **kwargs)
        self.update_checks('BIA', reduced_files, False)
        return

    def combine_bias(self, overwrite=False):
        """Produce and link necessary BIAScombined.fits files."""        
        self.combine_calibrator('bias', overwrite=overwrite)
        return

    def link_bias(self, overwrite=False):
        """Make necessary symbolic links for BIAScombined.fits files."""
        self.link_calibrator('bias', overwrite=overwrite)
        return

    def reduce_dark(self, overwrite=False, **kwargs):
        """Reduce all dark frames."""
        reduced_files = self.reduce_calibrator(
            'dark', overwrite=overwrite, **kwargs)
        self.update_checks('DRK', reduced_files, False)
        return
        
    def combine_dark(self, overwrite=False):
        """Produce and link necessary DARKcombinedXXXX.fits files."""
        self.combine_calibrator('dark', overwrite=overwrite)
        return

    def link_dark(self, overwrite=False):
        """Make necessary symbolic links for DARKcombinedXXXX.fits files."""
        self.link_calibrator('dark', overwrite=overwrite)
        return

    def reduce_lflat(self, overwrite=False, **kwargs):
        """Reduce all lflat frames."""
        reduced_files = self.reduce_calibrator(
            'lflat', overwrite=overwrite, **kwargs)
        self.update_checks('LFL', reduced_files, False)
        return

    def combine_lflat(self, overwrite=False):
        """Produce and link necessary LFLATcombined.fits files."""
        self.combine_calibrator('lflat', overwrite=overwrite)
        return

    def link_lflat(self, overwrite=False):
        """Make necessary symbolic links for LFLATcombined.fits files."""
        self.link_calibrator('lflat', overwrite=overwrite)
        return

    def make_tlm(self, overwrite=False, leave_reduced=False, **kwargs):
        """Make TLMs from all files matching given criteria."""
        file_iterable = self.files(ndf_class='MFFFF', do_not_use=False,
                                   **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite, tlm=True, 
            leave_reduced=leave_reduced)
        self.update_checks('TLM', reduced_files, False)
        return

    def reduce_arc(self, overwrite=False, **kwargs):
        """Reduce all arc frames matching given criteria."""
        file_iterable = self.files(ndf_class='MFARC', do_not_use=False,
                                   **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite)
        for fits in reduced_files:
            bad_fibres(fits.reduced_path, save=True)
        self.update_checks('ARC', reduced_files, False)
        return

    def reduce_fflat(self, overwrite=False, **kwargs):
        """Reduce all fibre flat frames matching given criteria."""
        file_iterable = self.files(ndf_class='MFFFF', do_not_use=False,
                                   **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite)
        self.update_checks('FLT', reduced_files, False)
        return

    def reduce_sky(self, overwrite=False, fake_skies=True, **kwargs):
        """Reduce all offset sky frames matching given criteria."""
        file_iterable = self.files(ndf_class='MFSKY', do_not_use=False,
                                   **kwargs)
        reduced_files = self.reduce_file_iterable(
            file_iterable, overwrite=overwrite)
        self.update_checks('SKY', reduced_files, False)
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
        new_num = old_num + 1000 * (9 - (old_num / 1000))
        new_filename = (
            fits.filename[:6] + '{:04d}'.format(new_num) + fits.filename[10:])
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
        # Add paths to the new FITSFile instance.
        # Don't use Manager.set_reduced_path because the raw location is
        # unusual
        new_fits.raw_dir = fits.reduced_dir
        new_fits.raw_path = new_path
        new_fits.reduced_dir = fits.reduced_dir
        new_fits.reduced_link = new_path
        new_fits.reduced_path = os.path.join(
            fits.reduced_dir, new_fits.reduced_filename)
        return new_fits

    def copy_path(self, path):
        """Return the path for a copy of the specified file."""
        directory = os.path.dirname(path)
        old_filename = os.path.basename(path)
        old_num = int(old_filename[6:10])
        new_num = old_num + 1000 * (9 - (old_num / 1000))
        new_filename = (
            old_filename[:6] + '{:04d}'.format(new_num) + old_filename[10:])
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
            file_iterable_long, overwrite=overwrite)
        # Check how good the sky subtraction was
        for fits in reduced_files:
            self.qc_sky(fits)
        if sky_residual_limit is not None:
            # Switch to sky line throughputs if the sky residuals are bad
            fits_list = self.files_with_bad_dome_throughput(
                reduced_files, sky_residual_limit=sky_residual_limit)
            self.reduce_file_iterable(
                fits_list, throughput_method='skylines',
                overwrite=True)
            bad_fields = np.unique([fits.field_id for fits in fits_list])
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
        reduced_files.extend(self.reduce_file_iterable(
            file_iterable_sky_lines, overwrite=overwrite,
            throughput_method='skylines'))
        reduced_files.extend(self.reduce_file_iterable(
            file_iterable_default, overwrite=overwrite))
        # Mark these files as not checked
        self.update_checks('OBJ', reduced_files, False)
        # Check how good the sky subtraction was
        for fits in reduced_files:
            self.qc_sky(fits)
        return

    def reduce_file_iterable(self, file_iterable, throughput_method='default', 
                             overwrite=False, tlm=False, leave_reduced=True):
        """Reduce all files in the iterable."""
        # First establish the 2dfdr options for all files that need reducing
        # Would be more memory-efficient to construct a generator
        input_list = [(fits, 
                       self.idx_files[fits.ccd],
                       tuple(self.tdfdr_options(
                           fits, throughput_method=throughput_method,
                           tlm=tlm)),
                       self.cwd,
                       self.imp_scratch,
                       self.scratch_dir)
                      for fits in file_iterable
                      if (overwrite or 
                          not os.path.exists(self.target_path(fits, tlm=tlm)))]
        reduced_files = [item[0] for item in input_list]
        # Send the items out for reducing. Keep track of which ones were done.
        while input_list:
            print len(input_list), 'files remaining.'
            with self.patch_if_demo(
                    'sami.dr.tdfdr.run_2dfdr_single', fake_run_2dfdr_single):
                finished = np.array(self.map(
                    run_2dfdr_single_wrapper, input_list))
            input_list = [item for i, item in enumerate(input_list) 
                          if not finished[i]]
        # Delete unwanted reduced files
        for fits in reduced_files:
            if (fits.ndf_class == 'MFFFF' and tlm and not leave_reduced and
                os.path.exists(fits.reduced_path)):
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
            if fits.epoch < 2013.0:
                # SAMI v1 had awful throughput at blue end of blue, need to
                # trim that data
                n_trim = 3
            else:
                n_trim = 0
            inputs_list.append({'path_pair': path_pair, 'n_trim': n_trim, 
                                'model_name': model_name, 'smooth': smooth})
        with self.patch_if_demo(
                'sami.dr.fluxcal2.derive_transfer_function',
                fake_derive_transfer_function):
            self.map(derive_transfer_function_pair, inputs_list)
        return

    def combine_transfer_function(self, overwrite=False, **kwargs):
        """Combine and save transfer functions from multiple files."""
        # First sort the spectrophotometric files into date/field/CCD/name 
        # groups. Wouldn't need name except that currently different stars
        # are on different units; remove the name matching when that's
        # sorted.
        groups = self.group_files_by(('date', 'field_id', 'ccd', 'name'),
            ndf_class='MFOBJECT', do_not_use=False,
            spectrophotometric=True, **kwargs)
        # Now combine the files within each group
        for fits_list in groups.values():
            path_list = [fits.reduced_path for fits in fits_list]
            path_out = os.path.join(os.path.dirname(path_list[0]),
                                    'TRANSFERcombined.fits')
            if overwrite or not os.path.exists(path_out):
                print 'Combining files to create', path_out
                fluxcal2.combine_transfer_functions(path_list, path_out)
                # Run the QC throughput measurement
                self.qc_throughput_spectrum(path_out)
            # Copy the file into all required directories
            done = [os.path.dirname(path_list[0])]
            for path in path_list:
                if os.path.dirname(path) not in done:
                    path_copy = os.path.join(os.path.dirname(path),
                                             'TRANSFERcombined.fits')
                    done.append(os.path.dirname(path_copy))
                    if overwrite or not os.path.exists(path_copy):
                        print 'Copying combined file to', path_copy
                        shutil.copy2(path_out, path_copy)
            self.update_checks('FLX', fits_list, False)
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
                print 'Flux calibrating file:', fits.reduced_path
                if os.path.exists(fits.fluxcal_path):
                    os.remove(fits.fluxcal_path)
                path_transfer_fn = os.path.join(
                    fits_spectrophotometric.reduced_dir,
                    'TRANSFERcombined.fits')
                fluxcal2.primary_flux_calibrate(
                    fits.reduced_path,
                    fits.fluxcal_path,
                    path_transfer_fn)
        return

    def telluric_correct(self, overwrite=False, model_name=None, **kwargs):
        """Apply telluric correction to object frames."""
        # First make the list of file pairs to correct
        inputs_list = []
        for fits_2 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd='ccd_2', 
                                 **kwargs):
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
                    model_name_out = 'ref_centre_alpha_dist_circ_hdratm'
                else:
                    model_name_out = model_name
            inputs_list.append({
                'fits_1': fits_1,
                'fits_2': fits_2,
                'n_trim': n_trim,
                'use_PS': use_PS,
                'scale_PS_by_airmass': scale_PS_by_airmass,
                'PS_spec_file': PS_spec_file,
                'model_name': model_name_out})
        # Now send this list to as many cores as we are using
        # Limit this to 10, because of semaphore issues I don't understand
        old_n_cpu = self.n_cpu
        if old_n_cpu > 10:
            self.n_cpu = 10
        with self.patch_if_demo(
                'sami.dr.telluric.derive_transfer_function',
                fake_derive_transfer_function):
            done_list = self.map(telluric_correct_pair, inputs_list)
        self.n_cpu = old_n_cpu
        # Mark telluric corrections as not checked
        fits_2_list = [inputs['fits_2'] for inputs, done in 
                       zip(inputs_list, done_list) if done]
        self.update_checks('TEL', fits_2_list, False)
        for inputs in [inputs for inputs, done in
                       zip(inputs_list, done_list) if done]:
            # Copy the FWHM measurement to the QC header
            self.qc_seeing(inputs['fits_1'])
            self.qc_seeing(inputs['fits_2'])
        return

    def get_stellar_photometry(self, refresh=False):
        """Get photometry of stars, with help from the user."""
        if refresh:
            catalogue = None
        else:
            catalogue = read_stellar_mags()
        new = get_sdss_stellar_mags(self, catalogue=catalogue)
        if not new:
            # No new magnitudes were downloaded
            return
        path_in = raw_input('Enter the path to the downloaded file:\n')
        idx = 1
        path_out = 'standards/secondary/sdss_stellar_mags_{}.csv'.format(idx)
        while os.path.exists(path_out):
            idx += 1
            path_out = (
                'standards/secondary/sdss_stellar_mags_{}.csv'.format(idx))
        shutil.move(path_in, path_out)
        return

    def scale_frames(self, overwrite=False, **kwargs):
        """Scale individual RSS frames to the secondary standard flux."""
        # First make the list of file pairs to scale
        inputs_list = []
        for fits_2 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd='ccd_2', 
                                 telluric_corrected=True, **kwargs):
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
            **kwargs)
        complete_groups = []
        for key, fits_list in groups.items():
            fits_list_other_arm = [self.other_arm(fits) for fits in fits_list]
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
                    break
        with self.patch_if_demo('sami.manager.find_dither', fake_find_dither):
            self.map(measure_offsets_group, complete_groups)
        for group in complete_groups:
            self.update_checks('ALI', group[1], False)
        return

    def cube(self, overwrite=False, min_exposure=599.0, name='main', 
             star_only=False, drop_factor=None, tag='', update_tol=0.02,
             size_of_grid=50, output_pix_size_arcsec=0.5,
             min_transmission=0.333, max_seeing=4.0, **kwargs):
        """Make datacubes from the given RSS files."""
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, **kwargs)
        # Add in the root path as well, so that cubing puts things in the 
        # right place
        cubed_root = os.path.join(self.root, 'cubed')
        inputs_list = []
        for (field_id, ccd), fits_list in groups.items():
            good_fits_list = self.qc_for_cubing(
                fits_list, min_transmission=min_transmission,
                max_seeing=max_seeing, min_exposure=min_exposure)
            path_list = [best_path(fits) for fits in good_fits_list]
            if not path_list:
                # All frames failed the QC checks!
                continue
            if star_only:
                objects = [
                    pf.getval(path_list[0], 'STDNAME', 'FLUX_CALIBRATION')]
            else:
                objects = get_object_names(path_list[0])
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
        # Send the cubing tasks off to multiple CPUs
        with self.patch_if_demo('sami.manager.dithered_cubes_from_rss_wrapper',
                                fake_dithered_cube_from_rss_wrapper):
            self.map(cube_object, inputs_list)

        # groups = [(key, fits_list, cubed_root, overwrite, star_only) 
        #           for key, fits_list in groups.items()]
        # # Send the cubing tasks off to multiple CPUs
        # with self.patch_if_demo('sami.manager.dithered_cubes_from_rss_list',
        #                         fake_dithered_cubes_from_rss_list):
        #     self.map(cube_group, groups)
        # Mark all cubes as not checked. Ideally would only mark those that
        # actually exist. Maybe set dithered_cubes_from_rss_list to return a 
        # list of those it created?
        for fits_list in groups.values():
            self.update_checks('CUB', [fits_list[0]], False)
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
            fits_pair = (fits, self.other_arm(fits))
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
            reduced=True, name=name, **kwargs)
        input_list = []
        for (field_id, ), fits_list in groups.items():
            table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
            objects = table['NAME'][table['TYPE'] == 'P']
            objects = np.unique(objects).tolist()
            for name in objects:
                if telluric.is_star(name):
                    break
            else:
                print 'No star found in field, skipping: ' + field_id
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
        with self.patch_if_demo('sami.manager.stellar_mags_cube_pair',
                                fake_stellar_mags_cube_pair):
            self.map(scale_cubes_field, input_list)
        return

    def bin_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                  min_transmission=0.333, max_seeing=4.0, tag=None, **kwargs):
        """Apply default binning schemes to datacubes."""
        path_pair_list = []
        groups = self.group_files_by(
            'field_id', ccd='ccd_1', ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, **kwargs)
        for (field_id, ), fits_list in groups.items():
            table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
            objects = table['NAME'][table['TYPE'] == 'P']
            objects = np.unique(objects).tolist()
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
        return

    def record_dust(self, overwrite=False, min_exposure=599.0, name='main',
                    min_transmission=0.333, max_seeing=4.0, tag=None, **kwargs):
        """Record information about dust in the output datacubes."""
        groups = self.group_files_by(
            'field_id', ccd='ccd_1', ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, **kwargs)
        for (field_id, ), fits_list in groups.items():
            table = pf.getdata(fits_list[0].reduced_path, 'FIBRES_IFU')
            objects = table['NAME'][table['TYPE'] == 'P']
            objects = np.unique(objects).tolist()
            for name in objects:
                for arm in ('blue', 'red'):
                    path = self.cubed_path(
                        name, arm, fits_list, field_id,
                        exists=True, min_exposure=min_exposure,
                        min_transmission=min_transmission,
                        max_seeing=max_seeing, tag=tag)
                    if path:
                        dust.dustCorrectSAMICube(path, overwrite=overwrite)
        return
            
    def gzip_cubes(self, overwrite=False, min_exposure=599.0, name='main',
                   star_only=False, min_transmission=0.333, max_seeing=4.0,
                   tag=None, **kwargs):
        """Gzip the final datacubes."""
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, name=name, **kwargs)
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
            for obj in objects:
                input_path = self.cubed_path(
                    obj, arm, fits_list, field_id,
                    exists=True, min_exposure=min_exposure,
                    min_transmission=min_transmission,
                    max_seeing=max_seeing, tag=tag)
                if input_path:
                    output_path = input_path + '.gz'
                    if os.path.exists(output_path) and overwrite:
                        os.remove(output_path)
                    if not os.path.exists(output_path):
                        input_list.append(input_path)
        with self.patch_if_demo('sami.manager.gzip', fake_gzip):
            self.map(gzip_wrapper, input_list)
        return

    def reduce_all(self, overwrite=False, **kwargs):
        """Reduce everything, in order. Don't use unless you're sure."""
        self.reduce_bias(overwrite)
        self.combine_bias(overwrite)
        self.reduce_dark(overwrite)
        self.combine_dark(overwrite)
        self.reduce_lflat(overwrite)
        self.combine_lflat(overwrite)
        self.make_tlm(overwrite, **kwargs)
        self.reduce_arc(overwrite, **kwargs)
        self.reduce_fflat(overwrite, **kwargs)
        self.reduce_sky(overwrite, **kwargs)
        self.reduce_object(overwrite, **kwargs)
        self.derive_transfer_function(overwrite, **kwargs)
        self.combine_transfer_function(overwrite, **kwargs)
        self.flux_calibrate(overwrite, **kwargs)
        self.telluric_correct(overwrite, **kwargs)
        self.scale_frames(overwrite, **kwargs)
        self.measure_offsets(overwrite, **kwargs)
        self.cube(overwrite, **kwargs)
        self.scale_cubes(overwrite, **kwargs)
        self.bin_cubes(overwrite, **kwargs)
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
            print 'Warning: No mean throughput file found for QC checks.'
            found_mean = False
        if found_mean:
            relative_throughput = absolute_throughput / mean_throughput
            data = np.vstack((absolute_throughput, relative_throughput))
            median_relative_throughput = np.median(relative_throughput)
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
                          'MEDRELTH', 'THROUGHPUT') / 
                pf.getval(path, 'RESCALE', 'FLUX_CALIBRATION'))
        except KeyError:
            # Not all the data is available
            print 'Warning: Not all data required to calculate transmission is available.'
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
        text += '='*75+'\n'
        text += 'Use space bar and cursor keys to move up and down; q to quit.\n'
        text += 'If one file in a pair is disabled it is marked with a +\n'
        text += 'If both are disabled it is marked with a *\n'
        text += '\n'
        for (field_id,), fits_list in self.group_files_by(
                'field_id', ndf_class='MFOBJECT', min_exposure=min_exposure,
                ccd=ccd, **kwargs).items():
            text += '+'*75+'\n'
            text += field_id+'\n'
            text += '-'*75+'\n'
            text += 'File        Exposure  FWHM (")  Transmission  Sky residual\n'
            for fits in fits_list:
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
            text += '+'*75+'\n'
            text += '\n'
        pager(text)
        return

    def reduce_file(self, fits, overwrite=False, tlm=False,
                    leave_reduced=False):
        """Select appropriate options and reduce the given file.

        For MFFFF files, if tlm is True then a tramline map is produced; if it
        is false then a full reduction is done. If tlm is True and leave_reduced
        is false, then any reduced MFFFF produced as a side-effect will be
        removed.

        Returns True if the file was reduced; False otherwise."""
        target = self.target_path(fits, tlm=tlm)
        if os.path.exists(target) and not overwrite:
            # File to be created already exists, abandon this.
            return False
        options = self.tdfdr_options(fits, tlm=tlm)
        # All options have been set, so run 2dfdr
        tdfdr.run_2dfdr_single(fits, self.idx_files[fits.ccd], 
                               options=options, cwd=self.cwd)
        if (fits.ndf_class == 'MFFFF' and tlm and not leave_reduced and
            os.path.exists(fits.reduced_path)):
            os.remove(fits.reduced_path)
        return True

    def tdfdr_options(self, fits, throughput_method='default', tlm=False):
        """Set the 2dfdr reduction options for this file."""
        options = []
        if fits.ccd == 'ccd_2':
            if fits.exposure >= self.min_exposure_for_sky_wave:
                # Adjust wavelength calibration of red frames using sky lines
                options.extend(['-SKYSCRUNCH', '1'])
            else:
                options.extend(['-SKYSCRUNCH', '0'])
            # Turn off bias and dark subtraction
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
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap']
            # Arc frames can't use optimal extraction because 2dfdr screws up
            # and marks entire columns as bad when it gets too many saturated
            # pixels
            options.extend(['-EXTR_OPERATION', 'GAUSS'])
        elif fits.ndf_class == 'MFFFF' and not tlm:
            if fits.lamp == 'Flap':
                # Flap flats should use their own tramline maps, not those
                # generated by dome flats
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap_flap', 
                                  'wavel']
            else:
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel']
        elif fits.ndf_class == 'MFSKY':
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                              'fflat']
        elif fits.ndf_class == 'MFOBJECT':
            if throughput_method == 'default':
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                  'fflat', 'thput']
                options.extend(['-TPMETH', 'OFFSKY'])
            elif throughput_method == 'external':
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                  'fflat']
                options.extend(['-TPMETH', 'OFFSKY'])
                options.extend(['-THPUT_FILENAME',
                                'thput_'+fits.reduced_filename])
            elif throughput_method == 'skylines':
                if fits.exposure >= self.min_exposure_for_throughput:
                    files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                      'fflat']
                    options.extend(['-TPMETH', 'SKYFLUX(MED)'])
                else:
                    files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                      'fflat', 'thput_object']
                    options.extend(['-TPMETH', 'OFFSKY'])
        else:
            raise ValueError('Unrecognised NDF_CLASS: '+fits.ndf_class)
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
            filename_match = self.match_link(fits, match_class)
            if filename_match is None:
                # What to do if no match was found
                if match_class == 'bias':
                    print ('Warning: Bias frame not found. '
                           'Turning off bias subtraction for '+fits.filename)
                    options.extend(['-USEBIASIM', '0'])
                    continue
                elif match_class == 'dark':
                    print ('Warning: Dark frame not found. '
                           'Turning off dark subtraction for '+fits.filename)
                    options.extend(['-USEDARKIM', '0'])
                    continue
                elif match_class == 'lflat':
                    print ('Warning: LFlat frame not found. '
                           'Turning off LFlat division for '+fits.filename)
                    options.extend(['-USEFLATIM', '0'])
                    continue
                elif match_class == 'thput':
                    # Try to find a fake MFSKY made from a dome flat
                    filename_match = self.match_link(fits, 'thput_fflat')
                    if filename_match is None:
                        if fits.exposure < self.min_exposure_for_throughput:
                            # Try to find a suitable object frame instead
                            filename_match = self.match_link(
                                fits, 'thput_object')
                            # Really run out of options here
                            if filename_match is None:
                                # Still nothing
                                print ('Warning: Offsky (or substitute) frame '
                                       'not found. Turning off throughput '
                                       'calibration for '+fits.filename)
                                options.extend(['-THRUPUT', '0'])
                                continue
                        else:
                            # This is a long exposure, so use the sky lines
                            options[options.index('-TPMETH') + 1] = (
                                'SKYFLUX(MED)')
                elif match_class == 'tlmap':
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
                                print ('Warning: No good flat found for TLM. '
                                    'Using flap flat from different field '
                                    'for ' + fits.filename)
                        else:
                            print ('Warning: No dome flat found for TLM. '
                                'Using flap flat instead for ' + fits.filename)
                    else:
                        print ('Warning: No matching flat found for TLM. '
                            'Using flat from different field for ' + 
                            fits.filename)
                elif match_class == 'fflat':
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
                                    'No matching fflat found for ' + 
                                    fits.filename)
                            else:
                                print ('Warning: No good flat found for '
                                    'flat fielding. '
                                    'Using flap flat from different field '
                                    'for ' + fits.filename)
                        else:
                            print ('Warning: No dome flat found for flat '
                                'fielding. '
                                'Using flap flat instead for ' + fits.filename)
                    else:
                        print ('Warning: No matching flat found for flat '
                            'fielding. '
                            'Using flat from different field for ' + 
                            fits.filename)
                elif match_class == 'wavel':
                    # Try with looser criteria
                    filename_match = self.match_link(fits, 'wavel_loose')
                    if filename_match is None:
                        # Still nothing. Raise an exception
                        raise MatchException('No matching wavel found for ' +
                                             fits.filename)
                    else:
                        print ('Warning: No good arc found for wavelength '
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
                if match_class == 'tlmap_flap':
                    match_class = 'tlmap'
                elif match_class == 'thput_object':
                    match_class = 'thput'
                options.extend(['-'+match_class.upper()+'_FILENAME',
                                filename_match])
        return options        

    def run_2dfdr_auto(self, dirname):
        """Run 2dfdr in auto mode in the specified directory."""
        # First find a file in that directory to get the date and ccd
        for fits in self.file_list:
            if os.path.realpath(dirname) == os.path.realpath(fits.reduced_dir):
                date_ccd = fits.filename[:6]
                break
        else:
            # No known files in that directory, best not do anything
            return
        with self.visit_dir(dirname):
            # Make the 2dfdr AutoScript
            script = ['AutoScript:SetAutoUpdate ' + date_ccd,
                      'AutoScript:InvokeButton .auto.buttons.update',
                      'AutoScript:InvokeButton .auto.buttons.start']
            script_filename = '2dfdr_script.tcl'
            f_script = open(script_filename, 'w')
            f_script.write('\n'.join(script))
            f_script.close()
            # Run 2dfdr
            idx_file = self.idx_files[date_ccd[-1]]
            command = ['drcontrol',
                       idx_file,
                       '-AutoScript',
                       '-ScriptName',
                       script_filename]
            print 'Running 2dfdr in', dirname
            with open(os.devnull, 'w') as f:
                subprocess.call(command, stdout=f)
            # Clean up
            os.remove(script_filename)
        return

    def run_2dfdr_combine(self, file_iterable, output_path):
        """Use 2dfdr to combine the specified FITS files."""
        input_path_list = [fits.reduced_path for fits in file_iterable]
        print 'Combining files to create', output_path
        tdfdr.run_2dfdr_combine(
            input_path_list, output_path, unique_imp_scratch=True, 
            return_to=self.cwd, restore_to=self.imp_scratch, 
            scratch_dir=self.scratch_dir)
        return

    def files(self, ndf_class=None, date=None, plate_id=None,
              plate_id_short=None, field_no=None, field_id=None,
              ccd=None, exposure_str=None, do_not_use=None,
              min_exposure=None, max_exposure=None,
              reduced_dir=None, reduced=None, copy_reduced=None,
              tlm_created=None, flux_calibrated=None, telluric_corrected=None,
              spectrophotometric=None, name=None, lamp=None,
              central_wavelength=None):
        """Generator for FITS files that satisfy requirements."""
        for fits in self.file_list:
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
                (exposure_str is None or fits.exposure_str in exposure_str) and
                (do_not_use is None or fits.do_not_use == do_not_use) and
                (min_exposure is None or fits.exposure >= min_exposure) and
                (max_exposure is None or fits.exposure <= max_exposure) and
                (reduced_dir is None or
                 os.path.realpath(reduced_dir) ==
                 os.path.realpath(fits.reduced_dir)) and
                (reduced is None or
                 (reduced and os.path.exists(fits.reduced_path)) or
                 (not reduced and not os.path.exists(fits.reduced_path))) and
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
                 (not telluric_corrected and hasattr(fits, 'telluric_path') and
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

    def group_files_by(self, keys, **kwargs):
        """Return a dictionary of FITSFile objects grouped by the keys."""
        if isinstance(keys, str):
            keys = [keys]
        groups = defaultdict(list)
        for fits in self.files(**kwargs):
            combined_key = []
            for key in keys:
                combined_key.append(getattr(fits, key))
            combined_key = tuple(combined_key)
            groups[combined_key].append(fits)
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

    def other_arm(self, fits):
        """Return the FITSFile from the other arm of the spectrograph."""
        if fits.ccd == 'ccd_1':
            other_number = '2'
        elif fits.ccd == 'ccd_2':
            other_number = '1'
        else:
            raise ValueError('Unrecognised CCD: ' + fits.ccd)
        other_filename = fits.filename[:5] + other_number + fits.filename[6:]
        other_fits = self.fits_file(other_filename)
        return other_fits
        
    def cubed_path(self, name, arm, fits_list, field_id, gzipped=False,
                   exists=False, tag=None, **kwargs):
        """Return the path to the cubed file."""
        n_file = len(self.qc_for_cubing(fits_list, **kwargs))
        path = os.path.join(
            self.abs_root, 'cubed', name,
            name+'_'+arm+'_'+str(n_file)+'_'+field_id)
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

    @contextmanager
    def visit_dir(self, dir_path, cleanup_2dfdr=False):
        """Context manager to temporarily visit a directory."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        os.chdir(dir_path)
        try:
            yield
        finally:
            os.chdir(self.cwd)
            if cleanup_2dfdr:
                self.cleanup()

    def cleanup(self):
        """Clean up 2dfdr rubbish."""
        with open(os.devnull, 'w') as f:
            subprocess.call(['cleanup'], stdout=f)
        os.chdir(self.cwd)
        return

    def matchmaker(self, fits, match_class):
        """Return the file that should be used to help reduce the FITS file.

        match_class is one of the following:
        tlmap            -- Find a tramline map from the dome lamp
        tlmap_loose      -- As tlmap, but with less strict criteria
        tlmap_flap       -- As tlmap, but from the flap lamp
        tlmap_flap_loose -- As tlmap_flap, but with less strict criteria
        wavel            -- Find a reduced arc file
        wavel_loose      -- As wavel, but with less strict criteria
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
        # Define some functions for figures of merit
        time_difference = lambda fits, fits_test: (
            abs(fits_test.epoch - fits.epoch))
        recent_reduction = lambda fits, fits_test: (
            -1.0 * os.stat(fits_test.reduced_path).st_mtime)
        copy_recent_reduction = lambda fits, fits_test: (
            -1.0 * os.stat(self.copy_path(fits_test.reduced_path)).st_mtime)
        def time_difference_min_exposure(min_exposure):
            def retfunc(fits, fits_test):
                if fits_test.exposure <= min_exposure:
                    return np.inf
                else:
                    return time_difference(fits, fits_test)
            return retfunc
        # Determine what actually needs to be matched, depending on match_class
        if match_class.lower() == 'tlmap':
            # Find a tramline map, so need a dome fibre flat field
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
                do_not_use=False,
                ):
            test_fom = fom(fits, fits_test)
            if test_fom < best_fom:
                fits_match = fits_test
                best_fom = test_fom
        return fits_match

    def match_link(self, fits, match_class):
        """Match and make a link to a file, and return the filename."""
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
        elif match_class.lower() == 'thput_fflat':
            filename = self.copy_path(fits_match.reduced_filename)
            raw_filename = self.copy_path(fits_match.filename)
            raw_dir = fits_match.reduced_dir
        else:
            filename = fits_match.reduced_filename
            raw_filename = fits_match.filename
            raw_dir = fits_match.raw_dir
        # These are the cases where we do want to make a link
        require_link = [
            'tlmap', 'tlmap_loose', 'tlmap_flap', 'tlmap_flap_loose', 
            'fflat', 'fflat_loose', 'fflat_flap', 'fflat_flap_loose',
            'wavel', 'wavel_loose', 'thput', 'thput_fflat', 'thput_object']
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
            print "You must install the pysftp package to do that!"
        if username is None:
            if self.aat_username is None:
                username = raw_input('Enter AAT username: ')
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
            print 'Authentication failed! Check username and password.'
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
            idx_file = self.idx_files[fits_or_dirname.ccd]
        else:
            # A directory name has been provided
            dirname = fits_or_dirname
            if 'ccd_1' in dirname:
                idx_file = self.idx_files['ccd_1']
            elif 'ccd_2' in dirname:
                idx_file = self.idx_files['ccd_2']
            else:
                # Can't work out the CCD, let the GUI sort it out
                idx_file = None
        tdfdr.load_gui(dirname=dirname, idx_file=idx_file, 
                       unique_imp_scratch=True, return_to=self.cwd, 
                       restore_to=self.imp_scratch, 
                       scratch_dir=self.scratch_dir)
        return

    def remove_directory_locks(self):
        """Remove all 2dfdr locks from directories."""
        for dirname, subdirname_list, filename_list in os.walk(self.abs_root):
            if '2dfdrLockDir' in subdirname_list:
                os.rmdir(os.path.join(dirname, '2dfdrLockDir'))
        return

    def update_checks(self, key, file_iterable, value, force=False):
        """Set flags for whether the files have been manually checked."""
        for fits in file_iterable:
            fits.update_checks(key, value, force)
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
            complete_list.extend(self.list_checks('ever'))
            # Should ditch the duplicate checks, but will work anyway
            complete_list.extend(self.list_checks('recent'))
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
            for key in [key for key, value in items if value is False]:
                check_dict_key = []
                for group_by_key in CHECK_DATA[key]['group_by']:
                    check_dict_key.append(getattr(fits, group_by_key))
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
            print '{}: {}'.format(index, check_data['name'])
            if key[2] == 'ever':
                print 'Never been checked'
            else:
                print 'Not checked since last re-reduction'
            for group_by_key, group_by_value in zip(
                check_data['group_by'], key[1]):
                print '   {}: {}'.format(group_by_key, group_by_value)
            for fits in fits_list:
                print '      {}'.format(fits.filename)
        return

    def check_next_group(self, *args, **kwargs):
        """Perform required checks on the highest priority group."""
        self.check_group(0, *args, **kwargs)

    def check_group(self, index, *args, **kwargs):
        """Perform required checks on the specified group."""
        key, fits_list = self.list_checks(*args, **kwargs)[index]
        check_method = getattr(self, 'check_' + key[0].lower())
        check_method(fits_list)
        print 'Have you finished checking all the files? (y/n)'
        print 'If yes, the check will be removed from the list.'
        y_n = raw_input(' > ')
        finished = (y_n.lower()[0] == 'y')
        if finished:
            print 'Removing this test from the list.'
            for fits in fits_list:
                fits.update_checks(key[0], True)
        else:
            print 'Leaving this test in the list.'
        print 'If any files need to be disabled, use commands like:'
        print ">>> mngr.disable_files(['" + fits_list[0].filename + "'])"
        return

    def check_2dfdr(self, fits_list, message, filename_type='reduced_filename'):
        """Use 2dfdr to perform a check of some sort."""
        print 'Use 2dfdr to plot the following files.'
        print 'You may need to click on the triangles to see reduced files.'
        print 'If the files are not listed, use the plot commands in the 2dfdr menu.'
        for fits in fits_list:
            print '   ' + getattr(fits, filename_type)
        print message
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

    @contextmanager
    def patch_if_demo(self, target, new, requires_data=True):
        """If in demo mode, patch the target, otherwise do nothing."""
        if self.demo:
            if requires_data:
                func = new(self.demo_data_source)
            else:
                func = new
            with patch(target, func):
                yield
        else:
            yield




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
        self.set_exposure()
        self.set_epoch()
        self.set_lamp()
        self.set_central_wavelength()
        self.set_do_not_use()
        self.set_coords_flags()
        self.set_copy()
        self.hdulist.close()
        del self.hdulist

    def set_ndf_class(self):
        """Save the NDF_CLASS of an AAT fits file."""
        for hdu in self.hdulist:
            if ('EXTNAME' in hdu.header.keys() and
                (hdu.header['EXTNAME'] == 'STRUCT.MORE.NDF_CLASS' or
                 hdu.header['EXTNAME'] == 'NDF_CLASS')):
                # It has a class
                ndf_class = hdu.data['NAME'][0]
                # Change DFLAT to LFLAT
                if ndf_class == 'DFLAT':
                    hdulist_write = pf.open(self.source_path, 'update')
                    hdulist_write[hdu.header['EXTNAME']].data['NAME'][0] = (
                        'LFLAT')
                    hdulist_write.flush()
                    hdulist_write.close()
                    ndf_class = 'LFLAT'
                self.ndf_class = ndf_class
                break
        else:
            self.ndf_class = None

    def set_reduced_filename(self):
        """Set the filename for the reduced file."""
        self.reduced_filename = self.filename_root + 'red.fits'
        if self.ndf_class == 'MFFFF':
            self.tlm_filename = self.filename_root + 'tlm.fits'
        elif self.ndf_class == 'MFOBJECT':
            self.fluxcal_filename = self.filename_root + 'fcal.fits'
            self.telluric_filename = self.filename_root + 'sci.fits'
        return

    def set_date(self):
        """Save the observation date as a 6-digit string yymmdd."""
        try:
            file_orig = self.header['FILEORIG']
            finish = file_orig.rfind('/', 0, file_orig.rfind('/'))
            self.date = file_orig[finish-6:finish]
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
        finish = self.plate_id.find('_', self.plate_id.find('_')+1)
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
                for i in xrange(self.field_no):
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
                for i in xrange(self.field_no - 1):
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
            # This isn't guaranteed to work, but it's ok for our normal lamp
            _object = self.header['OBJECT']
            self.lamp = _object[_object.rfind(' ')+1:]
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
        old_header = pf.getheader(path)
        # Only update if necessary
        if (key not in old_header or
            old_header[key] != value or
            type(old_header[key]) != type(value) or
            (comment is not None and old_header.comments[key] != comment)):
            hdulist = pf.open(path, 'update',
                              do_not_scale_image_data=True)
            hdulist[0].header[key] = value_comment
            hdulist.close()
        return


def safe_for_multiprocessing(function):
    @wraps(function)
    def safe_function(*args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except KeyboardInterrupt:
            print "Handling KeyboardInterrupt in worker process"
            print "You many need to press Ctrl-C multiple times"
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
    print ('Deriving transfer function for ' + 
            os.path.basename(path_pair[0]) + ' and ' + 
            os.path.basename(path_pair[1]))
    try:
        fluxcal2.derive_transfer_function(
            path_pair, n_trim=n_trim, model_name=model_name, smooth=smooth)
    except ValueError:
        print ('Warning: No star found in dataframe, skipping ' + 
               os.path.basename(path_pair[0]))
        return
    good_psf = pf.getval(path_pair[0], 'GOODPSF',
                         'FLUX_CALIBRATION')
    if not good_psf:
        print ('Warning: Bad PSF fit in ' + os.path.basename(path_pair[0]) + 
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
        print ('Matching blue arm not found for ' + fits_2.filename +
               '; skipping this file.')
        return
    path_pair = (fits_1.fluxcal_path, fits_2.fluxcal_path)
    print ('Deriving telluric correction for ' + fits_1.filename +
           ' and ' + fits_2.filename)
    try:
        telluric.derive_transfer_function(
            path_pair, PS_spec_file=PS_spec_file, use_PS=use_PS, n_trim=n_trim,
            scale_PS_by_airmass=scale_PS_by_airmass, model_name=model_name)
    except ValueError as err:
        if err.message.startswith('No star identified in file:'):
            # No standard star found; probably a star field
            print err.message
            print 'Skipping telluric correction for file:', fits_2.filename
            return False
        else:
            # Some other, unexpected error. Re-raise it.
            raise err
    for fits in (fits_1, fits_2):
        print 'Telluric correcting file:', fits.filename
        if os.path.exists(fits.telluric_path):
            os.remove(fits.telluric_path)
        telluric.apply_correction(fits.fluxcal_path, 
                                  fits.telluric_path)
    return True

@safe_for_multiprocessing
def measure_offsets_group(group):
    """Measure offsets between a set of dithered observations."""
    field, fits_list, copy_to_other_arm, fits_list_other_arm = group
    print 'Measuring offsets for field ID: {}'.format(field[0])
    path_list = [best_path(fits) for fits in fits_list]
    print 'These are the files:'
    for path in path_list:
        print '  ', os.path.basename(path)
    if len(path_list) < 2:
        # Can't measure offsets for a single file
        print 'Only one file so no offsets to measure!'
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
    print 'Cubing field ID: {}, CCD: {}'.format(field[0], field[1])
    path_list = [best_path(fits) for fits in fits_list]
    print 'These are the files:'
    for path in path_list:
        print '  ', os.path.basename(path)
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
        path_list, suffix='_'+field[0], size_of_grid=50, write=True,
        nominal=True, root=root, overwrite=overwrite, do_dar_correct=True,
        objects=objects, clip=True, drop_factor=drop_factor)
    return

@safe_for_multiprocessing
def cube_object(inputs):
    """Cube a single object in a set of RSS files."""
    (field_id, ccd, path_list, name, cubed_root, drop_factor, tag,
     update_tol, size_of_grid, output_pix_size_arcsec, overwrite) = inputs
    print 'Cubing {} in field ID: {}, CCD: {}'.format(name, field_id, ccd)
    print '{} files available'.format(len(path_list))
    suffix = '_'+field_id
    if tag:
        suffix += '_'+tag
    dithered_cube_from_rss_wrapper(
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
    fits, idx_file, options, cwd, imp_scratch, scratch_dir = group
    try:
        tdfdr.run_2dfdr_single(
            fits, idx_file, options=options, return_to=cwd, 
            unique_imp_scratch=True, restore_to=imp_scratch, 
            scratch_dir=scratch_dir)
    except tdfdr.LockException:
        message = ('Postponing ' + fits.filename + 
                   ' while other process has directory lock.')
        print message
        return False
    return True

@safe_for_multiprocessing
def scale_cubes_field(group):
    """Scale a field to the correct magnitude."""
    star_path_pair, object_path_pair_list, star = group
    print 'Scaling field with star', star
    stellar_mags_cube_pair(star_path_pair, save=True)
    # Previously tried reading the catalogue once and passing it, but for
    # unknown reasons that was corrupting the data when run on aatmacb.
    found = assign_true_mag(star_path_pair, star, catalogue=None)
    if found:
        scale = scale_cube_pair_to_mag(star_path_pair)
        for object_path_pair in object_path_pair_list:
            scale_cube_pair(object_path_pair, scale)
    else:
        print 'No photometric data found for', star
    return

@safe_for_multiprocessing
def scale_frame_pair(path_pair):
    """Scale a pair of RSS frames to the correct magnitude."""
    print 'Scaling RSS files to give star correct magnitude:'
    print os.path.basename(path_pair[0]), os.path.basename(path_pair[1])
    stellar_mags_frame_pair(path_pair, save=True)
    star = pf.getval(path_pair[0], 'STDNAME', 'FLUX_CALIBRATION')
    # Previously tried reading the catalogue once and passing it, but for
    # unknown reasons that was corrupting the data when run on aatmacb.
    found = assign_true_mag(path_pair, star, catalogue=None,
                            hdu='FLUX_CALIBRATION')
    if found:
        scale_cube_pair_to_mag(path_pair, hdu='FLUX_CALIBRATION')
    else:
        print 'No photometric data found for', star
    return

@safe_for_multiprocessing
def bin_cubes_pair(path_pair):
    """Bin a pair of datacubes using each of the default schemes."""
    path_blue, path_red = path_pair
    print 'Binning datacubes:'
    print os.path.basename(path_blue), os.path.basename(path_red)
    binning_settings = (
        ('adaptive', {'mode': 'adaptive'}),
        ('annular', {'mode': 'prescriptive', 'sectors': 1}),
        ('sectors', {'mode': 'prescriptive'}))
    for name, kwargs in binning_settings:
        binning.bin_cube_pair(path_blue, path_red, name=name, **kwargs)
    return

@safe_for_multiprocessing
def gzip_wrapper(path):
    """Gzip a single file."""
    print 'Gzipping file: ' + path
    gzip(path)
    return

# @safe_for_multiprocessing
# def test_function(variable):
#     import time
#     print "starting", variable
#     start_time = time.time()
#     current_time = time.time()
#     while current_time < (start_time + 5):
#         print "waiting..."
#         time.sleep(1); current_time = time.time()
#     print "finishing", variable

def assign_true_mag(path_pair, name, catalogue=None, hdu=0):
    """Find the magnitudes in a catalogue and save them to the header."""
    if catalogue is None:
        catalogue = read_stellar_mags()
    if name in catalogue:
        mag_g = catalogue[name]['g']
        mag_r = catalogue[name]['r']
    else:
        return False
    for path in path_pair:
        hdulist = pf.open(path, 'update')
        hdulist[hdu].header['CATMAGG'] = (mag_g, 'g mag from catalogue')
        hdulist[hdu].header['CATMAGR'] = (mag_r, 'r mag from catalogue')
        hdulist.flush()
        hdulist.close()
    return True
    
def read_stellar_mags():
    """Read stellar magnitudes from the various catalogues available."""
    data_dict = {}
    for (path, catalogue_type, extinction) in stellar_mags_files():
        if catalogue_type == 'ATLAS':
            names = ('PHOT_ID', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                     'sigma', 'radius')
            formats = ('S20', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                       'f8', 'f8')
            skiprows = 2
            delimiter = None
            name_func = lambda d: d['PHOT_ID']
        elif catalogue_type == 'SDSS_cluster':
            names = ('obj_id', 'ra', 'dec', 'u', 'g', 'r', 'i', 'z',
                     'priority')
            formats = ('S30', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i')
            skiprows = 0
            delimiter = None
            name_func = lambda d: '888' + d['obj_id'][-9:]
        elif catalogue_type == 'SDSS_GAMA':
            names = ('name', 'obj_id', 'ra', 'dec', 'type', 'u', 'sig_u',
                     'g', 'sig_g', 'r', 'sig_r', 'i', 'sig_i', 'z', 'sig_z')
            formats = ('S20', 'S30', 'f8', 'f8', 'S10', 'f8', 'f8',
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

def fake_run_2dfdr_single(demo_data_source):
    """Return a function that pretends to reduce a data file."""
    source = os.path.join(demo_data_source, 'tdfdr')
    def inner(fits, *args, **kwargs):
        """Pretend to reduce a data file."""
        print 'Reducing file: ' + fits.filename
        suffixes = ('im', 'tlm', 'ex', 'red')
        filename_list = [
            fits.filename.replace('.', suff+'.') for suff in suffixes]
        for filename in filename_list:
            copy_demo_data(filename, source, fits.reduced_dir)
    return inner

def fake_derive_transfer_function(demo_data_source):
    """Return a function that pretends to derive a transfer function."""
    source = os.path.join(demo_data_source, 'transfer_function')
    def inner(path_pair, *args, **kwargs):
        """Pretend to derive a transfer function."""
        for path in path_pair:
            filename = os.path.basename(path)
            destination = os.path.dirname(path)
            copy_demo_data(filename, source, destination)
    return inner

def fake_find_dither(demo_data_source):
    """Return a function that pretends to find a dither pattern."""
    source = os.path.join(demo_data_source, 'offset')
    def inner(path_list, *args, **kwargs):
        """Pretend to find a dither pattern."""
        for path in path_list:
            filename = os.path.basename(path)
            destination = os.path.dirname(path)
            copy_demo_data(filename, source, destination)
    return inner

def fake_dithered_cubes_from_rss_list(demo_data_source):
    """Return a function that pretends to make datacubes."""
    source = os.path.join(demo_data_source, 'cubes')
    def inner(path_list, suffix='', root='', overwrite=True, *args, **kwargs):
        """Pretend to make datacubes."""
        object_names = get_object_names(path_list[0])
        for name in object_names:
            ifu_list = [IFU(path, name, flag_name=True) for path in path_list]
            directory = os.path.join(root, name)
            try:
                os.makedirs(directory)
            except OSError:
                print "Directory Exists", directory
                print "Writing files to the existing directory"
            else:
                print "Making directory", directory
            # Filename to write to
            arm = ifu_list[0].spectrograph_arm            
            outfile_name = (
                str(name)+'_'+str(arm)+'_'+str(len(path_list))+suffix+'.fits')
            outfile_name_full = os.path.join(directory, outfile_name)
            # Check if the filename already exists
            if os.path.exists(outfile_name_full):
                if overwrite:
                    os.remove(outfile_name_full)
                else:
                    print 'Output file already exists:'
                    print outfile_name_full
                    print 'Skipping this object'
                    continue
            copy_demo_data(outfile_name, source, directory)
    return inner

def fake_dithered_cube_from_rss_wrapper(demo_data_source):
    """Return a function that pretends to make cubes for 1 object."""
    def inner(path_list, name, suffix='', root='', overwrite=True,
              *args, **kwargs):
        """Pretend to make datacubes."""
        ifu_list = [IFU(path, name, flag_name=True) for path in path_list]
        directory = os.path.join(root, name)
        try:
            os.makedirs(directory)
        except OSError:
            print "Directory Exists", directory
            print "Writing files to the existing directory"
        else:
            print "Making directory", directory
        # Filename to write to
        arm = ifu_list[0].spectrograph_arm            
        outfile_name = (
            str(name)+'_'+str(arm)+'_'+str(len(path_list))+suffix+'.fits')
        outfile_name_full = os.path.join(directory, outfile_name)
        # Check if the filename already exists
        if os.path.exists(outfile_name_full):
            if overwrite:
                os.remove(outfile_name_full)
            else:
                print 'Output file already exists:'
                print outfile_name_full
                print 'Skipping this object'
                return False
        copy_demo_data(outfile_name, source, directory)
        return True
    return inner    

def fake_stellar_mags_cube_pair(demo_data_source):
    """Return a function that pretends to measure and scale stellar mags."""
    source = os.path.join(demo_data_source, 'scaled_cubes')
    def inner(star_path_pair, *args, **kwargs):
        """Pretend to measure and scale stellar magnitudes."""
        for path in star_path_pair:
            filename = os.path.basename(path)
            destination = os.path.dirname(path)
            copy_demo_data(filename, source, destination)
    return inner

def fake_gzip(demo_data_source):
    """Return a function that pretends to gzip files."""
    source = os.path.join(demo_data_source, 'gzipped_cubes')
    def inner(path, *args, **kwargs):
        """Pretend to gzip a file."""
        filename = os.path.basename(path) + '.gz'
        destination = os.path.dirname(path)
        copy_demo_data(filename, source, destination)
        os.remove(path)
    return inner

def copy_demo_data(filename, source, destination, skip_missing=True):
    """Find and copy demo data from source dir to destination dir."""
    for root, dirs, files in os.walk(source):
        if filename in files:
            path = os.path.join(root, filename)
            if os.path.isfile(path) and not os.path.islink(path):
                dest_path = os.path.join(destination, filename)
                shutil.copy2(path, dest_path)
                sleep(0.5)
                break
    else:
        # No file found. Raise an error if skip_missing set to False
        if not skip_missing:
            raise IOError('File not found: ' + filename)

class MatchException(Exception):
    """Exception raised when no matching calibrator is found."""
    pass
