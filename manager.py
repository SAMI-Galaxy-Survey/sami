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
auto-reduce everything in a specified directory. This is used for
reducing bias, dark and long-slit flat frames, which do not need
individual calibrators specified.

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
from contextlib import contextmanager
from collections import defaultdict, deque

import astropy.coordinates as coord
from astropy import units
import astropy.io.fits as pf
import numpy as np
from .utils.other import find_fibre_table
from .general.cubing import dithered_cubes_from_rss_list
from .general.align_micron import find_dither
from .dr import fluxcal2, telluric


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

    Checking outputs
    ================

    As the reductions are done, the manager keeps an internal list of reduced
    files that need to be plotted to check that the outputs are ok. These are
    saved on a directory-by-directory basis. To print the contents of the
    list:

    >>> mngr.print_check_list()

    It will print the directories that need to be visited, and the files to
    check within each directory. The manager can't plot the files directly but
    can open up the 2dfdr GUI in the required directory, making it easy to use
    the 2dfdr plotting tools to do the checks. To open 2dfdr in the next
    directory in the list:

    >>> mngr.plot_next_dir()

    No more commands can be entered until you close 2dfdr. When you do so, you
    will be asked whether the directory can be removed from the list; enter 'y'
    if you have checked all the files, 'n' otherwise. You can also load a
    particular directory by specifying [part of] its path or its index in the
    list:

    >>> mngr.plot_dir('Y13SAR1_P002_09T004/main/ccd_1')
    >>> mngr.plot_dir_by_index(3)

    Note index numbers start at 0, and the directories are in the order listed
    when mngr.print_check_list() is called.

    If you want to remove a directory from the list without loading 2dfdr:

    >>> mngr.remove_dir_from_checklist('Y13SAR1_P002_09T004/main/ccd_1')
    >>> mngr.remove_dir_from_checklist_by_index(3)

    Only do this with good reason! Checking the outputs is a crucial part of
    data reduction.

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

    Other functions
    ===============

    The other functions defined probably aren't useful to you.
    """

    def __init__(self, root, copy_files=False, move_files=False, fast=False,
                 gratlpmm=GRATLPMM):
        if fast:
            self.speed = 'fast'
        else:
            self.speed = 'slow'
        self.idx_files = IDX_FILES[self.speed]
        self.gratlpmm = gratlpmm
        self.root = root
        self.abs_root = os.path.abspath(root)
        # Match objects within 1'
        self.matching_radius = \
            coord.AngularSeparation(0.0, 0.0, 0.0, 1.0, units.arcmin)
        self.file_list = []
        self.extra_list = []
        self.dark_exposure_str_list = []
        self.dark_exposure_list = []
        self.check_list = deque()
        self.inspect_root(copy_files, move_files)
        self.cwd = os.getcwd()

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
        if (fits.coords.separation(fits.cfg_coords) < self.matching_radius):
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
            if (fits.coords.separation(extra['coords']) < self.matching_radius):
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
                 'coords':fits.coords,
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
                    self.import_file(dirname, filename,
                                     trust_header=trust_header,
                                     copy_files=True, move_files=False)
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

    def reduce_calibrator(self, calibrator_type, overwrite=False):
        """Reduce all biases, darks of lflats."""
        self.check_calibrator_type(calibrator_type)
        if overwrite:
            # Delete all the old reduced files
            for fits in self.files(ndf_class=calibrator_type.upper(),
                                   reduced=True):
                os.remove(fits.reduced_path)
        for dir_path in self.reduced_dirs(calibrator_type.lower()):
            self.run_2dfdr_auto(dir_path)
        return

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
                self.check_list.append((dirname, [filename]))
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

    def reduce_bias(self, overwrite=False):
        """Reduce all bias frames."""
        self.reduce_calibrator('bias', overwrite=overwrite)
        return

    def combine_bias(self, overwrite=False):
        """Produce and link necessary BIAScombined.fits files."""        
        self.combine_calibrator('bias', overwrite=overwrite)
        return

    def link_bias(self, overwrite=False):
        """Make necessary symbolic links for BIAScombined.fits files."""
        self.link_calibrator('bias', overwrite=overwrite)
        return

    def reduce_dark(self, overwrite=False):
        """Reduce all dark frames."""
        self.reduce_calibrator('dark', overwrite=overwrite)
        return
        
    def combine_dark(self, overwrite=False):
        """Produce and link necessary DARKcombinedXXXX.fits files."""
        self.combine_calibrator('dark', overwrite=overwrite)
        return

    def link_dark(self, overwrite=False):
        """Make necessary symbolic links for DARKcombinedXXXX.fits files."""
        self.link_calibrator('dark', overwrite=overwrite)
        return

    def reduce_lflat(self, overwrite=False):
        """Reduce all lflat frames."""
        self.reduce_calibrator('lflat', overwrite=overwrite)
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
        self.reduce_file_iterable(file_iterable, overwrite=overwrite, 
                                  tlm=True, leave_reduced=leave_reduced)
        return

    def reduce_arc(self, overwrite=False, **kwargs):
        """Reduce all arc frames matching given criteria."""
        file_iterable = self.files(ndf_class='MFARC', do_not_use=False,
                                   **kwargs)
        self.reduce_file_iterable(file_iterable, overwrite=overwrite)
        return

    def reduce_fflat(self, overwrite=False, **kwargs):
        """Reduce all fibre flat frames matching given criteria."""
        file_iterable = self.files(ndf_class='MFFFF', do_not_use=False,
                                   **kwargs)
        self.reduce_file_iterable(file_iterable, overwrite=overwrite)
        return

    def reduce_sky(self, overwrite=False, **kwargs):
        """Reduce all offset sky frames matching given criteria."""
        file_iterable = self.files(ndf_class='MFSKY', do_not_use=False,
                                   **kwargs)
        self.reduce_file_iterable(file_iterable, overwrite=overwrite)
        return

    def reduce_object(self, overwrite=False, **kwargs):
        """Reduce all object frames matching given criteria."""
        # Reduce them in reverse order of exposure time, to ensure the best
        # possible throughput measurements are always available
        key = lambda fits: fits.exposure
        file_iterable = sorted(self.files(ndf_class='MFOBJECT',
                                          do_not_use=False, **kwargs),
                               key=key, reverse=True)
        self.reduce_file_iterable(file_iterable, overwrite=overwrite)
        return

    def reduce_file_iterable(self, file_iterable, overwrite=False, tlm=False,
                             leave_reduced=True):
        """Reduce all files in the iterable."""
        extra_check_dict = defaultdict(list)
        for fits in file_iterable:
            reduced = self.reduce_file(fits, overwrite=overwrite, tlm=tlm,
                                       leave_reduced=leave_reduced)
            if reduced:
                if tlm:
                    check_filename = fits.tlm_filename
                else:
                    check_filename = fits.reduced_filename
                extra_check_dict[fits.reduced_dir].append(check_filename)
        self.check_list.extend(extra_check_dict.items())
        return

    def derive_transfer_function(self, overwrite=False, **kwargs):
        """Derive flux calibration transfer functions and save them."""
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
            print ('Deriving transfer function for ' + fits.filename + 
                   ' and ' + fits_2.filename)
            try:
                fluxcal2.derive_transfer_function(path_pair)
            except ValueError:
                print ('Warning: No star found in dataframe, skipping ' + 
                       fits.filename)
                continue
            good_psf = pf.getval(fits.reduced_path, 'GOODPSF',
                                 'FLUX_CALIBRATION')
            if not good_psf:
                print ('Warning: Bad PSF fit in ' + fits.filename + 
                       '; will skip this one in combining.')
        return

    def combine_transfer_function(self, overwrite=False, **kwargs):
        """Combine and save transfer functions from multiple files."""
        # First sort the spectrophotometric files into date/field/CCD/name 
        # groups. Wouldn't need name except that currently different stars
        # are on different units; remove the name matching when that's
        # sorted.
        date_field_ccd_name_dict = defaultdict(list)
        for fits in self.files(ndf_class='MFOBJECT', do_not_use=False,
                               spectrophotometric=True, **kwargs):
            path = fits.reduced_path
            key = fits.date + fits.field_id + fits.ccd + fits.name
            date_field_ccd_name_dict[key].append(path)
        # Now combine the files within each group
        for path_list in date_field_ccd_name_dict.values():
            path_out = os.path.join(os.path.dirname(path_list[0]),
                                    'TRANSFERcombined.fits')
            if overwrite or not os.path.exists(path_out):
                print 'Combining files to create', path_out
                fluxcal2.combine_transfer_functions(path_list, path_out)
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

    def telluric_correct(self, overwrite=False, **kwargs):
        """Apply telluric correction to object frames."""
        for fits_2 in self.files(ndf_class='MFOBJECT', do_not_use=False,
                                 spectrophotometric=False, ccd='ccd_2', 
                                 **kwargs):
            if os.path.exists(fits_2.telluric_path) and not overwrite:
                # Already been done; skip to the next file
                continue
            fits_1 = self.other_arm(fits_2)
            if fits_1 is None or not os.path.exists(fits_1.fluxcal_path):
                print ('Matching blue arm not found for ' + fits_2.filename +
                       '; skipping this file.')
            path_pair = (fits_1.fluxcal_path, fits_2.fluxcal_path)
            if fits_1.epoch < 2013.0:
                # SAMI v1 had awful throughput at blue end of blue, need to
                # trim that data
                n_trim = 3
            else:
                n_trim = 0
            print ('Deriving telluric correction for ' + fits_1.filename +
                   ' and ' + fits_2.filename)
            telluric.correction_linear_fit(path_pair, n_trim=n_trim)
            print 'Telluric correcting file:', fits_2.filename
            if os.path.exists(fits_2.telluric_path):
                os.remove(fits_2.telluric_path)
            telluric.apply_correction(fits_2.fluxcal_path, 
                                      fits_2.telluric_path)
        return

    def cube(self, overwrite=False, **kwargs):
        """Make datacubes from the given RSS files."""
        # overwrite not yet implemented
        target_dir = os.path.join(self.abs_root, 'cubed')
        if 'min_exposure' in kwargs:
            min_exposure = kwargs['min_exposure']
            del kwargs['min_exposure']
        else:
            min_exposure = 599.0
        if 'name' in kwargs:
            name = kwargs['name']
            del kwargs['name']
        else:
            name = 'main'
        groups = self.group_files_by(
            ['field_id', 'ccd'], ndf_class='MFOBJECT', do_not_use=False,
            reduced=True, min_exposure=min_exposure, name=name, **kwargs)
        with self.visit_dir(target_dir):
            for field, fits_list in groups.items():
                print 'Cubing field ID: {}, CCD: {}'.format(field[0], field[1])
                path_list = []
                for fits in fits_list:
                    if os.path.exists(fits.telluric_path):
                        path = fits.telluric_path
                    elif os.path.exists(fits.fluxcal_path):
                        path = fits.fluxcal_path
                    else:
                        path = fits.reduced_path
                    path_list.append(path)
                # First calculate the offsets
                find_dither(path_list, path_list[0], centroid=True, 
                            remove_files=True)
                # Now do the actual cubing
                dithered_cubes_from_rss_list(path_list, suffix='_'+field[0], 
                                             write=True)
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
        self.cube(overwrite, **kwargs)
        return

    def reduce_file(self, fits, overwrite=False, tlm=False,
                    leave_reduced=False):
        """Select appropriate options and reduce the given file.

        For MFFFF files, if tlm is True then a tramline map is produced; if it
        is false then a full reduction is done. If tlm is True and leave_reduced
        is false, then any reduced MFFFF produced as a side-effect will be
        removed.

        Returns True if the file was reduced; False otherwise."""
        if fits.ndf_class == 'MFFFF' and tlm:
            target = fits.tlm_path
        else:
            target = fits.reduced_path
        if os.path.exists(target) and not overwrite:
            # File to be created already exists, abandon this.
            return False
        options = []
        # For now, setting all files to use GAUSS extraction
        options.extend(['-EXTR_OPERATION', 'GAUSS'])
        if fits.ndf_class == 'MFFFF' and tlm:
            files_to_match = ['bias', 'dark', 'lflat']
        elif fits.ndf_class == 'MFARC':
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap']
        elif fits.ndf_class == 'MFFFF' and not tlm:
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel']
        elif fits.ndf_class == 'MFSKY':
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                              'fflat']
        elif fits.ndf_class == 'MFOBJECT':
            if fits.exposure <= 899.0:
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                  'fflat', 'thput']
                options.extend(['-TPMETH', 'OFFSKY'])
            else:
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                  'fflat']
        else:
            raise ValueError('Unrecognised NDF_CLASS: '+fits.ndf_class)
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
                    # Try to find a suitable object frame instead
                    filename_match = self.match_link(fits, 'thput_object')
                    if filename_match is None:
                        # Still nothing
                        print ('Warning: Offsky (or substitute) frame not '
                               'found. Turning off throughput calibration '
                               'for '+fits.filename)
                        options.extend(['-THRUPUT', '0'])
                        continue
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
                        # Try using a dome flat instead
                        filename_match = self.match_link(fits, 'fflat_dome')
                        if filename_match is None:
                            # Try with looser criteria
                            filename_match = self.match_link(
                                fits, 'fflat_dome_loose')
                            if filename_match is None:
                                # Still nothing. Raise an exception
                                raise MatchException(
                                    'No matching fflat found for ' + 
                                    fits.filename)
                            else:
                                print ('Warning: No good flat found for '
                                    'flat fielding. '
                                    'Using dome flat from different field '
                                    'for ' + fits.filename)
                        else:
                            print ('Warning: No flap flat found for flat '
                                'fielding. '
                                'Using dome flat instead for ' + fits.filename)
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
                options.extend(['-'+match_class.upper()+'_FILENAME',
                                filename_match])
        # All options have been set, so run 2dfdr
        self.run_2dfdr_single(fits, options)
        if (fits.ndf_class == 'MFFFF' and tlm and not leave_reduced and
            os.path.exists(fits.reduced_path)):
            os.remove(fits.reduced_path)
        return True

    def extra_options(self, fits):
        """Return a list of extra reduction options suitable for the file."""
        options = []
        if fits.ndf_class == 'MFOBJECT' and fits.exposure <= 899.0:
            # Use offsky throughput values for short exposures
            options.extend(['-TPMETH', 'OFFSKY'])
        return options

    def run_2dfdr_single(self, fits, options=None):
        """Run 2dfdr on a single FITS file."""
        print 'Reducing file:', fits.filename
        if fits.ndf_class == 'BIAS':
            task = 'reduce_bias'
        elif fits.ndf_class == 'DARK':
            task = 'reduce_dark'
        elif fits.ndf_class == 'LFLAT':
            task = 'reduce_lflat'
        elif fits.ndf_class == 'MFFFF':
            task = 'reduce_fflat'
        elif fits.ndf_class == 'MFARC':
            task = 'reduce_arc'
        elif fits.ndf_class == 'MFSKY':
            task = 'reduce_sky'
        elif fits.ndf_class == 'MFOBJECT':
            task = 'reduce_object'
        else:
            raise ValueError('Unrecognised NDF_CLASS')
        command = ['drcontrol', task, fits.filename,
                   '-idxfile', self.idx_files[fits.ccd]]
        if options is not None:
            command.extend(options)
        with self.visit_dir(fits.reduced_dir):
            with open(os.devnull, 'w') as f:
                subprocess.call(command, stdout=f)
        self.cleanup()
        return

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
        output_dir, output_filename = os.path.split(output_path)
        # Need to extend the default timeout value; set to 5 hours here
        timeout = '300'
        # Write the 2dfdr AutoScript
        script = []
        for fits in file_iterable:
            script.append('lappend glist ' +
                          os.path.relpath(fits.reduced_path, output_dir))
        if len(script) < 2:
            raise ValueError('Need at least 2 files to combine!')
        script.extend(['proc Quit {status} {',
                       '    global Auto',
                       '    set Auto(state) 0',
                       '}',
                       'set task DREXEC1',
                       'global Auto',
                       'set Auto(state) 1',
                       ('ExecCombine $task $glist ' + output_filename +
                        ' -success Quit')])
        script_filename = '2dfdr_script.tcl'
        with self.visit_dir(output_dir):
            # Print the script to file
            f_script = open(script_filename, 'w')
            f_script.write('\n'.join(script))
            f_script.close()
            # Run 2dfdr
            command = ['drcontrol',
                       '-AutoScript',
                       '-ScriptName',
                       script_filename,
                       '-Timeout',
                       timeout]
            print 'Combining files to create', output_path
            with open(os.devnull, 'w') as f:
                subprocess.call(command, stdout=f)
            # Clean up
            os.remove(script_filename)
        return

    def files(self, ndf_class=None, date=None, plate_id=None,
              plate_id_short=None, field_no=None, field_id=None,
              ccd=None, exposure_str=None, do_not_use=None,
              min_exposure=None, max_exposure=None,
              reduced_dir=None, reduced=None, tlm_created=None,
              flux_calibrated=None, telluric_corrected=None,
              spectrophotometric=None, name=None, lamp=None):
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
                (lamp is None or fits.lamp == lamp)):
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
        fflat            -- Find a reduced fibre flat field from the flap lamp
        fflat_loose      -- As fflat, but with less strict criteria
        fflat_dome       -- As fflat, but from the dome lamp
        fflat_dome_loose -- As fflat_dome, but with less strict criteria
        thput            -- Find a reduced offset sky (twilight) file
        thput_object     -- As thput, but find a suitable object frame
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
        tlm_created = None
        flux_calibrated = None
        telluric_corrected = None
        spectrophotometric = None
        lamp = None
        # Define some functions for figures of merit
        time_difference = lambda fits, fits_test: (
            abs(fits_test.epoch - fits.epoch))
        recent_reduction = lambda fits, fits_test: (
            -1.0 * os.stat(fits_test.reduced_path).st_mtime)
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
            # Find a reduced fibre flat field from the flap lamp
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'fflat_loose':
            # Find a reduced fibre flat field with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            reduced = True
            lamp = 'Flap'
            fom = time_difference
        elif match_class.lower() == 'fflat_dome':
            # Find a reduced dome fibre flat field
            ndf_class = 'MFFFF'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            lamp = 'Dome'
            fom = time_difference
        elif match_class.lower() == 'fflat_dome_loose':
            # Fibre flat field from dome lamp with looser criteria
            ndf_class = 'MFFFF'
            ccd = fits.ccd
            reduced = True
            lamp = 'Dome'
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
        elif match_class.lower() == 'thput_object':
            # Find a reduced object field to take the throughput from
            ndf_class = 'MFOBJECT'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            fom = time_difference_min_exposure(899.0)
        elif match_class.lower() == 'fcal':
            # Find a spectrophotometric standard star
            ndf_class = 'MFOBJECT'
            date = fits.date
            plate_id = fits.plate_id
            field_id = fits.field_id
            ccd = fits.ccd
            reduced = True
            spectrophotometric = True
            fom = time_difference
        elif match_class.lower() == 'fcal_loose':
            # Spectrophotometric with less strict criteria
            ndf_class = 'MFOBJECT'
            ccd = fits.ccd
            reduced = True
            spectrophotometric = True
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
            for exposure_str in self.dark_exposure_strs(ccd=fits.ccd):
                test_fom = abs(float(exposure_str) - fits.exposure)
                if test_fom < best_fom:
                    exposure_str_match = exposure_str
                    best_fom = test_fom
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
        else:
            filename = fits_match.reduced_filename
        # These are the cases where we do want to make a link
        require_link = [
            'tlmap', 'tlmap_loose', 'tlmap_flap', 'tlmap_flap_loose', 
            'fflat', 'fflat_loose', 'fflat_dome', 'fflat_dome_loose',
            'wavel', 'wavel_loose', 'thput', 'thput_object']
        if match_class.lower() in require_link:
            link_path = os.path.join(fits.reduced_dir, filename)
            source_path = os.path.join(fits_match.reduced_dir, filename)
            raw_link_path = os.path.join(fits.reduced_dir, fits_match.filename)
            raw_source_path = os.path.join(fits_match.raw_dir,
                                           fits_match.filename)
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

    def print_check_list(self):
        """Print the reduced files that need to be checked."""
        for reduced_dir, filename_list in self.check_list:
            print reduced_dir
            for filename in filename_list:
                print ' - ' + filename
        return

    def plot_dir(self, dirname):
        """Load 2dfdr to plot a set of files from the checklist.

        The first match to dirname - which may be incomplete - will be selected
        from the checklist, and 2dfdr will be loaded in that directory."""
        for index, check_tuple in enumerate(self.check_list):
            if dirname in check_tuple[0]:
                self.plot_dir_by_index(index)
                break
        else:
            print 'No match found in checklist.'
        return

    def plot_next_dir(self):
        """Load 2dfdr to plot the next set of reduced files to check."""
        if len(self.check_list) == 0:
            print 'No directories are in the checklist.'
            return
        self.plot_dir_by_index(0)
        return

    def plot_dir_by_index(self, index):
        """Load 2dfdr to plot a specified set of files from the checklist."""
        reduced_dir, filename_list = self.check_list[index]
        print 'Loading 2dfdr in directory:'
        print reduced_dir
        print 'Use 2dfdr to plot and check the following files.'
        print '(Click on the triangles to see reduced files)'
        for filename in filename_list:
            print ' - ' + filename
        print 'Make a note of any that need to be disabled or re-run.'
        try:
            idx_file = self.idx_files[os.path.basename(reduced_dir)]
        except KeyError:
            try:
                idx_file = self.idx_files[filename_list[0][5]]
            except KeyError:
                # Can't work out which idx file to use; just grab the first one
                idx_file = self.idx_files.values()[0]
        command = ['drcontrol',
                   idx_file]
        with self.visit_dir(reduced_dir):
            with open(os.devnull, 'w') as f:
                subprocess.call(command, stdout=f)
        yn = raw_input('Have you finished checking all the files? '
                       'The directory will be removed from the checklist. '
                       '(y/n)\n > ')
        remove = (yn.lower()[0] == 'y')
        if remove:
            print 'Removing this directory from the checklist.'
            del self.check_list[index]
        else:
            print 'Leaving this directory in the checklist.'
        print ('If any files need to be disabled, you can do so using commands'
               ' like:')
        print ">>> mngr.disable_files(['" + filename_list[0] + "'])"
        return

    def remove_dir_from_checklist(self, dirname):
        """Remove the first instance of a directory from the checklist."""
        for index, check_tuple in enumerate(self.check_list):
            if dirname in check_tuple[0]:
                del self.check_list[index]
                break
        else:
            print 'No match found in checklist'
        return

    def remove_dir_from_checklist_by_index(self, index):
        """Remove the specified directory from the plotting checklist."""
        del self.check_list[index]
        return

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
        self.set_do_not_use()
        self.set_coords_flags()
        self.hdulist.close()
        del self.hdulist

    def set_ndf_class(self):
        """Save the NDF_CLASS of an AAT fits file."""
        for hdu in self.hdulist:
            if ('EXTNAME' in hdu.header.keys() and
                (hdu.header['EXTNAME'] == 'STRUCT.MORE.NDF_CLASS' or
                 hdu.header['EXTNAME'] == 'NDF_CLASS')):
                # It has a class
                self.ndf_class = hdu.data['NAME'][0]
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
            file_orig = self.hdulist[0].header['FILEORIG']
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
            self.plate_id = self.hdulist[0].header['PLATEID']
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
                if (self.cfg_coords.separation(
                    coord.ICRSCoordinates(pilot_field['coords'])).arcsecs < 1.0
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
            self.cfg_coords = \
                coord.ICRSCoordinates(ra=header['CFGCENRA'],
                                      dec=header['CFGCENDE'],
                                      unit=(units.radian, units.radian))
            if self.ndf_class == 'MFOBJECT':
                self.coords = \
                    coord.ICRSCoordinates(ra=header['CENRA'],
                                          dec=header['CENDEC'],
                                          unit=(units.radian, units.radian))
            else:
                self.coords = None
        else:
            self.cfg_coords = None
        return
    
    def set_ccd(self):
        """Set the CCD name."""
        if self.ndf_class:
            spect_id = self.hdulist[0].header['SPECTID']
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
            self.exposure = self.hdulist[0].header['EXPOSED']
            self.exposure_str = '{:d}'.format(int(np.round(self.exposure)))
        else:
            self.exposure = None
            self.exposure_str = None
        return

    def set_epoch(self):
        """Set the observation epoch."""
        if self.ndf_class:
            self.epoch = self.hdulist[0].header['EPOCH']
        else:
            self.epoch = None
        return

    def set_lamp(self):
        """Set which lamp was on, if any."""
        if self.ndf_class == 'MFARC':
            # This isn't guaranteed to work, but it's ok for our normal lamp
            _object = self.hdulist[0].header['OBJECT']
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

    def set_do_not_use(self):
        """Set whether or not to use this file."""
        try:
            self.do_not_use = self.hdulist[0].header['DONOTUSE']
        except KeyError:
            # By default, don't use fast readout files
            self.do_not_use = (self.hdulist[0].header['SPEED'] != 'NORMAL')
        return

    def set_coords_flags(self):
        """Set whether coordinate corrections have been done."""
        try:
            self.coord_rot = self.hdulist[0].header['COORDROT']
        except KeyError:
            self.coord_rot = None
        try:
            self.coord_rev = self.hdulist[0].header['COORDREV']
        except KeyError:
            self.coord_rev = None
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


class MatchException(Exception):
    """Exception raised when no matching calibrator is found."""
    pass
