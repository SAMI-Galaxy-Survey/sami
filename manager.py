"""Code for organising and reducing SAMI data."""

import shutil
import os
import re
import subprocess
from contextlib import contextmanager

import astropy.coordinates as coord
from astropy import units
import astropy.io.fits as pf
import numpy as np
from sami.utils.other import find_fibre_table


IDX_FILES = {'1': 'sami580V_v1_2.idx',
             '2': 'sami1000R_v1_2.idx',
             'ccd_1': 'sami580V_v1_2.idx',
             'ccd_2': 'sami1000R_v1_2.idx'}


class Manager:
    """Object for organising and reducing SAMI data.

    Initial setup
    =============

    You start a new manager by creating an object, and telling it where
    to put its data, e.g.:

    >>> import sami
    >>> mngr = sami.manager.Manager('data_directory')

    In this case, 'data_directory' should be non-existent or empty. At this
    point the manager is not aware of any actual data - skip to "Importing
    data" and carry on from there.

    Continuing a previous session
    =============================

    If you quit python and want to get back to where you were, just restart
    the manager on the same directory (after importing sami):

    >>> mngr = sami.manager.Manager('data_directory')

    It will search through the subdirectories and restore its previous
    state.

    Importing data
    ==============

    After creating the manager, you can import data into it:

    >>> mngr.import_dir('path/to/data')

    It will copy the data into the data_directory you defined earlier,
    putting it into a neat directory structure. You will typically need to
    import all of your data before you start to reduce anything, to ensure
    you have all the bias, dark and lflat frames.

    When importing data, the manager will do its best to work out what the
    telescope was pointing at in each frame. Sometimes it wont be able to
    and will ask you for the object name to go with a particular file.
    Depending on the file, you should give an actual object name - e.g.
    HR7950 or NGC2701 - or a more general description - e.g. SNAFU or
    blank_sky.

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
        reduced_dir     ('data_directory/reduced/130305/'
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

    Disabling files
    ===============

    If there are any problems with some files, either because an issue is
    noted in the log or because they wont reduce properly, you can disable
    them, preventing them from being used in any later reductions:

    >>> mngr.disable_files(['06mar10003', '06mar20003', '06mar10047'])

    If you only have one file you want to disable, you still need the
    square brackets. You can disable lots of files at a time using the
    files generator:

    >>> mngr.disable_files(mngr.files(
                date='130306', field_id='Y13SAR1_P002_09T004'))

    This allows the same keywords as described earlier, as well as:

        ndf_class           'MFFFF'
        reduced             False
        tlm_created         False
        flux_calibrated     False
        telluric_corrected  False

    For example, specifying these options as given would disable all
    fibre flat fields that had not yet been reduced and had not yet had
    tramline maps created.

    To re-enable files:

    >>> mngr.enable_files(['06mar10003', '06mar20003', '06mar10047'])

    This function follows exactly the same syntax as disable_files.

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

    def __init__(self, root, copy_files=False, move_files=False):
        self.idx_files = IDX_FILES
        self.root = root
        # Match objects within 1'
        self.matching_radius = \
            coord.AngularSeparation(0.0, 0.0, 0.0, 1.0, units.arcmin)
        self.file_list = []
        self.extra_list = []
        self.dark_exposure_str_list = []
        self.dark_exposure_list = []
        self.inspect_root(copy_files, move_files)
        self.cwd = os.getcwd()

    def inspect_root(self, copy_files, move_files):
        """Add details of existing files to internal lists."""
        reduced_path = os.path.join(self.root, 'reduced')
        for dirname, subdirname_list, filename_list in os.walk(reduced_path):
            for filename in filename_list:
                if self.file_filter(filename):
                    self.import_file(dirname, filename,
                                     trust_header=True, copy_files=copy_files,
                                     move_files=move_files)
        raw_path = os.path.join(self.root, 'raw')
        for dirname, subdirname_list, filename_list in os.walk(raw_path):
            for filename in filename_list:
                if self.file_filter_raw(filename):
                    try:
                        self.import_file(
                            dirname, filename,
                            trust_header=True, copy_files=copy_files,
                            move_files=move_files)
                    except Exception as e:
                        print 'Error importing file:', \
                            os.path.join(dirname, filename)
        return

    def file_filter(self, filename):
        """Return True if the file should be added."""
        # Match filenames of the form 01jan10001.fits
        return re.match(r'[0-3][0-9]'
                        '(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
                        '[1-2][0-9]{4}\.(fit|fits|FIT|FITS)$',
                        filename)

    def file_filter_raw(self, filename):
        """Return True if the raw file should be added."""
        return self.file_filter(filename) and self.fits_file(filename) is None

    def import_file(self, dirname, filename,
                    trust_header=True, copy_files=True, move_files=False):
        """Add details of a file to the manager"""
        source_path = os.path.join(dirname, filename)
        ff = FITSFile(source_path)
        if ff.ndf_class == 'DARK':
            if ff.exposure_str not in self.dark_exposure_str_list:
                self.dark_exposure_str_list.append(ff.exposure_str)
                self.dark_exposure_list.append(ff.exposure)
        self.set_raw_path(ff)
        if os.path.abspath(ff.source_path) != os.path.abspath(ff.raw_path):
            if copy_files:
                print 'Copying file:', filename
                self.update_copy(ff.source_path, ff.raw_path)
            if move_files:
                print 'Moving file: ', filename
                self.move(ff.source_path, ff.raw_path)
            if not copy_files and not move_files:
                print 'Warning! Adding', filename, 'in unexpected location'
                ff.raw_path = ff.source_path
        else:
            print 'Adding file: ', filename
        self.set_name(ff, trust_header=trust_header)
        self.set_reduced_path(ff)
        if not ff.do_not_use:
            ff.make_reduced_link()
        self.file_list.append(ff)
        return

    def set_raw_path(self, ff):
        """Set the raw path for a FITS file."""
        if ff.ndf_class == 'BIAS':
            rel_path = os.path.join('bias', ff.ccd, ff.date)
        elif ff.ndf_class == 'DARK':
            rel_path = os.path.join('dark', ff.ccd, ff.exposure_str, ff.date)
        elif ff.ndf_class == 'LFLAT':
            rel_path = os.path.join('lflat', ff.ccd, ff.date)
        else:
            rel_path = os.path.join(ff.date, ff.ccd)
        ff.raw_dir = os.path.join(self.root, 'raw', rel_path)
        ff.raw_path = os.path.join(ff.raw_dir, ff.filename)
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

    def set_name(self, ff, trust_header=True):
        """Set the object name for a FITS file."""
        ff.name = None
        if ff.ndf_class != 'MFOBJECT':
            # Don't try to set a name for calibration files
            return
        # First check if there's already a name in the header
        if trust_header:
            try:
                ff.name = pf.getval(ff.raw_path, 'MNGRNAME')
            except KeyError:
                pass
        if ff.name is None:
            # Failing that, see if the telescope was pointing in the right
            # direction
	    if (ff.coords.separation(ff.cfg_coords) < self.matching_radius):
                # Yes it was
	        ff.name = 'main'
            else:
                # No it wasn't. Now see if it matches any previous fields
		for extra in self.extra_list:
		    if (ff.coords.separation(extra['coords']) <
			self.matching_radius):
                        # Yes it does
			ff.name = extra['name']
                        break
                else:
                    # No match. As a last resort, ask the user
                    ff.name = raw_input('Enter object name for file ' +
                                        ff.filename + '\n > ')
            # At this point, a name has definitely been set.
            # Put it in the header.
            self.add_header_item(ff, 'MNGRNAME', ff.name,
                                 'Object name set by SAMI_Manager')
        # Check if the field was new
        for extra in self.extra_list:
            if ff.coords.separation(extra['coords']) < self.matching_radius:
                break
        else:
            # No match found: field was new, so add it to the list.
            self.extra_list.append({'name':ff.name,
                                    'coords':ff.coords})
        return

    def set_reduced_path(self, ff):
        """Set the reduced path for a FITS file."""
        if ff.ndf_class == 'BIAS':
            rel_path = os.path.join('bias', ff.ccd, ff.date)
        elif ff.ndf_class == 'DARK':
            rel_path = os.path.join('dark', ff.ccd, ff.exposure_str, ff.date)
        elif ff.ndf_class == 'LFLAT':
            rel_path = os.path.join('lflat', ff.ccd, ff.date)
        elif ff.ndf_class in ['MFFFF', 'MFARC', 'MFSKY']:
            rel_path = os.path.join(ff.date, ff.plate_id, ff.field_id,
                                    'calibrators', ff.ccd)
        else:
            rel_path = os.path.join(ff.date, ff.plate_id, ff.field_id,
                                    ff.name, ff.ccd)
        ff.reduced_dir = os.path.join(self.root, 'reduced', rel_path)
        ff.reduced_link = os.path.join(ff.reduced_dir, ff.filename)
        ff.reduced_path = os.path.join(ff.reduced_dir, ff.reduced_filename)
        if ff.ndf_class == 'MFFFF':
            ff.tlm_path = os.path.join(ff.reduced_dir, ff.tlm_filename)
        elif ff.ndf_class == 'MFOBJECT':
            ff.fluxcal_path = os.path.join(ff.reduced_dir, ff.fluxcal_filename)
            ff.telluric_path = os.path.join(ff.reduced_dir,
                                            ff.telluric_filename)
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
        for ff in self.file_list:
            if ff.filename in filename_options:
                return ff

    def disable_files(self, file_iterable):
        """Disable (delete links to) files in provided list (or iterable)."""
        for ff in file_iterable:
            if isinstance(ff, str):
                ff = self.fits_file(ff)
            ff.do_not_use = True
            if os.path.exists(ff.reduced_link):
                os.remove(ff.reduced_link)
            self.add_header_item(ff, 'DONOTUSE', True,
                                 'Do Not Use flag for SAMI_manager')
        return

    def enable_files(self, file_iterable):
        """Enable files in provided list (or iterable)."""
        for ff in file_iterable:
            if isinstance(ff, str):
                ff = self.fits_file(ff)
            ff.do_not_use = False
            ff.make_reduced_link()
            self.add_header_item(ff, 'DONOTUSE', False,
                                 'Do Not Use flag for SAMI_manager')
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
        return os.path.join(self.root, 'reduced', 'bias', ccd,
                            self.bias_combined_filename())

    def dark_combined_path(self, ccd, exposure_str):
        """Return the path for DARKcombined.fits"""
        return os.path.join(self.root, 'reduced', 'dark', ccd, exposure_str,
                            self.dark_combined_filename(exposure_str))
    
    def lflat_combined_path(self, ccd):
        """Return the path for LFLATcombined.fits"""
        return os.path.join(self.root, 'reduced', 'lflat', ccd,
                            self.lflat_combined_filename())

    def reduce_calibrator(self, calibrator_type, overwrite=False):
        """Reduce all biases, darks of lflats."""
        self.check_calibrator_type(calibrator_type)
        if overwrite:
            # Delete all the old reduced files
            for ff in self.files(ndf_class=calibrator_type.upper(),
                                 reduced=True):
                os.remove(ff.reduced_path)
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
        self.link_calibrator(calibrator_type, overwrite)
        return

    def link_calibrator(self, calibrator_type, overwrite=False):
        """Make necessary symbolic links for XXXXcombined.fits files."""
        if calibrator_type.lower() == 'bias':
            dir_type_list = ['dark', 'lflat', 'calibrators', 'object']
        elif calibrator_type.lower() == 'dark':
            dir_type_list = ['lflat', 'calibrators', 'object']
        elif calibrator_type.lower() == 'lflat':
            dir_type_list = ['calibrators', 'object']
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
        for ff in self.files(ndf_class='MFFFF', do_not_use=False, **kwargs):
            self.reduce_file(ff, overwrite=overwrite, tlm=True,
                             leave_reduced=leave_reduced)
        return

    def reduce_arc(self, overwrite=False, **kwargs):
        """Reduce all arc frames matching given criteria."""
        for ff in self.files(ndf_class='MFARC', do_not_use=False, **kwargs):
            self.reduce_file(ff, overwrite)
        return

    def reduce_fflat(self, overwrite=False, **kwargs):
        """Reduce all fibre flat frames matching given criteria."""
        for ff in self.files(ndf_class='MFFFF', do_not_use=False, **kwargs):
            self.reduce_file(ff, overwrite)
        return

    def reduce_sky(self, overwrite=False, **kwargs):
        """Reduce all offset sky frames matching given criteria."""
        for ff in self.files(ndf_class='MFSKY', do_not_use=False, **kwargs):
            self.reduce_file(ff, overwrite)
        return

    def reduce_object(self, overwrite=False, **kwargs):
        """Reduce all object frames matching given criteria."""
        for ff in self.files(ndf_class='MFOBJECT', do_not_use=False, **kwargs):
            self.reduce_file(ff, overwrite)
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
        return

    def reduce_file(self, ff, overwrite=False, tlm=False, leave_reduced=False):
        """Select appropriate options and reduce the given file."""
        if ff.ndf_class == 'MFFFF' and tlm:
            target = ff.tlm_path
        else:
            target = ff.reduced_path
        if os.path.exists(target) and not overwrite:
            # File to be created already exists, abandon this.
            return
        options = []
        if ff.ndf_class == 'MFFFF' and tlm:
            files_to_match = ['bias', 'dark', 'lflat']
        elif ff.ndf_class == 'MFARC':
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap']
        elif ff.ndf_class == 'MFFFF' and not tlm:
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel']
        elif ff.ndf_class == 'MFSKY':
            files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                              'fflat']
        elif ff.ndf_class == 'MFOBJECT':
            if ff.exposure <= 899.0:
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                  'fflat', 'thput']
                options.extend(['-TPMETH', 'OFFSKY'])
            else:
                files_to_match = ['bias', 'dark', 'lflat', 'tlmap', 'wavel',
                                  'fflat']
        else:
            raise ValueError('Unrecognised NDF_CLASS: '+ff.ndf_class)
        for match_class in files_to_match:
            filename_match = self.match_link(ff, match_class)
            if filename_match is None:
                # What to do if no match was found
                if match_class == 'bias':
                    print ('Warning: Bias frame not found. '
                           'Turning off bias subtraction.')
                    options.extend(['-USEBIASIM', '0'])
                    continue
                elif match_class == 'dark':
                    print ('Warning: Dark frame not found. '
                           'Turning off dark subtraction.')
                    options.extend(['-USEDARKIM', '0'])
                    continue
                elif match_class == 'lflat':
                    print ('Warning: LFlat frame not found. '
                           'Turning off LFlat division.')
                    options.extend(['-USEFLATIM', '0'])
                    continue
                elif match_class == 'thput':
                    # Try to find a suitable object frame instead
                    filename_match = self.match_link(ff, 'thput_object')
                    if filename_match is None:
                        # Still nothing
                        print ('Warning: Offsky (or substitute) frame not '
                               'found. Turning off throughput calibration.')
                        options.extend('THRUPUT', '0')
                        continue
                else:
                    # Anything else missing is fatal
                    raise MatchException('No matching ' + match_class +
                                         ' found for ' + ff.filename)
            options.extend(['-'+match_class.upper()+'_FILENAME',
                            filename_match])
        # All options have been set, so run 2dfdr
        self.run_2dfdr_single(ff, overwrite, options)
        if ff.ndf_class == 'MFFFF' and tlm and not leave_reduced:
            os.remove(ff.reduced_path)
        return

    def extra_options(self, ff):
        """Return a list of extra reduction options suitable for the file."""
        options = []
        if ff.ndf_class == 'MFOBJECT' and ff.exposure <= 899.0:
            # Use offsky throughput values for short exposures
            options.extend(['-TPMETH', 'OFFSKY'])
        return options

    def run_2dfdr_single(self, ff, options=None):
        """Run 2dfdr on a single FITS file."""
        print 'Reducing file:', ff.filename
        if ff.ndf_class == 'BIAS':
            task = 'reduce_bias'
        elif ff.ndf_class == 'DARK':
            task = 'reduce_dark'
        elif ff.ndf_class == 'LFLAT':
            task = 'reduce_lflat'
        elif ff.ndf_class == 'MFFFF':
            task = 'reduce_fflat'
        elif ff.ndf_class == 'MFARC':
            task = 'reduce_arc'
        elif ff.ndf_class == 'MFSKY':
            task = 'reduce_sky'
        elif ff.ndf_class == 'MFOBJECT':
            task = 'reduce_object'
        else:
            raise ValueError('Unrecognised NDF_CLASS')
        command = ['drcontrol', task, ff.filename,
                   '-idxfile', IDX_FILES[ff.ccd]]
        if options is not None:
            command.extend(options)
        with self.visit_dir(ff.reduced_dir):
            with open(os.devnull, 'w') as f:
                subprocess.call(command, stdout=f)
        self.cleanup()
        return

    def run_2dfdr_auto(self, dirname):
        """Run 2dfdr in auto mode in the specified directory."""
        # First find a file in that directory to get the date and ccd
        for ff in self.file_list:
            if os.path.realpath(dirname) == os.path.realpath(ff.reduced_dir):
                date_ccd = ff.filename[:6]
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
        # Write the 2dfdr AutoScript
        script = []
        for ff in file_iterable:
            script.append('lappend glist ' +
                          os.path.relpath(ff.reduced_path, output_dir))
        if len(script) < 2:
            raise ValueError('Need at least 2 files to combine!')
        script.extend(['proc Quit {status} {',
                       '    global Auto',
                       '    set Auto(state) 0',
                       '}',
                       'set task DREXEC1',
                       'global Auto',
                       'set Auto(state) 1',
                       'global AutoScript',
                       'set AutoScript(TimeOut) 300',
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
                       script_filename]
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
              flux_calibrated=None, telluric_corrected=None):
        """Generator for FITS files that satisfy requirements."""
        for ff in self.file_list:
            if ff.ndf_class is None:
                continue
            if ((ndf_class is None or ff.ndf_class in ndf_class) and
                (date is None or ff.date in date) and
                (plate_id is None or ff.plate_id in plate_id) and
                (plate_id_short is None or
                 ff.plate_id_short in plate_id_short) and
                (field_no is None or ff.field_no == field_no) and
                (field_id is None or ff.field_id in field_id) and
                (ccd is None or ff.ccd in ccd) and
                (exposure_str is None or ff.exposure_str in exposure_str) and
                (do_not_use is None or ff.do_not_use == do_not_use) and
                (min_exposure is None or ff.exposure >= min_exposure) and
                (max_exposure is None or ff.exposure <= max_exposure) and
                (reduced_dir is None or
                 os.path.realpath(reduced_dir) ==
                 os.path.realpath(ff.reduced_dir)) and
                (reduced is None or
                 (reduced and os.path.exists(ff.reduced_path)) or
                 (not reduced and not os.path.exists(ff.reduced_path))) and
                (tlm_created is None or
                 (tlm_created and hasattr(ff, 'tlm_path') and
                  os.path.exists(ff.tlm_path)) or
                 (not tlm_created and hasattr(ff, 'tlm_path') and
                  not os.path.exists(ff.tlm_path))) and
                (flux_calibrated is None or
                 (flux_calibrated and hasattr(ff, 'fluxcal_path') and
                  os.path.exists(ff.fluxcal_path)) or
                 (not flux_calibrated and hasattr(ff, 'fluxcal_path') and
                  not os.path.exists(ff.fluxcal_path))) and
                (telluric_corrected is None or
                 (telluric_corrected and hasattr(ff, 'telluric_path') and
                  os.path.exists(ff.telluric_path)) or
                 (not telluric_corrected and hasattr(ff, 'telluric_path') and
                  not os.path.exists(ff.telluric_path)))):
                yield ff
        return

    def ccds(self, do_not_use=False):
        """Generator for ccd names in the data."""
        ccd_list = []
        for ff in self.files(do_not_use=do_not_use):
            if ff.ccd not in ccd_list:
                ccd_list.append(ff.ccd)
                yield ff.ccd
        return

    def reduced_dirs(self, dir_type=None, **kwargs):
        """Generator for reduced directories containing particular files."""
        reduced_dir_list = []
        if dir_type is None:
            ndf_class = None
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
        for ff in self.files(ndf_class=ndf_class, **kwargs):
            if (dir_type.lower() == 'spectrophotometric' and
                (ff.name == 'main' or 'ngc' in ff.name.lower())):
                # This is a galaxy field, not a spectrophotometric standard
                continue
            if ff.reduced_dir not in reduced_dir_list:
                reduced_dir_list.append(ff.reduced_dir)
                yield ff.reduced_dir
        return

    def dark_exposure_strs(self, ccd, do_not_use=False):
        """Generator for dark exposure strings for a given ccd name."""
        exposure_str_list = []
        for ff in self.files(ndf_class='DARK', ccd=ccd, do_not_use=do_not_use):
            if ff.exposure_str not in exposure_str_list:
                exposure_str_list.append(ff.exposure_str)
                yield ff.exposure_str
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

    @contextmanager
    def visit_dir(self, dir_path):
        """Context manager to temporarily visit a directory."""
        os.chdir(dir_path)
        yield
        os.chdir(self.cwd)

    def cleanup(self):
        """Clean up 2dfdr rubbish."""
        with open(os.devnull, 'w') as f:
            subprocess.call(['cleanup'], stdout=f)
        os.chdir(self.cwd)
        return

    def add_header_item(self, ff, key, value, comment=None):
        """Add a header item to the FITS file."""
        if comment is None:
            value_comment = value
        else:
            value_comment = (value, comment)
        hdulist = pf.open(ff.raw_path, 'update',
                          do_not_scale_image_data=True)
        hdulist[0].header[key] = value_comment
        hdulist.close()
        return

    def matchmaker(self, ff, match_class):
        """Return the file that should be used to help reduce the FITS file ff.

        match_class is one of the following:
        tlmap -- Find a tramline map
        wavel -- Find a reduced arc file
        fflat -- Find a reduced fibre flat field
        thput -- Find a reduced offset sky (twilight) file
        thput_object -- As thput, but find a suitable object frame
        bias  -- Find a combined bias frame
        dark  -- Find a combined dark frame
        lflat -- Find a combined long-slit flat frame

        The return type depends on what is asked for:
        tlmap, wavel, fflat, thput, thput_object -- A FITS file object
        bias, dark, lflat       -- The path to the combined file
        """
        ff_match = None
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
        # Define some functions for figures of merit
        time_difference = lambda ff, ff_test: (
            abs(ff_test.epoch - ff.epoch))
        recent_reduction = lambda ff, ff_test: (
            -1.0 * os.stat(ff_test.reduced_path).st_mtime)
        def time_difference_min_exposure(min_exposure):
            def retfunc(ff, ff_test):
                if ff_test.exposure <= min_exposure:
                    return np.inf
                else:
                    return time_difference(ff, ff_test)
            return retfunc
        # Determine what actually needs to be matched, depending on match_class
        if match_class.lower() == 'tlmap':
            # Find a tramline map, so need a fibre flat field
            return_type = 'fits_file'
            ndf_class = 'MFFFF'
            date = ff.date
            plate_id = ff.plate_id
            field_id = ff.field_id
            ccd = ff.ccd
            tlm_created = True
            fom = time_difference
        elif match_class.lower() == 'wavel':
            # Find a reduced arc field
            return_type = 'fits_file'
            ndf_class = 'MFARC'
            date = ff.date
            plate_id = ff.plate_id
            field_id = ff.field_id
            ccd = ff.ccd
            reduced = True
            fom = time_difference
        elif match_class.lower() == 'fflat':
            # Find a reduced fibre flat field
            return_type = 'fits_file'
            ndf_class = 'MFFFF'
            date = ff.date
            plate_id = ff.plate_id
            field_id = ff.field_id
            ccd = ff.ccd
            reduced = True
            fom = time_difference
        elif match_class.lower() == 'thput':
            # Find a reduced offset sky field
            return_type = 'fits_file'
            ndf_class = 'MFSKY'
            date = ff.date
            plate_id = ff.plate_id
            field_id = ff.field_id
            ccd = ff.ccd
            reduced = True
            fom = recent_reduction
        elif match_class.lower() == 'thput_object':
            # Find a reduced object field to take the throughput from
            return_type = 'fits_file'
            ndf_class = 'MFOBJECT'
            date = ff.date
            plate_id = ff.plate_id
            field_id = ff.field_id
            ccd = ff.ccd
            reduced = True
            fom = time_difference_min_exposure(899.0)
        elif match_class.lower() == 'bias':
            # Just return the standard BIAScombined filename
            return_type = 'file_path'
            filename = self.bias_combined_filename()
            if os.path.exists(os.path.join(ff.reduced_dir, filename)):
                return filename
            else:
                return None
        elif match_class.lower() == 'dark':
            # This works a bit differently. Return the filename of the
            # combined dark frame with the closest exposure time.
            return_type = 'file_path'
            best_fom = np.inf
            for exposure_str in self.dark_exposure_strs(ccd=ff.ccd):
                test_fom = abs(float(exposure_str) - ff.exposure)
                if test_fom < best_fom:
                    exposure_str_match = exposure_str
                    best_fom = test_fom
            filename = self.dark_combined_filename(exposure_str_match)
            if os.path.exists(os.path.join(ff.reduced_dir, filename)):
                return filename
            else:
                return None
        elif match_class.lower() == 'lflat':
            # Just return the standard LFLATcombined filename
            return_type = 'file_path'
            filename = self.lflat_combined_filename()
            if os.path.exists(os.path.join(ff.reduced_dir, filename)):
                return filename
            else:
                return None
        else:
            raise ValueError('Unrecognised match_class')
        # Perform the match
        best_fom = np.inf
        for ff_test in self.files(
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
                do_not_use=False,
                ):
            test_fom = fom(ff, ff_test)
            if test_fom < best_fom:
                ff_match = ff_test
                best_fom = test_fom
        return ff_match

    def match_link(self, ff, match_class):
        """Match and make a link to a file, and return the filename."""
        ff_match = self.matchmaker(ff, match_class)
        if ff_match is None:
            # No match was found, send the lack of match onwards
            return None
        if match_class.lower() in ['bias', 'dark', 'lflat']:
            # matchmaker returns a filename in these cases; send it straight on
            return ff_match
        if match_class.lower() == 'tlmap':
            filename = ff_match.tlm_filename
        else:
            filename = ff_match.reduced_filename
        link_path = os.path.join(ff.reduced_dir, filename)
        source_path = os.path.join(ff_match.reduced_dir, filename)
        raw_link_path = os.path.join(ff.reduced_dir, ff_match.filename)
        raw_source_path = os.path.join(ff_match.raw_dir, ff_match.filename)
        # If the link path is occupied by a link, delete it
        # Leave actual files in place
        if os.path.islink(link_path):
            os.remove(link_path)
        if os.path.islink(raw_link_path):
            os.remove(raw_link_path)
        # Make a link, unless the file is already there
        if not os.path.exists(link_path):
            os.symlink(os.path.relpath(source_path, ff.reduced_dir),
                       link_path)
        if not os.path.exists(raw_link_path):
            os.symlink(os.path.relpath(raw_source_path, ff.reduced_dir),
                       raw_link_path)
        return filename

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
	    self.set_plate_id()
	    self.set_plate_id_short()
	    self.set_field_no()
	    self.set_field_id()
	else:
	    self.fibres_extno = None
	    self.plate_id = None
	    self.plate_id_short = None
	    self.field_no = None
	    self.field_id = None
	self.set_ccd()
	self.set_coords()
        self.set_exposure()
        self.set_epoch()
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
	self.plate_id = self.hdulist[self.fibres_extno].header['PLATEID']
        if self.plate_id == '':
            self.plate_id = 'none'

    def set_plate_id_short(self):
	"""Save the shortened plate ID."""
        if self.plate_id == 'none':
            self.plate_id_short = 'none'
        else:
            finish = self.plate_id.find('_', self.plate_id.find('_')+1)
            self.plate_id_short = self.plate_id[:finish]

    def set_field_no(self):
	"""Save the field number."""
	filename = self.hdulist[self.fibres_extno].header['FILENAME']
        if filename == '':
            self.field_no = 0
        else:
            start = filename.rfind('_f') + 2
            self.field_no = int(filename[start:filename.find('.', start)])

    def set_field_id(self):
	"""Save the field ID."""
        if self.plate_id == 'none':
            self.field_id = 'none'
        else:
            start = len(self.plate_id_short)
            for i in range(self.field_no):
                start = self.plate_id.find('_', start) + 1
            finish = self.plate_id.find('_', start)
            if finish == -1:
                field_id = self.plate_id[start:]
            else:
                field_id = self.plate_id[start:finish]
            self.field_id = self.plate_id_short + '_' + field_id
        
    def set_coords(self):
	"""Save the RA/Dec and config RA/Dec."""
	if self.ndf_class == 'MFOBJECT':
	    header = self.hdulist[self.fibres_extno].header
	    self.coords = \
		coord.ICRSCoordinates(ra=header['CENRA'],
				      dec=header['CENDEC'],
				      unit=(units.radian, units.radian))
	    self.cfg_coords = \
		coord.ICRSCoordinates(ra=header['CFGCENRA'],
				      dec=header['CFGCENDE'],
				      unit=(units.radian, units.radian))
	else:
	    self.coords = None
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


class MatchException(Exception):
    """Exception raised when no matching calibrator is found."""
    pass
