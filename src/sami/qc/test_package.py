"""
Functionality to automatically put together test packages.

A test package in this context is a collection of data used to test out
potential improvements to the data reduction. The test package provides
a fiducial reduction against which the improvement can be compared.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
from glob import glob
import re

from ..manager import Manager

PACKAGES = {
    'sami_test_package_1': [
        '06mar10045.fits',
        '06mar10047.fits',
        '06mar10048.fits',
        '06mar10050.fits',
        '06mar10051.fits',
        '06mar10052.fits',
        '06mar10053.fits',
        '06mar10054.fits',
        '06mar10055.fits',
        '06mar10056.fits',
        '06mar10057.fits',
        '06mar10059.fits',
        '06mar10060.fits',
        '06mar10061.fits',
        '06mar10062.fits',
        '06mar10066.fits',
        '06mar20045.fits',
        '06mar20047.fits',
        '06mar20048.fits',
        '06mar20050.fits',
        '06mar20051.fits',
        '06mar20052.fits',
        '06mar20053.fits',
        '06mar20054.fits',
        '06mar20055.fits',
        '06mar20056.fits',
        '06mar20057.fits',
        '06mar20059.fits',
        '06mar20060.fits',
        '06mar20061.fits',
        '06mar20062.fits',
        '06mar20066.fits',
        ],
}

def make_package(name_or_file_list, source_root, output_name=None):
    """Make and re-reduce a package."""
    if isinstance(name_or_file_list, str):
        file_list = PACKAGES[name_or_file_list]
        if output_name is None:
            output_name = name_or_file_list
    else:
        file_list = name_or_file_list
        if output_name is None:
            raise ValueError(
                'Must provide an output_name if file list is provided')
    mngr = Manager(output_name)
    import_raw_files(file_list, source_root, mngr)
    import_calibrators(source_root, mngr)
    reduce_data(mngr)
    return

def import_raw_files(filename_list, source_root, mngr):
    """Copy raw files to temp directory and import into manager."""
    dest_dir = os.path.join(mngr.abs_root, 'tmp')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for filename in filename_list:
        source_path = (
            glob(os.path.join(
                source_root, 'raw', '*', '*', filename)) +
            glob(os.path.join(
                source_root, 'raw', '*', '*', '*', filename)) + 
            glob(os.path.join(
                source_root, 'raw', '*', '*', '*', '*', filename))
            )[0]
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(source_path, dest_path)
    mngr.import_dir(dest_dir)
    return

def import_calibrators(source_root, mngr):
    """Import BIAScombined.fits and similar into manager."""
    for ccd in ('ccd_1', 'ccd_2'):
        for calibrator in ('bias', 'lflat'):
            filename = calibrator.upper()+'combined.fits'
            source_dir = os.path.join(source_root, 'reduced', calibrator, ccd)
            dest_dir = os.path.join(mngr.abs_root, 'reduced', calibrator, ccd)
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            shutil.copy2(source_path, dest_path)
    dark_list = []
    match_expr = (r'(?P<root>.*)(?P<reddark>reduced/dark)/(?P<ccd>ccd_.)/'
                  r'(?P<time>\d+)/(?P<filename>DARKcombined(?P=time).fits)')
    for source_path in glob(os.path.join(
            source_root, 'reduced', 'dark', '*', '*', 'DARKcombined*.fits')):
        match = re.match(match_expr, source_path)
        dest_dir = os.path.join(
            mngr.abs_root, 'reduced', 'dark', match.group('ccd'),
            match.group('time'))
        dest_path = os.path.join(dest_dir, match.group('filename'))
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        shutil.copy2(source_path, dest_path)
        # Find a matching raw fits file
        raw_path = glob(os.path.join(
            source_root, 'reduced', 'dark', match.group('ccd'),
            match.group('time'), '*', '??????????.fits'))[0]
        dark_list.append(os.path.basename(raw_path))
    import_raw_files(dark_list, source_root, mngr)
    mngr.link_bias()
    mngr.link_dark()
    mngr.link_lflat()
    return

def reduce_data(mngr):
    """Do all of the necessary reduction steps."""
    mngr.make_tlm()
    mngr.reduce_arc()
    mngr.reduce_fflat()
    mngr.reduce_sky()
    mngr.reduce_object()
    mngr.derive_transfer_function()
    mngr.combine_transfer_function()
    mngr.flux_calibrate()
    mngr.telluric_correct()
    mngr.measure_offsets()
    mngr.cube()
    return
