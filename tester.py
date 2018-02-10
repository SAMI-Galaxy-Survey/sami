from __future__ import print_function

import os
import warnings
import shutil
import re

import numpy as np

import sami

from astropy.io import fits



class Tester():

    """This class handles the testing of the SAMI pipeline. To run it requires
    additional data:
    1) the folder ``sami_ppl_test_data``, i.e. the raw data to be reduced
       locally (~1.1 GB - ~0.5 GB xz archive)
    2) the reduced data (folders ``slow_test_reference`` or
       ``fast_test_reference`` for the ``slow`` and ``fast`` reduction
       respectively (~7.5 GB - ~4.1 GB xz archive).

    To run the test (fast mode):
    >>> import sami.tester
    >>> mytest = tester.Tester(fast=True)
    >>> mngr = mytest.dr_reduce()
    >>> comp = mytest.dr_comparison()

    1) mngr is the instance of ``sami.manager.Manager`` that controls the local
       data reduction. It can be used to perform additional operations on the
       local data reduction, but strictly speaking should not be needed.
    2) comp is the result of the comparison. Should be True.


    Parameters
    ----------

    fast : bool, optional
        Whether to use the ``fast`` or ``slow`` data reduction option for the
        ``sami.manager.Manager``.
    rtol : float, optional
        The relative tolerance to assess whether any two numbers are the same.
        The default value of 1.e-07 is appropriate for single precision numbers.
    output_dir : str, optional
        The directory where to write the local data reduction. Default
        is None, which uses either ``slow_test`` or  ``fast_test``, depending on
        the value of the keyword ``fast``.
    reference_dir : str, optional
        The directory where to search for the reference data reduction. Default
        is None, which uses either ``slow_test_reference`` or 
        ``fast_test_reference``, depending on the value of the keyword ``fast``.
    create_reduction : bool, optional
        This flag is set to True in order to create the reference data
        reduction. When testing one should leave this keyword to its default
        value (False).

    """

    def __init__(self, fast=True, rtol=1.e-07, output_dir=None,
                 reference_dir=None, create_reduction=False):

        self.fast=fast
        self.rtol=rtol

        if output_dir is None:
            self.output_dir = 'fast_test' if fast else 'slow_test'
        else:
            self.output_dir = output_dir

        if reference_dir is None:
            self.reference_dir = 'fast_test_reference' if fast \
                else 'slow_test_reference'
        else:
            self.reference_dir = reference_dir

        if not create_reduction:
            # Check that ``reference_dir`` contains the expected files.
            self._check_reference_exists()
    


    def _check_reference_exists(self):

        """Method to assess whether the required reference data exists in the
        directory ``self.reference_dir``.

        """

        if not _check_existing_cubing(self.reference_dir):
            error_message = ('The directory "{}" does not appear to contain ' \
                + ' the required (reduced) data. Please contact the pipeline ' \
                + ' support team to obtain the reference dataset.').format(
                self.reference_dir)
            raise IOError(error_message)
        else:
            pass



    def dr_reduce(self,
                  overwrite_bias=False, overwrite_dark=False,
                  overwrite_lflat=False, overwrite_tlm=False,
                  overwrite_arc_n_flat=False, overwrite_sky_n_object=False,
                  overwrite_fluxcal=False, overwrite_cubing=False):

        mngr = dr_reduce(
            fast=self.fast, output_dir=self.output_dir,
            overwrite_bias=overwrite_bias, overwrite_dark=overwrite_dark,
            overwrite_lflat=overwrite_lflat, overwrite_tlm=overwrite_tlm,
            overwrite_arc_n_flat=overwrite_arc_n_flat,
            overwrite_sky_n_object=overwrite_sky_n_object,
            overwrite_fluxcal=overwrite_fluxcal,
            overwrite_cubing=overwrite_cubing)

        return mngr



    def dr_comparison(self):

        return dr_comparison(self.output_dir, self.reference_dir,
                             rtol=self.rtol)



# Example usage:
# >>> import sami.tester
# >>> res = tester.dr_reduce(fast=True)
# Test to relative tolerance.
# >>> comp = tester.dr_comparison('fast_test', 'fast_test_reference', rtol=1.e-07)


# +---+------------------------------------------------------------------------+
# |1. | Performing the data reduction.                                         |
# +---+------------------------------------------------------------------------+

def dr_reduce(fast=True, output_dir=None,
            overwrite_bias=False, overwrite_dark=False, overwrite_lflat=False,
            overwrite_tlm=False, overwrite_arc_n_flat=False,
            overwrite_sky_n_object=False, overwrite_fluxcal=False,
            overwrite_cubing=False):

    """This method does the data reduction on the test data suite.

    Parameters
    ----------

    fast: bool, True
        whether to perform the fast or slow data reduction (see the
        relevant documentation for sami.manager.Manager for more information).
    overwrite_<function>: bool, False
        whether to manually overwrite the preexisting data reduction step
        corresponding to <function> (if exists).

    Return
    ------

    mngr: ``sami.manager.Manager`` instance
        The Manager instance with the data reduction.

    """

    # Declare the output directory.
    if output_dir is None:
         output_dir = 'fast_test' if fast else 'slow_test'

    # If an old reduction exists, ask the user whether to delete it or keep it.
    if _check_existing_reduction(output_dir):
        _delete_existing_reduction(output_dir)

    # Importing the data.
    mngr = sami.manager.Manager(output_dir, fast=fast)
    mngr.import_dir('sami_ppl_test_data')
    mngr.remove_directory_locks()

    message('Processing the bias data...')
    mngr.reduce_bias(overwrite=overwrite_bias)
    mngr.combine_bias(overwrite=overwrite_bias)

    message('Processing the dark data...')
    mngr.reduce_dark(overwrite=overwrite_dark)
    mngr.combine_dark(overwrite=overwrite_dark)

    message('Processing the detector flat (lflat) data...')
    mngr.reduce_lflat(overwrite=overwrite_lflat)
    mngr.combine_lflat(overwrite=overwrite_lflat)

    message('Tracing the fibres (tramlines)...')
    mngr.make_tlm(overwrite=overwrite_tlm)

    message('Reducing the arc & flat frames...')
    mngr.reduce_arc(overwrite=overwrite_arc_n_flat)
    mngr.reduce_fflat(overwrite=overwrite_arc_n_flat)

    message('Reducing the sky and object frames...')
    mngr.reduce_sky(overwrite=overwrite_sky_n_object)
    mngr.reduce_object(overwrite=overwrite_sky_n_object)

    message('Flux calibration...')
    mngr.derive_transfer_function(overwrite=overwrite_fluxcal)
    mngr.combine_transfer_function(overwrite=overwrite_fluxcal)
    mngr.flux_calibrate(overwrite=overwrite_fluxcal)
    mngr.telluric_correct(overwrite=overwrite_fluxcal)
    mngr.get_stellar_photometry()
    mngr.scale_frames(overwrite=overwrite_fluxcal)
    mngr.measure_offsets(overwrite=overwrite_fluxcal)

    message('Cubing...')

    # Check whether cubing has been done in the past. If yes, use the keyword
    # ``overwrite_cubing`` to determine whether or not to redo this process.
    # This step is necessary because the keyword ``overwrite`` does not work
    # for ``sami.manager.Manager.cube``.
    if (not _check_existing_cubing(output_dir)) or overwrite_cubing:
        warn_message = 'Notice: sami.manager.Manger.cube`` is time consuming.' \
            + '\nThis tester will only cube one IFU (the secondary star one).'
        warnings.warn(warn_message)
        mngr.cube(overwrite=overwrite_cubing, star_only=True)
        mngr.scale_cubes(overwrite=overwrite_cubing)

        #mngr.record_dust(overwrite=overwrite_cubing)
        #mngr.gzip_cubes(overwrite=overwrite_cubing) # Unsupported

    mngr.qc_summary()

    check_results(mngr)

    return mngr




# +---+------------------------------------------------------------------------+
# |2. | Assessing the results.                                                 |
# +---+------------------------------------------------------------------------+

def dr_comparison(output_dir, reference_dir, rtol=1.e-07):

    comparison = []

    for prod_name in ['bias', 'dark', 'lflat', 'calibrators', 'ststar', 'main',
                      'cubed']:
        filelist_a, filelist_b = _retrieve_filenames(prod_name, output_dir,
                                                     reference_dir)

        for fn_a, fn_b in zip(filelist_a, filelist_b):
            fa, fb = os.path.basename(fn_a), os.path.basename(fn_b)
            comparison.append([fa, fb, _compare_files(fn_a, fn_b, rtol=rtol)])

    all_equal = [comp[2] for comp in comparison]

    if np.all(all_equal):
        return True
    else:
        warnings.warn('Not all comparisons have been successfull')
    return comparison


def _retrieve_filenames(input_product_name, output_dir, reference_dir):

    """This method retrieves the filenames from the data reduction (directory
    ``output_dir``) and from the standard (or reference) data reduction 
    (directory ``reference_dir``). It returns couples of filenames that need to
    be compared.

    input_product_name: str
        ['bias'|'dark'|'lflat'|'main'|'calibrators'|'ststar'|'cubed']

    Return
    ------

    Two lists of filenames, with the names of the files that need to be compared
    as in:
    [file_1a, file_2a, ..., file_Na], [file_1b, file_2b, ..., file_Nb]

    """

    pn_dic = {'bias': 'bias', 'dark': 'dark', 'lflat': 'lflat', 'cubed': 'cubed',
              'main': 'main', 'calibrators': 'calibrators', 'ststar': 'EG21'}
    product_name = pn_dic[input_product_name]

    pn_regex = {'bias': '.*' + product_name + \
                    '.*/([0-9]{2,2}[a-z]{3,3}[0-9]{5,5}red.fits|' + \
                    '.*BIAScombined.*fits)',
                'dark': '.*' + product_name + \
                    '.*/([0-9]{2,2}[a-z]{3,3}[0-9]{5,5}red.fits|' + \
                    '.*DARKcombined.*fits)',
                'lflat': '.*' + product_name + \
                    '.*/([0-9]{2,2}[a-z]{3,3}[0-9]{5,5}red.fits|' + \
                    '.*LFLATcombined.*fits)',
                'main': '.*' + product_name + \
                    '.*/[0-9]{2,2}[a-z]{3,3}[0-9]{5,5}sci.fits',
                'calibrators': '.*' + product_name + \
                    '.*/[0-9]{2,2}[a-z]{3,3}[0-9]{5,5}red.fits',
                'ststar': '.*' + product_name + \
                    '.*/[0-9]{2,2}[a-z]{3,3}[0-9]{5,5}im.fits',
                'cubed': '.*' + product_name + \
                    '.*/[0-9]{4,}_(blue|red)_.*_Y.*.fits$'}
    regex = pn_regex[input_product_name]

    # Loop over all directories of processed galaxies.
    match_files = re.compile(regex)

    # Result files.
    result_file_list = [dirpath + '/' + filename
        for dirpath, dirnames, filenames in os.walk(output_dir)
        for filename in filenames
        if match_files.search(dirpath + '/' + filename)]

    reference_file_list = [dirpath + '/' + filename
        for dirpath, dirnames, filenames in os.walk(reference_dir)
        for filename in filenames
        if match_files.search(dirpath + '/' + filename)]

    if input_product_name == 'main':
        result_file_list = _replace_names(result_file_list,
                                          instr='sci', outstr='red')
        reference_file_list = _replace_names(reference_file_list,
                                             instr='sci', outstr='red')
    elif input_product_name == 'ststar':
        result_file_list = _replace_names(result_file_list,
                                          instr='im', outstr='red')
        reference_file_list = _replace_names(reference_file_list,
                                             instr='im', outstr='red')

    result_file_list.sort(), reference_file_list.sort()

    # Assess whether the files are correctly matched (i.e. result_file_list[n]
    # has the same filename as reference_file_list[n], for every n).
    matched = _assess_matched_files(reference_file_list, result_file_list)

    if not matched:
        error_message = ('The filenames associated with the product "{}" ' \
            + 'between the directories "{}" and "{}" do not match.\n' \
            + 'Please inspect the relevant directories manually.').format(
            product_name, output_dir, reference_dir)
        raise AssertionError(error_message)
    else:
        return result_file_list, reference_file_list



def _compare_files(file_a, file_b, rtol=1.e-07):

    """Compare the contents of the FITS files ``file_a`` and ``file_b`` to a
    precision of ``rtol``.

    Return
    ------

    True if the data is equal to the required precision, False otherwise.

    """

    data_a, data_b = fits.open(file_a), fits.open(file_b)

    # List to store the results
    are_close = [True for ext in data_a]

    for n, (ext_a, ext_b) in enumerate(zip(data_a, data_b)):

        try:
            are_close[n] = np.testing.assert_allclose(
                ext_a.data, ext_b.data, rtol=rtol,
                atol=0., verbose=True)
        except TypeError:
            warnings.warn('Skipping binary table(s): testing not implemented yet.')
            pass
        except AssertionError as ae:
            warnings.warn(ae.message)
            return False

    return True



def _assess_matched_files(filenames_list_a, filenames_list_b):

    """Assess whether the filenames in the input lists ``filenames_list_a`` and
    ``filenames_list_b`` are the same (apart from the path). This is a helper
    method for ``_retrieve_filenames``.

    Return
    ------

    True if the check is passed, False otherwise.

    """

    if len(filenames_list_a) != len(filenames_list_b):
        return False
    else:
        for fn_a, fn_b in zip(filenames_list_a, filenames_list_b):
            if os.path.basename(fn_a) != os.path.basename(fn_b):
                return False

    # If it gets here, all checks have been passed and the list have the 
    # same filenames in the correct order.
    return True

        


# +---+------------------------------------------------------------------------+
# |3. | Utilities. Hic sunt leones.                                            |
# +---+------------------------------------------------------------------------+

def _check_existing_reduction(dir_name):

    reduction_exists = os.path.exists(dir_name)

    if reduction_exists:
        warn_message = 'An old reduction has been detected.\n' \
            + 'Please notice that it might (or might not) be incomplete.'
        warnings.warn(warn_message)

    return reduction_exists



def _check_existing_cubing(dir_name):

    regex = '.*cubed.*/[0-9]{4,}_(blue|red)_.*_Y.*.fits$'

    # Loop over all directories of processed galaxies.
    match_files = re.compile(regex)

    # Result files.
    cubes_file_list = [dirpath + '/' + filename
        for dirpath, dirnames, filenames in os.walk(dir_name)
        for filename in filenames
        if match_files.search(dirpath + '/' + filename)]

    return len(cubes_file_list)==2 # 1 IFU (star_only=True) x 2 spectrogroph arms.
    return len(cubes_file_list)==26 # 13 IFUs x 2 spectrogroph arms.



def _delete_existing_reduction(dir_name):

    """Ask the user whether to delete the old data reduction.

    """

    delete_er = input(('Delete the old reduction ({}) or resume it? ' \
        + '(y=yes / any other key=continue):\n').format(dir_name))
    
    if delete_er == 'y':
        shutil.rmtree(dir_name)
    else:
        warnings.warn('Continuing the existing data reduction...')
        pass



def _replace_names(input_list, instr='sci', outstr='red'):

    """Replace ``instr`` with ``outstr`` in every element of ``input_list``.

    """

    instr += '.fits'
    outstr += '.fits'

    for n in range(len(input_list)):
        input_list[n] = input_list[n].replace(instr, outstr)

    return input_list



def check_results(input_mngr):

    """Asks the user whether to perform manual checking of the results. """

    do_checks = True
    while do_checks:

        do_checks = input('Check the results? (y=yes, any other key=no):\n')

        if do_checks == 'y':
            try:
                input_mngr.check_next_group()
                do_checks = True
            except IndexError:
                message('No more checks to be done')
                do_checks = False
        else:
            message('No more checks will be done')
            do_checks = False
       

def message(message):

    """Yapp, yet another pritty printer. ``message`` is the text to be printed.

    """

    print()
    print('*******************************************************************')
    print('* {0:<{1}} *'.format(message, 63))
    print('*******************************************************************')
    print()
