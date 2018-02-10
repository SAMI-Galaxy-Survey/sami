"""
Module for controlling 2dfdr.

The actual 2dfdr call is done from run_2dfdr(). Other functions provide
more convenient access for different tasks: reducing a file
(run_2dfdr_single), combining files (run_2dfdr_combine) or loading the GUI
(load_gui).

The visit_dir context manager temporarily changes the working directory to
the one in which the file to be reduced is.

The temp_imp_scratch context manager temporarily sets the IMP_SCRATCH
environment variable to a temporary directory, allowing 2dfdr to be run in
parallel without conflicts. This is becoming unnecessary with recent
versions of 2dfdr which are ok with multiple instances, but there is no
particular reason to take it out.

Could easily be spun off as an independent module for general use. The
only thing holding this back is the assumed use of FITSFile objects as
defined in sami.manager, in run_2dfdr_single.

Note this module currently exists in two forms in two branches of this
repository: the default branch is compatible with 2dfdr v5, and the aaorun
branch is compatible with 2dfdr v6. The original plan had been to phase
out support for 2dfdr v5 and merge aaorun into default, but it may be
better to merge them in a way that retains the v5 functionality.
"""

import subprocess
import os
import tempfile
import shutil
from contextlib import contextmanager

LOCKDIR = '2dfdrLockDir'

COMMAND_GUI = 'drcontrol'
COMMAND_REDUCE = 'aaorun'

def call(command_line, debug=False, **kwargs):
    """Simply passes the command out to a subprocess, unless debug is True."""
    if debug:
        print('CWD: ' + os.getcwd())
        print(command_line)
    else:
        subprocess.call(command_line, **kwargs)

def run_2dfdr(dirname, options=None, return_to=None, unique_imp_scratch=False,
              lockdir=LOCKDIR, command=COMMAND_REDUCE, debug=False, **kwargs):
    """Run 2dfdr with a specified set of command-line options."""
    command_line = [command]
    if options is not None:
        command_line.extend(options)
    if unique_imp_scratch:
        with temp_imp_scratch(**kwargs):
            with visit_dir(dirname, return_to=return_to, 
                           cleanup_2dfdr=True, lockdir=lockdir):
                with open(os.devnull, 'w') as dump:
                    call(command_line, stdout=dump, debug=debug)
    else:
        with visit_dir(dirname, return_to=return_to, 
                       cleanup_2dfdr=True, lockdir=lockdir):
            with open(os.devnull, 'w') as dump:
                call(command_line, stdout=dump, debug=debug)
    return

def load_gui(dirname=None, idx_file=None, lockdir=LOCKDIR, **kwargs):
    """Load the 2dfdr GUI in the specified directory."""
    if dirname is None:
        dirname = os.getcwd()
    if idx_file is not None:
        options = [idx_file]
    else:
        options = None
    run_2dfdr(dirname, options, lockdir=lockdir, command=COMMAND_GUI, **kwargs)
    return

def run_2dfdr_single(fits, idx_file, options=None, lockdir=LOCKDIR, **kwargs):
    """Run 2dfdr on a single FITS file."""
    print('Reducing file:', fits.filename)
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
    out_dirname = fits.filename[:fits.filename.rindex('.')] + '_outdir'
    out_dirname_full = os.path.join(fits.reduced_dir, out_dirname)
    if not os.path.exists(out_dirname_full):
        os.makedirs(out_dirname_full)
    options_all = [task, fits.filename, '-idxfile', idx_file,
                   '-OUT_DIRNAME', out_dirname]
    if options is not None:
        options_all.extend(options)
    run_2dfdr(fits.reduced_dir, options=options_all, lockdir=lockdir, **kwargs)
    return

# def run_2dfdr_combine(input_path_list, output_path, return_to=None, 
#                       lockdir=LOCKDIR, **kwargs):
#     """Run 2dfdr to combine the specified FITS files."""
#     if len(input_path_list) < 2:
#         raise ValueError('Need at least 2 files to combine!')
#     output_dir, output_filename = os.path.split(output_path)
#     # Need to extend the default timeout value; set to 5 hours here
#     timeout = '300'
#     # Write the 2dfdr AutoScript
#     script = []
#     for input_path in input_path_list:
#         script.append('lappend glist ' +
#                       os.path.relpath(input_path, output_dir))
#     script.extend(['proc Quit {status} {',
#                    '    global Auto',
#                    '    set Auto(state) 0',
#                    '}',
#                    'set task DREXEC1',
#                    'global Auto',
#                    'set Auto(state) 1',
#                    ('ExecCombine $task $glist ' + output_filename +
#                     ' -success Quit')])
#     script_filename = '2dfdr_script.tcl'
#     with visit_dir(output_dir, return_to=return_to, lockdir=lockdir):
#         # Print the script to file
#         with open(script_filename, 'w') as f_script:
#             f_script.write('\n'.join(script))
#         # Run 2dfdr
#         options = ['-AutoScript',
#                    '-ScriptName',
#                    script_filename,
#                    '-Timeout',
#                    timeout]
#         run_2dfdr(output_dir, options, lockdir=None, **kwargs)
#         # Clean up the script file
#         os.remove(script_filename)
#     return

def run_2dfdr_combine(input_path_list, output_path, idx_file, **kwargs):
    """Run 2dfdr to combine the specified FITS files."""
    if len(input_path_list) < 2:
        raise ValueError('Need at least 2 files to combine!')
    output_dir, output_filename = os.path.split(output_path)
    options = ['combine_image',
               ' '.join([os.path.relpath(input_path, output_dir)
                         for input_path in input_path_list]),
               '-COMBINEDFILE',
               output_filename,
               '-idxfile',
               idx_file]
    run_2dfdr(output_dir, options=options, **kwargs)

def cleanup():
    """Clean up 2dfdr crud."""
    with open(os.devnull, 'w') as dump:
        subprocess.call(['cleanup'], stdout=dump)

@contextmanager
def visit_dir(dir_path, return_to=None, cleanup_2dfdr=False, lockdir=LOCKDIR):
    """Context manager to temporarily visit a directory."""
    if return_to is None:
        return_to = os.getcwd()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    os.chdir(dir_path)
    if lockdir is not None:
        try:
            os.mkdir(lockdir)
        except OSError:
            # Lock directory already exists, i.e. another task is working here
            # Run away!
            os.chdir(return_to)
            raise LockException(
                'Directory locked by another process: ' + dir_path)
    try:
        yield
    finally:
        if cleanup_2dfdr:
            cleanup()
        if lockdir is not None:
            os.rmdir(lockdir)
        os.chdir(return_to)

@contextmanager
def temp_imp_scratch(restore_to=None, scratch_dir=None, do_not_delete=False):
    """
    Create a temporary directory for 2dfdr IMP_SCRATCH,
    allowing multiple instances of 2dfdr to be run simultaneously.
    """
    try:
        old_imp_scratch = os.environ['IMP_SCRATCH']
    except KeyError:
        old_imp_scratch = None
    # Use current value for restore_to if not provided
    if restore_to is None:
        restore_to = old_imp_scratch
    # Make the parent directory, if specified
    # If not specified, tempfile.mkdtemp will choose a suitable location
    if scratch_dir is None and old_imp_scratch is not None:
        scratch_dir = old_imp_scratch
    if scratch_dir is not None and not os.path.exists(scratch_dir):
        os.makedirs(scratch_dir)
    # Make a temporary directory with a unique name
    imp_scratch = tempfile.mkdtemp(dir=scratch_dir)
    # Set the IMP_SCRATCH environment variable to that directory, so that
    # 2dfdr will use it
    os.environ['IMP_SCRATCH'] = imp_scratch
    try:
        yield
    finally:
        # Change the IMP_SCRATCH environment variable back to what it was
        if restore_to is not False:
            if restore_to is not None:
                os.environ['IMP_SCRATCH'] = restore_to
            else:
                del os.environ['IMP_SCRATCH']
        if not do_not_delete:
            # Remove the temporary directory and all its contents
            shutil.rmtree(imp_scratch)
            # No longer remove parent directories, as it can screw things up
            # next time around
            # # Remove any parent directories that are empty
            # try:
            #     os.removedirs(os.path.dirname(imp_scratch))
            # except OSError:
            #     # It wasn't empty; never mind
            #     pass
    return

class TdfdrException(Exception):
    """Base 2dfdr exception."""
    pass

class LockException(Exception):
    """Exception raised when attempting to work in a locked directory."""
    pass

