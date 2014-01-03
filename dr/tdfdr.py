import subprocess
import os
import tempfile
import shutil
from contextlib import contextmanager

def run_2dfdr(dirname, options=None, cwd=None, unique_imp_scratch=False, 
              **kwargs):
    """Run 2dfdr with a specified set of command-line options."""
    command = ['drcontrol']
    if options is not None:
        command.extend(options)
    if unique_imp_scratch:
        with temp_imp_scratch(**kwargs):
            with visit_dir(dirname, return_to=cwd, 
                           cleanup_2dfdr=True):
                with open(os.devnull, 'w') as dump:
                    subprocess.call(command, stdout=dump)
    else:
        with visit_dir(dirname, return_to=cwd, 
                       cleanup_2dfdr=True):
            with open(os.devnull, 'w') as dump:
                subprocess.call(command, stdout=dump)
    return

def run_2dfdr_single(fits, idx_file, options=None, **kwargs):
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
    options_all = [task, fits.filename, '-idxfile', idx_file]
    if options is not None:
        options_all.extend(options)
    run_2dfdr(fits.reduced_dir, options=options_all, **kwargs)
    return

def cleanup():
    """Clean up 2dfdr crud."""
    with open(os.devnull, 'w') as dump:
        subprocess.call(['cleanup'], stdout=dump)

@contextmanager
def visit_dir(dir_path, return_to=None, cleanup_2dfdr=False):
    """Context manager to temporarily visit a directory."""
    if return_to is None:
        return_to = os.getcwd()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    os.chdir(dir_path)
    try:
        yield
    finally:
        if cleanup_2dfdr:
            cleanup()
        os.chdir(return_to)

@contextmanager
def temp_imp_scratch(restore_to=None, scratch_dir=None, do_not_delete=False):
    """
    Create a temporary directory for 2dfdr IMP_SCRATCH,
    allowing multiple instances of 2dfdr to be run simultaneously.
    """
    # Use current value for restore_to if not provided
    if restore_to is None:
        restore_to = os.environ['IMP_SCRATCH']
    # Make the parent directory, if specified
    # If not specified, tempfile.mkdtemp will choose a suitable location
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
        os.environ['IMP_SCRATCH'] = restore_to
        if not do_not_delete:
            # Remove the temporary directory and all its contents
            shutil.rmtree(imp_scratch)
            # Remove any parent directories that are empty
            try:
                os.removedirs(os.path.dirname(imp_scratch))
            except OSError:
                # It wasn't empty; never mind
                pass
    return

