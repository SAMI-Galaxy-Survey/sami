import subprocess
import os
from contextlib import contextmanager

def run_2dfdr_single(fits, idx_file, options=None, cwd=None):
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
               '-idxfile', idx_file]
    if options is not None:
        command.extend(options)
    with visit_dir(fits.reduced_dir, return_to=cwd, cleanup_2dfdr=True):
        with open(os.devnull, 'w') as dump:
            subprocess.call(command, stdout=dump)
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
