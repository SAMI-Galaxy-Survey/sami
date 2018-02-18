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
import re
from contextlib import contextmanager
import asyncio
from asyncio.subprocess import PIPE

# Set up logging
from .. import slogging
log = slogging.getLogger(__name__)
log.setLevel(slogging.WARNING)
# log.enable_console_logging()

LOCKDIR = '2dfdrLockDir'

COMMAND_GUI = 'drcontrol'
COMMAND_REDUCE = 'aaorun'

# Check that 2dfdr is available.
try:
    with open(os.devnull, 'w') as dump:
        subprocess.call([COMMAND_REDUCE, 'help'], stdout=dump)
except (OSError, FileNotFoundError):
    error_message = (
        'Cannot find the 2dfdr executable ``{}``\n'.format(COMMAND_REDUCE)
        + 'Please ensure that 2dfdr is correctly installed.')
    raise ImportError(error_message)
try:
    assert len(os.environ["DISPLAY"]) > 0
except (AssertionError, TypeError):
    raise ImportError("2dfdr requires a working DISPLAY. If you are running remotely, try enabling X-forwarding.")



@asyncio.coroutine
def async_call(command_line, **kwargs):
    """Generic function to run a command in an asynchronous way, capturing STDOUT and returning it."""
    formatted_command = " ".join(command_line)
    log.info("async call: {}".format(formatted_command))
    log.debug("Starting async processs: %s", formatted_command)

    # Create subprocess
    proc = yield from asyncio.create_subprocess_exec(*command_line, stdout=PIPE, stderr=PIPE, **kwargs)

    log.debug("Waiting for finish of async processs: %s", formatted_command)

    # Wait (asyncronously) for process to complete and capture stdout (stderr is none unless it is also piped above).
    stdout, stderr = (yield from proc.communicate())
    log.debug("Async process finished: %s", formatted_command)

    stdout = stdout.decode("utf-8")
    # Note: stderr is not currently captured, so this will return None.
    stderr = stderr.decode("utf-8") if stderr else None

    log.debug("Output from command '%s'", formatted_command)
    if log.isEnabledFor(slogging.DEBUG):
        for line in stdout.splitlines():
            log.debug("   " + line)

    return stdout


@asyncio.coroutine
def call_2dfdr_reduce(dirname, options=None):
    """Call 2dfdr in pipeline reduction mode using `aaorun`"""
    # Make a temporary directory with a unique name for use as IMP_SCRATCH
    with tempfile.TemporaryDirectory() as imp_scratch:

        command_line = [COMMAND_REDUCE]
        if options is not None:
            command_line.extend(options)

        # Set up the environment:
        environment = dict(os.environ)
        environment["IMP_SCRATCH"] = imp_scratch

        with directory_lock(dirname):
            tdfdr_stdout = yield from async_call(command_line, cwd=dirname, env=environment)

        # Confirm that the above command ran to completion, otherwise raise an exception
        try:
            confirm_line = tdfdr_stdout.splitlines()[-2]
            assert re.match(r"Data Reduction command \S+ completed.", confirm_line)
        except (IndexError, AssertionError):
            message = "2dfdr did not run to completion for command: %s" % " ".join(command_line)
            raise TdfdrException(message)


def call_2dfdr_gui(dirname, options=None):
    """Call 2dfdr in GUI mode using `drcontrol`"""
    # Make a temporary directory with a unique name for use as IMP_SCRATCH
    with tempfile.TemporaryDirectory() as imp_scratch:

        command_line = [COMMAND_GUI]
        if options is not None:
            command_line.extend(options)

        # Set up the environment:
        environment = dict(os.environ)
        environment["IMP_SCRATCH"] = imp_scratch

        with directory_lock(dirname):
            subprocess.run(command_line, cwd=dirname, check=True, env=environment)


def load_gui(dirname, idx_file=None):
    """Load the 2dfdr GUI in the specified directory."""
    if idx_file is not None:
        options = [idx_file]
    else:
        options = None
    call_2dfdr_gui(dirname, options)
    return


@asyncio.coroutine
def run_2dfdr_single(fits, idx_file, options=None):
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
    yield from call_2dfdr_reduce(fits.reduced_dir, options=options_all)
    return '2dfdr Reduced file:' + fits.filename


def run_2dfdr_combine(input_path_list, output_path, idx_file):
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
    asyncio.get_event_loop().run_until_complete(call_2dfdr_reduce(output_dir, options=options))


def cleanup():
    """Clean up 2dfdr crud."""
    log.warning("It is generally not safe to cleanup 2dfdr in any other way than interactively!")
    subprocess.call(['cleanup'], stdout=subprocess.DEVNULL)


@contextmanager
def directory_lock(working_directory):
    """Create a context where 2dfdr can be run that is isolated from any other instance of 2dfdr."""

    lockdir = os.path.join(working_directory, LOCKDIR)

    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    # Attempt to obtain a directory lock in this directory
    try:
        os.mkdir(lockdir)
    except OSError:
        # Lock directory already exists, i.e. another task is working here
        # Run away!
        raise LockException(
            'Directory locked by another process: ' + working_directory)
    else:
        assert os.path.exists(lockdir)
        log.debug("Lock Directory '{}' created".format(lockdir))
    try:
        yield
    finally:
        log.debug("Will delete lock directory '{}'".format(lockdir))
        os.rmdir(lockdir)


class TdfdrException(Exception):
    """Base 2dfdr exception."""
    pass


class LockException(Exception):
    """Exception raised when attempting to work in a locked directory."""
    pass
