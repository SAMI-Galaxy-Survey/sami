'''
slogging: the SAMI Logging utility

This provides a simple logging utility suitable for use in SAMI modules.

Generically, one only needs to add the following lines to the top of a file:

    # Set up logging
    import slogging
    log = slogging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

Then one can add messages to the log by calling, e.g.,

    log.debug("Just reached step n")
    log.info("More information")
    log.warning("I'm doing something stupid maybe you should pay attention")
    log.error("Bad things have just happened")

The logging destinations are configured by calling
`slogging.configure_logging`, which must be done before the "getLogger"
command, and which will affect all subsequent loggers to be initialised from
slogging.

Created on Apr 30, 2014

@author: agreen

This module does not seem to be used anywhere. I don't know if anyone has
been using it interactively. (JTA)
'''

# Set up logging
import logging
# log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)

logging_configured = False
file_handler = None
console_handler = None

# Copy some things from parent package
DEBUG = logging.DEBUG
WARN = logging.WARN
WARNING = logging.WARNING
ERROR = logging.ERROR
INFO = logging.INFO

def configure_logging(output_filename=None,console_logging_enabled=False):
    
    # Set up logging. We assume here we are the only thing running in this
    # interpreter, so we define the behaviour of the root logger. Loggers for
    # each module will inherit this behaviour.
    #
    # NOTE: The logging level should not be set here, but rather at the top of
    # each file.
    root_log = logging.getLogger()
    formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)s %(funcName)s: %(message)s')

    if output_filename is not None:
        file_handler = logging.FileHandler(output_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_log.addHandler(file_handler)
    
    if console_logging_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        root_log.addHandler(console_handler)
    logging_configured = True

def getLogger(name):
    
    if not logging_configured:
        configure_logging()
    log = logging.getLogger(name)
    if console_handler is not None:
        log.addHandler(console_handler)
    if file_handler is not None:
        log.addHandler(file_handler)
    
    return log
        