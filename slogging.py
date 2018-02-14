'''
slogging: the SAMI Logging utility

This provides a simple logging utility suitable for use in libraries, etc. It
was originally written for the SAMI Python Package by Andy Green, but has been
adapted for use elsewhere.

Generically, one only needs to add the following lines to the top of a file:

    # Set up logging
    import slogging
    log = slogging.getLogger(__name__)
    log.setLevel(slogging.DEBUG)

Additional logging may be configured at the top of each file by, e.g.,

    log.add_file('/path/to/filename')
    log.enable_console_logging()

Then one can add messages to the log by calling, e.g.,

    log.debug("Just reached step n")
    log.info("More information")
    log.warning("I'm doing something stupid maybe you should pay attention")
    log.error("Bad things have just happened")

Logging levels are: (see https://docs.python.org/2/howto/logging.html)

    DEBUG   Detailed information, typically of interest only when diagnosing
        problems.
    INFO    Confirmation that things are working as expected.
    WARNING An indication that something unexpected happened, or indicative of
        some problem in the near future (e.g. 'disk space low'). The software
        is still working as expected.
    ERROR   Due to a more serious problem, the software has not been able to
        perform some function.
    CRITICAL    A serious error, indicating that the program itself may be
        unable to continue running.


History:

    Created on Apr 30, 2014 by Andy Green

    Module level configuration options updated 27 April 2015 by Andy Green

    Extensive modifications to better handle interactions with other logging
    systems, namely Django, 23 March 2016 by Andy Green


@author: agreen
'''

import os
import logging
import types


logging_configured = False
external_loggers = False
file_handler = None
console_handler = None

# Setting complement_existing_loggers to True will make this add to any existing
# handlers at the module level, otherwise this package will not set up handlers
# if there are any already present.
complement_existing_loggers = False

PACKAGE_LOGGER_NAME = 'sami'

# Copy some things from parent package
DEBUG = logging.DEBUG
WARN = logging.WARN
WARNING = logging.WARNING
ERROR = logging.ERROR
INFO = logging.INFO
VDEBUG = 9
VVDEBUG = 8

logging.addLevelName(9, 'VDEBUG')
logging.addLevelName(8, 'VVDEBUG')

def configure_logging():
    """Configure logging for this package and return the package level logger"""

    global logging_configured, complement_existing_loggers, external_loggers

    package_log = logging.getLogger(PACKAGE_LOGGER_NAME)

    if logging_configured:
        return package_log

    # Set up logging for the package. Note that any modules who set up logging
    # below this package logger (using the `slogging.getLogger(__name__)`
    # functionality) will propogate messages to this logger.
    #

    if not complement_existing_loggers and len(package_log.handlers) > 0:
        print("slogging not set up because existing log handlers found: %s" % package_log.handlers)
        logging_configured = True
        external_loggers = True
        return

    # NOTE: The logging level should not be set here, but rather at the top of
    # each file. The setting below ensures that this logger will catch all
    # requests from child loggers.
    package_log.setLevel(0)


    # The code below should only ever be called once, so we raise an exception if this doesn't seem to be the case.
    # if len(package_log.handlers) > 0:
    #     raise Exception("Logging configured more than once for 'samiDB'")

    formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)s %(funcName)s: %(message)s')

    # Default is to log messages to the
    if 'FIDIA_LOG_FILE' in os.environ:
        filename = os.environ['FIDIA_LOG_FILE']
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        package_log.addHandler(file_handler)

    logging_configured = True

    return package_log

def add_file(self, filename, override=False):
    """Add a file handler to this module's logger which writes to the given filename.

    The file will only contain messages from the module corresponding to the
    logger (no parent or sub-modules).

    """

    # assert isinstance(self, logging.Logger)

    if external_loggers and not override:
        return

    formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)s %(funcName)s: %(message)s')

    class LogFilter:
        """Filters log messages not created by the provided logger name."""
        def __init__(self, logger_name):
            self.logger_name = logger_name
        def filter(self, record):
            return record.name == self.logger_name

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(LogFilter(self.name))
    self.addHandler(file_handler)

def enable_console_logging(self, override=False):
    """Add console logging to this module's logger.

    If external logging is available and this module is not set to
    `complement_existing_loggers`, then nothing is done unless `override=True`.
    """
    global console_handler, external_loggers

    if external_loggers and not override:
        return

    if console_handler is None:
        formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)s %(funcName)s: %(message)s')

        console_handler = logging.StreamHandler()
        console_handler.setLevel(0)
        console_handler.setFormatter(formatter)

    if console_handler not in self.handlers:
        self.addHandler(console_handler)

def disable_console_logging(self):
    """Remove console logging previously added using `enable_console_logging`."""
    if len(self.handlers) == 0:
        return

    global console_handler

    if console_handler is None:
        return

    for hndlr in self.handlers:
        if hndlr is console_handler:
            self.removeHandler(hndlr)


def vdebug(self, msg, *args, **kwargs):
    """Debug function for verbose debugging"""
    self.log(VDEBUG, msg, *args, **kwargs)


def vvdebug(self, msg, *args, **kwargs):
    """Debug function for very verbose debugging."""
    self.log(VVDEBUG, msg, *args, **kwargs)


def getLogger(name):
    """Get the logger for `name`, adding some special convenience functions and setup if required."""
    package_log = configure_logging()

    if PACKAGE_LOGGER_NAME != name[:len(PACKAGE_LOGGER_NAME)]:
        package_log.warn('Logging setup request for non-%s-package member "%s"', PACKAGE_LOGGER_NAME, name)

    log = logging.getLogger(name)

    # Add some convenience functions to the log object.
    # See http://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object
    log.add_file = types.MethodType(add_file, log)
    log.enable_console_logging = types.MethodType(enable_console_logging, log)
    log.disable_console_logging = types.MethodType(disable_console_logging, log)
    # log.set_console_level = types.MethodType(set_console_level, log)

    log.vdebug = types.MethodType(vdebug, log)
    log.vvdebug = types.MethodType(vvdebug, log)

    return log
        