'''
Created on Nov 12, 2013

@author: agreen
'''
import logging
import os

# First, set up the SAMI root logger, and add the handlers
logger = logging.getLogger('sami')
logger.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)s %(funcName)s: %(message)s')

# create file handler which logs even debug messages
try:
    logfile = os.environ['SAMI_LOG_FILE']
except:
    # Fail gracefully if this is not defined, and simply don't write a log
    # file.
    pass
finally:
    _file_handler = logging.FileHandler(logfile)
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(formatter)
    logger.addHandler(_file_handler)

# create console handler with a higher default log level
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.ERROR)
_console_handler.setFormatter(formatter)
logger.addHandler(_console_handler)

