'''
Created on Nov 12, 2013

@author: agreen
'''
import logging

# Initialise the logger:

logger = logging.getLogger('sami')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
#fh = logging.FileHandler('spam.log')
#fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(levelname)s %(filename)s:%(lineno)s %(funcName)s: %(message)s')

#fh.setFormatter(formatter)
_console_handler.setFormatter(formatter)

# add the handlers to the logger
#logger.addHandler(fh)
logger.addHandler(_console_handler)

