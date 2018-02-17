"""
Package for assorted SAMI-related tasks.

This package contains code for observing with SAMI and reducing the data. For
the general user, the important modules are:

	* sami.update_csv for probe assignment during observing
	* sami.general.display for useful plots during observing
	* sami.utils for assorted utility functions
	* sami.centroid for tasks related to the position/size of observed objects
	* sami.manager for data reduction

The other modules are normally not used directly, but are used by those listed
above.
"""

import warnings

__all__ = ['utils', 'samifitting', 'update_csv', 'manager', 'dr', 'qc']

# sami package init file
# import the modules at package level
from . import utils
from . import samifitting
from . import update_csv
from . import manager
from .manager import Manager
from . import dr
from . import qc

from .log import logger

# the config file which contains some constants and stuff
from . import config

# Bring all subpackage modules up to the package name space.
from .general import *
from .observing import *
from .sdss import *


# Disable some numerical warnings:
# We get lots of invalid value warnings arising because of divide by zero errors.
warnings.filterwarnings('ignore', r'invalid value', RuntimeWarning)
