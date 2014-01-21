# sami package init file
# import the modules at package level
from log import logger


logger.debug("Loading sami.utils module...")
import utils
logger.debug("Loading sami.samifitting module...")
import samifitting
logger.debug("Loading sami.update_csv module...")
import update_csv
logger.debug("Loading sami.manager module...")
import manager
logger.debug("Loading sami.dr module...")
import dr



# the config file which contains some constants and stuff
logger.debug("Loading sami.config module...")
import config

# Bring all subpackage modules up to the package name space.
logger.debug("Loading sami.general submodules...")
from general import *
logger.debug("Loading sami.observing submodules...")
from observing import *
logger.debug("Loading sami.sdss submodules...")
from sdss import *

logger.debug("sami/__init__ completed")