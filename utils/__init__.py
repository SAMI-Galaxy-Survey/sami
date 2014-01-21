
import logging
logger = logging.getLogger(__name__)

try:
    logger.debug("Attempting import of cCirc (C version of circle code)")
    import cCirc as circ
except:
    logger.debug("Using slower circle mapping code for drizzling.")
    import circ

# Bring module namespaces up to the package level.
logger.debug("Importing all from sami.utils.ifu...")
from ifu import *
logger.debug("Importing all from sami.utils.other...")
from other import *

