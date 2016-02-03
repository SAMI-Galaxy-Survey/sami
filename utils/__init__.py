"""
Various utility functions used by other parts of the package.
"""

try:
	# If available, use the compiled C++ code for calculating drizzle overlaps
    import cCirc as circ
except:
	# The compiled C++ code is not available; fall back to the python version
    print("Using slower circle mapping code for drizzling.")
    import circ

# Bring module namespaces up to the package level.
from ifu import *
from other import *

