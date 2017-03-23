"""
Various utility functions used by other parts of the package.
"""

import warnings

try:
	# If available, use the compiled C++ code for calculating drizzle overlaps
    import cCirc as circ
except:
    warning_message = (
        '\n' + '*'*70 + '\n'
      + 'The compiled C++ version of `circ` is not available. Falling back to\n'
      + ' the python implementation.\n\n'
      + 'We recommend that you use the C++ implementation. In order to do so\n'
      + ' please exit python, then navigate to the directory where the SAMI\n'
      + 'pipeline is located (enter sami.__path__ in the python interpreter\n'
      + 'to find the directory).'
      + 'Now run the command \n'
      + '`make` on the terminal. If the output is:\n'
      + '    `g++ -O3 -shared -fPIC -c cCirc.cc` (or equivalent)\n'
      + '    `g++ -shared -o libcCirc.so cCirc.o` (or equivalent)\n'
      + 'and if you can find the file `libcCirc.so` in the subdirectory \n'
      + '`utils` of the sami path, you can open the python interpreter and \n'
      + 'try to import the sami manager again. If problems persists, please\n'
      + 'see a doctor.\n\n'
      + 'Notes applicable should you decide to continue the current session:\n'
      + ' 1) This implementation is slower\n'
      + ' 2) The drizzling implementation in python has a bug, I recommend not\n'
      + '    to use it. Try to compile the C++ version\n'
      + '*'*70 + '\n')

    warnings.warn(warning_message)
    
	# The compiled C++ code is not available; fall back to the python version
    import circ

# Bring module namespaces up to the package level.
from ifu import *
from other import *

