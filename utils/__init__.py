
try:
    import cCirc as circ
except:
    print("Using slower circle mapping code for drizzling.")
    import circ

# Bring module namespaces up to the package level.
from ifu import *
from other import *

