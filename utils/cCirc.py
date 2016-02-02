"""
Wrapper for the C++ drizzle overlap code.
"""

import ctypes as C
import numpy as np
import os.path

# Load the shared library
libcm = C.CDLL(os.path.dirname(__file__) + "/libcCirc.so")

# Specify the arguments our function takes:
#  First argument is a 1D array of doubles.  It must be contiguous in memory.
#  Second argument is a regular C long.
#  Third argument is a pointer to a double.
libcm.weight_map.argtypes = [C.c_long, C.c_long, C.c_double, C.c_double, C.c_double, np.ctypeslib.ndpointer(dtype='d', ndim=1, flags='C_CONTIGUOUS')]

def resample_circle(nx, ny, xc, yc, r):
  output = np.zeros(nx*ny)
  libcm.weight_map(nx, ny, xc, yc, r, output)
  return output.reshape(ny, nx)

if __name__ == "__main__":
  import pylab as plt
  o = resample_circle(1000, 1000, 500, 500, 200)
  plt.imshow(o, origin='lower')
  plt.show()
  #np.savetxt("t", o)
