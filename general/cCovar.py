from __future__ import print_function

"""
Wrapper for the C++ function that calculates the covariance matrix.

History
-------

* 16/02/2017 - Created by Francesco D'Eugenio

Author
------

Francesco D'Eugenio  <fdeugenio@gmail.com>

Notes
-----
This module contains a testing function. At the moment it requires that the libraries path be hardcoded (FDE).

"""

import ctypes as C
import numpy as np
import os.path

# Load the shared library
try:
    libccovar = C.CDLL(os.path.join(os.path.dirname(__file__), "libcCovar.so"))
except:
    pass
    #libccovar = C.CDLL(os.path.join('/home/franz/software/dev/sami-software-dev/dr0.10/utils', 'libcCirc.so'))

# Specify the arguments our function takes:
#  First argument is a regular C long.
#  Second argument is a regular C long.
#  Third argument is a regular C long.
#  Fourth argument is a regular C long.
#  Fifth argument is a 1D array of doubles. It must be contiguous in memory.
#  Sixth argument is a 1D array of doubles.
#  Seventh argument is a pointer to a double.

libccovar.create_covar_matrix.argtypes = [
    C.c_long, C.c_long, C.c_long, C.c_long,
    np.ctypeslib.ndpointer(dtype='d', ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype='d', ndim=1),
    np.ctypeslib.ndpointer(dtype='d', ndim=1, flags='C_CONTIGUOUS')]


def create_covar_matrix(overlap_array, variances, covarRad=2):

  nx, ny, n_fibres = overlap_array.shape

  nc = 2 * covarRad + 1

  overlap_array = overlap_array.ravel()

  output = np.zeros(nx*ny*nc*nc, dtype=np.float64)

  libccovar.create_covar_matrix(covarRad, nx, ny, n_fibres,
                                 overlap_array, variances, output)
  return output.reshape(nx, ny, nc, nc)



if __name__ == "__main__":

    # import pickle
    import time
    import warnings

    from covar import create_covar_matrix_original
    from covar import create_covar_matrix_vectorised

    try:
        from texttable import Texttable

        def print_table(headers, times):
            table = Texttable()
            headers = ['Method'] + headers 
            times = ['Elapsed time (ms)'] + times 
            table.add_rows([[h, t] for h, t in zip(headers, times)])
            table.set_cols_align(["l", "c"])
            print(table.draw())

    except ImportError:
        warnings.warn('To print formatted output, please install `texttable`')

        def print_table(headers, times):
            for h, e in zip(headers, elapsed):
                print(h, '{:8.2f}'.format(e/float(n_iterations)*1.e3), 'ms')

    # Declare the methods being tested.
    ccmo = create_covar_matrix_original
    ccmv = create_covar_matrix_vectorised
    ccmc = create_covar_matrix

    # These arrays are realistic inputs that I saved on disk.
    # covariance_array = pickle.load(open('temp2', 'r'))[0]
    # variances, overlap_array = pickle.load(open('temp', 'r'))

    n_iterations = 1000
    n_iterations = 10
    covarRad = 2
    nc = 2 * covarRad + 1
    nx, ny = 10, 10
    nx, ny = 50, 50
    n_fibres = 9
    n_fibres = 427
    np.random.seed(10042016)

    # Create the input data.
    overlap_array = [np.random.random((nx, ny, n_fibres))
                     for _ in range(n_iterations)]
    variances = [np.random.random(n_fibres) for _ in range(n_iterations)]
    
    covariances_o = np.empty((n_iterations, nx, ny, nc, nc))
    covariances_v = np.empty((n_iterations, nx, ny, nc, nc))
    covariances_c = np.empty((n_iterations, nx, ny, nc, nc))

    start_time, elapsed = [], []

    print('Benchmarking function `create_covar_matrix_original` (python, non vectorised)...')
    start_time.append(time.time())
    covariances_o = np.array(
        [ccmo(o, v) for o,v in zip(overlap_array, variances)])
    elapsed.append(time.time() - start_time[-1])

    print('Benchmarking function `create_covar_matrix_vectorised` (python, vectorised)...')
    start_time.append(time.time())
    covariances_v = np.array(
        [ccmv(o, v) for o,v in zip(overlap_array, variances)])
    elapsed.append(time.time() - start_time[-1])

    print('Benchmarking function `create_covar_matrix` (C++, vectorised)...')
    start_time.append(time.time())
    covariances_c = np.array(
        [ccmc(o, v) for o,v in zip(overlap_array, variances)])
    elapsed.append(time.time() - start_time[-1])

    print('Summary:')
    headers = [
        '`create_covar_matrix_original` (python, non vectorised)',
        '`create_covar_matrix_vectorised` (python, vectorised)',
        '`create_covar_matrix` (C++, vectorised)']
    print_table(headers, elapsed)

    print('Differences (should always be 0)')
    print(np.nanmedian(covariances_c - covariances_o))
    print(np.nanstd(covariances_c - covariances_o))
    print(np.nanmedian(covariances_v - covariances_o))
    print(np.nanstd(covariances_v - covariances_o))
