from __future__ import print_function

"""
Wrapper for the C++ drizzle overlap code.

History
-------

Created by Jon Nielsen in 2012
Updated for the new cubing algorithm by Francesco D'Eugenio 16/02/2017

Notes
-----
This module contains a testing function. At the moment it requires that the libraries path be hardcoded (FDE).
"""

import ctypes as C
import numpy as np
import os
from glob import glob

# Load the shared library
__path_to_lib__ = os.path.join(os.getcwd(), os.path.dirname(__file__))
__library_file__ = glob(os.path.join(__path_to_lib__, 'circ/weight_map*.so'))
libcm = C.CDLL(__library_file__[0])

# Specify the arguments our function takes:
#  First argument is a 1D array of doubles.  It must be contiguous in memory.
#  Second argument is a regular C long.
#  Third argument is a pointer to a double.
libcm.weight_map.argtypes = [
    C.c_long, C.c_long, C.c_double, C.c_double, C.c_double,
    np.ctypeslib.ndpointer(dtype='d', ndim=1, flags='C_CONTIGUOUS')]



def resample_circle(nx, ny, xc, yc, r, *args):
  output = np.zeros(nx*ny)
  libcm.weight_map(nx, ny, xc, yc, r, output)
  return output.reshape(ny, nx)




# Specify the arguments our function takes:
#  First argument is a 1D array of doubles.  It must be contiguous in memory.
#  Second argument is a regular C long.
#  Third argument is a pointer to a double.
libcm.weight_map_Gaussian.argtypes = [
     C.c_long, C.c_long, C.c_double, C.c_double, C.c_double, C.c_double,
     np.ctypeslib.ndpointer(dtype='d', ndim=1, flags='C_CONTIGUOUS')]



def inteGrauss2d(nx, ny, xc, yc, sigma, n_sigma):
  output = np.zeros(nx*ny)
  libcm.weight_map_Gaussian(nx, ny, xc, yc, sigma, n_sigma, output)
  return output.reshape(ny, nx)



if __name__ == "__main__":

    import time
    import pylab as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    plt.switch_backend('pdf')

    try:
        from texttable import Texttable

        def print_table(headers, times):
            table = Texttable()
            headers = ['Method'] + headers
            times = [u'Elapsed time (\u03bcs)'] + times
            table.add_rows([[h, t] for h, t in zip(headers, times)])
            table.set_cols_align(["l", "c"])

            print(table.draw())

    except ImportError:
        warnings.warn('To print formatted output, please install `texttable`')

        def print_table(headers, times):
            for h, e in zip(headers, elapsed):
                print(h, '{:8.2f}'.format(e/float(n_iterations)*1.e6),
                      u'\u03bc'+'s')


    from circ import resample_circle as rc_py
    from circ import inteGrauss2d as ig_py
    rc_C = resample_circle
    ig_C = inteGrauss2d

    # This circle show that the python implementation of cCirc wraps around,
    # which is dumb
    # (50, 50, 27.527636662069785, 2.716882503265406, 4.9423403572267581,
    #  2.0454945513296701))

    # Define some representative cases.
    # Each has (xpix, ypix, xc, yc, [r/sigma], [clip])
    tests = ((10, 10, 5, 5, 1, 3),
             (10, 10, 3, 5, 1, 3),
             (10, 10, 3.1, 5.7, 1, 3))
    tests = ((50, 50, 27.527636662069785, 23.716882503265406, 4.9423403572267581, 2.0454945513296701),
             (50, 50, 27.527636662069785, 2.716882503265406, 4.9423403572267581, 2.0454945513296701)) # <=

    # Define some circles to plot. Notice that for `resample_circle` the last
    # element of each tuple is ignored, while the penultimate is the radius.
    # for `inteGrauss2d` the last element defines the truncation and the
    # penultimate element defines the standard deviation of the Gaussian.
    tests = (
        (10, 10, 5., 1.0, 2., 2.),
        (10, 10, 5.65, 1.2, 2.31, 2.001))
    tests = ((50, 50, 9., 9., 1.6, 5.),)

    # Number of iterations for the benchmark.
    n_iterations = 10000

    n_tests = len(tests)

    plt.clf()
    fig = plt.figure(figsize=(18, 3 * n_tests))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_tests, 6),
                     cbar_location='top', cbar_mode='edge',
                     cbar_pad='0%', cbar_size='5%', direction='row',
                     axes_pad=(0, 0))
 
    wmaps = [[rc_py(*t), rc_C(*t), ig_py(*t), ig_C(*t)] for t in tests]
    wmaps = [[w[0], w[1], w[0]-w[1], w[2], w[3], w[2]-w[3]] for w in wmaps]
    wmaps = sum(wmaps, []) # Flatten the list.
 
    # Reformat `tests` so that we can iterate over it for each subplot.
    tests = ((t,)*6 for t in tests) # Repeat each test for every plot that uses it.
    tests = sum(tests, ())          # Flatten this tuple.
    

    y_labels = ['$\mathrm{Test\;'+ '{}'.format(n) + '}$'
                for n in range(1, n_tests+1)]
    x_labels = ['$\mathrm{resample\_circ \; (python)}$',
                '$\mathrm{resample\_circ \; (C++)}$',
                '$\mathrm{residuals}$',
                '$\mathrm{inteGrauss2d \; (python)}$',
                '$\mathrm{inteGrauss2d \; (C++)}$',
                '$\mathrm{residuals}$']

    for n, (wm, t, ax) in enumerate(zip(wmaps, tests, grid)):
        img = ax.imshow(wm, origin='lower', interpolation='none')
        ax.plot([t[2]-.5], [t[3]-.5], 'mx')
        if n >= (n_tests - 1) * 6: # Trigger the colourbar.
            cbar = ax.cax.colorbar(img)
            cbar.solids.set_edgecolor('face')

        if n % 6 == 0:              # We are at the first column.
            ax.set_ylabel(y_labels[n // 6])
        if n // 6 >= (n_tests - 1): # We are at the bottom row.
            ax.set_xlabel(x_labels[n % 6])
 
    #plt.subplots_adjust(wspace=0., hspace=0.)
    plt.savefig('test_ccircs.pdf')
 
 
 
    # Benchmarking.
    
    # Create random tests to prevent system caching of interemdiate results in
    # the C implementation - that would be unfair.
 
    grid_size = np.repeat(50, n_iterations)
    xc, yc = np.random.uniform(10, 40, (2, n_iterations))
    radius = np.random.uniform(1, 5, n_iterations)
    n_sigma = np.random.uniform(1, 5, n_iterations)
 
    tests = [
        (xpix, ypix, x, y, r, ns)
        for xpix, ypix, x, y, r, ns in zip(
            grid_size, grid_size, xc, yc, radius, n_sigma)]
 
    start_time, elapsed, results = [], [], []
 
    print('Benchmarking function `resample_circle` (Top hat cubing, python)...')
    start_time.append(time.time())
    results.append(np.array([rc_py(*t) for t in tests]))
    elapsed.append(time.time() - start_time[-1])
 
    print('Benchmarking function `resample_circle` (Top hat cubing, C++)...')
    start_time.append(time.time())
    results.append(np.array([rc_C(*t) for t in tests]))
    elapsed.append(time.time() - start_time[-1])
 
    print('Benchmarking function `inteGrauss2d` (Gaussian cubing, python)...')
    start_time.append(time.time())
    results.append(np.array([ig_py(*t) for t in tests]))
    elapsed.append(time.time() - start_time[-1])
 
    print('Benchmarking function `inteGrauss2d` (Gaussian cubing, C++)...')
    start_time.append(time.time())
    results.append(np.array([ig_C(*t) for t in tests]))
    elapsed.append(time.time() - start_time[-1])
 
    print('Summary:')
    headers = ['`resample_circle` (Top hat cubing, python)',
               '`resample_circle` (Top hat cubing, C++)',
               '`inteGrauss2d` (Gaussian cubing, python)',
               '`inteGrauss2d` (Gaussian cubing, C++)']
    print_table(headers, elapsed)
