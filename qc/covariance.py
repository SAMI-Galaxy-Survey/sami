import numpy as np
import astropy.io.fits as pf

import itertools

def read_full_covariance(path):
    """Return a full 50x50x2048x5x5 covariance array."""
    hdulist = pf.open(path)
    covar_cut = hdulist['COVAR'].data
    header = hdulist['COVAR'].header
    n_wave_out = hdulist[0].header['NAXIS3']
    wave_out = np.arange(n_wave_out)
    n_covar = header['COVAR_N']
    covar_loc = [header['COVARLOC_{}'.format(i+1)] for i in xrange(n_covar)]
    dim = [n_wave_out]
    dim.extend(header['NAXIS{}'.format(i)] for i in xrange(4, 0, -1))
    covar = np.zeros(dim)
    product_args = [xrange(n) for n in dim[1:]]
    for i, j, k, l in itertools.product(*product_args):
        covar[:, i, j, k, l] = np.interp(
            wave_out, covar_loc, covar_cut[:, i, j, k, l])
    return covar

def sum_variance(variance, covariance, x=None, y=None):
    """Sum the variance between spaxels, accounting for covariance."""
    if x is None and y is None:
        x, y = np.meshgrid(np.arange(variance.shape[1]),
                           np.arange(variance.shape[2]))
        x = np.ravel(x)
        y = np.ravel(y)
    if x is None or y is None:
        raise ValueError('sum_variance requires x and y, or neither')
    coords = zip(x, y)
    variance_out = np.zeros(variance.shape[0])
    for (x_i, y_i) in coords:
        for (delta_x, delta_y) in itertools.product(
                xrange(-2, 3), xrange(-2, 3)):
            if (x_i + delta_x, y_i + delta_y) in coords:
                covar_rel = covariance[:, delta_x + 2, delta_y + 2, x_i, y_i]
                variance_extra = variance[:, x_i, y_i] * covar_rel
                good = np.isfinite(variance_extra)
                variance_out[good] += variance_extra[good]
    return variance_out

def compare_variance_list(path_pair_list):
    return np.array([[compare_variance(path) for path in path_pair]
                     for path_pair in path_pair_list])

def compare_variance(path):
    """Compare variance with and without accounting for covariance."""
    var = pf.getdata(path, 'VARIANCE')
    covar = read_full_covariance(path)
    x, y = np.meshgrid(0.5*(np.arange(50)-24.5), 0.5*(np.arange(50)-24.5))
    radii = 0.5 * (np.arange(15) + 1)
    var_right = np.zeros(len(radii))
    var_wrong = np.zeros(len(radii))
    for i_rad, radius in enumerate(radii):
        x_keep, y_keep = np.where((x**2 + y**2) < radius**2)
        var_right[i_rad] = np.median(sum_variance(var, covar, x_keep, y_keep))
        var_wrong[i_rad] = np.median(np.nansum(var[:, x_keep, y_keep], 1))
    return var_right, var_wrong
