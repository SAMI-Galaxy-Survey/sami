import numpy as np
import astropy.io.fits as pf

import itertools

def read_norm_covariance(hdulist):
    """Return a full 50x50x2048x5x5 normalised covariance array."""
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

# def sum_variance(variance, covariance, x=None, y=None):
#     """Sum the variance between spaxels, accounting for covariance."""
#     if x is None and y is None:
#         x, y = np.meshgrid(np.arange(variance.shape[1]),
#                            np.arange(variance.shape[2]))
#         x = np.ravel(x)
#         y = np.ravel(y)
#     if x is None or y is None:
#         raise ValueError('sum_variance requires x and y, or neither')
#     coords = zip(x, y)
#     variance_out = np.zeros(variance.shape[0])
#     for (x_i, y_i) in coords:
#         for (delta_x, delta_y) in itertools.product(
#                 xrange(-2, 3), xrange(-2, 3)):
#             if (x_i + delta_x, y_i + delta_y) in coords:
#                 covar_rel = covariance[:, delta_x + 2, delta_y + 2, x_i, y_i]
#                 variance_extra = variance[:, x_i, y_i] * covar_rel
#                 good = np.isfinite(variance_extra)
#                 variance_out[good] += variance_extra[good]
#     return variance_out

def compare_variance_list(path_pair_list):
    return np.array([[compare_variance(path) for path in path_pair]
                     for path_pair in path_pair_list])

def compare_variance(path):
    """Compare variance with and without accounting for covariance."""
    hdulist = pf.open(path)
    var = hdulist['VARIANCE'].data
    # covar = read_norm_covariance(hdulist)
    x, y = np.meshgrid(0.5*(np.arange(50)-24.5), 0.5*(np.arange(50)-24.5))
    radii = 0.5 * (np.arange(15) + 1)
    var_right = np.zeros(len(radii))
    var_wrong = np.zeros(len(radii))
    for i_rad, radius in enumerate(radii):
        mask = (x**2 + y**2) < radius**2
        x_keep, y_keep = np.where(mask)
        # var_right[i_rad] = np.median(sum_variance(var, covar, x_keep, y_keep))
        var_right[i_rad] = np.median(bin_hdulist(hdulist, mask)[1])
        var_wrong[i_rad] = np.median(np.nansum(var[:, x_keep, y_keep], 1))
    hdulist.close()
    return var_right, var_wrong




def bin_hdulist(hdulist, mask):
    """
    Return the binned spectrum and variance.

    The inputs are an open HDUList object and a pixel mask of 0s and 1s.
    """
    c_data      = hdulist['Primary'].data
    # c_hdr       = hdulist['Primary'].header
    # c_var       = hdulist['VARIANCE'].data
    c_weight    = hdulist['WEIGHT'].data
    # c_covar     = hdulist['COVAR'].data
    # c_covar_hdr = hdulist['COVAR'].header
    
    # c_var_w = c_var*(c_weight**2)
    c_var_f_w = full_covar(hdulist)
    
    c_data_w = c_data*(c_weight**1)
    
    bin_x, bin_y = np.where(mask != 0)
    
    spec_data_w, spec_var_w = bin_data(
        c_data_w*(mask), c_var_f_w*(mask**2), bin_x, bin_y)

    sum_weight = np.nansum(c_weight[:, bin_x, bin_y], axis=1)
    spec_data = spec_data_w / sum_weight
    spec_var = spec_var_w / sum_weight**2
    
    return spec_data, spec_var

def full_covar(hdulist):
    """Return the full covariance array, after multiplying by variance."""
    
    #Reconstruct the covariance array
    norm_covar = read_norm_covariance(hdulist)
    var = hdulist['VARIANCE'].data
    covar = np.zeros(norm_covar.shape)
    for i in range(0,2048):
        for j in range(0,50):
            for k in range(0,50):
                covar[i,:,:,j,k] = var[i,j,k] * norm_covar[i,:,:,j,k]
    
    return covar

def bin_data(data, covar, binx, biny):
    
    cov = np.zeros((data.shape[0], len(binx), len(biny))) #covariance matrix
    for i in range(0,len(binx)):
        for j in range(0,len(biny)):
            if abs(binx[i]-binx[j]) <= 2 and abs(biny[i]-biny[j]) <= 2:
                cov[:, i, j] = covar[:, binx[j]-binx[i]+2, biny[j]-biny[i]+2,
                                     binx[i], biny[i]]
    data_bin = np.nansum(data[:, binx, biny], axis=1)
    var_bin = np.nansum(cov, axis=(1, 2))
    
    return data_bin, var_bin
