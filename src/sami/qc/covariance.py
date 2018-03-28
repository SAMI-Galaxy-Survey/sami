"""
Functions for handling the covariance as stored in SAMI data cubes.

The functions can read the data stored in the COVAR extension and
extrapolate to a full covariance array, and do simple binning of the data
(but see also sami.dr.binning).

compare_variance() calculates the binned variance with and without
covariance, to see how strong its effect is.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import astropy.io.fits as pf

import itertools

def read_norm_covariance(hdulist):
    """Return full Npix x Npix x Nwav x Ncov x Ncov normalised covariance."""
    covar_cut = hdulist['COVAR'].data
    header = hdulist['COVAR'].header
    n_wave_out = hdulist[0].header['NAXIS3']
    wave_out = np.arange(n_wave_out)
    n_covar = header['COVAR_N']
    covar_loc = [header['COVARLOC_{}'.format(i+1)] for i in range(n_covar)]
    dim = [n_wave_out]
    dim.extend(header['NAXIS{}'.format(i)] for i in range(4, 0, -1))
    covar = np.zeros(dim)
    product_args = [range(n) for n in dim[1:]]
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
#                 range(-2, 3), range(-2, 3)):
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
    variance = hdulist['VARIANCE'].data
    weight = hdulist['WEIGHT'].data
    n_pix = hdulist[0].header['NAXIS1']
    # Pixel scale in arcseconds
    pix_scale = 3600.0 * np.abs(hdulist[0].header['CDELT1'])
    # Construct coordinates along a single axis
    x_tmp = pix_scale * (np.arange(n_pix) - (n_pix - 1)/2.0)
    # Expand that to a grid of coordinate values
    x, y = np.meshgrid(x_tmp, x_tmp)
    radii = 0.5 * (np.arange(15) + 1)
    mask_list = [(x**2 + y**2) < radius**2 for radius in radii]
    var_right = [np.median(var) for spec, var in 
                 bin_hdulist_multi(hdulist, mask_list)]
    var_right = np.array(var_right)
    var_wrong = np.zeros(len(radii))
    for i_mask, mask in enumerate(mask_list):
        x_keep, y_keep = np.where(mask)
        var_wrong_spec = (
            np.nansum(variance[:, x_keep, y_keep] * 
                      weight[:, x_keep, y_keep]**2, 1) /
            np.nansum(weight[:, x_keep, y_keep], 1)**2)
        var_wrong[i_mask] = np.median(var_wrong_spec)
    hdulist.close()
    return var_right, var_wrong




def bin_hdulist(hdulist, mask):
    """
    Return the binned spectrum and variance.

    The inputs are an open HDUList object and a pixel mask of 0s and 1s.
    """
    c_data = hdulist['PRIMARY'].data
    c_weight = hdulist['WEIGHT'].data
    c_var_f_w = full_covar(hdulist)
    
    bin_x, bin_y = np.where(mask)    
    spec_data, spec_var = bin_data_weights(
        c_data, c_var_f_w, c_weight, bin_x, bin_y)

    return spec_data, spec_var

def bin_hdulist_multi(hdulist, mask_list):
    """
    Return a list of binned spectra and variances, one for each mask.
    """
    c_data = hdulist['PRIMARY'].data
    c_weight = hdulist['WEIGHT'].data
    c_var_f_w = full_covar(hdulist)
    
    result = []
    for mask in mask_list:
        bin_x, bin_y = np.where(mask)    
        spec_data, spec_var = bin_data_weights(
            c_data, c_var_f_w, c_weight, bin_x, bin_y)
        result.append((spec_data, spec_var))

    return result

def full_covar(hdulist):
    """
    Return the full covariance array, after multiplying by weighted
    variance.
    """
    
    #Reconstruct the covariance array
    norm_covar = read_norm_covariance(hdulist)
    var = hdulist['VARIANCE'].data
    weight = hdulist['WEIGHT'].data
    covar = np.zeros(norm_covar.shape)
    header = hdulist[0].header
    for i in range(header['NAXIS3']):
        for j in range(header['NAXIS2']):
            for k in range(header['NAXIS1']):
                covar[i,:,:,j,k] = (var[i,j,k] * norm_covar[i,:,:,j,k] *
                                    weight[i,j,k]**2)
    return covar

def bin_data_weights(data, covar, weight, bin_x, bin_y):
    """Bin the data in a weighted fashion."""
    data_w = data * weight
    spec_data_w, spec_var_w = bin_data(data_w, covar, bin_x, bin_y)
    sum_weight = np.nansum(weight[:, bin_x, bin_y], axis=1)
    spec_data = spec_data_w / sum_weight
    spec_var = spec_var_w / sum_weight**2
    return spec_data, spec_var

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
