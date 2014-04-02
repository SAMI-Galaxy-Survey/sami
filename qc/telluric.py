from ..dr import telluric

import astropy.io.fits as pf
import numpy as np

def compare_star_field(path_pair, output=None):
    """Compare measurements between all stars in a field."""
    n_hexa = 13
    n_pixel = pf.getval(path_pair[1], 'NAXIS1')
    result = np.zeros((n_hexa, 4, n_pixel))
    for i_hexa in xrange(n_hexa):
        telluric.derive_transfer_function(path_pair, use_probe=i_hexa+1)
        result[i_hexa, :, :] = pf.getdata(path_pair[1], 'FLUX_CALIBRATION')
    if output is not None:
        pf.PrimaryHDU(result).writeto(output)
    return result
