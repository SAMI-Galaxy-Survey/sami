"""
Tests for SAMI Cubing routines.

Run all tests using run_tests().

Individual tests described with each function.
"""

# Absolute Imports
import numpy as np
from numpy.random import standard_normal
from scipy.special import erfc
from random import randint
import cProfile
import timeit
import datetime

# Relative Imports
from .. import utils
from .cubing import *


def dummy_ifu():
    """Returns a dummy IFU object which can have arbitrary data inserted."""
    
    return utils.IFU('/Users/agreen/Research/sami_survey/data/april/sampling_test/10G1F2/12apr20037red.fits', '63027', flag_name=True)
    
def speed_test():
    """Time the standard code on a cutdown cube."""
    
    n_obs = 7
    # Number of observations.
    
    lambda_size = 100
    # Number of wavelength slices
    
    # First, we create a set of n_obs mock observations. 
    ifu_list = []
    for i in xrange(n_obs):
        ifu = dummy_ifu()
        
        ifu.data = standard_normal(size=( np.shape(ifu.data)[0], lambda_size)) + 100
        # Random data with stddev = 1 and mean = 100
        
        ifu.var = np.empty_like(ifu.data)
        ifu.var[:] = 1 
        
        print(np.shape(ifu.var))
        
        ifu_list.append(ifu)
    
    start_time = datetime.datetime.now()
    cProfile.runctx('f(ifu_list,offsets=None)',
                    {'f':dithered_cube_from_rss, 'ifu_list':ifu_list},
                    {},
                    filename='dithered_cube_from_rss.pstats')
    print("Wall time: ", (datetime.datetime.now() - start_time).seconds, " seconds")

def hot_pixel_clipping(n_obs=7, n_hot=10,verbose=False):
    """Test that hot pixels are clipped.
    
    This creates a small number of very hot pixels in otherwise almost uniform
    random data, and then checks that they are correctly removed.
    """
    
    # These must match the actual numbers used in the cubing code.
    # @TODO: These should be passed to the cubing code to ensure consistency.
    drop_size = 1.6
    output_size = 0.5
    
    lambda_size = 20
    # Number of wavelength slices

    # First, we create a set of n_obs mock observations. 
    ifu_list = []
    for i in xrange(n_obs):
        ifu = dummy_ifu()
        
        ifu.data = standard_normal(size=( np.shape(ifu.data)[0], lambda_size)) + 100
        # Random data with stddev = 1 and mean = 100

        ifu_list.append(ifu)

    # Run the null test on the data
    data_cube, var_cube, weight_cube, diagnostic_info = dithered_cube_from_rss(ifu_list,offsets=None)
    
    before = diagnostic_info['unmasked_pixels_before_sigma_clip']
    after = diagnostic_info['unmasked_pixels_after_sigma_clip']

    expected_rej = n_obs * np.asarray(ifu_list[0].data.shape).prod() * erfc(5.0/np.sqrt(2))

    if (before - after > expected_rej) or (before - after < 0):
        print("Failed test: hot_pixel_clipping 1")
        print("    The sigma clipping removed {0} pixels, expected {1}.".format(
            before - after, expected_rej))
    elif verbose:
        print("Passed test: hot_pixel_clipping 1")
        print("    The sigma clipping removed {0} pixels, expected {1}.".format(
            before - after, expected_rej))
        


    # Test 2:
    #
    #     Now we add in some hot pixels, and check that the expected number of
    #     pixels are masked from the output
    for i in xrange(n_obs):

        for noise_i in xrange(n_hot):
            ifu_list[i].data[
                randint(0,np.shape(ifu.data)[0] - 1), 
                randint(0,np.shape(ifu.data)[1] - 1)] = 10000.0        
    
    data_cube, var_cube, weight_cube, diagnostic_info = dithered_cube_from_rss(ifu_list,offsets=None)
    
    before = diagnostic_info['unmasked_pixels_before_sigma_clip']
    after = diagnostic_info['unmasked_pixels_after_sigma_clip']

    n_masked_actual = before - after
    n_masked_expected = n_obs * n_hot * 3.14 * ((drop_size + 1.2*output_size)/2)**2 / (output_size**2)
    # NOTE: The factor or 1.2*output_size is empirically determined, but seems
    # reasonable. The actual right answer will depend on the ratio
    # drop_size/output_size

    if (np.abs(n_masked_actual - n_masked_expected) > 0.1*n_masked_expected):
        print("Failed test: hot_pixel_clipping 2")
        print("    Expected number of pixels clipped: {0}, actual number clipped: {1}".format(
            n_masked_expected,
            before - after))
    elif verbose:
        print("Passed test: hot_pixel_clipping 2")
        print("    Expected number of pixels clipped: {0}, actual number clipped: {1}".format(
            n_masked_expected,
            before - after))
        

            
            
            