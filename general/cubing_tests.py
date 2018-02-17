"""
Tests for SAMI Cubing routines.

Individual tests described with each function. Currently available:

speed_test() prints time taken to cube 100 wavelength slices

hot_pixel_clipping() checks that artifically inserted hot pixels (e.g. cosmic
rays) are correctly clipped

dar_correction_test() cubes a star and fits its position at different
wavelengths to check that it doesn't shift around

The tests were developed by Andy Green. The DAR correction test is related
to the QC analysis in section 5.3.3 of the EDR paper (Allen et al 2015) but
is designed to check the code hasn't screwed up rather than actually measure
the accuracy.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

# Absolute Imports
import numpy as np
from numpy.random import standard_normal
from scipy.special import erfc
from random import randint
import cProfile
import timeit
import datetime
import os

# Package imports
import sami.samifitting as samifitting

# Relative Imports
from .. import utils
from .cubing import *
from .. import config
from ..utils.other import *

# Test data directory. Default assumes that it is in subdir "test_data" at the
# same level as the sami package.
#test_data_dir = os.path.dirname(__file__) + '/../../test_data/'
#test_data_dir = '/Users/agreen/Research/sami_survey/data/april/sampling_test/10G1F2/'

testing = True

def dummy_ifu():
    """Returns a dummy IFU object which can have arbitrary data inserted."""
    
    return utils.IFU(test_data_dir +'12apr10036red.fits', '63027', flag_name=True)
    
def speed_test():
    """Time the standard code on a cutdown cube."""
    
    n_obs = 7
    # Number of observations.
    
    lambda_size = 100
    # Number of wavelength slices
    
    # First, we create a set of n_obs mock observations. 
    ifu_list = []
    for i in range(n_obs):
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
    
    # The following lines produce some nice graphics if the user has gprof2dot.py 
    ### print("Creating profile graphics: dithered_cube_from_rss.png")
    ### os.system("gprof2dot.py -f pstats dithered_cube_from_rss.pstats |dot -Tpng -o dithered_cube_from_rss.png")

def hot_pixel_clipping(n_obs=7, n_hot=10,verbose=False):
    """Test that hot pixels are clipped.
    
    This creates a small number of very hot pixels in otherwise almost uniform
    random data, and then checks that they are correctly removed.
    """
    
    # These must match the actual numbers used in the cubing code.
    # @TODO: These should be passed to the cubing code to ensure consistency.
    drop_size = 1.6
    output_size = 0.5
    
    pixel_ratio = 3.14 * ((drop_size + 1.2*output_size)/2)**2 / (output_size**2)
    # This is the approximate number of output pixels each input pixel maps onto.
    # NOTE: The factor or 1.2*output_size is empirically determined, but seems
    # reasonable. The actual right answer will depend on the ratio
    # drop_size/output_size
    
    lambda_size = 20
    # Number of wavelength slices

    # First, we create a set of n_obs mock observations. 
    ifu_list = []
    for i in range(n_obs):
        ifu = dummy_ifu()
        
        ifu.data = standard_normal(size=( np.shape(ifu.data)[0], lambda_size)) + 100
        # Random data with stddev = 1 and mean = 100

        ifu_list.append(ifu)

    # Run the null test on the data
    data_cube, var_cube, weight_cube, diagnostic_info = dithered_cube_from_rss(ifu_list,offsets=None)
    
    before = diagnostic_info['unmasked_pixels_before_sigma_clip']
    after = diagnostic_info['unmasked_pixels_after_sigma_clip']

    expected_rej = n_obs * np.asarray(ifu_list[0].data.shape).prod() * erfc(5.0/np.sqrt(2)) * pixel_ratio

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
    for i in range(n_obs):

        for noise_i in range(n_hot):
            ifu_list[i].data[
                randint(0,np.shape(ifu.data)[0] - 1), 
                randint(0,np.shape(ifu.data)[1] - 1)] = 10000.0        
        
    data_cube, var_cube, weight_cube, diagnostic_info = dithered_cube_from_rss(ifu_list,offsets=None)
    
    before = diagnostic_info['unmasked_pixels_before_sigma_clip']
    after = diagnostic_info['unmasked_pixels_after_sigma_clip']

    n_masked_actual = before - after
    n_masked_expected = n_obs * n_hot * pixel_ratio + expected_rej
    
    


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
        
def dar_correction_test(verbose=False):
    """DAR is working correctly
    
    This uses a few planes from real star images to ensure everything is working
    in DAR without needing to do a whole 2048 slice cube.
    
    """
    
    
    
    # These must match the actual numbers used in the cubing code.
    # @TODO: These should be passed to the cubing code to ensure consistency.
    drop_size = 1.6
    output_size = 0.5
        
    # Take every lambda_step slice from the observations to generate test data
    lambda_step = 100

    # First, we create a set of n_obs mock observations. 
    ifu_list = [
        utils.IFU(test_data_dir + "12apr10036red.fits", 3, flag_name=False),
        utils.IFU(test_data_dir + "12apr10037red.fits", 3, flag_name=False),
        utils.IFU(test_data_dir + "12apr10038red.fits", 3, flag_name=False),
        utils.IFU(test_data_dir + "12apr10039red.fits", 3, flag_name=False),
        utils.IFU(test_data_dir + "12apr10040red.fits", 3, flag_name=False),
        utils.IFU(test_data_dir + "12apr10041red.fits", 3, flag_name=False),
        utils.IFU(test_data_dir + "12apr10042red.fits", 3, flag_name=False),
        ]
 
    n_obs = len(ifu_list)
 
    for i in range(n_obs):
        ifu_list[i].data = ifu_list[i].data[:,100:2000:lambda_step]
        ifu_list[i].var = ifu_list[i].var[:,100:2000:lambda_step]
        ifu_list[i].lambda_range = ifu_list[i].lambda_range[100:2000:lambda_step]
 
    # Run the test on the data
    data_cube, var_cube, weight_cube, diagnostic_info = \
        dithered_cube_from_rss(ifu_list,offsets=None)
    
    output_size = data_cube.shape
    
    tests_passed = True
    
    for i_obs in range(n_obs):
    
        print("\nTesting frame {}".format(i_obs))
    
        # Compute the output centroid plane by plane:
        offsets = []
        for i_slice in range(output_size[2]):
            
            # Convert fibre positions to arcseconds (approx)
            xpos = diagnostics.DAR.xfib[i_obs,:,i_slice] * 15.22/1000.0
            ypos = diagnostics.DAR.yfib[i_obs,:,i_slice] * 15.22/1000.0
    
            
            com_distr=utils.comxyz(xpos,ypos,ifu_list[i_obs].data[:,i_slice])
            
            mask = np.isfinite(np.ravel(ifu_list[i_obs].data[:,i_slice]))
    
            start =[np.nansum(ifu_list[i_obs].data[:,i_slice])/len(mask),
                    com_distr[0], 
                    com_distr[1], 
                    5, 0.0]
    
            gf = samifitting.TwoDGaussFitter(
                                             start,
                                             (np.ravel(xpos))[mask],
                                             (np.ravel(ypos))[mask],
                                             (np.ravel(ifu_list[i_obs].data[:,i_slice]))[mask]
                                             )
            fitting.fibre_integrator(gf, 1.6)
            try:
                gf.fit()
                if not np.allclose(gf.p,start):
                    print("Residual Offset: ({0:.4}, {1:.4}), norm: {2:.4}".format(gf.p[1],gf.p[2],np.linalg.norm(gf.p[1:3]) ) )
                    offsets.append( [gf.p[1], gf.p[2]] )
                else:
                    print("Fit failed for slice {}".format(i_slice))
            except:
                print("Fit failed for slice {}".format(i_slice))
    
        offsets = np.asarray(offsets)
        residual = mad(offsets,axis=0)
        if not ((residual < 0.1).all):
            print("Failed test: dar_correction")
            print("    Residual N-S: {1:.3}, Residual E-W: {0:.3}".format(
                residual[0], residual[1]))
            tests_passed = False
            
 
    if tests_passed:
        print("\nPassed test: dar_correction")
   
        
            
