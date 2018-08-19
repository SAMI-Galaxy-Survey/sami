"""
This module covers functions required to create cubes from a dithered set
of RSS frames. See Sharp et al (2015) for a detailed description of the
cubing algorithm.

The hard work is done by dithered_cube_from_rss which takes a list of IFU
objects as input and returns a datacube, variance and other information.
Various levels of wrappers give other ways of accessing this function -
the manager uses dithered_cube_from_rss_wrapper.

A few things to be aware of:

* Alignment should be done before calling the cubing. If it hasn't, the
  cubing will fall back to measuring the offsets itself, but this is much
  worse than the proper alignment code.
* By default, the absolute astrometry is measured by comparison to SDSS
  images. However, this doesn't work properly, so until someone fixes it
  you should always ask for nominal astrometry by setting nominal=True.
* The covariance information would be too large to store at every
  wavelength slice, so is currently only stored when the drizzle data is
  updated. James Allen is currently (Dec 2015) looking into alternative
  storage strategies to allow the drizzle information to be updated at
  every slice.
* Ned Taylor is currently (Dec 2015) investigating alternative cubing
  methods that may replace some of all of this code.

This module is in need of refactoring to break up some very long
functions.


The most likely command a user will want to run is one of:

   dithered_cubes_from_rss_files
   dithered_cubes_from_rss_list

These functions are wrappers for file input/output, with the actual work
being done in dithered_cube_from_rss. These three functions are briefly
described below.

**dithered_cubes_from_rss_files**

** notes from RGS
>cd ~/SAMI/
>ipython
>import sami
> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1.txt")   - no actual output made
> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1.txt",write=True,clip=False,drop_factor=0.5,suffix="_NoClip",covar_mode="none",nominal=True,root="testing/",overwrite=True)

> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1.txt",write=True,clip=False,drop_factor=0.5,suffix="_NoClip",covar_mode="none",nominal=True,root="testing_small/",overwrite=True,objects=["36894","47286"])
> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1.txt",write=True,clip=True,drop_factor=1.0,suffix="__Full_Clip",covar_mode="none",nominal=True,root="testing_small/",overwrite=True,objects=["36894","47286"])
> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1.txt",write=True,clip=True,drop_factor=0.5,suffix="_Clip",covar_mode="none",nominal=True,root="testing_small/",overwrite=True,objects=["36894","47286"])
> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1.txt",write=True,clip=True,drop_factor=0.5,suffix="_FullClip",covar_mode="none",nominal=True,root="testing_small/",overwrite=True,objects=["36894","47286"])

> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1blue.txt",write=True,clip=False,drop_factor=0.5,suffix="_ReducedDrop_NoClip",covar_mode="none",nominal=True,root="testing_small/",overwrite=True,objects=["36894","47286"])
> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1blue.txt",write=True,clip=True,drop_factor=0.5,suffix="_ReducedDrop_ClipWithReducedDrop",covar_mode="none",nominal=True,root="testing_small/",overwrite=True,objects=["36894","47286"])
> sami.general.cubing.dithered_cubes_from_rss_files("testfiles1blue.txt",write=True,clip=True,drop_factor=0.5,suffix="_ReducedDrop_ClipWithFullDrop",covar_mode="none",nominal=True,root="testing_small/",overwrite=True,objects=["36894","47286"])


INPUTS:
inlist - a file containing the names of the RSS files to be cubed. These file
namesare one per line in this file.

objects - a python list containing the names of the galaxies to make cubes for.
The default is the special value (string) 'all', which means all galaxies in the
input RSS files will be cubed.

size_of_grid - The size of the square output grid in pixels. The default is 50 pixels.

output_pix_size_arcsec - The size of the pixels within the square output grid in
arcseconds. The default is 0.5 arcseconds.

drop_factor - The size reduction factor for the input fibres when they are drizzled
onto the output grid.

clip - Clip the data when combining. This should provide cleaner output cubes. Default
is True.

plot - Make diagnostic plots when creating cubes. This seems to be defunct as no plots are
made. Default is True.

write - Write data cubes to file after creating them. The default is False. (Should change?)

suffix - A suffix to add to the output file name. Should be a string. Default is a null string.

nominal - Use the nominal tabulated object positions when determining WCS. Default is False and
a full comparison with SDSS is done.

root - Root directory for writing files to. Should be a string. Default is null string.

overwrite - Overwrite existing files with the same output name. Default is False.

covar_mode - Option are 'none', 'optimal' or 'full'. Default is 'optimal'.

OUTPUTS:
If write is set to True two data cubes will be produced for each object cubed. These are
written by default to a new directory, created within the working directory, with the name
of the object.

**dithered_cubes_from_rss_list**
INPUTS:

files: a python list of files to be cubed. 

All other inputs as above!

**dithered_cube_from_rss**
INPUTS:

ifu_list - A list of IFU objects. In most cases this is passed from
dithered_cubes_from_rss_list.

############################################################################################

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pylab as py
import numpy as np
import scipy as sp

from scipy import integrate

import astropy.io.fits as pf
import astropy.wcs as pw

import itertools
import os
import sys
import datetime
import warnings
warnings.simplefilter('always', DeprecationWarning)
warnings.simplefilter('always', ImportWarning)
from glob import glob

# This switch selects wich implementation to use to compute the covariance
# matrix. The two implementations differ only in their technical aspects, but
# the results are the same (within the numerical precision). The C
# implementation is the fastest.
try:
    from .cCovar import create_covar_matrix
except ImportError:
    # Here we use the original implementation from James (or Nic?). The module
    # contains a python, vectorised implementation that is significantly faster,
    # but still slower than the C implementation. Should you prefer the python,
    # vectorised implementation, please run:
    # from covar import create_covar_matrix_vectorised as create_covar_matrix
    warn_message = ('Failed to import the C version of covar: using '
        + ' the python implementation (this takes longer).\n'
        + 'To use the C implementation, please navigate to the folder where '
        + 'the sami pipeline is located, and run `Make` from the terminal. ')
    warnings.warn(warn_message, ImportWarning)
    from .covar import create_covar_matrix_original as create_covar_matrix


try:
    from tqdm import tqdm # `tqdm` required to show progress meter bars.
    use_old_progress_meter = False
except ImportError:
    warning_message = ('\n' + '*'*80 + '\n'
        '* `tqdm` module is not available. No progress bars will be shown.\n'
        + '* We currently recommend to install `tqdm`. This can be done e.g. by \n'
        + '* `~# pip install tqdm`\n'
        + '*'*80 + '\n')
    warnings.warn(warning_message, DeprecationWarning, stacklevel=2)
    use_old_progress_meter = True
    tqdm = lambda x: x    # If `tqdm` is not available, overload it to identity.

# Cross-correlation function from scipy.signal (slow)
from scipy.signal import correlate

# Import the sigma clipping algorithm from astropy
from astropy.stats import sigma_clip

# Attempt to import bottleneck to improve speed, but fall back to old routines
# if bottleneck isn't present
try:
    from bottleneck import nanmedian, nansum, nanmean
except:
    from numpy import nanmedian, nanmean
    nansum = np.nansum

# Utils code.
from .. import utils
from .. import samifitting as fitting
from ..utils.mc_adr import DARCorrector, parallactic_angle, zenith_distance
from .. import diagnostics

# importing everything defined in the config file
from ..config import *

# WCS code
from . import wcs

# Function for reading a filter response
from ..qc.fluxcal import read_filter, get_coords

import code

from pdb import set_trace

# This simple switch allows to test the alternative cubing methods. Do not change.
if 1:
    try:
        from ..utils.cCirc import resample_circle as compute_weights
    except ImportError: # Assume no compiled version of the C++ library exists. Switch to python.
        warn_message = ('Failed to import the C version of compute_weights: using'
            + ' the python implementation (this takes longer).\n'
            + 'To use the C implementation, please navigate to the folder where '
            + 'the sami pipeline is located, and run `Make` from the terminal. ')
        from ..utils.circ import resample_circle as compute_weights
    cubing_method = 'Tophat'
else:
    warning_message = 'The Gaussian weighting scheme is not fully assessed.'
    warnings.warn(warning_message, FutureWarning, stacklevel=2)
    try:
        from ..utils.cCirc import inteGrauss2d as compute_weights
    except ImportError: # Assume no compiled version of the C++ library exists. Switch to python.
        from ..utils.circ import inteGrauss2d as compute_weights
    cubing_method = 'Gaussian'

# Some global constants:
HG_CHANGESET = utils.hg_changeset(__file__)

epsilon = np.finfo(np.float).eps
# Store the value of epsilon for quick access.

def get_object_names(infile):
    """Get the object names observed in the file infile."""

    # Open up the file and pull out list of observed objects.
    try:
        table=pf.getdata(infile, 'FIBRES_IFU')
    except KeyError:
        table=pf.getdata(infile, 'MORE.FIBRES_IFU')
    names=table.field('NAME')

    # Find the set of unique values in names
    names_unique=list(set(names))

    # Pick out the object names, rejecting SKY and empty strings
    object_names_unique = [s for s in names_unique if ((s.startswith('SKY')==False)
                            and (s.startswith('Sky')==False)) and len(s)>0]

    return object_names_unique

def _get_probe_all(files, name, verbose=True):
    """Obtain a list of probe names, and write it to the cube header."""

    probes = [_get_probe_single(fl, name, verbose=verbose) for fl in files]
    probes = np.unique(probes)

    #if len(probes)==1:
    #    return probes[0]
    #else:
    #    message = ('Object {} appears to have dithers from multiple IFU ' \
    #        + 'probes.\n The input files are {}').format(name, files)
    #    raise IOError(message)

    probes_string = np.array2string(probes,separator=',')[1:-1]

    return probes_string

def _get_probe_single(infile, object_name, verbose=True):
    """ This should read in the RSS files and return the probe number the object was observed in"""

    # First find the IFU the object was returned in
    hdulist=pf.open(infile)
    fibre_table=hdulist['FIBRES_IFU'].data
    
    mask_name=fibre_table.field('NAME')==object_name

    table_short=fibre_table[mask_name]

    # Now find the ifu the galaxy was observed with in this file
    ifu_array=table_short.field('PROBENUM')

    # The ifu
    ifu=ifu_array[0]

    if verbose==True:
        print("Object", object_name, "was observed in IFU", ifu, "in file", infile)

    hdulist.close()

    # Return the probe number
    return ifu

def dar_correct(ifu_list, xfibre_all, yfibre_all, method='simple',update_rss=False):
    """Update the fiber positions as a function of wavelength to reflect DAR correction.
    
    """

    n_obs, n_fibres, n_slices = xfibre_all.shape

    # Set up the differential atmospheric refraction correction. Each frame
    # requires it's own correction, so we create a list of DARCorrectors.
    dar_correctors = []
                      
    for obs in ifu_list:
        darcorr = DARCorrector(method=method)
    
        ha_offset = obs.ra - obs.meanra  # The offset from the HA of the field centre

        ha_start = obs.primary_header['HASTART'] + ha_offset
        # The header includes HAEND, but this goes very wrong if the telescope
        # slews during readout. The equation below goes somewhat wrong if the
        # observation was paused, but somewhat wrong is better than very wrong.
        ha_end = ha_start + (obs.exptime / 3600.0) * 15.0
    
        if hasattr(obs, 'atmosphere'):
            # Take the atmospheric parameters from the file, as measured
            # during telluric correction
            darcorr.temperature = obs.atmosphere['temperature']
            darcorr.air_pres = obs.atmosphere['pressure']
            darcorr.water_pres = obs.atmosphere['vapour_pressure']
            darcorr.zenith_distance = np.rad2deg(obs.atmosphere['zenith_distance'])
        else:
            # Get the atmospheric parameters from the fibre table header
            darcorr.temperature = obs.fibre_table_header['ATMTEMP'] 
            darcorr.air_pres = obs.fibre_table_header['ATMPRES'] * millibar_to_mmHg
            #                     (factor converts from millibars to mm of Hg)
            darcorr.water_pres = \
                utils.saturated_partial_pressure_water(darcorr.air_pres, darcorr.temperature) * \
                obs.fibre_table_header['ATMRHUM']
            darcorr.zenith_distance = \
                integrate.quad(lambda ha: zenith_distance(obs.dec, ha),
                               ha_start, ha_end)[0] / (ha_end - ha_start)

        darcorr.hour_angle = 0.5 * (ha_start + ha_end)
    
        darcorr.declination = obs.dec

        dar_correctors.append(darcorr)

        del darcorr # Cleanup since this is meaningless outside the loop.


    wavelength_array = ifu_list[0].lambda_range
    
    # Iterate over wavelength slices
    for l in range(n_slices):
        
        # Iterate over observations
        for i_obs in range(n_obs):
            # Determine differential atmospheric refraction correction for this slice
            dar_correctors[i_obs].update_for_wavelength(wavelength_array[l])
            
            # Parallactic angle is direction to zenith measured north through east.
            # Must move light away from the zenith to correct for DAR.
            dar_x = dar_correctors[i_obs].dar_east * 1000.0 / plate_scale 
            dar_y = dar_correctors[i_obs].dar_north * 1000.0 / plate_scale 
            # TODO: Need to change to arcsecs!

            xfibre_all[i_obs,:,l] = xfibre_all[i_obs,:,l] + dar_x
            yfibre_all[i_obs,:,l] = yfibre_all[i_obs,:,l] + dar_y
            
#             print("DAR lambda: {:5.0f} x: {:5.2f}, y: {:5.2f}, pa : {:5.0f}".format(wavelength_array[l],
#                                                                            dar_x * plate_scale/1000.0,
#                                                                            dar_y * plate_scale/1000.0,
#                                                                            dar_correctors[i_obs].parallactic_angle()))
    if diagnostics.enabled:
        diagnostics.DAR.xfib = xfibre_all
        diagnostics.DAR.yfib = yfibre_all


def dithered_cubes_from_rss_files(inlist, **kwargs):
    """A wrapper to make a cube from reduced RSS files, passed as a filename containing a list of filenames. Only input files that go together - ie have the same objects."""

    # Read in the list of all the RSS files input by the user.
    files=[]
    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])

        files.append(np.str(cols[0]))

    dithered_cubes_from_rss_list(files, **kwargs)
    return

def dithered_cubes_from_rss_list(files, objects='all', **kwargs):
    """A wrapper to make a cube from reduced RSS files, passed as a list. Only input files that go together - ie have the same objects."""
        
    start_time = datetime.datetime.now()

    # Set a few numpy printing to terminal things
    np.seterr(divide='ignore', invalid='ignore') # don't print division or invalid warnings.
    np.set_printoptions(linewidth=120) # can also use to set precision of numbers printed to screen

    # For first file get the names of galaxies observed - assuming these are the same in all RSS files.
    # @TODO: This is not necessarily true, and we should either through an exception or account for it
    if objects=='all':
        object_names=get_object_names(files[0])
    else:
        object_names=objects
    print("--------------------------------------------------------------")
    print("The following objects will be cubed:\n\n")
    for name in object_names:
        print(name, end=' ')
    print("\n--------------------------------------------------------------")
    for name in tqdm(object_names):
        print()
        print("--------------------------------------------------------------")
        print("Cubing object:", name)
        print()
        dithered_cube_from_rss_wrapper(files, name, **kwargs)            
    print("Time dithered_cubes_from_files wall time: {0}".format(datetime.datetime.now() - start_time))
    return

def dithered_cube_from_rss_wrapper(files, name, size_of_grid=50, 
                                   output_pix_size_arcsec=0.5, drop_factor=0.5,
                                   clip=True, plot=True, write=True, suffix='',
                                   nominal=False, root='', overwrite=False,
                                   offsets='file', covar_mode='optimal',
                                   do_dar_correct=True, clip_throughput=True,
                                   update_tol=0.02):
    """Cubes and saves a single object."""
    n_files = len(files)

    ifu_list = []
    
    for filename in files:
        ifu_list.append(utils.IFU(filename, name, flag_name=True))

    if write:
        # First check if the object directory already exists or not.
        directory = os.path.join(root, name)
        try:
            os.makedirs(directory)
        except OSError:
            print("Directory Exists", directory)
            print("Writing files to the existing directory")
        else:
            print("Making directory", directory)

        # Filename to write to
        arm = ifu_list[0].spectrograph_arm            
        outfile_name=str(name)+'_'+str(arm)+'_'+str(n_files)+suffix+'.fits'
        outfile_name_full=os.path.join(directory, outfile_name)

        # Check if the filename already exists
        if os.path.exists(outfile_name_full):
            if overwrite:
                os.remove(outfile_name_full)
            else:
                print('Output file already exists:')
                print(outfile_name_full)
                print('Skipping this object')
                return False

        if overwrite:
            # Check for old cubes that used a different number of files
            for path in glob(os.path.join(
                    directory, str(name)+'_'+str(arm)+'_*'+suffix+'.fits')):
                os.remove(path)

    # Call dithered_cube_from_rss to create the flux, variance and weight cubes for the object.
    #try:
    flux_cube, var_cube, weight_cube, diagnostics, covariance_cube, covar_locs = \
        dithered_cube_from_rss(
            ifu_list, size_of_grid=size_of_grid,
            output_pix_size_arcsec=output_pix_size_arcsec,
            drop_factor=drop_factor, clip=clip, plot=plot,
            offsets=offsets, covar_mode=covar_mode,
            do_dar_correct=do_dar_correct, clip_throughput=clip_throughput,
            update_tol=update_tol)

    #except Exception:
    #    print "Cubing Failed."
    #    continue

    # Write out FITS files.
    if write==True:

        if ifu_list[0].gratid == '580V':
            band = 'g'
        elif ifu_list[0].gratid == '1000R':
            band = 'r'
        elif not nominal:
            # Need an identified band if we're going to do full WCS matching.
            # Should try to work something out, like in scale_cube_pair_to_mag()
            raise ValueError('Could not identify band. Exiting')
        else:
            # When nominal is True, wcs_solve doesn't even look at band
            band = None

        # Equate Positional WCS
        WCS_pos, WCS_flag=wcs.wcs_solve(ifu_list[0], flux_cube, name, band, size_of_grid, output_pix_size_arcsec, plot, nominal=nominal)
        
        # First get some info from one of the headers.
        list1=pf.open(files[0])
        hdr=list1[0].header

        hdr_new = create_primary_header(ifu_list, name, files, WCS_pos, WCS_flag)

        # Define the units for the datacube
        hdr_new['BUNIT'] = ('10**(-16) erg /s /cm**2 /angstrom /pixel', 
                            'Units')

        # Add the drop factor used to the datacube header
        hdr_new['DROPFACT'] = (drop_factor, 'Drizzle drop scaling')

        hdr_new['CBINGMET'] = (cubing_method, 'Method adopted for cubing')

        hdr_new['IFUPROBE'] = (_get_probe_all(files, name, verbose=False),
                               'Id number(s) of the SAMI IFU probe(s)')

        # Create HDUs for each cube.
        
        list_of_hdus = []

        # @NOTE: PyFITS writes axes to FITS files in the reverse of the sense
        # of the axes in Numpy/Python. So a numpy array with dimensions
        # (5,10,20) will produce a FITS cube with x-dimension 20,
        # y-dimension 10, and the cube (wavelength) dimension 5.  --AGreen
        list_of_hdus.append(pf.PrimaryHDU(np.transpose(flux_cube, (2,1,0)), hdr_new))
        list_of_hdus.append(pf.ImageHDU(np.transpose(var_cube, (2,1,0)), name='VARIANCE'))
        list_of_hdus.append(pf.ImageHDU(np.transpose(weight_cube, (2,1,0)), name='WEIGHT'))

        if covar_mode != 'none':
            hdu4 = pf.ImageHDU(np.transpose(covariance_cube,(4,3,2,1,0)),name='COVAR')
            hdu4.header['COVARMOD'] = (covar_mode, 'Covariance mode')
            if covar_mode == 'optimal':
                hdu4.header['COVAR_N'] = (len(covar_locs), 'Number of covariance locations')
                for i in range(len(covar_locs)):
                    hdu4.header['HIERARCH COVARLOC_'+str(i+1)] = covar_locs[i]
            list_of_hdus.append(hdu4)

        # Create HDUs for meta-data
        #metadata_table = create_metadata_table(ifu_list)

        list_of_hdus.append(create_qc_hdu(files, name))
        
        # Put individual HDUs into a HDU list
        hdulist = pf.HDUList(list_of_hdus)

        # Write the file
        print("Writing", outfile_name_full)
        print("--------------------------------------------------------------")
        hdulist.writeto(outfile_name_full)

        # Close the open file
        list1.close()
    return True


def dithered_cube_from_rss(ifu_list, size_of_grid=50, output_pix_size_arcsec=0.5, drop_factor=0.5,
                           clip=True, plot=True, offsets='file', covar_mode='optimal',
                           do_dar_correct=True, clip_throughput=True, update_tol=0.02):
    diagnostic_info = {}

    n_obs = len(ifu_list)
    # Number of observations/files
    
    n_slices = np.shape(ifu_list[0].data)[1]
    # The number of wavelength slices
    
    n_fibres = ifu_list[0].data.shape[0]
    # Number of fibres

    # Create an instance of SAMIDrizzler for use later to create individual overlap maps for each fibre.
    # The attributes of this instance don't change from ifu to ifu.
    overlap_maps=SAMIDrizzler(
        size_of_grid, output_pix_size_arcsec, drop_factor, n_obs * n_fibres,
        update_tol=update_tol)

    ### RGS 27/2/14 We also need a second copy of this IFF we are clipping the data AND drop_factor != 1
    if drop_factor != 1 and clip:
        ### But, check this does not bugger up intermediate calculation elements in the code?
        overlap_maps_fulldrop=SAMIDrizzler(
            size_of_grid, output_pix_size_arcsec, 1, n_obs * n_fibres,
            update_tol=update_tol)
                
    # Empty lists for positions and data.
    xfibre_all = np.empty((n_obs, n_fibres))
    yfibre_all = np.empty((n_obs, n_fibres))
    data_all= np.empty((n_obs, n_fibres, n_slices))
    var_all = np.empty((n_obs, n_fibres, n_slices))
    ifus_all= np.empty(n_obs)

    # The following loop:
    #
    #   2. Uses a Centre of Mass to as an initial guess to gaussian fit the position 
    #      of the object within the bundle
    #   3. Computes the positions of the fibres within each file including an offset
    #      for the galaxy position from the gaussian fit (e.g., everything is now on 
    #      the same coordiante system.
    #

    for j in range(n_obs):

        # Get the data.
        galaxy_data=ifu_list[j]
        
        # Smooth the spectra and median.
        data_smoothed=np.zeros_like(galaxy_data.data)
        for p in range(np.shape(galaxy_data.data)[0]):
            data_smoothed[p,:]=utils.smooth(galaxy_data.data[p,:], 10) #default hanning

        # Collapse the smoothed data over a large wavelength range to get continuum data
        data_med=nanmedian(data_smoothed[:,300:1800], axis=1)

        # Pick out only good fibres (i.e. those allocated as P)
        #
        # @TODO: This will break the subsequent code when we actually break a
        # fibre, as subsequent code assumes 61 fibres. This needs to be fixed.
        goodfibres=np.where(galaxy_data.fib_type=='P')
        x_good=galaxy_data.x_microns[goodfibres]
        y_good=galaxy_data.y_microns[goodfibres]
        data_good=data_med[goodfibres]
    
        # First try to get rid of the bias level in the data, by subtracting a median
        data_bias=np.median(data_good)
        if data_bias<0.0:
            data_good=data_good-data_bias
            
        # Mask out any "cold" spaxels - defined as negative, due to poor
        # throughtput calibration from CR taking out 5577.
        msk_notcold=np.where(data_good>0.0)
    
        # Apply the mask to x,y,data
        x_good=x_good[msk_notcold]
        y_good=y_good[msk_notcold]
        data_good=data_good[msk_notcold]

        # Below is a diagnostic print out.
        #print("data_good.shape: ",np.shape(data_good))

        # Check that we're not trying to use data that isn't there
        # Change the offsets method if necessary
        if offsets == 'file' and not hasattr(galaxy_data, 'x_refmed'):
            print('Offsets have not been pre-measured! Fitting them now.')
            offsets = 'fit'

        if (offsets == 'fit'):
            # Fit parameter estimates from a crude centre of mass
            com_distr=utils.comxyz(x_good,y_good,data_good)
        
            # First guess sigma
            sigx=100.0
        
            # Peak height guess could be closest fibre to com position.
            dist=(x_good-com_distr[0])**2+(y_good-com_distr[1])**2 # distance between com and all fibres.
        
            # First guess Gaussian parameters.
            p0=[data_good[np.sum(np.where(dist==np.min(dist)))], com_distr[0], com_distr[1], sigx, sigx, 45.0, 0.0]
       
            gf1=fitting.TwoDGaussFitter(p0, x_good, y_good, data_good)
            gf1.fit()

            # Adjust the micron positions of the fibres - for use in making final cubes.
            xm=galaxy_data.x_microns-gf1.p[1]
            ym=galaxy_data.y_microns-gf1.p[2]

        elif (offsets == 'file'):
            # Use pre-measured offsets saved in the file itself
            x_shift_full = galaxy_data.x_refmed + galaxy_data.x_shift
            y_shift_full = galaxy_data.y_refmed - galaxy_data.y_shift
            xm=galaxy_data.x_microns-x_shift_full
            ym=galaxy_data.y_microns-y_shift_full
            


        else:
            # Perhaps use this place to allow definition of the offsets manually??
            # Hopefully only useful for test purposes. LF 05/06/2013
            xm=galaxy_data.x_microns - np.mean(galaxy_data.x_microns)
            ym=galaxy_data.y_microns - np.mean(galaxy_data.y_microns)

        if clip_throughput:
            # Clip out fibres that have suspicious throughput values
            bad_throughput = ((galaxy_data.fibre_throughputs < 0.5) |
                              (galaxy_data.fibre_throughputs > 1.5))
            galaxy_data.data[bad_throughput, :] = np.nan
            galaxy_data.var[bad_throughput, :] = np.nan
    
        xfibre_all[j, :] = xm
        yfibre_all[j, :] = ym

        data_all[j, :, :] = galaxy_data.data
        var_all[j, :, :] = galaxy_data.var

        ifus_all[j] = galaxy_data.ifu

    # Scale these up to have a wavelength axis as well
    xfibre_all = xfibre_all.reshape(n_obs, n_fibres, 1).repeat(n_slices,2)
    yfibre_all = yfibre_all.reshape(n_obs, n_fibres, 1).repeat(n_slices,2)
    
    # @TODO: Rescaling between observations.
    #
    #     This may be done here (_before_ the reshaping below). There also may
    #     be a case to do it for whole files, although that may have other
    #     difficulties.


    # DAR Correction
    #
    #     The correction for differential atmospheric refraction as a function
    #     of wavelength must be applied for each file/observation independently.
    #     Therefore, it is applied to the positions of the fibres before the
    #     individual fibres are considered as independent observations
    #
    #     DAR correction is handled by another function in this module, which
    #     updates the fibre positions in place.
    
    if do_dar_correct:
        dar_correct(ifu_list, xfibre_all, yfibre_all)

    # Reshape the arrays
    #
    #     What we are doing is combining the first two dimensions, which are
    #     files and fibres. Effectively, what we will do is treat each fibre in
    #     each file as a completely independent observation for the purposes of
    #     building grided data cube.
    #
    #     old.shape -> (n_obs,            n_fibres, n_slices)
    #     new.shape -> (n_obs * n_fibres, n_slices)
    #     NOTE: the fibre position arrays are simply (n_files * n_fibres), no n_slices dimension.
    xfibre_all = np.reshape(xfibre_all, (n_obs * n_fibres, n_slices) )
    yfibre_all = np.reshape(yfibre_all, (n_obs * n_fibres, n_slices) )
    data_all   = np.reshape(data_all,   (n_obs * n_fibres, n_slices) )
    var_all    = np.reshape(var_all,    (n_obs * n_fibres, n_slices) )

    # Change the units on the input data
    #
    # We assume here that the input data is in units of surface brightness,
    # where the number in each input RSS pixel is the surface brightness in
    # units of the fibre area (e.g., ergs/s/cm^2/fibre). We want the surface
    # brightness in units of the output pixel area.
    fibre_area_pix = np.pi * (fibre_diameter_arcsec/2.0)**2 / output_pix_size_arcsec**2
    data_all = data_all / fibre_area_pix
    var_all = var_all / (fibre_area_pix)**2
    
    # We must renormalise the spectra in order to sigma_clip the data.
    #
    # Because the data is undersampled, there is an aliasing effect when making
    # offsets which are smaller than the sampling. The result is that multiple
    # sub-pixels can have wildly different values. These differences are not
    # physical, but rather just an effect of the sampling, and so we do not want
    # to clip devient pixels purely because of this variation. Renormalising the
    # spectra first allow us to flag devient pixels in a sensible way. See the
    # data reduction paper for a full description of this reasoning.
    data_norm=np.empty_like(data_all)
    for ii in range(n_obs * n_fibres):
        data_norm[ii,:] = data_all[ii,:]/nanmedian( data_all[ii,:])
        

    # Now create a new array to hold the final data cube and build it slice by slice
    flux_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))
    var_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))
    weight_cube=np.empty((size_of_grid, size_of_grid, np.shape(data_all)[1]))
    
    print("data_all.shape: ", np.shape(data_all))

    if clip:
        # Set up some diagostics if you have the clip flag set.
        diagnostic_info['unmasked_pixels_after_sigma_clip'] = 0
        diagnostic_info['unmasked_pixels_before_sigma_clip'] = 0

        diagnostic_info['n_pixels_sigma_clipped'] = []
           
            
    # Load the wavelength solution for the datacubes. 
    #
    # TODO: This should change when the header keyword propagation is improved
    # and we have confirmed that all RSS files are on the same wavelength
    # solution.
    wavelength_array = ifu_list[0].lambda_range


    # Initialise some covariance variables
    overlap_array = 0
    overlap_array_old = 0
    overlap_array_older = 0
    overlap_array_oldest = 0
    recompute_tracker = 1
    recompute_flag = 0
    covariance_array = []
    covariance_slice_locs = []
    
    # This loops over wavelength slices (e.g., 2048).
    for l in tqdm(range(n_slices)):

        # In this loop, we will map the RSS fluxes from individual fibres
        # onto the output grid.
        #
        # Variables with "grid" are on the output grid, and variables with
        # "rss" are in the input fibres space. The "fibres" part of the name
        # shows variables with individual planes for each fibre.
        #
        # e.g., 
        #     np.shape(data_rss_slice)         -> (n_fibres * n_files)
        #     np.shape(data_grid_slice_fibres) -> (outsize, outsize, n_fibres * n_files)
        #     np.shape(data_rss_slice_final)   -> (outsize, outsize)
        
        """ Olden time estimate, we ditched it for `tqdm` which is A)rabic
        B)etter and C)ool.
        """
        if use_old_progress_meter:
            # Estimate time to loop completion, and display to user:
            if (l == 1):
                start_time = datetime.datetime.now()
            elif (l == 10):
                time_diff = datetime.datetime.now() - start_time
                print("Mapping slices onto output grid, wavelength slice by slice...")
                print("Estimated time to complete all {0} slices: {1}".format(
                    n_slices, n_slices * time_diff / 9))
                sys.stdout.flush()
                del start_time
                del time_diff



        # Create pointers to slices of the RSS data for convenience (these are
        # NOT copies)
        norm_rss_slice=data_norm[:,l]
        data_rss_slice=data_all[:,l]
        var_rss_slice=var_all[:,l]

        # Compute drizzle maps for this wavelength slice.
        # Store previous slices drizzle map for optimal covariance approach
        overlap_array_oldest = np.copy(overlap_array_older)
        overlap_array_older = np.copy(overlap_array_old)
        overlap_array_old = np.copy(overlap_array)
        overlap_array = overlap_maps.drizzle(xfibre_all[:,l], yfibre_all[:,l])

        ### RGS 7/2/12 - If a small drop size is being used, AND if clipping is selected
        ### then we make a second overlap array, which maps the full size fibres to the output array
        ### we then use this full size mapping in order to make a smoothed version for clipping bad data points
        ### then we actyally combined only the data on the reduced drop size mapping
        if drop_factor != 1 and clip:            
            #print("Clipping with a small drop_factor, so calculating a full size overlap map.")
            ### This needs a new overlap map instance as that is where the drop size gets set...
            overlap_array_fulldrop = overlap_maps_fulldrop.drizzle(xfibre_all[:,l], yfibre_all[:,l])
        
        #######################################
        # Determine covariance array at either i) all slices (mode = full)
        # or ii) either side of DAR corrected slices (mode = optimal)
        # NB - Code between the #### could be parceled out into a separate function,
        # but because of the number of variables required I've opted to leave it here
        if (l == 0) and (covar_mode != 'none'):
            covariance_array_slice = create_covar_matrix(overlap_array,var_rss_slice)
            s_covar_slice = np.shape(covariance_array_slice)
            if np.nansum(covariance_array_slice == 0):
                covariance_array_slice = np.ones(s_covar_slice)*np.nan
            covariance_array = covariance_array_slice.reshape(np.append(s_covar_slice,1))
            covariance_slice_locs = [0]

        elif (covar_mode == 'optimal') and (recompute_flag == 1):
            covariance_array_slice = create_covar_matrix(overlap_array,var_rss_slice)
            covariance_array_slice = covariance_array_slice.reshape(np.append(s_covar_slice,1))
            covariance_array_slice_prev = create_covar_matrix(overlap_array_old,var_all[:,l-1])
            covariance_array_slice_prev = covariance_array_slice_prev.reshape(np.append(s_covar_slice,1))
            covariance_array_slice_prev2 = create_covar_matrix(overlap_array_older,var_all[:,l-2])
            covariance_array_slice_prev2 = covariance_array_slice_prev2.reshape(np.append(s_covar_slice,1))
            covariance_array_slice_prev3 = create_covar_matrix(overlap_array_oldest,var_all[:,l-3])
            covariance_array_slice_prev3 = covariance_array_slice_prev3.reshape(np.append(s_covar_slice,1))
            covariance_array = np.append(covariance_array,covariance_array_slice_prev3,axis=len(s_covar_slice))
            covariance_array = np.append(covariance_array,covariance_array_slice_prev2,axis=len(s_covar_slice))
            covariance_array = np.append(covariance_array,covariance_array_slice_prev,axis=len(s_covar_slice))
            covariance_array = np.append(covariance_array,covariance_array_slice,axis=len(s_covar_slice))
            covariance_slice_locs.append(l-3)
            covariance_slice_locs.append(l-2)
            covariance_slice_locs.append(l-1)
            covariance_slice_locs.append(l)
            recompute_tracker = overlap_maps.n_drizzle_recompute
            recompute_flag = 0

        #elif (((l%200 == 0) and (l != 0)) or (l == (n_slices-2)) or (l == (n_slices-1))) and (covar_mode != 'none'):
        #    covariance_array_slice = create_covar_matrix(overlap_array,var_rss_slice)
        #    covariance_array_slice = covariance_array_slice.reshape(np.append(s_covar_slice,1))
        #    covariance_array = np.append(covariance_array,covariance_array_slice,axis=len(s_covar_slice))
        #    covariance_slice_locs.append(l)

        elif ((l == (n_slices-2)) or (l == (n_slices-1))) and (covar_mode != 'none'):
            covariance_array_slice = create_covar_matrix(overlap_array,var_rss_slice)
            covariance_array_slice = covariance_array_slice.reshape(np.append(s_covar_slice,1))
            covariance_array = np.append(covariance_array,covariance_array_slice,axis=len(s_covar_slice))
            covariance_slice_locs.append(l)
        
        elif covar_mode == 'full':
            covariance_array_slice = create_covar_matrix(overlap_array,var_rss_slice)
            covariance_array_slice = covariance_array_slice.reshape(np.append(s_covar_slice,1))
            covariance_array = np.append(covariance_array,covariance_array_slice,axis=len(s_covar_slice))
        
        if recompute_tracker != overlap_maps.n_drizzle_recompute:
            recompute_flag = 1

        ##########################################
        
        # Map RSS slices onto gridded slices
        ### rgs 7/2/14 - if cosmic ray clipping is set AND a small drop size is used, then we will need an additional 
        ### map here which shows the the full fibre size mapping in ordero to perform the clipping on that "smoothed data.
        norm_grid_slice_fibres=overlap_array*norm_rss_slice        
        data_grid_slice_fibres=overlap_array*data_rss_slice
        var_grid_slice_fibres=(overlap_array*overlap_array)*var_rss_slice

        if clip:

            ### RGS 7/2/14 - I think all that is needed here, to fix the clipping for reduced drop size
            ### is to run the mask_grid... generation using the FULL drop size overlap_array_FULL
            ### then apply that mask directly to a "data_grid_slice" which uses the reduced drop size overlap_array
            ### However, I need to check that the fibre mapping works for that.
            ### Are the overlap arrays sparse arrays (i.e., full "cubes" with zero where there is no overlap?
            ### Or are they very specific remappings of "pointer like" elements? 

            if drop_factor != 1:
            #if 2 != 2: # kludge to turn this off for testing
            #print("A reduced drop_factor was detected")
            #print("Cliping using the full size drop mask")
                
                # Perform sigma clipping of the data if the clip flag is set.
                n_unmasked_pixels_before_clipping = np.isfinite(overlap_array_fulldrop*data_rss_slice).sum()
                
                # Sigma clip it - pixel by pixel and make a master mask
                # array. Current clipping, sigma=5 and 1 iteration.        
                mask_grid_slice_fibres = sigma_clip_mask_slice_fibres(overlap_array_fulldrop*norm_rss_slice/overlap_array_fulldrop)
                #mask_grid_slice_fibres = sigma_clip_mask_slice_fibres(norm_rss_slice)

            else:
                # Perform sigma clipping of the data if the clip flag is set.
                n_unmasked_pixels_before_clipping = np.isfinite(data_grid_slice_fibres).sum()
            
                # Sigma clip it - pixel by pixel and make a master mask
                # array. Current clipping, sigma=5 and 1 iteration.        
                mask_grid_slice_fibres = sigma_clip_mask_slice_fibres(norm_grid_slice_fibres/overlap_array)

            
            # Below is without the normalised spectra for the clip.
            #mask_grid_slice_fibres = sigma_clip_mask_slice_fibres(data_grid_slice_fibres/weight_grid_slice)
        
            # Apply the mask to the data slice array and variance slice array
            data_grid_slice_fibres[np.logical_not(mask_grid_slice_fibres)] = np.NaN 
            var_grid_slice_fibres[np.logical_not(mask_grid_slice_fibres)] = np.NaN # Does this matter?

            # Record diagnostic information about the number of pixels masked.
            n_unmasked_pixels_after_clipping = np.isfinite(data_grid_slice_fibres).sum()
            diagnostic_info['n_pixels_sigma_clipped'].append(
                n_unmasked_pixels_before_clipping - n_unmasked_pixels_after_clipping
                )
            diagnostic_info['unmasked_pixels_before_sigma_clip'] += n_unmasked_pixels_before_clipping
            diagnostic_info['unmasked_pixels_after_sigma_clip'] += n_unmasked_pixels_after_clipping
            #         print("Pixels Clipped: {0} ({1}%)".format(\
            #             n_unmasked_pixels_before_clipping - n_unmasked_pixels_after_clipping,
            #             (n_unmasked_pixels_before_clipping - n_unmasked_pixels_after_clipping) / float(n_unmasked_pixels_before_clipping)
            #             ))

        # Now, at this stage want to identify ALL positions in the data array
        # (before we collapse it) where there are NaNs (from clipping, cosmics
        # flagged by 2dfdr and bad columns flagged by 2dfdr). This allows the
        # correct weight map to be created for the wavelength slice in question.
        valid_grid_slice_fibres=np.isfinite(data_grid_slice_fibres).astype(int)

        # valid_grid_slice_fibres should now be an array of ones and zeros
        # reflecting where there is any valid input data. We multiply this by
        # the fibre weighting to get the final weighting.
        weight_grid_slice_fibres=overlap_array*valid_grid_slice_fibres

        # Combine (sum) the individual observations. See Section 6.1: "Simple
        # summation" of the data reduction paper. Note that these arrays are
        # "unweighted" cubes C' and V', not the weighted C and V given in the
        # paper.
        data_grid_slice_final = nansum(data_grid_slice_fibres, axis=2) 
        var_grid_slice_final = nansum(var_grid_slice_fibres, axis=2)
        weight_grid_slice_final = nansum(weight_grid_slice_fibres, axis=2)
        
        # Where the weight map is within epsilon of zero, set it to NaN to
        # prevent divide by zero errors later.
        weight_grid_slice_final[weight_grid_slice_final < epsilon] = np.NaN
        
        flux_cube[:, :, l] = data_grid_slice_final
        var_cube[:, :, l] = var_grid_slice_final
        weight_cube[:, :, l] = weight_grid_slice_final

    print("Total calls to drizzle: {}, recomputes: {}, ({}%)".format(
                overlap_maps.n_drizzle, overlap_maps.n_drizzle_recompute,
                float(overlap_maps.n_drizzle_recompute)/overlap_maps.n_drizzle*100.))

    # Finally, divide by the weight cube to remove variations in exposure time
    # (and hence surface brightness sensitivity) from the output data cube.
    flux_cube_unprimed = flux_cube / weight_cube 
    var_cube_unprimed = var_cube / (weight_cube * weight_cube)

    return flux_cube_unprimed, var_cube_unprimed, weight_cube, diagnostic_info, covariance_array, covariance_slice_locs

def sigma_clip_mask_slice_fibres(grid_slice_fibres):
    """Return a mask with outliers removed."""

    med = nanmedian(grid_slice_fibres, axis=2)
    stddev = utils.mad(grid_slice_fibres, axis=2)
     
    # We rearrange the axes so that numpy.broadcasting works in the subsequent
    # operations. See: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    t_grid_slice_fibres = np.transpose(grid_slice_fibres, axes=(2,0,1))
    
    mask = np.transpose( 
            np.less_equal(np.abs(t_grid_slice_fibres - med), stddev * 5 ),
            axes=(1,2,0))
    
    return mask
          
class SAMIDrizzler:
    """Make an overlap map for a single fibre. This is the same at all lambda slices for that fibre (neglecting
    DAR)"""  

    def __init__(self, size_of_grid, output_pix_size_arcsec, drop_factor,
                 n_fibres, n_sigma=5., update_tol=0.02):
        """Construct a new SAMIDrizzler isntance with the necessary information.
        
        Parameters
        ----------
        size_of_grid: the number of pixels along each dimension of the 
            square output pixel grid
        output_pix_size_arcsec: required output pixel size in arcseconds
        drop_factor: fraction to multiply the true fibre diameter by
        n_fibres: the total number of unique fibres which will be 
            mapped onto the grid (usually n_fibres * n_obs) 
        update_tol: how far, as a fraction of a pixel, to allow the footprint
            to wander before recalculating
        
        """

        # The input values
        self.pix_size_arcsec = output_pix_size_arcsec
        self.pix_size_micron = output_pix_size_arcsec * (1000.0 / plate_scale)
        # Set the size of the output grid - should probably be calculated somehow.
        self.output_dimension = size_of_grid

        self.plate_scale = plate_scale    # (in arcseconds per mm)
        self.drop_diameter_arcsec = fibre_diameter_arcsec * drop_factor
        
        # Drop dimensions in units of output pixels
        self.drop_diameter_pix = self.drop_diameter_arcsec / self.pix_size_arcsec
        self.drop_area_pix = np.pi * (self.drop_diameter_pix / 2.0) ** 2
        
        self.drizzle_update_tol = update_tol * self.pix_size_micron

        # Output grid abscissa in microns
        self.grid_coordinates_x = (np.arange(self.output_dimension) - self.output_dimension / 2.0) * self.pix_size_micron
        self.grid_coordinates_y = (np.arange(self.output_dimension) - self.output_dimension / 2.0) * self.pix_size_micron

        # Empty array for all overlap maps - i.e. need one for each fibre!
        self.drop_to_pixel = np.empty((self.output_dimension, self.output_dimension, n_fibres))
        self.pixel_coverage = np.empty((self.output_dimension, self.output_dimension, n_fibres))

        # These are used to cache the arguments for the last drizzle.
        self._last_drizzle_x = np.zeros(1)
        self._last_drizzle_y = np.zeros(1)        

        # Number of times drizzle has been called in this instance
        self.n_drizzle = 0
        
        # Number of times drizzle has been recomputed in this instance
        self.n_drizzle_recompute = 0

        # If using the Gaussian drizzling, `n_sigma` is the number of
        # standard deviations inside which the Gaussian is integrated. Beyond
        # a radius equal to `n_sigma` * `drop_diameter_pix` the Gaussian is set
        # to zero.
        self.n_sigma = n_sigma

    def single_overlap_map(self, fibre_position_x, fibre_position_y):
        """Compute the mapping from a single input drop to output pixel grid.
        
        (drop_fraction, pixel_fraction) = single_overlap_map(fibre_position_x, fibre_position_y)
        
        Parameters
        ----------
        fibre_position_x: (float) The grid_coordinates_x-coordinate of the fibre.
        fibre_position_y: (float) The grid_coordinates_y-coordinate of the fibre.
        
        Returns
        -------
        
        This returns a tuple of two arrays. Both arrays have the same dimensions
        (that of the output pixel grid). 
        
        drop_fraction: (array) The fraction of the input drop in each output pixel.
        pixel_fraction: (arra) The fraction of each output pixel covered by the drop.
                
        """

        # Map fibre positions onto pixel positions in the output grid.
        xfib = (fibre_position_x - self.grid_coordinates_x[0]) / self.pix_size_micron
        yfib = (fibre_position_y - self.grid_coordinates_y[0]) / self.pix_size_micron

        # Create the overlap map from the circ.py code. This returns values in
        # the range [0,1] that represent the amount by which each output pixel
        # is covered by the input drop.
        #
        # @NOTE: The circ.py code returns an array which has the x-coodinate in
        # the second index and the y-coordinate in the first index. Therefore,
        # we transpose the result here so that the x-cooridnate (north positive)
        # is in the first index, and y-coordinate (east positive) is in the
        # second index.
        weight_map = np.transpose(
            compute_weights(
                self.output_dimension, self.output_dimension, 
                xfib, yfib,
                self.drop_diameter_pix / 2.0, self.n_sigma))

        return weight_map

    def drizzle(self, xfibre_all, yfibre_all):
        """Compute a mapping from fibre drops to output pixels for all given fibre locations."""
            
        # Increment the drizzle counter
        self.n_drizzle = self.n_drizzle + 1

        if (np.allclose(xfibre_all,self._last_drizzle_x, rtol=0,atol=self.drizzle_update_tol) and
            np.allclose(yfibre_all,self._last_drizzle_y, rtol=0,atol=self.drizzle_update_tol)):
            # We've been asked to recompute an asnwer that is less than the tolerance to recompute
            return self.drop_to_pixel
        else:
            self.n_drizzle_recompute = self.n_drizzle_recompute + 1
        
        for i_fibre, xfib, yfib in zip(itertools.count(), xfibre_all, yfibre_all):
    
            # Feed the grid_coordinates_x and grid_coordinates_y fibre positions to the overlap_maps instance.
            drop_to_pixel_fibre = self.single_overlap_map(xfib, yfib)
    
            # Padding with NaNs instead of zeros (do I really need to do this? Probably not...)
            drop_to_pixel_fibre[np.where(drop_to_pixel_fibre < epsilon)]=np.nan
    
            self.drop_to_pixel[:,:,i_fibre]=drop_to_pixel_fibre
    
        self._last_drizzle_x = xfibre_all
        self._last_drizzle_y = yfibre_all
    
        return self.drop_to_pixel

def create_primary_header(ifu_list,name,files,WCS_pos,WCS_flag):
    """Create a primary header to attach to each cube from the RSS file headers"""

    hdr = ifu_list[0].primary_header
    fbr_hdr = ifu_list[0].fibre_table_header

    # Create the wcs. Note 2dfdr uses non-standard CTYPE and CUNIT, so these
    # are not copied
    wcs_new=pw.WCS(naxis=3)
    wcs_new.wcs.crpix = [WCS_pos["CRPIX1"], WCS_pos["CRPIX2"], hdr['CRPIX1']]
    wcs_new.wcs.cdelt = np.array([WCS_pos["CDELT1"], WCS_pos["CDELT2"], hdr['CDELT1']])
    wcs_new.wcs.crval = [WCS_pos["CRVAL1"], WCS_pos["CRVAL2"], hdr['CRVAL1']]
    wcs_new.wcs.ctype = [WCS_pos["CTYPE1"], WCS_pos["CTYPE2"], "AWAV"]
    wcs_new.wcs.equinox = 2000
    wcs_new.wcs.radesys = 'FK5'
            
    # Create a header
    hdr_new=wcs_new.to_header(relax=True)
    #hdr_new.update('WCS_SRC',WCS_flag,'WCS Source')
    hdr_new['WCS_SRC'] = (WCS_flag, 'WCS Source')

    # Putting in the units by hand, because otherwise astropy converts
    # 'Angstrom' to 'm'. Note 2dfdr uses 'Angstroms', which is non-standard.
    hdr_new['CUNIT1'] = WCS_pos['CUNIT1']
    hdr_new['CUNIT2'] = WCS_pos['CUNIT2']
    hdr_new['CUNIT3'] = 'Angstrom'
            
    # Add the name to the header
    hdr_new['NAME'] = (name, 'Object ID')
    # Need to implement a database specific-specific OBSTYPE keyword to indicate galaxies
    # *NB* This is different to the OBSTYPE keyword already in the header below
    
    # Determine total exposure time and add to header
    total_exptime = 0.
    for ifu in ifu_list:
        total_exptime+=ifu.primary_header['EXPOSED']
    hdr_new['TOTALEXP'] = (total_exptime, 'Total exposure (seconds)')

    # Add the mercurial changeset ID to the header
    hdr_new['HGCUBING'] = (HG_CHANGESET, 'Hg changeset ID for cubing code')
    # Need to implement a global version number for the database
    
    # Put the RSS files into the header
    for num in range(len(files)):
        rss_key='HIERARCH RSS_FILE '+str(num+1)
        rss_string='Input RSS file '+str(num+1)
        hdr_new[rss_key] = (os.path.basename(files[num]), rss_string)

    # Extract header keywords of interest from the metadata table, check for consistency
    # then append to the main header

    primary_header_keyword_list = ['DCT_DATE','DCT_VER','DETECXE','DETECXS','DETECYE','DETECYS',
                                   'DETECTOR','XPIXSIZE','YPIXSIZE','METHOD','SPEED','READAMP','RO_GAIN',
                                   'RO_NOISE','ORIGIN','TELESCOP','ALT_OBS','LAT_OBS','LONG_OBS',
                                   'RCT_VER','RCT_DATE','INSTRUME','SPECTID',
                                   'GRATID','GRATTILT','GRATLPMM','ORDER','TDFCTVER','TDFCTDAT','DICHROIC',
                                   'OBSTYPE','TOPEND','AXIS','AXIS_X','AXIS_Y','TRACKING','TDFDRVER']

    primary_header_conditional_keyword_list = ['HGCOORDS','COORDROT','COORDREV']

    for keyword in primary_header_keyword_list:
        if keyword in ifu.primary_header.keys():
            val = []
            for ifu in ifu_list: val.append(ifu.primary_header[keyword])
            if len(set(val)) == 1: 
                hdr_new.append(hdr.cards[keyword])
            else:
                print('Non-unique value for keyword:',keyword)

    # Extract the couple of relevant keywords from the fibre table header and again
    # check for consistency of keyword values

    fibre_header_keyword_list = ['PLATEID','LABEL']

    for keyword in fibre_header_keyword_list:
        val = []
        for ifu in ifu_list: val.append(ifu.fibre_table_header[keyword])
        if len(set(val)) == 1:
            hdr_new.append(ifu_list[0].fibre_table_header.cards[keyword])
        else:
            print('Non-unique value for keyword:', keyword)

    # Append HISTORY from the initial RSS file header, assuming HISTORY is
    # common for all RSS frames.

    hdr_new.append(hdr.cards['SCRUNCH'])
    hist_ind = np.where(np.array(hdr.keys()) == 'HISTORY')[0]
    for i in hist_ind: 
        hdr_new.append(hdr.cards[i])

    # Add catalogue RA & DEC to header
    hdr_new.set('CATARA', ifu_list[0].obj_ra[0], after='CRVAL3')
    hdr_new.set('CATADEC', ifu_list[0].obj_dec[0], after='CATARA')

    # Additional header items from random extensions
    # Each key in `additional` is a FITS extension name
    # Each value is a list of FITS header keywords to copy from that extension
    additional = {'FLUX_CALIBRATION': ('STDNAME',)}
    for extname, key_list in additional.items():
        try:
            add_hdr_list = [pf.getheader(f, extname) for f in files]
        except KeyError:
            print('Extension not found:', extname)
            continue
        for key in key_list:
            val = []
            try:
                for add_hdr in add_hdr_list:
                    val.append(add_hdr[key])
            except KeyError:
                print('Keyword not found:', key, 'in extension', extname)
                continue
            if len(set(val)) == 1:
                hdr_new.append(add_hdr.cards[key])
            else:
                print('Non-unique value for keyword:', key, 'in extension', extname)

    return hdr_new

def create_metadata_table(ifu_list):
    """Build a FITS binary table HDU containing all the meta data from individual IFU objects."""
    
    # List of columns for the meta-data table
    columns = []
    
    # Number of rows to appear in the table (equal to the number of files used to create the cube)
    n_rows = len(ifu_list)
    
    # For now we assume that all files have the same set of keywords, and complain if they don't.
    
    first_header = ifu_list[0].primary_header
    
    primary_header_keywords = first_header.keys()
    
    # We must remove COMMENT and HISTORY keywords, as they must be treated separately.
    for i in range(primary_header_keywords.count('HISTORY')):
        primary_header_keywords.remove('HISTORY')
    for i in range(primary_header_keywords.count('COMMENT')):
        primary_header_keywords.remove('COMMENT')
    for i in range(primary_header_keywords.count('SIMPLE')):
        primary_header_keywords.remove('SIMPLE')
    for i in range(primary_header_keywords.count('EXTEND')):
        primary_header_keywords.remove('EXTEND')
    for i in range(primary_header_keywords.count('SCRUNCH')):
        primary_header_keywords.remove('SCRUNCH')
    
    # TODO: Test/check that all ifu's have the same keywords and error if not

    # Determine the FITS binary table column types types for each header keyword
    # and create the corresponding columns. See the documentation here:
    # 
    #     https://astropy.readthedocs.org/en/v0.1/io/fits/usage/table.html#creating-a-fits-table

    def get_primary_header_values(keyword, dtype):
        return np.asarray(map(lambda x: x.primary_header[keyword],ifu_list), dtype)

    for keyword in primary_header_keywords:
        if (isinstance(first_header[keyword],bool)):
            # Output type is Logical (boolean)
            columns.append(pf.Column(
                name=keyword,
                format='L',
                array=get_primary_header_values(keyword,np.bool)
                )) 
        elif (isinstance(first_header[keyword],int)):
            columns.append(pf.Column(
                name=keyword,
                format='K',
                array=get_primary_header_values(keyword,np.int)
                )) # 64-bit integer
        elif (isinstance(first_header[keyword],str)):
            columns.append(pf.Column(
                name=keyword,
                format='128A',
                array=get_primary_header_values(keyword,'|S128')                
                )) # 128 character string
        elif (isinstance(first_header[keyword],float)):
            columns.append(pf.Column(
                name=keyword,
                format='E',
                array=get_primary_header_values(keyword,np.float)
                )) # single-precision float

    del get_primary_header_values
    
    # TODO: Add columns for comments and history information.
    #
    # The code below tries to put these in a variable size character array
    # column in the binary table, but it is very messy. Better might be a
    # separate ascii table.
    #
    #columns.append(pf.Column(name='COMMENTS', format='80PA(100)')) # Up to 100 80-character lines
    #columns.append(pf.Column(name='HISTORY', format='80PA(100)')) # Up to 100 80-character lines
   
    return pf.BinTableHDU.from_columns(columns)

    

def scale_cube_pair(file_pair, scale, **kwargs):
    """Scale both blue and red cubes by a given value."""
    for path in file_pair:
        scale_cube(path, scale, **kwargs)

def scale_cube(path, scale, hdu='PRIMARY', band=None):
    """Scale a single cube by a given value."""
    hdulist = pf.open(path, 'update')
    try:
        old_scale = hdulist[hdu].header['RESCALE']
    except KeyError:
        old_scale = 1.0
    hdulist['PRIMARY'].data *= (scale / old_scale)
    hdulist['VARIANCE'].data *= (scale / old_scale)**2
    hdulist[hdu].header['RESCALE'] = (scale, 'Scaling applied to data')
    if band is not None:
        hdulist[hdu].header['SCALEBND'] = (band, 'Band used to scale data')
    hdulist.flush()
    hdulist.close()

def scale_cube_pair_to_mag(file_pair, axis, hdu='PRIMARY', band=None):
    """
    Scale both cubes according to their pre-measured magnitude.

    If no band is supplied, it will try to choose the best one based on
    wavelength coverage.
    """
    header = pf.getheader(file_pair[0], hdu)
    if band is None:
        overlap = 0.0
        # Loop in decreasing order of preference, as a tie-breaker
        for band_new in 'griuz':
            key = 'MAG' + band_new.upper()
            if key not in header or header[key] == -99999:
                # No magnitude available for this band
                continue
            overlap_new = wavelength_overlap(file_pair, band_new, axis)
            if overlap_new > overlap:
                # Better coverage than previous best
                band = band_new
                overlap = overlap_new
    key = 'MAG' + band.upper()
    measured_mag = header[key]
    catalogue_mag = header['CAT' + key]
    scale = 10.0 ** ((measured_mag - catalogue_mag) / 2.5)
    if measured_mag == -99999:
        # No valid magnitude found
        scale = 1.0
    scale_cube_pair(file_pair, scale, hdu=hdu, band=band)
    return scale

def wavelength_overlap(file_pair, band, axis):
    """Return the weighted fraction of the band that is covered by the data."""
    overlap = 0.0
    response, wavelength_band = read_filter(band)
    response /= np.sum(response)
    coverage = np.zeros_like(response)
    for path in file_pair:
        wavelength_data = get_coords(pf.getheader(path), axis)
        coverage[(wavelength_band >= wavelength_data[0]) &
                 (wavelength_band <= wavelength_data[-1])] = 1.0
    overlap = np.sum(response * coverage)
    return overlap

def create_qc_hdu(file_list, name):
    """Create and return an HDU of QC information."""
    # The name of the object is passed, so that information specific to that
    # object can be included, but at the moment it is not used.
    qc_keys = (
        'SKYMDCOF',
        'SKYMDLIF',
        'SKYMDCOA',
        'SKYMDLIA',
        'SKYMNCOF',
        'SKYMNLIF',
        'SKYMNCOA',
        'SKYMNLIA',
        'TRANSMIS',
        'FWHM')
    rel_transp = []
    qc_data = {key: [] for key in qc_keys}
    for path in file_list:
        hdulist = pf.open(path)
        try:
            rel_transp.append(
                1.0 / hdulist['FLUX_CALIBRATION'].header['RESCALE'])
        except KeyError:
            rel_transp.append(-9999)
        if 'QC' in hdulist:
            qc_header = hdulist['QC'].header
            for key in qc_keys:
                if key in qc_header:
                    qc_data[key].append(qc_header[key])
                else:
                    qc_data[key].append(-9999)
        else:
            for key in qc_keys:
                qc_data[key].append(-9999)
        hdulist.close()
    filename_list = [os.path.basename(f) for f in file_list]
    columns = [
        pf.Column(name='filename', format='20A', array=filename_list),
        pf.Column(name='rel_transp', format='E', array=rel_transp),
        ]
    for key in qc_keys:
        columns.append(pf.Column(name=key, format='E', array=qc_data[key]))
    hdu = pf.BinTableHDU.from_columns(columns, name='QC')
    return hdu
