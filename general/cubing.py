import pylab as py
import numpy as np
import scipy as sp

import astropy.io.fits as pf
import astropy.wcs as pw

import itertools
import os
import sys
import datetime

# Cross-correlation function from scipy.signal (slow)
from scipy.signal import correlate

# Import the sigma clipping algorithm from astropy
from astropy.stats import sigma_clip

# Attempt to import bottleneck to improve speed, but fall back to old routines
# if bottleneck isn't present
try:
    from bottleneck import nanmedian, nansum, nanmean
except:
    from scipy.stats import nanmedian, nanmean
    nansum = np.nansum


# Utils code.
from .. import utils
from .. import samifitting as fitting
from ..utils.mc_adr import DARCorrector, compute_parallactic_angle

# importing everything defined in the config file
from ..config import *

"""
This module covers functions required to create cubes from a dithered set of RSS frames.

USAGE:
The relevant function is dithered_cube_from_rss. It has the following inputs:

inlist: A text file with a list of RSS frames (one per line, typically a dither set) for which to make cubes.
sample_size: Size of the output pixel (defaults to 0.5 arcseconds).
objects: Which objects to make cubes for. Default is all objects, or provide a list of strings.
plot: Make plots? Defaults to True.
write: Write FITS files of the resulting data cubes? Defaults to False.

Example call 1:

dithered_cube_from_rss('all_rss_files.list', write=True)

will make cubes for all objects with the default output pixel size and writes the files to disk.

Example call 2:

dithered_cube_from_rss('all_rss_files.list', sample_size=0.8, write=True)

Varies the sample size from above.

"""

HG_CHANGESET = utils.hg_changeset(__file__)

epsilon = np.finfo(np.float).eps
# Store the value of epsilon for quick access.

def get_object_names(infile):
    """Get the object names observed in the file infile."""

    # Open up the file and pull out list of observed objects.
    table=pf.open(infile)[2].data
    names=table.field('NAME')

    # Find the set of unique values in names
    names_unique=list(set(names))

    # Pick out the object names, rejecting SKY and empty strings
    object_names_unique = [s for s in names_unique if s.startswith('SKY')==False and len(s)>0]

    return object_names_unique

def get_probe(infile, object_name, verbose=True):
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
        print "Object", object_name, "was observed in IFU", ifu, "in file", infile

    hdulist.close()

    # Return the probe number
    return ifu

def dithered_cubes_from_rss_files(inlist, sample_size=0.5, drop_factor=0.5, objects='all', clip=True, plot=True, write=False):
    """A wrapper to make a cube from reduced RSS files. Only input files that go together - ie have the same objects."""

    start_time = datetime.datetime.now()

    # Set a few numpy printing to terminal things
    np.seterr(divide='ignore', invalid='ignore') # don't print division or invalid warnings.
    np.set_printoptions(linewidth=120) # can also use to set precision of numbers printed to screen

    # Read in the list of all the RSS files input by the user.
    files=[]
    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])

        files.append(np.str(cols[0]))
        
    n_files = len(files)

    # For first file get the names of galaxies observed - assuming these are the same in all RSS files.
    # @TODO: This is not necessarily true, and we should either through an exception or account for it
    if objects=='all':
        object_names=get_object_names(files[0])
    else:
        object_names=objects
        
    
    
    print "--------------------------------------------------------------"
    print "The following objects will be cubed:"
    print

    for name in object_names:
        print name

    print "--------------------------------------------------------------"

    for name in object_names:

        print
        print "--------------------------------------------------------------"
        print "Starting with object:", name
        print
        
        ifu_list = []
        
        for j in xrange(n_files):
            ifu_list.append(utils.IFU(files[j], name, flag_name=True))


        # Call dithered_cube_from_rss to create the flux, variance and weight cubes for the object.
        #flux_cube, var_cube, weight_cube, diagnostics = dithered_cube_from_rss(ifu_list, sample_size=sample_size,
        #                                  drop_factor=drop_factor, clip=clip, plot=plot)
        
        flux_cube = np.zeros((10,11,12))
        var_cube = np.zeros((10,11,12))
        weight_cube = np.zeros((10,11,12))
        
        # Write out FITS files.
        if write==True:

            # First check if the object directory already exists or not.
            if os.path.isdir(name):
                print "Directory Exists", name
                print "Writing files to the existing directory"
            else:
                print "Making directory", name
                os.mkdir(name)
            
            # NOTE - At this point we will want to generate accurate WCS information and create a proper header.
            # The addition of ancillary data to the header will be valuable. I think the creation of the header should
            # be separated off into a separate function (make_cube_header?)
            
            # This could proceed several ways, depending on what the WCS coders want from us and what we want back.
            # Need to liase.
    
            # For now create a rudimentary WCS with ONLY the correct wavelength scale.
    
            # First get some info from one of the headers.
            list1=pf.open(files[0])
            hdr=list1[0].header
    
            arm = ifu_list[0].spectrograph_arm
                
            # Create the wcs.
            wcs_new=pw.WCS(naxis=3)
            wcs_new.wcs.crpix = [1, 1, hdr['CRPIX1']]
            print sample_size
            #print str(sample_size)
            wcs_new.wcs.cdelt = np.array([sample_size, sample_size, hdr['CDELT1']])
            wcs_new.wcs.crval = [1, 1, hdr['CRVAL1']]
            wcs_new.wcs.ctype = ["ARCSEC", "ARCSEC", hdr['CTYPE1']]
            wcs_new.wcs.equinox = 2000
            
            # Create a header
            hdr_new=wcs_new.to_header()
            
            # Add the name to the header
            hdr_new.update('NAME', name, 'Object ID')
    
            # Add the mercurial changeset ID to the header
            hdr_new.update('HGCUBING', HG_CHANGESET, 'Hg changeset ID for cubing code')
    
            # Put the RSS files into the header
            for num in xrange(len(files)):
                #print files[num]
                rss_key='HIERARCH RSS_FILE '+str(num+1)
                rss_string='Input RSS file '+str(num+1)
                hdr_new.update(rss_key, os.path.basename(files[num]), rss_string)
            
            # Create HDUs for each cube - note headers generated automatically for now.
            #
            # @NOTE: PyFITS writes axes to FITS files in the reverse of the sense
            # of the axes in Numpy/Python. So a numpy array with dimensions
            # (5,10,20) will produce a FITS cube with x-dimension 20,
            # y-dimension 10, and the cube (wavelength) dimension 5.  --AGreen
            hdu1=pf.PrimaryHDU(np.transpose(flux_cube, (2,1,0)), hdr_new)
            hdu2=pf.ImageHDU(np.transpose(var_cube, (2,1,0)), name='VARIANCE')
            hdu3=pf.ImageHDU(np.transpose(weight_cube, (2,1,0)), name='WEIGHT')

            # Create HDUs for meta-data
            metadata_table = create_metadata_table(ifu_list)
            

            # Put individual HDUs into a HDU list
            hdulist=pf.HDUList([hdu1,hdu2,hdu3,metadata_table])
        
            # Write to FITS file.
            # NOTE - In here need to add the directory structure for the cubes.
            outfile_name=str(name)+'_'+str(arm)+'_'+str(len(files))+'.fits'
            outfile_name_full=os.path.join(name, outfile_name)

            print "Writing", outfile_name_full
            "--------------------------------------------------------------"
            hdulist.writeto(outfile_name_full)
    
            # Close the open file
            list1.close()
            
    print("Time dithered_cubes_from_files wall time: {0}".format(datetime.datetime.now() - start_time))

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
    for i in xrange(primary_header_keywords.count('HISTORY')):
        primary_header_keywords.remove('HISTORY')
    for i in xrange(primary_header_keywords.count('COMMENT')):
        primary_header_keywords.remove('COMMENT')
    for i in xrange(primary_header_keywords.count('SIMPLE')):
        primary_header_keywords.remove('SIMPLE')
    for i in xrange(primary_header_keywords.count('EXTEND')):
        primary_header_keywords.remove('EXTEND')
    for i in xrange(primary_header_keywords.count('SCRUNCH')):
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
        if (isinstance(first_header[keyword],int)):
            columns.append(pf.Column(
                name=keyword,
                format='K',
                array=get_primary_header_values(keyword,np.int)
                )) # 64-bit integer
        if (isinstance(first_header[keyword],str)):
            columns.append(pf.Column(
                name=keyword,
                format='128A',
                array=get_primary_header_values(keyword,'|S128')                
                )) # 128 character string
        if (isinstance(first_header[keyword],float)):
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
    
    return pf.new_table(columns)

def dithered_cube_from_rss(ifu_list, sample_size=0.5, drop_factor=0.5, clip=True, plot=True, offsets='fit'):
        
    # When resampling need to know the size of the grid in square output pixels
    # @TODO: Compute the size of the grid instead of hard code it!
    size_of_grid=50 
    

    diagnostic_info = {}

    n_obs = len(ifu_list)
    # Number of observations/files
    
    n_slices = np.shape(ifu_list[0].data)[1]
    # The number of wavelength slices
    
    n_fibres = ifu_list[0].data.shape[0]
    # Number of fibres

    # Create an instance of SAMIDrizzler for use later to create individual overlap maps for each fibre.
    # The attributes of this instance don't change from ifu to ifu.
    overlap_maps=SAMIDrizzler(sample_size, size_of_grid, n_obs * n_fibres)

    # Empty lists for positions and data. Could be arrays, might be faster? Should test...
    xfibre_all=[]
    yfibre_all=[]
    data_all=[]
    var_all=[]

    ifus_all=[]

    # The following loop:
    #
    #   2. Uses a Centre of Mass to as an initial guess to gaussian fit the position 
    #      of the object within the bundle
    #   3. Computes the positions of the fibres within each file including an offset
    #      for the galaxy position from the gaussian fit (e.g., everything is now on 
    #      the same coordiante system.
    #

    for j in xrange(n_obs):

        # Get the data.
        galaxy_data=ifu_list[j]
        
        # Smooth the spectra and median.
        data_smoothed=np.zeros_like(galaxy_data.data)
        for p in xrange(np.shape(galaxy_data.data)[0]):
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

        else:
            # Perhaps use this place to allow definition of the offsets manually??
            # Hopefully only useful for test purposes. LF 05/06/2013
            xm=galaxy_data.x_microns - np.mean(galaxy_data.x_microns)
            ym=galaxy_data.y_microns - np.mean(galaxy_data.y_microns)
    
        xfibre_all.append(xm)
        yfibre_all.append(ym)

        data_all.append(galaxy_data.data)
        var_all.append(galaxy_data.var)

        ifus_all.append(galaxy_data.ifu)


    xfibre_all=np.asanyarray(xfibre_all)
    yfibre_all=np.asanyarray(yfibre_all)
    data_all=np.asanyarray(data_all)
    var_all=np.asanyarray(var_all)

    ifus_all=np.asanyarray(ifus_all)

    # @TODO: Rescaling between observations.
    #
    #     This may be done here (_before_ the reshaping below). There also may
    #     be a case to do it for whole files, although that may have other
    #     difficulties.


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
    xfibre_all = np.reshape(xfibre_all, (n_obs * n_fibres)           )
    yfibre_all = np.reshape(yfibre_all, (n_obs * n_fibres)           )
    data_all   = np.reshape(data_all,   (n_obs * n_fibres, n_slices) )
    var_all    = np.reshape(var_all,    (n_obs * n_fibres, n_slices) )

    
    
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
    for ii in xrange(n_obs * n_fibres):
        data_norm[ii,:] = data_all[ii,:] / nanmedian( data_all[ii,:])
        

    # Now create a new array to hold the final data cube and build it slice by slice
    flux_cube   = np.zeros( (size_of_grid, size_of_grid, n_obs * n_fibres) )
    var_cube    = np.zeros( (size_of_grid, size_of_grid, n_obs * n_fibres) )
    weight_cube = np.empty( (size_of_grid, size_of_grid, n_obs * n_fibres) )
    
    print("data_all.shape: ", np.shape(data_all))

    if clip:
        # Set up some diagostics if you have the clip flag set.
        diagnostic_info['unmasked_pixels_after_sigma_clip'] = 0
        diagnostic_info['unmasked_pixels_before_sigma_clip'] = 0

        diagnostic_info['n_pixels_sigma_clipped'] = []
           
    # Set up the differential atmospheric refraction correction:
    dar_corrector = DARCorrector(method='simple')
    
    dar_corrector.temperature = ifu_list[0].fibre_table_header['ATMTEMP'] 
    dar_corrector.air_pres = ifu_list[0].fibre_table_header['ATMPRES'] * millibar_to_mmHg
    #                     (factor converts from millibars to mm of Hg)
    dar_corrector.water_pres = \
        utils.saturated_partial_pressure_water(dar_corrector.air_pres, dar_corrector.temperature) * \
        ifu_list[0].fibre_table_header['ATMRHUM']
    
    # TODO: This is the field ZD, not the target ZD. Also, the mean is probably
    # not the right computation to determine the best ZD for the correction.
    dar_corrector.zenith_distance = \
        (ifu_list[0].primary_header['ZDSTART'] + ifu_list[0].primary_header['ZDEND']) / 2
 
    # TODO: This is the field HA, not the target HA.
    dar_corrector.hour_angle = \
        (ifu_list[0].primary_header['HASTART'] + ifu_list[0].primary_header['HAEND']) / 2
    
    parallactic_angle = compute_parallactic_angle(dar_corrector.hour_angle, 
                                          dar_corrector.zenith_distance, 
                                          latitude.degrees)
    
    dar_corrector.print_setup()
    
    # Load the wavelength solution for the datacubes. 
    #
    # TODO: This should change when the header keyword propagation is improved
    # and we have confirmed that all RSS files are on the same wavelength
    # solution.
    wavelength_array = ifu_list[0].lambda_range
    
    # This loops over wavelength slices (e.g., 2048). 
    for l in xrange(n_slices):

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
        norm_rss_slice = data_norm[:,l]
        data_rss_slice = data_all[:,l]
        var_rss_slice = var_all[:,l]

        # Determine differential atmospheric refraction correction for this slice
        dar_r = dar_corrector.correction(wavelength_array[l]) * 1000.0 / plate_scale 
        # @NOTE: Could change to arcsecs instead of microns.
        dar_x = dar_r * np.cos(np.radians(parallactic_angle))
        dar_y = dar_r * np.sin(np.radians(parallactic_angle))

        print( "DAR lambda: {:5.0f} r: {:5.2f}, x: {:5.2f}, y: {:5.2f}".format(wavelength_array[l],
                                                                       dar_r * plate_scale/1000.0, 
                                                                       dar_x * plate_scale/1000.0,
                                                                       dar_y * plate_scale/1000.0))

        # Compute drizzle map for this wavelength slice.
        overlap_array, weight_grid_slice = overlap_maps.drizzle(xfibre_all + dar_x, yfibre_all + dar_y)
        
        # Map RSS slices onto gridded slices
        norm_grid_slice_fibres=overlap_array*norm_rss_slice        
        data_grid_slice_fibres=overlap_array*data_rss_slice
        var_grid_slice_fibres=(overlap_array*overlap_array)*var_rss_slice

        if clip:
            # Perform sigma clipping of the data if the clip flag is set.
            
            n_unmasked_pixels_before_clipping = np.isfinite(data_grid_slice_fibres).sum()
        
            # Sigma clip it - pixel by pixel and make a master mask
            # array. Current clipping, sigma=5 and 1 iteration.        
            mask_grid_slice_fibres = sigma_clip_mask_slice_fibres(norm_grid_slice_fibres/weight_grid_slice)

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
        weight_grid_slice_fibres=weight_grid_slice*valid_grid_slice_fibres

        # Collapse the slice arrays
        data_grid_slice_final = nansum(data_grid_slice_fibres, axis=2)
        var_grid_slice_final = nansum(var_grid_slice_fibres, axis=2)
        weight_grid_slice_final = nansum(weight_grid_slice_fibres, axis=2)
        
        # Where the weight map is within epsilon of zero, set it to NaN to
        # prevent divide by zero errors later.
        weight_grid_slice_final[weight_grid_slice_final < epsilon] = np.NaN
        
        flux_cube[:,:,l]=data_grid_slice_final
        var_cube[:,:,l]=var_grid_slice_final
        weight_cube[:,:,l]=weight_grid_slice_final

    # I have now got: flux cube, variance cube, weight cube. These have been made assuming no drop-size reduction factor.
    # Apply the drop size reduction factor to all three cubes.
    flux_cube=flux_cube/(drop_factor**2)
    var_cube=var_cube/(drop_factor**4)
    weight_cube=weight_cube/(drop_factor**2)

    # Now need to scale the flux and variance cubes appropriately by the weight cube
    flux_cube=flux_cube/weight_cube # flux cube scaling by weight map
    image=nanmedian(flux_cube, axis=2)

    var_cube=var_cube/(weight_cube*weight_cube) # variance cube scaling by weight map

    return flux_cube, var_cube, weight_cube, diagnostic_info

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

    

    def __init__(self, sample_size_arcsec, size_of_grid, n_fibres):
        """Construct a new SAMIDrizzler isntance with the necessary information.
        
        Parameters
        ----------
        sample_sie_arcsec: the size of each output pixel in arcseconds
        size_of_grid: the number of pixels along each dimension of the 
            square output pixel grid
        n_fibres: the total number of unique fibres which will be 
            mapped onto the grid (usually n_fibres * n_obs) 
        
        """

        # The input values
        self.sample_size_arcsec=sample_size_arcsec
        # Set the size of the output grid - should probably be calculated somehow.
        self.size_of_grid=size_of_grid

        # Some unchanging SAMI stuff
        self.plate_scale=plate_scale # (in arcseconds per mm)
        self.fib_diam_arcsec=1.6 # (in arcseconds)

        # Work out stuff for the resampling 
        self.oversample=self.fib_diam_arcsec/self.sample_size_arcsec
        self.dx=1000*self.fib_diam_arcsec/(self.oversample*self.plate_scale) # in microns

        # Fibre area in pixels
        self.fib_area_pix=np.pi*(self.oversample/2.0)**2

        # Fibre diameter in pixels
        self.fib_diam_pix=(1000*self.fib_diam_arcsec)/(self.plate_scale*self.dx)

        # Output grid in microns
        self.x=(np.arange(self.size_of_grid)-self.size_of_grid/2)*self.dx
        self.y=(np.arange(self.size_of_grid)-self.size_of_grid/2)*self.dx
        
        # Empty array for all overlap maps - i.e. need one for each fibre!
        self.drop_to_pixel=np.empty((self.size_of_grid, self.size_of_grid, n_fibres))
        self.pixel_coverage=np.empty((self.size_of_grid, self.size_of_grid, n_fibres))
        
        self._last_drizzle = np.empty(1)
        # This is used to cache the arguments for the last drizzle.

    def single_overlap_map(self, fibrex, fibrey):
        """Compute the mapping from a single input drop to output pixel grid.
        
        (drop_fraction, pixel_fraction) = single_overlap_map(fibrex, fibrey)
        
        Parameters
        ----------
        fibrex: (float) The x-coordinate of the fibre.
        fibery: (float) The y-coordinate of the fibre.
        
        Returns
        -------
        
        This returns a tuple of two arrays. Both arrays have the same dimensions
        (that of the output pixel grid). 
        
        drop_fraction: (array) The fraction of the input drop in each output pixel.
        pixel_fraction: (arra) The fraction of each output pixel covered by the drop.
                
        """

        # Map fibre positions onto pixel positions in the output grid.
        xfib=(fibrex-self.x[0])/self.dx
        yfib=(fibrey-self.y[0])/self.dx
        
        # Create the overlap map from the circ.py code
        #
        # @NOTE: The circ.py code returns an array which has the x-coodinate in
        # the first index and the y-coordinate in the zeroth index. Therefore,
        # we transpose the result here so that the x-cooridnate (north positive)
        # is in the zeroth index, and y-coordinate (east positive) is in the
        # first index.
        overlap_map = np.transpose(utils.circ.resample_circle(self.size_of_grid, self.size_of_grid, xfib, yfib, \
                                         self.oversample/2.0))
        
        input_frac_map=overlap_map/self.fib_area_pix # Fraction of input fibre/drop in each output pixel
        output_frac_map=overlap_map/1.0 # divided by area of ind. sq. (output) pixel.

        return input_frac_map, output_frac_map
    
    def drizzle(self, xfibre_all, yfibre_all):
        """Compute a mapping from fibre drops to output pixels for all given fibre locations."""
        if (np.array_equal(self._last_drizzle,np.asarray([xfibre_all,yfibre_all]))):
            # We've been asked to recompute the same answer as last time, so don't recompute.
            return self.drop_to_pixel, self.pixel_coverage
        
        for i_fibre, xfib, yfib in itertools.izip(itertools.count(), xfibre_all, yfibre_all):
    
            # Feed the x and y fibre positions to the overlap_maps instance.
            drop_to_pixel_fibre, pixel_coverage_fibre=self.single_overlap_map(xfib, yfib)
    
            # These lines are ONLY for debugging. The plotting and overwriting is very slow! 
            #py.imshow(drop_to_pixel_fibre, origin='lower', interpolation='nearest')
            #py.draw()
    
            # Padding with NaNs instead of zeros (do I really need to do this? Probably not...)
            drop_to_pixel_fibre[np.where(drop_to_pixel_fibre < epsilon)]=np.nan
            pixel_coverage_fibre[np.where(pixel_coverage_fibre < epsilon)]=np.nan
    
            self.drop_to_pixel[:,:,i_fibre]=drop_to_pixel_fibre
            self.pixel_coverage[:,:,i_fibre]=pixel_coverage_fibre
    
        self._last_drizzle = np.asarray([xfibre_all, yfibre_all])
    
        return self.drop_to_pixel, self.pixel_coverage

