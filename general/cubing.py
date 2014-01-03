"""
This module covers functions required to create cubes from a dithered set of RSS
frames.

The most likely command a user will want to run is one of:

   dithered_cubes_from_rss_files
   dithered_cubes_from_rss_list

(these differ only very slightly)

These functions are merely wrappers for file input/output, with the actual work
being done in "dithered_cube_from_rss"

Drop size and output pixel grid size and dimensions are set with the following
module variables:

    output_pix_size_arcsec (Default: 0.5) - Size of output spaxel in arcsec
    drop_factor (Default: 0.5) - Size of drop size as a fraction of the fibre 
        size
    size_of_grid (Default: 50) - Number of pixels along each dimension of the 
        output grid   

To change the output, these variables can be changed after the module has been
loaded, but before calling any of the module functions. In particular, any pre-
existing SAMIDrizzler instances will have strange behaviour if these variables
are changed.

"""

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
from ..utils.mc_adr import DARCorrector, parallactic_angle, zenith_distance
from .. import diagnostics

# importing everything defined in the config file
from ..config import *

# WCS code
import astropy.io.ascii as ascii
from scipy.interpolate import griddata
import urllib


# Some global constants:

HG_CHANGESET = utils.hg_changeset(__file__)

epsilon = np.finfo(np.float).eps
# Store the value of epsilon for quick access.

output_pix_size_arcsec = 0.5    # Size of output spaxel in arcsec
drop_factor = 0.5    # Size of drop as a fraction of the fibre size
size_of_grid = 50    # Size of a side of the cube such that the cube has 50x50 spaxels
# @TODO: Compute the size of the grid instead of hard code it!??!

def get_object_names(infile):
    """Get the object names observed in the file infile."""

    # Open up the file and pull out list of observed objects.
    table=pf.open(infile)[2].data
    names=table.field('NAME')

    # Find the set of unique values in names
    names_unique=list(set(names))

    # Pick out the object names, rejecting SKY and empty strings
    object_names_unique = [s for s in names_unique if ((s.startswith('SKY')==False)
                            and (s.startswith('Sky')==False)) and len(s)>0]

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

def dar_correct(ifu_list, xfibre_all, yfibre_all, method='simple',update_rss=False):
    """Update the fiber positions as a function of wavelength to reflect DAR correction.
    
    """

    n_obs, n_fibres, n_slices = xfibre_all.shape

    # Set up the differential atmospheric refraction correction. Each frame
    # requires it's own correction, so we create a list of DARCorrectors.
    dar_correctors = []
                      
    for obs in ifu_list:
        darcorr = DARCorrector(method=method)
    
        darcorr.temperature = obs.fibre_table_header['ATMTEMP'] 
        darcorr.air_pres = obs.fibre_table_header['ATMPRES'] * millibar_to_mmHg
        #                     (factor converts from millibars to mm of Hg)
        darcorr.water_pres = \
            utils.saturated_partial_pressure_water(darcorr.air_pres, darcorr.temperature) * \
            obs.fibre_table_header['ATMRHUM']

        ha_offset = obs.ra - obs.meanra  # The offset from the HA of the field centre
    
        darcorr.zenith_distance = \
            integrate.quad(lambda ha: zenith_distance(obs.dec, ha),
                           obs.primary_header['HASTART'] + ha_offset,
                           obs.primary_header['HAEND'] + ha_offset)[0] / (
                              obs.primary_header['HAEND'] - obs.primary_header['HASTART'])

        darcorr.hour_angle = \
            (obs.primary_header['HASTART'] + obs.primary_header['HAEND']) / 2 + ha_offset
    
        darcorr.declination = obs.dec

        dar_correctors.append(darcorr)

        del darcorr # Cleanup since this is meaningless outside the loop.


    wavelength_array = ifu_list[0].lambda_range
    
    # Iterate over wavelength slices
    for l in xrange(n_slices):
        
        # Iterate over observations
        for i_obs in xrange(n_obs):
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



def dithered_cubes_from_rss_files(inlist, 
                                  objects='all', clip=True, plot=True, 
                                  write=False, suffix='', root='',
                                  overwrite=False):
    """A wrapper to make a cube from reduced RSS files, passed as a filename containing a list of filenames. Only input files that go together - ie have the same objects."""

    # Read in the list of all the RSS files input by the user.
    files=[]
    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])

        files.append(np.str(cols[0]))

    dithered_cubes_from_rss_list(files,
                                 objects=objects, 
                                 clip=clip, plot=plot, write=write, 
                                 root=root, suffix=suffix, overwrite=overwrite)
    return

def dithered_cubes_from_rss_list(files, 
                                 objects='all', clip=True, plot=True, 
                                 write=False, suffix='', nominal=False, root='',
                                 overwrite=False):
    """A wrapper to make a cube from reduced RSS files, passed as a list. Only input files that go together - ie have the same objects."""
        
    start_time = datetime.datetime.now()

    # Set a few numpy printing to terminal things
    np.seterr(divide='ignore', invalid='ignore') # don't print division or invalid warnings.
    np.set_printoptions(linewidth=120) # can also use to set precision of numbers printed to screen

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

        if write:
            # First check if the object directory already exists or not.
            directory = os.path.join(root, name)
            try:
                os.makedirs(directory)
            except OSError:
                print "Directory Exists", directory
                print "Writing files to the existing directory"
            else:
                print "Making directory", directory

            # Filename to write to
            arm = ifu_list[0].spectrograph_arm            
            outfile_name=str(name)+'_'+str(arm)+'_'+str(len(files))+suffix+'.fits'
            outfile_name_full=os.path.join(directory, outfile_name)

            # Check if the filename already exists
            if os.path.exists(outfile_name_full):
                if overwrite:
                    os.remove(outfile_name_full)
                else:
                    print 'Output file already exists:'
                    print outfile_name_full
                    print 'Skipping this object'
                    continue


        # Call dithered_cube_from_rss to create the flux, variance and weight cubes for the object.
        
        # For now, putting in a try/except block to skip over any errors
        try:
            flux_cube, var_cube, weight_cube, diagnostics = \
                dithered_cube_from_rss(ifu_list, clip=clip, plot=plot)
        except Exception:
            print 'Cubing failed! Skipping to next galaxy.'
            print 'Object:', name, 'files:', files
            continue
            #raise

        # Write out FITS files.
        if write==True:

            if ifu_list[0].gratid == '580V':
                band = 'g'
            elif ifu_list[0].gratid == '1000R':
                band = 'r'
            else:
                raise ValueError('Could not identify band. Exiting')

            # Equate Positional WCS
            WCS_pos,WCS_flag = WCS_position(ifu_list[0],flux_cube,name,band,plot,nominal=nominal)   
            
            # First get some info from one of the headers.
            list1=pf.open(files[0])
            hdr=list1[0].header
    
            hdr_new = create_primary_header(ifu_list,name,files,WCS_pos,WCS_flag)

            # Define the units for the datacube
            hdr_new['BUNIT'] = ('10**(-16) erg /s /cm**2 /angstrom /pixel', 
                                'Units')

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
            #metadata_table = create_metadata_table(ifu_list)
            
            # Put individual HDUs into a HDU list
            hdulist=pf.HDUList([hdu1,hdu2,hdu3]) #,metadata_table])
        
            # Write the file
            print "Writing", outfile_name_full
            "--------------------------------------------------------------"
            hdulist.writeto(outfile_name_full)
    
            # Close the open file
            list1.close()
            
    print("Time dithered_cubes_from_files wall time: {0}".format(datetime.datetime.now() - start_time))

def create_primary_header(ifu_list,name,files,WCS_pos,WCS_flag):
    """Create a primary header to attach to each cube from the RSS file headers"""

    hdr = ifu_list[0].primary_header
    fbr_hdr = ifu_list[0].fibre_table_header

    # Get positional information from WCS_pos
    #### Do something here!!!

    # Create the wcs.
    wcs_new=pw.WCS(naxis=3)
    wcs_new.wcs.crpix = [WCS_pos["CRPIX1"], WCS_pos["CRPIX2"], hdr['CRPIX1']]
    wcs_new.wcs.cdelt = np.array([WCS_pos["CDELT1"], WCS_pos["CDELT2"], hdr['CDELT1']])
    wcs_new.wcs.crval = [WCS_pos["CRVAL1"], WCS_pos["CRVAL2"], hdr['CRVAL1']]
    wcs_new.wcs.ctype = [WCS_pos["CTYPE1"], WCS_pos["CTYPE2"], hdr['CTYPE1']]
    wcs_new.wcs.equinox = 2000
            
    # Create a header
    hdr_new=wcs_new.to_header()
    hdr_new.update('WCS_SRC',WCS_flag,'WCS Source')
            
    # Add the name to the header
    hdr_new.update('NAME', name, 'Object ID')
    # Need to implement a database specific-specific OBSTYPE keyword to indicate galaxies
    # *NB* This is different to the OBSTYPE keyword already in the header below
    
    # Determine total exposure time and add to header
    total_exptime = 0.
    for ifu in ifu_list:
        total_exptime+=ifu.primary_header['EXPOSED']
    hdr_new.update('TOTALEXP',total_exptime,'Total exposure (seconds)')

    # Add the mercurial changeset ID to the header
    hdr_new.update('HGCUBING', HG_CHANGESET, 'Hg changeset ID for cubing code')
    # Need to implement a global version number for the database
    
    # Put the RSS files into the header
    for num in xrange(len(files)):
        rss_key='HIERARCH RSS_FILE '+str(num+1)
        rss_string='Input RSS file '+str(num+1)
        hdr_new.update(rss_key, os.path.basename(files[num]), rss_string)

    # Extract header keywords of interest from the metadata table, check for consistency
    # then append to the main header

    primary_header_keyword_list = ['DCT_DATE','DCT_VER','DETECXE','DETECXS','DETECYE','DETECYS',
                                   'DETECTOR','XPIXSIZE','YPIXSIZE','METHOD','SPEED','READAMP','RO_GAIN',
                                   'RO_NOISE','ORIGIN','TELESCOP','ALT_OBS','LAT_OBS','LONG_OBS',
                                   'RCT_VER','RCT_DATE','RADECSYS','INSTRUME','SPECTID',
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
                print 'Non-unique value for keyword:',keyword

    # Extract the couple of relevant keywords from the fibre table header and again
    # check for consistency of keyword values

    fibre_header_keyword_list = ['PLATEID','LABEL']

    for keyword in fibre_header_keyword_list:
        val = []
        for ifu in ifu_list: val.append(ifu.fibre_table_header[keyword])
        if len(set(val)) == 1:
            hdr_new.append(ifu_list[0].fibre_table_header.cards[keyword])
        else:
            print 'Non-unique valie for keyword:',keyword

    # Append HISTORY from the initial RSS file header, assuming HISTORY is
    # common for all RSS frames.

    hdr_new.append(hdr.cards['SCRUNCH'])
    hist_ind = np.where(np.array(hdr.keys()) == 'HISTORY')[0]
    for i in hist_ind: 
        hdr_new.append(hdr.cards[i])

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
   
    return pf.new_table(columns)

def dithered_cube_from_rss(ifu_list, clip=True, plot=True, offsets='file'):
        
    diagnostic_info = {}

    n_obs = len(ifu_list)
    # Number of observations/files
    
    n_slices = np.shape(ifu_list[0].data)[1]
    # The number of wavelength slices
    
    n_fibres = ifu_list[0].data.shape[0]
    # Number of fibres

    # Create an instance of SAMIDrizzler for use later to create individual overlap maps for each fibre.
    # The attributes of this instance don't change from ifu to ifu.
    overlap_maps=SAMIDrizzler(size_of_grid, n_obs * n_fibres)

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

        # Check that we're not trying to use data that isn't there
        # Change the offsets method if necessary
        if offsets == 'file' and not hasattr(galaxy_data, 'x_refmed'):
            print 'Offsets have not been pre-measured! Fitting them now.'
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
    
        xfibre_all.append(xm)
        yfibre_all.append(ym)

        data_all.append(galaxy_data.data)
        var_all.append(galaxy_data.var)

        ifus_all.append(galaxy_data.ifu)


    xfibre_all=np.asanyarray(xfibre_all)
    yfibre_all=np.asanyarray(yfibre_all)

    # Scale these up to have a wavelength axis as well
    xfibre_all = xfibre_all.reshape(n_obs, n_fibres, 1).repeat(n_slices,2)
    yfibre_all = yfibre_all.reshape(n_obs, n_fibres, 1).repeat(n_slices,2)
    
    
    data_all=np.asanyarray(data_all)
    var_all=np.asanyarray(var_all)

    ifus_all=np.asanyarray(ifus_all)

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


        # Compute drizzle maps for this wavelength slice.
        overlap_array, weight_grid_slice = overlap_maps.drizzle(xfibre_all[:,l], yfibre_all[:,l])
        
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

        # Combine (sum) the individual observations. See Section 6.1: "Simple
        # summation" of the data reduction paper. Note that these arrays are
        # "unweighted" cubes C' and V', not the weighted C and V given in the
        # paper.
        data_grid_slice_final = nansum(data_grid_slice_fibres, axis=2) / n_obs
        var_grid_slice_final = nansum(var_grid_slice_fibres, axis=2) / (n_obs ** 2)
        weight_grid_slice_final = nansum(weight_grid_slice_fibres, axis=2) / n_obs
        
        # Where the weight map is within epsilon of zero, set it to NaN to
        # prevent divide by zero errors later.
        weight_grid_slice_final[weight_grid_slice_final < epsilon] = np.NaN
        
        flux_cube[:, :, l] = data_grid_slice_final
        var_cube[:, :, l] = var_grid_slice_final
        weight_cube[:, :, l] = weight_grid_slice_final

    print("Total calls to drizzle: {}, recomputes: {}, ({}%)".format(
                overlap_maps.n_drizzle, overlap_maps.n_drizzle_recompute,
                float(overlap_maps.n_drizzle_recompute)/overlap_maps.n_drizzle_recompute))

    # The flux and variance cubes must be rescaled to account for the reduction
    # in drop size. See Section 9.3: "Flux Scaling" of the data reduction paper.
    # Note also, that this scaling is immediately nullified by the division by the
    # weight cube below.
    flux_cube_scaled = flux_cube / (drop_factor ** 2)
    var_cube_scaled = var_cube / (drop_factor ** 4)
    weight_cube_scaled = weight_cube / (drop_factor ** 2)

    # Finally, divide by the weight cube to remove variations in exposure time
    # (and hence surface brightness sensitivity) from the output data cube.
    flux_cube_unprimed = flux_cube_scaled / weight_cube_scaled 
    var_cube_unprimed = var_cube_scaled / (weight_cube_scaled * weight_cube_scaled)

    return flux_cube_unprimed, var_cube_unprimed, weight_cube_scaled, diagnostic_info

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

    

    def __init__(self, size_of_grid, n_fibres):
        """Construct a new SAMIDrizzler isntance with the necessary information.
        
        Parameters
        ----------
        size_of_grid: the number of pixels along each dimension of the 
            square output pixel grid
        n_fibres: the total number of unique fibres which will be 
            mapped onto the grid (usually n_fibres * n_obs) 
        
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

        
        self.drizzle_update_tol = 0.1 * self.pix_size_micron

        # Output grid abscissa in microns
        self.grid_coordinates_x = (np.arange(self.output_dimension) - self.output_dimension / 2) * self.pix_size_micron
        self.grid_coordinates_y = (np.arange(self.output_dimension) - self.output_dimension / 2) * self.pix_size_micron

        # Empty array for all overlap maps - i.e. need one for each fibre!
        self.drop_to_pixel = np.empty((self.output_dimension, self.output_dimension, n_fibres))
        self.pixel_coverage = np.empty((self.output_dimension, self.output_dimension, n_fibres))

        self.drop_to_pixel=np.empty((self.output_dimension, self.output_dimension, n_fibres))
        self.pixel_coverage=np.empty((self.output_dimension, self.output_dimension, n_fibres))
        
        # These are used to cache the arguments for the last drizzle.
        self._last_drizzle_x = np.zeros(1)
        self._last_drizzle_y = np.zeros(1)        

        # Number of times drizzle has been called in this instance
        self.n_drizzle = 0
        
        # Number of times drizzle has been recomputed in this instance
        self.n_drizzle_recompute = 0

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

        # Create the overlap map from the circ.py code
        #
        # @NOTE: The circ.py code returns an array which has the x-coodinate in
        # the second index and the y-coordinate in the first index. Therefore,
        # we transpose the result here so that the x-cooridnate (north positive)
        # is in the first index, and y-coordinate (east positive) is in the
        # second index.
        overlap_map = np.transpose(
            utils.circ.resample_circle(
                self.output_dimension, self.output_dimension, 
                xfib, yfib,
                self.drop_diameter_pix / 2.0))

        # Fraction of input drop in each output pixel
        input_frac_map = overlap_map / self.drop_area_pix
        # Fraction of each output pixel covered by drop
        output_frac_map = overlap_map / 1.0

        return input_frac_map, output_frac_map

    def drizzle(self, xfibre_all, yfibre_all):
        """Compute a mapping from fibre drops to output pixels for all given fibre locations."""
            
        # Increment the drizzle counter
        self.n_drizzle = self.n_drizzle + 1

        if (np.allclose(xfibre_all,self._last_drizzle_x, rtol=0,atol=self.drizzle_update_tol) and
            np.allclose(yfibre_all,self._last_drizzle_y, rtol=0,atol=self.drizzle_update_tol)):
            # We've been asked to recompute an asnwer that is less than the tolerance to recompute
            return self.drop_to_pixel, self.pixel_coverage
        else:
            self.n_drizzle_recompute = self.n_drizzle_recompute + 1
        
        for i_fibre, xfib, yfib in itertools.izip(itertools.count(), xfibre_all, yfibre_all):
    
            # Feed the grid_coordinates_x and grid_coordinates_y fibre positions to the overlap_maps instance.
            drop_to_pixel_fibre, pixel_coverage_fibre=self.single_overlap_map(xfib, yfib)
    
            # Padding with NaNs instead of zeros (do I really need to do this? Probably not...)
            drop_to_pixel_fibre[np.where(drop_to_pixel_fibre < epsilon)]=np.nan
            pixel_coverage_fibre[np.where(pixel_coverage_fibre < epsilon)]=np.nan
    
            self.drop_to_pixel[:,:,i_fibre]=drop_to_pixel_fibre
            self.pixel_coverage[:,:,i_fibre]=pixel_coverage_fibre
    
        self._last_drizzle_x = xfibre_all
        self._last_drizzle_y = yfibre_all
    
        return self.drop_to_pixel, self.pixel_coverage

def WCS_position(myIFU,object_flux_cube,object_name,band,plot=False,write=False,nominal=False,
                 remove_thput_file=True):
    """Wrapper for WCS_position_coords, extracting coords from IFU.
    
    This function cross-correlates a g-band convolved SAMI cube with its
    respective SDSS g-band image and pins down the positional WCS for the
    central spaxel of the cube.
    """

    # Get Object RA + DEC from fibre table (this is the input catalogues RA+DEC in deg)
    object_RA = np.around(myIFU.obj_ra[myIFU.n == 1][0], decimals=6)
    object_DEC = np.around(myIFU.obj_dec[myIFU.n == 1][0], decimals=6)
    
    # Build wavelength axis.
    CRVAL3 = myIFU.crval1
    CDELT3 = myIFU.cdelt1
    Nwave  = np.shape(object_flux_cube)[0]
    
    # -- crval3 is middle of range and indexing starts at 0.
    # -- this wave-axis agrees with QFitsView interpretation.
    CRVAL3a = CRVAL3 - ((Nwave-1)/2)*CDELT3
    wave = CRVAL3a + CDELT3*np.arange(Nwave)
        
    object_flux_cube = np.transpose(object_flux_cube, (2,0,1))
    
    return WCS_position_coords(object_RA, object_DEC, wave, object_flux_cube,
                               object_name, band, plot=plot, write=write,
                               nominal=nominal)


def WCS_position_coords(object_RA, object_DEC, wave, object_flux_cube, object_name, band,
                        plot=False, write=False, nominal=False, remove_thput_file=True):
    """Equate the WCS position information from a cross-correlation between a 
    g-band SAMI cube and a g-band SDSS image."""

    if nominal:
        img_crval1 = object_RA
        img_crval2 = object_DEC
        xcube = size_of_grid
        ycube = size_of_grid
        img_cdelt1 = -1.0 * output_pix_size_arcsec / 3600.0
        img_cdelt2 = output_pix_size_arcsec / 3600.0
    else:

        # Get SDSS g-band throughput curve
        if not os.path.isfile("sdss_"+str(band)+".dat"):
            urllib.urlretrieve("http://www.sdss.org/dr3/instruments/imager/filters/"+str(band)+".dat", "sdss_"+str(band)+".dat")
        
        # and convolve with the SDSS throughput
        sdss = ascii.read("SDSS_"+str(band)+".dat", quotechar="#", names=["wave", "pt_secz=1.3", "ext_secz=1.3", "ext_secz=0.0", "extinction"])
        
        # re-grid g["wave"] -> wave
        thru_regrid = griddata(sdss["wave"], sdss["ext_secz=1.3"], wave, method="cubic", fill_value=0.0)
        
        # initialise a 2D simulated g' band flux array.
        len_axis = np.shape(object_flux_cube)[1]
        Nwave = len(wave)
        reconstruct = np.zeros((len_axis,len_axis))
        tester = np.zeros((len_axis,len_axis))
        data_bit = np.zeros((Nwave,len_axis,len_axis))
        
        # Sum convolved flux:
        for i in range(Nwave):
            data_bit[i] = object_flux_cube[i]*thru_regrid[i]
        
        reconstruct = np.nansum(data_bit,axis=0) # not absolute right now
        reconstruct[np.isnan(reconstruct)] = 0. # replacing nan with 0.0
        reconstruct[reconstruct < 0] = 0.       # replacing negative fluxes with 0.0
        
        cube_image = reconstruct
        xcube = len(cube_image[0])
        ycube = len(cube_image[1])
        cube_image_crop = cube_image[(len(cube_image[0])/2)-10:(len(cube_image[0])/2)+10,(len(cube_image[1])/2)-10:(len(cube_image[1])/2)+10]
        cube_image_crop = sp.ndimage.zoom(cube_image_crop, 5, order=3)
        cube_image_crop_norm = (cube_image_crop - np.min(cube_image_crop))/np.max(cube_image_crop - np.min(cube_image_crop))
        
        # Check if the user supplied a red RSS file, throw exception.
        if np.array_equal(cube_image, tester):
            raise SystemExit("All values are zero: please provide the cube corresponding to the requested spectral band of the image!")

    ##########
        
        cube_size = np.around((size_of_grid*output_pix_size_arcsec)/3600, decimals=6)
        
        # Get SDSS Image
        if not os.path.isfile(str(object_name)+"_SDSS_"+str(band)+".fits"):
            getSDSSimage(object_name=object_name, RA=object_RA, DEC=object_DEC, 
                         band=str(band), size=cube_size, number_of_pixels=size_of_grid)
        
        # Open SDSS image and extract data & header information
        image_file = pf.open(str(object_name)+"_SDSS_"+str(band)+".fits")
        image_data = image_file['Primary'].data

        
        image_header = image_file['Primary'].header
        img_crval1 = float(image_header['CRVAL1']) #RA
        img_crval2 = float(image_header['CRVAL2']) #DEC
        img_crpix1 = float(image_header['CRPIX1']) #Reference x-pixel
        img_crpix2 = float(image_header['CRPIX2']) #Reference y-pixel
        img_cdelt1 = float(image_header['CDELT1']) #Delta RA
        img_cdelt2 = float(image_header['CDELT2']) #Delta DEC

        SDSS_image = image_data
        SDSS_image_crop = SDSS_image[(len(SDSS_image[0])/2)-10:(len(SDSS_image[0])/2)+10,(len(SDSS_image[1])/2)-10:(len(SDSS_image[1])/2)+10]
        SDSS_image_crop_norm = (SDSS_image_crop - np.min(SDSS_image_crop))/np.max(SDSS_image_crop - np.min(SDSS_image_crop))

        ##########

    if (not nominal) and np.size(np.where(image_data == 0.0)) != 2*np.size(image_data):
        # Cross-correlate normalised SAMI-cube g-band image and SDSS g-band image
        WCS_flag = 'SDSS'
        crosscorr_image = sp.signal.correlate2d(SDSS_image_crop_norm, cube_image_crop_norm)

        # 2D Gauss Fit the cross-correlated cropped image
        crosscorr_image_1d = np.ravel(crosscorr_image)
        #use for loops to recover indicies in x and y positions of flux values
        x_pos = []
        y_pos = []
        for i in xrange(np.shape(crosscorr_image)[0]):
            for j in xrange(np.shape(crosscorr_image)[1]):
                x_pos.append(i)
                y_pos.append(j)
        x_pos=np.array(x_pos)
        y_pos=np.array(y_pos)

        #define guess parameters for TwoDGaussFitter:
        amplitude = max(crosscorr_image_1d)
        mean_x = (np.shape(crosscorr_image)[0])/2
        mean_y = (np.shape(crosscorr_image)[1])/2
        sigma_x = 5.0 
        sigma_y = 6.0 
        rotation = 60.0 
        offset = 4.0
        p0 = [amplitude, mean_x, mean_y, sigma_x, sigma_y, rotation, offset]

        # call SAMI TwoDGaussFitter
        GF2d = fitting.TwoDGaussFitter(p0, x_pos, y_pos, crosscorr_image_1d)
        # execute gauss fit using
        GF2d.fit()
        GF2d_xpos = GF2d.p[2]
        GF2d_ypos = GF2d.p[1]

        # reconstruct the fit
        GF2d_reconstruct=GF2d(x_pos, y_pos)

        x_shape = len(crosscorr_image[0])
        y_shape = len(crosscorr_image[1])
        x_offset_pix = GF2d_xpos - x_shape/2
        y_offset_pix = GF2d_ypos - y_shape/2
        x_offset_arcsec = -x_offset_pix * output_pix_size_arcsec/5
        y_offset_arcsec = y_offset_pix * output_pix_size_arcsec/5
        x_offset_degree = ((x_offset_arcsec/3600)/24)*360
        y_offset_degree = (y_offset_arcsec/3600)
    
    else:
        WCS_flag = 'Nominal'
        y_offset_degree = 0.0
        x_offset_degree = 0.0

        # Create dictionary of positional WCS
    if isinstance(xcube/2, int):
            WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2 + 0.5), 
                     "CRPIX2":(ycube/2 + 0.5), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"DEGREE", "CTYPE2":"DEGREE"}
    else:
            WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2), 
                     "CRPIX2":(ycube/2), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"DEGREE", "CTYPE2":"DEGREE"}


##########

    # Remove temporary files
    if remove_thput_file and os.path.exists("sdss_"+str(band)+".dat"):
        os.remove("sdss_"+str(band)+".dat")
    if os.path.exists(str(object_name)+"_SDSS_"+str(band)+".fits"):
        os.remove(str(object_name)+"_SDSS_"+str(band)+".fits")
    
    return WCS_pos,WCS_flag


def update_WCS_coords(filename, nominal=False, remove_thput_file=True):
    """Recalculate the WCS data in a SAMI datacube."""
    # Pick out the relevant data
    header = pf.getheader(filename)
    ra = (header['CRVAL1'] + 
          (1 + np.arange(header['NAXIS1']) - header['CRPIX1']) * 
          header['CDELT1'])
    dec = (header['CRVAL2'] + 
           (1 + np.arange(header['NAXIS2']) - header['CRPIX2']) * 
           header['CDELT2'])
    wave = (header['CRVAL3'] + 
            (1 + np.arange(header['NAXIS3']) - header['CRPIX3']) * 
            header['CDELT3'])
    object_RA = np.mean(ra)
    object_DEC = np.mean(dec)
    object_flux_cube = pf.getdata(filename)
    object_name = header['NAME']
    if header['GRATID'] == '580V':
        band = 'g'
    elif header['GRATID'] == '1000R':
        band = 'r'
    else:
        raise ValueError('Could not identify band. Exiting')
    # Calculate the WCS
    WCS_pos, WCS_flag = WCS_position_coords(
        object_RA, object_DEC, wave, object_flux_cube, object_name, band,
        nominal=nominal, remove_thput_file=remove_thput_file)
    # Update the file
    hdulist = pf.open(filename, 'update', do_not_scale_image_data=True)
    header = hdulist[0].header
    for key, value in WCS_pos.items():
        header[key] = value
    header['WCS_SRC'] = WCS_flag
    hdulist.close()
    return

def getSDSSimage(object_name="unknown", RA=0, DEC=0, band="g", size=0.006944, 
                 number_of_pixels=50, projection="Tan", url_show="False"):
    """This function queries the SDSS surver at skyview.gsfc.nasa.gov and returns an image
    with a user supplied set of parameters. 

    A full description of the input parameters is given at -
    http://skyview.gsfc.nasa.gov/docs/batchpage.html

    The parameters that can be set here are:

        name - object name to include in file name for reference
        RA - in degrees
        DEC - in degrees
        band - u,g,r,i,z filters
        size - size of side of image in degrees
        number_of_pixels - number of pixels of side of image (i.e 50 will return 50x50)
        projection - 2D mapping of onsky projection. Tan is standard.
        url_show - this is a function variable if the user wants the url printed to terminal

    """
    
    # Construct URL
    RA = str(RA).split(".")
    DEC = str(DEC).split(".")
    size = str(size).split(".")
    
    URL = "http://skyview.gsfc.nasa.gov//cgi-bin/pskcall?position="+(str(RA[0])+"%2e"+str(RA[1])+"%2c"+str(DEC[0])+"%2e"+str(DEC[1])+
        "&Survey=SDSSdr7"+str(band)+"&size="+str(size[0])+"%2e"+str(size[1])+"&pixels="+str(number_of_pixels)+"&proj="+str(projection))
    
    # Get SDSS image
    urllib.urlretrieve(str(URL), str(object_name)+"_SDSS_"+str(band)+".fits")
    
    if url_show=="True":
        print ("SDSS "+str(band)+"-band image of object "+str(object_name)+" has finished downloading to the working directory with the file name: "
               +str(object_name)+"_SDSS_"+str(band)+".fits")
        
        print "The URL for this object is: ", URL
