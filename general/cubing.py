import pylab as py
import numpy as np
import scipy as sp

import astropy.io.fits as pf
import astropy.wcs as pw

import itertools
import os

# Cross-correlation function from scipy.signal (slow)
from scipy.signal import correlate

# Stats functions from scipy
from scipy.stats import stats

# Import the sigma clipping algorithm from astropy
from astropy.stats import sigma_clip

# Attempt to import bottleneck to improve speed, but fall back to old routines
# if bottleneck isn't present
try:
    from bottleneck import nanmedian
except:
    nanmedian = stats.nanmedian
    print("Not Using bottleneck: Speed will be improved if you install bottleneck")


# Utils code.
from .. import utils
from .. import samifitting as fitting

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

def dithered_cubes_from_rss_files(inlist, sample_size=0.5, objects='all', plot=True, write=False):
    """A wrapper to make a cube from reduced RSS files. Only input files that go together - ie have the same objects."""

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

    # For first files get the names of galaxies observed - assuming these are the same in all RSS files.
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
        
        ifu_list = []
        
        for j in xrange(len(files)):
            ifu_list.append(utils.IFU(files[j], name, flag_name=True))
    
        flux_cube, var_cube, weight_cube = dithered_cube_from_rss(ifu_list,sample_size=sample_size, plot=plot, write=write)
        
        
        if write==True:
            # NOTE - At this point we will want to generate accurate WCS information and create a proper header.
            # The addition of ancillary data to the header will be valuable. I think the creation of the header should
            # be separated off into a separate function (make_cube_header?)
            
            # This could proceed several ways, depending on what the WCS coders want from us and what we want back.
            # Need to liase.
    
            # For now create a rudimentary WCS with ONLY the correct wavelength scale.
    
            # First get some info from one of the headers.
            list1=pf.open(files[0])
            hdr=list1[0].header
    
            grating=hdr['GRATID']
    
            if grating=='580V':
                print "Files are blue."
                arm='blue'
                
            elif grating=='1000R':
                print "Files are red."
                arm='red'
                
            else:
                print "I've no idea what grating you used, you crazy person!"
                arm='unknown'
    
            # Create the wcs.
            wcs_new=pw.WCS(naxis=3)
            wcs_new.wcs.crpix = [1, 1, hdr['CRPIX1']]
            wcs_new.wcs.cdelt = np.array([1, 1, hdr['CDELT1']])
            wcs_new.wcs.crval = [1, 1, hdr['CRVAL1']]
            wcs_new.wcs.ctype = ["PIXEL", "PIXEL", hdr['CTYPE1']]
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
            # Note - there is a 90-degree rotation in the cube, which I can't track down. I'm rolling the axes before
            # writing the FITS files to compensate.
            hdu1=pf.PrimaryHDU(np.transpose(flux_cube, (2,0,1)), hdr_new)
            hdu2=pf.ImageHDU(np.transpose(var_cube, (2,0,1)), name='VARIANCE')
            hdu3=pf.ImageHDU(np.transpose(weight_cube, (2,0,1)), name='WEIGHT')
                
            # Put individual HDUs into a HDU list
            hdulist=pf.HDUList([hdu1,hdu2,hdu3])
        
            # Write to FITS file.
            outfile_name=str(name)+'_'+str(arm)+'_'+str(len(files))+'.fits'
            print "Writing", outfile_name
            "--------------------------------------------------------------"
            hdulist.writeto(outfile_name)
    
            # Close the open file
            list1.close()


def dithered_cube_from_rss(ifu_list, sample_size=0.5, plot=True, write=False, offsets='fit'):
        
    # When resampling need to know the size of the grid in square output pixels
    size_of_grid=40 # should prob calculate this somehow?? Maybe when the cube is created, collapse

    # Create an instance of fibre_overlap_map for use later to create individual overlap maps for each fibre.
    # The attributes of this instance don't change from ifu to ifu.
    overlap_maps=fibre_overlap_map(sample_size, size_of_grid)

    diagnostic_info = {}

    n_obs = len(ifu_list)

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

    for j in xrange(len(ifu_list)):

        # Get the data.
        galaxy_data=ifu_list[j]
        
        # Smooth the spectra and median.
        data_smoothed=np.zeros_like(galaxy_data.data)
        for p in xrange(np.shape(galaxy_data.data)[0]):
            data_smoothed[p,:]=utils.smooth(galaxy_data.data[p,:], 10) #default hanning

        # Collapse the smoothed data over a large wavelength range to get continuum data
        data_med=stats.nanmedian(data_smoothed[:,300:1800], axis=1)

        # Pick out only good fibres (i.e. those allocated as P)
        goodfibres=np.where(galaxy_data.fib_type=='P')
        x_good=galaxy_data.x_microns[goodfibres]
        y_good=galaxy_data.y_microns[goodfibres]
        data_good=data_med[goodfibres]
    
        # First try to get rid of the bias level in the data, by subtracting a median
        data_bias=np.median(data_good)
        if data_bias<0.0:
            data_good=data_good-data_bias
            
        # Mask out any "cold" spaxels - defined as negative, due to poor throughtput calibration from CR taking out 5577.
        msk_notcold=np.where(data_good>0.0)
    
        # Apply the mask to x,y,data
        x_good=x_good[msk_notcold]
        y_good=y_good[msk_notcold]
        data_good=data_good[msk_notcold]

        print("data_good.shape: ",np.shape(data_good))

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
            xm=galaxy_data.x_microns - np.mean(galaxy_data.x_microns)
            ym=galaxy_data.y_microns - np.mean(galaxy_data.y_microns)
    
        xfibre_all.append(xm)
        yfibre_all.append(ym)

        data_all.append(galaxy_data.data)
        var_all.append(galaxy_data.var)

        ifus_all.append(galaxy_data.ifu)


    print ifus_all

    xfibre_all=np.asanyarray(xfibre_all)
    yfibre_all=np.asanyarray(yfibre_all)
    data_all=np.asanyarray(data_all)
    var_all=np.asanyarray(var_all)

    # For debugging run on only a subset of the data
    ##data_all=data_all[:,:,500:700]

    ifus_all=np.asanyarray(ifus_all)

    print "Data shape", np.shape(data_all)

    # Use data_all to find a bad pixel mask per IFU
    badpix_all=np.copy(data_all)
    badpix_all_final=np.zeros_like(badpix_all)

    # @TODO: This next section of code is wrong: The order of the ifus_used list is not necessarily the same
    #        as the order of the other lists it addresses here.

    # Now, find what the discrete IFUs used were
    ifus_used=list(set(ifus_all))
    
    k0=0
    k1=0
    for j in xrange(len(ifus_used)):

        ifu_single=ifus_used[j]
        
        msk_ifu_single=np.where(ifus_all==ifu_single)
        
        k1=k0+(np.shape(msk_ifu_single)[1])

        # Note badpix_all and ifu_all are in the same order.
        badpix_ifu_single=np.squeeze(badpix_all[msk_ifu_single,:,:])
        badpix_ifu_single_med=stats.nanmedian(badpix_ifu_single, axis=0)

        # Replicate this y times where y is the number of frames the galaxy was observed in this ifu.
        badpix_ifu_final=np.tile(badpix_ifu_single_med, (np.shape(msk_ifu_single)[1],1,1))

        # Put the replicated array into the final array.
        badpix_all_final[k0:k1,:,:]=badpix_ifu_final

        k0=k1

    # Convert the bad pixel mask into ones and zeros.
    badpix_all_final[np.where(np.isfinite(badpix_all_final))]=1.0
    badpix_all_final[np.where(np.isnan(badpix_all_final))]=0.0

    # Reshape the arrays
    xfibre_all=np.reshape(xfibre_all,(np.shape(xfibre_all)[0]*np.shape(xfibre_all)[1]))
    yfibre_all=np.reshape(yfibre_all,(np.shape(yfibre_all)[0]*np.shape(yfibre_all)[1]))
    data_all=np.reshape(data_all,(np.shape(data_all)[0]*np.shape(data_all)[1], np.shape(data_all)[2]))
    var_all=np.reshape(var_all,(np.shape(var_all)[0]*np.shape(var_all)[1], np.shape(var_all)[2]))

    badpix_all_final=np.reshape(badpix_all_final,(np.shape(badpix_all_final)[0]*np.shape(badpix_all_final)[1],
                                                  np.shape(badpix_all_final)[2]))
    
    # Empty array for all overlap maps - i.e. need one for each fibre!
    overlap_array=np.zeros((size_of_grid, size_of_grid, np.shape(xfibre_all)[0]))
    output_frac_array=np.zeros((size_of_grid, size_of_grid, np.shape(xfibre_all)[0]))

    #print "Galaxy is", name

    # An array of zeros, same size as the data.
    data_norm=np.zeros_like(data_all)
    print np.shape(data_all)
    
    # Now, want to normalise each spectrum by its median
    for ii in xrange(n_obs):

        spectrum = data_all[ii,:]
        med_spectrum=stats.nanmedian(spectrum)

        spec_norm=spectrum/med_spectrum

        # Put into the array
        data_norm[ii,:]=spec_norm
    
    
    # Now feed all x,y fibre position values to the overlap_maps class instance.
    for p, xfib, yfib in itertools.izip(itertools.count(), xfibre_all, yfibre_all):

        # Feed the x and y fibre positions to the overlap_maps instance.
        input_frac_map_fib, output_frac_map_fib=overlap_maps.create_overlap_map(xfib, yfib)

        # These lines are ONLY for debugging. The plotting and overwriting is very slow! 
        #py.imshow(input_frac_map_fib, origin='lower', interpolation='nearest')
        #py.draw()

        # Padding with NaNs instead of zeros (do I really need to do this? Probably not...)
        input_frac_map_fib[np.where(input_frac_map_fib==0)]=np.nan
        output_frac_map_fib[np.where(output_frac_map_fib==0)]=np.nan

        overlap_array[:,:,p]=input_frac_map_fib
        output_frac_array[:,:,p]=output_frac_map_fib

    print "OVERLAP ARRAY:", np.shape(overlap_array)
    #return

    # Create an empty weight cube
    weight_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))

    # Now create a new array to hold the final data cube and build it slice by slice
    flux_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))
    var_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))
    
    print("data_all.shape: ", np.shape(data_all))

    diagnostic_info['unmasked_pixels_after_sigma_clip'] = 0
    diagnostic_info['unmasked_pixels_before_sigma_clip'] = 0

    diagnostic_info['n_pixels_sigma_clipped'] = []
           
    # This loops over wavelength slices (e.g., 2048). 
    for l in xrange(np.shape(data_all)[1]):

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
        

        print l

        # Normalised spectra.
        norm_rss_slice=data_norm[:,l]

        # Spectra and variance.
        data_rss_slice=data_all[:,l]
        var_rss_slice=var_all[:,l]
        badpix_rss_slice=badpix_all_final[:,l]

        # Weight map parameters for a single slice - copy the output frac array each time.
        # This will NOT be correct for each slice when ADC is implemented, will have to calculate it several times.
        weight_grid_slice=np.copy(output_frac_array)

        # ----------------------------------------------------------------------------------------------------------
        # Find the pixel which need to be clipped using the normalised spectra.

        # Make the normalised slice array (need to determine deviant pixels using this array not the good data)
        norm_grid_slice_fibres=overlap_array*norm_rss_slice
        
        data_grid_slice_fibres = overlap_array*data_rss_slice
        
        unmasked_pixels_before_clipping = np.isfinite(data_grid_slice_fibres).sum()
        
        # Sigma clip it - pixel by pixel and make a master mask
        # array. Current clipping, sigma=5 and 1 iteration.
        
        #mask_master = sigma_clip_mask_slice_fibres(norm_grid_slice_fibres/weight_grid_slice)
        
        
        
        mask_master = sigma_clip_mask_slice_fibres(data_grid_slice_fibres/weight_grid_slice)
        
        
        mask_master=np.asanyarray(mask_master).astype(float)
        mask_master[np.where(mask_master==0.0)]=np.nan
        # NOTE - mask_master is an array of ones and NaNs.

        # Apply the mask to the data slice array and variance slice array
        data_grid_slice_fibres=overlap_array*data_rss_slice*mask_master
        var_grid_slice_fibres=(overlap_array*overlap_array)*var_rss_slice*mask_master

        unmasked_pixels_after_clipping = np.isfinite(data_grid_slice_fibres).sum()

        diagnostic_info['n_pixels_sigma_clipped'] = \
            unmasked_pixels_before_clipping - unmasked_pixels_after_clipping
        diagnostic_info['unmasked_pixels_before_sigma_clip'] += unmasked_pixels_before_clipping
        diagnostic_info['unmasked_pixels_after_sigma_clip'] += unmasked_pixels_after_clipping

        print("Pixels Clipped: {0} ({1}%)".format(\
            unmasked_pixels_before_clipping - unmasked_pixels_after_clipping,
            (unmasked_pixels_before_clipping - unmasked_pixels_after_clipping) / float(unmasked_pixels_before_clipping)
            ))

        # Now, at this stage want to identify ALL positions in the data array (before we collapse it) where there
        # are NaNs (from clipping, cosmics flagged by 2dfdr and bad columns flagged by 2dfdr). This allows the
        # correct weight map to be created for the wavelength slice in question.
        data_total_nanmask=np.copy(data_grid_slice_fibres)
        data_total_nanmask[np.where(np.isfinite(data_total_nanmask))]=1.0

        # data_total_nanmask should now be an array of ones and NaNs. This should be applied to the weight map.
        weight_grid_slice_fibres=weight_grid_slice*data_total_nanmask

        # Collapse the slice arrays
        data_grid_slice_final=np.nansum(data_grid_slice_fibres, axis=2)
        var_grid_slice_final=np.nansum(var_grid_slice_fibres, axis=2)
        weight_grid_slice_final=np.nansum(weight_grid_slice_fibres, axis=2)
        
        flux_cube[:,:,l]=data_grid_slice_final
        var_cube[:,:,l]=var_grid_slice_final
        weight_cube[:,:,l]=weight_grid_slice_final

    # Now need to scale the cubes by the weight cube
    flux_cube=flux_cube/weight_cube # flux cube scaling by weight map
    image=stats.nanmedian(flux_cube, axis=2)

    var_cube=var_cube/(weight_cube*weight_cube) # variance cube scaling by weight map

    return flux_cube, var_cube, weight_cube, diagnostic_info

def sigma_clip_mask_slice_fibres(grid_slice_fibres):
    """Return a mask with outliers removed."""

    med = nanmedian(grid_slice_fibres, axis=2)
    var = utils.mad(grid_slice_fibres, axis=2)

    # We rearrange the axes so that numpy.broadcasting works in the subsequent
    # operations. See: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    t_grid_slice_fibres = np.transpose(grid_slice_fibres, axes=(2,0,1))
    
    mask = np.transpose( 
            np.less_equal(t_grid_slice_fibres - med, var * 5 **2),
            axes=(1,2,0))
    
    return mask
          
class fibre_overlap_map:
    """Make an overlap map for a single fibre. This is the same at all lambda slices for that fibre (neglecting
    DAR)""" 

    def __init__(self, sample_size_arcsec, size_of_grid):

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

    def create_overlap_map(self, fibrex, fibrey):

        # Map fibre positions onto pixel positions in the output grid.
        self.xfib=(fibrex-self.x[0])/self.dx
        self.yfib=(fibrey-self.y[0])/self.dx
        
        # Create the overlap map from the circ.py code
        overlap_map=utils.resample_circle(self.size_of_grid, self.size_of_grid, self.xfib, self.yfib, \
                                         self.oversample/2.0)
        
        input_frac_map=overlap_map/self.fib_area_pix # Fraction of input pixel in each output pixel
        output_frac_map=overlap_map/1.0 # divided by area of ind. sq. (output) pixel.

        return input_frac_map, output_frac_map
    
