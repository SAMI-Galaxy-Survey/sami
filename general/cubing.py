import pylab as py
import numpy as np
import scipy as sp

import astropy.io.fits as pf
import astropy.wcs as pw

import itertools

# Cross-correlation function from scipy.signal (slow)
from scipy.signal import correlate

# Stats functions from scipy
from scipy.stats import stats

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

def dithered_cube_from_rss(inlist, sample_size=0.5, objects='all', plot=True, write=False):
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

    #N=len(files)

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

    # When resampling need to know the size of the grid in square output pixels
    size_of_grid=60 # should prob calculate this somehow??

    # Create an instance of fibre_overlap_map for use later to create individual overlap maps for each fibre.
    # The attributes of this instance don't change from ifu to ifu.
    overlap_maps=fibre_overlap_map(sample_size, size_of_grid)

    # For each ifu find the centroid etc.
    for name in object_names:

        print
        print "--------------------------------------------------------------"
        print "Starting with object:", name

        # Pull out the data for the galaxy in question.
        
        if plot==True:
            # Make a figure to plot the images and fits to galaxy position.
            f1=py.figure()
    
            if len(files)==1:
                r=1
                c=1
            elif len(files)==2:
                r=1
                c=2
            elif len(files)==3:
                r=1
                c=3
            elif len(files)==4:
                r=2
                c=2
            elif len(files)>3 and len(files)<=6:
                r=2
                c=3
            elif len(files)>6 and len(files)<=9:
                r=3
                c=3
            elif len(files)>9 and len(files)<=12:
                r=4
                c=3
            elif len(files)>12 and len(files)<=16:
                r=4
                c=4

        # Empty lists for positions and data. Could be arrays, might be faster? Should test...
        xfibre_all=[]
        yfibre_all=[]
        data_all=[]
        var_all=[]

        ifus_all=[]

        # Read in data for each file
        for j in xrange(len(files)):

            # Get the data.
            galaxy_data=utils.IFU(files[j], name, flag_name=True)
            
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
        
            # First try to get rid of the bias laval in the data, by subtracting a median
            data_bias=np.median(data_good)
            if data_bias<0.0:
                data_good=data_good-data_bias
                
            # Mask out any "cold" spaxels - defined as negative, due to poor throughtput calibration from CR taking out 5577.
            msk_notcold=np.where(data_good>0.0)
        
            # Apply the mask to x,y,data
            x_good=x_good[msk_notcold]
            y_good=y_good[msk_notcold]
            data_good=data_good[msk_notcold]

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
            
            if plot==True:
                
                # Plot the data (scatter plot).
                ax=f1.add_subplot(r,c,j+1)
                scatterplot=ax.scatter(x_good,y_good,s=24,c=data_good)
        
                x0=np.median(galaxy_data.x_microns)-24*overlap_maps.dx
                y0=np.median(galaxy_data.y_microns)-24*overlap_maps.dx
        
                xlin=x0+overlap_maps.dx*np.arange(50)
                ylin=y0+overlap_maps.dx*np.arange(50)
                
                # Reconstruct the model.
                model=np.zeros((len(xlin), len(ylin))) # 2d Gaussian
                
                # Reconstruct the Gaussian fit.
                for ii in xrange(len(xlin)):
                    x_val=xlin[ii]
                    for jj in xrange(len(ylin)):
                        y_val=ylin[jj]
                        model[ii,jj]=gf1.fitfunc(gf1.p, x_val, y_val)

                con=ax.contour(xlin, ylin, np.transpose(model), cmap=py.cm.winter)

                # Get rid of the tick labels.
                py.setp(ax.get_xticklabels(), visible=False)
                py.setp(ax.get_yticklabels(), visible=False)

            # Adjust the micron positions of the fibres - for use in making final cubes.
            xm=galaxy_data.x_microns-gf1.p[1]
            ym=galaxy_data.y_microns-gf1.p[2]
    
            xfibre_all.append(xm)
            yfibre_all.append(ym)

            data_all.append(galaxy_data.data)
            var_all.append(galaxy_data.var)

            ifus_all.append(galaxy_data.ifu)

        if plot==True:
            f1.suptitle(name)
            f1.subplots_adjust(wspace=0.0)

        print ifus_all

        xfibre_all=np.asanyarray(xfibre_all)
        yfibre_all=np.asanyarray(yfibre_all)
        data_all=np.asanyarray(data_all)
        var_all=np.asanyarray(var_all)

        ifus_all=np.asanyarray(ifus_all)

        print "Data shape", np.shape(data_all)

        # Use data_all to find a bad pixel mask per IFU
        badpix_all=np.copy(data_all)
        badpix_all_final=np.zeros_like(badpix_all)

        # Now, find what the discrete IFUs used were
        ifus_used=list(set(ifus_all))
        
        k0=0
        k1=0
        for j in xrange(len(ifus_used)):

            ifu_single=ifus_used[j]
            #print str(ifu_single)
            
            msk_ifu_single=np.where(ifus_all==ifu_single)
            #print np.shape(msk_ifu_single)[1]
            
            
            k1=k0+(np.shape(msk_ifu_single)[1])
            #print k0, k1

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

        ## # test only
        ## test=np.reshape(badpix_all_final, (np.shape(badpix_all_final)[0]*np.shape(badpix_all_final)[1], np.shape(badpix_all_final)[2]))
        ## print np.shape(test)
        ## f2=py.figure()

        ## ax2=f2.add_subplot(111)
        ## im2=ax2.imshow(test, origin='lower', interpolation='nearest')
        
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
        
        # Now feed all x,y values to the overlap_maps class instance.
        for p in xrange(len(xfibre_all)):
            xfib=xfibre_all[p]
            yfib=yfibre_all[p]

            # Feed the x and y fibre positions to the overlap_maps instance.
            input_frac_map_fib, output_frac_map_fib=overlap_maps.create_overlap_map(xfib, yfib)

            # These lines are ONLY for debugging. The plotting and overwriting is very slow! 
            #py.imshow(input_frac_map_fib, origin='lower', interpolation='nearest')
            #py.draw()

            input_frac_map_fib[np.where(input_frac_map_fib==0)]=np.nan
            output_frac_map_fib[np.where(output_frac_map_fib==0)]=np.nan

            overlap_array[:,:,p]=input_frac_map_fib
            output_frac_array[:,:,p]=output_frac_map_fib

            #del(input_frac_map_fib)
            #del(output_frac_map_fib)
            #print test.fib_area_pix

        # Create the final weight map. This is the same for each wavelength slice
        weight_map_final=np.nansum(output_frac_array, axis=2)
        #py.imshow(weight_map_final, interpolation='nearest', origin='lower')
        
        # Create a weight cube - for now containing the same weight map at each wavelength as positions of fibres
        # do not vary with wavelength. This will change when we are accounting for DAR.
        weight_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))
        #weightcube=weightcube*weight_map_final

        # Now create a new array to hold the final data cube and build it slice by slice
        flux_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))
        var_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))

        # Array to hold mosaiced bad pixel cube
        badpix_cube=np.zeros((size_of_grid, size_of_grid, np.shape(data_all)[1]))
        
        for l in xrange(np.shape(data_all)[1]):

            data_single_slice=data_all[:,l]
            var_single_slice=var_all[:,l]
            badpix_single_slice=badpix_all_final[:,l]
            #print np.shape(data_single_slice), np.shape(badpix_single_slice)
            
            #print np.shape(data_slice), np.shape(overlap_array)

            data_slice_array=overlap_array*data_single_slice
            var_slice_array=(overlap_array*overlap_array)*var_single_slice
            badpix_slice_array=overlap_array*badpix_single_slice
            #print np.shape(slices_array)

            data_slice_final=np.nansum(data_slice_array, axis=2)
            var_slice_final=np.nansum(var_slice_array, axis=2)
            badpix_slice_final=np.nansum(badpix_slice_array, axis=2)
            
            flux_cube[:,:,l]=data_slice_final
            var_cube[:,:,l]=var_slice_final
            weight_cube[:,:,l]=weight_map_final
            badpix_cube[:,:,l]=badpix_slice_final

        # badpix_cube is still not quite what I want, want to normalise each spectrum separately to get an idea of
        # where bad columns are spectrally for each spaxel.

        for mm in xrange(np.shape(badpix_cube)[0]):
            for nn in xrange(np.shape(badpix_cube)[1]):
                badpix_spectrum=badpix_cube[mm,nn,:]

                if np.isnan(np.nansum(badpix_spectrum)):
                    pass
                else:
                    badpix_spectrum_max=np.nanmax(badpix_spectrum)
                    badpix_spectrum_norm=badpix_spectrum/badpix_spectrum_max

                    badpix_cube[mm,nn,:]=badpix_spectrum_norm

        # Mask the bad columns
        msk_badcols=np.where(badpix_cube!=1.0)
        badpix_cube[msk_badcols]=np.nan

        flux_cube=flux_cube*badpix_cube/weight_cube # flux cube scaling by weight map
        image=stats.nanmedian(flux_cube, axis=2)

        var_cube=var_cube*badpix_cube/(weight_cube*weight_cube) # variance cube scaling by weight map

        #py.imshow(image, interpolation='nearest', origin='lower')

        #Write the bad pixel cube
        hdu1=pf.PrimaryHDU(np.transpose(badpix_cube, (2,0,1)))

        # Put individual HDUs into a HDU list
        hdulist=pf.HDUList([hdu1])
        
        #hdulist.writeto('TEST_2_badpix.fits')

        if write==True:
            # NOTE - At this point we will want to generate accurate WCS information and create a proper header.
            # This could proceed several ways, depending on what the WCS coders want from us and what we want back.
            # Need to liase.

            # For now create a rudimentary WCS with at least the correct wavelength scale.

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

            # Put the RSS files into the header
            for num in xrange(len(files)):
                #print files[num]
                rss_key='HIERARCH RSS_FILE '+str(num+1)
                rss_string='Input RSS file '+str(num+1)
                hdr_new.update(rss_key, files[num], rss_string)
            
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

def _plotcentroids(inlist, probe):
    """A test of the new samifitting code."""

    f1=py.figure()
    
    # The list of all the RSS files input by the user.
    files=[]
    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])

        files.append(np.str(cols[0]))

    N=len(files)
    r=3.0
    c=np.ceil(N/r)

    # Read in data for each file
    for j in xrange(len(files)):

        # Pick out data for that particular file.
        probe_properties=utils.IFU(files[j],probe)
        #xm,ym,xpos,ypos,data,var,num,fib_type,P,name=utils_IV.IFU_pick(files[j], ifu)
        
        # Collapse the data over a large wavelength range to get continuum data
        # Smooth the spectra and median.
        data_smoothed=np.zeros_like(probe_properties.data)
        for p in xrange(np.shape(probe_properties.data)[0]):
            data_smoothed[p,:]=utils.smooth(probe_properties.data[p,:], 10) #default hanning
            
            data_med=stats.nanmedian(data_smoothed[:,300:1800], axis=1)
            
        # Pick out only good fibres (i.e. those allocated as P)
        goodfibres=np.where(probe_properties.fib_type=='P')
        x_good=probe_properties.x_microns[goodfibres]
        y_good=probe_properties.y_microns[goodfibres]
        data_good=data_med[goodfibres]
        
        # First try to get rid of the bias laval in the data, by subtracting a median
        data_bias=np.median(data_good)
        if data_bias<0.0:
            # print data_bias
            data_good=data_good-data_bias
            
        # Mask out any "cold" spaxels - defined as negative, due to poor throughtput calibration from CR taking out 5577.
        msk_notcold=np.where(data_good>0.0)
        
        # Apply the mask to x,y,data
        x_good=x_good[msk_notcold]
        y_good=y_good[msk_notcold]
        data_good=data_good[msk_notcold]
        
        # Find the centroid for the file - using micron positions and smoothed, median collapsed data.
        # Returned fitted_params in format [peak_flux, x_position, y_position, x_sigma, y_sigma, angle, offset]

        # Fit parameter estimates from a crude centre of mass
        com_distr=utils.comxyz(x_good,y_good,data_good) #-data_bias)
    
        # First guess sigma
        sigx=100.0
        
        # Peak height guess could be closest fibre to com position.
        dist=(x_good-com_distr[0])**2+(y_good-com_distr[1])**2 # distance between com and all fibres.
        
        # First guess Gaussian parameters.
        p0=[data_good[np.sum(np.where(dist==np.min(dist)))], com_distr[0], com_distr[1], sigx, sigx, 45.0, 0.0]
   
        gf1=fitting.TwoDGaussFitter(p0, x_good, y_good, data_good)
        gf1.fit()

        p_out=gf1.p
        print p0
        print p_out
    
        #params_guess, gfit=find_centroid(x_good,y_good,data_good)
        
        # Make two figures.
        ax=f1.add_subplot(r,c,j+1)
        scatterplot=ax.scatter(x_good,y_good,s=24,c=data_good)
        #start_pos=ax.scatter(fit_guess[1], fit_guess[2], c='k')
        # Reconstruct the Gaussian
        # Make an even (x,y) grid spanning the entire range and reconstruct the Gaussian on it.
        
        # dx=overlap_maps.dx #fudge the fineness of the grid to make a pretty plot

        overlap_maps=fibre_overlap_map(0.5, 60)

        #x0=np.median(xm)-24*overlap_maps.dx
        #y0=np.median(ym)-24*overlap_maps.dx
        
        #xlin=x0+overlap_maps.dx*np.arange(50)
        #ylin=y0+overlap_maps.dx*np.arange(50)
        
        x0=np.median(probe_properties.x_microns)-24*overlap_maps.dx
        y0=np.median(probe_properties.y_microns)-24*overlap_maps.dx
        
        xlin=x0+overlap_maps.dx*np.arange(50)
        ylin=y0+overlap_maps.dx*np.arange(50)
        print np.shape(xlin)
        
        # Reconstruct the model.
        model=np.zeros((len(xlin), len(ylin))) # 2d Gaussian
        
        # Reconstruct the Gaussian fit.
        #print gfit
        #print gfit.p
        print gf1
        

        #print fit_out
        #print np.shape(xlin), np.shape(fit_out)
        for ii in xrange(len(xlin)):
            x_val=xlin[ii]
            for jj in xrange(len(ylin)):
                y_val=ylin[jj]
                model[ii,jj]=gf1.fitfunc(p_out, x_val, y_val)
                #model[ii,jj]=fit_out[x_val, y_val]

        print np.shape(model)
        print xlin, ylin, model
        con=ax.contour(xlin, ylin, np.transpose(model), cmap=py.cm.winter)

        # Get rid of the tick labels.
        py.setp(ax.get_xticklabels(), visible=False)
        py.setp(ax.get_yticklabels(), visible=False)
        
        #py.title(ifu)

    #pass
    
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
