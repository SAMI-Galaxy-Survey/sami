#########################################################################################
###                                                                                   ###
###################################--- sdss.py ---#######################################
###                                                                                   ###
#########################################################################################
#
# Written by: Samuel Richards & Iraklis Konstantopoulos
#
# Contact: Samuel Richards - samuel@physics.usyd.edu.au
# Maintainer: Iraklis Konstantopoulos - iraklis@aao.gov.au
#
# Version 5
# 22/07/2013
#
#
# Dependencies:
#
#       [1] sami.utils.ifu.py file has been edited to include GRP_MRA & GRP_MDEC, so
#           get Lisa to add them permanently. The three lines of code to be added just
#           before the "del hdulist" are:
#
#           -----
#           # Object RA & DEC
#           self.obj_ra=table_new.field('GRP_MRA')
#           self.obj_dec=table_new.field('GRP_MDEC')
#           -----
#
#       [2] Files are written/read in from this code, so check the files names match
#           throughout this code if you change them.
#
#
# Explanation of functions:
#
#   "RSS_SDSS_g"
#
#       This function convolves the sdss g-band filter with a Blue RSS frame to create
#       a collapse IFU g-band image.
#
#   "RSS_SDSS_resampled"
#   
#       This function extracts the SDSS g-band image of the object given using the
#       "GRP_MRA" & "GRP_MDEC" data in the RSS Fibre Table "FIBRES_IFU", and extracts
#       the flux for a given IFU footprint.
#
#   "residual_data"
#
#       This function creates residual maps of functions "RSS_SDSS_g" -
#       "RSS_SDSS_resampled". It outputs many variables that are used in the
#       optimisation script for plotting/writing.
#
#   "residual_function"
#
#       This function creates residual maps of functions "RSS_SDSS_g" -
#       "RSS_SDSS_resampled". It outputs one variable (the residual array) that is used
#       in the optimisation function.
#
#   "bestfit_SDSS_g"
#
#       This function performs a best fit on the residual_function. It uses the scipy
#       "leastsq" algorithm to do this with an initial guess of the minimum of the brute
#       force residual grid. At the moment it still somewhat struggles with supious
#       features like cosmic rays that give a high flux in an otherwise low flux fibre,
#       and high flux foreground objects, but it is very good with "normal" targets,
#       fitting in about 210secs per object per RSS. The output "p" is the offset in
#       arcsec, and the output RA and DEC are the equated RA and DECs for that frame in
#       degrees.
#
# "sim_SDSS_resampled"
#
#       This function is essentiall the same as "RSS_SDSS_resampled" but returns
#       variables with the same names as an RSS frame to supply the code with a simulated
#       frame instead of an RSS frame. This is used mainly in the testing of the code to
#       see how robust the optimisation algorithms are, as it should be able to find
#       itself.
#
#   "getSDSSimage"
#
#       This function queries the SDSS surver at skyview.gsfc.nasa.gov and returns an
#       image with a user supplied set of parameters, which are explained with the
#       function.
#
#   "getSDSSspectra"
#
#       NOTE: Not implemented, just a place holder with brief notes on what to do.
#
#   "gauss_kern" & "blur_image"
#
#       Following two functions from scipy cookbook to use a Gaussian kernal to blur an
#       image. It was taken from: http://www.scipy.org/Cookbook/SignalSmooth
# 
#   "sb" 
#
#       Plot a surface brightness map for an SDSS image. Takes arguments controlling the 
#       scale to which SB is extrapolated. Warning: this code does not resample images, 
#       but translates the flux of a given pixel to a surface brightness to the requested 
#       scale. Mental arithmetics required for interpretation. 
#
#########################################################################################

import numpy as np
import pylab as py
import scipy as sp
import sami.utils as utils
import sami.samifitting as sf
import astropy.io.fits as pf
import astropy.io.ascii as ascii
from scipy.interpolate import griddata
import scipy.optimize as optimize
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle
import urllib
import os
import datetime
py.ioff()

#########################################################################################
#
# "RSS_SDSS_g"
#
#   This function convolves the sdss g-band filter with a Blue RSS frame to create a
#   collapse IFU g-band image.
#

def RSS_SDSS_g(RSS_file="unknown", GAMA_ID="unknown", IFU=1, show="False", write="False"):

    # Get RSS file name without ".fits"
    RSS_file_name = RSS_file[(len(RSS_file)-18):(len(RSS_file)-5)]
    
    # Read RSS file (GAMA_ID-specific input).
    if GAMA_ID!="unknown":
        myIFU = utils.IFU(RSS_file, GAMA_ID)
    else: myIFU = utils.IFU(RSS_file, IFU, flag_name=False)
    GAMA_ID=str(myIFU.name)

    # Get wavelength axis as an aspect of the sami.utils.IFU class
    wave = myIFU.lambda_range

    # Get SDSS g-band throughput curve
    if not os.path.isfile("sdss_g.dat"):
        urllib.urlretrieve("http://www.sdss.org/dr3/instruments/imager/filters/g.dat", "sdss_g.dat")

    # and convolve with the SDSS throughput (should live in a library)
    sdss_g = ascii.read("SDSS_g.dat", quotechar="#", names=["wave", "pt_secz=1.3", "ext_secz=1.3", "ext_secz=0.0", "extinction"])

    # re-grid g["wave"] -> wave
    thru_regrid = griddata(sdss_g["wave"], sdss_g["ext_secz=1.3"], wave, method="cubic", fill_value=0.0)

    # initialise a simulated g' band flux array
    reconstruct = np.zeros(len(myIFU.n))
    tester = np.zeros(len(myIFU.n))

    # Sum convolved flux:
    for i in range(len(myIFU.n)):
        reconstruct[i] = np.nansum(np.absolute(myIFU.data[i] * thru_regrid))
        
    reconstruct[np.isnan(reconstruct)] = 0. # replacing nan with 0.0
    reconstruct[reconstruct < 0] = 0.       # replacing negative fluxes with 0.0
        
    # Check if the user supplied a red RSS file, throw exception.
    if np.array_equal(reconstruct, tester):
        raise SystemExit("All values are zero: please check if you might have input a RED spectrum!")

    # Relative fibre positions to Fibre#1 (Plate scale of 15.22"/mm)
    fibre_xpos_arcsec = (15.22/1000)*(myIFU.x_microns - myIFU.x_microns[myIFU.n == 1])
    fibre_ypos_arcsec = (15.22/1000)*(myIFU.y_microns - myIFU.y_microns[myIFU.n == 1])

    # Position of GAMA object from the RSS file, but defined by the target selection .dat file on the wiki. Might have slight manual offset from true galaxy position, but is the SAMI working position, so is used here.
    obj_ra = np.around(myIFU.obj_ra[myIFU.n == 1], decimals=6)
    obj_dec = np.around(myIFU.obj_dec[myIFU.n == 1], decimals=6)

    RSS_SDSS_g_flux = reconstruct

    # Self normalisation
    RSS_SDSS_g_flux = (RSS_SDSS_g_flux - np.min(RSS_SDSS_g_flux))/np.max(RSS_SDSS_g_flux - np.min(RSS_SDSS_g_flux))

    # Write out g-band convolved data
    if write=="True":
        # Write data
        outstring = zip(fibre_xpos_arcsec, fibre_ypos_arcsec, RSS_SDSS_g_flux)
        f = open(str(myIFU.name)+"_RSS_"+str(RSS_file_name)+"_SDSS_g.txt", "w")
        f.write("# Column 1: Respective x position in arcseconds"+"\n")
        f.write("# Column 2: Respective y position in arcseconds"+"\n")
        f.write("# Column 3: SDSS g-band flux"+"\n")
        f.write("# Position of Object ["+str(GAMA_ID)+"](deg): "+str(obj_ra[0])+", "+str(obj_dec[0])+"\n"+"\n")
        for line in outstring:
            f.write(" ".join(str(x) for x in line) + "\n")
        f.close()

    if show=="True" or write=="True":
    
        # Plot fibres with normalised colourmap
        x = fibre_xpos_arcsec
        y = fibre_ypos_arcsec
        radii = np.zeros(len(fibre_xpos_arcsec)) + 0.8
        patches = []
        for x1,y1,r in zip(x, y, radii):
            circle = Circle((x1,y1), r)
            patches.append(circle)
        fig1 = py.figure()
        fig1.set_size_inches(6,6)
        ax = fig1.add_subplot(1,1,1)
        ax.set_aspect('equal')
        colors = RSS_SDSS_g_flux
        pa = PatchCollection(patches, cmap=py.cm.Blues)
        pa.set_array(colors)
        ax.add_collection(pa)
        py.colorbar(pa)
        py.axis([-8., 8., -8., 8.])
        py.xlabel("Delta(RA) [arcsec]")
        py.ylabel("Delta(DEC) [arcsec]")
        py.title("GAMA_ID = "+str(GAMA_ID)+"\n"+"RSS_file = "+RSS_file+"\n"+"RSS g-band image")
        if write=="True":
            py.savefig(str(myIFU.name)+"_RSS_"+str(RSS_file_name)+"_SDSS_g.png", bbox_inches=0)
        if show=="True":
            py.show()


    return fibre_xpos_arcsec, fibre_ypos_arcsec, RSS_SDSS_g_flux, obj_ra, obj_dec

#########################################################################################
#
# "RSS_SDSS_resampled"
#
#   This function extracts the SDSS g-band image of the object given using the "GRP_MRA"
#   & "GRP_MDEC" data in the RSS Fibre Table "FIBRES_IFU", and extracts the flux for a
#   given IFU footprint.
#

def RSS_SDSS_resampled(RSS_file="unknown", GAMA_ID="unknown", IFU=1, x_offset=0, y_offset=0, band="g", show="False", write="False"):

    # Get RSS file name without ".fits"
    RSS_file_name = RSS_file[(len(RSS_file)-18):(len(RSS_file)-5)]
        
    # Read RSS file (GAMA_ID-specific input)
    if GAMA_ID!="unknown":
        myIFU = utils.IFU(RSS_file, GAMA_ID)
    else: myIFU = utils.IFU(RSS_file, IFU, flag_name=False)
    GAMA_ID=str(myIFU.name)

    # Converts arcsec offset into degrees as the SDSS image WCS are in degrees
    x_offset_deg = x_offset/3600.
    y_offset_deg = y_offset/3600.

    # Relative fibre degree positions to Fibre#1 with applied offset
    pos_off_x = (((myIFU.x_microns - myIFU.x_microns[myIFU.n==1])*(15.22/1000.))/3600.) + x_offset_deg
    pos_off_y = (((myIFU.y_microns - myIFU.y_microns[myIFU.n==1])*(15.22/1000.))/3600.) + y_offset_deg

    fibre_xpos_raw = (((myIFU.x_microns - myIFU.x_microns[myIFU.n==1])*(15.22/1000.)))
    fibre_ypos_raw = (((myIFU.y_microns - myIFU.y_microns[myIFU.n==1])*(15.22/1000.)))

    # Relative fibre positions in arcsec with applied offset to original RSS position
    fibre_xpos_arcsec = pos_off_x*3600
    fibre_ypos_arcsec = pos_off_y*3600

    # Position of GAMA object from the RSS file, but defined by the target selection .dat file on the SAMI_Wiki. Might have slight manual offset from true galaxy position, but is the SAMI working position, so is used here.
    obj_ra = np.around(myIFU.obj_ra[myIFU.n == 1], decimals=6)
    obj_dec = np.around(myIFU.obj_dec[myIFU.n == 1], decimals=6)

    # Extract SDSS g-band image if not already in working directory.
    # Returns 250x250 image with a side of ~25" and a pixel size of ~0.1". These are "~" because the SDSS surver works in degrees.
    if not os.path.isfile(str(myIFU.name)+"_SDSS_g.fits"):
        getSDSSimage(GAMA_ID=GAMA_ID, RA=obj_ra[0], DEC=obj_dec[0], band=str(band), size=0.00695, number_of_pixels=250, projection="Tan")

    # Open SDSS image and extract data & header information
    image_file = pf.open(str(myIFU.name)+"_SDSS_g.fits")
    
    image_data = image_file['Primary'].data
    image_header = image_file['Primary'].header
    
    img_crval1 = float(image_header['CRVAL1']) #RA
    img_crval2 = float(image_header['CRVAL2']) #DEC
    img_crpix1 = float(image_header['CRPIX1']) #Reference x-pixel
    img_crpix2 = float(image_header['CRPIX2']) #Reference y-pixel
    img_cdelt1 = float(image_header['CDELT1']) #Delta RA
    img_cdelt2 = float(image_header['CDELT2']) #Delta DEC
    img_naxis1 = float(image_header['NAXIS1']) #Number of pixels in x-direction
    img_naxis2 = float(image_header['NAXIS2']) #Number of pixels in y-direction

    # Smooth SDSS image to median seeing of 1.8arcsec
    sigma = 2*np.sqrt(2*np.log(2))*(1.8/1)
    n = sigma
    image_data_smoothed = blur_image(image_data, n, ny=None)
    image_data = image_data_smoothed

    # Get SDSS g-band fluxes of each fibre from the SDSS image
    SDSS_flux = []
    x_pix_positions = []
    y_pix_positions = []
        
    for i in xrange(np.shape(fibre_xpos_raw)[0]):
            
        # Image pixel positions of a fibre
        x_pix_pos = ((pos_off_x[i]) / img_cdelt1) + img_crpix1
        y_pix_pos = ((pos_off_y[i]) / img_cdelt2) + img_crpix2
        
        x_pix_positions.append(x_pix_pos)
        y_pix_positions.append(y_pix_pos)
            
        # Fibre radius in units of pixels (Fibre core diamter = 1.6")
        fibre_radius = ((1.6/2) / 3600) / img_cdelt1
            
        # Weight map array of SDSS image for the footprint of one fibre on the image
        weight = utils.resample_circle(img_naxis1, img_naxis2, (-1)*x_pix_pos, y_pix_pos, fibre_radius)
            
        # Multiply weight map & SDSS image data
        SDSS_flux_fibre_array = image_data * weight
            
        # NaNsum eitre flux array to get flux in one fibre
        SDSS_flux_fibre = np.nansum(SDSS_flux_fibre_array)
            
        # Append each fibre flux to list
        SDSS_flux.append(SDSS_flux_fibre)
    
    # Self normalisation
    SDSS_flux = (SDSS_flux - np.min(SDSS_flux))/np.max(SDSS_flux - np.min(SDSS_flux))

    # Write out g-band convolved data
    if write=="True":
        # Write data
        outstring = zip(fibre_xpos_arcsec, fibre_ypos_arcsec, SDSS_flux)
        f = open(GAMA_ID+"_RSS_"+str(RSS_file_name)+"_SDSS_"+str(band)+"_resampled.txt", "w")
        f.write("# Column 1: Respective x position in arcseconds"+"\n")
        f.write("# Column 2: Respective y position in arcseconds"+"\n")
        f.write("# Column 3: Normalised SDSS "+str(band)+"-band flux"+"\n")
        f.write("# Position of Object ["+str(GAMA_ID)+"](deg): "+str(obj_ra[0])+", "+str(obj_dec[0])+"\n"+"\n")
        for line in outstring:
            f.write(" ".join(str(x) for x in line) + "\n")
        f.close()

    if show=="True" or write=="True":
    
        # Plot fibres with normalised colourmap
        x = fibre_xpos_arcsec
        y = fibre_ypos_arcsec
        radii = np.zeros(len(fibre_xpos_arcsec)) + 0.8
        patches = []
        for x1,y1,r in zip(x, y, radii):
            circle = Circle((x1,y1), r)
            patches.append(circle)
        fig1 = py.figure()
        fig1.set_size_inches(6,6)
        ax = fig1.add_subplot(1,1,1)
        ax.set_aspect('equal')
        colors = SDSS_flux
        pa = PatchCollection(patches, cmap=py.cm.Blues)
        pa.set_array(colors)
        ax.add_collection(pa)
        py.colorbar(pa)
        py.axis([-8., 8., -8., 8.])
        py.xlabel("Delta(RA) [arcsec]")
        py.ylabel("Delta(DEC) [arcsec]")
        py.title("GAMA_ID = "+str(GAMA_ID)+"\n"+"RSS_file = "+RSS_file+"\n"+"SDSS g-band resampled image")
        if write=="True":
            py.savefig(str(myIFU.name)+"_RSS_"+str(RSS_file_name)+"_SDSS_"+str(band)+"_resampled.png", bbox_inches=0)
        if show=="True":
            py.show()
        

    return fibre_xpos_arcsec, fibre_ypos_arcsec, SDSS_flux, obj_ra, obj_dec

#########################################################################################
#
# "residual_data"
#
#   This function creates residual maps of functions "RSS_SDSS_g" - "RSS_SDSS_resampled".
#
#   It outputs many variables that are used in the optimisation script for plotting/
#   writing.
#

def residual_data(p0, RSS_file, GAMA_ID, IFU=1, sim="False", sim_x_off=1, sim_y_off=1, show="False", write="False"):
    
    # Print x_offset, y_offset, and flux scale factor to see what the optimisation code is doing. The offsets need to be in arcsec.
    print "x_offset: ",p0[0]
    print "y_offset: ",p0[1]
    print "#####" # This is just a line spacer to make it easier to read in the terminal.
    
    ### Step 1 ###
    if sim=="True":
        fibre_xpos_arcsec, fibre_ypos_arcsec, RSS_SDSS_g_flux, obj_ra, obj_dec = sim_SDSS_resampled(RSS_file=RSS_file, GAMA_ID=GAMA_ID, IFU=IFU, sim_x_off=sim_x_off, sim_y_off=sim_y_off, show=show, write=write)
    else:
        # Get collapse normalised g-band fluxes from RSS file using SDSS g-band filter
        fibre_xpos_arcsec, fibre_ypos_arcsec, RSS_SDSS_g_flux, obj_ra, obj_dec = RSS_SDSS_g(RSS_file=RSS_file, GAMA_ID=GAMA_ID, IFU=IFU, show=show, write=write)
    
    # Define variables ("raw" denotes the original RSS)
    raw_xpos_arcsec = np.asarray(fibre_xpos_arcsec)
    raw_ypos_arcsec = np.asarray(fibre_ypos_arcsec)
    raw_g_flux = np.asarray(RSS_SDSS_g_flux)
    raw_obj_ra = np.asarray(obj_ra[0]) #in degrees
    raw_obj_dec = np.asarray(obj_dec[0]) #in degrees
    
    ### Step 2 ###
    # Get collapse normalised g-band fluxes from IFU overlay on SDSS g-band image
    fibre_xpos_arcsec, fibre_ypos_arcsec, SDSS_flux, obj_ra, obj_dec = RSS_SDSS_resampled(RSS_file=RSS_file, GAMA_ID=GAMA_ID, IFU=IFU, x_offset=p0[0], y_offset=p0[1], band="g", show=show, write=write)
    
    # Define variables ("off" denotes the resampled image after offsets applied)
    off_xpos_arcsec = np.asarray(fibre_xpos_arcsec)
    off_ypos_arcsec = np.asarray(fibre_ypos_arcsec)
    off_g_flux = np.asarray(SDSS_flux)
    off_obj_ra = np.asarray(obj_ra[0]) #in degrees
    off_obj_dec = np.asarray(obj_dec[0]) #in degrees
    
    ### Step 3 ###
    # Find residual
    # use original fibre positions (just relative offsets)
    x_pos_arcsec = raw_xpos_arcsec
    residual_flux = off_g_flux - raw_g_flux
    sumsq_flux = np.nansum(np.power(residual_flux,2))
        
    # Plot residual maps
    if show=="True" or write=="True":
        
        # Plot fibres with colourmap
        x = raw_xpos_arcsec
        y = raw_ypos_arcsec
        radii = np.zeros(len(raw_xpos_arcsec)) + 0.8
        patches = []
        for x1,y1,r in zip(x, y, radii):
            circle = Circle((x1,y1), r)
            patches.append(circle)
        fig1 = py.figure()
        fig1.set_size_inches(6,6)
        ax = fig1.add_subplot(1,1,1)
        ax.set_aspect('equal')
        colors = residual_flux
        pa = PatchCollection(patches, cmap="RdBu")
        pa.set_array(colors)
        ax.add_collection(pa)
        py.colorbar(pa)
        py.axis([-8., 8., -8., 8.])
        min = np.min(residual_flux)
        max = np.max(residual_flux)
        if np.absolute(min) > np.absolute(max):
            pa.set_clim([min,-min])
        else: pa.set_clim([-max,max])
        py.xlabel("Delta(RA) [arcsec]")
        py.ylabel("Delta(DEC) [arcsec]")
        py.title("GAMA_ID = "+GAMA_ID+"\n"+"RSS_file = "+RSS_file+"\n"+"g-band residual image")
        if write=="True":
            # Get RSS file name without ".fits"
            RSS_file_name = RSS_file[(len(RSS_file)-18):(len(RSS_file)-5)]
            py.savefig(str(GAMA_ID)+"_RSS_"+str(RSS_file_name)+"_SDSS_g_residual.png", bbox_inches=0)
        if show=="True":
            py.show()
        
    
    return raw_xpos_arcsec, raw_ypos_arcsec, raw_obj_ra, raw_obj_dec, raw_g_flux, off_xpos_arcsec, off_ypos_arcsec, off_g_flux, residual_flux, sumsq_flux

#########################################################################################
#
# "residual_function"
#
#   This function creates residual maps of functions "RSS_SDSS_g" - "RSS_SDSS_resampled"
#   for both the fibre map and also the grids.
#
#   It outputs one variable (the residual array) that is used in the optimisation
#   function.     
#

def residual_function(p0, RSS_file, GAMA_ID, sim, sim_x_off, sim_y_off, IFU=1, show="False", write="False"):

    # Print x_offset, y_offset, and flux scale factor to see what the optimisation code is doing. The offsets need to be in arcsec.
    print "x_offset: ",p0[0]
    print "y_offset: ",p0[1]
    print "#####" # This is just a line spacer to make it easier to read in the terminal.
    
    ### Step 1 ###
    if sim=="True":
        fibre_xpos_arcsec, fibre_ypos_arcsec, RSS_SDSS_g_flux, obj_ra, obj_dec = sim_SDSS_resampled(RSS_file=RSS_file, GAMA_ID=GAMA_ID, IFU=IFU, sim_x_off=sim_x_off, sim_y_off=sim_y_off, show=show, write=write)
    else:
        # Get collapse normalised g-band fluxes from RSS file using SDSS g-band filter
        fibre_xpos_arcsec, fibre_ypos_arcsec, RSS_SDSS_g_flux, obj_ra, obj_dec = RSS_SDSS_g(RSS_file=RSS_file, GAMA_ID=GAMA_ID, IFU=IFU, show=show, write=write)

    # Define variables ("raw" denotes the original RSS)
    raw_g_flux = np.asarray(RSS_SDSS_g_flux)
    
    ### Step 2 ###
    # Get collapse normalised g-band fluxes from IFU overlay on SDSS g-band image
    fibre_xpos_arcsec, fibre_ypos_arcsec, SDSS_flux, obj_ra, obj_dec = RSS_SDSS_resampled(RSS_file=RSS_file, GAMA_ID=GAMA_ID, IFU=IFU, x_offset=p0[0], y_offset=p0[1], band="g", show=show, write=write)
    
    # Define variables ("off" denotes the resampled image after offsets applied)
    off_g_flux = np.asarray(SDSS_flux)
    
    ### Step 3 ###
    # Find residual
    residual_flux = off_g_flux - raw_g_flux
    
    residual_flux = residual_flux.ravel()

    return residual_flux

#########################################################################################
#
# "bestfit_SDSS_g"
#
#   This function performs a best fit on the residual_function. It uses the scipy
#   "leastsq" algorithm to do this with an initial guess of the minimum of the brute
#   force residual grid. At the moment it still somewhat struggles with supious features
#   like cosmic rays that give a high flux in an otherwise low flux fibre, and high flux
#   foreground objects, but it is very good with "normal" targets, fitting in about
#   210secs per object per RSS. The output "p" is the offset in arcsec, and the output RA
#   and DEC are the equated RA and DECs for that frame in degrees.
#

def bestfit_SDSS_g(RSS_file="unknown", GAMA_ID="unknown", IFU=1, sim="False", sim_off="unkown", sim_x_off=0, sim_y_off=0, show="False", write="False"):
    
    start_time = datetime.datetime.now()
    print("### Best Fit Start Time: "+start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Get RSS file name without ".fits"
    RSS_file_name = RSS_file[(len(RSS_file)-18):(len(RSS_file)-5)]
        
    # Read RSS file (GAMA_ID-specific input)
    if GAMA_ID!="unknown":
        myIFU = utils.IFU(RSS_file, GAMA_ID)
    else: myIFU = utils.IFU(RSS_file, IFU, flag_name=False)
    GAMA_ID=str(myIFU.name)

    # Relative fibre positions to Fibre#1 (Plate scale of 15.22"/mm)
    fibre_xpos_arcsec = (15.22/1000)*(myIFU.x_microns - myIFU.x_microns[myIFU.n == 1])
    fibre_ypos_arcsec = (15.22/1000)*(myIFU.y_microns - myIFU.y_microns[myIFU.n == 1])


    # Offset pattern for sim data
    if sim_off=="A":
        sim_x_off=0
        sim_y_off=0
    if sim_off=="B":
        sim_x_off=-0.6
        sim_y_off=0.4
    if sim_off=="C":
        sim_x_off=0
        sim_y_off=-0.7
    if sim_off=="D":
        sim_x_off=0.6
        sim_y_off=0.4
    if sim_off=="E":
        sim_x_off=-0.6
        sim_y_off=-0.4
    if sim_off=="F":
        sim_x_off=0.6
        sim_y_off=-0.4
    if sim_off=="G":
        sim_x_off=0
        sim_y_off=0.7

    ########## OPTIMIZE ############
    
    ### My own grid ### ~185s
    x = np.linspace(-2,2,20)
    y = np.linspace(-2,2,20)
    x_coors = []
    y_coors = []
    sumsq_vals = []
    for x_coor in x:
        for y_coor in y:
            raw_xpos_arcsec, raw_ypos_arcsec, raw_obj_ra, raw_obj_dec, raw_g_flux, off_xpos_arcsec, off_ypos_arcsec, off_g_flux, residual_flux, sumsq_flux = residual_data([x_coor, y_coor], str(RSS_file), str(GAMA_ID), sim=sim, sim_x_off=sim_x_off, sim_y_off=sim_y_off)
            x_coors.append(x_coor)
            y_coors.append(y_coor)
            sumsq_vals.append(sumsq_flux)
    print x_coors[sumsq_vals.index(np.min(sumsq_vals))], y_coors[sumsq_vals.index(np.min(sumsq_vals))], sumsq_vals[sumsq_vals.index(np.min(sumsq_vals))]

    g = [x_coors[sumsq_vals.index(np.min(sumsq_vals))],y_coors[sumsq_vals.index(np.min(sumsq_vals))]]

    ### leastsq ### ~25s (including SDSS g-band image download 262KB)
    p0_guess = [g[0],g[1]] # Initial guess is minimum of above sumsq grid
    print p0_guess
    # The "epsfcn" variable was asigned by changing it until the best fit happened for a variety of objects and offsets. This is the value for SAMI, but would be different for another purpose.
    p, cov_p, infodict, mesg, ier = optimize.leastsq(residual_function,p0_guess,args=(str(RSS_file), str(GAMA_ID), sim, sim_x_off, sim_y_off), epsfcn=2.3, full_output=1)
    print p

    ########## OPTIMIZE - COMPLETED ############

    # get residual data from new set of offsets from optimisation
    raw_xpos_arcsec, raw_ypos_arcsec, raw_obj_ra, raw_obj_dec, raw_g_flux, off_xpos_arcsec, off_ypos_arcsec, off_g_flux, residual_flux, sumsq_flux  = residual_data([p[0], p[1]], str(RSS_file), str(GAMA_ID), sim=sim, sim_x_off=sim_x_off, sim_y_off=sim_y_off)

    RA = raw_obj_ra - (p[0]/3600)
    DEC = raw_obj_dec + (p[1]/3600)
            
    print " RA: ", RA
    print "DEC: ", DEC

    if show=="True" or write=="True":
        
        # Close all open plots to allow overlay of subplots
        py.close("all")
        
        ### FIGURE 1 ###
        # Best fit plots of fibres
        fig = py.figure(1)
        fig.set_size_inches(36, 6.5)
        fig.suptitle("Offset Fit for GAMA_ID = "+str(GAMA_ID)+",     RSS = "+str(RSS_file)+",     Onsky Position: {0:.6f}".format(RA)+", {0:.6f}".format(DEC)+" degrees,     Offset = {0:.3f}".format(p[0])+", {0:.3f}".format(p[1])+" arcsecs", fontsize=18)

        # SUBPLOT-1 Plot raw_flux with onsky positions
        ax = fig.add_subplot(1,5,1)
        ax.set_aspect('equal')
        x = fibre_xpos_arcsec
        y = fibre_ypos_arcsec
        radii = np.zeros(len(fibre_xpos_arcsec)) + 0.8
        patches = []
        for x1,y1,r in zip(x, y, radii):
            circle = Circle((x1,y1), r)
            patches.append(circle)
        colors =(raw_g_flux - np.min(raw_g_flux))/np.max(raw_g_flux - np.min(raw_g_flux))
        pa = PatchCollection(patches, cmap=py.cm.Blues)
        pa.set_array(colors)
        ax.add_collection(pa)
        py.axis([-10, 10, -10, 10])
        py.xlabel("Delta(RA) [arcsec]")
        py.ylabel("Delta(DEC) [arcsec]")
        py.title("RSS g-band image")

        # SUBPLOT-2 Overlay figure for with IFU
        image_file = pf.open(str(GAMA_ID)+"_SDSS_g.fits")
        image_data = image_file['Primary'].data
        image_header = image_file['Primary'].header
        img_crval1 = float(image_header['CRVAL1']) #RA
        img_crval2 = float(image_header['CRVAL2']) #DEC
        img_crpix1 = float(image_header['CRPIX1']) #Reference x-pixel
        img_crpix2 = float(image_header['CRPIX2']) #Reference y-pixel
        img_cdelt1 = float(image_header['CDELT1']) #Delta RA
        img_cdelt2 = float(image_header['CDELT2']) #Delta DEC
        img_naxis1 = float(image_header['NAXIS1']) #Number of pixels in x-direction
        img_naxis2 = float(image_header['NAXIS2']) #Number of pixels in y-direction
        x_pix_positions = []
        y_pix_positions = []
        for k in xrange(61):
            # Image pixel positions of a fibre
            x_pix_pos = ((off_xpos_arcsec[k]/3600) / img_cdelt1) + img_crpix1
            y_pix_pos = ((off_ypos_arcsec[k]/3600) / img_cdelt2) + img_crpix2
            x_pix_positions.append(x_pix_pos)
            y_pix_positions.append(y_pix_pos)
        ax = fig.add_subplot(1,5,2)
        ax.set_aspect('equal')
        py.imshow(image_data, cmap='gist_yarg', aspect="equal")
        py.xlim(0,250)
        py.ylim(0,250)
        x_pix_positions_inv = ((-1)*(np.asarray(x_pix_positions)-img_crpix1)) + img_crpix1
        x = x_pix_positions_inv
        y = y_pix_positions
        patches = []
        for x1,y1 in zip(x, y):
            circ = Circle((x1,y1), ((0.8/3600)/img_cdelt1))
            patches.append(circ)
        pa = PatchCollection(patches, edgecolor='red', facecolor='none', linewidth=0.75)
        ax.add_collection(pa)
        py.scatter(img_crpix1, img_crpix2, marker="+", s=100, color="b")
        py.setp(ax.get_xticklabels(), visible=False)
        py.setp(ax.get_yticklabels(), visible=False)
        py.title("SDSS g-band image with offset overlay")
        py.xlabel("<--- 25 arcsec --->")
        py.ylabel("<--- 25 arcsec --->")
    
        # SUBPLOT-3 Plot off_flux with onsky positions
        ax = fig.add_subplot(1,5,3)
        ax.set_aspect('equal')
        x = fibre_xpos_arcsec
        y = fibre_ypos_arcsec
        radii = np.zeros(len(fibre_xpos_arcsec)) + 0.8
        patches = []
        for x1,y1,r in zip(x, y, radii):
            circle = Circle((x1,y1), r)
            patches.append(circle)
        colors =(off_g_flux - np.min(off_g_flux))/np.max(off_g_flux - np.min(off_g_flux))
        pa = PatchCollection(patches, cmap=py.cm.Blues)
        pa.set_array(colors)
        ax.add_collection(pa)
        py.axis([-10, 10, -10, 10])
        py.xlabel("Delta(RA) [arcsec]")
        py.ylabel("Delta(DEC) [arcsec]")
        py.title("SDSS g-band image resampled")
        
        # SUBPLOT-4 Plot sumsq_grid
        ax = fig.add_subplot(1,5,4)
        ax.set_aspect('equal')
        x = np.asarray(x_coors)
        y = np.asarray(y_coors)
        patches = []
        for x1,y1 in zip(x, y):
            rect = Rectangle((x1-0.1,y1-0.1), 0.2,0.2 , ec="none")
            patches.append(rect)
        colors = np.asarray(sumsq_vals)
        pa = PatchCollection(patches, cmap=py.cm.jet)
        pa.set_array(colors)
        ax.add_collection(pa)
        pa.set_clim([np.min(sumsq_vals),np.max(sumsq_vals)])
        py.axis([-2.5, 2.5, -2.5, 2.5])
        py.scatter(g[0], g[1],color='r',marker='x', s=100)
        py.scatter(p[0], p[1], color='1.0',marker='x', s=100)
        py.xlabel("Delta(RA) [arcsec]")
        py.ylabel("Delta(DEC) [arcsec]")
        py.title("Grid of E(d2) values for each offset, best = {0:.4f}".format(sumsq_flux))

        # SUBPLOT-5 Plot res_flux with onsky positions
        ax = fig.add_subplot(1,5,5)
        ax.set_aspect('equal')
        x = fibre_xpos_arcsec
        y = fibre_ypos_arcsec
        radii = np.zeros(len(fibre_xpos_arcsec)) + 0.8
        patches = []
        for x1,y1,r in zip(x, y, radii):
            circle = Circle((x1,y1), r)
            patches.append(circle)
        colors = residual_flux
        pa = PatchCollection(patches, cmap=py.cm.RdBu)
        pa.set_array(colors)
        ax.add_collection(pa)
        py.colorbar(pa)
        py.axis([-10, 10, -10, 10])
        min = np.min(residual_flux)
        max = np.max(residual_flux)
        if np.absolute(min) > np.absolute(max):
            pa.set_clim([min,-min])
        else: pa.set_clim([-max,max])
        py.xlabel("Delta(RA) [arcsec]")
        py.ylabel("Delta(DEC) [arcsec]")
        py.title("Offset residuals")
            
        if write=="True":
            if sim=="True":
                # Save figure in high-resolution
                py.savefig(str(GAMA_ID)+"_"+str(RSS_file_name)+"_SDSS_g_bestfit_fibres_sim_"+str(sim_off)+".png", bbox_inches=0, dpi=300)
            else:
                # Save figure in high-resolution
                py.savefig(str(GAMA_ID)+"_"+str(RSS_file_name)+"_SDSS_g_bestfit_fibres_chi2.png", bbox_inches=0, dpi=300)
        if show=="True":
            py.show()
        

    end_time = datetime.datetime.now()
    print("### End Time: "+end_time.strftime("%Y-%m-%d %H:%M:%S"))
    
    print("### Time to solve best fit for object "+str(GAMA_ID)+": {0}".format(end_time-start_time))
    
    notes = "############# ---- This is the end of the script ---- #############"
    
    return p, RA, DEC, sumsq_flux, notes

#########################################################################################
#
# "sim_SDSS_resampled"
#
#   This function is essentiall the same as "RSS_SDSS_resampled" but returns variables
#   with the same names as an RSS frame to supply the code with a simulated frame instead
#   of an RSS frame. This is used mainly in the testing of the code to see how robust the
#   optimisation algorithms are, as it should be able to find itself.
#

def sim_SDSS_resampled(RSS_file="unknown", GAMA_ID="unknown", IFU=1, sim_x_off=1, sim_y_off=1, band="g", show="False", write="False"):
    
    # Get RSS file name without ".fits"
    RSS_file_name = RSS_file[(len(RSS_file)-18):(len(RSS_file)-5)]
    
    # Read RSS file (GAMA_ID-specific input)
    if GAMA_ID!="unknown":
        myIFU = utils.IFU(RSS_file, GAMA_ID)
    else: myIFU = utils.IFU(RSS_file, IFU, flag_name=False)
    GAMA_ID=str(myIFU.name)
    
    # Converts arcsec offset into degrees as the SDSS image WCS are in degrees
    x_offset_deg = sim_x_off/3600.
    y_offset_deg = sim_y_off/3600.
    
    # Relative fibre degree positions to Fibre#1 with applied offset
    pos_off_x = (((myIFU.x_microns - myIFU.x_microns[myIFU.n==1])*(15.22/1000.))/3600.) + x_offset_deg
    pos_off_y = (((myIFU.y_microns - myIFU.y_microns[myIFU.n==1])*(15.22/1000.))/3600.) + y_offset_deg
    
    fibre_xpos_raw = (((myIFU.x_microns - myIFU.x_microns[myIFU.n==1])*(15.22/1000.)))
    fibre_ypos_raw = (((myIFU.y_microns - myIFU.y_microns[myIFU.n==1])*(15.22/1000.)))
    
    # Relative fibre positions in arcsec with applied offset to original RSS position
    fibre_xpos_arcsec = pos_off_x*3600
    fibre_ypos_arcsec = pos_off_y*3600
    
    # Position of GAMA object from the RSS file, but defined by the target selection .dat file on the SAMI_Wiki. Might have slight manual offset from true galaxy position, but is the SAMI working position, so is used here.
    obj_ra = np.around(myIFU.obj_ra[myIFU.n == 1], decimals=6)
    obj_dec = np.around(myIFU.obj_dec[myIFU.n == 1], decimals=6)
    
    # Extract SDSS g-band image if not already in working directory.
    # Returns 250x250 image with a side of ~25" and a pixel size of ~0.1". These are "~" because the SDSS surver works in degrees.
    if not os.path.isfile(str(myIFU.name)+"_SDSS_g.fits"):
        getSDSSimage(GAMA_ID=GAMA_ID, RA=obj_ra[0], DEC=obj_dec[0], band=str(band), size=0.00695, number_of_pixels=250, projection="Tan")
    
    # Open SDSS image and extract data & header information
    image_file = pf.open(str(myIFU.name)+"_SDSS_g.fits")
    
    image_data = image_file['Primary'].data
    image_header = image_file['Primary'].header
    
    img_crval1 = float(image_header['CRVAL1']) #RA
    img_crval2 = float(image_header['CRVAL2']) #DEC
    img_crpix1 = float(image_header['CRPIX1']) #Reference x-pixel
    img_crpix2 = float(image_header['CRPIX2']) #Reference y-pixel
    img_cdelt1 = float(image_header['CDELT1']) #Delta RA
    img_cdelt2 = float(image_header['CDELT2']) #Delta DEC
    img_naxis1 = float(image_header['NAXIS1']) #Number of pixels in x-direction
    img_naxis2 = float(image_header['NAXIS2']) #Number of pixels in y-direction
    
    # Smooth SDSS image to median seeing of 1.8arcsec
    sigma = 2*np.sqrt(2*np.log(2))*(1.8/1)
    n = sigma
    image_data_smoothed = blur_image(image_data, n, ny=None)
    image_data = image_data_smoothed
    
    # Get SDSS g-band fluxes of each fibre from the SDSS image
    SDSS_flux = []
    x_pix_positions = []
    y_pix_positions = []
    
    for i in xrange(np.shape(fibre_xpos_raw)[0]):
        
        # Image pixel positions of a fibre
        x_pix_pos = ((pos_off_x[i]) / img_cdelt1) + img_crpix1
        y_pix_pos = ((pos_off_y[i]) / img_cdelt2) + img_crpix2
        
        x_pix_positions.append(x_pix_pos)
        y_pix_positions.append(y_pix_pos)
        
        # Fibre radius in units of pixels (Fibre core diamter = 1.6")
        fibre_radius = ((1.6/2) / 3600) / img_cdelt1
        
        # Weight map array of SDSS image for the footprint of one fibre on the image
        weight = utils.resample_circle(img_naxis1, img_naxis2, (-1)*x_pix_pos, y_pix_pos, fibre_radius)
        
        # Multiply weight map & SDSS image data
        SDSS_flux_fibre_array = image_data * weight
        
        # NaNsum eitre flux array to get flux in one fibre
        SDSS_flux_fibre = np.nansum(SDSS_flux_fibre_array)
        
        # Append each fibre flux to list
        SDSS_flux.append(SDSS_flux_fibre)
    
    # Self normalisation
    SDSS_flux = (SDSS_flux - np.min(SDSS_flux))/np.max(SDSS_flux - np.min(SDSS_flux))
    
    fibre_xpos_arcsec = fibre_xpos_arcsec
    fibre_ypos_arcsec = fibre_ypos_arcsec
    RSS_SDSS_g = SDSS_flux # Note: This is not RSS, but used so code is compatible
    obj_ra = obj_ra
    obj_dec = obj_dec

    return fibre_xpos_arcsec, fibre_ypos_arcsec, RSS_SDSS_g, obj_ra, obj_dec 

#########################################################################################
#
# "getSDSSimage"
#
#   This function queries the SDSS surver at skyview.gsfc.nasa.gov and returns an image
#   with a user supplied set of parameters. A full description of the input parameters is
#   given at - http://skyview.gsfc.nasa.gov/docs/batchpage.html
#
#   The parameters that can be set here are:
#
#   RA - in degrees
#   DEC - in degrees
#   band - u,g,r,i,z filters
#   size - size of side of image in degrees
#   number_of_pixels - number of pixels of side of image (i.e 720 will return 720x720)
#   projection - 2D mapping of onsky projection. Tan is standard.
#   url_show - this is a function variable if the user wants the url printed to terminal
#

def getSDSSimage(GAMA_ID="unknown", RA=0.0, DEC=0.0, band="g", size=0.02, number_of_pixels=720, projection="Tan", url_show="False"):
    
    #function to retrieve SDSS g-band image
    
    # Contruct URL
    RA = str(RA).split(".")
    DEC = str(DEC).split(".")
    size = str(size).split(".")
    
    URL = "http://skyview.gsfc.nasa.gov//cgi-bin/pskcall?position="+str(RA[0])+"%2e"+str(RA[1])+"%2c"+str(DEC[0])+"%2e"+str(DEC[1])+"&Survey=SDSSdr7"+str(band)+"&size="+str(size[0])+"%2e"+str(size[1])+"&pixels="+str(number_of_pixels)+"&proj="+str(projection)
    
    urllib.urlretrieve(str(URL), str(GAMA_ID)+"_SDSS_g.fits")
    
    if url_show=="True":
        print "SDSS g-band image of object "+str(GAMA_ID)+" has finished downloading to the working directory with the file name: "+str(GAMA_ID)+"_SDSS_g.fits"
        
        print "The URL for this object is: ", URL

#########################################################################################
#
# "getSDSSspectra"
#
#   NOTE: Not implemented, just a place holder with notes on what to do.
#

def getSDSSspectra(inlist, show="True", write="True"):
    
    
    #function to retrieve SDSS spectra
    
    #example (GAMA-91999):
    
    #url for GAMA-91999 spectra (main object is first in list, then two repeat observations of secondary object) [find a way to distiguish between them and extract all spectra for S/N comparison] http://api.sdss3.org/spectrumQuery?ra=214.5090d&dec=0.48423&radius=7
    
    #returns list: ["sdss.303.51615.552.26", "sdss.304.51609.405.26", "sdss.304.51957.422.26"]
    
    #urllib.urlretrieve("http://api.sdss3.org/spectrum?plate=303&mjd=51615&fiber=552", "spectra_test_91999.fits")
    
    print "end"

#########################################################################################
#
# "gauss_kern" & "blur_image"
#
#   Following two functions from scipy cookbook to use a Gaussian kernal to blur an
#   image. It was taken from: http://www.scipy.org/Cookbook/SignalSmooth
#

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
        """
    g = gauss_kern(n, sizey=ny)
    improc = sp.signal.convolve(im, g, mode='same')
    return(improc)


#########################################################################################
#
# "sb"
#
#   Plot a surface brightness map for an SDSS image. Takes arguments controlling the 
#   scale to which SB is extrapolated. Warning: this code does not resample images, 
#   but translates the flux of a given pixel to a surface brightness to the requested 
#   scale. Mental arithmetics required for interpretation. 
#

def sb(image, scale=1.0, contour=True, vmin=None, vmax=None, 
       sky=None, levels=None):  

    import matplotlib.pyplot as plt

    """ 
    INPUTS: 
    -------
    scale      [flt] Surface brightness extrapolation scale in sq.asec.
    vmin, vmax [flt] Levels for image display. 
    levels     [flt] List of levels for optional contour overplot. 

    ToDo list: 
    ----------
    * Add a contour export function in DS9 format (.con). 
    * Add error estimation. 
    """

    """ (1) Image IO """
    hdu = pf.open(image)

    flux = hdu[0].data
    hdr0 = hdu[0].header

    hdu.close()

    """ (2) Photometry 

    Methodology: 

    Flux to magnitude conversion
    1 nanomaggie : 22.5 mag
    -> m - 22.5 = -2.5 * alog10(F/1)
    -> m = 22.5 - 2.5 * alog10(F)
    
    Error calculation, from the SDSS DR5 manual
    (http://www.sdss.org/dr5/algorithms/fluxcal.html): 

    Pogson error: 
          mag  = -2.5 * log10(f/f0)
    error(mag) = 2.5 / ln(10) * error(counts) / counts
    
    where 
    error(counts) = sqrt([counts+sky]/gain + Npix*(dark_variance+skyErr)),
    """

    if sky == None: 
        sky = np.median(flux)

    def brightness(F_in): 
        brightness = 22.5 - 2.5* np.log10(F_in - sky)
        return brightness

    # ALSO define error function here. 

    """ (3) Get surface brightness to requested scale: default is 1 sq.asec  """
    img_scale = 0.396127 # asec/px
    area_norm = scale / img_scale**2

    sb = brightness(flux*area_norm)

    """ (4) Plot surface photometry map """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.title("Surface brightness plot [mag/"+str(scale).zfill(3)+" sq.asec]\n("+image+")")

    if (vmin==None) & (vmax==None):
        vmin = np.round(np.nanmin(sb) + 2.0, 1)
        vmax = np.round(np.nanmax(sb) - 2.0, 1)
        print("Setting range to ["+str(vmin).zfill(4)+", "+
              str(vmax).zfill(4)+"] mag /"+str(scale)+" sq.asec")
    im = plt.imshow(sb, cmap='gray', interpolation='nearest', 
               origin='lower left', vmin=vmin, vmax=vmax)
    cb1 = plt.colorbar(im, shrink=0.8)

    # Also plot contours for every mag increment. 
    if levels == None: # i.e., leave as is for default levels. 
        levels = [23., 22., 21., 20., 19., 18., 17., 16.0]
    cplot = plt.contour(sb, cmap='Oranges', levels=levels, 
                        linewidths=1.0, alpha=0.8)
    strannot = "levels between [" + str(float(min(levels))).zfill(4) + ", " +\
               str(float(max(levels))).zfill(4) + "] mag"
    plt.text(0.04, 0.95, strannot, 
             horizontalalignment='left',verticalalignment='center', 
             transform=ax.transAxes)


#########################################################################################
###                                                                                   ###
#################################--- END OF FILE ---#####################################
###                                                                                   ###
#########################################################################################
