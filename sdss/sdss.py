# sami.sdss
# 
# Contains functions used to interface between SAMI and SDSS. 
#
# Written by: Iraklis Konstantopoulos & Samuel Richards
#
# Initial development September 2012 -- March 2013, Iraklis Konstantopoulos. 
# 
# Contact: Samuel Richards - samuel@physics.usyd.edu.au
# Maintainer: Iraklis Konstantopoulos - iraklis@aao.gov.au
#
# Version 7
# 25/02/2014
#
#   "getSDSSimage"
#
#       This function queries the SDSS surver at skyview.gsfc.nasa.gov and returns an
#       image with a user supplied set of parameters, which are explained with the
#       function.
#
#   "sb" 
#
#       Plot a surface brightness map for an SDSS image. Takes arguments controlling the 
#       scale to which SB is extrapolated. Warning: this code does not resample images, 
#       but translates the flux of a given pixel to a surface brightness to the requested 
#       scale. Mental arithmetics required for interpretation. 
#
#  "ten"
#
#       Converts from sexagessimal to decimal. 
# 
#  "sixty"
#
#       Converts from decimal to sexagessimal. 
# 
#  "sim_sdss_cube"
#
#       Produce a mock g-band image from a blue SAMI cube. Throughputs from: 
#       - SDSS:    http://www.sdss.org/dr5/instruments/imager/index.html#filters
#       - AAOmega: http://www.aao.gov.au/cgi-bin/aaomega_calc.cgi
#       WARNING! A couple of lines pertaining to NaNs in the reconstruct arrays were 
#       commented out, as the nansum should take care of this. Testing required. 
# 
#  "sim_sdss_rss"
#
#       Produce a mock g-band image from a SAMI row-stacked spectrum (RSS). 
# 
#  "overlay"
#
#       Overlay a SAMI bundle schematic onto a fits image. 
# 
#  "bundle_definition"
#
#       Make a definition file containing a schematic of a SAMI fibre bundle. 
# 
#########################################################################################

import numpy as np
import scipy as sp
import astropy.io.fits as pf
import urllib

import pylab as py
import sami.utils as utils
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import astropy.io.ascii as tab
from scipy.interpolate import griddata
import sys

#########################################################################################

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
    
    URL = "http://skyview.gsfc.nasa.gov//cgi-bin/pskcall?position="+(str(RA[0])+"%2e"+str(RA[1])+"%2c"+str(DEC[0])+"%2e"+str(DEC[1])+"&Survey=SDSSdr7"+str(band)+"&size="+str(size[0])+"%2e"+str(size[1])+"&pixels="+str(number_of_pixels)+"&proj="+str(projection))
                                                                     
    # Get SDSS image
    urllib.urlretrieve(str(URL), str(object_name)+"_SDSS_"+str(band)+".fits")
                                                                     
    if url_show=="True":
        print ("SDSS "+str(band)+"-band image of object "+str(object_name)+" has finished downloading to the working directory with the file name: "+str(object_name)+"_SDSS_"+str(band)+".fits")
                                                                                
        print "The URL for this object is: ", URL



#########################################################################################
#
# "sb"
#
#   Plot a surface brightness map for an SDSS image. Takes arguments controlling the 
#   scale to which SB is extrapolated. Warning: this code does not resample images, 
#   but translates the flux of a given pixel to a surface brightness to the requested 
#   scale. Mental arithmetics required for interpretation. 
#

def sb(image, scale=1.0, contour=True, vmin=None, vmax=None, sky=None, levels=None):

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






def ten(h, m, s, RA=False):
    result = np.sign(h) * \
        (np.absolute(float(h)) + (float(m) + float(s)/60.)/60.)
    if RA==True: result = 15.*result
    return result


def sixty(deg, RA=False):
    sign = np.sign(deg)
    if RA==True: deg = float(np.abs(deg)/15.)
    else: deg = float(np.abs(deg))
    h = np.fix(deg)
    m = np.fix(60.*(deg-h))
    s = 60.*np.around(60*(deg-h) - m, decimals=3)
    sex = [int(h)*sign, int(m), float(s)]
    return sex


def sim_sdss_cube(cubein, verbose=True, invert=True, log=False):
    """ 
    Produce a mock g-band image from a blue SAMI cube. 
    
    -- A future incarnation should convert the estimated SDSS flux into 
       a brightness (in nanomaggies).

    -- Need to add axis labels in arcseconds. 

    Issues: 
    1) Cubes do not include valid CRVAL1, just CRVAL3. Need an extra step
       and some trickery to build wavelength axis. 
    """
    
    # Open FITS file. 
    hdulist = pf.open(cubein)
    
    if verbose:  # print file info to screen
        print('')
        hdulist.info()
        print('')

    # Build wavelength axis. 
    crval3 = hdulist[0].header['CRVAL3']
    cdelt3 = hdulist[0].header['CDELT3']
    nwave  = len(hdulist[0].data)

    # -- crval3 is middle of range and indexing starts at 0. 
    # -- this wave-axis agrees with QFitsView interpretation. 
    crval1 = crval3 - ((nwave-1)/2)*cdelt3
    wave = crval1 + cdelt3*np.arange(len(hdulist[0].data))
    
    # Read in SDSS throughput (should live in a library)
    path_curve = '/Users/iraklis/Progs/Python/SAMI_manual/SDSS/SDSS_curves/'
    sdss_g = tab.read(path_curve+'SDSS_g.dat', quotechar="#", \
                       names=['wave', 'pt_secz=1.3', 'ext_secz=1.3', \
                                  'ext_secz=0.0', 'extinction'])
    
    # re-grid g['wave'] -> wave
    thru_regrid = griddata(sdss_g['wave'], sdss_g['ext_secz=1.3'], 
                           wave, method='cubic', fill_value=0.0)
    
    # initialise a 2D simulated g' band flux array. 
    len_axis = np.shape(hdulist[0].data[1][0])
    len_axis = np.float32(len_axis)
    reconstruct = np.zeros((len_axis,len_axis))
    tester = np.zeros((len_axis,len_axis))
    data_bit = np.zeros((len(hdulist[0].data),len_axis,len_axis))
    
    # Sum convolved flux:  
    for i in range(len(hdulist[0].data)):
        data_bit[i] = hdulist[0].data[i]*thru_regrid[i]
        """
        reconstruct[i,j] = \
        np.nansum(np.absolute(hdulist[0].data[:][i][j] * thru_regrid))
        """
    reconstruct = np.nansum(data_bit,axis=0) # not absolute right now
    
    ### reconstruct[np.isnan(reconstruct)] = 0. # replacing nan with 0.0
    reconstruct[reconstruct < 0] = 0.       # replacing negative fluxes with 0.0
    
    # Check if the user supplied a red RSS file, throw exception. 
    if np.array_equal(reconstruct, tester): 
        raise SystemExit('All values are zero: please check if you might ' 
                         'have input a RED spectrum!')
    
    # PLOT
    if invert: colmap = 'gist_yarg'
    else: colmap = 'gray'
    if log: reconstruct = np.log10(reconstruct)
    plt.imshow(reconstruct,cmap=colmap,interpolation='nearest')
    
    # Close the FITS buffer. 
    hdulist.close()
    

def sim_sdss_rss(file_in,        # A 2dfdr-processed row-stacked spectrum. 
                 ifu_num=1,      # Probe number (or galaxy name). 
                 log=True,       # Logarithmic scaling. 
                 c_invert=True,  # Invert colour map. 
                 path='./'):     # Where does file_in live? 
    """ 
    Produce a mock g-band image from a SAMI row-stacked spectrum (RSS).

    Description
    ------------
    This function reads in a row-stacked SAMI spectrum and simulates an SDSS 
    g-band image. This is be achieved by convolving the flux in each fibre
    through the SDSS filter transmission curve.  

    Notes 
    ------
    > Log scaling: once absolute flux calibration has been formalised, the 
                   function will plot/output brightness. 

    > ifu input: at the moment this only accepts a single ifu. Multiplexing 
                 will be implemented in the next version (not essential for 
                 survey use). 

    > SDSS throughput: this is now read in, but should live in a repository
                       of some sort. Otherwise, it should be offered as an 
                       input. 

    > Need to add a 'cube' function, to perform a px-by-px conversion. 

    > It might be intersting to have a 'rest frame' function that simulates 
      the image as it would appear in the rest frame. 

    > ISK, 11/4/13: Adapted to the new SAMI utils package and the IFU class. 
    """

    # ---------------------
    # INPUT and PROCESSING
    # ---------------------

    # Read RSS file (IFU-specific input).
    myIFU = utils.IFU(path+'/'+file_in, ifu_num, flag_name=False)
    # -- add a bunch of info on the IFU class outputs.

    # Get wavelength axis as an aspect of the sami.utils.IFU class
    wave = myIFU.lambda_range

    # and convolve with the SDSS throughput (should live in a library)
    path_curve = '/Users/iraklis/Progs/Python/SAMI_manual/SDSS/SDSS_curves/'
    sdss_g = tab.read(path_curve+'SDSS_g.dat', quotechar="#", \
                       names=['wave', 'pt_secz=1.3', 'ext_secz=1.3', \
                                  'ext_secz=0.0', 'extinction'])
    
    # re-grid g['wave'] -> wave
    thru_regrid = griddata(sdss_g['wave'], sdss_g['ext_secz=1.3'], 
                           wave, method='cubic', fill_value=0.0)
    
    # initialise a simulated g' band flux array
    reconstruct = np.zeros(len(myIFU.n))
    tester = np.zeros(len(myIFU.n))

    # Sum convolved flux:  
    for i in range(len(myIFU.n)): 
        reconstruct[i] = np.nansum(np.absolute(myIFU.data[i] * thru_regrid))
    
    ### reconstruct[np.isnan(reconstruct)] = 0. # replacing nan with 0.0
    reconstruct[reconstruct < 0] = 0.       # replacing negative fluxes with 0.0

    # Check if the user supplied a red RSS file, throw exception. 
    if np.array_equal(reconstruct, tester): 
        raise SystemExit('All values are zero: please check if you might ' 
                         'have input a RED spectrum!')

    # -----
    # PLOT 
    # -----    
    # produce a colour index for the reconstructed data: 
    if not log: norm1 = reconstruct - np.min(reconstruct)
    else: norm1 = np.log10(reconstruct) - np.min(np.log10(reconstruct))

    # normalise to unity
    norm_reconstruct = norm1/np.max(norm1)
    norm_reconstruct[norm_reconstruct < 0] = 0.0  # wipe negative fluxes
    
    # base plot: set aspect ratio, do not define size: 
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    xaxis_ctr = myIFU.xpos[myIFU.n == 1]
    yaxis_ctr = myIFU.ypos[myIFU.n == 1]
    
    plt.axis([-8., 8., -8., 8.]) # in arcseconds. 
    
    # Plot fibres as circle patches. 
    for i in range(len(myIFU.n)):

        # Define (xy) coordinates. 
        xy = 3600.*(xaxis_ctr - myIFU.xpos[i]), \
            3600.*(yaxis_ctr - myIFU.ypos[i])

        if c_invert: 
            this_col = str(norm_reconstruct[i]) # colour
            mappable = plt.cm.ScalarMappable(cmap='gray')
        else: 
            this_col = str(1.0-norm_reconstruct[i])
            mappable = plt.cm.ScalarMappable(cmap='gist_yarg') 

        # Create patch, plot it. 
        circ = patches.Circle(xy, radius=0.8,  
                              facecolor=this_col, edgecolor='None')
        ax.add_patch(circ) 
        # fib_size/2.

    mappable.set_array(norm_reconstruct)
    plt.colorbar(mappable)

    # and some plot cosmetics to finish it up. 
    plt.xlabel('Delta(RA) [arcsec]')
    plt.ylabel('Delta(DEC) [arcsec]')
    plt.title('Probe #'+str(ifu_num)+' / CATID = '+myIFU.name)


def overlay(image, bdf,                  # basic inputs
            l1=0.0, l2=1.0,              # levels
            shift=[0.0, 0.0],            # target shift from reference value 
            radec=[0.0, 0.0],            # target coords, if not ref value (d/s)
            nod=[0,0],                   # shift from default ctr (in asec)
            showGrid=False,              # show coordinate grid?
            stretch='linear',            # 'log', 'linear', etc.     
            invert=False,                # invert a grayscale image?
            hdu=0,                       # HDU, if not 0. 
            gray=True, readAVM=False):   # plot colour, AVM read
    """ 
    Overlay a SAMI bundle onto a fits image 

    Adapted to astropy input. 

    Inputs 
    -------
      image: fits image used for overlay; 
      bdf:   a 'bundle definition file', generate with the 'bundle_definition' 
             function in this module. 
    """

    import aplpy
    import astropy.wcs as pywcs
    
    # is the input image a fits file? 
    isfits = (image[len(image)-5:]=='.fits') or (image[len(image)-4:]=='.fit')

    # Use APLPy to read in the FITS file. 
    fig = aplpy.FITSFigure(image, north=True, hdu=hdu)
    if (gray==True): 
        fig.show_grayscale(vmin=l1, vmax=l2, stretch=stretch, invert=invert)
    else:
        fig.show_rgb()
    if showGrid==True: fig.show_grid()

    # Read the AVM of a jpg or tiff image: 
    if readAVM==True:
        from pyavm import AVM
        avm = AVM(image)

    # read BDF
    tab = tab.read(bdf)
    
    # Get field centre coordinates -- quite messy, clean up. 
    ctr = [0., 0.] # just initiate the field centre (list ok)

    # Input type: image loaded with Astro Visualisation Metadata --
    if (np.mean(radec)==0.0) and (readAVM!=True) and (isfits!=True): 
        ctr = [0.0, 0.0]                          # if cannot find AVM, ctr=0,0
        print("Warning: did not find a valid field centre definition")
    if (np.mean(radec)!=0.0) and (isfits!=True):  # respec' user input
        radec = np.array(radec)
        if radec.size > 2:                        # if input in sex, not dec
            from SAMI_sdss import ten
            ctr[0] = ten(radec[0], radec[1], radec[2], RA=True)
            ctr[1] = ten(radec[3], radec[4], radec[5])
        else: ctr = radec                               
                    
    if readAVM==True and (np.mean(radec)==0.0): 
        ctr=avm.Spatial.ReferenceValue            # read AVM field centre

    # Input type: fits file -- 
    if isfits:
        data = pf.open(image)
        wcs = pywcs.WCS(data[0].header)
        ctr = wcs.wcs.crval

    # apply 'nod' (fine positioning shift)
    if (nod[0]!=0) and (nod[1]!=0): 
        nod[0] = nod[0]*15
        nod = np.array(nod)/3600.
        ctr = np.array(ctr)-nod
        from SAMI_sdss import sixty
        stringer1 = 'Recentering to: ' + str(ctr[0]) + ' '+str(ctr[1])
        stringer2 = '            ie: ' + str(sixty(ctr[0],RA=True)) + \
            ' ' + str(sixty(ctr[1]))
        print('')
        print(stringer1)
        print(stringer2)

    # shift SAMI bundle into place
    ra  = tab['RA'] / np.cos(np.radians(ctr[1])) + ctr[0] 
    dec = tab['DEC'] + ctr[1]
    
    # SAMI bundle (individual fibres)
    fig.show_circles(ra, dec, 0.8/3600., 
                     edgecolor='cyan', facecolor='cyan', alpha=0.5)
    
    # Exclusion radius
    fig.show_circles(ctr[0], ctr[1], 165./3600., 
                     edgecolor='green', facecolor='none', linewidth=3.)
    

def bundle_definition(file_in, ifu=1, path='./', 
                      diagnose=False, pilot=False):
    """ 
    Make a definition file containing a schematic of a fibre bundle 

    There is some duplication in this code, as it includes a test for 
    two different methods to plot the so-called bundle definition file. 
    This can be removed. 

    Adapted to new IFU object input. Kept old input (still appropriate 
    for Pilot Sample data). 
    """
    
    if pilot:
        # Follow old input style, appropriate for RSS files from Pilot Sample. 
        # Open file and mask single IFU
        hdu    = pf.open(path+file_in)
        fibtab = hdu[2].data   # binary table containing fibre information
        
        mask_ifu = fibtab.field('PROBENAME')==ifu # index the selected IFU
        fibtab = fibtab[mask_ifu]                 # and mask fibre data
        
        nfib = len(fibtab)                        # count the number of fibres
        fib1 = np.where(fibtab['FIBNUM'] == 1)[0] # identify the central fibre
                
    if not pilot:
        myIFU = utils.IFU(file_in, ifu, flag_name=False)
        nfib = len(myIFU.n)

    # get true angular separation (a) between each fibre and Fib1
    # ra and dec separations will then be cos(a), sin(a)
    offset_ra  = np.zeros(nfib, dtype='double')
    offset_dec = np.zeros(nfib, dtype='double')

    for i in range(nfib):

        if pilot:
            ra1 = np.radians(fibtab['FIB_MRA'][fib1])
            ra_fib = np.radians(fibtab['FIB_MRA'][i])
            dec1  = np.radians(fibtab['FIB_MDEC'][fib1])
            dec_fib  = np.radians(fibtab['FIB_MDEC'][i])

        if not pilot:
            ra1    = np.radians(myIFU.xpos[np.where(myIFU.n == 1)])
            dec1   = np.radians(myIFU.ypos[np.where(myIFU.n == 1)])
            ra_fib  = np.radians(myIFU.xpos[i])
            dec_fib = np.radians(myIFU.ypos[i])

        # Angular distance
        cosA = np.cos(np.pi/2-dec1) * np.cos(np.pi/2-dec_fib) + \
            np.sin(np.pi/2-dec1) * np.sin(np.pi/2-dec_fib) * np.cos(ra1-ra_fib) 

        # DEC offset
        cos_dRA  = np.cos(np.pi/2-dec1) * np.cos(np.pi/2-dec1) + \
            np.sin(np.pi/2-dec1) * np.sin(np.pi/2-dec1) * np.cos(ra1-ra_fib) 

        # RA offset
        cos_dDEC = np.cos(np.pi/2-dec1) * np.cos(np.pi/2-dec_fib) + \
            np.sin(np.pi/2-dec1) * np.sin(np.pi/2-dec_fib) * np.cos(ra1-ra1) 

        # Sign check; trig collapses everything to a single quadrant, so need 
        # to check which I am on: 
        if (ra_fib >= ra1) and (dec_fib >= dec1):  # 1. quadrant (+, +)
            offset_ra[i]  = np.degrees(np.arccos(cos_dRA[0]))
            offset_dec[i] = np.degrees(np.arccos(cos_dDEC[0]))

        if (ra_fib <= ra1) and (dec_fib >= dec1):  # 2. quadrant (-, +)
            offset_ra[i]  = np.negative(np.degrees(np.arccos(cos_dRA[0])))
            offset_dec[i] = np.degrees(np.arccos(cos_dDEC[0]))

        if (ra_fib <= ra1) and (dec_fib <= dec1):  # 3. quadrant (-, -)
            offset_ra[i]  = np.negative(np.degrees(np.arccos(cos_dRA[0])))
            offset_dec[i] = np.negative(np.degrees(np.arccos(cos_dDEC[0])))

        if (ra_fib >= ra1) and (dec_fib <= dec1):  # 4. quadrant (+, -)
            offset_ra[i]  = np.degrees(np.arccos(cos_dRA[0]))
            offset_dec[i] = np.negative(np.degrees(np.arccos(cos_dDEC[0])))

    # Write a dictionary of relative RA, DEC lists
    datatab = {'RA': offset_ra, 
               'DEC': offset_dec} # proper, spherical trig, sky-projected

    """
    datatab2 = {'RA': fibtab['FIB_MRA'] - fibtab['FIB_MRA'][fib1], 
                'DEC': fibtab['FIB_MDEC'] - fibtab['FIB_MDEC'][fib1]} # simple
    """

    # Write to file
    file_out = './bundle'+str(ifu)+'.bdf'
    tab.write(datatab, file_out, names=['RA', 'DEC']) # need 'names' in order
    
    # And test the positioning:
    if diagnose==True:
        ctr = [0.0, 0.0]
        fig = plt.gcf()
        fig.clf()
        ax = fig.add_subplot(111)
        axis = [-8./3600.+ctr[0], 8./3600.+ctr[0], 
                 -8./3600.+ctr[1], 8./3600.+ctr[1]]
        plt.axis(axis)
        plt.title('Bundle '+str(ifu))
        plt.xlabel('RA Offset [degrees]')
        plt.ylabel('DEC Offset [degrees]')
        ax.set_aspect('equal')
        
        for i in range(61):
            circ = patches.Circle((datatab['RA'][i] + ctr[0], 
                                   datatab['DEC'][i] + ctr[1]), 0.8/3600.,
                                  edgecolor='none', facecolor='cyan', alpha=.5)
            ax.add_patch(circ)
            
            """
            circ2 = patches.Circle((datatab2['RA'][i] + ctr[0], 
                                    datatab2['DEC'][i] + ctr[1]), 0.8/3600.,
                                   edgecolor='none', facecolor='cyan',alpha=.5)
            ax.add_patch(circ2)
            """
    
        big_circ = patches.Circle(ctr, 7.5/3600., edgecolor='green', 
                                  facecolor='none', lw=3)
        
        #ax.add_patch(big_circ)
        plt.savefig('/Users/iraklis/Desktop/bundle.pdf', transparent=True)
        plt.show()


#########################################################################################
###                                                                                   ###
#################################--- END OF FILE ---#####################################
###                                                                                   ###
#########################################################################################

