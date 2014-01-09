#
# Written by: Samuel Richards & Iraklis Konstantopoulos
#
# Contact: Samuel Richards - samuel@physics.usyd.edu.au
# Maintainer: Iraklis Konstantopoulos - iraklis@aao.gov.au
#
# Version 6
# 08/01/2014
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
#########################################################################################

import numpy as np
import scipy as sp
import astropy.io.fits as pf
import urllib

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


#########################################################################################
###                                                                                   ###
#################################--- END OF FILE ---#####################################
###                                                                                   ###
#########################################################################################
