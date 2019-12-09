"""
This file contains some functions used during SAMI observing. These revolve around fitting stars in the RSS data.

1) centroid(infile, ifus='all', outfile=None, plot=True)

-infile should be a reduced RSS file, already passed through 2dfdr.
-ifus should be a list of the probe numbers you want to run on, e.g. [11,12].
-outfile should be a string, the desired name of the output file.
-plot should be set to True if you want to see and save the images with overlaid fits.

This function is primarily for use on commissioning star field observations. The main purpose of this function is to
calculate the offsets between the fitted positions of the stars and the centres of the hexabundles (i.e. where they
should be). Example usage:

centroid('12mar20044red.fits', ifus=[11,12], outfile='test', plot=True)

This will print to the screen the calculated offsets like so:

-------------------------------------------------
Offsets RA Dec:
Probe 11 -1.12940162731 -0.138415127654
Probe 12 0.0365293069473 1.75226680276

and will save two files: test.txt and test.pdf, the former for feeding Tony Farrell's code to calculate plate scale
errors (including global plate rotation) and the latter a pdf of the images with overlaid fits.

The widths of the fits are also printed to the screen like so:

-------------------------------------------------
FWHM of fits (in \").
[ 1.28656199  0.566648  ]

this gives you a handy measure of seeing. (Please note in this case we did not have 0.6\" seeing, there is no star in
probe 12, it's junk, designed only to illustrate use!)

2) focus(inlist, ifu)

-inlist should be a list of files to run the focus script on, one file name per line.
-ifu should be the ifu containing the star (ONE only) e.g. [11]

This function is for use during the daily telescope focus check. Example usage:

focus('focus.list', ifu=[11])

The output to the screen includes the telescope focus values and the FWHM values like so:

Focus values are (in mm): [ 38.83754565  38.1         38.3         38.5         38.7         38.9       ]
FWHM values are (in \"): [ 1.58563038  2.49753397  1.58024517  1.28656199  1.3452223   1.50470957]

Two figures will also be produced, the first showing the images and fits for the probe in question for all files. The
second plots the focus values vs fwhm values with a fitted parabola. The minimum of the fit can be picked by eye or the
official value is also printed to screen.

3) seeing(infile, ifu):

-infile should be a reduced RSS file, already passed through 2dfdr.
-ifu should be an integer (1-13).

Calculates the seeing from the star observation in a particular field.

Prints values to screen and makes a plot showing the fit.

Takes one ifu at a time so if you have many stars (i.e. a star field) then use the centroid function above.

4) centroid_fit(x,y,data,microns=True)

You shouldn't need to touch this one, it is called by the functions above (as well as by the align_micron module).

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pylab as py
import numpy as np
import scipy as sp

import os

import photutils 

# astropy fits file io (replacement for pyfits)
import astropy.io.fits as pf

import string
import itertools

from scipy.ndimage.filters import median_filter

# Circular patch.
from matplotlib.patches import Circle

from .. import utils
from .. import samifitting as fitting

# importing everything defined in the config file
from ..config import *


def centroid(infile, ifus='all', savefile=True, plot=True):
    """Fits to positions of the stars in ifus for infile. Primary purpose is to produce the files needed as imput for
    Tony's code."""

    # Define IFUs.
    if ifus=='all':
        ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        ifus=ifus

    # Number of IFUs to display
    n=len(ifus)
    print()
    print("--------------------------------------------------------------------------")
    print("I am running the centroid script on", n, "IFU(s) in file", infile) 
    print("--------------------------------------------------------------------------")
    print()

    # Number of rows and columns needed in the final display box
    # This is a bit of a fudge...
    if n==1:
        r=1
        c=1
    elif n==2:
        r=1
        c=2
    elif n==3:
        r=1
        c=3
    elif n==4:
        r=2
        c=2
    elif n>3 and n<=6:
        r=2
        c=3
    elif n>6 and n<=9:
        r=3
        c=3
    elif n>=9 and n<=12:
        r=3
        c=4
    elif n>=13 and n<=16:
        r=4
        c=4

    if plot==True:
        # Create the figure
        f0=py.figure()
        #f1=py.figure() # Add a figure for the sky coords plots.

    # Open the output file for writing
    if savefile:

        # Find the name of the input file
        outfile=str.split(os.path.basename(infile), '.')[0]
        out_txt=string.join([outfile, ".txt"],'')
        
        print("Output text file is:", out_txt)
        
        # Open the text file for writing. Note this will overwrite existing files.
        f=open(out_txt, 'w')

    else:
        outfile = None

    # List for the size of the Gaussian
    fwhm_arr=[]
    fwhm_conv_arr=[] # For the converted numbers

    # List for the x and y offsets in arcseconds.
    x_off_arr=[]
    y_off_arr=[]

    # Print the heading for the offsets
    print("-------------------------------------------------")
    print("Offsets RA Dec:")
    
    for i, ifu in enumerate(ifus):

        # Use the utils module to extract data from a single IFU.
        ifu_data=utils.IFU(infile, ifu, flag_name=False)

        # Remove the position of fibre 1, the central fibre, from all other positions to get a grid of relative positions
        idx0=np.where(ifu_data.n==1)

        x_degrees=ifu_data.xpos-ifu_data.xpos[idx0]
        y_degrees=ifu_data.ypos-ifu_data.ypos[idx0]

        x_microns=ifu_data.x_microns-ifu_data.x_microns[idx0]
        y_microns=ifu_data.y_microns-ifu_data.y_microns[idx0]

        # Feed the wrapped fitter both the micron and sky values
        p_sky, data_sky, xlin_sky, ylin_sky, model_sky=centroid_fit(x_degrees, y_degrees, ifu_data.data, microns=False,
                                                                    circular=True)
        p_mic, data_mic, xlin_mic, ylin_mic, model_mic=centroid_fit(x_microns, y_microns, ifu_data.data, circular=True)

        # Expand out the returned fitted values.
        amplitude_sky, xout_sky, yout_sky, sig_sky, bias_sky=p_sky
        amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic=p_mic
        
        # Find offsets in arcseconds using both methods
        x_off=3600*xout_sky # Note - no need to subract the central fibre as that was done before the fit.
        y_off=3600*yout_sky 

        # Use the micron values to calculate the offsets...
        centroid_microns_converted=utils.plate2sky(xout_mic, yout_mic)
        
        # Subtract the star postion from the hexabundle centre position
        x_off_conv=-1*(centroid_microns_converted[0]) # plate2sky keeps micron sign convention
        y_off_conv=centroid_microns_converted[1]
        
        # Find the widths
        x_w=sig_sky*3600.0
        xm_w=sig_mic*15.22/1000.0

        # FWHM (a measure of seeing)
        fwhm=x_w*2.35
        fwhm_arr.append(fwhm)

        fwhm_conv=xm_w*2.35
        fwhm_conv_arr.append(fwhm_conv)

        #print "FWHM from four techniques:", fwhm, fwhm_corr, fwhm_conv, fwhm_conv_corr

        print("Probe", ifu_data.ifu, x_off, y_off) #,  x_off_conv, y_off_conv  #, xm_off, ym_off, x_off, y_off

        # Add the offsets to the lists
        x_off_arr.append(x_off)
        y_off_arr.append(y_off)

        # Make an image of the bundle with the fit overlaid in contours. NOTE - plotting is done with the fit using
        # the micron values. This is more aesthetic and simple.
        if plot==True:
            
            # The limits for the axes (plotting in microns).
            xm_lower=np.min(x_microns)-100
            xm_upper=np.max(x_microns)+100
            
            ym_lower=np.min(y_microns)-100
            ym_upper=np.max(y_microns)+100

            # Debugging.
            # The limits for the axes (plotting in sky coords).
            #xs_lower=np.min(x_degrees)-0.001
            #xs_upper=np.max(x_degrees)+0.001
            
            #ys_lower=np.min(y_degrees)-0.001
            #ys_upper=np.max(y_degrees)+0.001

            #print np.min(ifu_data.xpos), np.max(ifu_data.xpos)
            #print np.min(ifu_data.ypos), np.max(ifu_data.ypos)
            
            # Add axes to the figure
            ax0=f0.add_subplot(r,c,i+1, xlim=(xm_lower, xm_upper), ylim=(ym_lower, ym_upper), aspect='equal')

            # For sky co-ords.
            #ax1=f1.add_subplot(r,c,i+1, xlim=(xs_lower, xs_upper), ylim=(ys_lower, ys_upper), aspect='equal')
            
            data_norm=data_mic/np.nanmax(data_mic)
            mycolormap=py.get_cmap('YlGnBu_r')

            # Iterate over the x, y positions (and data value) making a circle patch for each fibre, with the
            # appropriate color.
            for xmval, ymval, dataval in zip(x_microns, y_microns, data_norm):

                # Make and add the fibre patch to the axes.
                fibre_microns=Circle(xy=(xmval,ymval), radius=52.5) # 52.5
                ax0.add_artist(fibre_microns)

                fibre_microns.set_facecolor(mycolormap(dataval))

            # Add the model fit as contors.
            con0=ax0.contour(xlin_mic, ylin_mic, np.transpose(model_mic), origin='lower')
            #con1=ax1.contour(xlin_sky, ylin_sky, np.transpose(model_sky), origin='lower')

            # Title and get rid of ticks.
            title_string=string.join(['Probe ', str(ifu_data.ifu)])
            py.title(title_string)

            py.setp(ax0.get_xticklabels(), visible=False)
            py.setp(ax0.get_yticklabels(), visible=False)

            # Needed in future for debugging...
            #for xval, yval, dataval in zip(x_degrees, y_degrees, data_norm):

                # Make and add the fibre patch to the axes.
                #fibre_sky=Circle(xy=(xval,yval), radius=2.22e-4) # 52.5
                #ax1.add_artist(fibre_sky)

                #fibre_sky.set_facecolor(mycolormap(dataval))

            #py.setp(ax1.get_xticklabels(), visible=False)
            #py.setp(ax1.get_yticklabels(), visible=False)
        
        # -------------------------------------------------------
        # Write the results to file
        if outfile is not None:
            # Probe number, offset in RA ("), offset in Dec (")
            s=str(ifu_data.ifu)+' '+str(x_off)+' '+str(y_off)+'\n' # the data to write to file
            f.write(s)

    print()
    print("-------------------------------------------------")

    if plot:
        py.suptitle(infile)
        py.show()
        
        # Save the figure
        if savefile:
            # Save the figure
            out_fig=string.join([outfile, ".pdf"],'') # outfile has been defined above
            print("Output pdf file is:", out_fig)
            
            py.savefig(out_fig, format='pdf')
    
    if savefile:
        f.close() # close the output file

    # Print out the measured width values from the sky coords calculation
    fwhm_arr=np.asanyarray(fwhm_arr)
    fwhm_conv_arr=np.asanyarray(fwhm_conv_arr)
    
    print("-------------------------------------------------")
    print()
    print("FWHM of fits (in \"):")
    print(fwhm_arr)
    print() 
    # Now print the average offsets
    x_off_arr=np.asarray(x_off_arr)
    y_off_arr=np.asarray(y_off_arr)

    RA_med=np.median(x_off_arr)
    Dec_med=np.median(y_off_arr)
    
    #print "Median offsets RA/Dec (in \"):"
    #print "RA:", np.median(x_off_arr)
    #print "Dec:", np.median(y_off_arr)

    
    if RA_med < 0.0:
        RA_flag='W'
    else:
        RA_flag='E'

    if Dec_med < 0.0:
        Dec_flag='S'

    else:
        Dec_flag='N'

    print("To centre the objects in the bundles you should offset the telescope:")
    print("RA", np.abs(RA_med), RA_flag)
    print("Dec", np.abs(Dec_med), Dec_flag)

    #print fwhm_conv_arr

def focus(inlist, ifu):

    # Read in the files from the list of files.
    files=[]
    for line in open(inlist):
        cols=line.split()
        cols[0]=str.strip(cols[0])
        
        files.append(np.str(cols[0]))

    # Number of files 
    n=len(files)

    # Check the ifu makes some sense
    all=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    if ifu in all:

        print()
        print("--------------------------------------------------------------------------")
        print("I am running the focus script on probe", ifu, "for", n, "files.")
        print("--------------------------------------------------------------------------")
        print()

    else:
        print()
        print("-----------------------------------------------------------------------")
        print("You have not provided a vaild probe number. Must be between 1 and 13.")
        print("-----------------------------------------------------------------------")

        # Exit the function
        return

    # Number of rows and columns needed in the final display box
    # This is a bit of a fudge...
    if n==1:
        r=1
        c=1
    elif n==2:
        r=1
        c=2
    elif n==3:
        r=1
        c=3
    elif n==4:
        r=2
        c=2
    elif n>3 and n<=6:
        r=2
        c=3
    elif n>6 and n<=9:
        r=3
        c=3
    elif n>=9 and n<=12:
        r=3
        c=4
    elif n>=13 and n<=16:
        r=4
        c=4

    # It is possible you will have to put the values in a list here, due to header values being wrong!
    # For example, replace the line below with this: focus_vals=np.array([38.8,39.0,39.2,39.4,39.6,38.6])
    # then comment out line 308.

    focus_values=np.empty((len(files)))  # empty array for focus values from header
    fwhm_values=np.empty((len(files))) # as above for calculated fwhm values

    f0=py.figure() # figure to contain the fits
    
    for i, infile in enumerate(files):

        # Pull the focus value out of the header - NB this isn't always right!!
        focus=pf.getval(infile,'TELFOC')
        focus_values[i]=focus

        # Use the utils module to extract data from a single IFU.
        ifu_data=utils.IFU(infile, ifu, flag_name=False)

        # Remove the position of fibre 1, the central fibre, from all other positions to get a grid of relative positions
        idx0=np.where(ifu_data.n==1)

        x_degrees=ifu_data.xpos-ifu_data.xpos[idx0]
        y_degrees=ifu_data.ypos-ifu_data.ypos[idx0]

        x_microns=ifu_data.x_microns-ifu_data.x_microns[idx0]
        y_microns=ifu_data.y_microns-ifu_data.y_microns[idx0]


        # Feed the wrapped fitter both the micron and sky values
        p_sky, data_sky, xlin_sky, ylin_sky, model_sky=centroid_fit(x_degrees, y_degrees, ifu_data.data,
                                                                    microns=False, circular=True)
        p_mic, data_mic, xlin_mic, ylin_mic, model_mic=centroid_fit(x_microns, y_microns,
                                                                    ifu_data.data, circular=True)

        # Expand out the returned fitted values.
        amplitude_sky, xout_sky, yout_sky, sig_sky, bias_sky=p_sky
        amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic=p_mic

        # Find the widths
        x_w=sig_sky*3600.0 # from sky coords fit
        xm_w=sig_mic*15.22/1000.0 # from plate coords (microns) fit.

        # FWHM (a measure of seeing)
        fwhm=x_w*2.35
        fwhm_values[i]=fwhm

        # The limits for the axes (plotting in microns).
        xm_lower=np.min(x_microns)-100
        xm_upper=np.max(x_microns)+100
        
        ym_lower=np.min(y_microns)-100
        ym_upper=np.max(y_microns)+100
        
        # Add axes to the figure
        ax0=f0.add_subplot(r,c,i+1, xlim=(xm_lower, xm_upper), ylim=(ym_lower, ym_upper), aspect='equal')
        
        data_norm=data_mic/np.nanmax(data_mic)
        mycolormap=py.get_cmap('YlGnBu_r')
        
        # Iterate over the x, y positions (and data value) making a circle patch for each fibre, with the
        # appropriate color.
        for xmval, ymval, dataval in zip(x_microns, y_microns, data_norm):
            
            # Make and add the fibre patch to the axes.
            fibre=Circle(xy=(xmval,ymval), radius=52.5) # 52.5
            ax0.add_artist(fibre)
            
            fibre.set_facecolor(mycolormap(dataval))
            
        # Add the model fit as contors.
        con0=ax0.contour(xlin_mic, ylin_mic, np.transpose(model_mic), origin='lower')

        subtitle_string=string.join(['Focus ', str(focus), '\n', str(infile)])
        py.title(subtitle_string, fontsize=11)
        
        py.setp(ax0.get_xticklabels(), visible=False)
        py.setp(ax0.get_yticklabels(), visible=False)

    # Title and get rid of ticks.
    title_string=string.join(['Focus Run: Probe ', str(ifu)])
    py.suptitle(title_string)

    # Now make a plot of the focus values vs FWHM of the Gaussian fit.    
    f1=py.figure()
    ax1=f1.add_subplot(1,1,1)

    ax1.plot(focus_values, fwhm_values, 'bo') #, label=IFUlist[j])

    print()
    print("Focus values are (in mm):", focus_values)
    print("FWHM values are (in \"):", fwhm_values)
    print()

    p=np.polyfit(focus_values, fwhm_values, 2)
    focus_lin=np.arange(np.min(focus_values), np.max(focus_values)+0.1, 0.1)
    
    fit=np.polyval(p, focus_lin)
    ax1.plot(focus_lin, fit, 'r')

    ax1.set_xlabel('Telescope focus (mm)')
    ax1.set_ylabel('Star FWHM (\")')

    py.show()

    print("Focus value at minimum of fitted parabola: ", focus_lin[np.where(fit==np.min(fit))][0])

def seeing(infile, ifu):
    """
    Calculate the seeing from the star observation in a particular field.
    
    Takes one ifu at a time so if you have many stars (i.e. a star field) then use the centroid function above.
    """
    
    # Check the ifu makes some sense
    all=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    if ifu in all:
        print()
        print("------------------------------------------------------")
        print("You have told me that the PSF star is in probe", ifu)
        print("------------------------------------------------------")

    else:
        print()
        print("-----------------------------------------------------------------------")
        print("You have not provided a vaild probe number. Must be between 1 and 13.")
        print("-----------------------------------------------------------------------")

        # Exit the function
        return

    # Use the utils module to extract data from a single IFU.
    ifu_data=utils.IFU(infile, ifu, flag_name=False)

    # Remove the position of fibre 1, the central fibre, from all other positions to get a grid of relative positions
    idx0=np.where(ifu_data.n==1)

    x_degrees=ifu_data.xpos-ifu_data.xpos[idx0]
    y_degrees=ifu_data.ypos-ifu_data.ypos[idx0]

    x_microns=ifu_data.x_microns-ifu_data.x_microns[idx0]
    y_microns=ifu_data.y_microns-ifu_data.y_microns[idx0]

    # Feed the data to the fitter, using both types of coordinates.
    p_sky, data_sky, xlin_sky, ylin_sky, model_sky=centroid_fit(x_degrees, y_degrees, ifu_data.data,
                                                                    microns=False, circular=False)
    p_mic, data_mic, xlin_mic, ylin_mic, model_mic=centroid_fit(x_microns, y_microns, ifu_data.data, circular=False)

    # Expand out the returned fitted values.
    amplitude_sky, xout_sky, yout_sky, sigx_sky, sigy_sky, rot_sky, bias_sky=p_sky
    amplitude_mic, xout_mic, yout_mic, sigx_mic, sigy_mic, rot_mic, bias_mic=p_mic

    # Find the widths
    x_w=sigx_sky*3600 # from sky coords fit
    y_w=sigy_sky*3600
    
    xm_w=sigx_mic*15.22/1000 # from plate coords (microns) fit. Rough conversion
    ym_w=sigy_mic*15.22/1000 

    # FWHM (a measure of seeing)
    fwhmx_sky=x_w*2.35
    fwhmy_sky=y_w*2.35
    
    fwhmx_mic=xm_w*2.35
    fwhmy_mic=ym_w*2.35

    print()
    print("FWHM X:", np.around(fwhmx_sky, 4))
    print("FWHM Y:", np.around(fwhmy_sky, 4))
    print()
    print("Seeing (average):", np.mean([fwhmx_sky, fwhmy_sky]))

    print() 
    print("FWHM X/FWHM Y:", np.around(fwhmx_sky, 4)/np.around(fwhmy_sky, 4))
    print()

    # The limits for the axes (plotting in microns).
    xm_lower=np.min(x_microns)-100
    xm_upper=np.max(x_microns)+100
    
    ym_lower=np.min(y_microns)-100
    ym_upper=np.max(y_microns)+100

    # Create the figure
    f0=py.figure()
    
    # Add axes to the figure
    ax0=f0.add_subplot(1,1,1, xlim=(xm_lower, xm_upper), ylim=(ym_lower, ym_upper), aspect='equal')
    
    data_norm=data_mic/np.nanmax(data_mic)
    mycolormap=py.get_cmap('YlGnBu_r')
    
    # Iterate over the x, y positions (and data value) making a circle patch for each fibre, with the
    # appropriate color.
    for xmval, ymval, dataval in zip(x_microns, y_microns, data_norm):
        
        # Make and add the fibre patch to the axes.
        fibre=Circle(xy=(xmval,ymval), radius=52.5) # 52.5
        ax0.add_artist(fibre)
        
        fibre.set_facecolor(mycolormap(dataval))

    ax0.contour(xlin_mic, ylin_mic, np.transpose(model_mic), origin='lower')

    # A title for the axes
    title_string=string.join(['Probe ', str(ifu_data.ifu)])
    ax0.set_title(title_string, fontsize=14)    

def centroid_fit(x,y,data,reference=None,rssframe=None,galaxyid=None,microns=True, circular=True): #** reference,rssframe,galaxyid added 

    """Fit the x,y,data values, regardless of what they are and return some useful stuff. Data is an array of spectra"""

    working_dir = rssframe.strip('sci.fits')

    # Smooth the data spectrally to get rid of cosmics
    data_smooth=np.zeros_like(data)
    for q in range(np.shape(data)[0]):
        # data_smooth[q,:]=utils.smooth(data[q,:], 11) #default hanning smooth
        data_smooth[q,:]=median_filter(data[q,:], 15)
        
    # Now sum the data over a large range to get broad band "image"
    data_sum=np.nansum(data_smooth[:,200:1800],axis=1)
    data_med=np.nanmedian(data_smooth[:,200:1800], axis=1)

#** New masking method starts ————————————————————————————————————————————————
    from scipy.ndimage.filters import gaussian_filter
    from astropy.stats import sigma_clipped_stats
    from photutils import find_peaks

   # Parameter initializations
    x0, y0 = x-np.min(x), y-np.min(y) # image x,y
    xc, yc = (np.max(x)-np.min(x))/2.+np.min(x),(np.max(y)-np.min(y))/2.+np.min(y) # central pixel
    width = 85. # default gaussian filtering size
    checkind = 'None' # for check list
    img = np.zeros((np.max(x0)+1,np.max(y0)+1)) # rss image
    x_good, y_good, data_sum_good = x, y, data_sum  # good fibres to use
    tx,ty,trad = xc,yc,1000     #target x,y centre and masking radius (1000 means no masking)
    if not os.path.exists(working_dir+'_centroid_fit_reference/'): # path to save centre of reference frame & checklist
        os.makedirs(working_dir+'_centroid_fit_reference')

   # Load fibre flux to image
    for i in range(len(x0)):
        img[x0[i],y0[i]] = data_sum[i]

   # Gaussian filtering
    img1 = gaussian_filter(img, sigma=(width, width), order=0, mode='constant') # width = diameter of a core in degrees/microns

   # Find peaks
    mean, median, std = sigma_clipped_stats(img1, sigma=3.0)
    threshold = median + std
    tbl = find_peaks(img1, threshold, box_size=105)

   # Case1: If no peaks are found, masking is not applied. Actually I don't find any.
    if tbl == None:
        checkind = 'nopeak'

    elif(len(tbl) < 1): 
        checkind = 'nopeak'

   # Case2: A single peak is found
    elif(len(tbl) == 1): 
        checkind = 'single'
        dist = (tbl['y_peak']+np.min(x)-xc)**2+(tbl['x_peak']+np.min(y)-yc)**2    # separation between a peak and centre 
        if(dist < (310)**2): # Single peak near the centre
            tx,ty,trad = tbl['y_peak']+np.min(x), tbl['x_peak']+np.min(y),105*2  # y_peak is x. yes. it's right.
        else:  # When a peak is near the edge. High possibility that our target is not detected due to low brightness
            for k in range(1,100):  # repeat until it finds multiple peaks with reduced filtering box
                width = width*0.98
                img3 = gaussian_filter(img, sigma=(width, width), order=0, mode='constant',cval=np.min(img)) # width = diameter of a core in degrees/microns
                mean, median, std = sigma_clipped_stats(img3, sigma=3.0)
                threshold = median + std*0.1
                tbl = find_peaks(img3, threshold, box_size=width) #find peaks

                if(len(tbl)==1): # only a single peak is found until maximum iteration (=100)
                    tx,ty,trad=tbl['y_peak']+np.min(x), tbl['x_peak']+np.min(y),1000 # fibre masking is not applied (trad = 1000)
                    checkind = 'single_edge'

                if(len(tbl)>1):  # multiple peaks are found, go to Case3: multiple peaks
                    checkind = 'multi_faint'
                    break

    # Case3: When there are multiple peaks
    elif(len(tbl) > 1):
        if checkind is not 'multi_faint':
            checkind = 'multi'
        xx,yy = tbl['y_peak']+np.min(x), tbl['x_peak']+np.min(y) # y_peak is x. yes. it's right.

        # The assumption is that dithering is relatively small, and our target is near the target centre from the (1st) reference frame
        if reference is not None and rssframe != reference and os.path.exists(working_dir+'_centroid_fit_reference/centre_'+galaxyid+'_ref.txt') != False:
            fileref = open(working_dir+'_centroid_fit_reference/centre_'+galaxyid+'_ref.txt','r')
            rx,ry=np.loadtxt(fileref, usecols=(0,1))
            coff = (xx-rx)**2+(yy-ry)**2  # If not reference frame, the closest object from the reference
        else:
            coff = (xx-xc)**2+(yy-yc)**2  # If reference frame, the closest object from the centre
        
        tx, ty = xx[np.where(coff == np.min(coff))[0][0]], yy[np.where(coff == np.min(coff))[0][0]]  # target centre 
        xx, yy = xx[np.where(xx*yy != tx*ty)], yy[np.where(xx*yy != tx*ty)]
        osub = np.where(((xx-tx)**2+(yy-ty)**2 - np.min((xx-tx)**2+(yy-ty)**2)) < 0.1)   # the 2nd closest object
        trad = np.sqrt((xx[osub]-tx)**2+(yy[osub]-ty)**2)/2.   # masking radius = (a separation btw the target and 2nd closest object)/2.
        if(trad > 105*2): # when masking radius is too big
            trad = 105*2
        if(trad < 105*1.5): # when masking radius is too small
            trad = 105*1.5

    # Use fibres only within masking radius
    gsub = np.where(np.sqrt((x-tx)**2+(y-ty)**2) < trad)
    if len(gsub) < 5:
        tdist = np.sqrt((x-tx)**2+(y-ty)**2)
        inds = np.argsort(tdist)
        gsub = inds[:5]
    x_good, y_good, data_sum_good = x[gsub], y[gsub], data_sum[gsub]

    # Save the target centre of reference frame
    if reference is not None and rssframe == reference:
        ref=open(working_dir+'_centroid_fit_reference/centre_'+galaxyid+'_ref.txt','w')
        try:
            ref.write(str(tx.data[0])+' '+str(ty.data[0]))
        except:
            ref.write(str(tx)+' '+str(ty))
        ref.close()

#** New masking method ends ————————————————————————————————————————————————

   
 # Use the crude distributed centre-of-mass to get the rough centre of mass
    com=utils.comxyz(x_good,y_good,data_sum_good) #**use good data within masking

    # Peak height guess could be closest fibre to com position.
    dist=(x-com[0])**2+(y-com[1])**2 # distance between com and all fibres.
 
    # First guess at width of Gaussian - diameter of a core in degrees/microns.
    if microns==True:
        sigx=105.0
        core_diam=105.0
    else:
        sigx=4.44e-4
        core_diam=4.44e-4
  
    # First guess Gaussian parameters.
    if circular==True:
        p0=[data_sum[np.sum(np.where(dist==np.min(dist)))], com[0], com[1], sigx, 0.0]

        #print "Guess Parameters:", p0 #here

    elif circular==False:
        p0=[data_sum[np.sum(np.where(dist==np.min(dist)))], com[0], com[1], sigx, sigx, 45.0, 0.0]
        #print "Guess Parameters:", p0

    # Fit two circular 2D Gaussians.
    gf=fitting.TwoDGaussFitter(p0,x_good,y_good,data_sum_good)     #** use good data within masking
    amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic = gf.p    


    fitting.fibre_integrator(gf, core_diam) # fibre integrator
    gf.fit() #### gaussian fitting
    # Make a linear grid to reconstruct the fitted Gaussian over.
    x_0=np.min(x) 
    y_0=np.min(y)

    # dx should be 1/10th the fibre diameter (in whatever units)
    dx=sigx/10.0
    
    xlin=x_0+np.arange(100)*dx # x axis
    ylin=y_0+np.arange(100)*dx # y axis

    # Reconstruct the model
    model=np.zeros((len(xlin), len(ylin)))
    # Reconstructing the Gaussian over the proper grid.
    for ii in range(len(xlin)):
        xval=xlin[ii]
        for jj in range(len(ylin)):
            yval=ylin[jj]
            model[ii,jj]=gf.fitfunc(gf.p, xval, yval)

    amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic = gf.p    
  #  print('gx,gy final',xout_mic,yout_mic) #test
    
    return gf.p, data_sum, xlin, ylin, model


def guider_focus(values):
    
    """
    #
    # "guider_focus"
    #
    #   This function finds the best focus position for the telescope using the
    #   FWHM pix values from the guide camera as obtained via the Night Assistant
    #   using the View > Pick Object -> FWHM function in the GAIA Guide Camera
    #   software (Telescope Control Software on main Control Desk).
    #
    #   Function Example:
    #
    #   quicklook.guider_focus([[36.7,26],[36.9,19],[37.1,19],[37.3,23],[37.5,28]])
    #
    #   Input Parameters:
    #
    #     values.......Array with each cell containing the Telescope focus
    #                  positions in mm and the Guide Camera FWHM in pixels.
    #
    #                  quicklook.guider_focus([[mm,pix],[mm,pix],[mm,pix],etc...])
    #
    """
    
    focus_positions=[]
    FWHMs=[]
    
    # Get focus values from function input
    for value in values:
        
        focus_position = value[0]
        FWHM = value[1]
        
        focus_positions.append(focus_position)
        FWHMs.append(FWHM)
    
    # Fit 2nd order polynomial to data
    p=np.polyfit(focus_positions, FWHMs, 2)
    focus_lin=np.arange(np.min(focus_positions)-0.1, np.max(focus_positions)+0.1, 0.01)
    fit=np.polyval(p, focus_lin)
    
    # Equate minimum
    min_x = -p[1]/(p[0]*2)
    min_y = p[0]*(min_x**2) + p[1]*min_x + p[2]
    
    min_FWHM = min_y*0.0787 #0.0787"/pix is the image scale on the SAMI guide camera

    # Plot
    fig = py.figure()
    py.scatter(focus_positions, FWHMs)
    py.scatter(min_x, min_y,marker="o",color="r")
    py.plot(focus_lin, fit, "r")
    py.title("Telescope focus from Guider"+"\n"+"Best focus position: {0:.2f}".format(min_x)+"mm        FWHM = {0:.2f}".format(min_FWHM)+'"')
    py.xlabel("Telescope Focus Position (mm)")
    py.ylabel("FWHM (Guider Pixels)")
    
    print("---> START")
    print("--->")
    print("---> The best focus position is: {0:.2f}".format(min_x))
    print("--->")
    print("---> END")

