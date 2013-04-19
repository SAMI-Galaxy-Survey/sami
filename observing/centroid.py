import pylab as py
import numpy as np
import scipy as sp
import pyfits as pf

import string
import itertools

from scipy.stats import stats

# Circular patch.
from matplotlib.patches import Circle

from .. import utils
from .. import samifitting as fitting

# importing everything defined in the config file
from ..config import *

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

3) centroid_fit(x,y,data,microns=True)

You shouldn't need to touch this one, it is called by the two functions above.

"""

def centroid(infile, ifus='all', outfile=None, plot=True):
    """Fits to positions of the stars in ifus for infile. Primary purpose is to produce the files needed as imput for
    Tony's code."""

    # Define IFUs.
    if ifus=='all':
        ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        ifus=ifus

    # Number of IFUs to display
    n=len(ifus)
    print
    print "--------------------------------------------------------------------------"
    print "I am running the centroid script on", n, "IFU(s) in file", infile 
    print "--------------------------------------------------------------------------"
    print

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

    # Open the output file for writing
    if outfile!=None:
        out_txt=string.join([outfile, ".txt"],'')
        print "Output text file is:", out_txt
        f=open(out_txt, 'w')

    # List for the size of the Gaussian
    fwhm_arr=[]

    # Print the heading for the offsets
    print "-------------------------------------------------"
    print "Offsets RA Dec:"
    
    for i, ifu in enumerate(ifus):

        # Use the utils module to extract data from a single IFU.
        ifu_data=utils.IFU(infile, ifu, flag_name=False)

        # Feed the wrapped fitter both the micron and sky values
        p_sky, data_sky, xlin_sky, ylin_sky, model_sky=centroid_fit(ifu_data.xpos, ifu_data.ypos, ifu_data.data,
                                                                    microns=False, circular=True)
        p_mic, data_mic, xlin_mic, ylin_mic, model_mic=centroid_fit(ifu_data.x_microns, ifu_data.y_microns,
                                                                    ifu_data.data, circular=True)

        # Expand out the returned fitted values.
        amplitude_sky, xout_sky, yout_sky, sig_sky, bias_sky=p_sky
        amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic=p_mic
        
        # Find offsets in arcseconds using both methods
        x_off=3600*(xout_sky-ifu_data.xpos[np.sum(np.where(ifu_data.n==1))])*np.cos(np.pi*ifu_data.ypos[np.sum(np.where(ifu_data.n==1))]/180)
        y_off=3600*(yout_sky-ifu_data.ypos[np.sum(np.where(ifu_data.n==1))])

        xm_off=-1*(xout_mic-ifu_data.x_microns[np.where(ifu_data.n==1)])*plate_scale/1000
        ym_off=(yout_mic-ifu_data.y_microns[np.where(ifu_data.n==1)])*plate_scale/1000

        # Find the widths
        x_w=sig_sky*3600
        xm_w=sig_mic*15.22/1000

        # FWHM (a measure of seeing)
        fwhm=x_w*2.35
        fwhm_arr.append(fwhm)

        # Compare the offsets and widths. Do something with these?!
        #print "Differences (x,y,width)", np.abs(x_off-xm_off), np.abs(y_off-ym_off), np.abs(x_w-xm_w)

        print "Probe", ifu_data.ifu, x_off, y_off #, xm_off, ym_off, x_w, xm_w

        # Make an image of the bundle with the fit overlaid in contours. NOTE - plotting is done with the fit using
        # the micron values. This is more aesthetic and simple.
        if plot==True:
            
            # The limits for the axes (plotting in microns).
            xm_lower=np.min(ifu_data.x_microns)-100
            xm_upper=np.max(ifu_data.x_microns)+100
            
            ym_lower=np.min(ifu_data.y_microns)-100
            ym_upper=np.max(ifu_data.y_microns)+100
            
            # Add axes to the figure
            ax0=f0.add_subplot(r,c,i+1, xlim=(xm_lower, xm_upper), ylim=(ym_lower, ym_upper), aspect='equal')
            
            data_norm=data_mic/np.nanmax(data_mic)
            mycolormap=py.get_cmap('YlGnBu_r')

            # Iterate over the x, y positions (and data value) making a circle patch for each fibre, with the
            # appropriate color.
            for xmval, ymval, dataval in itertools.izip(ifu_data.x_microns, ifu_data.y_microns, data_norm):

                # Make and add the fibre patch to the axes.
                fibre=Circle(xy=(xmval,ymval), radius=52.5) # 52.5
                ax0.add_artist(fibre)

                fibre.set_facecolor(mycolormap(dataval))

            # Add the model fit as contors.
            con0=ax0.contour(xlin_mic, ylin_mic, np.transpose(model_mic), origin='lower')

            # Title and get rid of ticks.
            title_string=string.join(['Probe ', str(ifu_data.ifu)])
            py.title(title_string)

            py.setp(ax0.get_xticklabels(), visible=False)
            py.setp(ax0.get_yticklabels(), visible=False)
        
        # -------------------------------------------------------
        # Write the results to file
        if outfile!=None:
            # Probe number, offset in RA ("), offset in Dec (")
            s=str(ifu_data.ifu)+' '+str(x_off)+' '+str(y_off)+'\n' # the data to write to file
            f.write(s)


    if plot==True:
        py.suptitle(infile)
        py.show()
        
        # Save the figure
        if outfile!=None:
            out_fig=outfile+".pdf"
            py.savefig(out_fig, format='pdf')
    
    if outfile!=None:
        f.close() # close the output file

    # Print out the measured width values from the sky coords calculation
    fwhm_arr=np.asanyarray(fwhm_arr)
    print
    print "-------------------------------------------------"
    print "FWHM of fits (in \")."
    print fwhm_arr


def focus(inlist, ifu):

    if len(ifu)!=1:
        # Do an exception here.
        pass

    files=[]
    for line in open(inlist):
        cols=line.split()
        cols[0]=str.strip(cols[0])
        
        files.append(np.str(cols[0]))

    # Number of files 
    n=len(files)

    print
    print "--------------------------------------------------------------------------"
    print "I am running the focus script on probe", ifu[0], "for", n, "files."
    print "--------------------------------------------------------------------------"
    print

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

        # Feed the wrapped fitter both the micron and sky values
        p_sky, data_sky, xlin_sky, ylin_sky, model_sky=centroid_fit(ifu_data.xpos, ifu_data.ypos, ifu_data.data,
                                                                    microns=False, circular=True)
        p_mic, data_mic, xlin_mic, ylin_mic, model_mic=centroid_fit(ifu_data.x_microns, ifu_data.y_microns,
                                                                    ifu_data.data, circular=True)

        # Expand out the returned fitted values.
        amplitude_sky, xout_sky, yout_sky, sig_sky, bias_sky=p_sky
        amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic=p_mic

        # Find the widths
        x_w=sig_sky*3600 # from sky coords fit
        xm_w=sig_mic*15.22/1000 # from plate coords (microns) fit.

        # FWHM (a measure of seeing)
        fwhm=x_w*2.35
        fwhm_values[i]=fwhm

        # Make a figure with the fits displayed
        #ax0=f0.add_subplot(r,c,i+1)

        # The limits for the axes (plotting in microns).
        xm_lower=np.min(ifu_data.x_microns)-100
        xm_upper=np.max(ifu_data.x_microns)+100
        
        ym_lower=np.min(ifu_data.y_microns)-100
        ym_upper=np.max(ifu_data.y_microns)+100
        
        # Add axes to the figure
        ax0=f0.add_subplot(r,c,i+1, xlim=(xm_lower, xm_upper), ylim=(ym_lower, ym_upper), aspect='equal')
        
        data_norm=data_mic/np.nanmax(data_mic)
        mycolormap=py.get_cmap('YlGnBu_r')
        
        # Iterate over the x, y positions (and data value) making a circle patch for each fibre, with the
        # appropriate color.
        for xmval, ymval, dataval in itertools.izip(ifu_data.x_microns, ifu_data.y_microns, data_norm):
            
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
    title_string=string.join(['Focus Run: Probe ', str(ifu_data.ifu[0])])
    py.suptitle(title_string)

    # Now make a plot of the focus values vs FWHM of the Gaussian fit.    
    f1=py.figure()
    ax1=f1.add_subplot(1,1,1)

    ax1.plot(focus_values, fwhm_values, 'bo') #, label=IFUlist[j])

    print
    print "Focus values are (in mm):", focus_values
    print "FWHM values are (in \"):", fwhm_values
    print

    p=np.polyfit(focus_values, fwhm_values, 2)
    focus_lin=np.arange(np.min(focus_values), np.max(focus_values)+0.1, 0.1)
    
    fit=np.polyval(p, focus_lin)
    ax1.plot(focus_lin, fit, 'r')

    ax1.set_xlabel('Telescope focus (mm)')
    ax1.set_ylabel('Star FWHM (\")')

    py.show()

    print "Focus value at minimum of fitted parabola: ", focus_lin[np.where(fit==np.min(fit))]

def centroid_fit(x,y,data,microns=True, circular=True):
    """Fit the x,y,data values, regardless of what they are and return some useful stuff. Data is an array of spectra"""

    # Smooth the data spectrally to get rid of cosmics
    data_smooth=np.zeros_like(data)
    for q in xrange(np.shape(data)[0]):
        data_smooth[q,:]=utils.smooth(data[q,:], 11) #default hanning smooth
        
    # Now sum the data over a large range to get broad band "image"
    data_sum=np.nansum(data_smooth[:,200:1800],axis=1)
    data_med=stats.nanmedian(data_smooth[:,200:1800], axis=1)

    # Use the crude distributed centre-of-mass to get the rough centre of mass
    com=utils.comxyz(x,y,data_sum)
    
    # Peak height guess could be closest fibre to com position.
    dist=(x-com[0])**2+(y-com[1])**2 # distance between com and all fibres.
        
    # First guess at width of Gaussian - diameter of a core in degrees/microns (distance between core 1 and core 2?) in whichever direction that is larger....
    if microns==True:
        sigx=105.0
        core_diam=52.5

    else:
        sigx=4.44e-4
        core_diam=2.22e-4

    
    # First guess Gaussian parameters.
    if circular==True:
        p0=[data_sum[np.sum(np.where(dist==np.min(dist)))], com[0], com[1], sigx, 0.0]

    elif circular==False:
        p0=[data_sum[np.sum(np.where(dist==np.min(dist)))], com[0], com[1], sigx, sigx, 45.0, 0.0]
        
    # Fit two circular 2D Gaussians.
    gf=fitting.TwoDGaussFitter(p0,x,y,data_sum)
    fitting.fibre_integrator(gf, core_diam)
    gf.fit()

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
    for ii in xrange(len(xlin)):
        xval=xlin[ii]
        for jj in xrange(len(ylin)):
            yval=ylin[jj]
            model[ii,jj]=gf.fitfunc(gf.p, xval, yval)
    
    return gf.p, data_sum, xlin, ylin, model
    
