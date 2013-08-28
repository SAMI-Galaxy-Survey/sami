import pylab as py
import numpy as np
import scipy as sp

import astropy.io.fits as pf

import string
import itertools

from scipy.stats import stats

# Circular patch.
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from .. import utils # Module containing utilities functions for SAMI

"""
This script contains the display functions for SAMI data.

"""
def display(infile, ifus='all', log=True):
    """Plot broad band images of each of the ifus asked for, from infile"""
   
    # Define the list of IFUs to display
    if ifus=='all':
        ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        ifus=ifus

    # Number of IFUs to display
    n=len(ifus)
    print "I have received", len(ifus), 'IFU(s) to display.'

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
    elif n>12 and n<=16:
        r = 4
        c = 4
    
    # Create the figure.
    fig=py.figure()

    for i, ifu in enumerate(ifus):

        # Get the data.
        ifu_data=utils.IFU(infile, ifu, flag_name=False)

        # Sum up the data - across a broad band, 200:1800 pixels.
        data_sum=np.nansum(ifu_data.data[:,200:1800],axis=1)
        data_med=stats.nanmedian(ifu_data.data[:,200:1800],axis=1)

        # X and Y positions in microns.
        x_m=ifu_data.x_microns
        y_m=ifu_data.y_microns

        x_lower=np.min(x_m)-100
        x_upper=np.max(x_m)+100

        y_lower=np.min(y_m)-100
        y_upper=np.max(y_m)+100

        # Mask out the negative values that occur because of bad tp correction.
        for value in data_sum:
            if value<=0:
                value=1

        # If the log tag is True log the data array.
        if log==True:
            data_sum=np.log10(data_sum)

        # Add a subplot to the figure.
        ax=fig.add_subplot(r,c,i+1, xlim=(x_lower, x_upper), ylim=(y_lower, y_upper), aspect='equal')
        
        # Normalise the data and select the colormap.
        data_norm=data_sum/np.nanmax(data_sum)
        mycolormap=py.get_cmap('YlGnBu_r')

        fibres=[]
        # Iterate over the x, y positions making a circle patch for each fibre, with the appropriate color.
        for xval, yval, dataval in itertools.izip(x_m, y_m, data_norm):
            #Add the fibre patch.
            fibre=Circle(xy=(xval,yval), radius=52.5)
            fibres.append(fibre)
            
            #fibre.set_facecolor(mycolormap(dataval))

        allpatches=PatchCollection(fibres, cmap=mycolormap) 
        allpatches.set_array(data_norm)

        ax.add_collection(allpatches)
        py.colorbar(allpatches)

        # Give each subplot a title (necessary?).
        title_string=string.join(['Probe ', str(ifu_data.ifu)])
        ax.set_title(title_string, fontsize=11)

        # Get rid of the tick labels.
        py.setp(ax.get_xticklabels(), visible=False)
        py.setp(ax.get_yticklabels(), visible=False)

    # Super title for plot - the filename.
    py.suptitle(infile)

def display_list(inlist, ifu, log=True):
    """For each file in inlist, plot a collapsed broad band image of the hexabundle"""

    files=[]

    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])
        
        files.append(np.str(cols[0]))

    # Number of files 
    n=len(files)

    print "I have received", n, "files to plot."

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

    # Create the figure
    fig=py.figure()

    for i, infile in enumerate(files):

        # Get the data.
        ifu_data=utils.IFU(infile, ifu, flag_name=False)

        # Sum up the data - across a broad band, 200:1800 pixels.
        data_sum=np.nansum(ifu_data.data[:,200:1800],axis=1)
        data_med=stats.nanmedian(ifu_data.data[:,200:1800],axis=1)

        # X and Y positions in microns.
        x_m=ifu_data.x_microns
        y_m=ifu_data.y_microns

        x_lower=np.min(x_m)-100
        x_upper=np.max(x_m)+100

        y_lower=np.min(y_m)-100
        y_upper=np.max(y_m)+100

        # Mask out the negative values that occur because of bad tp correction.
        for value in data_sum:
            if value<=0:
                value=1

        # If the log tag is True log the data array.
        if log==True:
            data_sum=np.log10(data_sum)

        # Add a subplot to the figure.
        ax=fig.add_subplot(r,c,i+1, xlim=(x_lower, x_upper), ylim=(y_lower, y_upper), aspect='equal')
        
        # Normalise the data and select the colormap.
        data_norm=data_sum/np.nanmax(data_sum)
        mycolormap=py.get_cmap('YlGnBu_r')

        # Iterate over the x, y positions making a circle patch for each fibre, with the appropriate color.
        for xval, yval, dataval in itertools.izip(x_m, y_m, data_norm):
            #Add the fibre patch.
            fibre=Circle(xy=(xval,yval), radius=52.5)
            ax.add_artist(fibre)

            fibre.set_facecolor(mycolormap(dataval))

        # Get rid of the tick labels.
        py.setp(ax.get_xticklabels(), visible=False)
        py.setp(ax.get_yticklabels(), visible=False)

        subtitle_string=str(infile)
        py.title(subtitle_string, fontsize=11)

    # Title for the plot - the probe number.
    title_string=string.join(['Probe ', str(ifu)])
    fig.suptitle(title_string)

def summed_spectrum(infile, ifu, overplot=False):
    """Sums all spectra in an RSS file and plots the summed spectrum against wavelength"""

    # Only close windows if overplot tag is False (note, strings not Booleans)
    if overplot==False:
        # Close any active figure windows
        py.close('all')

    if overplot==True:
        print "Overplotting..."

    # Get the data.
    ifu_data=utils.IFU(infile, ifu, flag_name=False)

    # Wavelength range.
    L=ifu_data.lambda_range

    # Sum up the data - across all fibres in the hexabundle.
    data_sum=np.nansum(ifu_data.data,axis=0)
    data_med=stats.nanmedian(ifu_data.data,axis=0)
    
    # Plot the spectrum.
    py.plot(L, data_sum)

    # Put the file name as a title.
    py.title(infile)
    
