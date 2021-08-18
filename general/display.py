"""
This module contains many useful plotting functions, mostly for observers.

Each plotting function is described in its docstring. This module has quite
a lot of fudges and magic numbers that work fine for the SAMI Galaxy Survey
but may not always work for other data sets.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import pylab as py
import numpy as np
import scipy as sp

import astropy.io.fits as pf

import string
import itertools

# Circular patch.
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from .. import utils # Module containing utilities functions for SAMI

def display(infile, ifus='all', log=True):
    """Plot broad band images of each of the ifus asked for, from infile.

    infile is the path to a reduced FITS file.
    ifus is a list of probe numbers, or the string 'all'.
    log determines whether the images are plotted on a log scale.
    """
   
    # Define the list of IFUs to display
    if ifus=='all':
        ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        ifus=ifus

    # Number of IFUs to display
    n=len(ifus)
    print(("I have received", len(ifus), 'IFU(s) to display.'))

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
        data_med=np.nanmedian(ifu_data.data[:,200:1800],axis=1)

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
        for xval, yval, dataval in zip(x_m, y_m, data_norm):
            #Add the fibre patch.
            fibre=Circle(xy=(xval,yval), radius=52.5)
            fibres.append(fibre)
            
            #fibre.set_facecolor(mycolormap(dataval))

        allpatches=PatchCollection(fibres, cmap=mycolormap, edgecolors='none') 
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
    """Plot collapsed broad band images for one probe across several files.

    inlist is the path to an ASCII file that contains a list of paths to
        reduced FITS files (one per line).
    ifu is an integer determining the probe number to plot.
    """

    files=[]

    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])
        
        files.append(np.str(cols[0]))

    # Number of files 
    n=len(files)

    print(("I have received", n, "files to plot."))

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
        data_med=np.nanmedian(ifu_data.data[:,200:1800],axis=1)

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
        for xval, yval, dataval in zip(x_m, y_m, data_norm):
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
    """Sums all spectra in a probe and plots the summed spectrum.

    infile is the path to a reduced FITS file.
    ifu is an integer determining the probe number to plot.
    if overplot is False a new figure is made (and existing ones closed!).
    """

    # Only close windows if overplot tag is False
    if overplot==False:
        # Close any active figure windows
        # Shouldn't we just open a new figure instead?
        py.close('all')

    if overplot==True:
        print("Overplotting...")

    # Get the data.
    ifu_data=utils.IFU(infile, ifu, flag_name=False)

    # Wavelength range.
    L=ifu_data.lambda_range

    # Sum up the data - across all fibres in the hexabundle.
    data_sum=np.nansum(ifu_data.data,axis=0)
    data_med=np.nanmedian(ifu_data.data,axis=0)
    
    # Plot the spectrum.
    py.plot(L, data_sum)

    # Put the file name as a title.
    py.title(infile)


def field(infile, ifus='all', log=True):
    """Plots images of galaxies in their positions in the field.

    Inputs are as for display()
    """

    # Define the list of IFUs to display
    if ifus=='all':
        ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        ifus=ifus

    # Create a figure
    f1=py.figure()
    ax=f1.add_subplot(1,1,1, xlim=(118365,-118365), ylim=(-118365,118365), aspect='equal')
    
    for i, ifu in enumerate(ifus):

        # Get the data.
        ifu_data=utils.IFU(infile, ifu, flag_name=False)
        
        # Sum up the data - across a broad band, 200:1800 pixels.
        data_sum=np.nansum(ifu_data.data[:,200:1800],axis=1)
        data_med=np.nanmedian(ifu_data.data[:,200:1800],axis=1)

        # X and Y positions in microns.
        x_m=ifu_data.x_microns
        y_m=ifu_data.y_microns

        # Blow up the hexabundle
        x_m0=x_m[np.where(ifu_data.n==1)]
        y_m0=y_m[np.where(ifu_data.n==1)]

        x_m_delta=(x_m-x_m0)*20
        y_m_delta=(y_m-y_m0)*20

        x_m_new=x_m0+x_m_delta
        y_m_new=y_m0+y_m_delta
        
        print(x_m_delta)
        print(y_m_delta)

        #x_lower=np.min(x_m)-100
        #x_upper=np.max(x_m)+100

        #y_lower=np.min(y_m)-100
        #y_upper=np.max(y_m)+100

        # Mask out the negative values that occur because of bad tp correction.
        for value in data_sum:
            if value<=0:
                value=1

        # If the log tag is True log the data array.
        if log==True:
            data_sum=np.log10(data_sum)

        # Add a subplot to the figure.
        field_circle=Circle(xy=(0,0), radius=118265, fc='none')
        ax.add_artist(field_circle)
        
        # Normalise the data and select the colormap.
        data_norm=data_sum/np.nanmax(data_sum)
        mycolormap=py.get_cmap('YlGnBu_r')

        fibres=[]
        # Iterate over the x, y positions making a circle patch for each fibre, with the appropriate color.
        for xval, yval, dataval in zip(x_m_new, y_m_new, data_norm):
            #Add the fibre patch.
            fibre=Circle(xy=(xval,yval), radius=1500)
            fibres.append(fibre)
            
            fibre.set_facecolor(mycolormap(dataval))

        allpatches=PatchCollection(fibres, cmap=mycolormap) 
        allpatches.set_array(data_norm)

        ax.add_collection(allpatches)
        #py.colorbar(allpatches)

        # Give each subplot a title (necessary?).
        #title_string=string.join(['Probe ', str(ifu_data.ifu)])
        #ax.set_title(title_string, fontsize=11)

        # Get rid of the tick labels.
        #py.setp(ax.get_xticklabels(), visible=False)
        #py.setp(ax.get_yticklabels(), visible=False)

        

def raw(flat_file, object_file, IFU="unknown", sigma_clip=False, log=True,
        pix_waveband=100, pix_start="unknown",
        old_plot_style=False):
    """
    #
    # "raw"
    #
    #   Takes in a raw flat field and a raw object frame. Performs a cut on the flat
    #   field along the centre of the CCD to get fibre row positions.
    #   Collapses +/- 50 wavelength pix on the object frame at those positions
    #   and plots them.
    #
    #   Function Example:
    #
    #       sami.display.raw("02sep20045.fits","02sep20053.fits",Probe_to_fit=2,
    #                   sigma_clip=True)
    #
    #   Input Parameters:
    #
    #       flat_file.......File name string of the raw flat frame to find tramlines
    #                       on (e.g. "02sep20045.fits").
    #
    #       object_file.....File name string of the object frame wanting to be
    #                       displayed (e.g. "02sep20048.fits").
    #
    #       IFU.............Integer value to only display that IFU
    #
    #       sigma_clip......Switch to turn sigma clip on and off. If it is on the
    #                       code will run ~20s slower for a pix_waveband of 100. If
    #                       turned off there is a chance that cosmic rays/bad pixels
    #                       will dominate the image stretch and 2D Gauss fits. It is
    #                       strongly advised to turn this on when dealing with the
    #                       Blue CCD as there are many bad pixels. In the Red CCD you
    #                       can normally get away with leaving it off for the sake of
    #                       saving time.
    #
    #       log.............Switch to select display in log or linear (default is log)
    #
    #       pix_waveband....Number of pixels in wavelength direction to bin over,
    #                       centered at on the column of the spatial cut. 100pix is
    #                       enough to get flux contrast for a good fit/image display.
    #
    #       pix_start.......This input is for times where the spatial cut finds 819
    #                       peaks but doesn't find only fibres (e.g. 817 fibres and
    #                       2 cosmic rays). This will be visible in the display
    #                       output and if such a case happens, input the pixel
    #                       location where the previous spatial cut was performed and
    #                       the code will search for better place where 819 fibres
    #                       are present. Keep doing this until 819 are found, and if
    #                       none are found then something is wrong with the flat
    #                       frame and use another.
    #
    """
    
    print("---> START")
    print("--->")
    print(("---> Object frame: "+str(object_file)))
    print("--->")
    
    # Import flat field frame
    flat = pf.open(flat_file)
    flat_data = flat['Primary'].data

    # Range to find spatial cut
    if pix_start != "unknown":
        cut_loc_start = np.float(pix_start+5)/np.float(2048)
        cut_locs = np.linspace(cut_loc_start,0.75,201)
    else:
        cut_locs = np.linspace(0.25,0.75,201)
    
    print("---> Finding suitable cut along spatial dimension...")
    # Check each spatial slice until 819 fibres (peaks) have been found
    for cut_loc in cut_locs:
        # perform cut along spatial direction
        flat_cut = flat_data[:,int(np.shape(flat_data)[1]*cut_loc)]
        flat_cut_leveled = flat_cut - 0.1*np.max(flat_cut)
        flat_cut_leveled[flat_cut_leveled < 0] = 0.
        # find peaks (fibres)
        peaks = peakdetect(flat_cut_leveled, lookahead = 3)
        Npeaks = np.shape(peaks[0])[0]
        if Npeaks == 819:
            break
        else:
            continue
    
    print("--->")
    
    # If 819 fibres can't be found then exit script. At the moment this script can't cope with broken or missing fibres.
    if Npeaks != 819:
        raise ValueError("---> Can't find 819 fibres. Check [1] Flat Field is correct "+
            "[2] Flat Field is supplied as the first variable in the function. If 1+2"+
            " are ok then use the 'pix_start' variable and set it at least 10 pix beyond"+
            " the previous value (see terminal for value)")
    
    print(("---> Spatial cut at pixel number: ",int(cut_loc*2048)))
    print(("---> Number of waveband pixels: ",pix_waveband))
    print(("---> Number of fibres found: ",np.shape(peaks[0])[0]))
    print("--->")
    
    # Location of fibre peaks for linear tramline
    tram_loc=[]
    for i in np.arange(np.shape(peaks[0])[0]):
        tram_loc.append(peaks[0][i][0])
    
    # Import object frame
    object = pf.open(object_file)
    object_data = object['Primary'].data
    object_fibtab = object['MORE.FIBRES_IFU'].data
    object_guidetab = object['MORE.FIBRES_GUIDE'].data
    object_guidetab = object_guidetab[object_guidetab['TYPE']=='G']

    # Perform cut along spatial direction at same position as cut_loc
    s = np.shape(object_data)
    object_cut = object_data[:,int((s[1]*cut_loc)-pix_waveband/2):int((s[1]*cut_loc)+pix_waveband/2)]
    
    # "Sigma clip" to get set bad pixels as row median value
    if sigma_clip == True:
        print("---> Performing 'Sigma-clip'... (~20s)")
        for i in np.arange(np.shape(object_cut)[0]):
            for j in np.arange(np.shape(object_cut)[1]):
                med = np.median(object_cut[i,:])
                err = np.absolute((object_cut[i,j]-med)/med)
                if err > 0.25:
                    object_cut[i,j] = med
        print("--->")
    
    # Collapse spectral dimension
    object_cut_sum = np.nansum(object_cut,axis=1)
    
    # Extract intensities at fibre location and log
    object_spec = object_cut_sum[tram_loc]
    
    Probe_list = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    
    # Plot the data
    print("---> Plotting...")
    print("--->")

    if old_plot_style:
        fig = py.figure()
        if IFU != "unknown":
            fig.suptitle("SAMI Display of raw frame: "+str(object_file),fontsize=15)
            ax = fig.add_subplot(1,1,1)
            ax.set_aspect('equal')
            ind_all = np.where((object_fibtab.field('TYPE')=="P") & 
                                (object_fibtab.field('PROBENUM')==IFU))
            ind_one = np.where((object_fibtab.field('TYPE')=="P") & 
                                (object_fibtab.field('PROBENUM')==IFU) &
                                (object_fibtab.field('FIBNUM')==1))
            Probe_data = object_spec[ind_all]
            x = object_fibtab.field('FIB_PX')[ind_all] - object_fibtab.field('FIB_PX')[ind_one][3*(-IFU+14) - 2]
            y = -(object_fibtab.field('FIB_PY')[ind_all] - object_fibtab.field('FIB_PY')[ind_one][3*(-IFU+14) - 2])
            radii = np.zeros(len(x)) + 52.5
            patches = []
            for x1,y1,r in zip(x[0:len(x)], y[0:len(y)], radii):
                circle = Circle((x1,y1), r)
                patches.append(circle)
            if log:
                colors = np.log(Probe_data)
            else:
                colors = Probe_data
            pa = PatchCollection(patches, cmap=py.cm.YlGnBu_r)
            pa.set_array(colors)
            ax.add_collection(pa)
            py.axis([-600, 600, -600, 600])
            py.setp(ax.get_xticklabels(), visible=False)
            py.setp(ax.get_yticklabels(), visible=False)
            py.title("Probe "+str(IFU), fontsize=10)

        else:
            fig.suptitle("SAMI Display of raw frame: "+str(object_file),fontsize=15)
            for Probe in Probe_list:
                ax = fig.add_subplot(4,4,Probe)
                ax.set_aspect('equal')
                ind_all = np.where((object_fibtab.field('TYPE')=="P") & 
                                    (object_fibtab.field('PROBENUM')==Probe))
                ind_one = np.where((object_fibtab.field('TYPE')=="P") & 
                                (object_fibtab.field('PROBENUM')==Probe) &
                                (object_fibtab.field('FIBNUM')==1))
                Probe_data = object_spec[ind_all]
                x = object_fibtab.field('FIB_PX')[ind_all] - object_fibtab.field('FIB_PX')[ind_one]
                y = -(object_fibtab.field('FIB_PY')[ind_all] - object_fibtab.field('FIB_PY')[ind_one])
                radii = np.zeros(len(x)) + 52.5
                patches = []
                for x1,y1,r in zip(x[0:len(x)], y[0:len(y)], radii):
                    circle = Circle((x1,y1),r)
                    patches.append(circle)
                if log:
                    colors = np.log(Probe_data)
                else:
                    colors = Probe_data
                pa = PatchCollection(patches, cmap=py.cm.YlGnBu_r)
                pa.set_array(colors)
                ax.add_collection(pa)
                py.axis([-600, 600, -600, 600])
                py.setp(ax.get_xticklabels(), visible=False)
                py.setp(ax.get_yticklabels(), visible=False)
                py.title("Probe "+str(Probe), fontsize=10)
    else:

        scale_factor = 18

        def display_ifu(x_coords, y_coords, xcen, ycen, scaling, values):
            bundle_patches = []
            for x1,y1 in zip(x_coords, y_coords):
                circle = Circle((x1*scaling+xcen,y1*scaling+ycen), 52.5*scaling)
                bundle_patches.append(circle)
            pcol = PatchCollection(bundle_patches, cmap=py.get_cmap('afmhot'))
            pcol.set_array(values)
            pcol.set_edgecolors('none')
            return pcol

        fig = py.figure(figsize=(10,10))
        fig.suptitle("SAMI Display of raw frame: "+str(object_file),fontsize=15)

        ax = fig.add_subplot(1,1,1)
        ax.set_aspect('equal')

        ax.add_patch(Circle((0,0), 264/2*1000, facecolor="#cccccc", edgecolor='#000000', zorder=-1))

        for Probe in Probe_list:
            ind_all = np.where((object_fibtab.field('TYPE')=="P") & 
                                (object_fibtab.field('PROBENUM')==Probe))
            Probe_data = object_spec[ind_all]
            
            mask = np.logical_and(object_fibtab.field('TYPE')=="P",
                                  object_fibtab['PROBENUM']==Probe)

            mean_x = np.mean(object_fibtab.field('FIBPOS_X')[mask])
            mean_y = np.mean(object_fibtab.field('FIBPOS_Y')[mask])

            x = object_fibtab.field('FIBPOS_X')[mask] - mean_x
            y = object_fibtab.field('FIBPOS_Y')[mask] - mean_y

            ax.add_collection(display_ifu(x, y, mean_x, mean_y, scale_factor, Probe_data))
            ax.axis([-140000, 140000, -140000, 140000])
            py.setp(ax.get_xticklabels(), visible=False)
            py.setp(ax.get_yticklabels(), visible=False)
            ax.text(mean_x, mean_y - scale_factor*750, "Probe " + str(Probe),
                    verticalalignment="bottom", horizontalalignment='center')

        for probe_number, x, y in zip(
                object_guidetab['PROBENUM'], object_guidetab['CENX'], object_guidetab['CENY']):
            ax.add_patch(Circle((x,y), scale_factor*250, edgecolor='#009900', facecolor='none'))
            ax.text(x, y, "G" + str(probe_number),
                    verticalalignment='center', horizontalalignment='center')

        ax.arrow(100000,100000,0,15000, color="#aa0000", edgecolor='#aa0000', width=100)
        ax.text(101000,116000, 'North', verticalalignment="bottom", horizontalalignment='left')

        ax.arrow(100000,100000,15000,0, color="#aa0000", edgecolor='#aa0000', width=0)
        ax.text(116000,101000, 'East', verticalalignment="bottom", horizontalalignment='left')

        py.tight_layout()
        fig.show()

    print("---> END")

#########################################################################################

def raw2(flat_file, object_file, IFU="unknown", sigma_clip=False, log=True,
        pix_waveband=100, pix_start="unknown"):
    """
    #
    # "raw"
    #
    #   Takes in a raw flat field and a raw object frame. Performs a cut on the flat
    #   field along the centre of the CCD to get fibre row positions.
    #   Collapses +/- 50 wavelength pix on the object frame at those positions
    #   and plots them.
    #
    #   Function Example:
    #
    #       sami.display.raw("02sep20045.fits","02sep20053.fits",Probe_to_fit=2,
    #                   sigma_clip=True)
    #
    #   Input Parameters:
    #
    #       flat_file.......File name string of the raw flat frame to find tramlines
    #                       on (e.g. "02sep20045.fits").
    #
    #       object_file.....File name string of the object frame wanting to be
    #                       displayed (e.g. "02sep20048.fits").
    #
    #       IFU.............Integer value to only display that IFU
    #
    #       sigma_clip......Switch to turn sigma clip on and off. If it is on the
    #                       code will run ~20s slower for a pix_waveband of 100. If
    #                       turned off there is a chance that cosmic rays/bad pixels
    #                       will dominate the image stretch and 2D Gauss fits. It is
    #                       strongly advised to turn this on when dealing with the
    #                       Blue CCD as there are many bad pixels. In the Red CCD you
    #                       can normally get away with leaving it off for the sake of
    #                       saving time.
    #
    #       log.............Switch to select display in log or linear (default is log)
    #
    #       pix_waveband....Number of pixels in wavelength direction to bin over,
    #                       centered at on the column of the spatial cut. 100pix is
    #                       enough to get flux contrast for a good fit/image display.
    #
    #       pix_start.......This input is for times where the spatial cut finds 819
    #                       peaks but doesn't find only fibres (e.g. 817 fibres and
    #                       2 cosmic rays). This will be visible in the display
    #                       output and if such a case happens, input the pixel
    #                       location where the previous spatial cut was performed and
    #                       the code will search for better place where 819 fibres
    #                       are present. Keep doing this until 819 are found, and if
    #                       none are found then something is wrong with the flat
    #                       frame and use another.
    #
    """
    
    print("---> START")
    print("--->")
    print(("---> Object frame: "+str(object_file)))
    print("--->")
    
    # Import flat field frame
    flat = pf.open(flat_file)
    flat_data = flat['Primary'].data
    flat_fibtab = flat['MORE.FIBRES_IFU'].data

    # Range to find spatial cut
    if pix_start != "unknown":
        cut_loc_start = np.float(pix_start+5)/np.float(2048)
        cut_locs = np.linspace(cut_loc_start,0.75,201)
    else:
        cut_locs = np.linspace(0.25,0.75,201)
        
    Nfibs = sum(flat_fibtab['SELECTED'])
    
    print("---> Finding suitable cut along spatial dimension...")
    # Check each spatial slice until 819 fibres (peaks) have been found
    for cut_loc in cut_locs:
        # perform cut along spatial direction
        flat_cut = flat_data[:,int(np.shape(flat_data)[1]*cut_loc)]
        flat_cut_leveled = flat_cut - 0.1*np.max(flat_cut)
        flat_cut_leveled[flat_cut_leveled < 0] = 0.
        # find peaks (fibres)
        peaks = peakdetect(flat_cut_leveled, lookahead = 3)
        Npeaks = np.shape(peaks[0])[0]
        if Npeaks == Nfibs:
            break
        else:
            continue
    
    print("--->")
    
    # If 819 fibres can't be found then exit script. At the moment this script can't cope with broken or missing fibres.
    if Npeaks != Nfibs:
        raise ValueError("---> Can't find {} fibres. Check [1] Flat Field is correct ".format(Nfibs)+
            "[2] Flat Field is supplied as the first variable in the function. If 1+2"+
            " are ok then use the 'pix_start' variable and set it at least 10 pix beyond"+
            " the previous value (see terminal for value)")
    
    print(("---> Spatial cut at pixel number: ",int(cut_loc*2048)))
    print(("---> Number of waveband pixels: ",pix_waveband))
    print(("---> Number of fibres found: ",np.shape(peaks[0])[0]))
    print("--->")
    
    # Location of fibre peaks for linear tramline
    tram_loc=[]
    for i in np.arange(np.shape(peaks[0])[0]):
        tram_loc.append(peaks[0][i][0])
    
    # Import object frame
    object = pf.open(object_file)
    object_data = object['Primary'].data
    object_fibtab = object['MORE.FIBRES_IFU'].data
    object_guidetab = object['MORE.FIBRES_GUIDE'].data
    object_guidetab = object_guidetab[object_guidetab['TYPE']=='G']

    # Perform cut along spatial direction at same position as cut_loc
    s = np.shape(object_data)
    object_cut = object_data[:,int((s[1]*cut_loc)-pix_waveband/2):int((s[1]*cut_loc)+pix_waveband/2)]
    
    # "Sigma clip" to get set bad pixels as row median value
    if sigma_clip == True:
        print("---> Performing 'Sigma-clip'... (~20s)")
        for i in np.arange(np.shape(object_cut)[0]):
            for j in np.arange(np.shape(object_cut)[1]):
                med = np.median(object_cut[i,:])
                err = np.absolute((object_cut[i,j]-med)/med)
                if err > 0.25:
                    object_cut[i,j] = med
        print("--->")
    
    # Collapse spectral dimension
    object_cut_sum = np.nansum(object_cut,axis=1)
    
    # Extract intensities at fibre location and log
    object_spec = object_cut_sum[tram_loc]
    
    Probe_list = np.unique(flat_fibtab['GROUP_N'][np.where(flat_fibtab['SELECTED']==1)])
    Probe_list = Probe_list[Probe_list < 99]
    
    # Plot the data
    print("---> Plotting...")
    print("--->")

    scale_factor = 18

    def display_ifu(x_coords, y_coords, xcen, ycen, scaling, values):
            bundle_patches = []
            for x1,y1 in zip(x_coords, y_coords):
                circle = Circle((x1*scaling+xcen,y1*scaling+ycen), 52.5*scaling)
                bundle_patches.append(circle)
            pcol = PatchCollection(bundle_patches, cmap=py.get_cmap('afmhot'))
            pcol.set_array(values)
            pcol.set_edgecolors('none')
            return pcol

    fig = py.figure(figsize=(10,10))
    fig.suptitle("SAMI Display of raw frame: "+str(object_file),fontsize=15)

    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')

    ax.add_patch(Circle((0,0), 264/2*1000, facecolor="#cccccc", edgecolor='#000000', zorder=-1))
    ax.add_patch(Circle((0,0), 264*1000, facecolor="#cccccc", edgecolor='#000000', zorder=-1))
    
    object_fibtab = object_fibtab[np.where(object_fibtab.field('SELECTED')==1)]

    for Probe in Probe_list:
        ind_all = np.where((object_fibtab.field('TYPE')=="P") & 
                                (object_fibtab.field('PROBENUM')==Probe))
        try:
            Probe_data = object_spec[ind_all]
        except:
            import code
            code.interact(local=dict(globals(),**locals()))
        
        mask = np.logical_and(object_fibtab.field('TYPE')=="P",
                                  object_fibtab['PROBENUM']==Probe)

        mean_x = np.mean(object_fibtab.field('FIBPOS_X')[mask])
        mean_y = np.mean(object_fibtab.field('FIBPOS_Y')[mask])

        x = object_fibtab.field('FIBPOS_X')[mask] - mean_x
        y = object_fibtab.field('FIBPOS_Y')[mask] - mean_y

        ax.add_collection(display_ifu(x, y, mean_x, mean_y, scale_factor, Probe_data))
        ax.axis([-140000*2, 140000*2, -140000*2, 140000*2])
        py.setp(ax.get_xticklabels(), visible=False)
        py.setp(ax.get_yticklabels(), visible=False)
        ax.text(mean_x, mean_y - scale_factor*750, "Probe " + str(Probe),
                    verticalalignment="bottom", horizontalalignment='center')

    for probe_number, x, y in zip(
                object_guidetab['PROBENUM'], object_guidetab['CENX'], object_guidetab['CENY']):
        ax.add_patch(Circle((x,y), scale_factor*250, edgecolor='#009900', facecolor='none'))
        ax.text(x, y, "G" + str(probe_number),
                    verticalalignment='center', horizontalalignment='center')

    ax.arrow(200000,200000,0,15000, color="#aa0000", edgecolor='#aa0000', width=100)
    ax.text(201000,216000, 'North', verticalalignment="bottom", horizontalalignment='left')

    ax.arrow(200000,200000,15000,0, color="#aa0000", edgecolor='#aa0000', width=0)
    ax.text(216000,201000, 'East', verticalalignment="bottom", horizontalalignment='left')

    py.tight_layout()
    fig.show()

    print("---> END")

#########################################################################################

def peakdetect(y_axis, x_axis = None, lookahead = 300, delta=0):
    
    """
    #
    # "peakdetect"
    #
    #   Determines peaks from data. Translation of the MATLAB code "peakdet.m"
    #   and taken from https://gist.github.com/sixtenbe/1178136
    #
    #   Called by "raw"
    #
    """
    
    i = 10000
    x = np.linspace(0, 3.5 * np.pi, i)
    y = (0.3*np.sin(x) + np.sin(1.3 * x) + 0.9 * np.sin(4.2 * x) + 0.06 * np.random.randn(i))
    
    # Converted from/based on a MATLAB script at:
    # http://billauer.co.il/peakdet.html
    
    # function for detecting local maximas and minmias in a signal.
    # Discovers peaks by searching for values which are surrounded by lower
    # or larger values for maximas and minimas respectively
    
    # keyword arguments:
    # y_axis........A list containg the signal over which to find peaks
    # x_axis........(optional) A x-axis whose values correspond to the y_axis list
    #               and is used in the return to specify the postion of the peaks. If
    #               omitted an index of the y_axis is used. (default: None)
    # lookahead.....(optional) distance to look ahead from a peak candidate to
    #               determine if it is the actual peak (default: 200)
    #               '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    # delta.........(optional) this specifies a minimum difference between a peak and
    #               the following points, before a peak may be considered a peak. Useful
    #               to hinder the function from picking up false peaks towards to end of
    #               the signal. To work well delta should be set to delta >= RMSnoise *
    #               5. (default: 0)
    # delta.........function causes a 20% decrease in speed, when omitted. Correctly used
    #               it can double the speed of the function
    # return........two lists [max_peaks, min_peaks] containing the positive and negative
    #               peaks respectively. Each cell of the lists contains a tupple of:
    #               (position, peak_value) to get the average peak value do:
    #               np.mean(max_peaks, 0)[1] on the results to unpack one of the lists
    #               into x, y coordinates do: x, y = zip(*tab)
    #
    
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
    
    # check input data
    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
    
    return [max_peaks, min_peaks]

def _datacheck_peakdetect(x_axis, y_axis):
    """Used as part of "peakdetect" """
    if x_axis is None:
        x_axis = range(len(y_axis))
    
    if len(y_axis) != len(x_axis):
        raise ValueError("Input vectors y_axis and x_axis must have same length")

    #needs to be a numpy array
    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis
    
