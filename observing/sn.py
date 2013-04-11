import pylab as py
import numpy as np
import scipy as sp
import pyfits as pf

import asciitable as tab

from scipy.stats import stats
from scipy.interpolate import griddata

from matplotlib.patches import Circle

import sami.utils as utils
import sami.samifitting as fitting

import sys

"""
This file contains a S/N estimation code used predominantly during SAMI observing runs.

UPDATED: 8/4/13, Iraklis Konstantopoulos
         -- editing to comply with new conventions in sami_utils; 
         -- edited to accept new target table format; 

NOTES: 10/4/13, Iraklis Konstantopoulos
       -- I no longer return SN_all, but sn_Re, the median SN @Re. 
       -- I removed the SN_all array from the sn function. 

"""

def sn_list(inlist, tablein, l1, l2, ifus='all'):
    """ 
    Wrapper function to provide S/N estimates for >1 file 
    
    inlist   [ascii] list of files (format?)
    tablein  [ascii] 
    """

    #To print only two decimal places in all numpy arrays
    np.set_printoptions(precision=2)

    files=[]

    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])
        
        files.append(np.str(cols[0]))

    print "I have received", len(files), \
        "files for which to calculate and combine S/N measurements."

    # Define the list of IFUs to display
    if ifus == 'all':
        IFUlist = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        IFUlist = [ifus]

    print "I will calculate S/N for", len(IFUlist), "IFUs."

    SN_all_sq=np.empty((len(IFUlist), len(files)))

    for i in range(len(files)):

        insami=files[i]
        SN_all=sn(insami, tablein, plot=False, ifus=ifus, verbose=False)
        
        SN_all_sq[:,i]=SN_all*SN_all

    # Add the squared SN values and square root them
    SN_tot=np.sqrt(np.sum(SN_all_sq, axis=1))

    print IFUlist
    print SN_tot
    
def sn(insami, tablein, l1, l2, plot=False, ifus='all', 
       log=True, verbose=True, output=False, seek_centroid=True):

    """ 
    Purpose: Main function, estimates S/N for any or all probes in an RSS file. 

    Input variables:

     insami  [fits]  Input RSS file. 
     tablein [ascii] Observations table. 
     l1, l2  [flt]   Wavelength range for S/N estimation. 
     ifus    [str]   Probe number, or 'all' for all 13. 
     log     [bool]  Logarithimic scaling for plot -- CURRENTLY NOT ENVOKED. 
     verbose [bool]  Toggles diagnostic verbosity. 

    Process: 
     1) Interpret input. 
       [Set up plot]
     2) Read target table (new format for SAMI survey),  
       [Commence all-IFU loop, read data]
     3) Identify wavelength range over which to estimate SNR, 
     4) Calculate SNR for all cores in the RSS file. 
     5) Locate galaxy centre as peak SNR core. 
     6) Identify cores intercepted by Re (listed). 
     7) Get SNR @Re as median of collapsed wavelength region. 
       [End all-IFU loop]
    """

    # --------------------
    # (1) Interpret input
    # --------------------
    if ifus == 'all':
        IFUlist = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        IFUlist = ifu_num = [int(ifus)]

    n_IFU = len(IFUlist)

    if verbose: 
        print('')
        print('-----------------------------------')
        print('Process: SAMI_utils. Function: SNR.')
        print('-----------------------------------')
        print('')
        if n_IFU == 1: print 'Processing', n_IFU, 'IFU. Plotting is', 
        if n_IFU > 1:  print 'Processing', n_IFU, 'IFUs. Plotting is', 
        if not plot: print 'OFF.'
        if plot:  print 'ON.'
        print('')

    # --------------------
    # Set up plot process
    # --------------------
    
    # Define number of cores, core diameter (in arcsec). 
    # -- is this stored someplace in sami.utils/generic? 
    n_core = 61
    r_core = 1.6
    
    # Create the figure
    if plot: 

        # Get Field RA, DEC
        hdulist = pf.open(insami)
        primary_header = hdulist['PRIMARY'].header
        field_dec = primary_header['MEANDEC']

        # To create the even grid to display the cubes on 
        #  (accurate to 1/10th core diameter)
        dx = 4.44e-5 /np.cos(np.pi *field_dec /180.)
        dy = 4.44e-5
        
        fig = py.figure()
        # Number of rows and columns needed in the final display box
        # This is a bit of a fudge...
        if n_IFU==1:
            im_n_row = 1
            im_n_col = 1
        elif n_IFU==2:
            im_n_row = 1
            im_n_col = 2
        elif n_IFU==3:
            im_n_row = 1
            im_n_col = 3
        elif n_IFU==4:
            im_n_row = 2
            im_n_col = 2
        elif n_IFU>3 and n_IFU<=6:
            im_n_row = 2
            im_n_col = 3
        elif n_IFU>6 and n_IFU<=9:
            im_n_row = 3
            im_n_col = 3
        
        # ISK: trying to improve the rows and columns a bit: 
        # def isodd(num): return num & 1 and True or False
        # if n <= 3:
        #     r = 1
        #     c = n
        # elif n > 6: 
        #     r = 3
        #     c = 3
        
    # ----------------------
    # (2) Read target table
    # ----------------------
    tabname = ['name', 'ra', 'dec', 'r_petro', 'r_auto', 'z', 'M_r', 
               'Re', '<mu_Re>', 'mu(Re)', 'mu(2Re)', 'M*', 'g-i', 'A_g', 
               'CATID', 'SURV_SAMI', 'PRI_SAMI', 'BAD_CLASS']
    target_table = tab.read(tablein, names=tabname, data_start=0)

    # Start a little counter to keep track 
    # -- a fudge for the way the plot loop is set up... 
    counter = 0

    # --------------------------
    # Commence the all-IFU loop
    # --------------------------
    for ifu_num in IFUlist:

        counter = counter + 1

        # Read single IFU
        myIFU = utils.IFU(insami, ifu_num, flag_name=False)

        # ----------------------------
        # (3) Define wavelength range
        # ----------------------------
        z_target = target_table.z[target_table.CATID == myIFU.name]

        l_range = myIFU.lambda_range
        l_rest = l_range/(1+z_target)

        # identify array elements closest to l1, l2 **in rest frame**
        idx1 = (np.abs(l_rest - l1)).argmin()
        idx2 = (np.abs(l_rest - l2)).argmin()

        if verbose: 
            this_gal_z = target_table.z[target_table.name == myIFU.name]
            if n_IFU > 1: print('-- IFU #' + str(ifu_num))

            print('   Spectral range: ' + 
                  str(np.around([l_rest[idx1], l_rest[idx2]])))
            print('   Observed at:    ' + 
                  str(np.around([l_range[idx1], l_range[idx2]])))

            print('')
        
        # -------------------------
        # (4) Get SNR of all cores
        # -------------------------
        sn_spec = myIFU.data/np.sqrt(myIFU.var)
        
        # Sum up the data
        #sum = np.nansum(myIFU.data[:, idx1:idx2], axis=1)
        #med = stats.nanmedian(myIFU.data[:, idx1:idx2], axis=1)
        
        # Median SN over lambda range (per Angstrom)
        sn = stats.nanmedian(sn_spec[:, idx1:idx2], axis=1) * (1./myIFU.cdelt1)

        # ----------------------------------
        # (5) Find galaxy centre (peak SNR)
        # ----------------------------------
        # Initialise a couple of arrays for this loop
        core_distance = np.zeros(n_core)
        good_core     = np.zeros(n_core)
        centroid_ra  = 0.
        centroid_dec = 0.
        
        # Get target Re from table (i.e., match entry by name)
        re_target = target_table.Re[target_table.CATID == int(myIFU.name)]
        # if Re is not listed (i.e., Re = -99.99), then quote centroid SNR. 
        if re_target == -99.99: 
            print("*** No Re listed, calculating at centroid instead.")

        # Get either centroid, or table RA, DEC
        if seek_centroid: 
            centroid = np.where(sn == np.nanmax(sn))
            centroid_ra  = myIFU.xpos[centroid]
            centroid_dec = myIFU.ypos[centroid]

        if not seek_centroid: 
            centroid_ra = target_table.ra[target_table.CATID == int(myIFU.name)]
            centroid_dec=target_table.dec[target_table.CATID == int(myIFU.name)]

            test_distance = 3600.* np.sqrt(
                (myIFU.xpos - centroid_ra)**2 +
                (myIFU.ypos - centroid_dec)**2 )
            centroid = np.abs(test_distance - 0).argmin()

        if verbose: 
            print '   S/N @Centroid =', np.round(sn[centroid]), '[/Angstrom]'
            print ''

        # ---------------------------------------- 
        # (6) Identify cores at approximately Re
        # ---------------------------------------- 

        core_distance = 3600.* np.sqrt(
            (myIFU.xpos - centroid_ra)**2 +
            (myIFU.ypos - centroid_dec)**2 )

        good_core[(core_distance > re_target - 0.5*r_core) 
                  & (core_distance < re_target + 0.5*r_core)] = True
        
        # Get median S/N of cores @Re: 
        sn_Re = stats.nanmedian(sn[good_core == True])        

        if verbose == True: 
            print '=> Min, Max, Median S/N @Re = ',
            print '%0.2f' % min(sn[good_core == True]), ',',
            print '%0.2f' % max(sn[good_core == True]), ',',
            print '%0.2f' % sn_Re, '[/Angstrom]'
            print('')
        
        # ----------
        # DRAW PLOT 
        # ----------
        if plot:
            # Set image size to fit the bundle.
            size_im = 100
            N_im = np.arange(size_im)
            
            # Create a linear grid, centred at Fibre #1.
            x_ctr = myIFU.xpos[np.sum(np.where(myIFU.n == 1))]
            y_ctr = myIFU.ypos[np.sum(np.where(myIFU.n == 1))]
        
            # Set axis origin: highest RA, lowest DEC.
            x_0 = x_ctr + (size_im/2)*dx
            y_0 = y_ctr - (size_im/2)*dy
        
            # Direction of each axis: RA decreases, DEC increases. 
            x_lin = x_0-N_im*dx
            y_lin = y_0+N_im*dy
        
            # Create image --
            # 1) Find indices of nearest linear points to actual core positions.
            b = 0        # (reset index)
            core_x = []
            core_y = []
            
            for b in range(n_core):
                
                nx = np.abs(x_lin - myIFU.xpos[b]).argmin()
                ny = np.abs(y_lin - myIFU.ypos[b]).argmin()
                
                core_x.append(nx)
                core_y.append(ny)
                
            if verbose: 
                print("Displaying IFU #" + str(ifu_num))
                print('')
            
            # Make empty image.
            frame = np.empty((size_im,size_im)) + np.nan
            ax = fig.add_subplot(im_n_row, im_n_col, counter)
            ax.set_aspect('equal')

            # Colorise all fibres according to S/N; negatives set to zero. 
            sn_norm = sn/np.nanmax(sn)
            sn_norm[sn < 0] = 0.0
            
            # Loop through all cores: 
            a = 0 #reset index
            for a in range(n_core):

                # Find indices of points in appropriate Bresenham circles
                # Note 5 is chosen as the radius as 1 pixel=1/10th spaxel

                # *** NEED to replace this with circle patches. 

                # Make a Circle patch for each fibre in the bundle: 
                art_core = Circle(xy = (core_x[a], core_y[a]), 
                                  radius=4.8, color=str(sn_norm[a]))
                ax.add_artist(art_core)

                # and mark cores intersected by Re: 
                if good_core[a]: 
                    art_good = Circle(xy = (core_x[a], core_y[a]), 
                                  radius=4.8, alpha=0.7)
                    ax.add_artist(art_good)

                frame[core_x[a], core_y[a]] = sn[a]   #sum[a]
                
                """
                circle_x, \
                circle_y=SAMI_utils_V.bresenham_circle(core_x[a],core_y[a],5)

                frame[circle_x, circle_y] = sn[a]   #sum[a]
                
                # mark cores intersected by Re
                if good_core[a]: 
                    ax.plot(core_x[a], core_y[a], 'bo', ms=20, lw=3, mfc=None)
                """
                    
            ax = fig.add_subplot(im_n_row, im_n_col, counter)
            im = ax.imshow(np.transpose(frame), origin='lower', 
                           interpolation='nearest', cmap='gray')
            
            ax.set_title('Probe #'+str(ifu_num))
            fig.colorbar(im)

            # Write images
            if output: 
                outsnfile='sn_'+np.str(l1)+'_'+np.str(l2)+'_'+\
                    str(ifu_num)+'_'+insami
                pf.writeto(outsnfile, np.transpose(frame), clobber=True)
            
        # Super title for plot
        py.suptitle(insami+', S/N map')

    if verbose: 
        print ''
        print '-----------------------------------'

    #return SN_all
    return('Median S/N @Re = '+str(np.round(sn_Re, decimals=1)))
