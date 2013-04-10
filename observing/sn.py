import pylab as py
import numpy as np
import scipy as sp
import pyfits as pf

import asciitable as tab

from scipy.stats import stats
from scipy.interpolate import griddata

import sami.utils as utils
import sami.samifitting as fitting

"""
This file contains some S/N calculations used predominantly during SAMI observing runs.
"""

def sn_list(inlist, tablein, l1, l2, ifus='all'):

    #To print only two decimal places in all numpy arrays
    np.set_printoptions(precision=2)

    files=[]

    for line in open(inlist):
        cols=line.split(' ')
        cols[0]=str.strip(cols[0])
        
        files.append(np.str(cols[0]))

    print "I have received", len(files), "files for which to calculate and combine S/N measurements."

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
    
def sn(insami, tablein, l1, l2, plot=False, ifus='all', log='True', verbose=True):

    if verbose == True: 
        print ''
        print '-----------------------------------'
        print 'Process: SAMI_utils. Function: SNR.'
        print '-----------------------------------'
        print ''

    # Read FITS file
    list1  = pf.open(insami)
    data   = list1[0].data
    var    = list1[1].data
    hdr    = list1[0].header

    # Read target table ## TESTER: "sami_sel_120807.out"
    tabname = ['name', 'ra', 'dec', 'r_petro', 'z', 'M_r', 'Re', 
               'med_mu_Re', 'mu_Re', 'mu_2Re', 'isel', 'A_g', 'group']
    tabin = tab.read(tablein, names=tabname)

    # Field of view centre
    mra  = hdr['MEANRA']
    mdec = hdr['MEANDEC']
    #print mra, mdec

    # Wavelength range
    x = np.arange(2048)

    # Wavelength range
    Lc  = hdr['CRVAL1']
    pix = hdr['CRPIX1']
    dL  = hdr['CDELT1']

    L0 = Lc - pix*dL

    L = L0 + x*dL

    # Wavelength range for the cube
    idx1 = SAMI_utils_V.find_nearest(L,l1)
    idx2 = SAMI_utils_V.find_nearest(L,l2)

    # Define the list of IFUs to display
    if ifus == 'all':
        IFUlist = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        IFUlist = [ifus]
    #data_sum=np.nansum

    # Number of IFUs to display
    n = len(IFUlist)

    # Diagnostic screen output
    if verbose == True: 
        if n == 1: print '-- Processing', n, 'IFU. Plotting is', 
        if n > 1:  print '-- Processing', n, 'IFUs. Plotting is', 
        if plot == False: print 'OFF.'
        if plot == True:  print 'ON.'

    # --ISK: some variables
    ncore = 61                       # number of cores per bundle
    rcore = 1.6                      # core diameter in asec
    # ------
    
    # To create the even grid to display the cubes on 
    #  (accurate to 1/10th core diameter)
    dx = 4.44e-5 /np.cos(np.pi *mdec /180)
    dy = 4.44e-5
    #print dx, dy
    
    # Create the figure
    if plot == True: fig = py.figure()

    #LF - Adding an array to contian all SN values for each bundle
    SN_all=np.empty((n))
    
    # Looping over all IFUs: 
    for i in range(n):

        # --ISK: Initialise a couple of arrays for this loop
        dc      = np.zeros(ncore)
        good_dc = np.zeros(ncore)
        t_ra  = 0.
        t_dec = 0.

        x,y,data,var,num,fib_type,P,\
            name=SAMI_utils_V.IFU_pick(insami, IFUlist[i])

        # --ISK: Also read fibre table
        fibtab = list1[2].data
        n_fibtab = len(fibtab)           # N_lines in fibre table
        
        # first isolate lines pertaining to each bundle
        range_ = np.where(np.logical_and(
                fibtab['PROBENUM'] == IFUlist[i],
                fibtab['TYPE'] == "P"
                ))

        target = fibtab['NAME'][range_][0]       # target ID

        c_ra  = fibtab['FIB_MRA'][range_]        # RA, Dec of each core
        c_dec = fibtab['FIB_MDEC'][range_]
        
        match = tabin.name == target             # get Table index
        
        # *** t_radec should be replaced with fib_radec of  brightest core
        data_ = list1[0].data[range_]
        # print np.nanmax(data_)
        # centroid = np.where(data_ == np.nanmax(data_))
        # print 'CENTROID FLUX = ', data_[centroid]
        # t_ra  = fibtab['FIB_MRA'][centroid]
        # t_dec = fibtab['FIB_MDEC'][centroid]
        print 'DIAG: ', np.shape(fibtab), np.shape(data_)

        # t_ra = tabin.ra[match]                   # get RA, Dec of target
        # t_dec = tabin.dec[match]

        re_ = tabin.Re[match]                    # Re of target
        # -------

        if verbose == True: 
            print ' '
            print '-- IFU =', IFUlist[i]
            print '  > spectral range = [', l1, ',', l2, ']'
            print '  > in rest frame  = [', np.round(l1/(1+tabin.z[match])),',',
            print np.round(l2/(1+tabin.z[match])), ']'
            print ''

        sn_spec = data/np.sqrt(var)

        # Sum up the data
        sum = np.nansum(data[:, idx1:idx2], axis=1)
        med = stats.nanmedian(data[:, idx1:idx2], axis=1)

        # Median SN over lambda range
        sn = stats.nanmedian(sn_spec[:, idx1:idx2], axis=1)

        centroid = np.where(sn == np.nanmax(sn))
        print 'S/N @CENTROID', sn[centroid]

        t_ra  = c_ra[centroid]
        t_dec = c_dec[centroid]
        print ''

        # Sum the variances
        # varsum=np.nansum(var[:,idx1:idx2],axis=1)
        # sn=sum/np.sqrt(varsum)

        # -- ISK: identify cores at approximately Re: 
        # ------------------------------------------- 
        for k in range(ncore):
            dc[k] = 3600.* np.math.sqrt( (c_ra[k] - t_ra)**2 + 
                                         (c_dec[k] - t_dec)**2 )
            
            # Mark cores intersected by Re: 
            if dc[k] >  re_ - 0.5*rcore and dc[k] <  re_ + 0.5*rcore: 
                good_dc[k] = 1.
            
        # Get median S/N of cores @Re: 
        sn_Re = stats.nanmedian(sn[good_dc == True])
        SN_all[i]=sn_Re
        
        if verbose == True: 
            print '=> Min, Max, Median S/N @Re = ',
            print '%0.2f' % min(sn[good_dc == True]), ',',
            print '%0.2f' % max(sn[good_dc == True]), ',',
            print '%0.2f' % sn_Re

        # -- ISK: starting the plot loop here: 
        # ------------------------------------
        if plot == True: 

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
                
            # ISK: trying to improve the rows and columns a bit: 
            # def isodd(num): return num & 1 and True or False
            # if n <= 3:
            #     r = 1
            #     c = n
            # elif n > 6: 
            #     r = 3
            #     c = 3
                
            # Size of image - chosen to fit bundle.
            size = 100
            N = np.arange(size)
    
            # So now, linear grid
            x_c = x[np.sum(np.where(num == 1))]
            y_c = y[np.sum(np.where(num == 1))]
        
            x_0 = x_c + (size/2)*dx #Highest RA number (leftmost point)
            y_0 = y_c - (size/2)*dy #Lowest Dec Number (bottom point)

            x_lin = x_0-N*dx #RA goes +ve to -ve
            y_lin = y_0+N*dy #Dec goes -ve to +ve
            
            # Create image
            # Find indices of nearest linear points to actual core positions.
            b = 0 #reset index
            core_x = []
            core_y = []

            for b in range(int(len(x))):
                
                nx = SAMI_utils_V.find_nearest(x_lin, x[b])
                ny = SAMI_utils_V.find_nearest(y_lin, y[b])
                # print nx
                core_x.append(nx)
                core_y.append(ny)
                
            print ' '
            print "-- Displaying", IFUlist[i]

            # Make empty image.
            frame = np.empty((size,size)) + np.nan
            ax = fig.add_subplot(r, c, i+1)

            # Loop through all cores: 
            a = 0 #reset index
            for a in range(int(len(core_x))):
                # Find indices of points in appropriate Bresenham circles
                # Note 5 is chosen as the radius as 1 pixel=1/10th spaxel

                circle_x, \
                circle_y=SAMI_utils_V.bresenham_circle(core_x[a],core_y[a],5)
                
                frame[circle_x, circle_y] = sn[a]   #sum[a]

                # mark cores intersected by Re
                if good_dc[a] == 1: 
                    ax.plot(core_x[a], core_y[a], 'bo', ms=8, lw=3, mfc=None)
                    
            ax = fig.add_subplot(r, c, i+1)
            im = ax.imshow(np.transpose(frame), origin='lower', 
                           interpolation='nearest', cmap='gray')

            ax.set_title(IFUlist[i])
            fig.colorbar(im)
            
            # Write images
            # -- ISK: clobbering to overwrite existing file of same name
            outsnfile='sn_'+np.str(l1)+'_'+np.str(l2)+'_'+\
                np.str(IFUlist[i])+'_'+insami
            pf.writeto(outsnfile, np.transpose(frame), clobber=True)
            
        # Super title for plot
        py.suptitle(insami+'(S/N)')

    #Close the open hdu list
    list1.close()

    if verbose == True: 
        print ''
        print '-----------------------------------'

    return SN_all
