"""
This file contains a couple of S/N estimation codes 
 designed for use during SAMI observing runs.

UPDATED: 08.04.2013, Iraklis Konstantopoulos
         - Edited to comply with new conventions in sami_utils. 
         - Edited to accept new target table format. 

         23.08.2012, Iraklis Konstantopoulos
         - Changed name of "sn" function to "sn_re". 
         - Writing new S/N code based on the secondary star observation. 

NOTES: 10.04.2013, Iraklis Konstantopoulos
       - I no longer return SN_all, but sn_Re, the median SN @Re. 
       - Removed the SN_all array from the sn function. 

       26.08.2013, Iraklis Konstantopoulos
       - Updated fields for the SAMI target table. 
       - Also changed all mentions of 'z' to 'zpec'. 
       - Major bug fixes in case where target not found on target table.  

       27.08.2013, Iraklis Konstantopoulos
       - Writing surface brightness map function. 

For reasons I (JTA) don't remember, this code was never quite finished
or put into action. The intention had been to use S/N measurements to aid
the observers in deciding when a field was finished, but this code is not
mentioned in the observers' instructions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals


import pylab as py
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# use astropy for all astronomy related things.
import astropy.io.fits as pf
import astropy.io.ascii as tab

import sys

from matplotlib.patches import Circle

# Relative imports from sami package
from .. import utils
from .. import samifitting as fitting


def sn_map(rssin):
    """ 
    Plot SNR of all 12 SAMI targets across fraction of Re. 
    
    Process: 
    - Deduce the noise level from the standard star: 
     + obtain listed brightness, 
     + use existing 2D Gauss function to get SBP,
     + (photometric aperture and aperture correction?),  
     + normalise flux, 
     + calculate integrated S/N for star, 
     + establish noise level.  

    - Run the SDSS-SB fuction on all targets, 
     + Convert brightness to S/N, 
     + Plot all 12 targets:
      - x-axis: fraction of Re (from target selection table), 
      - y-axis: S/N, 
      - horizontal lines @S/N=5, 10.
    """

    print("HAY!")

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

    print("I have received", len(files), \
        "files for which to calculate and combine S/N measurements.")

    # Define the list of IFUs to display
    if ifus == 'all':
        IFUlist = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        IFUlist = [ifus]

    print("I will calculate S/N for", len(IFUlist), "IFUs.")

    SN_all_sq=np.empty((len(IFUlist), len(files)))

    for i in range(len(files)):

        insami=files[i]
        SN_all=sn_re(insami, tablein, plot=False, ifus=ifus, verbose=False)
        
        SN_all_sq[:,i]=SN_all*SN_all

    # Add the squared SN values and square root them
    SN_tot=np.sqrt(np.sum(SN_all_sq, axis=1))

    print(IFUlist)
    print(SN_tot)
    
def sn_re(insami, tablein, l1, l2, plot=False, ifus='all', 
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
        print('--------------------------------')
        print('Running sami.observing.sn.sn_re.')
        print('--------------------------------')
        print('')
        if n_IFU == 1: print('Processing', n_IFU, 'IFU. Plotting is', end=' ') 
        if n_IFU > 1:  print('Processing', n_IFU, 'IFUs. Plotting is', end=' ') 
        if not plot: print('OFF.')
        if plot:  print('ON.')
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
        elif n_IFU>9 and n_IFU<=12:
            im_n_row = 3
            im_n_col = 4
        elif n_IFU>12:
            im_n_row = 4
            im_n_col = 4
        
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
    tabname = ['name', 'ra', 'dec', 'r_petro', 'r_auto', 'z_tonry', 'zspec', 
               'M_r', 'Re', '<mu_Re>', 'mu(Re)', 'mu(2Re)', 'ellip', 'PA', 'M*',
               'g-i', 'A_g', 'CATID', 'SURV_SAMI', 'PRI_SAMI', 'BAD_CLASS']
    target_table = tab.read(tablein, names=tabname, data_start=0)
    CATID = target_table['CATID'].tolist()

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

        # And find the row index for this SAMI target. 
        try: 
            this_galaxy = CATID.index(int(myIFU.name))
            no_such_galaxy = False
        except:
            this_galaxy = []
            no_such_galaxy = True
            pass

        """
        There are other ways to do this with a numpy array as input. 
        Lists are far better at this, so have made a CATID list. 
        
        this_galaxy = np.where(target_table['CATID'] == int(myIFU.name))
        this_galaxy = np.where(CATID == int(myIFU.name))
        this_galaxy = [CATID == int(myIFU.name)]
        """

        # ----------------------------
        # (3) Define wavelength range
        # ----------------------------

        if no_such_galaxy:
            z_target = 0.0
            z_string = '0.0'
            
            # see below for explanation of this. 
            idx1 = l1
            idx2 = l2

            print(('-- IFU #' + str(ifu_num)))
            print("   This galaxy was not found in the Target Table. ")

        else: 
            z_target = target_table['zspec'][this_galaxy]
            z_string = str(z_target)

            l_range = myIFU.lambda_range
            l_rest = l_range/(1+z_target)
            
            # identify array elements closest to l1, l2 **in rest frame**
            idx1 = (np.abs(l_rest - l1)).argmin()
            idx2 = (np.abs(l_rest - l2)).argmin()
            
            if verbose: 
                print('-------------------------------------------------------')
                print((' IFU #' + str(ifu_num)))
                print('-------------------------------------------------------')
                print(('   Redshift:       ' + z_string))
                print(('   Spectral range: ' + 
                      str(np.around([l_rest[idx1], l_rest[idx2]]))))
                
                print(('   Observed at:    ' + 
                      str(np.around([l_range[idx1], l_range[idx2]]))))
                print('')
        
        # -------------------------
        # (4) Get SNR of all cores
        # -------------------------
        sn_spec = myIFU.data/np.sqrt(myIFU.var)
        
        # Median SN over lambda range (per Angstrom)
        sn = np.nanmedian(sn_spec[:, idx1:idx2], axis=1) * (1./myIFU.cdelt1)
        
        # ----------------------------------
        # (5) Find galaxy centre (peak SNR)
        # ----------------------------------
        # Initialise a couple of arrays for this loop
        core_distance = np.zeros(n_core)
        good_core     = np.zeros(n_core)
        centroid_ra  = 0.
        centroid_dec = 0.
        
        # Get target Re from table (i.e., match entry by name)
        if no_such_galaxy:
            print("   No Re listed, calculating SNR at centroid instead.")
            re_target = 0.

        else:
            re_target = target_table['Re'][this_galaxy]
            
        # Get either centroid, or table RA, DEC
        if seek_centroid: 
            if no_such_galaxy:
                centroid = np.where(myIFU.n ==1)
            else:
                centroid = np.where(sn == np.nanmax(sn))
                centroid_ra  = myIFU.xpos[centroid]
                centroid_dec = myIFU.ypos[centroid]

        if not seek_centroid: 
            if no_such_galaxy:
                centroid = np.where(myIFU.n ==1)
            else:
                centroid_ra = target_table['ra'][this_galaxy]
                centroid_dec = target_table['dec'][this_galaxy]
                
                test_distance = 3600.* np.sqrt(
                    (myIFU.xpos - centroid_ra)**2 +
                    (myIFU.ypos - centroid_dec)**2 )
                centroid = np.abs(test_distance - 0).argmin()
                
        if verbose: 
            print('   S/N @Centroid =', np.round(sn[centroid]), '[/Angstrom]')
            print('')

        # ---------------------------------------- 
        # (6) Identify cores at approximately Re
        # ---------------------------------------- 

        # Check that there is an Re listed, some times there isn't. 
        if no_such_galaxy:
            sn_Re = 0.
        else:
            core_distance = 3600.* np.sqrt(
                (myIFU.xpos - centroid_ra)**2 +
                (myIFU.ypos - centroid_dec)**2 )
            
            good_core[(core_distance > re_target - 0.5*r_core) 
                      & (core_distance < re_target + 0.5*r_core)] = True
            
            # Get median S/N of cores @Re: 
            if 1 in good_core:
                sn_Re = np.nanmedian(sn[good_core == True])        
                sn_min = min(sn[good_core == True])
                sn_max = max(sn[good_core == True])
                
            if verbose: 
                if not 1 in good_core:
                    sn_str = str(np.round(np.nanmedian(sn)))
                    print("** Could not match Re")
                    print(('=> Median overall S/N = '+sn_str))
                    print('')

                else:
                    print('=> [Min, Max, Median] S/N @Re = [', end=' ')
                    print('%0.2f' % min(sn[good_core == True]), ',', end=' ')
                    print('%0.2f' % max(sn[good_core == True]), ',', end=' ')
                    print('%0.2f' % sn_Re, '] [/Angstrom]')
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

                # Make a Circle patch for each fibre in the bundle: 
                art_core = Circle(xy = (core_x[a], core_y[a]), 
                                  radius=4.8, color=str(sn_norm[a]))
                ax.add_artist(art_core)

                # and mark cores intersected by Re: 
                if good_core[a]: 
                    art_good = Circle(xy = (core_x[a], core_y[a]), 
                                  radius=4.8, alpha=0.7)
                    ax.add_artist(art_good)

                frame[core_x[a], core_y[a]] = sn[a]
                
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
        print('-------------------------------------------------------')

 
def read_targ_tab(tablein):
    """ Read a SAMI target table. """
    tabname = ['name', 'ra', 'dec', 'r_petro', 'r_auto', 'z_tonry', 'zspec', 
               'M_r', 'Re', '<mu_Re>', 'mu(Re)', 'mu(2Re)', 'ellip', 'PA', 'M*',
               'g-i', 'A_g', 'CATID', 'SURV_SAMI', 'PRI_SAMI', 'BAD_CLASS']
    target_table = tab.read(tablein, names=tabname, data_start=0)
    return target_table


def sb(rssin, tablein, starin, ifus='all', 
       starIDcol=0, starMAGcol=[5,6], area='fibre'):
    """ Make surface brightness maps of all IFUs in rssin, indicate SNR. """

    from scipy.interpolate import griddata

    """ 
    Use the secondary star to deduce zeropoint. 
    Then translate flux to surface brightness. 

    This should make use of the Gauss-fit code to fit the SBP of the star. 
    For now I am just keeping the thing simple. 

    1) Identify secondary star. Should be only target not on 'tablein'. 
    2) Measure flux (for now of the whole probe). 
    3) Look up brightness of star on star table. 
    4) Deduce zeropoint. 
    5) Map SB of targets in all other probes. 

    The 'area' input corresponds to the area over which the surface brightness
    is inter/extrapolated. The default is to measure per SAMI fibre, but it is
    possible to provide any area (e.g., per sq/ arcsec). 
    """
    
    # ---------------------------
    # (1) Identify secondary star 
    # ---------------------------

    # First of all, read the colour of the spectrum in the primary header. 
    myHDU = pf.open(rssin)
    colour = myHDU[0].header['SPECTID']
    myHDU.close()

    # Interpret input
    if ifus == 'all':
        IFUlist = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    else:
        IFUlist = ifu_num = [int(ifus)]

    n_IFU = len(IFUlist)

    # Read star table
    star_table = tab.read(starin, header_start=0, data_start=1)
    RowID = star_table['RowID'].tolist()
    
    # Read SDSS throughputs
    sdss_col = ['wave', 'pt_secz=1.3', 'ext_secz=1.3', 
                'ext_secz=0.0', 'extinction']
    sdss_g = tab.read('SDSS_g.dat', quotechar="#", names=sdss_col)
    sdss_r = tab.read('SDSS_r.dat', quotechar="#", names=sdss_col)

    # Cycle through probes, identify star through CATID//RowID. 
    found_star = False
    for ifu_num in IFUlist: 
        
        # Read single IFU
        myIFU = utils.IFU(rssin, ifu_num, flag_name=False)
        nfib = np.shape(myIFU.data)[0]
        
        if int(myIFU.name) in RowID:
            found_star = True
            star = ifu_num
            print(("Star found in Probe #"+str(star)))

            # ----------------
            # (2) Measure flux
            # ----------------
            """
            This needs to take into account the flux in limited spectral and 
            spatial ranges. The spectral is taken care of (convolving with 
            SDSS filter throughput), but the spatial is not. Should use the 
            Gauss fit function and integrate, currently summing up all fibres.
            """
            wave = myIFU.lambda_range
            if colour == 'RD':
                thru_regrid = griddata(sdss_r['wave'], sdss_r['ext_secz=1.3'], 
                                       wave, method='cubic', fill_value=0.0)
            else:
                thru_regrid = griddata(sdss_g['wave'], sdss_g['ext_secz=1.3'], 
                                       wave, method='cubic', fill_value=0.0)

            # Convolve flux and sum in a per-core basis.
            conv_fib = np.zeros(len(myIFU.data))
            for fib in range(nfib):
                conv_fib[fib] = np.nansum(myIFU.data[fib]*thru_regrid)
                
            """ 
            Blue spectrum overlaps well with g' band, but r' does not, need 
            extrapolate a flux according to the fixed F-type star spec-slope. 
            The slope is straight, so a triangle approximation is alright. My 
            model is this F-star:
            
              http://www.sdss.org/dr5/algorithms/spectemplates/spDR2-007.gif
            
            which I approximate to a right-angle triangle. The opposing and 
            adjacent sides of the full (entire r' band) and curtailed (SAMI)
            triangles are [50, 1800] and [30, 1000], in units of [flux, Ang].  

            The relative areas are therefore differ by a factor of three and 
            the extrapolated flux contained in the area of overlap between 
            the SDSS r' and the SAMI red spectrum is 3. 
            """ 
            if colour == 'RD':
                flux = 3* np.nansum(conv_fib)
            else: 
                flux = np.nansum(conv_fib)

            print(("S(Flux) = "+str(np.round(flux))+" cts"))

            """ 
            Finally, need to check if the user is looking for a flux inter/
            extrapolated to an area different to that of the SAMI fibre. 
            pi * (0.8")**2 ~= 2.01 sq. asec.
            """ 
            if area != 'fibre':
                flux = flux * (np.pi*0.8**2)/area

            # -------------------------
            # (3) Get listed brightness
            # -------------------------

            # Get g (blue) or r (red) mag from stars catalogue.

            # ID is column zero, unless otherwise set by starIDcol, 
            # and g, r are 5, 6, unless set otherwise in starMAGcol.
            this_star = RowID.index(int(myIFU.name))

            if colour == 'RD':
                mag = star_table['r'][this_star]
            else:
                mag = star_table['g'][this_star]
            print(("[ID, brightness] = ", RowID[this_star], mag))
            
    # --------------------
    # (4) Deduce zeropoint
    # --------------------
    # Red zeropoint tricky, as not entire r' is covered. Secondary stars are 
    # F-class, so can assume a spectral slope. Going with flat, roughly OK. 

    if colour == 'RD':
        # SAMI spectra roughly run from 6250 to 7450 A. 
        # The SDSS r' band throughput between 5400 and 7230 A. 
    
        zmag = mag + 2.5 * np.log10(flux)
        print(("Calculated zeropoint as "+str(np.round(zmag,decimals=2))+" mag."))
            
    # -------------------------
    # (5) Map SB of all targets
    # -------------------------

    # Set up plot
    fig = plt.gcf()
    fig.clf()

    # Cycle through all IFUs. 
    for ifu_num in IFUlist: 
        
        if ifu_num != star: 
            myIFU = utils.IFU(rssin, ifu_num, flag_name=False)
            s_flux = np.zeros(nfib)

            # and some plotty things
            fibtab = myIFU.fibtab
            offset_ra  = np.zeros(nfib, dtype='double')
            offset_dec = np.zeros(nfib, dtype='double')

            # And loop through all fibres to get summed flux
            for fibnum in range(nfib):
                s_flux[fibnum] = np.nansum(myIFU.data[fibnum][:])

                # do some fibre positions while you're looping

                """
                Adapting the plotting method from the BDF creation code. 
                Not sure if this is the best. Check Lisa's display code. 
                Should do it that way. 
                """
                
                # Get RAs and DECs of all fibres. 
                ra1    = np.radians(myIFU.xpos[np.where(myIFU.n == 1)])
                dec1   = np.radians(myIFU.ypos[np.where(myIFU.n == 1)])
                ra_fib  = np.radians(myIFU.xpos[fibnum])
                dec_fib = np.radians(myIFU.ypos[fibnum])
                
                # Angular distance
                cosA = np.cos(np.pi/2-dec1) * np.cos(np.pi/2-dec_fib) + \
                       np.sin(np.pi/2-dec1) * np.sin(np.pi/2-dec_fib) * \
                       np.cos(ra1-ra_fib) 

                # DEC offset
                cos_dRA  = np.cos(np.pi/2-dec1) * np.cos(np.pi/2-dec1) + \
                           np.sin(np.pi/2-dec1) * np.sin(np.pi/2-dec1) * \
                           np.cos(ra1-ra_fib) 

                # RA offset
                cos_dDEC = np.cos(np.pi/2-dec1) * np.cos(np.pi/2-dec_fib) + \
                           np.sin(np.pi/2-dec1) * np.sin(np.pi/2-dec_fib) * \
                           np.cos(ra1-ra1) 

                # Sign check; trig collapses everything to a single quadrant
                if (ra_fib >= ra1) and (dec_fib >= dec1):  # 1. quadrant (+, +)
                    offset_ra[fibnum]  = np.degrees(np.arccos(cos_dRA[0]))
                    offset_dec[fibnum] = np.degrees(np.arccos(cos_dDEC[0]))

                if (ra_fib <= ra1) and (dec_fib >= dec1):  # 2. quadrant (-, +)
                    offset_ra[fibnum] = \
                                np.negative(np.degrees(np.arccos(cos_dRA[0])))
                    offset_dec[fibnum] = np.degrees(np.arccos(cos_dDEC[0]))

                if (ra_fib <= ra1) and (dec_fib <= dec1):  # 3. quadrant (-, -)
                    offset_ra[fibnum] = \
                                np.negative(np.degrees(np.arccos(cos_dRA[0])))
                    offset_dec[fibnum] = \
                                np.negative(np.degrees(np.arccos(cos_dDEC[0])))

                if (ra_fib >= ra1) and (dec_fib <= dec1):  # 4. quadrant (+, -)
                    offset_ra[fibnum]  = np.degrees(np.arccos(cos_dRA[0]))
                    offset_dec[fibnum] = \
                                np.negative(np.degrees(np.arccos(cos_dDEC[0])))

            # Write a dictionary of relative RA, DEC lists
            datatab = {'RA': offset_ra, 
                    'DEC': offset_dec} # proper, spherical trig, sky-projected

            # And finally get that surface brightness
            sb = zmag - 2.5 * np.log10(s_flux)
            
            # -------------------------
            # PLOT
            # -------------------------

            ax = fig.add_subplot(4,4,ifu_num)
            ax.set_aspect('equal')
            ax.set_xlim(-0.0022, 0.0022)
            ax.set_ylim(-0.0022, 0.0022)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            plt.title("Probe #"+str(ifu_num))

            # Normalise sb array for plot colouring
            norm = sb-min(sb)
            sb_norm = norm/max(norm)

            # Make a colorbar that maintains scale
            mappable = plt.cm.ScalarMappable(cmap='gray')
            mappable.set_array(sb)
            plt.colorbar(mappable)

            for i in range(nfib):
                this_col = str(sb_norm[i])
                circ = Circle((datatab['RA'][i], 
                               datatab['DEC'][i]), 0.8/3600.,
                              edgecolor='none', facecolor=this_col)
                ax.add_patch(circ)
            plt.show()



    # Report if no star was identified in the supplied RSS file or probe. 
    if not found_star:
        if ifus=='all':
            print(("Did not find a secondary star in RSS file '"+rssin+"'"))
        else:
            print(("Did not find a secondary star in Probe #"+str(ifus)+"."))
