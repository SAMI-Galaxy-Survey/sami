#! /usr/bin/env python

version = 2.0
date = '3 June, 2013'
day = ''

"""Flux calibration package for SAMI data.

Requires pyfits, astropy

Version %s beta -- %s; %s

Developed by Edward Taylor, James Allen, Matt Owers, and Scott Croom.
Maintained (for now) by Edward Taylor - <ent@ph.unimelb.edu.au>

To do list:
* explore alternatives to chunk+sigclipped polyn fit for PSF params
* simultaneous fits for PSF location/size/shape using red & blue arms
* account for foreground Galactic dust for the secondary
* better algorithmic selection for the secondary model spectrum
* devise a good smoothing strategy for the secondary scaling factor
* header management (standard names and QC params; DAR coefficients, etc.)

""" % (version, date, day)

import numpy as np
np.seterr(invalid='ignore')
from matplotlib import pyplot as plt

from scipy.optimize import leastsq, fmin, fmin_powell

from astropy import coordinates as coord
from astropy import units
from astropy.io import fits as pf
import atpy      # At some point, replace this with astropy equivalent
import os, sys, time, urllib, urllib2

from .. import utils

HG_CHANGESET = utils.hg_changeset(__file__)

i_am_ned = False
if i_am_ned :
    at_the_telescope = False

make_diagnostic_plots = True

do_full_fit = True
double_gaussian_fits = True

at_the_telescope  = False
if at_the_telescope :
    do_full_fit = False

path_to_standard_spectra = ['./standards/ESO/',
                            './standards/Bessell/']
standard_star_catalog = ['ESOstandards.dat',
                         'Bessellstandards.dat']
atm_extinction_table = './standards/extinsso.tab'

chunk_min = 24
n_chunks = 20
chunk_size = 100

fibre_radius_arcsec = 0.798

ifucmap = plt.cm.gist_earth

# ____________________________________________________________________________
# ____________________________________________________________________________

# def fluxcal( path_to_standards, path_to_data, 
#              do_secondary_calibration=True,
#              path_to_ESO_standards=path_to_standard_spectra,
#              extraid=None, datestamp=None, verbose=True ):

#     transfer_fn, transfer_var = get_transfer_function( 
#         path_to_standards, path_to_ESO_standards=path_to_ESO_standards,
#         verbose=verbose )

#     apply_flux_calibration( path_to_data, transfer_fn, transfer_var, 
#                             do_secondary_calibration=do_secondary_calibration,
#                             extraid=None, datestamp=None, verbose=verbose )

    # transfer_fn, transfer_var = get_transfer_function( 
    #     path_to_standards, path_to_ESO_standards=path_to_ESO_standards,
    #     verbose=verbose )
    # 
    # apply_flux_calibration( path_to_data, transfer_fn, transfer_var, 
    #                         extraid=extraid, verbose=verbose )

# ____________________________________________________________________________
# ____________________________________________________________________________

def apply_flux_calibration( path_to_data, transfer_fn, transfer_var,
                            do_secondary_calibration=False, 
                            extraid=None, datestamp=None, verbose=True ):

    datafiles = os.listdir( path_to_data )
    for filename in datafiles[ ::-1 ] :
        if not filename.endswith( 'red.fits' ):
            datafiles.remove( filename )
        elif datestamp != None :
            if not filename.startswith( datestamp ):
                datafiles.remove( filename )

    if len( datafiles ) == 0 :
        return

    if verbose > 0 :
        print '_' * 78
        print '\nApplying flux calibration to data in path:'
        print path_to_data ; print

    if verbose > 0 :
        print
        print 'Found %i data frames.  Starting work now.' % len( datafiles )

    plt.figure( 12, figsize=[14,4] ) ; plt.clf()

    for fitsfilename in datafiles :

        if extraid == None :
            newfitsfilename = fitsfilename.rstrip( 'red.fits' )+'fcal.fits'
        else :
            newfitsfilename = fitsfilename.rstrip( 'red.fits' )+'fcal_'+extraid+'.fits'

        path_in = os.path.join(path_to_data, fitsfilename)
        path_out = os.path.join(path_to_data, newfitsfilename)
        primary_flux_calibrate(path_in, path_out, transfer_fn, transfer_var)
        
        if do_secondary_calibration :
            perform_telluric_correction('%s/%s' % (path_to_data, 
                                                   newfitsfilename))


def primary_flux_calibrate(path_in, path_out, transfer_fn, transfer_var):
    """Apply primary flux calibration to a single file."""
    filename_in = os.path.basename(path_in)
    filename_out = os.path.basename(path_out)
    print '\nApplying primary calibration:'
    print '%s   --->   %s ' % ( filename_in, filename_out ),
    if os.path.exists( path_out ):
        print '(overwriting existing file)',
        os.remove( path_out )
    print '...',
    sys.stdout.flush()

    fitsfile = pf.open( path_in )

    ### XXX NOTE TO SELF: CHECK THAT VAR SHOULDN'T BE x 100 XXX ###
    data = fitsfile[0].data * 10.
    var  = fitsfile[1].data * 10.

    data *= transfer_fn
    var  *= transfer_fn
    var  *= ( 1. + np.sqrt( transfer_var / transfer_fn ) )

    fitsfile[0].data = data
    fitsfile[1].data = var

    hdr = fitsfile[0].header
    hdr.add_history( 'Flux calibrated using SAMI_fluxcal %s (%s)' % (
                version, time.ctime() ) )

    fitsfile.writeto( path_out )
    fitsfile.close()

    print 'done.' 

    return
                

# ____________________________________________________________________________
# ____________________________________________________________________________

def get_transfer_function( path_to_standards, 
                           path_to_ESO_standards=path_to_standard_spectra,
                           verbose=True,
                           save=True ):

    if verbose > 0 :
        print '_' * 78
        print '_' * 78
        print '\nDetermining flux calibration using standard star data in path:'
        print path_to_standards ; print

    standards_list = find_standard_star_dataframes( path_to_standards, 
        path_to_ESO_standards=path_to_ESO_standards, verbose=verbose )

    if make_diagnostic_plots :
        plt.figure( 4 ) ; plt.clf()
        if len( standards_list ) > 1 :
            colors=plt.cm.spectral(np.linspace(.95, 0., len(standards_list)+1))
        else:
            colors = 'k'

    transfer_fn = None
    for si, ( fitsfilename, probename, 
              standardfile, standardname, standardoffset
              ) in enumerate( standards_list ):
        transfer_fn_single, transfer_fn_var_single, sensitivity_single, \
            datadict, standardwl = \
            get_transfer_function_single(si, fitsfilename, probename, 
              standardfile, standardname, standardoffset, color=colors[si], 
              verbose=verbose, save=save)
        if transfer_fn is None:
            transfer_fn = transfer_fn_single
            transfer_fn_var = transfer_fn_var_single
            sensitives = sensitivity_single
        else :
            transfer_fn    = np.vstack((transfer_fn, transfer_fn_single))
            transfer_fn_var= np.vstack((transfer_fn_var, transfer_fn_var_single))
            sensitives     = np.vstack((sensitives, sensitivity_single))

    ymax, yref0, yrefstep = 18., 15.6, 0.8145
    if np.median( datadict[ 'wl' ] )  < 6000 :
        plotlims = np.array( ( 3350., 6650. ) )
        xref = 3500.
    else :
        plotlims = np.array( ( 6150., 7650. ) ) 
        scaleby = 31./18
        xref = 6865.
        ymax *= scaleby ; yrefstep *= scaleby
        yref0 = 4.3 + len( standards_list ) * yrefstep
    yref = yref0 - yrefstep*len( standards_list )

    if len( transfer_fn.shape ) > 1 :

        nmeasures = transfer_fn.shape[0]
        weighting = np.zeros( transfer_fn.shape )

        if verbose > 0 :
            print '_' * 78
            print '\nCombining results from %i standard stars to get final flux calibration.' % nmeasures


#        final_transfer_fn = ( np.nansum( transfer_fn/transfer_fn_var, axis=0 )
#                              / np.nansum( 1./transfer_fn_var, axis=0 ) )
#        final_transfer_var = np.nansum( transfer_fn_var, axis=0 ) \
#            / np.sum( np.isfinite( transfer_fn_var ), axis=0 )
#
#        final_transfer_var += np.std( transfer_fn, axis=0 )**2.

        print transfer_fn.shape
        for tfi in range( nmeasures ):
            comp    =  transfer_fn[  tfi, :  ]  -   transfer_fn
            compvar = transfer_fn_var[tfi, : ] + transfer_fn_var
            compvar+= ( 0.01 * transfer_fn[ tfi, :  ] )**2.

            likely = ( np.exp( -0.5 * comp**2./compvar ) 
                       / np.sqrt(2.*np.pi*compvar) )

            likely = np.where( np.isfinite( likely ), likely, 0. )
            likely[ tfi, : ] = 0.

            weighting += likely
            
        if make_diagnostic_plots or 1 : 
            plt.figure( 5 ) ; plt.clf()
            plt.imshow( weighting / weighting.sum( axis=0 ), 
                        origin='lower', aspect='auto', 
                        extent=(standardwl.min(), standardwl.max(), 
                                1., nmeasures+1 ) )
            plt.yticks( np.arange( nmeasures )+1.5, range( nmeasures ) )
            plt.xlim( plotlims[0], plotlims[1] ) ; 
            plt.title( 'relative weights given to each frame in the final calibration' )
            plt.ylabel( 'frame', fontsize='xx-large' )
            plt.xlabel( 'wavelength', fontsize='xx-large' )
            plt.colorbar()
            plt.draw()

        weighting /= weighting.sum( axis=0 )

        weighting = np.where( ( transfer_fn > 0 ) & ( weighting > 0 ),
                              weighting, 0 )
        transfer_fn = np.where( transfer_fn > 0, transfer_fn, 0 )
        sensitives = np.where( sensitives > 0, sensitives, 0 )

        final_transfer_fn = np.sum( weighting * transfer_fn , axis=0 )
        final_sensitivity = np.sum( weighting *  sensitives , axis=0 )



        s2nsquared = transfer_fn**2. / transfer_fn_var
        s2nsquared = np.where( s2nsquared > 0, s2nsquared, 0. )
        final_s2n2 = np.sum( weighting * s2nsquared, axis=0 )
        
        final_transfer_var = final_transfer_fn**2. / final_s2n2

        percenterr = np.sqrt(final_transfer_var) /final_transfer_fn*100.
        percenterr = percenterr[ np.isfinite( percenterr ) ]
        if verbose > 0 :
            print '\nMedian uncertainty in the final flux cal across the specrum is %.2f%% .' % ( np.median( percenterr ) )
                
               

    else :
        final_transfer_fn = transfer_fn 
        final_transfer_var= transfer_fn_var
        final_sensitivity = sensitives

        print "\nOnly the one spectrophotometric standard observation, so i'm all done here."

    if save:
        save_combined_transfer_fn(
            final_transfer_fn, 
            final_transfer_var,
            final_sensitivity,
            transfer_fn,
            transfer_fn_var,
            sensitives,
            standards_list,
            path_to_standards,
            )

    if make_diagnostic_plots or 1 :
        plt.figure( 4 ) 
        
        # xref, yref = np.mean( datadict[ 'wl' ] ), 
        if len( transfer_fn.shape ) > 1 :
            plt.plot( standardwl, np.clip( final_sensitivity, final_sensitivity, 0 )* 100., '-',
                      standardwl, (np.clip( np.sqrt( final_transfer_var )
                                            / final_transfer_fn, 0., 1. ) * 100.), '--',
                      standardwl, (np.std(transfer_fn, axis=0)
                                   / final_transfer_fn ) * 100., ':', 
                  [xref, xref], [yref, yref], 's', lw=3, ms=8, color=colors[-1] )
            plt.text( xref, yref, 'final combined', 
                      ha='left', va='center', fontsize='medium' )


        plt.plot( plotlims, np.ones(2) * np.median(percenterr), 'k-', lw=1 )

        plt.text( plotlims[1]-85., np.median(percenterr) + .2, 
                  'median uncert. = %.2f%%' % np.median(percenterr), 
                  va='bottom', ha='right' )

        #add_date_stamp( figfilename )
        plt.xlim( plotlims ) ;        plt.ylim( 0., ymax )
        plt.grid()
        plt.draw()

        plt.savefig( '%s/fluxcal_%s.png' % ( path_to_standards, standardname ) )
        print '  Look at %s/fluxcal_%s.png' % ( path_to_standards, standardname ) 

    final_transfer_fn   = np.interp( datadict['wl'], standardwl, 
                                     final_transfer_fn )
    final_transfer_var=np.interp(datadict['wl'],
                                 standardwl,final_transfer_var)

    if verbose > 0 :
        print '_' * 78

    return final_transfer_fn, final_transfer_var


def get_transfer_function_single(si, fitsfilename, probename, 
        standardfile, standardname, standardoffset, color='k', 
        verbose=True, save=True):
    if verbose > 0 :
        print '_' * 78
        print "\nLooking at %s; probe %s in %s (dataframe %i)." % ( standardname, probename, fitsfilename.split('/')[-1], 
    si+1)

    chunkfigurefile = fitsfilename.rstrip( '.fits' ) + '.chunk.png' 
    print chunkfigurefile
    
    startTime = time.time()
    
    standardspec, standardwl = get_standard_spectrum( 
        standardfile, verbose=verbose )

    # extract the SAMI IFS data for the standard, given the bundle name.
    datadict = IFU_pick( fitsfilename, probename, 
                         extincorr=True, verbose=verbose )

    onlyneed = np.interp( ( datadict['wl'].min(), datadict['wl'].max() ),
                          standardwl, np.arange(standardwl.shape[0]) )
    if onlyneed[0] > 0 :
        onlyneed[ 0 ] -= 1
    standardwl   =  standardwl[ onlyneed[0]:onlyneed[1]+2 ]
    standardspec = standardspec[ onlyneed[0]:onlyneed[1]+2 ]

    xfibre, yfibre = relative_fibre_positions( datadict )
    data, var, wl = datadict[ 'data' ], datadict[ 'var' ], datadict[ 'wl' ]

    startTime = time.time()
    # fit a 2D Gaussian to the PSF as a fn of wavelength
    PSFmodel = fit_psf_afo_wavelength( datadict, polydeg=1,
        chunk_min=24, n_chunks=n_chunks, chunk_size=chunk_size, 
        verbose=verbose, chunkfigurefile=chunkfigurefile )

    if make_diagnostic_plots :
        figfilename = '%s_%s.psf.png' % ( 
            fitsfilename.rstrip( '.fits' ), standardname )            
        add_date_stamp( figfilename, top=True )
        plt.draw()
        plt.figure( 2 )
        print 'Making diagnostic plot ...', ; sys.stdout.flush()
        plt.savefig( figfilename )
        print 'done.'
        print '  Look at', figfilename
        print

    # get Gaussian amplitudes from the IFS data as a fn of wavelength
#        amplitude, uncertainty = extract_total_flux( 
#                datadict, PSFmodel, verbose=verbose )

    amplitude, uncertainty = extract_total_flux_new( 
            datadict, PSFmodel, verbose=verbose )

    # median smooth the amplitudes on a small scale
    amp_smoothed, smoothed_scatter = median_smooth( 
        amplitude, binsize=16, givenmad=True )
    unc_smoothed = median_smooth( uncertainty, binsize=16 )

    # censor outlying data
    censor = ( np.abs( amplitude-amp_smoothed ) 
               / np.sqrt( unc_smoothed**2. + smoothed_scatter**2. ) ) > 5.
    uncensored = np.copy( amplitude )
    amplitude = np.where( censor, np.nan, uncensored )
    print 'Censoring %i wavelengths (%.1f%%)' % (
        censor.sum(), float(censor.sum())/datadict['wl'].shape[0]*100.)

    # rebin the Gaussian-fit amplitudes onto the wl grid of the standard
    amp_rebinned, unc_rebinned = rebin_spectrum( 
        standardwl, datadict[ 'wl' ], amplitude, unc_smoothed )

    ratio = standardspec / amp_rebinned
    unc_ratio = unc_rebinned / amp_rebinned * ratio

    avoid = [ 4861.-20, 4861.+20 ]
    cut = np.interp( avoid, standardwl, np.arange(standardwl.shape[0]))
     
    # std spec have units 10-16 erg/cm2/s/A
    h_in_erg_sec = 6.62606957e-27
    c_in_ang_per_sec = 2.997924581e18

    mirror_area_in_cm2 = 0.84 * np.pi * 195.**2.

    std_energy  = h_in_erg_sec * c_in_ang_per_sec / standardwl
    std_photons = standardspec / 1.e16 / std_energy 
    std_photons *= mirror_area_in_cm2
    
    sensitivity = ( datadict[ 'GAIN' ] * amp_rebinned 
                    / datadict[ 'CDELT1' ] / std_photons )

    ymax, yref0, yrefstep = 18., 15.6, 0.8145
    if np.median( datadict[ 'wl' ] )  < 6000 :
        plotlims = np.array( ( 3350., 6650. ) )
        xref = 3500.
    else :
        plotlims = np.array( ( 6150., 7650. ) ) 
        scaleby = 31./18
        xref = 6865.
        ymax *= scaleby ; yrefstep *= scaleby
        #yref0 = 4.3 + len( standards_list ) * yrefstep
        yref = 4.3 + 3 * yrefstep


    if cut[0] > 0 and 0 :
        standardwl = np.hstack( ( standardwl[ :cut[0] ], 
                              standardwl[ cut[1]+1: ] ) )
        standardspec = np.hstack( ( standardspec[ :cut[0] ], 
                              standardspec[ cut[1]+1: ] ) )
        ratio = np.hstack( ( ratio[ :cut[0] ], ratio[ cut[1]+1: ] ) )
        unc_ratio = np.hstack(( unc_ratio[ :cut[0]], 
                                unc_ratio[cut[1]+1: ] ))
        sensitivity = np.hstack(( sensitivity[ :cut[0]], 
                                  sensitivity[cut[1]+1: ] ))  

    if save:
        save_transfer_fn(fitsfilename, ratio, unc_ratio**2., sensitivity,
                         probename, standardfile, standardname,
                         standardoffset)

    transfer_fn    = ratio
    transfer_fn_var= unc_ratio**2.

    if make_diagnostic_plots :
        if verbose > 0 :
            sys.stdout.flush()

        maxfiber = np.argmax(np.median(datadict['data']/amplitude, axis=1))

        plt.figure( 3 ) ; plt.clf()
        plt.plot( datadict[ 'wl' ], amplitude, 'k-' )
        plt.scatter( datadict[ 'wl' ][ censor ], uncensored[ censor ],
                     20, 'r', facecolors='r', marker='x', lw=2 )
        
        plt.fill_between( datadict[ 'wl' ], 
                          2.*( datadict[ 'data' ][ maxfiber ]
                          -3. * np.sqrt( datadict[ 'var' ][ maxfiber ] ) ),
                          2.*(datadict[ 'data' ][ maxfiber ]
                          +3. * np.sqrt( datadict[ 'var' ][ maxfiber ] ) ),
                          color='b', alpha=0.2, zorder=2 )

        plt.plot( datadict[ 'wl' ], 2.*datadict['data'][ maxfiber ], 'b-' )

        plt.scatter( standardwl, amp_rebinned, 160, 'none', zorder=4, 
                     edgecolors='w', linewidths=3, marker='s' )
        plt.scatter( standardwl, amp_rebinned, 160, 'none', zorder=4, 
                     edgecolors='r', linewidths=1, marker='s' )

        yrange = np.nanmax(amp_rebinned) - min(np.nanmin(amp_rebinned), 0)

        plt.ylim( max( np.nanmin( amp_rebinned ), 0 )-0.035 * yrange, 
                  np.nanmax( amp_rebinned )+0.15 * yrange )
        plt.xlim( plotlims )
        plt.xlabel( 'wavelength', fontsize='xx-large' )
        plt.ylabel( 'flux', fontsize='xx-large' )
        plt.title( 'file: %s; probe: %s;\nspecphot star: %s' % ( 
                fitsfilename.split('/')[-1], probename, standardname ), 
                   fontsize='x-large' )

        plt.figtext( 0.15565, 0.735, 
                     'blue: brightest fibre (#%i) x 2\nblack: Gaussian fits\ncrosses: censored as outlying\nboxes: smoothed & rebinned\nshaded: 3 x uncertainties' % (maxfiber+1) )

        add_date_stamp( figfilename )
        plt.draw()
        figfilename = '%s_%s.amp.png' % ( 
            fitsfilename.rstrip( '.fits' ), standardname )

        plt.savefig( figfilename )

#            xref, yref = np.mean( datadict[ 'wl' ] ), 26. * 0.85**si
        yref = yref0 - yrefstep*si


        oktp = ( (np.isfinite( sensitivity )) & (np.isfinite( ratio ))
                & np.isfinite( unc_ratio ))

        unc_sens = unc_ratio/ratio
        unc_sens = np.clip( unc_sens, 0., 1. )
        
        plt.figure( 4 ) 
        plt.plot( standardwl[oktp], (sensitivity)[oktp]*100., '-', 
                  standardwl[oktp], np.clip( unc_sens[oktp], 0., 1. ) * 100., '--', 
                  [xref], [yref], 's', 
                  ms=8, lw = 3, color=color 
                  )
                  
        plt.text( xref+50, yref, fitsfilename.split('/')[-1], 
                  ha='left', va='center', fontsize='medium' )
        plt.xlim( plotlims[0], plotlims[1] )
        plt.ylim( 0., ymax )   
        plt.xlabel( 'wavelength', fontsize='xx-large' )
        plt.ylabel( 'net per-fibre sensitivity (%)', fontsize='xx-large' )
        plt.title( 'datadir: %s\nspecphot star: %s' % 
                   ( os.path.dirname(standardfile), standardname ), fontsize='x-large' )
        if si == 0 :
            plt.text( np.mean( plotlims ), .99 * ymax, # 0.45, 
"""dashed lines show *relative* uncertainties in each measurement as a percentange
dotted line shows RMS variation between multiple measurements in the same way"""
                      , ha='center', va='top', fontsize='small' )

        plt.draw()

    percenterr = unc_ratio / ratio *100. 
    percenterr = percenterr[ np.isfinite( percenterr ) ]

    if verbose > 0 :
        print 'Median uncertainty in the transfer function across the specrum is %.2f%% .' % ( 
            np.median( percenterr ) )
        if make_diagnostic_plots :
            print '  Look at %s' % figfilename
        print '\n Finished with that data frame. (Took %.1f sec.)' % (
            time.time()-startTime)

    return transfer_fn, transfer_fn_var, sensitivity, datadict, standardwl



def save_transfer_fn(fitsfilename, ratio, variance, sensitivity,
                     probename, standardfile, standardname, standardoffset):
    """Save the derived transfer function for an individual file."""
    # Turn the data into a single array
    data = np.vstack((ratio, variance, sensitivity))
    # Make the new HDU
    hdu_name = 'TRANSFER_FUNCTION'
    new_hdu = pf.ImageHDU(data, name=hdu_name)
    # Add info to the header
    header_item_list = [
        ('PROBE', probename, 'Name of the probe the star is in'),
        ('STDFILE', standardfile, 'Filename of standard spectrum'),
        ('STDNAME', standardname, 'Name of standard star'),
        ('STDOFF', standardoffset, 'Offset (arcsec) to standard star '
                                   'coordinates'),
        ('HGFLXCAL', HG_CHANGESET, 'Hg changeset ID for fluxcal code'),
        ]
    for key, value, comment in header_item_list:
        new_hdu.header[key] = (value, comment)
    # Update the file
    hdulist = pf.open(fitsfilename, 'update',
                      do_not_scale_image_data=True)
    # Check if there's already a transfer function, and delete if so
    try:
        existing_index = hdulist.index_of(hdu_name)
    except KeyError:
        pass
    else:
        del hdulist[existing_index]
    hdulist.append(new_hdu)
    hdulist.close()
    del hdulist
    return

def save_combined_transfer_fn(final_transfer_fn, final_transfer_var,
                              final_sensitivity, transfer_fn, transfer_fn_var,
                              sensitivity, standards_list, path_to_standards):
    """Save the combined transfer function in a new file."""
    filename = 'TRANSFERcombined.fits'
    # Stick together the combined data
    data = np.vstack((final_transfer_fn, final_transfer_var, final_sensitivity))
    # Make the primary HDU
    primary_hdu = pf.PrimaryHDU(data)
    # Add info to the header
    header_item_list = [
        ('HGFLXCAL', HG_CHANGESET, 'Hg changeset ID for fluxcal code'),
        ]
    for key, value, comment in header_item_list:
        primary_hdu.header[key] = (value, comment)
    # Make an HDU list, to include data from each contributing file
    hdulist = pf.HDUList([primary_hdu])
    hdu_name = 'INPUT_FUNCTION'
    for index, standard in enumerate(standards_list):
        # Stick together the data for this input file
        data = np.vstack((transfer_fn[index, :],
                          transfer_fn_var[index, :],
                          sensitivity[index, :]))
        hdu = pf.ImageHDU(data, name=hdu_name)
        # Add info to the header
        (fitsfilename, probename, standardfile, standardname, 
         standardoffset) = standard
        header_item_list = [
            ('EXTVER', index + 1, ''),
            ('FILENAME', fitsfilename, 'Filename of observed spectrum'),
            ('PROBE', probename, 'Name of the probe the star is in'),
            ('STDFILE', standardfile, 'Filename of standard spectrum'),
            ('STDNAME', standardname, 'Name of standard star'),
            ('STDOFF', standardoffset, 'Offset (arcsec) to standard star '
                                       'coordinates'),
            ]
        for key, value, comment in header_item_list:
            hdu.header[key] = (value, comment)
        hdulist.append(hdu)
    path = os.path.join(path_to_standards, filename)
    if os.path.exists(path):
        os.remove(path)
    hdulist.writeto(path)
    return

def read_combined_transfer_fn(filename):
    """Read the combined transfer function from a saved file."""
    data = pf.getdata(filename)
    transfer_fn = data[0,:]
    transfer_fn_var = data[1,:]
    return transfer_fn, transfer_fn_var    


# ____________________________________________________________________________
# ____________________________________________________________________________

def perform_telluric_correction( samifitsfilename=None, verbose=1 ):

    # Also, with the telluric bands, these are at:
    # 6850.0-6960.0, 7130.0-7360.0, 7560.0-7770.0, 8100.0-8360.0
    # email from Scott 14/5/13 re: discussion of flux cal tomorrow (Friday)?
    # for now, only derive/apply correction to this limited wavelength range.

#    modelwl, modelspec, modelnames = load_atlas_stars( )
    modelwl, modelspec, modelnames = load_CK04_models( )

    # filename management ... boring
    pieces = samifitsfilename.split( '/' )
    pieces[ -1 ] = pieces[ -1 ][ 0:10 ] #             filename-->datestamp only
    basefilename = pieces[0]   
    for bit in pieces[1:] : #                              reassemble path name
        basefilename += '/' + bit
    chunkfigurefile = basefilename + '.chunks.png' 
    corrfigfilename = basefilename + '.fcal.png'
    fcalfitsfilename= basefilename + 'sci.fits'

    # pull out the data from the SAMI dataframe
    datadict = extract_secondary_standard( samifitsfilename, verbose=verbose )
    xfibre, yfibre = relative_fibre_positions( datadict )
    data, var, wl = datadict[ 'data' ], datadict[ 'var' ], datadict[ 'wl' ]

    # limit correction to telluric bands only
    tellbands = [ [6850, 6960], [7130, 7360], [7560, 7770], [8100, 8360] ]
    tellurics = ( ( ( 6850. <= wl ) & ( wl <= 6960. ) )
                  | ( ( 7130. <= wl ) & ( wl <= 7360. ) )
                  | ( ( 7560. <= wl ) & ( wl <= 7770. ) )
                  | ( ( 8100. <= wl ) & ( wl <= 8360. ) ) )

    # characterise the PSF shape as a function of wavelength ...
    PSFmodel = fit_psf_afo_wavelength( datadict, 
        chunk_min=24, n_chunks=20, chunk_size=100, polydeg=0, verbose=1,
                                       chunkfigurefile=chunkfigurefile )
#    PSFmodel = fit_gaussian_afo_wavelength( datadict, 
#        chunk_min=24, n_chunks=10, chunk_size=200, polydeg=0, verbose=1 )

    if make_diagnostic_plots :
        figfilename = '%s.psf.png' % ( samifitsfilename.rstrip( '.fits' ) )
        add_date_stamp( figfilename, top=True )
        plt.draw()
        plt.figure( 2 )
        print 'Making diagnostic plot ...', ; sys.stdout.flush()
        plt.savefig( figfilename )
        print 'done.'
        print '  Look at', figfilename

    # ... and use that to solve for the total flux of the standard.
    amplitude, uncertainty = extract_total_flux_new( 
        datadict, PSFmodel, verbose=verbose )

    # now median smooth the spectrum on a small scale ...
    amp_smoothed, smoothed_scatter = median_smooth( 
        amplitude, binsize=16, givenmad=True )
    unc_smoothed = median_smooth( uncertainty, binsize=16 )

    # ... in order to censor outlying data.
    censor = ( np.abs( amplitude-amp_smoothed ) 
               / np.sqrt( unc_smoothed**2. + smoothed_scatter ) ) > 5.
    amplitude = np.where( censor, np.nan, amplitude )
    print 'Censoring %i wavelengths (%.1f%%)' % (
        censor.sum(), float(censor.sum())/datadict['wl'].shape[0]*100.)

    # isolate those parts of the model spectra in the right wavelength range
    onlyneed = np.where( ( datadict['wl'].min() <= modelwl ) 
                         & ( modelwl <= datadict['wl'].max() ) )[0]
    needlo, needhi = onlyneed.min()-1, onlyneed.max()+2
    modelspec, modelwl = modelspec[ :,needlo:needhi ], modelwl[ needlo:needhi ]

    # (prepare gaussian smoothing kernel for later)
    smoothfwhm = 8
    smoothstddev = smoothfwhm / 2.35482
    pixelsteps = int( 5 * smoothstddev ) + 1
    pixels = np.arange( -pixelsteps, pixelsteps+1 )
    kernel = np.exp( -0.5 * ( pixels / smoothstddev )**2. )
    # (or maybe just a tophat!)
#    kernel = np.ones( 2 * smoothfwhm )

    # now look to see which model best describes the data - nothing fancy here.
    bestchi2 = np.inf
    bestindex = None
    for modi, model in enumerate( modelspec ):
        # interpolate the model spectrum onto the data (SHOULD REBIN?)
        interspec = np.interp( wl, modelwl, model )

        # fit for the overall amplitude, with some sigma-clipping.
        fitto = (tellurics == False) 
        scaling, chi2 = fit_for_amplitude( 
            interspec, amplitude, uncertainty**2., fitto, sigclip=30. )

        # you're the best! better than all the rest ...
        if chi2 < bestchi2 :

            # multiplicative correction is just the ratio of fluxes
            bestcorr = amplitude / scaling / interspec
            sig2noise = amplitude / uncertainty

            gausssmooth, medsmooth = True, False

            if gausssmooth:
                # check for meaningful (finite) data
                check = sig2noise > 0.
                weighting = np.where( check, 1., 0 )

                masked = np.where( check, bestcorr, 1. )
#                masked = np.where( check & tellurics, correction, 1. )

                correction = np.convolve(weighting * masked , 
                                         kernel, mode='same' )
                sig2noise = np.convolve( weighting * sig2noise**2. , 
                                         kernel, mode='same' )
                weighting = np.convolve( weighting, 
                                         kernel, mode='same' )

                correction = correction / weighting
                sig2noise  =  np.sqrt( sig2noise )
                sig2noise = np.where( sig2noise > 0, sig2noise, 0 )

            elif mediansmooth:
                correction = median_smooth( correction, binsize=16 )
                sig2noise  = median_smooth( sig2noise , binsize=16 )

            # this one is only for plotting purposes ...
            smoothed = median_smooth( correction, binsize=16 )

            bestchi2 = chi2
            bestindex = modi
            bestfit = scaling * interspec

            plotthings = [ amplitude, amplitude / correction, bestfit ]
            plotcolors = [ 'k', 'b', 'r' ]
            plt.figure( 18, figsize=[14,4] ) ; plt.clf()
            for pthing, pcolor in zip( plotthings, plotcolors ):
                ytp, dytp = rebin_spectrum( modelwl, wl, pthing, sig2noise )
                plt.plot( modelwl, ytp, '-', color=pcolor )
            yskirt = 3.15 * ( bestfit.max() - bestfit.min() )
            plt.ylim( bestfit.min()-yskirt, bestfit.max() + yskirt )
            plt.draw()

            # ... like this right here.
            plt.figure( 12, figsize=[14,4] ) ; plt.clf()
            plt.scatter( wl, bestcorr, 1, 'k', zorder=0, alpha=0.2 )
            plt.plot( wl, correction, 'r-', zorder=5, lw=1 )
            plt.fill_between( wl, correction * (1. + 3./sig2noise),
                              correction * (1. - 3./sig2noise),
                              color='gray', zorder=1, alpha=.6 )
            plt.plot( wl, smoothed, 'r-', lw=2, zorder=2 )
            for ( tlo, thi ) in tellbands :
                plt.plot( [ tlo, tlo ], [ 0, 2 ], '-', color='gray', lw=.5, zorder=-9 )
                plt.plot( [ thi, thi ], [ 0, 2 ], '-', color='gray', lw=.5, zorder=-9 )
                plt.fill( [ tlo, thi, thi, tlo, tlo ], [0, 0, 2, 2, 0], zorder=-9,
                          fill=False, color='gray', lw=0.5, hatch='/',  )

            wllim = [ wl.min(), wl.max() ]
            plt.plot( wllim, [ 1. , 1. ], 'k-', lw=0.5 )
            plt.plot( wllim, [1.05,1.05], 'k-', lw=0.5 )
            plt.plot( wllim, [0.95,0.95], 'k-', lw=0.5 )
            plt.xlim( wllim ) ; plt.ylim( 0.6, 1.5 )
    plt.title( '%s: best fit model: %s' % (
            fcalfitsfilename, modelnames[ modi ] ) )
                    
#            print 1.486 * np.median( np.abs( correction - smoothed ) ),
#            print np.median( unc_smoothed / amp_smoothed ),
#            print np.median( 1./sig2noise )

    add_date_stamp( corrfigfilename )
    plt.draw()            
    plt.savefig( corrfigfilename )
            
    if not np.isfinite( bestchi2 ):
        print 'WARNING from SAMI.perform_telluric_correction:'
        print 'I have found no meaningful fit to secondary star data in frame:'
        print '    ', samifitsfilename
        print 'This probably means that the secondary is not centred in the bundle,'
        print 'or that the primary calibration has failed badly.'
        print

        return 1, 1

    print '\nApplying secondary calibration now.'
    print '    %s --> %s' % ( samifitsfilename, corrfigfilename ), 

    fitsfile = pf.open( samifitsfilename )
    hdr = fitsfile[0].header
    
    hdr.add_history( 'Telluric corrected using secondary standard star.' )
    hdr.add_history( 'Empirical DAR correction fit for.' )
    hdr.update( 'DARFITPA', datadict[ 'DAR_FIT_POSANG' ], 
             'DAR position angle - deg N of E' )
    hdr.update( 'DARLAM0', datadict[ 'DAR_FIT_LAM0' ], 
             'Reference wavelength for DAR fit' )
    hdr.update( 'DARNORM', datadict[ 'DAR_FIT_ACOEFF' ],
                'DAR fit: D R = DARNORM x (wl/DARLAM0)^DARPOWER' )
    hdr.update( 'DARPOWER', datadict[ 'DAR_FIT_PCOEFF' ],
                'DAR fit: D R = DARNORM x (wl/DARLAM0)^DARPOWER' )

    print hdr

    data = fitsfile[0].data 
    var  = fitsfile[1].data 

    data /= correction
    var  /= np.sqrt( correction )

    fitsfile[0].data = data
    fitsfile[1].data = var

    if os.path.exists( fcalfitsfilename ) :
        os.remove( fcalfitsfilename )
        print '(replacing existing file)', 

    fitsfile.writeto( fcalfitsfilename )
    fitsfile.close()

    print 'done.' ; print ; print '_' * 78

    return correction, sig2noise

# ____________________________________________________________________________
# ____________________________________________________________________________

def fit_for_amplitude( model, data, var, fitto=None, sigclip=30. ):

    if fitto != None :
        firstguess = np.median( data / model )
    else :
        firstguess = np.median( ( data / model )[ fitto ] )

#    print firstguess,
    result = fmin( amplitude_chi2, firstguess, 
                   args=( model, data, var, sigclip, fitto ), disp=False )
#    print result[0] / firstguess
    chi2 = amplitude_chi2( result, model, data, var, fitto=fitto )

    return result, chi2

def amplitude_chi2( amplitude, model, data, var, sigclip=30., fitto=None ):

    chi2 = ( amplitude * model - data )**2. / var
    useme = ( chi2 <= sigclip**2. ) & ( fitto )
    chi2 = np.clip( chi2, 0., sigclip**2. )
    if fitto != None :
        chi2 = chi2[fitto]

    if 0 and np.random.rand() < .01 :
        plt.figure( 18 ) ; plt.clf()
        xtp = np.arange( data.shape[0] )
        plt.plot( xtp, amplitude * model, 'r-' )
        plt.scatter( xtp, data, 2, 'r', edgecolors='none', zorder=10 )
        plt.scatter( xtp[ useme ], data[useme ], 4, 'k', zorder=1 )
        for ui in np.where( useme )[0]:
            plt.plot( [ ui, ui ], [ data[ui], amplitude*model[ui] ], 
                      'r-', lw=1, zorder=-5 )
        plt.title( np.nansum( chi2 ) )
        plt.draw()

        print amplitude, chi2[48:-48:100], np.nansum( chi2 )
    return np.nansum( chi2 )

# ____________________________________________________________________________
# ____________________________________________________________________________

def chunk_spectrum( data, wl, chunk_min=10, n_chunks=40, chunk_size=50 ):

    # break the data into largish chunks
    nfibres = data.shape[0]
    chunked_data = ( data[:,chunk_min:chunk_min+n_chunks*chunk_size ] 
                ).reshape( nfibres, n_chunks, chunk_size )
    chunked_wl = ( wl[ chunk_min:chunk_min+n_chunks*chunk_size ] 
                ).reshape( n_chunks, chunk_size )

    chunked_data = np.median( chunked_data, axis=2 )
    chunked_wl   = np.median( chunked_wl  , axis=1 )

    return chunked_data, chunked_wl

# ____________________________________________________________________________
# ____________________________________________________________________________
       
def fit_psf_afo_wavelength( datadict, 
    chunk_min=chunk_min, n_chunks=n_chunks, chunk_size=chunk_size,
    polydeg=1, wavelengthout = None, verbose=False,
                            chunkfigurefile=None ):

    data, var, wavelength = datadict[ 'data' ], datadict[  'var'  ], datadict[ 'wl' ]
    xfibre, yfibre = relative_fibre_positions( datadict )

    if verbose > 0 :
        print '_' * 78
        print
        print 'Making Moffat function fits to chunked data to get PSF shape.'

    if wavelengthout == None:
        wavelengthout = wavelength

    # first pass using chunked spectrum
    chunked_data, chunked_wl = chunk_spectrum( data, wavelength,
                chunk_min=chunk_min,n_chunks=n_chunks,chunk_size=chunk_size )

    PSFfitpars = {}
    keys = 'xcen ycen alphax alphay beta rho flux bkg'.split()
    keys += 'lam'.split()
    for key in keys :
        PSFfitpars[ key ] = []

    net_flux = np.nansum( chunked_data, axis=0 )

    chunk_start = chunk_min + np.arange( n_chunks ) * chunk_size

    if make_diagnostic_plots :
        plt.figure( 101 ) ; plt.clf()
        nsubx = n_chunks / 5
        nsuby = n_chunks / 4
        plt.subplots_adjust( left=0.005, right=0.9485, top=0.965, bottom=0.01,
                             hspace=0.185, wspace=0.135 )

    result = None
    for chunki, chunk_flux in enumerate( net_flux ):
        if np.isfinite( chunked_data[:,chunki] ).sum() > 10 : 
            hourglass( chunki, n_chunks, verbose=verbose )

            result, fit_flux_frac = fit_covariant_moffat(
                xfibre, yfibre, chunked_data[ :, chunki ], 
                guess=None, verbose=verbose )

            if make_diagnostic_plots and chunkfigurefile != None :
                plt.figure( 101 )
                plt.subplot( nsubx, nsuby, chunki+1 )
                psf_diagnostic_plot( 
                    xfibre, yfibre, chunked_data[:,chunki], None, result,
                    datasize=255/n_chunks*1.5, modelsize=1020/n_chunks*1.8, 
                    edgecolors='none', lw=0.5, simpletitle=True, 
                    title='lam ~ %i AA' % chunked_wl[ chunki ] )
                    
            if make_diagnostic_plots :
                plt.figure( 1 ) ; plt.clf()
                psf_diagnostic_plot( 
                    xfibre, yfibre, chunked_data[:,chunki], None, result,
                    title='%s\nchunk #%i : lam ~ %i AA' % (
                        datadict['filename'], chunki+1, chunked_wl[ chunki ]))
                plt.draw()
            if fit_flux_frac < 0.5 or fit_flux_frac > 1. and verbose>(-1):
                print ; print
                print 'WARNING from SAMI_fluxcal.fit_psf_afo_wavelength:'
                print 'PSF fit for chunk %i (wl ~ %iA) looks odd.' % (
                    chunki, chunked_wl[chunki] )
                print 'The fit implies the SAMI bundle is seeing {:.1%} of the star.'.format( fit_flux_frac )
                print 'Check to see if the star is in the bundle?'
                print 'If this happens a lot, flux calibration for this frame may be meaningless.'
                print

            for ki, value in enumerate( result ) :
    #                print ki, keys[ ki ], value
                PSFfitpars[ keys[ ki ] ].append( value )

            PSFfitpars[ 'lam'  ].append( chunked_wl[ chunki ] )

            amp, bkg = covariant_moffat_chi2( result, 
                xfibre, yfibre, chunked_data[ :, chunki ], returnamp=True )

            PSFfitpars[ 'flux' ].append( amp )
            PSFfitpars[ 'bkg' ].append( bkg )

            hourglass( chunki+1, n_chunks, verbose=verbose )

    PSFfitpars[ 'lam' ] = np.array( PSFfitpars[ 'lam' ] )

    PSFfitpars = fit_for_DAR_corrections( 
        PSFfitpars, keys, wavelength, datadict, polydeg=polydeg, 
        verbose=verbose, make_diagnostic_plots=make_diagnostic_plots ) 

    PSFmodels = build_moffat_fibre_profiles( 
        xfibre, yfibre, wavelength, PSFfitpars, verbose=verbose )

    if make_diagnostic_plots and chunkfigurefile != None :
        plt.figure( 101 )
        add_date_stamp( chunkfigurefile )
        plt.savefig( chunkfigurefile )

    return PSFmodels

    
    if verbose > 0 :
        print 'Making polynomial fits to chunk results to get PSF shape afo wavelength.'

    if make_diagnostic_plots:
        plt.figure( 2, figsize=(7.5, 12) ) ; plt.clf()
        plt.subplots_adjust( top=0.95, bottom=0.05, left=0.105, right=0.965, 
                             hspace=.585, wspace=.285  )

    lam = np.array( PSFfitpars[ 'lam' ] )
    lam0 = np.mean( lam )
    lam -= lam0

    for ki, key in enumerate( keys[ :-1 ] ):

        hourglass( ki, len( keys )-1, verbose=verbose )

        PSFfitpars[ key ] = np.array( PSFfitpars[ key ] )

        polycoeffs, nmad = polyfit_sigclipped( lam, PSFfitpars[ key ], 
                                               polydeg, verbose=verbose )

        if key != 'flux' :
            result = np.polyval( polycoeffs, wavelengthout-lam0 )
        else :
            result = np.interp( wavelengthout-lam0, lam, PSFfitpars[ key ] )

#        while nmad > 1 and polydeg > 1 :
#            polydeg -= 1
#            polycoeffs, nmad = polyfit_sigclipped( lam, Gaussfits[ key ], 
#                                                  polydeg, verbose=verbose )
#            result = np.polyval( polycoeffs, datadict[ 'wl' ]-lam0 )

        if make_diagnostic_plots :
            plt.figure( 2 ) ; plt.subplot( 4, 2, ki + 1 ) 

            plotbuffer = max( ( 0.1, 3.5 * nmad ) )
            plotmin = result.min() - plotbuffer
            plotmax = result.max() + plotbuffer

            plt.plot( lam+lam0, 
                      np.clip( PSFfitpars[ key ], plotmin, plotmax ), 'rs' )

            plt.plot( wavelengthout, result, 'k-' )
            plt.plot( wavelengthout, result - 3 * np.clip( nmad, 0, 1e2 ), 'k--' )
            plt.plot( wavelengthout, result + 3 * np.clip( nmad, 0, 1e2 ), 'k--' )
            plt.ylabel( key, fontsize='xx-large' )
            plt.xlabel( 'wavelength', fontsize='medium' )
            fitstring = '%s = ' % key
            for pi, coeff in enumerate( polycoeffs[ ::-1 ] ):
                if pi == 0 :   fitstring += '%+.2f ' % coeff
                elif pi == 1 : fitstring += '%+.2e . wl\n ' % coeff
                elif pi >= 2 : fitstring += '%+.2e . wl^%i ' % ( coeff, pi )
            fitstring += '(NMAD = %.2f)' % nmad
            plt.title( fitstring )
            plt.ylim( plotmin-0.5 * nmad, plotmax+0.5 * nmad )

        PSFfitpars[ key ] = result

        hourglass( ki+1, len( keys )-1, verbose=verbose )

    if make_diagnostic_plots :
        plt.figure( 2 ) ;
        plt.draw()

    PSFmodels = build_moffat_fibre_profiles( 
        xfibre, yfibre, wavelength, PSFfitpars, verbose=verbose )

    return PSFmodels

# ____________________________________________________________________________
# ____________________________________________________________________________

def psf_diagnostic_plot( xfibre, yfibre, data, 
                         psfmodel=None, psfparvec=None, 
                         datasize=320, modelsize=1280,
                         simpletitle=False, title=None, 
                         edgecolors='k', lw=None ):

    if title == None :
        title = 'PSF fitting'

    if psfmodel == None or psfparvec != None :
        thefit = SAMI_PSF( psfparvec, xfibre, yfibre )

        thefit = np.reshape( thefit, thefit.shape + (1,) )
        amp, const = covariant_moffat_chi2( psfparvec, xfibre, yfibre, data,
                                            returnamp=True )
        thefit = amp * thefit + const

        xgrid, ygrid = np.indices( (801, 801) )
        xgrid = xgrid.astype( 'f' ) / 50. - 8.
        ygrid = ygrid.astype( 'f' ) / 50. - 8.

        thepsf = SAMI_PSF( psfparvec, xgrid, ygrid, simple=True )
        thepsf = amp * thepsf + const

        if not simpletitle :
            try :
                plt.contour( xgrid, ygrid, thepsf, 12, edgecolors='k', 
                         vmin=thefit.min(), vmax=thefit.max(), lw=lw, 
                         cmap=ifucmap, linewidths=3, zorder=-10 )
            except :
                print 'BADNESS!!!'
                print 'unable to plot the fit!'
#                print thefit
#                print thefit.min(), thefit.max()

    if np.any( np.isfinite( thefit ) ):
        vmin, vmax = thefit.min(), thefit.max()
        vbuffer = 0.01 * ( vmax - vmin )
        vmin, vmax = vmin - vbuffer, vmax + vbuffer
        outedgecolors = ifucmap( ( (vmax - thefit) / (vmax - vmin) ).ravel() )
                                   
        plt.scatter( xfibre, yfibre, modelsize, thefit, 
                     zorder=1, cmap=ifucmap, edgecolors=outedgecolors, lw=lw,
                     vmin=vmin, vmax=vmax )
        thing = plt.scatter( xfibre, yfibre, datasize, data, zorder=4, 
                             cmap=ifucmap, edgecolors=edgecolors, lw=lw,
                             vmin=vmin, vmax=vmax )

    elif np.any( np.isfinite( data ) ) :
        thing = plt.scatter( xfibre, yfibre, datasize, data,
                             zorder=4, cmap=ifucmap, edgecolors=edgecolors )
        plt.scatter( xfibre, yfibre, modelsize, 'r', zorder=1 )

    if simpletitle :
        plt.xticks( np.arange( -8., 8.1 ), [ '' ] * 17 )
        plt.yticks( np.arange( -8., 8.1 ), [ '' ] * 17 )
        plt.title( title, fontsize='small' )
    else :
        plt.title( title )
        plt.xlabel( r'$\Delta$ RA from bundle centre [arcsec]' )
        plt.ylabel( r'$\Delta$ Dec from bundle centre [arcsec]' )
        plt.colorbar( thing )
#    plt.axes().set_aspect('equal')
    plt.xlim( 8, -8 ) ; plt.ylim( -8, 8 )

                         
    

# ____________________________________________________________________________
# ____________________________________________________________________________

def build_moffat_fibre_profiles( xfibre, yfibre, wavelengths, PSFfitpars, 
                                 verbose = 1 ):

    if verbose > 0 :
        print 'Constructing wavelength specific PSF models from Moffat fit results.'

    PSFmodel = np.zeros( xfibre.shape + wavelengths.shape )
    params = zip( PSFfitpars[ 'xcen' ], PSFfitpars[ 'ycen' ],
                  PSFfitpars[ 'alphax' ], PSFfitpars[ 'alphay' ],
                  PSFfitpars[ 'beta' ], PSFfitpars[ 'rho' ] )

    for wli, parvec in enumerate( params ):
        PSFmodel[ :, wli ] = SAMI_PSF( parvec, xfibre, yfibre )
        if verbose > 0 :
            hourglass( wli+1, wavelengths.shape[0] )

    return PSFmodel

# ____________________________________________________________________________
# ____________________________________________________________________________

def extract_total_flux( datadict, gaussfit, verbose=True, sigclip=10. ):

    thedata, thevar, wl = datadict[ 'data' ], datadict[  'var'  ], datadict[ 'wl' ]
    xfibre, yfibre = relative_fibre_positions( datadict )

    if verbose > 0 :
        print 'Fitting for Gaussian amplitude at each wavelegth (knowing the PSF shape).'
    
    okaytouse = ( np.isfinite( thedata ) )  & ( np.isfinite( thedata ) )
                  

    flxsum = np.zeros( thedata.shape[1] )
    varsum = np.zeros( thedata.shape[1] )
    amp = np.zeros( thedata.shape[1] ) * np.nan
    unc = np.zeros( thedata.shape[1] ) * np.nan
    
    timeoff = 0.
    startTime = time.time()
    for wli, ( data, var, psfweight, okay ) in enumerate( 
            zip( thedata.T, thevar.T, gaussfit.T, okaytouse.T ) ):
      
      sumover = np.where( okaytouse[ :, wli ] )
      if len( sumover[0] ) and np.any( np.isfinite( psfweight ) ):

#        print parvec+(0., 1.)
#        psfweight = Gauss2D( parvec + (1.,), xfibre, yfibre )
#        print psfweight
        # plt.figure( 1 ) ; plt.clf()
        # plt.scatter( xfibre, yfibre, 720, psfweight )
        # plt.scatter( xfibre, yfibre, 180, data[ :, wli ] )
        # plt.draw()

        offsetguess = 0. # np.median( data[sumover] )
        scaling = np.sum( psfweight[ sumover ] )
        amplitudeguess = np.sum( data[sumover] - offsetguess ) / scaling
        firstguess = [ max( amplitudeguess, 1 ), offsetguess ]

        args = ( psfweight[sumover], data[sumover], var[sumover], sigclip )
        ( amplitude, constant ) = fmin( 
            amplitude_chi2_sigclipped, [amplitudeguess, offsetguess], 
            args=args, disp=False )

        thefit = amplitude * psfweight + constant
        count = ( ( ( data - thefit )**2. / var < sigclip**2. ) & ( okay ) )
                            
        signal2noise = np.sqrt( np.sum( 
                ( ( thefit - constant )**2. / var )[ count ] ) )

        if count.sum() < 25 :
            amplitude = np.nan
                                
        amp[ wli ] = amplitude
        unc[ wli ] = amplitude / signal2noise

        if ( make_diagnostic_plots and not at_the_telescope 
                 and ( wli % 64 ) == 32 ):
            plt.figure( 1 ) ; plt.clf()

            if np.any( np.isfinite( thefit ) ):
                vmin, vmax = np.nanmin( thefit ), np.nanmax( thefit )
                vrange = vmax - vmin
                vmin -= 0.05 * vrange
                vmax += 0.15 * vrange

                plt.scatter( xfibre, yfibre, 1020, thefit,
                             vmin=vmin, vmax=vmax, edgecolors='k', 
                             cmap=ifucmap, zorder=1 )

                xgrid, ygrid = np.indices( (801, 801) )
                xgrid = xgrid.astype( 'f' ) / 50. - 8.
                ygrid = ygrid.astype( 'f' ) / 50. - 8.

            else : 
                vmin, vmax = None, None
                plt.scatter( xfibre, yfibre, 1020, 'r', zorder=1 )

            if count.sum() :
                thing = plt.scatter( xfibre[ count ], yfibre[ count ], 255, 
                                 data[count], cmap=ifucmap, 
                                 vmin=vmin, vmax=vmax, edgecolors='k', 
                                 zorder=4 )
                plt.colorbar( thing )

            dontcount = count == False
            if dontcount.sum() > 0 :
                plt.scatter( xfibre[dontcount], yfibre[dontcount], 255, 
                             'r', zorder=4 )
                    
            plt.axes().set_aspect('equal' )
            plt.xlim( -8, 8 ) ; plt.ylim( -8, 8 )
            plt.draw()

      hourglass( wli+1, thedata.shape[1], verbose=verbose )

    return amp, unc

# ____________________________________________________________________________
# ____________________________________________________________________________

def amplitude_chi2_sigclipped( params, fitvals, obsvals, obsvar, 
                    sigclip=12. ):

    [ amplitude, constant ] = params
    chi2 = ( amplitude * fitvals + constant - obsvals )**2. / obsvar
    chi2 = np.clip( chi2, 0., sigclip**2. )
    
    return np.sum( chi2 )

#
# ____________________________________________________________________________
# ____________________________________________________________________________

def extract_total_flux_new( datadict, gaussweight,
                            sigclip=5., verbose=True,  ):

    data, var, wl = datadict[ 'data' ], datadict[  'var'  ], datadict[ 'wl' ]
    xfibre, yfibre = relative_fibre_positions( datadict )

    if 0 :
        if verbose > 0 :
            print 'Median filtering to get empirical variance estimates.'

            empvar = np.zeros( var.shape )
            for i in range( empvar.shape[0] ):
                median, nmad = median_smooth( data[ i, : ], 20, givenmad=True )
                empvar[ i, : ] = nmad**2.
                if verbose > 0 :
                    hourglass( i+1, empvar.shape[0] )
            empvar = empvar + var
    else :
        sigclip = 5.
        empvar = var

    if verbose > 0 :
        print 'Iterative-progressive fitting for total flux (knowing the PSF shape).'

    badness = np.zeros( data.shape )
    
    use = ( ( np.isfinite( data ) ) & ( np.isfinite( var ) ) ) 
    dead = ( use == False )
    lastamplitude, constant = None, 0.
    maxchange, maxcount = 1., 100
    count, clipshift = 0, 12
    chi2 = np.zeros( data.shape )

    use = ( gaussweight > 10.**( -1.8 ) ) & ( dead == False )

    # first guesses
    amplitude = ( np.median( ( gaussweight * data )[ use ] ) 
                  / np.median( ( gaussweight**2. )[ use ] ) )
    constant = np.median( ( data - amplitude*gaussweight )[ use ] )
    slop = constant**2. + ( 0.015 * amplitude*gaussweight )**2. 

    bestamp = np.nan * np.ones( data.shape[1] )
    bestcon = np.nan * np.ones( data.shape[1] )
    maxlike = np.ones( data.shape[1] ) - np.inf

    while ( maxchange > 1e-3 and count < maxcount ) or count < 20 :

        numerator = use * gaussweight * (data-constant) / ( var+slop )
        denominator = use * gaussweight**2. / ( var+slop )

        amplitude = ( np.nansum( numerator, axis=0 ) 
                      / np.nansum( denominator, axis=0 ) )
            
        constant = ( 
            np.nansum( (data-(amplitude*gaussweight)) * use / var, axis=0 ) \
                / np.nansum( use / var, axis=0 ) )

        thefit = amplitude * gaussweight + constant
        delta = thefit - data 
        slop = constant**2. + ( 0.015 * amplitude*gaussweight )**2. 
        chi2 = delta**2. / ( empvar + slop ) 

        clipshift = np.clip( 12 - 2.5 *  count, 0., 12. )
        use = ( ( chi2 < ( sigclip + clipshift )**2. )
                & ( gaussweight > 10.**(-1.5-1.8*np.min([count/10.,np.inf]) ))
                & ( dead == False ) )

        loglike = np.nansum( -0.5 * np.clip( chi2, 0., (sigclip+clipshift)**2. )
                              - 0.5 * np.log( empvar + slop ), axis=0 )
        better = np.where( loglike > maxlike )[0]

        bestamp[ better ] = amplitude[ better ]
        bestcon[ better ] = constant[ better ]
        maxlike[ better ] = loglike[ better ]

        if 1 :
            badness += ( 1 - use )
            plt.figure( 20 ) ; plt.clf()
            plt.imshow( badness, origin='lower', aspect='auto' )
            plt.colorbar()
            plt.draw()

        signal2noise = np.sqrt(np.nansum(
                use * ( bestamp * gaussweight )**2. / var, axis=0))
        uncertainty = np.abs( bestamp / signal2noise )

        if lastamplitude != None :
          change = np.abs( 1. - amplitude / lastamplitude )
          changed = ( change > 0 ) & ( amplitude > 0. ) 

          if np.sum( changed ) :
            maxchange = np.nanmax( change[ changed ] )
            changesum = np.nansum( change[ changed ] )

          else :
              maxchange = 0.

        else :
            changed = np.isfinite( amplitude )

        lastamplitude = amplitude
        if better.shape[0] == 0 :
            maxchange = 0.


        if i_am_ned :
          plt.figure( 17 ) ; plt.clf()
          plt.plot( wl, amplitude, 'k-' )
          plt.scatter(wl[changed], amplitude[changed], 8, 'r', edgecolors='none')
          plt.fill_between( wl, ( amplitude-3.*uncertainty ), 
                            ( amplitude + 3.*uncertainty ), 
                            color='k', alpha=0.3 )
          plt.plot( wl, bestamp + np.median( amplitude )/2., 'k-' )
          plt.scatter( wl[better], bestamp[better] + np.median( amplitude )/2., 
                       8, 'r', edgecolors='none' )

          plt.plot( wl, bestcon, 'k-' )
          plt.scatter( wl[better], bestcon[better], 8, 'r', edgecolors='none' )
          
          sortedvals = np.sort( amplitude[ np.isfinite( amplitude ) ] )
          ymax = np.interp( 0.92, np.linspace( 0., 1., sortedvals.shape[0] ), 
                            sortedvals )
          plt.xlim( wl.min(), wl.max() )
          plt.ylim( min( 1.2 * bestcon[ np.isfinite( bestcon ) ].min(), 
                         -0.08 * ymax ), 1.52 * ymax )
          plt.title( np.sum( use ) )
          plt.draw()

        count += 1
        if verbose > 0 :
            pinwheel( count, maxcount, verbose=verbose+1 )
    if verbose > 0 :
        print 'Done after  '
    print 'Final total flux extraction based on {:8.3%} of available spexels'.format( float( use.sum() ) / float( data.shape[0]*data.shape[1] - dead.sum() ) )

    return np.clip( bestamp, 0., bestamp ), uncertainty
# ____________________________________________________________________________
# ____________________________________________________________________________

def fit_covariant_moffat( xfibre, yfibre, data, var=None, guess=None, 
                          sigclip=12, makeplot=False, verbose=1 ):

#    makeplot=True
    if var == None:
        var = 1.
    
    # initial guess for the centre of the PSF
    weighted = ( data / var ).T
    weighted = np.clip( weighted, 0., weighted )
    weighted /= np.nansum( weighted )

    if guess == None :
        xmean = np.nansum( xfibre * weighted ) 
        ymean = np.nansum( yfibre * weighted ) 
        alpha, beta = 2.5, 4.0
        guess = [ xmean, ymean, alpha, alpha, beta, 0.0 ]

    fitpars = fmin( covariant_moffat_chi2, guess,
                    args=[ xfibre, yfibre, data, var, 
                           sigclip, False, makeplot ], disp=False )
                    
    psfmodel = SAMI_PSF( fitpars, xfibre, yfibre )

    return fitpars, psfmodel.sum()

# ____________________________________________________________________________
# ____________________________________________________________________________

def covariant_moffat_chi2( parvec, xfibre, yfibre, data, var=None, 
                           sigclip=6, simple=False, makeplot=False, 
                           returnamp=False, verbose=0 ):

    if var == None :
        var = 1.

    if len( parvec ) == 6 :
        [ xcenter, ycenter, alphax, alphay, beta, rho ] = parvec
    elif len( parvec ) == 4 :
        [ xcenter, ycenter, alphax, beta ] = parvec

    if np.sqrt( xcenter**2. + ycenter**2. ) > 6.5 or beta < 1. \
            or np.abs( np.log10( alphax / alphay ) ) > 0.5 :
        if returnamp :
            return np.nan, np.nan
        else :
            return np.inf

    psfmodel = SAMI_PSF( parvec, xfibre, yfibre, simple=simple )

    constant=0.
    lastchi2, change, count = None, 1., 0
    okay = np.isfinite( data )
    use = okay[ :: ]
    while change > 1e-3 and count < 20 :
        
        amplitude = abs(np.nansum( use *psfmodel*(data -constant)/var, axis=0 )
                     / np.nansum( use * psfmodel*psfmodel/var, axis=0 ) )

        constant = (np.nansum( use*(data - amplitude*psfmodel)/ var, axis=0 )
                     / np.sum( use/var, axis=0 ) )
 
        delta = amplitude * psfmodel + constant - data
        slop = ( 0.01 * amplitude * psfmodel )**2. + constant**2.
        chi2 = delta**2. / ( var + slop ) + np.log( var + slop )
#        print data
#        print chi2
        use = (chi2 < sigclip**2.) & (okay)
        if use.sum() < 12 :
            use = chi2 > np.sort( chi2[ okay ] )[ 12 ]
#        if use.sum() < 8 :
#            use =  data > np.sort( data )[ -8 ]
        chi2 = np.where( use & okay, chi2, sigclip**2. )
        chi2 = chi2.sum()

        if lastchi2 != None :
            change = np.abs( lastchi2 - chi2 )
        lastchi2 = chi2
        count += 1

#    print parvec, chi2
#        print count, use.sum(), amplitude, constant, chi2, change, '\n', ;sys.stdout.flush()
#    print count, use.sum(), amplitude, constant, chi2, change, '\r', ;sys.stdout.flush() 

    if verbose > 3 :
        printstring = ''
        for par in parvec :
            printstring += '%+8.3f' % par
        printstring += '%10.1f ' % chi2
        printstring += ' %3i\r' % count
        sys.stdout.write( printstring )
        sys.stdout.flush()


    if returnamp :
        return amplitude, constant

    if makeplot :
        plt.figure( 72 ) ; plt.clf()
        thefit = amplitude * psfmodel
        vmax = 1.05 * thefit.max()
        if vmax > 0 :
            plt.scatter( xfibre, yfibre, 1020, data - constant, 
                         vmin=0., vmax=vmax, cmap=ifucmap )

            plt.scatter( xfibre, yfibre, 255, thefit, 
                         vmin=0., vmax=vmax, cmap=ifucmap )
        plt.colorbar()
        plt.draw() 
        time.sleep( 1 )

        print parvec, chi2

    return chi2


# ____________________________________________________________________________
# ____________________________________________________________________________

def covariant_moffat( parvec, xfiber, yfiber ):

    if len( parvec ) == 6 :
        [ xcenter, ycenter, alphax, alphay, beta, rho ] = parvec
    elif len( parvec ) == 4 :
        [ xcenter, ycenter, alphax, beta ] = parvec
        alphay, rho = alphax, 0.

    if alphax < 0 or alphay < 0 or beta < 0 or abs( rho ) > 1 :
        return np.inf * np.ones( xfiber.shape )
    
    xterm = ( xfiber - xcenter ) / alphax
    yterm = ( yfiber - ycenter ) / alphay

    return ( ( beta -  1. ) / ( np.pi * alphax * alphay * np.sqrt(1.-rho**2.) )
            * ( 1. + ( xterm * xterm + yterm * yterm
                        - 2. * rho * xterm * yterm ) / (1.-rho**2.)
                )**( -beta ) )


#    return ( beta-1 ) / ( np.pi * alphax**2. ) \
#        * ( 1. + ( x**2. + y**2. ) / alphax**2. )**( -beta )

#    return ( ( beta-1. ) / ( np.pi*alphax*alphay*np.sqrt( 1.-rho**2. ) )
#            * ( 1. + ( x**2. / alphax**2. + y**2. / alphay**2.
#                       - 2. * rho * x * y / alphax / alphay ) / (1.-rho**2. )
#                )**( -beta ) )

# ____________________________________________________________________________
# ____________________________________________________________________________

def SAMI_PSF( parvec, xvals, yvals, ninner=6, nrings=10, simple=False ):

    result = covariant_moffat( parvec, xvals, yvals )
    if simple :
        return result * ( np.pi * fibre_radius_arcsec**2. )

    if len( parvec ) == 6 :
        [ x0, y0, beta, alphax, alphay, rho ] = parvec
    else :
        [ x0, y0, beta, alpha ] = parvec
        alphax, alphay, rho = alpha, alpha, 0.
#    closein = np.where( ( (xvals-x0)**2. / alphax**2. 
#                          + (yvals-y0)**2. / alphay**2. ) < 15**2. )

    closein = np.where( np.sqrt( ( xvals-x0 )**2. + ( yvals-y0 )**2. ) < 5 )
                          
    radii = np.arange( 0., nrings ) + 0.5
    rotang = 0.
    for ri, ringrad in enumerate( radii ):
        npoints = ninner*ringrad
        tpoints = np.linspace( 0., 2.*np.pi, npoints+1 ) + rotang
        rotang += tpoints[1] / 2.
        if ri == 0 :
            radius = np.ones(npoints)*ringrad
            theta =  tpoints[ :-1 ]
        else :
            radius = np.hstack( ( radius, np.ones(npoints)*ringrad) )
            theta  = np.hstack( ( theta , tpoints[ :-1 ] ) )

    radius *= fibre_radius_arcsec / nrings

    xsub = radius * np.cos( theta )
    ysub = radius * np.sin( theta )

    for i in closein[ 0 ] :
        x, y = xvals[ i ], yvals[ i ]

        values = covariant_moffat( parvec, x+xsub, y+ysub )
        result[ i ] = values.mean() 

#    print parvec, result.sum() * ( np.pi * fibre_radius_arcsec**2. )
     
    return result * ( np.pi * fibre_radius_arcsec**2. )

# ____________________________________________________________________________
# ____________________________________________________________________________

def polyfit_sigclipped( xvals, yvals, polydeg=1,
                        sigclip=4, tolerance=.03, verbose=False ):

    keep = np.isfinite( yvals )

    lastcoeffs = np.polyfit( xvals[ keep ], yvals[ keep ], polydeg-1 )
    lastcoeffs = np.hstack( ( lastcoeffs, ( 0. ) ) )
    yfit = np.polyval( lastcoeffs, xvals )
    delta = ( yvals - yfit )
    scatter = 1.4826 * np.median( np.abs( delta - np.median( delta ) ) )
    
    count = 1
    gogo = True
    while gogo :
        cliplimit = np.max( ( sigclip*scatter, tolerance ) )
        keep = ( np.abs(delta) < cliplimit ) & np.isfinite( yvals )

        if keep.sum() < 5 :
            print '\n' * 5
            print 'badness - not enough points left.  How does this happen?!'
            print '\n' * 5
            gogo = False

            coeffs = np.hstack( ( np.median(yvals), np.zeros( polydeg-1 ) ) )
            yfit = np.polyval( lastcoeffs, xvals )
            delta = ( yvals - yfit )
            scatter = 1.4826 * np.median( np.abs( delta - np.median( delta ) ) )
            return coeffs, scatter


        coeffs = np.polyfit( xvals[ keep ], yvals[ keep ], polydeg )
        yfit = np.polyval( coeffs, xvals )
        delta = ( yvals - yfit )
        scatter = 1.4826 * np.median( np.abs( delta[keep] 
                                    - np.median( delta[keep] ) ) )
        count += 1 

        if verbose > 1 :
            print count, keep.sum(), coeffs, scatter

        if np.max( np.abs( lastcoeffs / coeffs - 1. ) < 1e-4 ) :
            gogo = False
            return coeffs, scatter
            
        if count >= 100 :
            print '\n' * 5
            print 'badness - too many iterations. Oscillating solution?'
            print '\n' * 5
            gogo = False
            return ( coeffs + lastcoeffs ) / 2., scatter

        lastcoeffs = coeffs

# ____________________________________________________________________________
# ____________________________________________________________________________


def fit_for_DAR_corrections( psffitparams, paramkeys, wavelength, datadict,
                             polydeg=1, sigclip=3.5, 
                             verbose=1, make_diagnostic_plots=True ):
    
    if verbose > 0 :
        print 'Making polynomial fits to get PSF shape afo wavelength.'

    xcen = np.array( psffitparams[ 'xcen' ] )
    ycen = np.array( psffitparams[ 'ycen' ] )
    lam = np.array( psffitparams[ 'lam' ] )
    lam0 = np.mean( lam )
    lam -= lam0

    rad = np.sqrt( xcen**2. + ycen**2. )
    maxrad = 6.
    useme = rad < maxrad
    ninrad = useme.sum()
    count, changed, lastused = 0, True, useme.copy()
    theta = np.arctan( np.polyfit( xcen[ useme ], ycen[ useme ], 1 )[ 1 ] )
    darfit = [ np.median(xcen[useme]), np.median(ycen[useme]), theta, 1., 0. ]
    while count < 12 and changed :
        
        darfit = fmin( psf_position_chi2, darfit, 
                       args=[ xcen, ycen, (lam+lam0)/lam0, useme, sigclip ], 
                       disp=False )
        [ x0, y0, theta, drdl, coeff ] = darfit

        xfit, yfit = psf_position_chi2( darfit, xcen, ycen, (lam+lam0)/lam0, returnfits=True )
                                    
        delrad = np.sqrt( ( xcen-xfit )**2. + ( ycen-yfit )**2. )
        med = np.median( delrad )
        nmad = 1.486 * np.median( np.abs( delrad - med ) )
        useme = ( np.abs( delrad ) < min( ( sigclip * nmad ), 0.3 ) ) \
            & ( rad < maxrad )
        if useme.sum() < 5 and ninrad > 3 :
            check = np.abs( delrad )
            values = np.sort( check[ rad < maxrad ] )
            value = values[ min(ninrad-1,5) ]
            useme = ( check < value )

        if useme.sum() < 3 or ninrad <= 3 :
            darfit = [ 0., 0., 0., 0., 0. ]
            useme = np.isfinite( xcen )
            count = np.inf
            change = False

        exclude = np.where( useme == False )[0]

        if 1 :
#            print count, med, nmad, useme.sum()
#            print lastused
#            print useme
#            print delrad
#            print
        
            plt.figure( 53 ) ; plt.clf()
            lcol = plt.cm.jet( ( lam - lam.min() ) / ( lam.max() - lam.min() ) )

            cfr = analytic_DAR( lam+lam0, datadict[ 'ZDSTART' ], wl0=lam0 )
            cfx = x0 + np.cos( -theta ) * cfr
            cfy = y0 - np.sin( -theta ) * cfr

            plt.scatter( xcen, ycen, 40, lcol )
            plt.plot( xfit, yfit, 'k-' )
            plt.plot( cfx, cfy, 'r--' )

            for i in range( len( xcen ) ) : 
                plt.plot( [ xcen[ i ], xfit[ i ] ], [ ycen[ i ], yfit[ i ] ], '-', 
                          color=lcol[i] )
            plt.scatter( xfit, yfit, 80, lam, zorder=-1,
                         facecolors='none', edgecolors=lcol, lw=2 )
            if len( exclude ) :
                plt.scatter( xcen[ exclude ], ycen[ exclude ], 240, 'r', 
                     marker='x', lw=2 )
            plt.xlim( xfit.mean() - .5, xfit.mean() + .5 )
            plt.ylim( yfit.mean() - .5, yfit.mean() + .5 )
            plt.draw()
#            time.sleep( 1 )

        if lastused != None :
            changed = np.any( lastused != useme )
        lastused = useme.copy()
        count += 1

    datadict[ 'DAR_FIT_POSANG' ] = theta * 180. / np.pi
    datadict[ 'DAR_FIT_ACOEFF' ] = coeff
    datadict[ 'DAR_FIT_LAM0' ] = lam0
    datadict[ 'DAR_FIT_PCOEFF' ] = drdl


    xfit, yfit = psf_position_chi2( darfit, xcen, ycen, wavelength/lam0, 
                                    returnfits=True )

    if make_diagnostic_plots:
        plt.figure( 2, figsize=(7.5, 12) ) ; plt.clf()
        plt.subplots_adjust( top=0.95, bottom=0.05, left=0.105, right=0.965, 
                             hspace=.585, wspace=.285  )

    for ki, key in enumerate( paramkeys[ :-1 ] ):
        if verbose > 0 :
            hourglass( ki, len( paramkeys )-1, verbose=verbose )
        
        if key == 'xcen' :
            result = xfit
        elif key == 'ycen' :
            result = yfit
        elif key == 'flux' :
            result = np.interp( wavelength-lam0, lam, psffitparams[ key ] )
        else :
            psffitparams[ key ] = np.array( psffitparams[ key ] )

            if polydeg >= 1 :
                polycoeffs, nmad = polyfit_sigclipped( 
                    lam[ useme ], psffitparams[ key ][ useme ], 
                    polydeg, verbose=verbose )

                result = np.polyval( polycoeffs, wavelength-lam0 )
            else :
                polycoeffs = [ np.mean( psffitparams[ key ][ useme ] ) ]

                count, change, lastone = 0, True, useme[ :: ]
                while count < 10 and change :
                    med = np.median( psffitparams[ key ] )
                    nmad = 1.486*np.median( np.abs(psffitparams[ key ] - med ))

                    use = ( ( abs( psffitparams[ key ] - med ) 
                              < (sigclip * nmad) ) & useme )

                    if lastone != None :
                        change = np.any( use != lastone )
                    lastone = use[ :: ]
                    count += 1
                    result = np.mean( psffitparams[ key ][ use ] )

                result = ( result * np.ones( wavelength.shape ) )

        if make_diagnostic_plots :
            if key.startswith( 'sec' ):
                plt.figure( 2 ) ; plt.subplot( 4, 2, ki + 2 ) 
            else :
                plt.figure( 2 ) ; plt.subplot( 4, 2, ki + 1 ) 

            plotbuffer = max( ( 0.1, 3.5 * nmad ) )
            plotmin = result.min() - plotbuffer
            plotmax = result.max() + plotbuffer

            toplot = np.array( psffitparams[ key ] )
            toplot = np.clip( toplot, plotmin, plotmax )
            plt.plot( ( lam+lam0 )[exclude], toplot[ exclude ], 'rx', lw=1.3 )
            plt.plot( ( lam+lam0 )[useme], toplot[ useme ], 'bs' )

            if key == 'xcen' :
                plt.plot( lam+lam0, cfx, 'r--' )
            if key == 'ycen' :
                plt.plot( lam+lam0, cfy, 'r--' )

            plt.plot( wavelength, result, 'k-' )
            if key != 'flux' :
                plt.plot( wavelength, result - 3 * np.clip( nmad, 0, 1e2 ), 'k--' )
                plt.plot( wavelength, result + 3 * np.clip( nmad, 0, 1e2 ), 'k--' )

            plt.ylabel( key, fontsize='xx-large' )
            plt.xlabel( 'wavelength', fontsize='medium' )
            fitstring = '%s = ' % key
            if not key in 'xcen ycen flux'.split() :
              for pi, coeff in enumerate( polycoeffs[ ::-1 ] ):
                if pi == 0 :   fitstring += '%+.2f ' % coeff
                elif pi == 1 : fitstring += '%+.2e . wl\n ' % coeff
                elif pi >= 2 : fitstring += '%+.2e . wl^%i ' % ( coeff, pi )
              fitstring += '(NMAD = %.2f)' % nmad
            plt.title( fitstring )
            plt.ylim( plotmin-0.5 * nmad, plotmax+0.5 * nmad )

        psffitparams[ key ] = result

        hourglass( ki+1, len( paramkeys )-1, verbose=verbose )

    if make_diagnostic_plots:
        plt.figure( 2 )
        plt.draw()

    return psffitparams


# ____________________________________________________________________________
# ____________________________________________________________________________

def psf_position_chi2( parvec, xcen, ycen, lam,
                       useme=None, sigclip=5., returnfits=False ):
    [ x0, y0, theta, drdl, coeff ] = parvec

    rpos = coeff * lam**drdl
    xpos = x0 + np.cos( theta ) * rpos 
    ypos = y0 + np.sin( theta ) * rpos
    
    if returnfits :
        return xpos, ypos

    dx, dy = xcen - xpos, ycen - ypos

    if 0 :
        delrad = np.sqrt( dx**2. + dy**2. )
        med = np.median( delrad )
        nmad = 1.486 * np.median( np.abs( delrad - med ) )
#    useme = ( np.abs( delrad ) < min( ( sigclip * nmad ), 0.3 ) )
        chi2 = delrad**2. / nmad**2.
        chi2 = np.clip( chi2, 0., sigclip**2. )
    else :
        chi2 = dx**2. + dy**2.

    if useme != None :
        chi2 = np.where( useme, chi2, sigclip**2. )

    if np.random.random() < .01 :
        plt.figure( 54 ) ; plt.clf()
        plt.scatter( xcen, ycen, 80, np.linspace( 0.,1., lam.shape[0] ) )
        plt.scatter( xpos, ypos, 80, np.linspace( 0.,1., lam.shape[0] ), marker='x', lw=4 )
        plt.draw()

    return np.sum( chi2 )
    
# ____________________________________________________________________________
# ____________________________________________________________________________

def find_standard_star_dataframes( directory, 
    path_to_ESO_standards=path_to_standard_spectra, 
    verbose=True ):

    if verbose >= 1 :
        print 'Looking for SAMI spectrophotometric standard star observations.'
    if not os.path.exists( directory ) :
        if verbose > 0 :
            print '\n\n\nWARNING: path %s does not exist.' % directory
            print 'Cannot look for standard stars in a non existent directory!'
            print '\n\n\n'
        return None

    standards = os.listdir( directory )

    standards_list = []
    for standard in standards :
      fitsfilename = '%s/%s' % ( directory, standard )
      if fitsfilename.endswith( 'red.fits' ) :
        # find which bundle contains the standard, and which standard it is.
        try:
            result = find_standard_in_dataframe( 
                fitsfilename, directory_list=path_to_ESO_standards, verbose=verbose )
        except IOError:
            if verbose >= 1:
                print 'Nothing found in that file'
            continue
            
        if result != None :
            standards_list.append( result )
                        
    if verbose >= 1 :
        print '\nFound %i spectrophotometric standards in %s.' % ( 
            len( standards_list ), directory )

    return standards_list
# ____________________________________________________________________________
# ____________________________________________________________________________

def find_standard_in_dataframe( fitsfilename, max_sep_arcsec=30., 
                                directory_list=path_to_standard_spectra,
                                tablename_list=standard_star_catalog,
                                verbose=False ):
    """
Identifies and extracts the IFS data for a standard star, given a dataframe.
Standard stars are identified by position using find_standard_spectrum.
SAMI data are extracted using SAMI_utils.IFU_pick.
Returns the SAMI data as well as the rebinned standard star spectrum.

"""
    if verbose > 1 :
        print """\n
Starting SAMI_fluxcal.find_standard_in_dataframe to get SAMI IFS data
and reference spectrum for a standard star.

Opening data fits file: %s\n""" % ( fitsfilename )
    elif verbose > 0 :
        print 'Searching for standard star in', fitsfilename.split('/')[-1]

    fitsfile = pf.open( fitsfilename )
    hdr=fitsfile[0].header
    table = fitsfile[2].data
    probenames = np.unique( table.field( 'PROBENAME' ) )

    nfound = 0
    for probename in probenames :
        if not probename.count( 'SKY' ):
            probe = np.where( table.field( 'PROBENAME' ) == probename )
            RA = np.mean( table.field( 'FIB_MRA' )[ probe ] )
            DEC= np.mean( table.field( 'FIB_MDEC' )[ probe ] )        
            if RA != 0. or DEC != 0. :
                if verbose > 1:
                    print 'Looking around bundle %s (RA=%.6f, Dec=%.6f) ...' % (
                        probename, RA, DEC ),   ;   sys.stdout.flush()

                starfile, starname, offset=find_standard_spectrum( 
                    RA, DEC, max_sep_arcsec=max_sep_arcsec,
                    directory_list=directory_list, verbose=verbose-2>0 )

                if starname != None :
                    nfound += 1
                    standards_list= [ fitsfilename, probename, 
                                             starfile, starname, offset ]
                    if verbose > 1:
                        print 'matches ', starname, '.'
                elif verbose > 1 :
                    print 'no match (closest at %.2f arcsec)' % offset

    if nfound > 1 and verbose >= 0 :
        print '\n\n\nWARNING from SAMI_fluxcal.find_standard_in_dataframe:'
        print 'Found %i possible standard stars in dataframe %s' % (
                    nfound, path_to_data, fitsfilename )
        print 'Some code revision is needed to accommodate this scenario.'
        print 'Only returning the last match considered.\n\n\n'

    if nfound < 1 : 
        if verbose >= 0 :    
            print '\n\n\nWARNING from SAMI_fluxcal.find_standard_in_dataframe:'
            print 'No standard stars found in dataframe:'
            print '       '+fitsfilename
            print 'Data frame header supplied MEANRA = %.6f, MEANDEC = %.6f.' % ( 
                hdr[ 'MEANRA' ], hdr[ 'MEANDEC' ] )
            print 'Nearest match is %.2f arcsec away (max radius is %2.f arcsec).' % ( offset, max_sep_arcsec )
            print "Maybe you need me to (be able to) download more spectrophot'c standard spectra?"
            print 'Not returning anything useful.\n\n\n'
        return None

    if verbose > 0 :
        print 'Found standard spectrum for star %s in probe %s.' % (
            standards_list[2], standards_list[1] )
        print 'Offset b/w bundle centre and known position is %.2f arcsec.' % (
            standards_list[ 4 ] )
    if verbose > 1 :
        print "\nFinished SAMI_fluxcal.find_standard_in_dataframe.\n"

    return standards_list

# ____________________________________________________________________________
# ____________________________________________________________________________

def find_standard_spectrum(ra,dec, max_sep_arcsec=15., 
    directory_list=path_to_standard_spectra, tablename_list=standard_star_catalog,
    status=None, infield=False, doplot=False, verbose=False ):
    """
Get the right calibration spectrum, based on the RA and Dec of the star. 
Returns the true spectrum of the standard star and wavelength array, as well 
as the name of the standard star, and the distance from the search position.

spectrum, wavelength, identifier, separation =
find_standard_spectrum( RA, Dec, max_sep_arcsec=1
                        directory='./ESOstandards/', tablename='ESOstandards.dat'
                        status=None, infield=False, doplot=False, verbose=False )

    RA, Dec          coordinates in decimal degrees

    max_step_arcsec  maximum separation for matching, in units of arcsec
    directory        the path to where the spectra are located
    tablename        the path to where the standards table is located
    status           not actually used
    url              the url for where the ESO standards list can be found
    verbose          Boolean controlling verbose outputs

"""
    # Find the best match in RA and Dec
    if verbose > 0:
        print 'Matching to observed coordinates RA =', ra, 'Dec =', dec

    min_sep = float('inf')

    for directory, tablename in zip(directory_list, tablename_list):
        standardcat = '%s/%s' % ( directory, tablename )

        # Read the index file
        if verbose > 0:
            print """
Starting SAMI_fluxcal.find_standard_spectrum to get standard star spectrum.
 
Looking for flux calibration data in %s .
Reading coordinates of standard stars from %s .
Maximum search radius is %.2f arcsec.
""" % ( directory, standardcat, max_sep_arcsec )

        if not os.path.exists( standardcat ) :
            print """\n\n\n\n\n
Cannot find file %s containing standard star info.
You may need to run SAMI_fluxcal.create_ESO_standards_table .
You may also need to run SAMI_fluxcal.get_ESO_standard_spectra .

Dying gracelessly in 3 seconds ...\n\n\n""" % standardcat 
            time.sleep( 3 )
        
        index = np.loadtxt( standardcat, dtype='S' )

        for star in index:
            RAstring = '%sh%sm%ss' % ( star[2], star[3], star[4] )
            Decstring= '%sd%sm%ss' % ( star[5], star[6], star[7] )

            coords_star = coord.ICRSCoordinates( RAstring, Decstring )
            ra_star = coords_star.ra.degrees
            dec_star= coords_star.dec.degrees

            ### BUG IN ASTROPY.COORDINATES ###
            if '-' in star[5] and dec_star > 0:
              dec_star *= -1
              if -1. <= dec and dec <= 0 :
                print '    fixing negative Dec bug in astropy.coordinates.'
                print '    astropy.coordinates treats Dec of -00 as +00'
                print '    future astropy.coordinates may fix this problem.'
                print '\n' * 2
                print "    i'll give you 10 seconds to think about that."
                print '\n' * 5
                for i in range( 5 ):
                    print '\a'
                    time.sleep( 0.3 )
                time.sleep( 8.5 )

            sep = coord.angles.AngularSeparation(
                    ra, dec, ra_star, dec_star, units.degree ).arcsecs

            if sep < min_sep:
                min_sep = sep
                closest_star = star
                closest_dir = directory

    # Check that the closest match is close enough
    if min_sep > max_sep_arcsec :
        if verbose > 0 :        
            print 'WARNING from SAMI_fluxcal.find_standard_spectrum:'
            print '    Closest standard star is too far away. (%.3f arcsec)' % (min_sep)
        return None, None, min_sep

    if verbose > 1:
        print 'Closest standard star found is', closest_star[1], \
              'at separation %.3f arcsec.' % min_sep
        print min_sep, max_sep_arcsec, min_sep > max_sep_arcsec

    return os.path.join(closest_dir, closest_star[0]), closest_star[1], min_sep
    
# ____________________________________________________________________________
# ____________________________________________________________________________

def get_standard_spectrum( standard_filename, 
                            verbose=False ):

    # Read the flux and wavelength from the correct file

    if not os.path.exists( standard_filename ) and verbose >= 0 :
        print """\n\n\n\n\n
Cannot find file %s containing spectrum for standard star.
You may need to run SAMI_fluxcal.get_ESO_standard_spectra .

Dying gracelessly in 3 seconds ...\n\n\n""" % ( standard_filename )
        time.sleep( 3 )

    if verbose > 1:
        print 'Reading spectrum from', standard_filename
    
    skiprows = 0
    with open(standard_filename) as f_spec:
        finished = False
        while not finished:
            line = f_spec.readline()
            try:
                number = float(line.split()[0])
            except ValueError:
                skiprows += 1
                continue
            else:
                finished = True

    star_data = np.loadtxt(standard_filename, dtype='d', skiprows=skiprows)
    wavelength= star_data[:,0]
    spectrum  = star_data[:,1]

    if verbose > 1 :
        print 'Spectrum ranges from %.1f to %.1f Angstrom.' % (
                    wavelength.min(), wavelength.max() )
        print "\nFinished SAMI_fluxcal.find_standard_spectrum.\n" ; print

    return spectrum, wavelength

# ____________________________________________________________________________
# ____________________________________________________________________________
#
# ESO spectrophotometric standard star information retrieval from www.
# ____________________________________________________________________________

def create_ESO_standards_table(
    path_to_standard_spectra=path_to_standard_spectra,
    standard_star_catalog=standard_star_catalog,
    status=None,
    url='http://www.eso.org/sci/observing/tools/standards/spectra/stanlis.html',
    verbose=True ):
    """
Reads the list of standard stars from the ESO website and converts it into
an index.dat file saved to the specified directory. If no url is provided
a default (correct as of 26/10/2012) is used.
    
create_ESO_standards_table( tablename='./ESOstandards/ESOstandards.dat', 
                            status=None, url=None, verbose=False )

    tablename      the path to where the table will be written
    status         not actually used
    url            the url for where the ESO standards list can be found
    verbose        Boolean controlling verbose outputs
    
by default, the standard star list is read from the following url:
    http://www.eso.org/sci/observing/tools/standards/spectra/stanlis.html 
"""

    tablename = '%s/%s' % ( path_to_standard_spectra, standard_star_catalog )
    if verbose :
        print """\n
Starting SAMI_fluxcal.create_ESO_standards_table to get ESO standard stars.
A catalogue of standard stars positions, etc., will be written to the file:"""
        print tablename
        print


    if not os.path.exists( path_to_standard_spectra ) :
        if verbose > 0 :
            print '\n\n\nWARNING: path %s does not exist.' % path_to_standard_spectra
            print '\n\n\n' 
            verbose = -9

    if verbose > 0 :
        print "Opening webpage:", url

    f = urllib2.urlopen(url)
    webpage = f.read()                       # Read the full standards webpage
    f.close()    
                                         # Find the start and end of the table
    dashes = webpage.find('-----------------------------'
                          '-----------------------------')
    endcode = webpage.find('</code>', dashes)
    table = webpage[dashes:endcode]

    file_out = open(tablename, 'w')                     # Open the output file

    line_start = 0                            # Examine the table line by line
    while True:                            # Find the start of the table entry
        line_start = table.find('<a href="', line_start)
        if line_start == -1:                                 # No more entries
            break        
                        
        filename = table[           # Find and extract the filename in the url
            line_start+9 : table.find( '"', line_start+9) ].split('/')[-1]
                                       # Convert the filename to f*.dat format
        filename = 'f' + filename[:filename.rfind('.')] + '.dat'

                                       # Find and extract the name of the star
        name = table[table.find('>', line_start) + 1:
                     table.find('</a>', line_start)]
        name = name.replace(' ', '')                    # Strip out whitespace

                            # Get the coordinates, magnitude and spectral type
        data = table[table.find('</a>', line_start) + 4:
                     table.find('<br>', line_start)].split()
        coords = data[:6]

        if len(data) > 6:
            mag = data[6]
        else:
            mag = 'NaN'
            if verbose :
                print 'magnitude not found for object', name
        if len(data) > 7:
            spec_type = data[7]
        else:   
            spec_type = 'unknown'
            if verbose :
                print 'spec type not found for object', name

                                           # Print the data to the output file
        tableline = \
            '{:18s}{:12s}    {:3s}{:3s}{:5s}    {:4s}{:3s}{:5s}{:8.2f}  {:s}\n'.format(
            filename, name, coords[0], coords[1], coords[2], 
            coords[3], coords[4], coords[5], float(mag), spec_type)
        # Surely there's a better way than coords[0], coords[1] etc?

        file_out.write( tableline )

        if verbose > 0 :
            print 'Copied data for', name
        if verbose > 1 :
            print tableline,

        line_start += 1

    # Close the output file
    file_out.close()

    if verbose :
        print 'Wrote file', tablename, '.'
        print "\nFinished SAMI_fluxcal.create_ESO_standards_table.\n"

    return
# ____________________________________________________________________________
# ____________________________________________________________________________

def get_ESO_standard_spectra(
    directory=path_to_standard_spectra, status=None, 
    url_list=None, update=False, verbose=True ):
    """
Download and save the ESO standard star spectra to the given directory.
url_list is an optional argument specifying which ftp (or other) directories
should be searched in for spectra, if not provided then a default set of
four (correct as of 16/10/2012) is used.

get_ESO_standard_spectra( directory='./ESOstandards/', 
                          status=None, url_list=None, verbose=False )

    directory      the path to where the spectra will be saved
    status         not actually used
    url_list       a list of urls where the ESO standard spectra can be found
    update         Boolean controlling whether to overwrite existing files
    verbose        Boolean controlling verbose outputs

by default, standard star spectra are taken from the following sources:
url_list = 
    [ 'ftp://ftp.eso.org/pub/stecf/standards/hststan/',
      'ftp://ftp.eso.org/pub/stecf/standards/okestan/',
      'ftp://ftp.eso.org/pub/stecf/standards/ctiostan/',
      'ftp://ftp.eso.org/pub/stecf/standards/wdstan/' ]

"""
    if verbose :
        print ; print """
Starting SAMI_fluxcal.get_ESO_standard_spectra to download ESO standards.
Spectra for standard stars will be placed in the following directory:"""
        print directory

    if not os.path.exists( directory ):
        print '\n' * 5
        print 'The path specified for path_to_ESO_standards does not exist!'
        print '\n' * 3
        print 'You have to make this directory for me!'
        print '\n' * 5

    if url_list == None:
        # Define the default URLs in which to search
        url_list = ['ftp://ftp.eso.org/pub/stecf/standards/hststan/',
                    'ftp://ftp.eso.org/pub/stecf/standards/okestan/',
                    'ftp://ftp.eso.org/pub/stecf/standards/ctiostan/',
                    'ftp://ftp.eso.org/pub/stecf/standards/wdstan/']

    for url in url_list:

        # Get the list of filenames in this directory
        if verbose:
            print "Searching for files in", url
        f = urllib2.urlopen(url+'/')
        contents = f.read()
        f.close()

        # Pick out f*.dat filenames
        filename_list = []
        dotdat = 0
        while True:
            # Find the end of the filename
            dotdat = contents.find('.dat', dotdat)
            if dotdat == -1:
                # No more files in the list
                break
            # Find the start of the filename
            whitespace = contents.rfind(' ', 0, dotdat)
            filename = contents[whitespace+1:dotdat+4]
            if filename[0] == 'f':
                # This is a flux file, save it to the list
                filename_list.append(filename)
            dotdat += 1

        if verbose:
            print len(filename_list), "files found"

        # Save the files one by one
        for filename in filename_list:
            if os.path.exists( directory+'/'+filename ) and not update:
                if verbose:
                    print '%18s already exists; no new download.' % filename
                pass
            else :                
                if verbose:
                    print 'Saving file:', filename, 'to directory', directory
                urllib.urlretrieve(url+'/'+filename, directory+'/'+filename)

    if verbose :
        print "\nFinished SAMI_fluxcal.get_ESO_standard_spectra.\n" ; print

    return

# ____________________________________________________________________________
# ____________________________________________________________________________

def extract_secondary_standard( samifitsfilename, verbose=True ):
    """Identify and extract secondary standard star from SAMI data frame"""
    if verbose > 0 :
        print '_' * 78
        print '\nLooking to identify secondary standard star in dataframe:'
        print ' '*8 + samifitsfilename
    
    fitsfile=pf.open(samifitsfilename)
    fulltable=fitsfile[2].data 

    names = fulltable[ 'NAME' ]

    careabout = ( ( names == '' ) | ( names.startswith( 'SKY' ) ) ) == False
    names = names[ careabout ]
    objects = np.unique( names )
    galaxies = objects.startswith( 'J' )
    
    secstandard = objects[ galaxies == False ]
    mask = np.where( fulltable[ 'NAME' ] == secstandard )
    probename = fulltable[ 'PROBENAME' ][ mask ][ 0 ]

    fitsfile.close()

    if verbose > 0:
        print 'Found star with ID %s in probe %s.' % ( secstandard, probename )

    datadict = IFU_pick( samifitsfilename, probename, 
                         extincorr=True, verbose=verbose )

    return datadict

# ____________________________________________________________________________
# ____________________________________________________________________________
#
# Castelli & Kurucz (2004) model stellar spectra for secondary calibration
# ____________________________________________________________________________

def load_CK04_models( path_to_ck04_models='./ck04models/', verbose=True ):
    """Load Castelli & Kurucz (2004) model stellar spectra from standard file."""

    if verbose > 0 :
        print '_' * 78
        print '\nLoading Castelli & Kurucz (2004) model stellar spectra.'
        print 'Looking for file %s/catalog.fits.' % path_to_ck04_models

    catalog = atpy.Table( '%s/%s' % ( 
            path_to_ck04_models, 'catalog.fits' ), type='fits' )
    loaded = []
    wavelength, allmodnames, allmodels = None, None, None
    for idi, identifier in enumerate( catalog.FILENAME ):
        filename = identifier.split( '[' )[0]
        temp = int( filename.split( '_' )[ 1 ].split( '.' )[0] )

        if ( 5860 <= temp and temp <= 8270 and not loaded.count( filename ) ) :
            # this restricts the models to roughly A3V -- G2V inclusive,
            # provided that 4 < log g < 4.4 (which is enforced below); see:
            # //www.stsci.edu/hst/observatory/cdbs/castelli_kurucz_atlas.html

            modellist, namelist = [], []
            model = atpy.Table( '%s/%s' % ( 
                    path_to_ck04_models, filename ), type='fits' )
            loaded.append( filename )
            
            if wavelength == None :
                wavelength = model.WAVELENGTH

            fstarkeys = 'g40 g45'.split()
            for key in fstarkeys : # model.keys() :
#                print key,
                if key != 'WAVELENGTH' and model.data[ key ].max() :
                    modellist.append( model.data[ key ] )
                    namelist.append( filename.split( '.fits' )[0]+'_'+key )
            if allmodels == None :
                allmodels = np.array( modellist )
                allnames = namelist
            else :
                allmodels = np.vstack( (allmodels, modellist ) )
                allnames += namelist
#            print allmodels.shape
            
            del( model )
#            print idi, identifier, model.WAVELENGTH.shape, \
#                model.WAVELENGTH.min(), model.WAVELENGTH.max(), len( loaded )
        hourglass( idi+1, len( catalog.FILENAME ) )

    if verbose > 0 :
        print 78 * '_'
        
#    global modelwl, modelspec, modelnames
#    modelwl, modelspec, modelnames = wavelength, allmodels, allnames

    return wavelength, allmodels, allnames
# ____________________________________________________________________________
# ____________________________________________________________________________
#
#    SAMI-specific data retrieval utilities, based on code
#    by Lisa Fogarty (SAMI_utils_IV), minor edits by Edward Taylor
# ____________________________________________________________________________
# ____________________________________________________________________________

def IFU_pick( samifitsfilename, IFU=None, 
              extincorr=False, returnfitsfile=False, verbose=False ):

    datadict = { 'filename':samifitsfilename }

    # open the fits file and extract data, variance, header, and IFS table
    fitsfile=pf.open(samifitsfilename)

    hdr=fitsfile[0].header
    data=fitsfile[0].data
    var=fitsfile[1].data
    fulltable=fitsfile[2].data 
    #Correct, but for fudged file (test_sami_coords_hdr.fits) is list1[1]

    #Exposure Time
    datadict[ 'EXPOSED' ] = hdr['EXPOSED']
    exptime = datadict[ 'EXPOSED' ]

    # Gain (NB. -- actually *inverse* gain; has units of electrons per ADU)
    datadict[ 'GAIN' ] = hdr[ 'RO_GAIN' ]

    datadict[ 'PROBENAME' ] = IFU
    #Mask the table with the IFU keyword to pick out the right IFU
    mask=fulltable.field( 'PROBENAME' )==IFU
    table=fulltable[mask]
    
    #X and Y positions of fibres in absolute degrees.
    datadict[ 'FIB_MRA' ] = table.field('FIB_MRA') 
    datadict[ 'FIB_MDEC'] = table.field('FIB_MDEC')

    #Fibre number - used for tests.
    datadict[ 'FIBNUM' ]=table.field('FIBNUM')

    #Fibre designation.
    datadict[ 'TYPE' ]=table.field('TYPE')

    #Probe number
    datadict[ 'PROBNUM' ]=table.field('PROBENUM')

    #Adding for tests only - LF 05/04/2012
    datadict[ 'FIBPOS_X' ]=table.field('FIBPOS_X')
    datadict[ 'FIBPOS_Y' ]=table.field('FIBPOS_Y')

    #Name of object
    datadict[ 'NAME' ] = table.field( 'NAME' )[0]

    #indices of the corresponding spectra (SPEC_ID counts from 1, image counts from 0)
    datadict[ 'SPEC_ID' ] = table.field( 'SPEC_ID' )
    ind = datadict[ 'SPEC_ID' ] - 1

    datadict[ 'CDELT1' ] = hdr[ 'cdelt1' ]
    
    # data and variance arrays
    datadict[ 'data' ] = data[ ind, : ] / exptime
    datadict[ 'var'  ] = var[  ind, : ] /(exptime * exptime)

    # wavelength
    datadict[ 'wl' ] = hdr[ 'CRVAL1' ] + ( 
        hdr[ 'cdelt1' ] * ( np.arange( data.shape[1] ) - hdr[ 'crpix1' ] ) )

    # zenith distance and airmass calculation
    datadict[ 'ZDSTART' ], datadict[ 'ZDEND' ] \
        = hdr[ 'ZDSTART' ], hdr[ 'ZDEND' ]

    zdstart, zdend = hdr[ 'ZDSTART' ], hdr[ 'ZDEND' ]
    airmass = calculate_airmass( zdstart, zdend )

    if extincorr :
        # read in extinction curve, expressed in mag
        extwl, atmext = get_atm_extinction_curve( verbose=verbose )
        # interpolate onto the data wavelength grid
        atmext = np.interp( datadict[ 'wl' ], extwl, atmext )
        # turn into mutiplicative flux scaling
        extcorr = 10**( -0.4 * airmass * atmext )

        if verbose > -1 :
            print 'Appling extinction correction with airmass = %.3f.' % airmass        

#        extcorr = 1.
#        print '\n\n\nNOT APPLYING ATM EXT CORR!!!\n\n\n' * 3

        # scale data --- NB. extcorr < 1, so dividing increases the flux
        datadict[ 'data' ] /= extcorr
        # scale variance
        datadict[ 'var'  ] /= extcorr**2.

    fitsfile.close()

    return datadict

# ____________________________________________________________________________
# ____________________________________________________________________________

def relative_fibre_positions( datadict ):

    rafib = datadict[ 'FIB_MRA' ]
    decfib = datadict[ 'FIB_MDEC' ]
    centre = np.where( datadict[ 'FIBNUM' ] == 1 )
    centralRA, centralDec = rafib[ centre ], decfib[ centre ]

    xfibre = -1. * (rafib-centralRA) * np.cos( np.deg2rad(centralDec)) *3600.
    yfibre = ( decfib - centralDec ) * 3600.

    return xfibre, yfibre

# ____________________________________________________________________________
# ____________________________________________________________________________

def calculate_airmass( zdstart, zdend ):
    zdmid = ( zdstart + zdend ) / 2.
    amstart, ammid, amend = zd2am( np.array( [zdstart, zdmid, zdend] ) )
    airmass = ( amstart + 4. * ammid + amend ) / 6.
    # Simpson integration across 3 steps
    return airmass
# ____________________________________________________________________________

def zd2am( zenithdistance ):
    # fitting formula from Pickering (2002)
    altitude = ( 90. - zenithdistance ) 
    airmass = 1./ ( np.sin( ( altitude + 244. / ( 165. + 47 * altitude**1.1 )
                            ) / 180. * np.pi ) )
    return airmass



# ____________________________________________________________________________
# ____________________________________________________________________________
#
# Mike Bessell's atmospheric exctinction curve for Siding Spring Observatory
# ____________________________________________________________________________

def get_atm_extinction_curve( 
    atm_extinction_table=atm_extinction_table,
    url='http://www.mso.anu.edu.au/~bessell/FTP/Spectrophotometry/', 
    verbose=False ):
    """Load standard SSO atmospheric extinction curve from Mike Bessell"""
    
    if not os.path.exists( atm_extinction_table ):
        
        print ; print '_' * 78 ; print
        print 'atmospheric extinction table not found.'
        print "i will try to download this from Mike Bessell's webpage:"
        print url
        print

        urllib.urlretrieve( 
            '%s/%s' % ( url, os.path.basename(atm_extinction_table) ), 
            atm_extinction_table )

        print 'done.'
        print ; print '_' * 78 ; print

    wl, ext = [], []

    for entry in open( atm_extinction_table, 'r' ).xreadlines() :
        line = entry.rstrip( '\n' )
        if not line.count( '*' ) and not line.count( '=' ):
            values = line.split()
            wl.append(  values[0] )
            ext.append( values[1] )
    
    wl = np.array( wl ).astype( 'f' )
    ext= np.array( ext ).astype( 'f' )
    
    return wl, ext
# ____________________________________________________________________________
# ____________________________________________________________________________

def analytic_DAR( wlinAngstrom, zenith_angle, wl0=5000.,
                  temperature=7., pressure=600., vapourPressure = 8. ):
                  
    wl = wlinAngstrom * 1e-4
    # analytic expectations from Fillipenko (1982)

    seaLevelDry = 1e-6 * ( 64.328 + ( 29498.1 / ( 146. - ( 1 / wl**2. ) ) )
                           + 255.4 / ( 41. - ( 1. / wl**2. ) ) )

    altitudeCorrection = ( 
        ( pressure * ( 1. + (1.049 - 0.0157*temperature ) * 1e-6 * pressure ) )
        / ( 720.883 * ( 1. + 0.003661 * temperature ) ) )

    vapourCorrection = ( ( 0.0624 - 0.000680 / wl**2. )
                         / ( 1. + 0.003661 * temperature ) ) * vapourPressure

    refindex = seaLevelDry * altitudeCorrection * vapourCorrection + 1
    
    ref0 = np.interp( wl0, wlinAngstrom, refindex )
    DAR = 206265. * ( refindex - ref0 ) * np.tan( zenith_angle * np.pi / 180. )

    return DAR
    



# ____________________________________________________________________________
# ____________________________________________________________________________

def rebin_IFU( targetwl, originalwl, datain, varin ):

    originaldata = datain[ : , 1:-1 ]
    originalvar  = varin[ : , 1:-1 ]
    originalbinlimits = ( originalwl[ :-1 ] + originalwl[ 1: ] ) / 2.

    # for the rebinning, we want to conserve total flux in each bin; 
    # but input spectrum is assumed to be in flux / unit wavelength
    okaytouse = ( np.isfinite( originaldata ) ) # & ( varin < 1e4 )

    originalweight = np.where(okaytouse, 1., 0.)
    originaldata = np.where(okaytouse, originaldata, 0.)
    originalvar  = np.where(okaytouse, originalvar , 0.)

    originalflux = originaldata * np.diff( originalbinlimits )
    originalvar = originalvar * np.diff( originalbinlimits )
    originalweight *= np.diff( originalbinlimits )

    nowlsteps = len( targetwl )
    rebinneddata   = np.zeros( (originaldata.shape[0], nowlsteps) )
    rebinnedvar    = np.zeros( (originaldata.shape[0], nowlsteps) )
    rebinnedweight = np.zeros( (originaldata.shape[0], nowlsteps) )

    binlimits = np.array( [ np.nan ] * (nowlsteps+1) )
    binlimits[ 0 ] = targetwl[ 0 ]
    binlimits[ 1:-1 ] = ( targetwl[ 1: ] + targetwl[ :-1 ] ) / 2.
    binlimits[ -1 ] = targetwl[ -1 ]
    binwidths = np.diff( binlimits )

    origbinindex = np.interp( binlimits, originalbinlimits, 
                              np.arange( originalbinlimits.shape[0] ),
                              left=np.nan, right=np.nan )

    fraccounted = np.zeros( originaldata.shape[1] )
    # use fraccounted to check what fraction of each orig pixel is counted,
    # and in this way check that flux is conserved.

    maximumindex = np.max( np.where( np.isfinite( origbinindex ) ) )

    for i, origindex in enumerate( origbinindex[ :-1 ] ):
      if np.isfinite( origindex ) :
        # deal with the lowest orig bin, which straddles the new lower limit
        lowlimit = int( origindex )
        lowfrac = 1. - ( origindex % 1 )

        indices = np.array( [ lowlimit] )
        weights = np.array( [ lowfrac ] )

        # deal with the orig bins that fall entirely within the new bin
        if np.isfinite( origbinindex[i+1] ):
            intermediate = np.arange( int( origindex )+1, \
                                  int(origbinindex[i+1]) )
        else :
            intermediate = np.arange( int( origindex )+1, \
                                        maximumindex )
#        print intermediate.shape
#        print intermediate
        indices = np.hstack( ( indices, intermediate ) )
        weights = np.hstack( ( weights, np.ones( intermediate.shape ) ) )

        # deal with the highest orig bin, which straddles the new upper limit
        if np.isfinite( origbinindex[i+1] ):
          upplimit = int( origbinindex[i+1] )
          uppfrac = origbinindex[ i+1 ] % 1
          indices = np.hstack( ( indices, np.array( [ upplimit ] ) ) )
          weights = np.hstack( ( weights, np.array( [ uppfrac  ] ) ) )

        fraccounted[ indices ] += weights

        rebinneddata[ :, i ] = np.sum( weights * originalflux[ :, indices ], axis=1 )
        rebinnedvar[  :, i ] = np.sum( weights * originalvar[  :, indices ], axis=1 )
        rebinnedweight[:,i ] = np.sum( weights * originalweight[:, indices ], axis=1 )

    # now go back from total flux in each bin to flux per unit wavelength
    rebinneddata = rebinneddata / rebinnedweight 
    rebinnedvar  = rebinnedvar  / rebinnedweight 

    return rebinneddata, rebinnedvar

# ____________________________________________________________________________
# ____________________________________________________________________________

def rebin_spectrum( targetwl, originalwl, originalspec, originalvarin=None,
                    centerbins=True ):

    if centerbins :
        originaldata = originalspec[ 1:-1 ]
        originalvar  = originalvarin[ 1:-1 ]
        originalbinlimits = ( originalwl[ :-1 ] + originalwl[ 1: ] ) / 2.
    else :
        originaldata = originalspec[ 1: ]
        originalvar  = originalvarin[ 1: ]
        originalbinlimits = originalwl

    # for the rebinning, we want to conserve total flux in each bin; 
    # but input spectrum is assumed to be in flux / unit wavelength
    okaytouse = ( ( np.isfinite( originaldata ) )
                  & ( np.isfinite( originalvar ) ) 
                  & ( originalvar > 0. ) ) # & ( originalvar < 1.e3 )

    originalweight = np.where(okaytouse, 1., 0.)
    originaldata = np.where(okaytouse, originaldata, 0.)
    originalvar  = np.where(okaytouse, originalvar , 0.)

    originalflux = originaldata * np.diff( originalbinlimits )
    originalvar = originalvar * np.diff( originalbinlimits )
    originalweight *= np.diff( originalbinlimits )

    nowlsteps = len( targetwl )
    rebinneddata   = np.zeros( nowlsteps )
    rebinnedvar    = np.zeros( nowlsteps )
    rebinnedweight = np.zeros( nowlsteps )

    binlimits = np.array( [ np.nan ] * (nowlsteps+1) )
    binlimits[ 0 ] = targetwl[ 0 ]
    binlimits[ 1:-1 ] = ( targetwl[ 1: ] + targetwl[ :-1 ] ) / 2.
    binlimits[ -1 ] = targetwl[ -1 ]
    binwidths = np.diff( binlimits )

    origbinindex = np.interp( binlimits, originalbinlimits, 
                              np.arange( originalbinlimits.shape[0] ),
                              left=np.nan, right=np.nan )

    fraccounted = np.zeros( originaldata.shape[0] )
    # use fraccounted to check what fraction of each orig pixel is counted,
    # and in this way check that flux is conserved.

    maximumindex = np.max( np.where( np.isfinite( origbinindex ) ) )

    for i, origindex in enumerate( origbinindex ):
      if np.isfinite( origindex ) :
        # deal with the lowest orig bin, which straddles the new lower limit
        lowlimit = int( origindex )
        lowfrac = 1. - ( origindex % 1 )
        indices = np.array( [ lowlimit] )
        weights = np.array( [ lowfrac ] )

        # deal with the orig bins that fall entirely within the new bin
        if np.isfinite( origbinindex[i+1] ):
            intermediate = np.arange( int( origindex )+1, \
                                  int(origbinindex[i+1]) )
        else :
            intermediate = np.arange( int( origindex )+1, \
                                        maximumindex )
        indices = np.hstack( ( indices, intermediate ) )
        weights = np.hstack( ( weights, np.ones( intermediate.shape ) ) )

        # deal with the highest orig bin, which straddles the new upper limit
        if np.isfinite( origbinindex[i+1] ):
          upplimit = int( origbinindex[i+1] )
          uppfrac = origbinindex[ i+1 ] % 1
          indices = np.hstack( ( indices, np.array( [ upplimit ] ) ) )
          weights = np.hstack( ( weights, np.array( [ uppfrac  ] ) ) )

        fraccounted[ indices ] += weights
        rebinneddata[ i ] = np.sum( weights * originalflux[ :, indices ] )
        rebinnedvar[  i ] = np.sum( weights * originalvar[  :, indices ] )
        rebinnedweight[i ]= np.sum( weights * originalweight[:,indices ] )

    # now go back from total flux in each bin to flux per unit wavelength
    rebinneddata = rebinneddata / rebinnedweight 
    rebinnedvar  = rebinnedvar  / rebinnedweight 

    return rebinneddata, rebinnedvar

# ____________________________________________________________________________
# ____________________________________________________________________________

def median_smooth( spectrum, binsize, givenmad=False ):

    bon2 = binsize/2.
    specsize = spectrum.shape[0]
    result = np.zeros( specsize )
    if givenmad :
        nmad = np.zeros( specsize )
    for i in range( specsize ):
        lo = max( i-bon2, 0 )
        hi = min( i+bon2+1, specsize )
        med = np.median( spectrum[ lo:hi ] )
        result[ i ] = med
        if givenmad :
            nmad[ i ] = 1.486 * np.median( np.abs( spectrum[ lo:hi ] - med ) )

    if givenmad :
        return result, nmad
    else :
        return result

# ____________________________________________________________________________
# ____________________________________________________________________________

def hourglass( current, target, done=False, verbose=True ):
    if verbose <= 0 :
        return

    if done :
        fraction = 1.
    else :
        fraction = float( current ) / float( target )

    ndashes = int( fraction * 50 )
    dashes = '=' * ndashes
    if ndashes == 0 :
        pass
    elif ndashes == 1 :
        dashes = 'o'
    elif fraction < 1 :
        dashes = '<' + dashes[ 1:-1 ] + '>'
    else :
        dashes = '(' + dashes[ 1:-1 ] + ')'
    
    if fraction < 1 and verbose <= 1 : endline = '\r'
    else : endline = '\n'

    sys.stdout.write( '   progress:  |{:^50s}| ({:4.0%}){:}'.format( 
                            dashes, fraction, endline ) )
    sys.stdout.flush()
# ____________________________________________________________________________

def pinwheel( current, maximum, done=False, verbose=True ):
    if verbose <= 0 :
        return
    
    if done :
        print '\n Done.'
        return

    spinners = '/ - \ | / - \ |'.split()
    spinners[ 1 ] = u"\u2013"
    spinners[ 5 ] = u"\u2013"

    printstr = '  iteration %3i (%i max.)' % ( current, maximum )
    printstr += '       %s      %s  %s         %s      %s    %s' % (
        spinners[ current % 8 ], spinners[ (3+current) % 8 ],
        spinners[ (1+current) % 8 ], spinners[ (current+4) % 8 ],
        spinners[ (3+current) % 8 ], spinners[ (current+1) % 8 ], )
    printstr += '\r'
   
    sys.stdout.write( printstr ) ; sys.stdout.flush()
    
# ____________________________________________________________________________

def add_date_stamp( filename, top=False ):
    datestring = '%s\n%s' % ( time.ctime(), filename )
    if top :
        plt.figtext( 0.5, 0.995, datestring, 
                    ha='center', va='top', color='gray', fontsize='small' )
    else :
        plt.figtext( 0.995, 0.5, datestring, rotation='vertical', 
                    ha='right', va='center', color='gray', fontsize='small' )
                    

# ____________________________________________________________________________
# ____________________________________________________________________________

if __name__ == '__main__' :

    verbose=1

    helpstring = """
useage:

first:

SAMI_fluxcal.py setup [/path/to/ESOstandards/]

This will fetch the required spectrophotometric standard star data (~15Mb),
and put in /path/to/ESOstandards. ( ./ESOstandards/ if unspecified. 

then

SAMI_fluxcal.py /path/to/standard/ /path/to/data/ [/path/to/ESOstandards].

default value for /path/to/ESOstandards/ is ./ESOstandards/ .

"""
    args = sys.argv[ 1: ]

    if len( args ) == 0 or len( args ) > 3 :
        print helpstring


    if len( args ) >= 1 :
        if args[0] == 'setup' :
            if len( args ) == 1 :
                create_ESO_standards_table(  ) 
                get_ESO_standard_spectra( )
            elif len( args ) == 2 :
                create_ESO_standards_table( args[1] )
                get_ESO_standard_spectra( args[1] )
            else :
                print helpstring
        elif len( args ) == 1 :
            print helpstring
        else :
            print """
Running SAMI_fluxcal.fluxcal.

I am assuming that the standard star data can be found here:
(This needs to be something like /path/to/standards/ccd_1/)

%s

I am assuming that the data to be calibrated can be found here:
(This needs to be something like /path/to/data/ccd_1/)

%s 
""" % ( args[0], args[1] )

            if len( args ) == 2 :
                print """
I will look for ESO spectrophotometric standard data here:
"""
                print path_to_standard_spectra

                fluxcal( args[0], args[1], verbose=verbose )

            else :
                print """
I will look for ESO spectrophotometric standard data here:
"""
                print args[2]

                fluxcal( args[0], args[1], 
                         path_to_ESO_standards=args[2], verbose=verbose )

        plt.close( 'all' )
        print '\nAll done!\n\n'

# ____________________________________________________________________________
# ____________________________________________________________________________


