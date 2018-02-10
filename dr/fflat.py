"""
This module addresses issues with fibre flat field frames

correct_bad_fibres() is used to identify fibres with 'bad' flat field values,
typical due to the presence of star in a twilight flat or an uncorrected
cosmic ray.

correct_bad_fibres takes as input a group of flats observed on the same field,
constructs a mean flat frame then calculates the residuals of each input frame
on a fibre-by-fibre basis (summing over the wavelength axis). Any fibres
identified as 'bad' (i.e. differing by more than 3 sigma from the mean) are
replaced with the average fibre profile.
"""

import numpy as np
import astropy.io.fits as pf

def correct_bad_fibres(path_list,debug=False):

    """ Replace bad fibre flat values with an average """

    # Define array to store fibre flats
    n_file = len(path_list)
    n_fibre = pf.getval(path_list[0],'NAXIS2','PRIMARY')
    n_spec = pf.getval(path_list[0],'NAXIS1','PRIMARY')

    fflats = np.zeros((n_file,n_fibre,n_spec))

    # Read in fibre flats from fits files to array
    for index in range(n_file):
        fflats[index,:,:] = pf.getdata(path_list[index],'PRIMARY')

    # Determine the average fibre flat, per-pixel residual images
    # and average residual per fibre
    avg_fflat = np.nanmean(fflats,axis=0)

    residual_fflats = fflats - avg_fflat

    fibre_residuals = np.sqrt(np.nanmean(residual_fflats**2,axis=2))

    # Identify all 'bad' fibres that lie more than 3 sigma from
    # the mean
    mean_residual = np.nanmean(fibre_residuals)
    sig_residual = np.nanstd(fibre_residuals)

    ww = np.where(fibre_residuals > mean_residual + 3*sig_residual)

    # Replace worst fibre with nans, recalculate the residuals,
    # then iterate until no bad fibres remain

    bad_fibres_index = []
    fflats_fixed = np.copy(fflats)
    i = 0
    n_bad_fibres = 1e9

    while n_bad_fibres > 0:
        worst_fibre_index = np.squeeze(np.where(fibre_residuals == 
            np.nanmax(fibre_residuals)))
        bad_fibres_index.append(worst_fibre_index)
        fflats_fixed[worst_fibre_index[0],worst_fibre_index[1],:] = np.nan
        avg_fixed = np.nanmean(fflats_fixed,axis=0)
        residual_fflats = fflats_fixed - avg_fixed
        fibre_residuals = np.sqrt(np.nanmean(residual_fflats**2,axis=2))
        still_bad_fibres_index = np.where(fibre_residuals > 
                mean_residual + 4*sig_residual)
        n_bad_fibres = len(still_bad_fibres_index[0])
        i = i+1

    # Replace all bad fibres with the average for that fibre

    bad_fibres_index = np.squeeze(bad_fibres_index)
    fflats_fixed[bad_fibres_index[:,0],bad_fibres_index[:,1],:] = avg_fixed[bad_fibres_index[:,1],:]

    # Write new fibre flat field values to file

    edited_files = np.unique(bad_fibres_index[:,0])
    for index in range(len(edited_files)):
        hdulist = pf.open(path_list[index],mode='update')
        hdulist['PRIMARY'].data = fflats_fixed[index,:,:]
        hdulist.flush()
        hdulist.close()

    if debug:
        for val in edited_files:
            ww = np.where(bad_fibres_index[:,0] == val)
            print(path_list[val])
            print(np.array(bad_fibres_index)[ww,1])
            print('--------')

    return
