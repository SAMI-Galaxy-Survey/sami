"""Code to deal with fibre throughputs."""

import os

import astropy.io.fits as pf
import numpy as np

def edit_throughput(hdulist, fibres, new_throughputs):
    """Change the throughput values in one or more fibres."""
    # We will update hdu and save a copy of the original
    hdu = hdulist['THPUT']
    hdu_old = hdu.copy()
    data = hdulist[0].data
    variance = hdulist['VARIANCE'].data
    sky = hdulist['SKY'].data
    # Update the data
    data[fibres, :] = (
        ((data[fibres, :] + sky).T * 
         hdu_old.data[fibres] / new_throughputs).T - sky)
    # Update the variance - this is wrong because we don't have the sky var
    variance[fibres, :] = (
        (variance[fibres, :].T * (hdu_old.data[fibres] / new_throughputs)**2).T)
    # Update the throughput table
    hdu.data[fibres] = new_throughputs
    return

def make_thput_file(path, throughput):
    """Makes a file containing a set of throughput values, for use by 2dfdr."""
    hdu = pf.ImageHDU(throughput, name='THPUT')
    primary_hdu = pf.PrimaryHDU()
    hdulist = pf.HDUList([primary_hdu, hdu])
    hdulist.writeto(path)
    return

def make_clipped_thput_files(path_list, overwrite=True):
    """Make thput files with bad values replaced by a mean."""
    n_file = len(path_list)
    n_fibre = pf.getval(path_list[0], 'NAXIS1', 'THPUT')
    thput = np.zeros((n_file, n_fibre))
    for index in xrange(n_file):
        thput[index, :] = pf.getdata(path_list[index], 'THPUT')
    new_thput = np.zeros((n_file, n_fibre))
    # 2dfdr now replaces dodgy throughput values with 0, which helps
    good = (thput > 0)
    new_thput[good] = thput[good]
    mean_thput = np.sum(thput * good, 0) / np.sum(good, 0)
    mean_thput = np.outer(np.ones(n_file), mean_thput)
    new_thput[~good] = mean_thput[~good]
    # Check which of the files have had their values edited
    edited = (np.sum(good, 1) != n_fibre)
    # Write new files where necessary
    for index in xrange(n_file):
        if edited[index]:
            path_in = path_list[index]
            filename_out = 'thput_' + os.path.basename(path_in)
            path_out = os.path.join(os.path.dirname(path_in), filename_out)
            if os.path.exists(path_out) and overwrite:
                os.remove(path_out)
            if not os.path.exists(path_out):
                make_thput_file(path_out, new_thput[index])
            else:
                edited[index] = False
    return edited
