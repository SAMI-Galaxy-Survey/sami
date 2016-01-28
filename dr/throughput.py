"""
Code to deal with fibre throughputs.

make_clipped_thput_files() is used by the manager to make FITS files with
a THPUT extension that contains values averaged over a few observations.
This is useful for when the throughput has been measured from night sky
lines: the S/N is increased by averaging, and frames with a bad throughput
measurement (e.g. if the 5577A line has been hit by a cosmic ray or bad
pixel) are clipped out.
"""

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

def make_thput_file(path, new_thput, old_thput, used, path_list):
    """Makes a file containing a set of throughput values, for use by 2dfdr."""
    primary_hdu = pf.PrimaryHDU()
    hdu = pf.ImageHDU(new_thput, name='THPUT')
    hdulist = [primary_hdu, hdu]
    for idx, (old_thput_i, used_i, path_i) in enumerate(zip(
            old_thput, used, path_list)):
        data = np.vstack((old_thput_i, used_i))
        hdu_i = pf.ImageHDU(data, name='INPUT')
        hdu_i.header['EXTVER'] = idx + 1
        hdu_i.header['INPATH'] = path_i
        hdulist.append(hdu_i)
    hdulist = pf.HDUList(hdulist)
    hdulist.writeto(path)
    return

def make_clipped_thput_files(path_list, overwrite=True, edit_all=False,
                             median=False):
    """Make thput files with bad values replaced by an average."""
    n_file = len(path_list)
    n_fibre = pf.getval(path_list[0], 'NAXIS1', 'THPUT')
    thput = np.zeros((n_file, n_fibre))
    for index in xrange(n_file):
        thput[index, :] = pf.getdata(path_list[index], 'THPUT')
    new_thput = np.zeros((n_file, n_fibre))
    # 2dfdr Gauss extraction now replaces dodgy throughput values with 0
    good = (thput > 0) & np.isfinite(thput)
    if median:
        avg_thput = np.array([np.median(thput[good[:, i], i])
                             for i in xrange(n_fibre)])
    else:
        avg_thput = np.sum(thput * good, 0) / np.sum(good, 0)
    avg_thput = np.outer(np.ones(n_file), avg_thput)
    if edit_all:
        new_thput = avg_thput
        edited = np.ones(n_file, bool)
    else:
        new_thput[good] = thput[good]
        new_thput[~good] = avg_thput[~good]
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
                make_thput_file(path_out, new_thput[index], thput, good,
                                path_list)
            else:
                edited[index] = False
    return edited
