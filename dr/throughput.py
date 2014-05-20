"""Code to deal with fibre throughputs."""

import astropy.io.fits as pf

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
    