"""
Configuration for extracting covariance matrix from file
"""

# Change settings and options below:
####################################
# Number of spectral or wavelength slice (out of the 2048) that is wished to be constructed:
Nspectral = 512
# provide path + filename for stored components of covariance cube
inputfile = '../SAMIcubing_output_dev/10000122red_50pix_05arcsec_nomarg_covar_compressed.fits'
# provide path + filename for output file; by default output file will be written in current run directory:
outputfile = '../SAMIcubing_output_dev/10000122red_50pix_05arcsec_nomarg_covar_reconstructed'
# specify array position in x (column number) of selected pixel, 
# set to None if full covariance between all pixels should be retrieved:
xpix = 25
# specify array position in y (column number) of selected pixel:
ypix = 25
# specify format, either 'csv' (default) or 'fits'
fileformat = 'fits'
# specify whether only covariance for 5x5 pixel cut-out with pixel selected as center (reduced = True), or 
# if full covariance between selected pixel and all other pixel in image (reduced = False)
reduced = False

