# SAMIplus
Sandbox for developing and testing probabilistic SAMI image construction 

The method is based on a generative model with a Gaussian Process Prior and a 2D radial kernel, which is adaptive and pre-optimised given wavelength dependend PSF. The main script for running the new probabilistic cubesolve method on SAMI data is currently run_datafuse.py (STILL IN TESTING). The log-probability and posterior is calculated and can be used for optimization (e.g. of GP parameter and spatial offsets). 

Installation:
See requirements.txt file for main dependencies; tested with Python 3.7.3
(Code also tested with Python 2.7)

How to run code:

1) Specify settings for cubing and GP in config.yaml file
2) execute cubing:  python run_datafuse.py 


Core algorithms for cubing can be found in gpcubesolve.py, see function logl() and predict().


Main Input: reduced fibre data in form of '...sci.fits' file list

Main Return: 
1) Cube for data + variance in fits format
2) Cube fits file of components for reconstructing the covariance matrix
3) Some preliminary diagnostic and evaluation plots


There are two possible options for binning/chunking along spectral axes:
1) Sum over spectral block of fibre data (length of each block: wavebin parameter), default is no binning.
2) Response (transformation) matrix is only calculated every N wavelength steps (responsebin parameter), default responsebin=32. The reason for doing this is to speed up computation since the response matrix calculation takes most of computational time but does not significantly change over a small wavlength range. Another advantage is that the response cube is by default written in fits file with same length as response matrix, which allows to store the components of covariance matrix at a reasonable file size (~500Mb).

The covariance is computed and its components stored as matrix components in a lossless format (not 100% implemented for log-tranform). This allows the user to fully reconstruct the covariance matrix at any wavelength (using gpcovariance.py)

There are two options for storing covariance cube:
1) Covariance can be reconstrcuted from response matrix, GP lenghtsale, and fiber variance.
With this method the response matrix size is reduced by a factor of ~2500/(61*7) * responsebin in comparison to the full covariance matrix and ensures lossless reconstruction.
2) Store entire covariance: not possible in most cases since too large in size. Only possible for low resolution and binning with wavebin

Note on pixelscale: 
Should be calculated with less than 0.5 arcsec, which results in less artifacts and preciser evaluation of reconstructed FWHM. If 0.5 arcsec required, one can calculate at pixelsize = 0.25 arcsec and then enable set zoom2 = True to smooth cube to 0.5 arcsec.

Note on GP kernel: 
Default is a squared exponential kernel which works reasonable well, other kernel options possible but could increase computational time. The lengthscale parameter gamma is currently set to half the FWHM of the PSF (Moffat), which is dependent on wavelength.
We have tested the optimal lengthscale by calculating log probabilities as a function of gamma, PSF, and S/N. Best range for gamma/ seeing is 0.3-0.5 
