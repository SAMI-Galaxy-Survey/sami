"""
Main script for running probabilistic cubesolve method on SAMI data
This method is a generative model with a Gaussian Process Prior and a 2D radial kernel 
which is adaptive and pre-optimised given wavelength dependend PSF. Still in testing version.
Seb Haan, 21 March 2018

Main Input: reduced fibre data in form of '...sci.fits' file list
Main Return: 1) Fits for Data + Variance Cube, 
			 2) Fits file of components for reconstructing coariance matrix
			 3) Some preliminay diagnostic and evaluation plots

Currently required files in same directory (in addition to SAMI pipeline modules):
datafuse_...py, gpcubesolve_....py, gpcovariance_...py

Note on binning/chunking along spectral axes - There are two possible ways to bin:
1) Sum over spetrcal block of fibre data (length of each block: wavebin parameter), default 1 --> no binnning
2) Choose that response (transfromation matrix) is only calculated every nsteps (responsebin parameter), default 32
The response cube is written in fits file with same length

Note on storing covariance - two options:
1) Covariance can be reconstrcuted from response matrix, GP lenghtsale, and fiber variance.
With this method the response matrix size is reduced by a factor of ~2500/(61*7) * nresponse 
in comparison to full covariance matrix and ensures lossless reconstruction
2) Store entire covariance: not advisable since too large in size. Only possible for low resolution and binning with wavebin

Note on pixelscale: Should be calculated with less than 0.5 arcsec, which results in less artifacts 
and more accurate evaluation of reconstructed FWHM. If 0.5 arcsec required, one can calculate
 at pixelsize = 0.25 arcsec and then enable set zoom2 = True to smooth cube to 0.5 arcsec

Note on GP kernel: Default is a squared exponential kernel which works reasonable well, other kernel options possible 
	but could increase computational time and have only limited testing. 
	The lengthscale parameter gamma is currently set to half the FWHM of the PSF (Moffat), which is dependent on wavelength.
	We have tested the optimal lengthscale by calculating log probabilities as function of gamma, PSF, and S/N. Best range 
	for gamma/ seeing is 0.3-0.5. May add in future option to run for multile gammas, and then marginalize over results.  
	(Gamma setting on line 837 in datafuse_r2.py)


TODO: 	1) enable multiprocessing (split along spectral axis not simple, use pool for working on different cubes instead, align with manager.py)
        2) optional: Add function to approxiate FWHM from S/N and PSF
		3) optional: add adaptive Signal-to-Noise in order to stabilize reconstructed FWHM along spectral axis 
			and suppress sidelobes/rings in covariance/reconstrcuted image. Potentially add 
			cleaning deconvolution algorithm for rings.
		4) Build Test-Suite python library for collecting tools to evaluate product quality 
		(DAR, spatial resolution, velcoity fields etc. )
		5) If more computational power available, use calculated log-probability and posterior for optimization of 
		GP parameter, spatial offsets, and PSF on subset of cube (perhaps with SGD) 
		6) marginalize over GP lengthscale.
 """
from __future__ import print_function
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from sami.gpcubing import datafuse_r2 as df
from sami.gpcubing import gpcovariance_r2 as gpcov
import matplotlib.pylab as plt
import scipy

#################################
"""
Specify parameters below for cubing:
"""

def new_cube(fitslist,name,ccdband='blue',Lpix=50,pixscale=0.5,wavebin=1,responsebin=32,
        path_out='/import/opus1/nscott/SAMI_Survey/new_cubes_output/',filename_ext='_50pix_05arcsec',
        write_fits=True):

    # Lpix: number of pixels for x and y in final cube
    # pixscale: preferred 18/60 = 0.3
    # wavebin: number of wavelengths to stack in each bin
    # responsebin: interval in wavelengths to re-calculate response matrix (same for storing response matrix)
    # ccdband: choose 'red 'or 'blue'
    # star_only: run only on star image for testing, specify identifier of star below

    gpmethod = 'squared_exp' #specify GP kernel used: 'squared_exp', or 'sparse', or 'wide', or 'moffat'
    marginalize = True # Uniform marginalisation over GP length scale
    zoom2 = False # Interpolate pixelsize of final cube by a factor of two
    avgpsf = False # Use average PSF of all exposures, change to False to include all
    gcovar = False # stores covariance cube, only possible for low resolution, probably don't set to True
    avgvar = False # averages variance  for all fibres and wavelengths (exposures can stib  be differnt) to make S2N and resolution stable
    test_reconstuct_covar = False # old option has been only use for internal testing 
							#to reconstruct covariance at certain wavelength use gpcovariance_r...py

    names = [name]

    # Start of Calculation:
    for identifier in names: 

        print('Start cubing', identifier)

        ####### First read and combine data
        data, variance, psf_alpha, psf_beta, xfibre, yfibre, xdar, ydar = df.sami_combine(fitslist,
                                                                  identifier, do_dar_correct = True)
		
        ####### Plot fibre data DAR offset as scatterplot:
        # df.sami_plotfibre(data, xfibre, yfibre, identifier, path_out = path_out)

        ### Just for testing
        if avgvar:
            for i in range(variance.shape[0]):
                var_flat = variance[i, :, :].flatten() 
                variance[i, :, :] = variance[i, :, :] * 0. + np.nanmean(var_flat[var_flat < 1e6])

        ######## Run the actual cubing class
        _Nexp = len(fitslist)
        fuse = df.DataFuse3D(data, (psf_alpha, psf_beta), (xfibre, yfibre), data_sigma = np.sqrt(variance),
                             dar_cor = (xdar, ydar), name = identifier, pixscale = pixscale, _Nexp = _Nexp,
                             avgpsf = avgpsf, gcovar= gcovar, gpmethod=gpmethod)

        ######## Return reconstructed cubes (covar_cube is zero if not explicitly called, see above):
        data_cube, var_cube, resp_cube, covar_cube = fuse.fusecube(Lpix = Lpix, binsize=wavebin, nresponse = responsebin)

        ######## Write fits files for data+variance cube and one fits file for components to reconstruct covaraince :
        if write_fits:
            filename_out=identifier +'_' + ccdband + filename_ext +'_gp.fits'
            df.sami_write_file(fitslist, identifier, data_cube, var_cube, 
				path_out = path_out, filename_out = filename_out, overwrite = True, covar_mode = None, pixscale = pixscale)
            filename_out_covar=identifier +'_' + ccdband+ filename_ext +'_gp_covar.fits'
            df.write_response_cube(identifier, resp_cube, variance, fuse.gamma, pixscale, Lpix, gpmethod,marginalize, 
                                   path_out = path_out, filename_out = filename_out_covar, 
                                   overwrite = True, _Nexp = _Nexp)

            ######## smooth cube to half the pixel size, not really needed:
            if zoom2:
                data_cube[np.isnan(data_cube)] = 0.
                var_cube[np.isnan(var_cube)] = 0.
                data_cube = scipy.ndimage.zoom(data_cube, zoom = [0.5,0.5,1])
                var_cube = scipy.ndimage.zoom(var_cube, zoom = [0.5,0.5,1])
                pixscale = pixscale * 2.
                yar,xar = np.mgrid[1:data_cube.shape[0]+1, 1:data_cube.shape[1]+1]
                r = np.sqrt((xar - data_cube.shape[0]/2)**2 + (yar - data_cube.shape[1]/2)**2)
                rmax = (14.7 + 1.6)/2. / pixscale
                data_cube[r > rmax, :] = np.nan
                var_cube[r > rmax, :] = np.nan
                if write_fits:
                    filename_out=identifier+'_'+ccdband+'_'+str(_Nexp)+'_'+filename_ext + '.fits'
                    df.sami_write_file(fitslist, identifier, data_cube, var_cube, 
					path_out = path_out, filename_out = filename_out, overwrite = True, covar_mode = None, pixscale = pixscale)
                    filename_out_covar=identifier+'_'+ccdband+'_'+str(_Nexp)+'_'+filename_ext + '_covar.fits'
                    df.write_response_cube(identifier, resp_cube, variance, fuse.gamma, pixscale, Lpix, gpmethod, 
					path_out = path_out, filename_out = filename_out_covar, overwrite = True)
                    
######### Calculation and plotting of reconstructed cube characteristics:
                    flux, flux_total, fwhm, fwhm2, fwhm_err, fwhm2_err, s2n_cube, s2n_fibre = df.test_dar_solution(data_cube, 
			var_cube, covar_cube, data, variance, path_out = path_out, plot = True, 
			pixelarcsec = pixscale, seeing = fuse.fwhm_seeing, gcovar = gcovar)

    print('CUBING FINISHED')
    print('-----------------------------------------')

    return True
