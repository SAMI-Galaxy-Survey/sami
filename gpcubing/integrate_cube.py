"""
Script for Reconstructing Covariance Matrix from file
1 March 2018 Seb Haan

Please change settings in config_covar.py
and then execute python gpcovariance_r2.py
"""
from __future__ import print_function
import sys
import os
import numpy as np
from astropy.io import fits as pf
import astropy.wcs as pw


class Covariance():
	"""
	    Class for calculating the covariance matrix from saved components: 
	    response matrix, GP gamma, and fibre data variance
	    See for more details gpcubesolve_r1.py for constructing cube and covariance
    """
    
	def __init__(self, filename):
	    """ Readin of header information and calculate matrix distances
	    Can eventually be replaced with standalone __main__ and called with 'python gpcovariance path+filename'
	    :param path: path for input file, string
	    :param filename: filename, string
	    """
	    print("Reconstructing Covariance from file", filename)
	    hdr = self.read_response_hdr(filename)
	    self.hdr = hdr

	def _cache_square_distances(self):
	    """
	    Initialize (squared) distance matrix for stationary kernel.
	    """
	    xrange = (np.arange(0, self.Lpix) - self.Lpix/2.0) * self.pixscale
	    yrange = (np.arange(0, self.Lpix) - self.Lpix/2.0) * self.pixscale
	    self._xg, self._yg = np.meshgrid(xrange, yrange)
	    xr, yr = self._xg.ravel(), self._yg.ravel()
	    Dx = xr[:, np.newaxis] - xr[np.newaxis,:]
	    Dy = yr[:, np.newaxis] - yr[np.newaxis,:]
	    self.D2 = Dx**2 + Dy**2

	def _gpkernel(self, D2, gamma):
	    """2-D round RBF kernel, with length scale = standard deviation of
	    the PSF of a Gaussian process scene drawn from this kernel.
	    Same as in gpcubesolve_r1.py
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    """
	    return np.exp(-0.25 * D2/gamma**2)

	def _gpkernel_sparse(self, D2, gamma):
	    """2-D round sparse RBF kernel, defined in Melkumyan and Ramos, 2009
	    Same as in gpcubesolve_r1.py
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    """
	    D2 = np.sqrt(D2)
	    gamma = 4 * gamma
	    res = (2 + np.cos(2*np.pi * D2/gamma))/3.*(1-D2/gamma) + 1/(2.*np.pi) * np.sin(2*np.pi*D2/gamma)
	    res[D2>=gamma] = 0.
	    return res

	def _gpkernel_rational(self, D2, gamma):
	    ''' use rational quadratic, change alpha (currently set at at 2)
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    '''
	    alpha = 2.
	    return (1 + 0.25*D2/gamma**2/alpha)**(-alpha)

	def _gpkernel_wide(self, D2, gamma):
	    """ use sersic profile or sigmoi multiplication
	    Same as in gpcubesolve_r1.py
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    """
	    return np.exp(-0.25 * D2/gamma**2) * 1./ (1 + np.exp((4./gamma**2 *(np.sqrt(D2) - 0.5*gamma)**2)))
	    #return np.exp(-0.25 * D2**1.0/gamma**2)

	def _gpkernel_moffat(self, D2, gamma):
	    """ use sersic profile
	    Same as in gpcubesolve_r1.py
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    """
	    beta = 4.7
	    alpha = 4* 2/2.335 * gamma # 2* 2.335 * gamma
	    norm = (beta - 1.0)/(np.pi * alpha**2)
	    return norm * (1.0 + (D2)/alpha**2)**(-beta)

	def _gpkernel_matern32(self, D2, gamma):
	    ''' Matern3/2 kernel
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    '''
	    nu = 0.5*np.sqrt(3) * np.sqrt(D2) / gamma    
	    return (1 + nu) * np.exp(-nu)

	def _gpkernel_matern52(self, D2, gamma):
	    ''' Matern5/2 kernel
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    '''
	    nu = 0.5 * np.sqrt(5) * np.sqrt(D2) / gamma      
	    return (1 + nu + 0.25 * 5./3 * D2/gamma**2) * np.exp(-nu)

	def _gpkernel_exp(self, D2, gamma):
	    ''' Exponentia; kernel
	    :param D2: pairwise square distances
	    :param gamma: kernel length scale
	    '''    
	    return np.exp(-0.5 * np.sqrt(D2)/gamma)

	def read_cube(self, filename):
	    """ Read fits file and returns response matrix/cube, fibre data variance, and gamma 
	    :param path: input path
	    :param filename: filename 
	    """
	    hdulist = pf.open(filename)
	    resp = np.transpose(hdulist[0].data, (2,1,0))
	    hdulist.close()
	    return resp

	def read_response_hdr(self, filename):
	    """ Read fits file and returns response matrix/cube, fibre data variance, and gamma 
	    :param path: input path
	    :param filename: filename 
	    """
	    hdulist = pf.open(filename)  
	    hdr = hdulist[0].header 
	    hdulist.close()
	    return hdr


	def calc_sum(self, resp, nbin):
	    """ Calculating covariance matrix from response matrix cube, fibre variance, and GP gamma 
	    input must be obtained from read_response_cube()
	    :param resp: response matrix or cube, shape = (Nfibre*Nexposure,Npix*Npix,Nwavelenth/Nresponse)
	    :param var: variance of fibre data, shape= (Nexposure, Nfibre, Nwavelenth)
	    :param gamma: GP lengthscale hyperparameter, shape= (Nwavelenth)
	    :param wavenumber: number of element on spectral axis for which to calculate covariance
	    """
	    print('Summing over cube')
	    result = np.zeros((resp.shape[0],resp.shape[1],resp.shape[2]/nbin))
	    for i in range(resp.shape[2]/nbin):
	    	result[:,:,i] = np.sum(resp[:,:,i*nbin : (i+1)*nbin],axis =2)
	    return result


	def write_res(self, resarray, filename_out, silent = False, overwrite = False):
		"""
		Write reconstructed covariance matrix to csv or fits file
		: param resarray: 2d array of covariance matrix
		: param filename: full filename including path for output file
		: param silent: set True if comments should be suppressed (e.g if used in batch or for loop)
		: param overwrite: bool, if True overwrite the output file if it exists
		"""
		if filename_out is None:
			raise ValueError("Provide output filename")
		# Check if the object directory already exists or not.
		data = np.transpose(resarray, (2,1,0))
		hdu = pf.PrimaryHDU(data)
		hdu.header = self.hdr
		hdu.writeto(filename_out +'.fits', clobber = overwrite)
    	# Filename to write to
    	# if not silent: print("Covariance array is saved in", path_out + filename_out +".csv'")



#inputfile = '../../results/cubes_Feb/567624red_50pix_05arcsec_cube'
inputfile = '../../data/for_cube_testing_only/cubed/567624/567624_red_7_Y13SAR1_P002_15T006'
covar = Covariance(inputfile + '.fits')
# Read response matix cube:
resp =  covar.read_cube(inputfile + '.fits')
# Reconstruct covariance for one wavelength and pixel
res_covar = covar.calc_sum(resp, nbin = 4)
# Write reconstructed covariance to fits or csv file
covar.write_res(res_covar, filename_out = inputfile + '_4bins', overwrite = True)
