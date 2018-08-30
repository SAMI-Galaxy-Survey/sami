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
from scipy import linalg
from .config_covar import *


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
	    self.Nexp = hdr['NEXP']
	    self.Nfib = hdr['NFIB']
	    self.Lpix = hdr['LPIX']
	    self.pixscale = hdr['PIXSCALE']
	    self.gpkernel_method = hdr['GPMETHOD']
	    self._cache_square_distances()
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

	def read_response_cube(self, filename):
	    """ Read fits file and returns response matrix/cube, fibre data variance, and gamma 
	    :param path: input path
	    :param filename: filename 
	    """
	    hdulist = pf.open(filename)
	    resp = np.transpose(hdulist[0].data, (2,1,0))
	    variance = np.transpose(hdulist[1].data, (2,1,0))
	    gamma = hdulist[2].data
	    hdulist.close()
	    return resp, variance, gamma

	def read_response_hdr(self, filename):
	    """ Read fits file and returns response matrix/cube, fibre data variance, and gamma 
	    :param path: input path
	    :param filename: filename 
	    """
	    hdulist = pf.open(filename)  
	    hdr = hdulist[0].header 
	    hdulist.close()
	    return hdr


	def calc_covar(self, resp, var, gamma, wavenumber):
	    """ Calculating covariance matrix from response matrix cube, fibre variance, and GP gamma 
	    input must be obtained from read_response_cube()
	    :param resp: response matrix or cube, shape = (Nfibre*Nexposure,Npix*Npix,Nwavelenth/Nresponse)
	    :param var: variance of fibre data, shape= (Nexposure, Nfibre, Nwavelenth)
	    :param gamma: GP lengthscale hyperparameter, shape= (Nwavelenth)
	    :param wavenumber: number of element on spectral axis for which to calculate covariance
	    """
	    print('Calculating reconstructed covariance matrix at spectral slice', wavenumber, '...')
	    gamma = np.asarray(gamma)
	    Nresponse = len(gamma)/resp.shape[2]
	    if len(gamma) % resp.shape[2] != 0:
	    	raise ValueError("length of gamma array is not multiple of binned spectral lenght of response cube!")
	    if len(gamma) > wavenumber:
	    	gamma_l = gamma[wavenumber]
	    else:
	    	raise ValueError("Wavenumber element must be smaller than length of gamma vector")
	    # Nresponse is the integer number of steps that the response matrix has been calculated, default should be 32
	    if self.gpkernel_method == 'squared_exp':
	    	self.Kxx = self._gpkernel(self.D2, gamma_l)
	    elif self.gpkernel_method == 'sparse':
	    	self.Kxx = self._gpkernel_sparse(self.D2, gamma_l)
	    elif self.gpkernel_method == 'wide':
	    	self.Kxx = self._gpkernel_wide(self.D2, gamma_l)
	    elif self.gpkernel_method == 'moffat':
	    	self.Kxx = self._gpkernel_moffat(self.D2, gamma_l)
	    elif self.gpkernel_method == 'matern32':
	    	self.Kxx = self._gpkernel_matern32(self.D2, gamma_l)
	    elif self.gpkernel_method == 'matern52':
	    	self.Kxx = self._gpkernel_matern52(self.D2, gamma_l)
	    else:
	    	print("Kernel method not supported, take default squared exponential!")
	    	self.Kxx = self._gpkernel(self.D2, gamma_l)
	    # calculate relative position of response matrix 
	    nresp = wavenumber // Nresponse
	    A = resp[:,:,nresp]
	    var_l = var[:,:,wavenumber]
	    var_l = var_l.reshape(self.Nexp * self.Nfib)
	    Ky = np.dot(A, np.dot(self.Kxx, A.T)) + np.diag(var_l**2)
	    Ky_chol = linalg.cholesky(Ky, lower=True)
	    V = linalg.solve_triangular(Ky_chol, np.dot(A, self.Kxx), lower=True)
	    return self.Kxx - np.dot(V.T, V)

	def calc_covar_pixel(self, resp, var, gamma, wavenumber, xypixel = None, reduced = False):
		""" Calculating covariance matrix from response matrix cube, fibre variance, and GP gamma
		for a given wavelength and pixel. 
		Input must be obtained from read_response_cube()
		:param resp: response matrix or cube, shape = (Nfibre*Nexposure,Npix*Npix,Nwavelenth/Nresponse)
		:param var: variance of fibre data, shape= (Nexposure, Nfibre, Nwavelenth)
		:param gamma: GP lengthscale hyperparameter, shape= (Nwavelenth)
		:param wavenumber: number of element on spectral axis for which to calculate covariance
		:param xypixel: (x,y) array row/coumn position of pixel for which covariance should be extracted, 
						if None, entire covariance array between all pixels will be returned.
		:param reduced: if True return only covariance for 5x5 pixel cutout around  pixel selected, 
					if False, return full covariance between selected  pixel and all other pixel in image
		"""
		# Calculate first entire covariance matrix for specified wavelength
		res_covar = self.calc_covar(resp, var, gamma, wavenumber)
		if xypixel is not None:
			xypixel = np.asarray(xypixel)
			if len(xypixel) == 2:
				# Select only covariance image for selected pixel
				Npix = np.sqrt(res_covar.shape[0]).astype(int) # number of rows(columns in image)
				mask = np.zeros((Npix,Npix))
				mask[xypixel[0],xypixel[1]] = 1
				mask = mask.flatten()
				sel = np.where(mask == 1)
				res_sel = (res_covar[sel, :]).reshape(Npix, Npix)
				if reduced:
					res_cut = np.zeros((5,5))
					for i in range(5):
						idx = xypixel[0]-2 + i
						for j in range(5):
							idy = xypixel[1]-2 + j
							if (idx >= 0) & (idy >= 0) & (idx < Npix) & (idy < Npix):
								res_cut[i,j] = res_sel[idx,idy]
					res_sel = res_cut
			else:
				raise ValueError('x and y pixel not given')

		else: 
			res_sel = res_covar
		return res_sel



	def write_res(self, covar, filename_out, fileformat = 'fits', silent = False, overwrite = False):
		"""
		Write reconstructed covariance matrix to csv or fits file
		: param covar: 2d array of covariance matrix
		: param filename: full filename including path for output file
		: param fileformat: either 'csv' or 'fits'
		: param silent: set True if comments should be suppressed (e.g if used in batch or for loop)
		: param overwrite: bool, if True overwrite the output file if it exists
		"""
		if filename_out is None:
			raise ValueError("Provide output filename")
		# Check if the object directory already exists or not.
		if fileformat == 'csv':
			if not os.path.exists(filename_out) | overwrite:
				np.savetxt(filename_out +'.csv', res_covar, delimiter=",")
		elif fileformat == 'fits':
			hdu = pf.PrimaryHDU(covar)
			hdu.header = self.hdr
			hdu.writeto(filename_out +'.fits', clobber = overwrite)
		else:
			print('Format for output file incorrect. Select csv or fits')
    	# Filename to write to
    	# if not silent: print("Covariance array is saved in", path_out + filename_out +".csv'")



if __name__ == "__main__":
	"""
	reads in covariance file and writes reconstructed covariance at a certain wavelength slice to csv or fits file
	"""
	if (xpix is not None) and (ypix is not None):
		xypixel = np.asarray([xpix, ypix]).astype(int)
	else:
		xypixel = None
	covar = Covariance(inputfile)
	# Read response matix cube:
	resp_c, var_c, gamma_c =  covar.read_response_cube(inputfile)
	# Reconstruct covariance for one wavelength and pixel
	res_covar = covar.calc_covar_pixel(resp_c, var_c, gamma_c, wavenumber = Nspectral, xypixel = xypixel, reduced = reduced)
	# Write reconstructed covariance to fits or csv file
	covar.write_res(res_covar, outputfile, fileformat = fileformat, overwrite = True)
