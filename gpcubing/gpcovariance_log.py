"""
Script for Reconstructing Covariance Matrix from file
Latest Change including marginalization over gamma
Nov 2018 Seb Haan

Please change settings in config_covar.py
and then execute python gpcovariance_r2.py
"""
from __future__ import print_function, division
import sys
import os
import numpy as np
from astropy.io import fits as pf
import astropy.wcs as pw
from scipy import linalg
import yaml
from settings_sami import _Rfib_arcsec
from config_covar import *


class Covariance():
    """
        Class for calculating the covariance matrix from saved components: 
        response matrix, GP gamma, and fibre data variance
        See for more details gpcubesolve.py for constructing cube and covariance
    """
    
    def __init__(self, filename, filename2):
        """ Readin of header information and calculate matrix distances
        Can eventually be replaced with standalone __main__ and called with 'python gpcovariance path+filename path+filename2'
        :param path: path for input file, string
        :param filename: filename, string
        :param filename2: filename, string
        """
        print("Reconstructing Covariance from file", filename, 'and', filename2)
        hdr = self.read_response_hdr(filename)
        self.Nexp = hdr['NEXP']
        self.Nfib = hdr['NFIB']
        self.Lpix = hdr['LPIX']
        self.pixscale = hdr['PIXSCALE']
        self.gpkernel_method = hdr['GPMETHOD']
        self.gpmarginalized = 'notdefined'
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

    def _gpkernel_mix(self, D2, gamma):
        ''' user specific mix of gp kernels
        :param D2: pairwise square distances
        :param gamma: kernel length scale
        '''    
        #beta = 4.7
        #alpha = 4* 2/2.335 * gamma # 2* 2.335 * gamma
        #norm = (beta - 1.0)/(np.pi * alpha**2)
        #moffat = norm * (1.0 + (D2)/alpha**2)**(-beta)
        return  np.exp(-0.125 * np.sqrt(D2)/gamma) * np.exp(-0.125 * D2/gamma**2) 

    def read_response_cube(self, filename):
        """ Read fits file and returns response matrix/cube, fibre data variance, and gamma 
        :param path: input path
        :param filename: filename 
        """
        hdulist = pf.open(filename)
        resp = np.transpose(hdulist[0].data, (2,1,0))
        variance = np.transpose(hdulist[1].data, (1,0))
        gamma = hdulist[2].data
        gp0 = hdulist[3].data
        gpoffset = hdulist[4].data
        hdulist.close()
        return resp, variance, gamma, gp0, gpoffset

    def read_data_cube(self, filename):
        """ Read fits file and returns cube with flux data
        :param path: input path
        :param filename: filename 
        """
        hdulist = pf.open(filename)
        data = np.transpose(hdulist[0].data, (2,1,0))
        hdulist.close()
        return data

    def read_response_hdr(self, filename):
        """ Read fits file and returns response matrix/cube, fibre data variance, and gamma 
        :param path: input path
        :param filename: filename 
        """
        hdulist = pf.open(filename)  
        hdr = hdulist[0].header 
        hdulist.close()
        return hdr


    def calc_covar(self, resp, var, gamma, gp0, gpoffset, ymean, wavenumber):
        """ Calculating covariance matrix from response matrix cube, fibre variance, and GP gamma 
        input must be obtained from read_response_cube()
        :param resp: response matrix or cube, shape = (Nfibre*Nexposure,Npix*Npix,Nwavelenth/Nresponse)
        :param var: variance of fibre data, shape= (Nexposure, Nfibre, Nwavelenth)
        :param gamma: GP lengthscale hyperparameter, shape= (Nwavelenth)
        :param gp0: GP amplitude, shape= (Nwavelenth)
        :param gpoffset: GP offset for log-transform of data, shape= (Nwavelenth)
        :param ymean: GP amplitude, shape= (Nwavelenth)
        :param wavenumber: number of element on spectral axis for which to calculate covariance
        """
        print('Calculating reconstructed covariance matrix at spectral slice', wavenumber, '...')
        gamma = np.asarray(gamma)
        Nresponse = int(len(gamma)/resp.shape[2])
        nresp = int(wavenumber // Nresponse) - 1
        if len(gamma) % resp.shape[2] != 0:
            raise ValueError("length of gamma array is not multiple of binned spectral lenght of response cube!")
        if len(gamma) > wavenumber:
            gamma_l = np.nanmean(gamma[wavenumber:wavenumber + Nresponse ])
            #gamma_l = gamma[nresp]
        else:
            raise ValueError("Wavenumber element must be smaller than length of gamma vector")
        # Nresponse is the integer number of steps that the response matrix has been calculated, default should be 32
        method_hash = { 'squared_exp': self._gpkernel,
                'rational':    self._gpkernel_rational,
                'sparse':      self._gpkernel_sparse,
                'wide':        self._gpkernel_wide,
                'moffat':      self._gpkernel_moffat,
                'matern32':    self._gpkernel_matern32,
                'matern52':    self._gpkernel_matern52,
                'exp':         self._gpkernel_exp,
                'mix':         self._gpkernel_mix, }
        if self.gpkernel_method in method_hash:
            gpKfunc = method_hash[self.gpkernel_method]
        else:
            print("Kernel method not supported, now taking default squared exponential!")
            gpKfunc = self._gpkernel
        Kxx = gpKfunc(self.D2, gamma_l)
        # calculate relative position of response matrix 
        gp0_l = gp0[nresp]
        A = resp[:,:,nresp]
        var_l = var[:, nresp]
        var_l = var_l.flatten()
        var_l[np.isnan(var_l)] = 1.e9

        ymean_w = ymean[:,:,nresp]
        ymean_w = ymean_w.flatten() + gpoffset[nresp]
        #offset = np.nanstd(y) # may need fixed standard deviation of entire data cube
        #scene_cov = np.zeros((self.Lpix**2, self.Lpix**2))
        Ky0 = np.dot(A, np.dot(Kxx, A.T)) 
        Ky = gp0_l * Ky0  + np.diag(var_l + 1e-6)
        Ky_chol = linalg.cholesky(Ky, lower=True)
        V = linalg.solve_triangular(Ky_chol, np.dot(A, gp0_l * Kxx), lower=True)
        log_cov = gp0_l * Kxx - np.dot(V.T, V)
        scene_cov = np.dot(np.diag(ymean_w), np.dot(np.exp(log_cov), np.diag(ymean_w))) * (np.exp(log_cov) - 1.)
        #scene_cov = np.outer(np.exp(log_gmean), np.exp(log_gmean)) * np.exp(log_cov) * (np.exp(log_cov) - 1.)
        return scene_cov

    def calc_covar_pixel(self, resp, var, gamma, gp0, gpoffset, ymean, wavenumber, xypixel = None, reduced = False):
        """ Calculating covariance matrix from response matrix cube, fibre variance, and GP gamma
        for a given wavelength and pixel. 
        Input must be obtained from read_response_cube()
        :param resp: response matrix or cube, shape = (Nfibre*Nexposure,Npix*Npix,Nwavelenth/Nresponse)
        :param var: variance of fibre data, shape= (Nexposure, Nfibre, Nwavelenth)
        :param gamma: GP lengthscale hyperparameter, shape= (Nwavelenth)
        :param gp0: GP amplitude, shape= (Nwavelenth)
        :param gpoffset: GP offset for log-transform of data, shape= (Nwavelenth)
        :param ymean: pixel flux, shape= (Npix,Npix)
        :param wavenumber: number of element on spectral axis for which to calculate covariance
        :param xypixel: (x,y) array row/coumn position of pixel for which covariance should be extracted, 
                        if None, entire covariance array between all pixels will be returned.
        :param reduced: if True return only covariance for 5x5 pixel cutout around  pixel selected, 
                    if False, return full covariance between selected  pixel and all other pixel in image
        """
        # Calculate first entire covariance matrix for specified wavelength
        res_covar = self.calc_covar(resp, var, gamma, gp0, gpoffset, ymean, wavenumber)
        # print('res_covar', res_covar)
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
            if (not os.path.exists(filename_out)) | (overwrite):
                np.savetxt(filename_out +'.csv', res_covar, delimiter=",")
        elif fileformat == 'fits':
            hdu = pf.PrimaryHDU(covar)
            hdu.header = self.hdr
            hdu.writeto(filename_out +'.fits', overwrite = overwrite)
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

    covar = Covariance(inputfile, inputfile2)
    # Read response matix cube:
    resp_c, var_c, gamma_c, gp0_c, gpoff_c =  covar.read_response_cube(inputfile)
    # Read data matix cube:
    ymean_c =  covar.read_data_cube(inputfile2)
    # Reconstruct covariance for one wavelength and pixel
    res_covar = covar.calc_covar_pixel(resp_c, var_c, gamma_c, gp0_c, gpoff_c, ymean_c, wavenumber = Nspectral, xypixel = xypixel, reduced = reduced)
    # Write reconstructed covariance to fits or csv file
    covar.write_res(res_covar, outputfile, fileformat = fileformat, overwrite = True)
