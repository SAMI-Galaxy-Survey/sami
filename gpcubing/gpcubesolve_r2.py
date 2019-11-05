""" Main functions for calculating respoinse matrix, GP, and cubing
Seb Haan 21 March 2018, still experimental
"""
from __future__ import print_function
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack, interpolate, linalg, signal, optimize
from .settings_sami import _Nfib, _Rfib_arcsec, _plate_scale

fwhm_2_std = 1. / (2.*np.sqrt(2*np.log(2.)))

np.random.seed(3121)


class CubeError(Exception):
    """
    General exception class for stuff in here
    """

    def __init__(self, msg):
        self.msg = msg

    def __unicode__(self):
        return self.msg


class VerboseMessager(object):
    """
    Class for flushed verbose debugging
    """

    def __init__(self, verbose):
        self.verbose = verbose

    def __call__(self, msg):
        if self.verbose:
            print(msg)
            sys.stdout.flush()


def profile_time(func, *args, **kwargs):
    """
    Measures a function's execution time.
    :param func: function to time
    :param args: ordered arguments to pass to func
    :param kwargs: keyword arguments to pass to func
    """
    t0 = time.time()
    result = func(*args, **kwargs)
    t1 = time.time()
   # print "time to run:  {}:  {:.3f} sec".format(func.__name__, t1 - t0)
    return result

def rebin(A, rb):
    """
    Rebins each axis of an array by a factor rb.  Right now rb must
    evenly divide each axis of the array, though in future more general
    rebinning patterns may be supported.
    :param A: array to rebin
    :param rb: rebinning factor (positive integer)
    """
    # Sanity checks
    if not isinstance(rb, int) or rb <= 0:
        raise CubeError("rb must be positive integer in rebin()")
    for i, dim in enumerate(A.shape):
        if dim % rb != 0:
            raise CubeError("rb doesn't evenly divide axis {} of array"
                            .format(i))
        new_shape = list(A.shape[:i])
        new_shape.extend([dim//rb, rb])
        new_shape.extend(list(A.shape[i+1:]))
        A = A.reshape(new_shape).sum(axis=i+1)
    return A

def gaussian_psf(dx, dy, sigma):
    """
    Round Gaussian PSF model, normalized to unit flux.
    :param dx: distance from PSF center along x-axis, in pixels
    :param dy: distance from PSF center along y-axis, in pixels
    :param sigma: standard deviation of Gaussian in pixels
    """
    return np.exp(-0.5*(dx**2 + dy**2)/sigma**2)/(2*np.pi*sigma)

def moffat_psf(dx, dy, alpha, beta):
    """
    Round Moffat PSF model, normalized to unit flux.
    :param dx: distance from PSF center along x-axis, in pixels
    :param dy: distance from PSF center along y-axis, in pixels
    :param alpha: core width parameter of Moffat in pixels
    :param beta: wings slope parameter of Moffat (dimensionless)
    """
    norm = (beta - 1.0)/(np.pi * alpha**2)
    return norm * (1.0 + (dx**2 + dy**2)/alpha**2)**(-beta)

def moffat_ellip_psf(dx, dy, alpha_x, alpha_y, rho, beta):
    """
    Round Moffat PSF model, normalized to unit flux.
    :param dx: distance from PSF center along x-axis, in pixels
    :param dy: distance from PSF center along y-axis, in pixels
    :param rho: ellpticity?
    :param alpha_x: core width parameter of Moffat in x-direction in pixels
    :param alpha_y: core width parameter of Moffat in y-direction in pixels
    :param beta: wings slope parameter of Moffat (dimensionless)
    """
    norm = (beta - 1) / (np.pi * alpha_x * alpha_y * np.sqrt(1 - rho**2))
    return norm * (1.0 + ((dx/alpha_x)**2 + (dy/alpha_y)**2 - 2*rho*dx/alpha_x*dy/alpha_y) / (1 - rho**2))**(-beta)

def sig2alpha(beta):
    """
    Given the beta parameter for a Moffat profile, solves for the ratio
    of alpha to the standard deviation of a Gaussian with the same FWHM.
    """
    return np.sqrt(2*np.log(2)/(2**(1.0/beta) - 1.0))

def fake_scene(Lpix, xsrc, ysrc, psf, *pars):
    """
    Generates a fake scene with a single round Gaussian source in it.
    :param Lpix: integer length of side of reconstructed scene,
        in pixels (assumed square)
    :param xsrc: x position of source in pixels (0 <= xsrc < Lpix)
    :param ysrc: y position of source in pixels (0 <= xsrc < Lpix)
    :param psf: PSF function of source (2-D callable + parameters)
    :param pars: parameters to pass to PSF
    :return: np.array of floats w/shape (Lpix, Lpix)
    """
    x, y = np.meshgrid(np.arange(Lpix) + 0.5, np.arange(Lpix) + 0.5)
    return psf(x - xsrc, y - ysrc, *pars)


# ---------------------------------------------------------------------------------------
#  The Response Matrix for circular fiber-optics geometry (below for AAO SAMI survey)
#  Change accordingly below for other instrumental geometry.
# ---------------------------------------------------------------------------------------


class ResponseMatrix(object):
    """
    Class for building response matrices.  All workhorse code due to Ned.
    """

    def __init__(self, coord, dar_cor, psf, Lpix, pixscale, fft=True, avgpsf=False, _Nexp = 7):
        """
        :param coord: array with (x,y) coordinates of fibres
        :param dar_cor: array with (x,y) DAR corrections, same shape as coord
        :param psf: callable accepting two ndarrays (x,y) of same shape;
            interpret it as convolution kernel to go from reconstructed
            scene to observed scene -- i.e. not strictly equal to the
            observed PSF if reconstructed scene has finite resolution!
        :param Lpix: integer length of side of reconstructed scene,
            in pixels (assumed square)
        :param pixscale: pixel scale in arcsec/pix
        :param fft: use FFTs instead of oversampling for convolution
        :param avgpsf: use mean psf of 7 exposures
        """
        self.psf = psf
        self.Lpix = Lpix
        self.pixscale = pixscale
        self.fft = fft
        self.avgpsf = avgpsf
        self._Nexp = _Nexp
        # Update fibre positions and convert to coordinates in scene with DAR offsets:
        self._xfib = (coord[0] - coord[0].mean() + dar_cor[0])/pixscale + 0.5*Lpix - 1.0
        self._yfib = (coord[1] - coord[1].mean() + dar_cor[1])/pixscale + 0.5*Lpix + 1.0
        # or without DAR correction
        #self._xfib = (coord[0] - coord[0].mean())/pixscale  + 0.5*Lpix 
        #self._yfib = (coord[1] - coord[1].mean())/pixscale  + 0.5*Lpix

    def _subsample_fiber(self, nrings, radius):
        """
        Generates an evenly spaced point cloud within the circumference
        of a fibre footprint.  Used in numerical integrals over PSFs.
        :param nrings: integer number of concentric annuli
        :param radius: radius of fibre in pixels (of reconstructed scene)
        :return: two 1-D ndarrays with (x,y) coordinates of point cloud
        """
        print('\n_subsample_fiber: note that this only gets run once.')
        delta = 1. / (nrings - 0.5)
        radii = np.arange(0., 1., delta)
        rotang = 0.
        for ri, ringrad in enumerate(radii):
            # Step through annuli
            npoints = int(np.round((2*np.pi*ringrad) / delta))
            tpoints = np.linspace(0., 2.*np.pi, npoints+1) + rotang
            # Only one point at the centre
            if ri == 0 :
                rad = np.zeros(1)
                theta = np.zeros(1)
            # Larger annuli have angular offsets for even spatial sampling
            else :
                rotang += (tpoints[1] - tpoints[0]) / 2.
                rad = np.hstack((rad, np.ones(npoints)*ringrad))
                theta = np.hstack((theta, tpoints[:-1]))
        xsub = radius * rad * np.cos(theta)
        ysub = radius * rad * np.sin(theta)
        print('_subsample_fiber: %i points in %i rings.' % (
            xsub.size, nrings ))
        return xsub, ysub


    def _point_response_fft(self, *pars):
        """
        Richard:
        Same as _point_response_matrix from Ned's simulation tool, except uses FFT to compute the
        convolution.
        Answers the question:  what fraction of a PSF, placed at each pixel
        center, does each fiber see?  (Convolves PSF with fibre acceptance.)
        Uses FFT convolutions to generate response matrix elements.  Fast.
        Sets self._response to a np.array of shape (_Nexp, _Nfib, Lpix, Lpix).
        Still experimental
        :param *pars:  accepts positional parameters for the PSF callable
        """
        vmsg = VerboseMessager(verbose=False)

        # Set up an image in pixel *coordinates*, but oversampled.
        # 16 arcsec max radius x 0.08 arcsec/grid x 2 = 400 x 400
        # We're going to over-oversample this below by another factor of rb,
        # so make sure the grid size is a "fast FFT" number divisible by rb.
        t0 = time.time()
        rb = 10 
        gridscale = 0.08/self.pixscale
        Lgrid = fftpack.next_fast_len(int(2.5*self.Lpix/gridscale))
        while Lgrid % rb != 0:
            Lgrid = fftpack.next_fast_len(Lgrid + 1)
        xrange = (np.arange(0, Lgrid) - 0.5*Lgrid) * gridscale
        yrange = (np.arange(0, Lgrid) - 0.5*Lgrid) * gridscale
        xg, yg = np.meshgrid(xrange, yrange)
        t1 = time.time()
        vmsg("time to set up oversampled grid:     {:.3f} sec".format(t1-t0))

        # Experimental:  Can we make this any smaller?  Truncate the PSF to
        # some radius that holds most of the light.
        # The next few lines calculate the actual cut-off PSF fraction,
        # but accuracy doesn't seem to depend strongly on this.
        """
        ri = np.arange(0, Lpix/2, 0.2)
        psf_ray = self.psf(ri, 0, *pars)
        psf_int = (psf_ray*ri).cumsum() / (psf_ray*ri).sum()
        rc = 2*min(ri[psf_int > 0.999]) + _Rfib_arcsec/self.pixscale
        """
        # Just use a box of fixed size.  0.4*Lpix seems to give quite good
        # accuracy over a range of conditions, which seems weird because
        # it's a sweet spot in the middle of several less optimal choices
        # (as measured by consistency with the point cloud method).
        rc = 0.4*self.Lpix
        Lfft = fftpack.next_fast_len(int(2.5*rc/gridscale))
        while Lfft % rb != 0:
            Lfft = fftpack.next_fast_len(Lfft + 1)
        ifft = (Lgrid - Lfft)//2
        xg = xg[ifft:ifft+Lfft, ifft:ifft+Lfft]
        yg = yg[ifft:ifft+Lfft, ifft:ifft+Lfft]
        if self.avgpsf:
            psf_image = self.psf(xg, yg, *pars)
            psf_image[(xg**2 + yg**2) > (xg.max())**2] = 0.
            #avoid sharp cut in psf truncation above and use:
            #psf_image = psf_image * 1./(1 + np.exp((4./xg.max())**2 *(xg**2 + yg**2 - (0.75*xg.max())**2)))
            #plt.clf()
            #plt.imshow(psf_image)
            #plt.colorbar()
            #plt.savefig('psfimage.png')
        else:
            psf_image = np.zeros((self._Nexp, xg.shape[0], xg.shape[1]))
            pars = np.asarray(pars)
            for i in range(self._Nexp):
                psf_image[i] = self.psf(xg, yg, pars[0,i], pars[1,i]) 
                #psf_image[i] = self.psf(xg, yg, *pars[:,i]) 
                psf_image[i, (xg**2 + yg**2) > (xg.max())**2] = 0.
                #avoid sharp cut in psf truncation above and use:
                #psf_image[i] = psf_image * psf_image * 1./(1 + np.exp((4./xg.max())**2 *(xg**2 + yg**2 - (0.75*xg.max())**2)))
            #plt.clf()
            #plt.imshow(psf_image[0])
            #plt.colorbar()
            #plt.savefig('psfimage.png')

        vmsg("Lgrid = {}, Lfft = {}, rb = {}".format(Lgrid, Lfft, rb))
        t2 = time.time()
        vmsg("time to truncate PSF:                {:.3f} sec".format(t2-t1))

        # Calculate the fibre acceptance.  This is just uniform on a circle,
        # but to get the boundary right we'll oversample by some additional
        # factor that neatly divides into our coordinate grid.
        fibre_subsamp = np.zeros((Lfft, Lfft))
        _Rfib_rebin = rb*_Rfib_arcsec/self.pixscale
        #assuming infinite sharp edge of fiber (comment line below in/out):
        fibre_subsamp[xg**2 + yg**2 < _Rfib_rebin**2] = 1.0  
        #smooth edges of fiber to avoid sidelobes in fft with 2D sigmoid function (comment line below in/out):
        #fibre_subsamp = 1./(1 + np.exp((4./_Rfib_rebin)**2 *(xg**2 + yg**2 - (0.75*_Rfib_rebin)**2)))
        fibre_subsamp = rebin(fibre_subsamp, rb)
        fibre_accept = np.zeros((Lfft, Lfft))
        Lgrb, igrb = Lfft//rb, (Lfft - Lfft//rb)//2
        fibre_accept[igrb:igrb+Lgrb, igrb:igrb+Lgrb] = fibre_subsamp
        fibre_accept /= fibre_accept.sum()
        t3 = time.time()
        vmsg("time to calculate fibre acceptance:  {:.3f} sec".format(t3-t2))

        # Convolve the fibre acceptance by the PSF to produce an oversampled
        # image of the instrumental response.  
        #signal.fftconvolve can take n-dimesnional PSF convolutions, but too heavy on memory, that's why we do it in for loop
        if self.avgpsf:
            conv = signal.fftconvolve(fibre_accept, psf_image, mode='same')
        else:
            conv = []
            for i in range(self._Nexp):
                conv_i = signal.fftconvolve(fibre_accept, psf_image[i], mode='same') # can take n-dimensional arrays, in case of sami 7 times 2d arrays
                conv.append(conv_i)
        t4 = time.time()
        vmsg("time to calculate FFT convolution:   {:.3f} sec".format(t4-t3))
        if self.avgpsf:
            response_image = np.zeros((Lgrid, Lgrid))
            response_image[ifft:ifft+Lfft, ifft:ifft+Lfft] = conv
        else:
            response_image = np.zeros((psf_image.shape[0], Lgrid, Lgrid))
            response_image[:,ifft:ifft+Lfft, ifft:ifft+Lfft] = conv


        t5 = time.time()
        vmsg("time to set up interpolator:         {:.3f} sec".format(t5-t4))

        # Run through a loop similar to the one in subsample_fiber(),
        # where we march the SAMI footprint across each pixel in the scene.
        response = np.zeros(self._xfib.shape + (self.Lpix, self.Lpix))  # sami shape: (7, 61, 50, 50) 
        if self.avgpsf:
            for xi in range(self.Lpix):
                dx = 0.5 + xi - self._xfib.ravel()
                ix = np.array(dx/gridscale + Lgrid/2, dtype=int)
                for yi in range(self.Lpix):
                    dy = 0.5 + yi - self._yfib.ravel()
                    iy = np.array(dy/gridscale + Lgrid/2, dtype=int)
                    response_int = response_image[ix, iy]
                    response[..., xi, yi] = response_int.reshape(self._Nexp, _Nfib)
        else: 
            for iexp in range(self._Nexp):
                for xi in range(self.Lpix):
                    dx = 0.5 + xi - self._xfib[iexp,:] 
                    ix = np.array(dx/gridscale + Lgrid/2, dtype=int)
                    for yi in range(self.Lpix):
                        dy = 0.5 + yi - self._yfib[iexp,:] 
                        iy = np.array(dy/gridscale + Lgrid/2, dtype=int)
                        response[iexp, :, xi, yi] = response_image[iexp, ix, iy]
        t6 = time.time()
        vmsg("time to evaluate interpolation:      {:.3f} sec".format(t6-t5))
        vmsg("-----------------------------------------------")
        vmsg("total time to run:                   {:.3f} sec".format(t6-t0))

        
        # responsesum = np.zeros((self.Lpix,self.Lpix))
        # for i in range(61):
        #     responsesum = responsesum + response[0,i,:,:]
        # plt.clf()
        # plt.imshow(responsesum)
        # plt.colorbar()
        # plt.savefig('responseimage.png')
        # plt.clf()
        # plt.imshow(response[0,0])
        # plt.colorbar()
        # plt.savefig('responseimage_single.png')
        return response

    def get_packed(self, *pars):
        """
        Returns a "packed" response matrix, an ndarray of floats with
        shape (_Nexp, _Nfib, Lpix, Lpix); this is basically the same as
        in Ned's original notebook implementation.
        :param pars:  parameter vector to be passed to the PSF
        """
        if self.fft:    # build response matrix using FFT convolution
            return profile_time(self._point_response_fft, *pars)
        else:           # build response matrix using Ned's point cloud method
            return profile_time(self._point_response_matrix, *pars)

    def get_unpacked(self, *pars):
        """
        Returns an "unpacked" response matrix, an ndarray of floats with
        shape (_Nexp*_Nfib, Lpix*Lpix); this is the form that's useful
        for GP implementations where the parameter vector is a raster
        scan of a square scene.
        :param pars:  parameter vector to be passed to the PSF
        """
        response = self.get_packed(*pars)
        return response.reshape(self._Nexp*_Nfib, self.Lpix**2)

    def __call__(self, *pars):
        """
        A convenience wrapper for ResponseMatrix.get_unpacked(), given
        that we won't need the get_packed() version too often.
        :param pars:  parameter vector to be passed to the PSF
        """
        return self.get_unpacked(*pars)


class SliceView(object):
    """
    Interface for a single wavelength slice of a SAMI datacube.
    Contains a representation of the model that can plot itself, and can
    unpack itself to interface with various samplers.
    """

    def __init__(self, response):
        """
        :param response: response matrix
        :param kernel_scale: correlation length of GP scene
        :param seeing_fwhm: seeing FWHM in arcseconds
        """
        self.Lpix = response.Lpix
        self.pixscale = response.pixscale
        self.scene = np.zeros((self.Lpix, self.Lpix))
        self.variance = np.zeros((self.Lpix, self.Lpix))
        self.response = response
        self.fibflux = None
        self.fibfluxerr = None

    def unpack_scene(self):
        """
        Return current scene as 1-d parameter vector.
        :returns:  np.array, shape = (L_pix**2, )
        """
        return self.scene.reshape(-1,), self.variance.reshape(-1,)

    def pack_scene(self, parvec):
        """
        Sets internal scene variable to contents of parameter vector.
        :param parvec:  np.array, shape = (L_pix**2, )
        """
        if parvec.shape != (self.Lpix**2, ):
            raise CubeError("param vector shape incompatible w/scene shape")
        self.scene = parvec.reshape(self.Lpix, self.Lpix)

    def pack_scene_var(self, parvec):
        """
        Sets internal scene variable to contents of parameter vector.
        :param parvec:  np.array, shape = (L_pix**2, )
        """
        if parvec.shape != (self.Lpix**2, ):
            raise CubeError("param vector shape incompatible w/scene shape")
        self.variance = parvec.reshape(self.Lpix, self.Lpix)

    def pack_scene_cov(self, parvec):
        """
        Sets internal scene variable to contents of parameter vector.
        :param parvec:  np.array, shape = (L_pix**2, )
        """
        #if parvec.shape != (self.Lpix**4, ):
        #    raise CubeError("covariance param vector shape incompatible w/scene shape")
        #self.covariance = parvec.rshape(self.Lpix, self.Lpix, self.Lpix, self.Lpix)
        self.covariance = parvec

    def predict_fibflux(self, *psfpars):
        """
        Predict fibre fluxes based on current scene and response matrix.
        Useful for simulating data.  Caches results in self.fibflux.
        :param psfpars: parameters to pass to PSF of response matrix
        """
        f = np.dot(self.response(*psfpars), self.unpack_scene())
        self.fibflux = f.reshape(self._Nexp, _Nfib)

    def add_fibnoise(self, sigabs=None, snr=None):
        """
        Adds random iid Gaussian noise to the cached fibre fluxes.
        :param sigabs: standard deviation of noise in absolute units
        :param snr: desired peak signal-to-noise of simulated fibre flux
            (i.e. peak fibre flux divided by standard deviation)
        """
        if sigabs is None and snr is None:
            raise CubeError("no noise specified in SliceView.add_fibnoise()")
        elif sigabs is not None and snr is not None:
            raise CubeError("specify only sigabs or snr, "
                            "but not both, in SliceView.add_fibnoise()")
        elif sigabs is not None:
            fibnoise = sigabs
        elif snr is not None:
            fibnoise = (self.fibflux.max() / snr)
        else:
            raise CubeError("shouldn't get here in SliceView.add_fibnoise()")
        self.fibflux += fibnoise * np.random.normal(size=fibflux.shape)

    def plot(self):
        """
        Displays a diagnostic plot of the (cached) scene.
        """
        plt.clf()
        plt.subplot(121, aspect='equal')
        plt_extent = (0, self.Lpix, 0, self.Lpix)
        plt.imshow(self.scene, extent=plt_extent)
        plt.colorbar()
        plt.subplot(122, aspect='equal')
        plt.scatter(self.response._xfib,
                    self.Lpix - self.response._yfib, 32, self.fibflux)
        plt.xlim(0, self.Lpix)
        plt.ylim(0, self.Lpix)
        plt.colorbar()
        plt.show()


class GPModel(SliceView):
    """
    Models a single wavelength slice of a datacube as a Gaussian
    process, using the instrument response to transform the GP prior.
    """

    def __init__(self, response, fibflux, fibfluxerr, calcresponse, gpmethod = 'squared_exp'):
        """
        :param response: response matrix
        :param fibflux: fibre fluxes (np.array of shape (_Nexp, _Nfib))
        :param fibfluxerr: standard deviations of fibre fluxes
            (either float, or np.array of shape (_Nexp, _Nfib)
        :param calcresponse: If True, recalculate response matrix; if False use response matrix of previosu step
        :param gpmethod: specify GP kernel used: 'squared_exp' (Default)
                                            'sparse', 
                                            'wide', 
                                            'moffat',
                                            'matern32',
                                            'matern52'
                Warning: Most tests done with squared_exp kernel, for other kernels only limited testing done.
        """
        # Explicitly call superclass constructor to make sure it gets done
        super(GPModel, self).__init__(response)
        # Initialize other class stuff
        # Assume zero mean function for now
        self._fmean, self._fstd = 0.0, fibflux.std()
        if self._fstd > 0.:
            self.fibflux = (fibflux - self._fmean)/self._fstd
            self.fibfluxerr = fibfluxerr/self._fstd
        else:
            self.fibflux = fibflux
            self.fibfluxerr = fibfluxerr
        self._cache_square_distances()
        self.calcresponse = calcresponse
        self._gpmethod = gpmethod

    def _cache_square_distances(self):
        """
        Initialize (squared) distance matrix for stationary kernel.
        Assuming the user wants to predict results on the same grid as
        the latent points for training, this needs to be done only once.
        Store square distances for RBF and/or Moffat kernels/PSFs.
        """
        xrange = (np.arange(0, self.Lpix) - self.Lpix/2.0) * self.pixscale
        yrange = (np.arange(0, self.Lpix) - self.Lpix/2.0) * self.pixscale
        self._xg, self._yg = np.meshgrid(xrange, yrange)
        xr, yr = self._xg.ravel(), self._yg.ravel()
        Dx = xr[:, np.newaxis] - xr[np.newaxis,:]
        Dy = yr[:, np.newaxis] - yr[np.newaxis,:]
        self.D2 = Dx**2 + Dy**2

    def fit(self):
        pass

    def _gpkernel(self, D2, gamma):
        """
        2-D round RBF kernel, with length scale = standard deviation of
        the PSF of a Gaussian process scene drawn from this kernel.
        :param D2: pairwise square distances
        :param gamma: kernel length scale
        """
        # The constant of 0.25 below (instead of 0.5) ensures that the kernel
        # represents the PSF standard deviation.  To see why, note that the
        # convolving a GP with some PSF gives a GP with a kernel convolved by
        # that PSF on both inputs.  An infinite-resolution scene has a
        # delta-function kernel, so convolving by a Gaussian PSF of standard
        # deviation sigma gives a kernel w/standard deviation sqrt(2)*sigma.
        return np.exp(-0.25 * D2/gamma**2)

    def _gpkernel_sparse(self, D2, gamma):
        """
        2-D round sparse RBF kernel, defined in Melkumyan and Ramos, 2009
        :param D2: pairwise square distances
        :param gamma: kernel length scale
        """
        D2 = np.sqrt(D2)
        gamma = 4 * gamma
        res = (2 + np.cos(2*np.pi * D2/gamma))/3.*(1-D2/gamma) + 1/(2.*np.pi) * np.sin(2*np.pi*D2/gamma)
        res[D2>=gamma] = 0.
        return res

    def _gpkernel_wide(self, D2, gamma):
        ''' use sersic profile or sigmoi multiplication
        :param D2: pairwise square distances
        :param gamma: kernel length scale
        '''
        return np.exp(-0.25 * D2/gamma**2) * 1./ (1 + np.exp((4./gamma**2 *(np.sqrt(D2) - 0.5*gamma)**2)))
        #return np.exp(-0.25 * D2**1.0/gamma**2)

    def _gpkernel_rational(self, D2, gamma):
        ''' use rational quadratic, change alpha (currently set at at 2)
        :param D2: pairwise square distances
        :param gamma: kernel length scale
        '''
        alpha = 2.
        return (1 + 0.25*D2/gamma**2/alpha)**(-alpha)

    def _gpkernel_moffat(self, D2, gamma):
        ''' use sersic profile
        :param D2: pairwise square distances
        :param gamma: kernel length scale
        '''
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
        ''' Matern5/2 kernel
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
        return  np.exp(-0.125 * np.sqrt(D2)/gamma) * np.exp(-0.125 * D2/gamma**2)  # * np.exp(-(np.sin(np.pi*np.sqrt(D2)/gamma))**2/gamma**2)


    def logL(self, hp):
        """
        Evaluates the marginal log likelihood of the data.
        :param hp: (hyper-)parameter 1-D np.array of floats
        """
        # Generate covariances and other relevant parameters and cache them,
        # since we'll need them again for prediction.
        psf_pars, gamma = hp[:-1], hp[-1]

        method_hash = { 'squared_exp': self._gpkernel,
                'rational':    self._gpkernel_rational,
                'sparse':      self._gpkernel_sparse,
                'wide':        self._gpkernel_wide,
                'moffat':      self._gpkernel_moffat,
                'matern32':    self._gpkernel_matern32,
                'matern52':    self._gpkernel_matern52,
                'exp':         self._gpkernel_exp,
                'mix':         self._gpkernel_mix, }
        if self._gpmethod in method_hash:
            gpKfunc = method_hash[self._gpmethod]
        else:
            print("Kernel method not supported, now taking default squared exponential!")
            self._Kxx = self._gpkernel(self.D2, gamma)
        if self.calcresponse:
            self._A = self.response(*psf_pars)
        y, yerr = self.fibflux, self.fibfluxerr
        Ky = np.dot(self._A, np.dot(self._Kxx, self._A.T)) + np.diag(yerr**2) # Shape(427,427)


        # The marginal likelihood is a multivariate Gaussian, of the form
        #     log(L) = -0.5 * (y.T*(Ky^-1)*y + log(det(Ky)) + n*log(2*pi))
        # with * above meaning matrix multiplication (np.dot) for matrices.
        # This system is usually solved by taking the Cholesky factor of Ky,
        # a triangular matrix satisfying Ky = np.dot(Kychol, Kychol.T),
        # in order to form both y.T*(Ky^-1)*y = u.T*u and the log det term.

        self._Ky_chol = linalg.cholesky(Ky, lower=True) # Shape(427,427)
        self._u = linalg.solve_triangular(self._Ky_chol, y, lower=True)
        log_det_Ky = np.log(np.diag(self._Ky_chol)**2).sum()
        n_log_2pi = self.Lpix**2 * np.log(2 * np.pi)
        result = -0.5 * (np.dot(self._u, self._u) + log_det_Ky + n_log_2pi)
        sys.stdout.flush()
        return result

    def logL_marg(self, hp, verbose=False, kl_approx=False):
        """
        Version of logL that marginalizes stuff out.
        WARNING:  this code does a numeric marginalization over possible
        values of gamma, and so it predicts the marginalized posterior
        in addition to everything else it does.  It leaves the internal
        variables self.scene and self.scene_cov in a state that is NOT
        consistent with using self.predict(), so calling self.predict()
        after calling this will give you UNPREDICTABLE, WRONG results.
        Instead, call this and then just access self.scene and
        self.scene_cov directly.  There may be better ways to do this
        but for now this is what the experimental code does.
        :param hp: (hyper-)parameter 1-D np.array of floats
        :param verbose: print status messages (esp. time taken)?
        :param kl_approx: experimental Gaussian posterior approximation
            that minimizes K-L divergence from the grid on which logL
            is calculated; intended to save cycles but doesn't really
        """
        vmsg = VerboseMessager(verbose=verbose)
        t0 = time.time()

        # Generate the response matrix from the hyperparameters.
        y, yerr = self.fibflux, self.fibfluxerr
        psf_pars, gdum = hp[:-1], hp[-1]
        if self.calcresponse:
            A = self._A = self.response(*psf_pars)
        else:
            A = self._A
        Lpix = self.response.Lpix
        pixscale = self.response.pixscale
        t1 = time.time()
        vmsg("time to set up response matrix:      {:.3f} sec".format(t1-t0))
        sys.stdout.flush()

        # Define the hyperparameter grid, wide range:
        #dG, dlogS = 0.2, 0.05
        #self._gamvals = np.arange(0.01, 5.0/pixscale, dG)
        #self._Svals = 10**np.arange(-2.0, 1.0, dlogS)
        # Define the hyperparameter grid, small range:
        dG, dlogS = 0.2, 0.2
        self._gamvals = np.arange(0.1, 1.0/pixscale, dG)
        self._Svals = 10**np.arange(-1.0, 1.0, dlogS)
       

        # Look up which of the many available kernels we're using
        method_hash = { 'squared_exp': self._gpkernel,
                        'rational':    self._gpkernel_rational,
                        'sparse':      self._gpkernel_sparse,
                        'wide':        self._gpkernel_wide,
                        'moffat':      self._gpkernel_moffat,
                        'matern32':    self._gpkernel_matern32,
                        'matern52':    self._gpkernel_matern52,
                        'exp':         self._gpkernel_exp,
                        'mix':         self._gpkernel_mix, }
        if self._gpmethod in method_hash:
            gpKfunc = method_hash[self._gpmethod]
        else:
            print("Kernel method not supported, now taking default squared exponential!")
            gpKfunc = self._gpkernel

        # Cache the grid of GP kernel matrices and their products with A.
        # We only need to do this if we've updated the response matrix.
        if self.calcresponse:
            self._K_gv, self._AK_gv, self._AKA_gv = [ ], [ ], [ ]
            for gamma in self._gamvals:
                self._K_gv.append(gpKfunc(self.D2, gamma))
                self._AK_gv.append(np.dot(A, self._K_gv[-1]))
                self._AKA_gv.append(np.dot(self._AK_gv[-1], A.T))
            self._K_gv = np.array(self._K_gv)
            self._AK_gv = np.array(self._AK_gv)
            self._AKA_gv = np.array(self._AKA_gv)
        t2 = time.time()
        vmsg("time to set up GP kernel cache:      {:.3f} sec".format(t2-t1))
        sys.stdout.flush()

        # Keep a record of parameter values during the grid search, on the
        # expectation that only a small number of them will matter and so
        # recalculating at the end should be only a small fraction of runtime.
        # it will cost us relatively little to recalculate at the end.
        evalpars = [ ]

        # The marginal likelihood is a multivariate Gaussian, of the form
        #     log(L) = -0.5 * (y.T*(Ky^-1)*y + log(det(Ky)) + n*log(2*pi))
        # with * above meaning matrix multiplication (np.dot) for matrices.
        # This system is usually solved by taking the Cholesky factor of Ky,
        # a triangular matrix satisfying Ky = np.dot(Kychol, Kychol.T),
        # in order to form both y.T*(Ky^-1)*y = u.T*u and the log det term.

        for i, gamma in enumerate(self._gamvals):
            for j, S in enumerate(self._Svals):
                # Given gamma, S, and A, calculate the log likelihood
                Ky = S**2 * self._AKA_gv[i] + np.diag(yerr**2)
                Ky_chol = linalg.cholesky(Ky, lower=True)
                u = linalg.solve_triangular(Ky_chol, y, lower=True)
                log_det_Ky = np.log(np.diag(Ky_chol)**2).sum()
                n_log_2pi = self.Lpix**2 * np.log(2 * np.pi)
                logL = -0.5 * (np.dot(u, u) + log_det_Ky + n_log_2pi)
                # Make sure we weight things appropriately; we want these
                # hyperpriors to be uniform in gamma and S not log-spaced,
                # even if the sampling is log spaced
                # d(log x) = dx/x; dx = x * d(log x)
                dV = S
                evalpars.append([gamma, S, logL, dV])

        evalpars = np.array(evalpars)
        ngrid = len(evalpars)
        t3 = time.time()
        vmsg("time to calculate logL grid:         {:.3f} sec".format(t3-t2))
        sys.stdout.flush()

        # The next part (prediction) is seriously expensive, so we want to
        # minimize the number of points we have to evaluate.  So far our
        # *real* marginal likelihood is unimodal in the hyperparameters,
        # so we can try approximating it by a Gaussian.  This is technically
        # a variational inference step, that might point the way to a
        # Bayesian optimization above.  For now we're just going to use it
        # to tell us which points to evaluate in the evaluation loop.
        def kldist(pars):
            # Parametrize by mu (mean) and L (covariance Cholesky factor).
            # logL(x) = -0.5*((mu-pars).T*C.inv*(mu-pars) + log(det C))
            mu_x, mu_y, L_xx, L_yy, L_xy, const = pars
            mu = np.array([mu_x, mu_y])
            Lcov = np.array([[L_xx, 0], [L_xy, L_yy]])
            x, logL_full = evalpars[:,0:2], evalpars[:,2] - const
            logdetC = np.sum(np.log([L_xx**2, L_yy**2]))
            u = linalg.solve_triangular(Lcov, (x-mu).T, lower=True).T
            logL_var = -0.5*(np.array([np.dot(ui.T, ui) for ui in u]) + logdetC)
            kl_var = np.mean(np.exp(logL_full)*(logL_full-logL_var))
            return kl_var
        if kl_approx:
            # Run the optimization
            imax = np.argmax(evalpars[:,2])
            p0 = np.array([evalpars[imax,0], evalpars[imax,1],
                           dG, dlogS, 0, evalpars[imax,2]])
            result = optimize.minimize(kldist, p0, tol=1e-2)
            # Evaluate the variational model to get evaluation points!
            mu_x, mu_y, L_xx, L_yy, L_xy, const = result.x
            Lcov = np.array([[L_xx, 0], [L_xy, L_yy]])
            n, kappa = 2, 3.0 - 2
            evalpars = np.array(
                    [[mu_x,        mu_y,        const, kappa/(n+kappa)],
                     [mu_x + L_xx, mu_y + L_xy, const,   0.5/(n+kappa)],
                     [mu_x - L_xx, mu_y - L_xy, const,   0.5/(n+kappa)],
                     [mu_x       , mu_y + L_yy, const,   0.5/(n+kappa)],
                     [mu_x       , mu_y - L_yy, const,   0.5/(n+kappa)]])
            t4 = time.time()
            vmsg("time to run variational model:       {:.3f} sec".format(t4-t3))
            sys.stdout.flush()
        else:
            # Renormalize weights for grid points we've already calculated.
            wint = evalpars[:, -1] * np.exp((evalpars[:,-2] - evalpars[:,-2].max())/1000.)
            weights = evalpars[:, -1] = wint**2 / np.sum(wint**2)
            idx = (weights > 1e-2 * weights.max())
            evalpars, weights = evalpars[idx], weights[idx]
            t4 = time.time()
            vmsg("time to renormalize weights:         {:.3f} sec".format(t4-t3))
            sys.stdout.flush()

        self.scene = np.zeros((Lpix**2,))
        self.scene_cov = np.zeros((Lpix**2, Lpix**2))

        # Recalculate the likelihood at the above points and run!
        for p in evalpars:
            gamma, S, logL, w = p
            if kl_approx:
                # No cache for these points, we need to recalculate
                K = gpKfunc(self.D2, gamma)
                AK = np.dot(A, K)
                AKA = np.dot(AK, A.T)
            else:
                # Use the cached points on the pre-calculated grid
                i = np.argmin(np.abs(self._gamvals - gamma))
                K = self._K_gv[i]
                AK = self._AK_gv[i]
                AKA = self._AKA_gv[i]
            # Do the rest of the solution
            Ky = S**2 * AKA + np.diag(yerr**2)
            Ky_chol = linalg.cholesky(Ky, lower=True)
            u = linalg.solve_triangular(Ky_chol, y, lower=True)
            V = linalg.solve_triangular(
                    Ky_chol, S**2 * AK, lower=True)
            self.scene += w*(np.dot(V.T, u)*self._fstd + self._fmean)
            self.scene_cov += w**2 * (S**2 * K - np.dot(V.T, V))*self._fstd**2


        t5 = time.time()
        vmsg("time to form final scene solution:   {:.3f} sec".format(t5-t4))
        vmsg("-----------------------------------------------")
        vmsg("total time to run:                   {:.3f} sec".format(t5-t0))
        vmsg("number of grid points contributing:  {:5d}/{}".format(
            len(evalpars), ngrid))
        vmsg("grid points contributing:")
        for evp in evalpars:
            vmsg("{:9.3f} {:6.3f} {:8.1f} {:.4f}".format(*evp))
       # sys.stdout.flush()

        vmsg("sum of weights {:.4f}".format(evalpars[:,-1].sum()))
        sys.stdout.flush()

        # Pack up everything
        self.scene = self.scene.reshape(self.response.Lpix, self.response.Lpix)
        self.pack_scene_var(np.diagonal(self.scene_cov))
        self.pack_scene_cov(self.scene_cov)

        # We have to return something, so return the MAP
        return np.max(evalpars[:,-2])

    def logL_margsimple(self, hp, verbose=False):
        """
        Version of logL that marginalizes stuff over with uniform space of gamma
        WARNING:  this code does a numeric marginalization over possible
        values of gamma, and so it predicts the marginalized posterior
        in addition to everything else it does.  It leaves the internal
        variables self.scene and self.scene_cov in a state that is NOT
        consistent with using self.predict(), so calling self.predict()
        after calling this will give you UNPREDICTABLE, WRONG results.
        Instead, call this and then just access self.scene and
        self.scene_cov directly.  
        :param hp: (hyper-)parameter 1-D np.array of floats
        :param verbose: print status messages (esp. time taken)?
        """
        vmsg = VerboseMessager(verbose=verbose)
        t0 = time.time()

        # Generate the response matrix from the hyperparameters.
        y, yerr = self.fibflux, self.fibfluxerr
        psf_pars, gdum = hp[:-1], hp[-1]
        if self.calcresponse:
            A = self._A = self.response(*psf_pars)
        else:
            A = self._A
        Lpix = self.response.Lpix
        pixscale = self.response.pixscale
        t1 = time.time()
        vmsg("time to set up response matrix:      {:.3f} sec".format(t1-t0))
        sys.stdout.flush()

        # Define the hyperparameter grid
        self._gamvals = np.linspace(gdum * 0.2, 2.* gdum, 10)

        # Look up which of the many available kernels we're using
        method_hash = { 'squared_exp': self._gpkernel,
                        'rational':    self._gpkernel_rational,
                        'sparse':      self._gpkernel_sparse,
                        'wide':        self._gpkernel_wide,
                        'moffat':      self._gpkernel_moffat,
                        'matern32':    self._gpkernel_matern32,
                        'matern52':    self._gpkernel_matern52,
                        'exp':         self._gpkernel_exp,
                        'mix':         self._gpkernel_mix, }
        if self._gpmethod in method_hash:
            gpKfunc = method_hash[self._gpmethod]
        else:
            print("Kernel method not supported, now taking default squared exponential!")
            gpKfunc = self._gpkernel

        # Cache the grid of GP kernel matrices and their products with A.
        # We only need to do this if we've updated the response matrix.
        if self.calcresponse:
            self._K_gv, self._AK_gv, self._AKA_gv = [ ], [ ], [ ]
            for gamma in self._gamvals:
                self._K_gv.append(gpKfunc(self.D2, gamma))
                self._AK_gv.append(np.dot(A, self._K_gv[-1]))
                self._AKA_gv.append(np.dot(self._AK_gv[-1], A.T))
            self._K_gv = np.array(self._K_gv)
            self._AK_gv = np.array(self._AK_gv)
            self._AKA_gv = np.array(self._AKA_gv)
        t2 = time.time()
        vmsg("time to set up GP kernel cache:      {:.3f} sec".format(t2-t1))
        sys.stdout.flush()

        # The marginal likelihood is a multivariate Gaussian, of the form
        #     log(L) = -0.5 * (y.T*(Ky^-1)*y + log(det(Ky)) + n*log(2*pi))
        # with * above meaning matrix multiplication (np.dot) for matrices.
        # This system is usually solved by taking the Cholesky factor of Ky,
        # a triangular matrix satisfying Ky = np.dot(Kychol, Kychol.T),
        # in order to form both y.T*(Ky^-1)*y = u.T*u and the log det term.

        # create arrays
        self.scene = np.zeros((Lpix**2,))
        self.scene_cov = np.zeros((Lpix**2, Lpix**2))
        logL = 0.

        # Set uniform weighting
        w = 1./self._gamvals.shape[0]

        for i in range(len(self._gamvals)):
            # Use the cached points on the pre-calculated grid
            K = self._K_gv[i]
            AK = self._AK_gv[i]
            AKA = self._AKA_gv[i]
            # Do the rest of the solution
            self._S = np.sqrt(np.mean(np.diag(AKA)))
            y, yerr = self.fibflux * self._S, self.fibfluxerr * self._S
            Ky = AKA + np.diag(yerr**2)
            Ky_chol = linalg.cholesky(Ky, lower=True)
            u = linalg.solve_triangular(Ky_chol, y, lower=True)
            V = linalg.solve_triangular(Ky_chol, AK, lower=True)
            self.scene += w * (np.dot(V.T, u) * self._fstd/self._S  + self._fmean)
            self.scene_cov += w**2 * (K - np.dot(V.T, V)) * (self._fstd/self._S)**2
            # Calculation of Logl
            log_det_Ky = np.log(np.diag(Ky_chol)**2).sum()
            n_log_2pi = self.Lpix**2 * np.log(2 * np.pi)
            logL += w* -0.5 * (np.dot(u, u) + log_det_Ky + n_log_2pi)

        t3 = time.time()
        vmsg("time to form final scene solution:   {:.3f} sec".format(t3-t2))
        vmsg("-----------------------------------------------")
        vmsg("total time to run:                   {:.3f} sec".format(t3-t0))
        sys.stdout.flush()

        # Pack up everything
        self.scene = self.scene.reshape(self.response.Lpix, self.response.Lpix)
        self.pack_scene_var(np.diagonal(self.scene_cov))
        self.pack_scene_cov(self.scene_cov)

        # return the LogL
        return logL


    def minus_logL(self, hp):
        """
        Wrapper around logL (with a minus sign) for minimization.
        :param hp: (hyper-)parameter 1-D np.array of floats
        """
        return -self.logL(hp)

    def predict(self, gcovar = False):
        """
        Evaluates the predictive density of the data, and caches the
        resulting scene and its covariance.
        :param gcovar: set for storing covariance directly
        """
        # Since we're predicting at the same points as the training data,
        # we can use a lot of our calculations from earlier.  Only reason
        # we're doing this is to deconvolve the instrument response.
        self._V = linalg.solve_triangular(
                self._Ky_chol, np.dot(self._A, self._Kxx), lower=True)
        self.pack_scene(np.dot(self._V.T, self._u)*self._fstd + self._fmean)
        self.scene_cov = self._Kxx - np.dot(self._V.T, self._V)
        self.pack_scene_var(np.diagonal(self.scene_cov)*self._fstd**2)
        if gcovar:
            self.pack_scene_cov(self.scene_cov*self._fstd**2)
