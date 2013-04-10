from scipy.optimize import leastsq
import scipy
from scipy.optimize.zeros import ridder
import scipy as sp
from scipy import arange
from scipy.ndimage import gaussian_filter1d, shift

from scipy import optimize

import numpy as np
import pylab as py
# Revise which of these are needed for the functions we keep.

"""
This file contains various fitting functions for general use with SAMI codes.
What we need:

Gaussian Fitter (1d)
Gaussian Fitter (2d, w/ PA and different widths)
Gaussian Fitter (2d, constrained to be circular?)
Binned (integrated) Gaussian fitter (Have three from James, Julia, and Ned. The best one of these will be included.)
Gauss-Hermite Fitter?

Example of the class format is below. Should have a list of things that can be accessed in all class definitions (chi-2
etc.)



"""

class FittingException(Exception):
    """Could I make this do something useful?"""
    pass

class GaussFitter:
    """ To fit a 1d Gaussian to data. Params in form list p (amplitude, mean, sigma, offset). Offset is optional."""

    def __init__(self, p, x, y, weights=None):
        self.p_start = p
        self.p = p
        self.x = x
        self.y = y
        if weights == None:
            self.weights = sp.ones(len(self.y))
        else:
            self.weights = weights

        self.perr = 0.
        self.var_fit = 0.

        if len(p) == 4 and p[0]>0.:
            self.fitfunc = self.f1
        elif len(p) == 3:
            # no base
            self.fitfunc = self.f2
        elif len(p) == 4 and p[0] < 0.:
            self.p[0] = abs(self.p[0])
            self.fitfunc = self.f3
    
    def f1(self, p, x): 
        return p[0]*sp.exp(-(p[1]-x)**2/(2*p[2]**2)) + p[3]
    
    def f2(self, p, x): 
        return p[0]*sp.exp(-(p[1]-x)**2/(2*p[2]**2)) + 0.
    
    def f3(self, p, x): 
        return -p[0]*sp.exp(-(p[1]-x)**2/(2*p[2]**2)) + p[3]

    def errfunc(self, p, x, y, weights):
        # if width < 0 return input
        if p[2] < 0. or p[0] < 0.:
            # if we get a negative sigma value then penalise massively because
            # that is silly.
            return 1e99
        else:
            return weights*(self.fitfunc(p, x) - y)

    def fit(self):

        self.p, self.cov_x, self.infodict, self.mesg, self.success = \
        leastsq(self.errfunc, self.p, \
                args=(self.x, self.y, self.weights), full_output=1)

        var_fit = (self.errfunc(self.p, self.x, \
            self.y, self.weights)**2).sum()/(len(self.y)-len(self.p))

        self.var_fit = var_fit

        if self.cov_x != None:
            self.perr = sp.sqrt(self.cov_x.diagonal())*self.var_fit

        if not self.success in [1,2,3,4]:
            print "Fit Failed" 
            #raise ExpFittingException("Fit failed") # This does nothing.
            
        self.linestr=self.p[0]*self.p[2]*sp.sqrt(2*S.pi)
        #self.line_err=S.sqrt(self.linestr*self.linestr*((self.perr[0]/self.p[0])**2+(self.perr[2]/self.p[2])**2))

    def __call__(self, x):
        return self.fitfunc(self.p, x)

class TwoDGaussFitter:
    """ To fit a 2d Gaussian with PA and ellipticity. Params in form (amplitude, mean_x, mean_y, sigma_x, sigma_y,
    rotation, offset). Offset is optional. To fit a circular Gaussian use (amplitude, mean_x, mean_y, sigma, offset),
    again offset is optional."""

    def __init__(self, p, x, y, z, weights=None):
        self.p_start = p
        self.p = p
        self.x = x
        self.y = y
        self.z = z
        
        if weights == None:
            self.weights = sp.ones(len(self.z))
        else:
            self.weights = weights

        self.perr = 0.
        self.var_fit = 0.

        if len(p) == 7:
            # 2d elliptical Gaussian with offset.
            self.p[0] = abs(self.p[0]) # amplitude should be positive.
            self.fitfunc = self.f1
            
        elif len(p) == 6:
            # 2d elliptical Gaussian witout offset.
            self.p[0] = abs(self.p[0]) # amplitude should be positive.
            self.fitfunc = self.f2
            
        elif len(p) == 5:
            # 2d circular Gaussian with offset.
            self.p[0] = abs(self.p[0]) # amplitude should be positive.
            self.fitfunc = self.f3
            
        elif len(p) == 4:
            self.p[0] = abs(self.p[0])
            self.fitfunc = self.f4

        else:
            raise Exception

    def f1(self, p, x, y):
        # f1 is an elliptical Gaussian with PA and a bias level.

        rot_rad=p[5]*sp.pi/180 # convert rotation into radians.

        rc_x=p[1]*sp.cos(rot_rad)-p[2]*sp.sin(rot_rad)
        rc_y=p[1]*sp.sin(rot_rad)+p[2]*sp.cos(rot_rad)
    
        return p[0]*sp.exp(-(((rc_x-(x*sp.cos(rot_rad)-y*sp.sin(rot_rad)))/p[3])**2\
                                    +((rc_y-(x*sp.sin(rot_rad)+y*sp.cos(rot_rad)))/p[4])**2)/2)+p[6]

    def f2(self, p, x, y):
        # f2 is an elliptical Gaussian with PA and no bias level.

        rot_rad=p[5]*sp.pi/180 # convert rotation into radians.

        rc_x=p[1]*sp.cos(rot_rad)-p[2]*sp.sin(rot_rad)
        rc_y=p[1]*sp.sin(rot_rad)+p[2]*sp.cos(rot_rad)
    
        return p[0]*sp.exp(-(((rc_x-(x*sp.cos(rot_rad)-y*sp.sin(rot_rad)))/p[3])**2\
                                    +((rc_y-(x*sp.sin(rot_rad)+y*sp.cos(rot_rad)))/p[4])**2)/2)+p[6]

    def f3(self, p, x, y):
        # f3 is a circular Gaussian, p in form (amplitude, mean_x, mean_y, sigma, offset).
        return p[0]*sp.exp(-(((p[1]-x)/p[3])**2+((p[2]-y)/p[3])**2)/2)+p[4]

    def f4(self, p, x, y):
        # f4 is a circular Gaussian as f3 but without an offset
        return p[0]*sp.exp(-(((p[1]-x)/p[3])**2+((p[2]-y)/p[3])**2)/2)

    def errfunc(self, p, x, y, z, weights):
        # if width < 0 return input
        if p[3] < 0. or p[4] < 0. or p[0] < 0.:
            # if we get negative sigma values then penalise massively because
            # that is silly.
            return 1e99
        else:
            return weights*(self.fitfunc(p, x, y) - z)

    def fit(self):

        #print np.shape(self.x), np.shape(self.y), np.shape(self.z)

        self.p, self.cov_x, self.infodict, self.mesg, self.success = \
        leastsq(self.errfunc, self.p, \
                args=(self.x, self.y, self.z, self.weights), full_output=1)

        var_fit = (self.errfunc(self.p, self.x, \
            self.y, self.z, self.weights)**2).sum()/(len(self.z)-len(self.p))

        self.var_fit = var_fit

        if self.cov_x != None:
            self.perr = sp.sqrt(self.cov_x.diagonal())*self.var_fit

        if not self.success in [1,2,3,4]:
            #print "Fit Failed" 
            raise ExpFittingException("Fit failed")

    def __call__(self, x, y):
        return self.fitfunc(self.p, x, y)

# Make this stuff into a class as well.  
def rotgaussianfibre(height, c_x, c_y, w_x, w_y, rot, b, diameter):
    # a 2d gaussian integrated over a fibre of diameter as given
    def retfunc(x, y):
        # this function will be returned, it has the parameters hardwired
        # so only requires the fibre positions to be provided
        n_pix = 51       # sampling points across the fibre
        n_fib = np.size(x)
        # first make a 1d array of subsample points
        x_sub = np.linspace(-0.5 * (diameter * (1 - 1.0/n_pix)),
                            0.5 * (diameter * (1 - 1.0/n_pix)),
                            num=n_pix)
        y_sub = x_sub
        # then turn that into a 2d grid of (x_sub, y_sub) centred on (0, 0)
        x_sub = ravel(np.outer(x_sub, np.ones(n_pix)))
        y_sub = ravel(np.outer(np.ones(n_pix), y_sub))
        # only keep the points within one radius
        keep = np.where(x_sub**2 + y_sub**2 < (0.5 * diameter)**2)[0]
        n_keep = np.size(keep)
        x_sub = x_sub[keep]
        y_sub = y_sub[keep]
        # now centre this grid on each fibre position
        x_sub = np.outer(x_sub, np.ones(n_fib)) + np.outer(np.ones(n_keep), x)
        y_sub = np.outer(y_sub, np.ones(n_fib)) + np.outer(np.ones(n_keep), y)
        # evaluate the gaussian at all of these positions
        gauss = rotgaussian(height, c_x, c_y, w_x, w_y, rot, b)(x_sub, y_sub)
        # take the mean in each fibre
        gauss = np.mean(gauss, 0)
        return gauss
    return retfunc

def fitrotgaussianfibre(params, x, y, data):
    #as fitrotgaussianirr, but flux is integrated across the fibre
    #params in form height, c_x, c_y, w_x, w_y, rot, bg, diameter
    #diameter is kept fixed
    params_short = params[:-1]
    diameter = params[-1]
    def errorfunction(p):
        p_long = np.hstack((p, diameter))
        return ravel(rotgaussianfibre(*p_long)(x, y) - data)
    p, success = optimize.leastsq(errorfunction, params_short)
    print success
    return np.hstack((p, diameter))
