from scipy.optimize import leastsq
import scipy as sp
import numpy as np

"""
This file contains various fitting functions for general use with SAMI codes.

Currently included:

GaussFitter - Gaussian Fitter (1d)
GaussHermiteFitter - Fits a truncated Gauss-Hermite expansion (1d)
TwoDGaussFitter - Gaussian Fitter (2d, optionally w/ PA and different widths)

Would be nice:

Exponential Fitter?
Others?

Example of the class format is below. Should have a list of things that can
be accessed in all class definitions (chi-2 etc.)

To use these fitting classes, initialise them using the initial guesses of the
parameters, along with the coordinates, data and (optionally) weights. Then
call the fit function to perform the fit. The best fit parameters are then
stored in p. For example:

my_fitter = TwoDGaussFitter(initial_p, x, y, data, weights)
my_fitter.fit()
best_fit_p = my_fitter.p

If you want to integrate over each fibre, use the fibre_integrator function
*before* performing the fit. The diameter must be provided in the same units
that x and y will be in. For example:

my_fitter = TwoDGaussFitter(initial_p, x, y, data, weights)
fibre_integrator(my_fitter, 1.6)
my_fitter.fit()
best_fit_p = my_fitter.p

Calling an instance of a fitter will return the model values at the provided
coordinates. So, after either of the above examples:

my_fitter(x, y)

would return the best-fit model values at the coordinates (x, y).

"""

class FittingException(Exception):
    """Could I make this do something useful?"""
    pass

class GaussFitter:
    """ Fits a 1d Gaussian to data. Params in form list p (amplitude, mean, sigma, offset). Offset is optional."""

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
            #raise FittingException("Fit failed") # This does nothing.
            
        self.linestr=self.p[0]*self.p[2]*sp.sqrt(2*S.pi)
        #self.line_err=S.sqrt(self.linestr*self.linestr*((self.perr[0]/self.p[0])**2+(self.perr[2]/self.p[2])**2))

    def __call__(self, x):
        return self.fitfunc(self.p, x)

class GaussHermiteFitter:
    """Parameters list p contains, in order, amplitude, mean, sigma, h3, h4, bias, where the bias is optional"""

    def __init__(self, p, x, y, weights=None):
        self.p_start = p
        self.p = p
        self.x = x
        self.y = y
        if weights == None:
            self.weights = S.ones(len(self.y))
        else:
            self.weights = weights

        self.perr = 0.
        self.var_fit = 0.

        # Which function to use depends on the input parameters 
        if len(p) == 5 and p[0]>0.:
            # Fit a truncated Gauss-Hermite sequence.
            self.fitfunc = self.f1
        elif len(p) == 6 and p[0]>0.:
            # Fit a truncated Gauss-Hermite sequence with a bias.
            self.fitfunc = self.f2
        else:
            raise Exception
    
    def f1(self, p, x):
        w=(p[1]-x)/(p[2])
        H3=(p[3]*sp.sqrt(2)/sp.sqrt(6))*((2*w**3)-(3*w))
        H4=(p[4]/sp.sqrt(24))*(4*w**4-12*w**2+3)
        gauss=p[0]*sp.exp(-w**2/2)
        
        gh=gauss*(1+H3+H4)
        return gh

    def f2(self, p, x):
        w=(p[1]-x)/(p[2])
        H3=(p[3]*sp.sqrt(2)/sp.sqrt(6))*((2*w**3)-(3*w))
        H4=(p[4]/sp.sqrt(24))*(4*w**4-12*w**2+3)
        gauss=p[0]*sp.exp(-w**2/2)
        
        gh2=gauss*(1+H3+H4)+p[5]
        return gh2

    def errfunc(self, p, x, y, weights):
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
            self.perr = S.sqrt(self.cov_x.diagonal())*self.var_fit

        # Would like to return the linestrength and associated error
        gamma=self.p[0]*self.p[2]*S.sqrt(2*S.pi)
        gamma_err=S.sqrt(gamma*gamma*((self.perr[0]/self.p[0])**2+(self.perr[1]/self.p[1])**2))
        self.linestr=gamma*(1+S.sqrt(6)*self.p[4]/4)
        self.line_err=S.sqrt(gamma_err**2*(1+S.sqrt(6)*self.p[4]/4)**2+self.perr[4]**2*(S.sqrt(6)*gamma_err/4)**2)

        if not self.success in [1,2,3,4]:
            print "Fit Failed..."
            #raise FittingException("Fit failed")

    def __call__(self, x):
        return self.fitfunc(self.p, x)

class TwoDGaussFitter:
    """ Fits a 2d Gaussian with PA and ellipticity. Params in form (amplitude, mean_x, mean_y, sigma_x, sigma_y,
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
                                    +((rc_y-(x*sp.sin(rot_rad)+y*sp.cos(rot_rad)))/p[4])**2)/2)

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
            print "Fit Failed..." 
            #raise ExpFittingException("Fit failed")

    def __call__(self, x, y):
        return self.fitfunc(self.p, x, y)
    
def fibre_integrator(fitter, diameter, pixel=False):
    """Edits a fitter's fitfunc so that it integrates over each SAMI fibre."""

    # Save the diameter; not used here but could be useful later
    fitter.diameter = diameter

    # Define the subsampling points to use
    n_pix = 51       # Number of sampling points across the fibre
    # First make a 1d array of subsample points
    delta_x = np.linspace(-0.5 * (diameter * (1 - 1.0/n_pix)),
                          0.5 * (diameter * (1 - 1.0/n_pix)),
                          num=n_pix)
    delta_y = delta_x
    # Then turn that into a 2d grid of (delta_x, delta_y) centred on (0, 0)
    delta_x = np.ravel(np.outer(delta_x, np.ones(n_pix)))
    delta_y = np.ravel(np.outer(np.ones(n_pix), delta_y))
    if pixel:
        # Square pixels; keep everything
        n_keep = n_pix**2
    else:
        # Round fibres; only keep the points within one radius
        keep = np.where(delta_x**2 + delta_y**2 < (0.5 * diameter)**2)[0]
        n_keep = np.size(keep)
        delta_x = delta_x[keep]
        delta_y = delta_y[keep]

    old_fitfunc = fitter.fitfunc

    def integrated_fitfunc(p, x, y):
        # The fitter's fitfunc will be replaced by this one
        n_fib = np.size(x)
        x_sub = (np.outer(delta_x, np.ones(n_fib)) +
                 np.outer(np.ones(n_keep), x))
        y_sub = (np.outer(delta_y, np.ones(n_fib)) +
                 np.outer(np.ones(n_keep), y))
        return np.mean(old_fitfunc(p, x_sub, y_sub), 0)

    # Replace the fitter's fitfunc
    fitter.fitfunc = integrated_fitfunc

    return

