"""
mc_adr.py: Atmospheric Differential Refraction Computation

Originally Written by Mike Childress
Modified for SAMI by Andy Green

Description:
    
    This computes atmospheric differential refraction as a function of
    wavelength. The method matches that of Filippenko (1982).
    
History:
    
    I have removed some duplicated and extra functions from the original version
    by MC so that this file now only contains the adr and nothing else. Doesn't
    really meet SAMI coding standards.

"""

import numpy

def adr_n1(lam):
    #convert angstroms to microns
    lmic = lam*1.0e-4
    term1 = 64.328
    term2 = 29498.1/(146.0 - (1.0/lmic)**2)
    term3 = 255.4/(41.0 - (1.0/lmic)**2)
    return 1.0e-6*(term1 + term2 + term3)

def adr_f1(p,t):
    term1 = p/720.883
    term2 = 1.0 + 1.0e-6*p*(1.049 - 0.0157*t)
    term3 = 1.0 + 0.003661*t
    return term1*term2/term3

def adr_g1(lam, t, f):
    lmic = lam*1.0e-4
    term1 = 0.0624 - 0.000680/(lmic**2)
    term2 = 1.0 + 0.003661*t
    return term1*f/term2

def adr_ntot(lam, p, t, f):
    real_n = 1.0e-6*(adr_n1(lam) * 1.0e6 - adr_g1(lam, t, f))
    return real_n * adr_f1(p, t) + 1.0

def adr_r(wavelength, zenith_distance, air_pres=700.0, temperature=0.0, water_pres=8.0):
    """
    Compute the absolute differential atmospheric refraction.
    
    Parameters
    ----------
        wavelength: list of wavelengths to compute, units: microns (array-like)
        zenith_distance: secant of the zenith distance, or airmass (float)
        air_pres: air pressure (at ground level) at time of 
             observation (in units of mm of Hg)
        temperature: air temperature (at ground level), units of degrees celcius
        water_pres: water partial pressure at ground level in units of mm Hg
    
    Returns
    -------
        absolute magnitude of correction in arcseconds
        
    """
    
    seczd = 1/numpy.cos(numpy.radians(zenith_distance))
    
    nlam = adr_ntot(wavelength, air_pres, temperature, water_pres)
    tanz = (seczd**2 - 1.0)**0.5
    return 206265.0 * nlam * tanz

def parallactic_angle(hour_angle, zenith_distance, latitude):
    """
    Return parallactic angle in degrees for a given observing condition.
    
    Inputs in degrees. Hour angle is positive if west of the meridian.
    
    Written by Andy Green, based on Fillipenko (1982) Equation 9.
    
    """
    
    # Define two convenience functions to simplify the following code.
    sin_d = lambda x: numpy.sin(numpy.radians(x))
    arcsin_d = lambda x: numpy.degrees(numpy.arcsin(x))

    return arcsin_d(
        sin_d(hour_angle) * 
        sin_d(numpy.pi/2 - latitude) /
        sin_d(zenith_distance) 
        )


class DARCorrector:
    """
    Tool to compute atmospheric refraction corrections via one of several methods
    """
    
    
    def __init__(self,method='none'):
        if (method == 'none'):
            self.correction = self.correction_none
        
        if (method == 'simple'):
            self.temperature = 10
            self.air_pres = 720
            self.water_pres = 10
            self.correction = self.correction_simple
            self.zenith_distance = 0

    def correction_none(self, wavelength):
        """Dummy function to return 0 (no correction) if DAR method is 'none'."""
        return 0
          
    def correction_simple(self,wavelength):
        """DAR correction using simple theoretical model."""
        return adr_r(wavelength, self.zenith_distance, self.air_pres, self.temperature, self.water_pres)
    
        
        
        
        
        