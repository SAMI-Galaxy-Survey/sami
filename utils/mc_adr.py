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
from scipy import integrate

from ..config import latitude, millibar_to_mmHg
from other import saturated_partial_pressure_water

from astropy import __version__ as astropy_version

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
        wavelength: list of wavelengths to compute, units: angstroms (array-like)
        zenith_distance: secant of the zenith distance, or airmass (float)
        air_pres: air pressure (at ground level) at time of 
             observation (in units of mm of Hg)
        temperature: air temperature (at ground level), units of degrees celcius
        water_pres: water partial pressure at ground level in units of mm Hg
    
    Returns
    -------
        absolute magnitude of correction in arcseconds
        
    """
    
    seczd = 1.0/numpy.cos(numpy.radians(zenith_distance))
    
    nlam = adr_ntot(wavelength, air_pres, temperature, water_pres)
    tanz = (seczd**2 - 1.0)**0.5
    return 206265.0 * nlam * tanz

def parallactic_angle(hour_angle, declination, latitude):
    """
    Return parallactic angle in degrees for a given observing condition.
    
    Inputs in degrees. Hour angle is positive if west of the meridian.
    
    The parallactic angle returned is the direction to the zenith measured north
    through east.
    
    Written by Andy Green, confirmed to give the same results as Fillipenko
    (1982) Equation 9, but with correct sign/direction for all areas of the sky.
    Actual formula from:
    
    https://github.com/brandon-rhodes/pyephem/issues/24
    
    "A treatise on spherical astronomy" By Sir Robert Stawell Ball
    (p. 91, as viewed on Google Books)
    
    """

    sin_dec = numpy.sin(numpy.radians(declination))
    cos_dec = numpy.cos(numpy.radians(declination))
    sin_lat = numpy.sin(numpy.radians(latitude))
    cos_lat = numpy.cos(numpy.radians(latitude))
    sin_ha =  numpy.sin(numpy.radians(hour_angle))
    cos_ha =  numpy.cos(numpy.radians(hour_angle))

    return numpy.degrees(
        -numpy.arctan2( cos_lat * sin_ha, sin_lat * cos_dec - cos_lat * sin_dec * cos_ha)
        )
    
def zenith_distance(declination, hour_angle):
    """Return the zenith distance in degrees of an object.
    
    All inputs are in degrees.
    
    This is based on "A treatise on spherical astronomy" By Sir Robert Stawell Ball, pg 91.
    
    """
    
    # astropy version catch to be backwards compatible
    if float(astropy_version[2]) >= 3.:
        Latitude=latitude.degree
    else:
        Latitude=latitude.degrees
    
    sin_lat = numpy.sin(numpy.radians(Latitude))
    sin_dec = numpy.sin(numpy.radians(declination))
    cos_lat = numpy.cos(numpy.radians(Latitude))
    cos_dec = numpy.cos(numpy.radians(declination))
    cos_ha =numpy.cos(numpy.radians(hour_angle))
    
    return numpy.degrees(
        numpy.arccos(sin_lat * sin_dec + cos_lat * cos_dec * cos_ha))

class DARCorrector(object):
    """
    Tool to compute atmospheric refraction corrections via one of several methods
    """
    
    
    def __init__(self,method='none'):
        self.method = method
        if (method == 'none'):
            self.correction = self.correction_none
        
        if (method == 'simple'):
            self.temperature = 10
            self.air_pres = 720
            self.water_pres = 10
            self.correction = self.correction_simple
            self.zenith_distance = 0
            self.ref_wavelength = 5000.0
        
        # astropy version catch to be backwards compatible
        if float(astropy_version[2]) >= 3.:
            self.latitude=latitude.degree
        else:
            self.latitude=latitude.degrees
        
        # Private variables
        self._pa = False

    def correction_none(self, wavelength):
        """Dummy function to return 0 (no correction) if DAR method is 'none'."""
        return 0
          
    def correction_simple(self,wavelength):
        """DAR correction using simple theoretical model from wavelength in angstroms."""
        
        dar = adr_r(wavelength, self.zenith_distance, self.air_pres, self.temperature, self.water_pres)
        dar_reference = adr_r(self.ref_wavelength, self.zenith_distance, self.air_pres, self.temperature, self.water_pres)
    
        return dar - dar_reference
    
    def setup_for_ifu(self, ifu):
        """Set all instance properties for the IFU class observation provided."""


        if (self.method == 'none'):
            pass
        elif (self.method == 'simple'):
            self.temperature = ifu.fibre_table_header['ATMTEMP']
            self.air_pres = ifu.fibre_table_header['ATMPRES'] * millibar_to_mmHg
            #                     (factor converts from millibars to mm of Hg)
            self.water_pres = \
                saturated_partial_pressure_water(self.air_pres, self.temperature) * \
                ifu.fibre_table_header['ATMRHUM']

            ha_offset = ifu.ra - ifu.meanra    # The offset from the HA of the field centre

            self.zenith_distance = \
                integrate.quad(lambda ha: zenith_distance(ifu.dec, ha),
                               ifu.primary_header['HASTART'] + ha_offset,
                               ifu.primary_header['HAEND'] + ha_offset)[0] / (
                                  ifu.primary_header['HAEND'] - ifu.primary_header['HASTART'])

            self.hour_angle = \
                (ifu.primary_header['HASTART'] + ifu.primary_header['HAEND']) / 2 + ha_offset

            self.declination = ifu.dec



    def print_setup(self):
        print("Method: {}".format(self.method))
        
        if (self.method != 'none'):
            print("Air Pressure: {}, Water Pressure: {}, Temperature: {}".format(self.air_pres, self.water_pres, self.temperature))
            print("Zenith Distance: {}, Reference Wavelength: {}".format(self.zenith_distance, self.ref_wavelength))
        
    def parallactic_angle(self):
        if self._pa is False:
            self._pa = parallactic_angle(self.hour_angle, 
                                         self.declination, 
                                         self.latitude) 
        return self._pa
        
    def update_for_wavelength(self,wavelength):
        """Update the valuse of dar_r, dar_east, and dar_north for the given wavelength.
        
        dar_r, dar_east, and dar_north are stored as instance attributes. They
        give the scale of the refraction from the refraction at the reference
        wavelength, e.g., a star observed at dec_obs would appear at dec_obs + dar_north 
        if there were no atmosphere.
        
        """
        
        self.dar_r = self.correction(wavelength)

        # Parallactic angle is direction to zenith measured north through east.
        # Must move light away from the zenith to correct for DAR.
        self.dar_east  = -self.dar_r * numpy.sin(numpy.radians(self.parallactic_angle()))
        self.dar_north = -self.dar_r * numpy.cos(numpy.radians(self.parallactic_angle()))
        
    @property
    def wavelength(self):
        """The wavelength of the current DAR correction."""
        return self._wavelength
         
    @wavelength.setter
    def wavelength(self,value):
        self._wavelength = value

        # Update the dar correction values because the wavelength has changed.
        self.dar_r = self.correction(value)
        # Parallactic angle is direction to zenith measured north through east.
        # Must move light away from the zenith to correct for DAR.
        self.dar_east  = -self.dar_r * numpy.sin(numpy.radians(self.parallactic_angle()))
        self.dar_north = -self.dar_r * numpy.cos(numpy.radians(self.parallactic_angle()))
        
        
        
