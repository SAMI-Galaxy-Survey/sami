import numpy as np
from scipy.ndimage.filters import median_filter

import os
import subprocess
import gzip as gz

from collections import namedtuple

# Attempt to import bottleneck to improve speed, but fall back to old routines
# if bottleneck isn't present
try:
    from bottleneck import nanmedian
except:
    from scipy.stats import nanmedian
    print("Not Using bottleneck: Speed will be improved if you install bottleneck")

from sami import update_csv

# import constants defined in the config file.
from sami.config import *

"""
This file is the utilities script for SAMI data. See below for description of functions.

IFU : a class that returns data and ancillary info from an RSS file for a single IFU.
Bresenham circle : A function that defines a Bresenham circle in a grid of pixels.
Bresenham ellipse : A function that defines a Bresenham ellipse in a grid of pixels.
centre of mass : a 
circle resampling (by Jon Nielsen)
other stuff?

"""

def offset_hexa(csvfile, guide=None, obj=None, linear=False,
                ignore_allocations=False):
    """Print the offsets to move a star from the centre, to a guide position or hexabundle.
    
    Guide and obj numbers must match those on the plate, *not* the probes.
    If the guide number is "skip" (or no guiders are found), that step is skipped.
    If no guide or obj number is given, it uses the closest to the centre.
    If ignore_allocations is True, any hole will be used, regardless of
    whether there is a probe there."""

    print '-' * 70

    csv = update_csv.CSV(csvfile)
    guide_probe = np.array(csv.get_values('Probe', 'guide'))
    guide_x = np.array(csv.get_values('Probe X', 'guide'))
    guide_y = np.array(csv.get_values('Probe Y', 'guide'))
    object_probe = np.array(csv.get_values('Probe', 'object'))
    object_x = np.array(csv.get_values('Probe X', 'object'))
    object_y = np.array(csv.get_values('Probe Y', 'object'))

    if ignore_allocations:
        valid_guides = np.arange(guide_probe.size)
        invalid_guides = np.array([])
    else:
        valid_guides = np.where(guide_probe != '')[0]
        invalid_guides = np.where(guide_probe == '')[0]
    n_valid_guides = valid_guides.size
    n_invalid_guides = invalid_guides.size
        
    if guide is None:
        if n_valid_guides == 0:
            # Asked to find a guide but there aren't any to find!
            print 'No guide probes found! Skipping that step'
            guide = 'skip'
        else:
            # Find the closest valid guide to the centre
            dist2 = guide_x**2 + guide_y**2
            if n_invalid_guides > 0:
                dist2[invalid_guides] = np.inf
            # 1-indexed for now, this will be taken off in a few lines
            guide = dist2.argmin() + 1
            
    if guide == 'skip':
        guide_name = 'the central hole'
        guide_x = 0.0
        guide_y = 0.0
        guide_offset_x, guide_offset_y = plate2sky(
            guide_x, guide_y, linear=linear)
    else:
        # Make the 1-indexed guide number 0-indexed
        guide = guide - 1
        guide_name = 'G' + str(guide+1) + ' on plate'
        try:
            guide_name = ('guider ' + str(int(guide_probe[guide])) +
                          ' (' + guide_name + ')')
        except ValueError:
            # No guider was assigned to this hole
            guide_name = guide_name + ' (no guide probe assigned!)'
        guide_x = guide_x[guide]
        guide_y = guide_y[guide]

        guide_offset_x, guide_offset_y = plate2sky(
            guide_x, guide_y, linear=linear)

        if guide_offset_x <= 0:
            offset_direction_x = 'E'
        else:
            offset_direction_x = 'W'
        if guide_offset_y <= 0:
            offset_direction_y = 'N'
        else:
            offset_direction_y = 'S'

        print('Move the telescope {0:,.2f} arcsec {1} and {2:,.2f} arcsec {3}'.format(
            abs(guide_offset_x), offset_direction_x, 
            abs(guide_offset_y), offset_direction_y))
        print 'The star will move from the central hole'
        print '    to', guide_name

    if ignore_allocations:
        valid_objects = np.arange(object_probe.size)
        invalid_objects = np.array([])
    else:
        valid_objects = np.where(object_probe != '')[0]
        invalid_objects = np.where(object_probe == '')[0]
        if valid_objects.size == 0:
            print 'No allocated object probes found! Using closest hole.'
            valid_objects = np.arange(object_probe.size)
            invalid_objects = np.array([])
    n_valid_objects = valid_objects.size
    n_invalid_objects = invalid_objects.size
        
    if obj is None:
        # Find the closest valid object to the guide
        dist2 = (object_x - guide_x)**2 + (object_y - guide_y)**2
        if n_invalid_objects > 0:
            dist2[invalid_objects] = np.inf
        # 1-indexed for now, this will be taken off in a few lines
        obj = dist2.argmin() + 1

    obj = obj - 1
    object_name = 'P' + str(obj+1) + ' on plate'
    try:
        object_name = ('object probe ' + str(int(object_probe[obj])) +
                      ' (' + object_name + ')')
    except ValueError:
        # No object was assigned to this hole
        object_name = object_name + ' (no object probe assigned!)'
    object_x = object_x[obj]
    object_y = object_y[obj]

    object_offset_x, object_offset_y = plate2sky(
        object_x, object_y, linear=linear)
    offset_x = object_offset_x - guide_offset_x
    offset_y = object_offset_y - guide_offset_y
    
    if offset_x <= 0:
        offset_direction_x = 'E'
    else:
        offset_direction_x = 'W'
    if offset_y <= 0:
        offset_direction_y = 'N'
    else:
        offset_direction_y = 'S'

    print('Move the telescope {0:,.2f} arcsec {1} and {2:,.2f} arcsec {3}'.format(
        abs(offset_x), offset_direction_x, 
        abs(offset_y), offset_direction_y))
    print 'The star will move from', guide_name
    print '    to', object_name
    print '-' * 70

    return
    
def plate2sky(x, y, linear=False):
    """Convert position on plate to position on sky, relative to plate centre.

    x and y are input as positions on the plate in microns, with (0, 0) at
    the centre. Sign conventions are defined as in the CSV allocation files.
    Return a named tuple (xi, eta) with the angular coordinates in arcseconds,
    relative to plate centre with the same sign convention. If linear is set
    to True then a simple linear scaling is used, otherwise pincushion
    distortion model is applied."""

    # Should implement something to cope with (0, 0)

    # Define the return named tuple type
    AngularCoords = namedtuple('AngularCoords', ['xi', 'eta'])

    # Make sure we're dealing with floats
    x = np.array(x, dtype='d')
    y = np.array(y, dtype='d')

    if np.size(x) == 1 and np.size(y) == 1 and x == 0.0 and y == 0.0:
        # Plate centre, return zeros before you get an error
        return AngularCoords(0.0, 0.0)

    if linear:
        # Just do a really simple transformation
        plate_scale = 15.2 / 1000.0   # arcseconds per micron
        coords = AngularCoords(x * plate_scale, y * plate_scale)
    else:
        # Include pincushion distortion, found by inverting:
        #    x = xi * f * P(xi, eta)
        #    y = eta * f * P(xi, eta)
        # where:
        #    P(xi, eta) = 1 + p * (xi**2 + eta**2)
        #    p = 191.0
        #    f = 13.515e6 microns, the telescope focal length
        p = 191.0
        f = 13.515e6
        a = p * (x**2 + y**2) * f
        twentyseven_a_squared_d = 27.0 * a**2 * (-x**3)
        root = np.sqrt(twentyseven_a_squared_d**2 +
                       4 * (3 * a * (x**2 * f))**3)
        xi = - (1.0/(3.0*a)) * ((0.5*(twentyseven_a_squared_d +
                                      root))**(1.0/3.0) -
                                (-0.5*(twentyseven_a_squared_d -
                                       root))**(1.0/3.0))
        # Convert to arcseconds
        xi *= (180.0 / np.pi) * 3600.0
        eta = y * xi / x
        if np.size(x) > 1 and np.size(y) > 1:
            # Check for (0,0) values in input arrays
            zeros = ((x == 0.0) & (y == 0.0))
            xi[zeros] = 0.0
            eta[zeros] = 0.0
        coords = AngularCoords(xi, eta)

    return coords
    

def comxyz(x,y,z):
    """Centre of mass given x, y and z vectors (all same size). x,y give position which has value z."""

    Mx=0
    My=0
    mass=0

    for i in range(len(x)):
        Mx=Mx+x[i]*z[i]
        My=My+y[i]*z[i]
        mass=mass+z[i]

    com=(Mx/mass, My/mass)
    return com

def smooth(x ,window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]

    if window == 'flat': #moving average
        w=ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]


# This is all the stuff for Andy's dome code. Hacked by Lisa to follow the same conventions as the rest of the code. 
# Want to put this stuff somewhere else, e.g. the config.py
degree = np.pi / 180.0 # this converts to and from radians *degree means answer in radians...not a fan, could use astropy?

# This next function allows us to return angles between -180 and 180 easily.
def bettermod(num, divisor, start):
    """Return num mod divisor, with offset."""
    res = np.mod(num,divisor)
    if (res > start + divisor):
        res -=divisor
    return res

def coord_rotate(x, y, z):
    """Three dimensional coordinate rotation."""

    xt  =  np.arcsin ( np.sin(x) * np.sin(y) +
                  np.cos(x) * np.cos(y) * np.cos(z) )
    yt  =  np.arccos ( ( np.sin(x) - np.sin(y) * np.sin(xt) ) /
                  ( np.cos(y) * np.cos(xt) ) )

    if  np.sin(z) > 0.0:
        yt  =  2.0*np.pi - yt
        
    return (xt, yt)

def altaz_from_hadec(ha, dec):
    """Compute altitude and azimuth from hour angle and declination at AAT."""

    alt, az = coord_rotate(dec * degree, latitude_radians, ha * degree) 

    return (alt / degree, az / degree)

def hadec_from_altaz(alt, az):
    """Compute hour angle and declination from altitude and azimuth at AAT."""

    ha, dec = coord_rotate(alt * degree, latitude_radians, az * degree)

    return (ha / degree, dec / degree)

def domewindscreenflat_pos(ha_h, ha_m, ha_s, dec_d, dec_m, dec_s):
    """Compute dome coordinates for flat patch in front of AAT for given HA and DEC."""

    DomeCoords=namedtuple('DomeCoords', ['azimuth', 'zd'])

    # Convert sexagesimal to degrees
    ha, dec=decimal_to_degree(ha_h, ha_m, ha_s, dec_d, dec_m, dec_s)

    print
    print "---------------------------------------------------------------------------"
    print "INPUT"
    print "Hour Angle:", ha
    print "Declination:", dec
    print "---------------------------------------------------------------------------"
    
    # Convert to radians
    ha = ha * degree
    dec = dec * degree

    xt = np.cos(ha) * np.cos(dec)
    yt = np.sin(ha) * np.cos(dec)
    zt = np.sin(dec)

    # Rotate to Az-Alt
    xta = -xt * np.sin(latitude_radians) + zt * np.cos(latitude_radians)
    zta = xt * np.cos(latitude_radians) + zt * np.sin(latitude_radians)
    
    # Position of intersection of optical axis with declination axis
    w = polar_declination_dist * (1.0 - np.cos(ha))
    dx = w * np.sin(latitude_radians)
    dy = polar_declination_dist * np.sin(ha)
    dz = -(w * np.cos(latitude_radians) + declination_dome_dist)

    # Compute coefficients of quadratic in r
    b = 2.0 * ( xta * dx + yt * dy + zta * dz )
    c = dx * dx + dy * dy + dz * dz - 1.0

    # Positive solution is in front of the telescope
    r = (-b + np.sqrt(b*b - 4.0 * c )) / 2.0

    # Windscreen x, y, z of optical axis intersection
    xw = r * xta + dx
    yw = r * yt + dy
    zw = r * zta + dz

    #print( (xw, yw, zw) )
    
    # Convert to azimuth and zenith distance
    a = np.arctan2( -yw, xw)
    z = np.arctan2( np.sqrt( xw*xw + yw*yw), zw)

    # Convert back to degrees
    a = bettermod(a * 180.0 / np.pi, 360, 0)
    z = z * 180.0 / np.pi

    # Offset for the windscreen
    #   (tested by AWG on the real AAT, 7 March 2013, but may need tweaking)
    z = z + 21

    output=DomeCoords(a,z) # output is a named tuple

    print
    print "---------------------------------------------------------------------------"
    print "OUTPUT"
    print output
    print "---------------------------------------------------------------------------"

def decimal_to_degree(ha_h, ha_m, ha_s, dec_d, dec_m, dec_s):

    # Simple conversion
    ha_deg=np.abs(ha_h)*15.0+np.abs(ha_m)*15.0/60.0+np.abs(ha_s)*15.0/3600.0
    dec_deg=np.abs(dec_d)+np.abs(dec_m)/60.0+np.abs(dec_s)/3600.0
    
    if ha_h<0.0 or ha_m<0.0 or ha_s<0.0:
        ha_deg=-1.0*ha_deg

    if dec_d<0.0 or dec_m<0.0 or dec_s<0.0:
        dec_deg=-1.0*dec_deg

    return ha_deg, dec_deg

# ----------------------------------------------------------------------------------------
# This function returns the probe numbers and objects observed in them.
def get_probes_objects(infile, ifus='all'):

    if ifus=='all':
        # List of probe numbers.
        ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]

    else:
        ifus=ifus
 
    print "Probe   Object"
    print "-----------------------------------"
    for ifu in ifus:

        ifu_data=IFU(infile, ifu, flag_name=False)
        print ifu,"\t", ifu_data.name

def hg_changeset(path=__file__):
    """Return the changeset ID for the current version of the code."""
    try:
        changeset = subprocess.check_output(['hg', '-q', 'id'],
                                            cwd=os.path.dirname(path))
        changeset = changeset[:-1]
    except (subprocess.CalledProcessError, OSError):
        changeset = ''
    return changeset

def mad(a, c=0.6745, axis=None):
    """
    Compute the median absolute deviation along the specified axis.

    median(abs(a - median(a))) / c

    Returns the median absolute deviation of the array elements.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : int, optional
        Axis along which the medians are computed. The default (axis=None)
        is to compute the median along a flattened version of the array.
    c : float, optional
        The scaling factor applied to the raw median aboslute deviation.
        The default is to scale to match the standard deviation.

    Returns
    -------
    mad : ndarray
        A new array holding the result. 

    """
    if (axis is None):
        _shape = a.shape
        a.shape = np.product(a.shape,axis=0)
        m = nanmedian(np.fabs(a - nanmedian(a))) / c
        a.shape = _shape
    else:
        m = np.apply_along_axis(
            lambda x: nanmedian(np.fabs(x - nanmedian(x))) / c, 
            axis, a)

    return m


def find_fibre_table(hdulist):
    """Returns the extension number for FIBRES_IFU or MORE.FIBRES_IFU,
    whichever is found. Raises KeyError if neither is found."""

    extno = None
    try:
        extno = hdulist.index_of('FIBRES_IFU')
    except KeyError:
        pass
    if extno is None:
        try:
            extno = hdulist.index_of('MORE.FIBRES_IFU')
        except KeyError:
            raise KeyError("Extensions 'FIBRES_IFU' and "
                           "'MORE.FIBRES_IFU' both not found")
    return extno


def gzip(filename, leave_original=False):
    """gzip a file, optionally leaving the original version in place."""
    with open(filename, 'rb') as f_in:
        with gz.open(filename + '.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    if not leave_original:
        os.remove(filename)
    return

def find_nearest(arr, val):
    
    # Finds the index of the array element in arr nearest to val
    idx=(np.abs(arr-val)).argmin()
    return idx

def saturated_partial_pressure_water(air_pressure, air_temperature):
    """Return saturated partial pressure of water given the air pressure and temperature.
    
    Units are "mm of Hg" for pressures, and "degree C" for temperatures.
    
    Formula is from Wikipedia:
    http://en.wikipedia.org/wiki/Relative_humidity
    
    """

    # Convert pressure units to hPa/millibar 
    air_pressure *= 1/(millibar_to_mmHg)
    
    term1 = 1.0007 + 3.46e-6 * air_pressure
    term2 = 6.1121 * np.exp(17.502 * air_temperature / (240.97 + air_temperature))

    return term1 * term2 * millibar_to_mmHg


def replace_xsym_link(path):
    """Replace an XSym type link with a proper POSIX symlink."""
    # This could be sped up if only the first few lines were read in, but I
    # (JTA) can't be bothered.
    with open(path) as f:
        contents = f.readlines()
    if len(contents) != 5 or contents[0] != 'XSym\n':
        # This wasn't an XSym link
        raise ValueError('Not an XSym file: ' + path)
    source = contents[-2][:-1]
    os.remove(path)
    os.symlink(source, path)
    return

def replace_all_xsym_link(directory='.'):
    """Replace all XSym links in directory and its subdirectories."""
    for dirname, subdirname_list, filename_list in os.walk(directory):
        for filename in filename_list:
            try:
                replace_xsym_link(os.path.join(dirname, filename))
            except (ValueError, IOError):
                # This generally just means it wasn't an XSym link in the 
                # first place
                pass
    return

def clip_spectrum(flux, noise, wavelength, limit_noise=0.35, limit_flux=10.0,
                  limit_noise_abs=100.0):
    """Return a "good" array, clipping mostly based on discrepant noise."""
    filter_width_noise = 21
    filter_width_flux = 21
    filtered_noise = median_filter(noise, filter_width_noise)
    # Only clipping positive deviations - negative deviations are mostly due
    # to absorption lines so should be left in
    # noise_ratio = (noise - filtered_noise) / filtered_noise
    # Clipping both negative and positive values, even though this means
    # clipping out several absorption lines
    noise_ratio = np.abs((noise - filtered_noise) / filtered_noise)
    filtered_flux = median_filter(flux, filter_width_flux)
    flux_ratio = np.abs((flux - filtered_flux) / filtered_noise)
    # This is mostly to get rid of very bad pixels at edges of good regions
    # The presence of NaNs is screwing up the median filter in theses places
    median_noise = np.median(noise)
    good = (np.isfinite(flux) &
            np.isfinite(noise) &
            (noise_ratio < limit_noise) &
            (flux_ratio < limit_flux) &
            (noise < (limit_noise_abs * median_noise)))
    return good

