import pylab as py
import numpy as np
import scipy as sp
import pyfits as pf

import os
import sys
import math
import itertools
import subprocess

from collections import namedtuple

from sami import update_csv

# import constants defined in the config file.
from config import *

"""
This file is the utilities script for SAMI data. See below for description of functions.

IFU : a class that returns data and ancillary info from an RSS file for a single IFU.
Bresenham circle : A function that defines a Bresenham circle in a grid of pixels.
Bresenham ellipse : A function that defines a Bresenham ellipse in a grid of pixels.
centre of mass : a 
circle resampling (by Jon Nielsen)
other stuff?

"""

class IFU:

    def __init__(self, infile, pick, flag_name=True):
        """A class containing data and other information from a single file pertaining to a particular object or
        probe."""
        
        self.infile=infile

        # Open the file (should I really be doing this here?)
        hdulist=pf.open(infile)

        data_in=hdulist['PRIMARY'].data
        variance_in=hdulist['VARIANCE'].data
        primary_header=hdulist['PRIMARY'].header

        fibre_table=hdulist['FIBRES_IFU'].data

        # Some useful stuff from the header
        self.exptime=primary_header['EXPOSED']
        self.crval1=primary_header['CRVAL1']
        self.cdelt1=primary_header['CDELT1']
        self.crpix1=primary_header['CRPIX1']
        self.naxis1=primary_header['NAXIS1']

        self.meanra=primary_header['MEANRA']
        self.meandec=primary_header['MEANDEC']
        
        # Wavelength range
        x=np.arange(self.naxis1)+1
        
        L0=self.crval1-self.crpix1*self.cdelt1 #Lc-pix*dL
        
        self.lambda_range=L0+x*self.cdelt1

        # Based on the given information (probe number or object name) find the other piece of information. NOTE - this
        # will fail for unassigned probes which will have empty strings as a name.
        if flag_name==True:
            if len(pick)>0:
                self.name=pick # Flag is true so we're selecting on object name.
                msk0=fibre_table.field('NAME')==self.name # First mask on name.
                table_find=fibre_table[msk0] 

                # Find the IFU name from the find table.
                self.ifu=np.unique(table_find.field('PROBENUM'))[0]

            else:
                # Write an exception error in here?
                pass
            
        else:
            self.ifu=pick # Flag is not true so we're selecting on probe (IFU) number.
            
            msk0=fibre_table.field('PROBENUM')==self.ifu # First mask on probe number.
            table_find=fibre_table[msk0]

            # Pick out the place in the table with object names, rejecting SKY and empty strings.
            object_names_nonsky = [s for s in table_find.field('NAME') if s.startswith('SKY')==False and len(s)>0]

            #print np.shape(object_names_nonsky)

            self.name=list(set(object_names_nonsky))[0]
            
        mask=np.logical_and(fibre_table.field('PROBENUM')==self.ifu, fibre_table.field('NAME')==self.name)
        table_new=fibre_table[mask]

        #X and Y positions of fibres in absolute degrees.
        self.xpos=table_new.field('FIB_MRA') #RA add -1*
        self.ypos=table_new.field('FIB_MDEC') #Dec
 
        # Fibre number - used for tests.
        self.n=table_new.field('FIBNUM')
    
        # Fibre designation.
        self.fib_type=table_new.field('TYPE')
        
        # Probe Name
        self.hexabundle_name=table_new.field('PROBENAME')
        
        # Adding for tests only - LF 05/04/2012
        self.x_microns=-1*table_new.field('FIBPOS_X') # To put into on-sky frame
        self.y_microns=table_new.field('FIBPOS_Y')
        
        # Name of object
        name_tab=table_new.field('NAME')
        self.name=name_tab[0]
        
        # indices of the corresponding spectra (SPEC_ID counts from 1, image counts from 0)
        ind=table_new.field('SPEC_ID')-1
        
        self.data=data_in[ind,:]/self.exptime
        self.var=variance_in[ind,:]/(self.exptime*self.exptime)

        # Added for Iraklis, might need to check this.
        self.fibtab=fibre_table

        del hdulist

def offset_hexa(csvfile, guide=None, obj=None, linear=False,
                ignore_allocations=False):
    """Print the offsets to move a star from the centre,
    to a guide position, to a hexabundle.
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

        print 'Move the telescope', abs(guide_offset_x), 'arcsec', offset_direction_x, \
            'and', abs(guide_offset_y), 'arcsec', offset_direction_y
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

    print 'Move the telescope', abs(offset_x), 'arcsec', offset_direction_x, \
        'and', abs(offset_y), 'arcsec', offset_direction_y
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

    if x == 0.0 and y == 0.0:
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

# ----------------------------------------------------------------------------------------
# Written by Jon Nielsen 2012
# A program to generate a weight map for the intersection of a circle with
# a square grid.  Squares on the grid that are completely within the circle
# receive a weight of 1.  Square that intersect the circle are given a weight
# that is proportional to the area of the square inside the circle.
#
# Requires numpy and matplotlib.  Although if you don't care about plotting
# then you can do without matplotlib.
# ----------------------------------------------------------------------------------------

def find_squares_in_circle(xc, yc, r):
  # Establish bounds in y
  ymin = int(math.ceil(yc - r))
  ymax = int(math.floor(yc + r))
  yspace = np.arange(ymin, ymax)

  # Use these to calculate, for each y, the bounds in x
  # Ensure that we check that the whole square is in the circle, not
  # just its lower left point.
  y = yspace - yc
  x1 = r*r - y*y
  x1[np.abs(x1) <= sys.float_info.epsilon] = 0

  x2 = r*r - (y+1)*(y+1)
  x2[np.abs(x2) <= sys.float_info.epsilon] = 0

  x = np.sqrt(np.minimum(x1, x2))
  xmin = np.cast[int](np.ceil(-x + xc))
  xmax = np.cast[int](np.floor(x + xc))

  # Now we have, for each y, the bounds in x
  # Use these to create a list of squares that are in the circle
  arr = np.column_stack((yspace, xmin, xmax, xmax-xmin))
  if (arr.shape[0] == 0):
    return None

  # Make sure we don't have any where max<min (can happen due to the way
  # we search for the bounds)
  keep = (arr[:,3] > 0)
  arr = arr[keep]
  npoints = np.add.reduce(arr[:,3])

  # Make sure there's something to work with
  if (npoints <= 0):
    return None

  points = np.empty((npoints,2), dtype=int)
  i = 0
  for row in arr:
    points[i:i+row[3]] = np.column_stack((np.arange(row[1],row[2]),np.repeat(row[0],row[2]-row[1])))
    i += row[3]

  return points

def find_intersections(xc, yc, r):
  # First establish the limits within which the intersections will lie
  xmin = int(math.ceil(xc - r))
  xmax = int(math.floor(xc + r)) + 1
  ymin = int(math.ceil(yc - r))
  ymax = int(math.floor(yc + r)) + 1

  # Generate the grid
  xspace = np.arange(xmin, xmax)
  yspace = np.arange(ymin, ymax)

  # Calculate the intersections with each integer x
  x = xspace - xc
  y2 = r*r - x*x
  # Deal with floating point issues
  y2[np.abs(y2) <= sys.float_info.epsilon] = 0
  y = np.sqrt(y2)
  # Ignore tangents
  keep = (y > sys.float_info.epsilon)
  x = x[keep]
  newx = xspace[keep]
  y = y[keep]

  # Make sure there's something to work with
  if (y.shape[0] <= 0):
    return None

  # Get +/- solutions
  x = np.tile(x, 2)
  newx = np.tile(newx, 2)
  y = np.hstack((y, -y))
  newy = y+yc
  # Decide if any of these intersections are also on an integer y
  on_y = (np.abs(newy-np.round(newy)) <= sys.float_info.epsilon)
  newy[on_y] = np.round(newy[on_y])
  # Calculate angles (+ve please)
  theta = np.arctan2(y, x)
  theta[(theta < 0)] += 2*math.pi
  # Store the points
  points = np.column_stack([newx, newy, theta, np.ones_like(x), on_y])

  # Calculate the intersections with each integer y
  y = yspace - yc
  x2 = r*r - y*y
  # Deal with floating point issues
  x2[np.abs(x2) <= sys.float_info.epsilon] = 0
  x = np.sqrt(x2)
  # Ignore tangents
  keep = (x > sys.float_info.epsilon)
  x = x[keep]
  y = y[keep]
  newy = yspace[keep]

  # Get +/- solutions
  x = np.hstack((x, -x))
  y = np.tile(y, 2)
  newy = np.tile(newy, 2)
  # Decide if any of these intersections are also on an integer x
  newx = x+xc
  on_x = (np.abs(newx-np.round(newx)) <= sys.float_info.epsilon)
  newx[on_x] = np.round(newx[on_x])
  # Calculate angles (+ve please)
  theta = np.arctan2(y, x)
  theta[(theta < 0)] += 2*math.pi
  # Store the points
  points = np.append(points, np.column_stack([newx, newy, theta, on_x, np.ones_like(y)]), axis=0)

  # Sort by theta, and repeat the first point at the end
  args = np.argsort(points[:,2])
  points = points[np.append(args, args[0])]
  points[-1,2] += 2*math.pi;

  # Remove duplicates
  # We don't need an abs on the diff because we have already sorted into
  # ascending order.
  args = (np.diff(points[:,2]) > sys.float_info.epsilon)

  # Don't forget to keep the second last point.  It will be diffed against the
  # repeated first point, and will get a -ve result.
  args[-1] = True
  # The very last point is the repeated first point, but the diff is one shorter
  # so we fix that here
  args = np.append(args, True)
  points = points[args]

  return points

def area_contribution(p1, p2, xc, yc, r):
  i = 0
  j = 0
  area = 0.0

  # We assume that p2 theta < p1 theta
  delta_theta = p2[2] - p1[2]

  # Work out which square we are dealing with here
  mid_theta = (p1[2] + p2[2]) / 2.0
  x = r * math.cos(mid_theta) + xc
  i = int(math.floor(x))
  y = r * math.sin(mid_theta) + yc
  j = int(math.floor(y))

  # First get the circle segment area
  area = 0.5*r*r*(delta_theta - math.sin(delta_theta))

  # Next get the polygonal area
  
  if (p1[3] and p2[3]):
    # Both points are on an x gridline
    delta_x = math.fabs(p1[0] - p2[0])
    if (delta_x <= sys.float_info.epsilon):
      # Both points are on the same x gridline
      # No polygonal contribution at all.
      pass
    else:
      # Points are on different x gridlines.  Note that they must both
      # have the same upper and lower y grid bounds, or else we would have
      # a point on a y gridline between them.
      delta_y = math.fabs(p1[1] - p2[1])
      if (y < yc):
	tmpy = max(p1[1], p2[1])
	# rectangular area
	area += math.ceil(tmpy) - tmpy
      else:
	tmpy = min(p1[1], p2[1])
	# rectangular area
	area += tmpy - math.floor(tmpy)

      # triangular area
      area += 0.5*delta_y

  elif (p1[4] and p2[4]):
    # Both points are on a y gridline
    delta_y = math.fabs(p1[1] - p2[1])
    if (delta_y <= sys.float_info.epsilon):
      # Both points are on the same y gridline
      # No polygonal contribution at all.
      pass
    else:
      # Points are on different y gridlines.  Note that they must both
      # have the same upper and lower x grid bounds, or else we would have
      # a point on an x gridline between them.
      delta_x = math.fabs(p1[0] - p2[0])
      if (x < xc):
	tmpx = max(p1[0], p2[0])
	# rectangular area
	area += math.ceil(tmpx) - tmpx
      else:
	tmpx = min(p1[0], p2[0])
	# rectangular area
	area += tmpx - math.floor(tmpx)

      # triangular area
      area += 0.5*delta_x

  else:
    # One is on x, the other on y
    # Call the point on x xp, and the point on y yp
    if (p1[3] and p2[4]):
      xp = p1
      yp = p2
    else:
      xp = p2
      yp = p1

    # Now we know which is which, construct point c, which is the
    # point on the same x gridline as xp, but also on the next y gridline
    # closer to the centre of the circle
    if (xp[1] < yc):
      cy = math.ceil(xp[1])
    else:
      cy = math.floor(xp[1])
    cx = xp[0]

    # Now also point d, which is on the same y gridline as yp,
    # but also on the next x gridline closer to the centre of the circle
    if (yp[0] < xc):
      dx = math.ceil(yp[0])
    else:
      dx = math.floor(yp[0])
    dy = yp[1]

    # Work out if c and d are different points, or the same point
    if (math.sqrt((cx-dx)**2 + (cy-dy)**2) <= sys.float_info.epsilon):
      # The same point, so it's a triangle
      area += math.fabs(0.5*(xp[1]-cy)*(yp[0]-cx))
    else:
      # Not the same point - it's a pentagon
      # Note that we ignore any intersections of the circle with other edges
      # of this square.  This is handled by the calling function, which uses
      # 1-area as a subtractive term.
      area += math.fabs(xp[1]-cy) + math.fabs((xp[1]-dy)*(yp[0]-dx)) + \
              math.fabs(0.5*(xp[0]-yp[0])*(xp[1]-yp[1]))

  return i, j, area

def resample_circle(xpix, ypix, xc, yc, r):
  # Create the output array
  out = np.zeros((ypix,xpix))

  # First find the squares that are entirely in the circle
  a = find_squares_in_circle(xc, yc, r)
  if not a is None:
    out[a[:,1],a[:,0]] = 1.0

  # Now work out the tricky bits around the circumference
  b = find_intersections(xc, yc, r)
  if b is None:
    # The whole circle fits in one square
    i = int(math.floor(xc))
    j = int(math.floor(yc))
    out[j,i] = math.pi*r*r
  else:
    # Work out way through the points, pairwise, calculating area as we go
    for (p1,p2) in pairwise(b):
      i,j,area = area_contribution(p1, p2, xc, yc, r)
      #print p1, p2
      #print "i,j,area",i,j,area,out[j,i]
      if (out[j,i] != 0.0):
	# We already had area for this square, so that means the circle
	# has intersected it again and we need to subtract off the new bit
	# from what we already calculated
	area = 1-area
	out[j,i] -= area
      else:
	# Just set the output for this square
	out[j,i] = area

  return out

# A useful function for iterating over a sequence in pairwise fashion
def pairwise(iterable):
  "s -> (s0,s1), (s1,s2), (s2, s3), ..."
  a, b = itertools.tee(iterable)
  next(b, None)
  return itertools.izip(a, b)

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

def mad(a, c=0.6745, axis=0):
    """
    Median Absolute Deviation:

    median(abs(a - median(a))) / c

    """
    
    _shape = a.shape
    a.shape = np.product(a.shape,axis=0)
    m = stats.nanmedian(np.fabs(a - stats.nanmedian(a))) / c
    a.shape = _shape

    return m

