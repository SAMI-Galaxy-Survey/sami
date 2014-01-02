# ----------------------------------------------------------------------------------------
# Written by Jon Nielsen 2012
# A program to generate a weight map for the intersection of a circle with
# a square grid.    Squares on the grid that are completely within the circle
# receive a weight of 1.    Square that intersect the circle are given a weight
# that is proportional to the area of the square inside the circle.
#
# Requires numpy and matplotlib.    Although if you don't care about plotting
# then you can do without matplotlib.
# ----------------------------------------------------------------------------------------

import sys
import math
import itertools

import numpy as np

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

    # Don't forget to keep the second last point.    It will be diffed against the
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
            # Points are on different x gridlines.    Note that they must both
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
            # Points are on different y gridlines.    Note that they must both
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
            # of this square.    This is handled by the calling function, which uses
            # 1-area as a subtractive term.
            area += math.fabs(xp[1]-cy) + math.fabs((xp[1]-dy)*(yp[0]-dx)) + \
                            math.fabs(0.5*(xp[0]-yp[0])*(xp[1]-yp[1]))

    return i, j, area

def resample_circle(xpix, ypix, xc, yc, r):
    """Resample a circle/drop onto an output grid.
    
    Parameters
    ----------
    xpix: (int) Number of pixels in the x-dimension of the output grid
    ypix: (int) Number of pixels in the y-dimension of the output grid
    xc: (float) x-position of the centre of the circle.
    yc: (float) y-position of the centre of the circle.
    r: (float) radius of the circle
    
    Output
    ------
    2D array of floats. Note that the zeroth axis is for the y-dimension and the
    first axis is for the x-dimension. i.e., out.shape -> (ypix, xpix)
    This can be VERY CONFUSING, particularly when one remembers that imshow's
    behaviour is to plot the zeroth axis as the vertical coordinate and the first
    axis as the horizontal coordinate.
    
    """
    
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
