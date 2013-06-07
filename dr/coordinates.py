"""Functions for modifying fibre coordinates in SAMI FITS files."""

import astropy.io.fits as pf
import numpy as np
from sami.utils.other import find_fibre_table
from scipy.optimize import leastsq

def reverse_probes(fibre_table):
    """Reverse the order of the probes in the fibre table.

    This function is to correct a fault before 6th March 2013 in which the
    probe numbers were in the wrong order. The code in fact changes the
    fibre numbers (SPEC_ID) to match the given probe numbers, and then
    sorts by SPEC_ID.
    """
    # Correct each fibre number (SPEC_ID)
    for fibre in fibre_table:
        probenum_0 = fibre['PROBENUM'] - 1
        # This is the correct mapping for the fibre numbers
        if 'SKY' in fibre['PROBENAME']:
            fibre['SPEC_ID'] = 819 - fibre['SPEC_ID']
        else:
            rel_spec_id = fibre['SPEC_ID'] - 63 * probenum_0
            fibre['SPEC_ID'] = 63 * (12 - probenum_0) + rel_spec_id
    # Sort the fibre_table by fibre number
    fibre_table.sort(order='SPEC_ID')
    return
            
def rotate_all_hexas(fibre_table):
    """Rotate all hexabundles by 180 degrees.

    See rotate_probe for further details.
    """
    for probenum in xrange(1,14):
        # Have to do things as a slice to avoid copying the data back and forth
        this_probe = np.where((fibre_table['PROBENUM'] == probenum) &
                              (fibre_table['TYPE'] == 'P'))[0]
        fibre_table_hexa = fibre_table[this_probe[0]:this_probe[-1]+1]
        rotate_hexa(fibre_table_hexa)
    return

def rotate_hexa(fibre_table_hexa):
    """Rotate hexabundle by 180 degrees.

    This function is to correct a fault before 1st April 2013 in which the
    hexabundles were given a rotation of 0 degrees, when they should have
    had 180 degrees.

    We know that FIPBOS_X/Y is on a nice square coordinate system, so these
    coordinates are rotated by 180 degrees, and then converted into all
    other coordinate systems by interpolating between the original
    FIBPOS_X/Y values.
    """
    # Define the centre of the hexabundle
    alpha, beta = define_hexa_centre(fibre_table_hexa)
    # Rotate FIBPOS_X/Y, but don't overwrite the old coordinates yet
    cen_x, cen_y = coordinate_centre(
        fibre_table_hexa, 'FIBPOS_X', 'FIBPOS_Y', alpha, beta)
    new_fibpos_x = cen_x - (fibre_table_hexa['FIBPOS_X'] - cen_x)
    new_fibpos_y = cen_y - (fibre_table_hexa['FIBPOS_Y'] - cen_y)
    # Now rotate each other coordinate pair in turn, using interpolation
    name_pair_list = [('XPOS', 'YPOS'),
                      ('FIB_MRA', 'FIB_MDEC'),
                      ('FIB_ARA', 'FIB_ADEC')]
    for x_name, y_name in name_pair_list:
        interpolate(fibre_table_hexa, x_name, y_name,
                    new_fibpos_x, new_fibpos_y)
    # Update the FIBPOS_X/Y positions
    fibre_table_hexa['FIBPOS_X'][:] = np.round(new_fibpos_x).astype(int)
    fibre_table_hexa['FIBPOS_Y'][:] = np.round(new_fibpos_y).astype(int)
    # Update the PORIENT values
    fibre_table_hexa['PORIENT'][:] = 180.0
    return

def define_hexa_centre(fibre_table_hexa):
    """Define the centre of a hexabundle relative to fibres 1-3.

    x_cen = x_0 + alpha * (x_1 - x_0) + beta * (x_2 - x_0)
    y_cen = y_0 + alpha * (y_1 - y_0) + beta * (y_2 - y_0)
    """
    order = np.argsort(fibre_table_hexa['FIBNUM'])
    x = fibre_table_hexa['FIB_PX'][order].astype(float)
    y = fibre_table_hexa['FIB_PY'][order].astype(float)
    alpha = ((y[0] * (x[2] - x[0]) - x[0] * (y[2] - y[0])) / 
             ((x[1] - x[0]) * (y[2] - y[0]) -
              (y[1] - y[0]) * (x[2] - x[0])))
    beta = ((y[0] * (x[1] - x[0]) - x[0] * (y[1] - y[0])) /
            ((x[2] - x[0]) * (y[1] - y[0]) -
             (y[2] - y[0]) * (x[1] - x[0])))
    return alpha, beta

def coordinate_centre(fibre_table_hexa, x_name, y_name, alpha, beta):
    """Return the centre of the hexabundle in the given coordinates."""
    order = np.argsort(fibre_table_hexa['FIBNUM'])
    x = fibre_table_hexa[x_name][order]
    y = fibre_table_hexa[y_name][order]
    cen_x = x[0] + alpha * (x[1] - x[0]) + beta * (x[2] - x[0])
    cen_y = y[0] + alpha * (y[1] - y[0]) + beta * (y[2] - y[0])
    return cen_x, cen_y
    
def interpolate(fibre_table_hexa, x_name, y_name, new_fibpos_x, new_fibpos_y):
    """Update the coordinates in x/y_name to the new fibpos_x/y positions.

    Works by interpolating between the old fibpos_x/y positions, which are
    in fibre_table_hexa. The coordinates are assumed to relate to
    fibpos_x/y according to:
        x = x_0 + a_x * fibpos_x + b_x * fibpos_y
        y = y_0 + a_y * fibpos_x + b_y * fibpos_y
    x_0, a_x, b_x, y_0, a_y, b_y are found by fitting to the old coordinates.
    """
    old_coords_x = fibre_table_hexa[x_name]
    old_coords_y = fibre_table_hexa[y_name]
    old_fibpos_x = fibre_table_hexa['FIBPOS_X']
    old_fibpos_y = fibre_table_hexa['FIBPOS_Y']
    # Define the function to fit
    fitfunc = lambda par, fibpos_x, fibpos_y: \
        par[0] + par[1]*fibpos_x + par[2]*fibpos_y
    errfunc = lambda par, fibpos_x, fibpos_y, coords: \
        coords - fitfunc(par, fibpos_x, fibpos_y)
    # Initial guess for x
    par_x_0 = np.zeros(3)
    par_x_0[1] = ((old_coords_x.max() - old_coords_x.min()) /
                  (old_fibpos_x.max() - old_fibpos_x.min()))
    par_x_0[0] = old_coords_x.mean() / (par_x_0[1] * old_fibpos_x.mean())
    # Do the fit for x
    args_x = (old_fibpos_x, old_fibpos_y, old_coords_x)
    par_x = leastsq(errfunc, par_x_0, args=args_x)[0]
    # Initial guess for x
    par_y_0 = np.zeros(3)
    par_y_0[2] = ((old_coords_y.max() - old_coords_y.min()) /
                  (old_fibpos_y.max() - old_fibpos_y.min()))
    par_y_0[0] = old_coords_y.mean() / (par_y_0[2] * old_fibpos_y.mean())
    # Do the fit for x
    args_y = (old_fibpos_x, old_fibpos_y, old_coords_y)
    par_y = leastsq(errfunc, par_y_0, args=args_y)[0]
    # Now use the new_fibpos_x/y to get the new coordinates
    new_coords_x = fitfunc(par_x, new_fibpos_x, new_fibpos_y)
    new_coords_y = fitfunc(par_y, new_fibpos_x, new_fibpos_y)
    # Finally, save the new coordinates
    fibre_table_hexa[x_name][:] = new_coords_x
    fibre_table_hexa[y_name][:] = new_coords_y
    return
    
def copy_coords(fibre_table, destination=None):
    """Copy the fibre coordinate information from a given fibre table.
    
    Destination is a different existing fibre table. If no destination is
    provided, a new hdu containing the coordinate data is returned."""
    name_list = ['XPOS', 'YPOS',
                 'FIBPOS_X', 'FIBPOS_Y',
                 'FIB_MRA', 'FIB_MDEC',
                 'FIB_ARA', 'FIB_ADEC',
                 'PORIENT']
    fmt_list = ['1D', '1D',
                '1J', '1J',
                '1D', '1D',
                '1D', '1D',
                '1D']
    if destination == None:
        # No destination is provided so we need to make a new hdu
        col_list = []
        for name, fmt in zip(name_list, fmt_list):
            col_list.append(pf.Column(name=name, format=fmt,
                                      array=fibre_table[name].copy()))
        tbhdu = pf.new_table(col_list)
        # Name the extension so it can be found later
        tbhdu.update_ext_name('OLD_COORDS')
        return tbhdu
    else:
        # We have an existing table to copy the coordinates into
        for name in name_list:
            for entry, value in zip(destination, fibre_table.field(name)):
                entry[name] = value
        return

def correct_coordinates(filename):
    """See which corrections are necessary and apply them to the file.

    If the hexabundles have PORIENT = 0.0, they will be rotated 180
    degrees. If the probes are in the wrong order, they will be
    re-ordered. If neither of these is the case, nothing is done."""
    pass
    
