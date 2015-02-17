import os
import numpy as np
from math import pi, sqrt, sin, cos
from matplotlib import pyplot as plt

import pyfits as pf

try :
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    # print 'Requires healpy -- developed with py27-healpy @1.8.1_0'
    # print 'If you use Macports to manage your python installation,'
    # print 'type "sudo port install py27-healpy".'

from astropy import coordinates as co
from astropy import units as u

MAPS = {}

MAPS_FILES = {
    'planck': ('dust/HFI_CompMap_ThermalDustModel_2048_R1.20.fits', 2),
    'sfd98': ('dust/lambda_sfd_ebv.fits', 0),
}

def load_map(name, force_reload=False):
    """Load the dust maps from various sources."""
    if name not in MAPS or force_reload:
        path, field = MAPS_FILES[name]
        try:
            MAPS[name] = hp.read_map(path, field=field)
        except IOError:
            return False
    return True


# planck_dust_map_filename = 'HFI_CompMap_ThermalDustModel_2048_R1.20.fits'
# if not os.path.exists( planck_dust_map_filename ):
#     print 'WARNING: Cannot find Planck dust map with file name:'
#     print ' '*8, planck_dust_map_filename
#     print
#     print 'This can be downloaded via the Planck Explanatory Supplement wiki:'
#     print ' '*8, 'http://wiki.cosmos.esa.int/planckpla'
#     print '(Look under Mission Products > CMB and astrophysical component maps,'
#     print 'and be sure to download the higher resolution Nside=2048 version.'
#     print 'Note that this is a ~1.6 Gb download!)'
#     print
# else :
#     print 'reading Planck dust map from file:'
#     print ' '*8, planck_dust_map_filename
#     planckDustMap = hp.read_map( planck_dust_map_filename, field=2 )
#     hp.mollview( planckDustMap, min=0., max=1., fig=0,
#                  title='Planck dust map: %s'
#                  % planck_dust_map_filename.split('/')[-1] )
# print ; print
    
# schlegel_dust_map_filename = 'lambda_sfd_ebv.fits'
# if not os.path.exists( schlegel_dust_map_filename ):
#     print 'WARNING: Cannot find Schlegel et al. dust map with file name:'
#     print ' '*8, schlegel_dust_map_filename
#     print
#     print "This can be downloaded via NASA-Goddard's LAMBDA data archive:"
#     print ' '*8, 'http://lambda.gsfc.nasa.gov/product/foreground/f_products.cfm'
#     print '(Look under data > foreground > products > Reddening (E(B-V)) Map,'
#     print 'and be sure to download the healpix version.'
#     print
# else :
#     print 'reading Schlegel et al. dust map from file:'
#     print ' '*8, schlegel_dust_map_filename
#     schlegelDustMap = hp.read_map( schlegel_dust_map_filename, field=0 )
#     hp.mollview( planckDustMap, min=0., max=1., fig=1,
#                  title='Schlegel, Finkbinder & Davis (1998) dust map: %s'
#                  % schlegel_dust_map_filename.split('/')[-1] )
# print ; print




def healpixAngularCoords( ra, dec ):
    pos = co.SkyCoord( ra*u.deg, dec*u.deg ).galactic
    theta, phi = pi/2. - pos.b.rad, pos.l.rad
    return theta, phi

# def Planck_EBV( theta, phi ):
#     return hp.get_interp_val( planckDustMap, theta, phi )

# def Schlegel_EBV( theta, phi ):
#     return hp.get_interp_val( schlegelDustMap, theta, phi )

def EBV(name, theta, phi):
    """
    Return E(B-V) for given map at given location.

    Valid names are 'planck' or 'sfd98'.
    """
    success = load_map(name)
    if success:
        return hp.get_interp_val(MAPS[name], theta, phi)
    else:
        return None




# def foregroundCorrection( ra, dec, wavelength ):
#     print 'Looking up MW dust redding at (RA, Dec) = (%10.6f, %+10.6f).'
#     theta, phi = healpixAngularCoords( ra, dec )

#     EBV1 = Schlegel_EBV( theta, phi )
#     print 'E(B-V) from Schlegel dust map is: %.4f' % EBV1
#     EBV2 = Planck_EBV( theta, phi )
#     print 'E(B-V) from Planck dust map is:   %.4f' % EBV2

#     correction = MilkyWayDustCorrection( wavelength, EBV2, dustlaw='CCM89' )
#     # this is the multiplicative scaling to correct for foreground dust
#     return correction, EBV1, EBV2
        



def dustCorrectSAMICube( path, overwrite=False ):
    hdulist = pf.open(path, 'update')
    try:
        hdu = hdulist['DUST']
    except KeyError:
        # HDU does not exist; make it
        hdu = pf.ImageHDU()
        hdu.name = 'DUST'
        hdulist.append(hdu)
    else:
        # HDU does exist. Do we want to overwrite it?
        if not overwrite:
            # Don't overwrite; get out of here!
            hdulist.close()
            return
    print 'Recording dust data for ' + os.path.basename(path)
    header = hdulist[0].header
    ra, dec = header[ 'CATARA' ], header[ 'CATADEC' ]
    wl = header[ 'CRVAL3' ] + ( header[ 'CDELT3' ] *
                                (1 + np.arange( header[ 'NAXIS3' ] ))
                                - header[ 'CRPIX3' ] )
    theta, phi = healpixAngularCoords( ra, dec )
    EBV_sfd98 = EBV( 'sfd98', theta, phi )
    if EBV_sfd98 is not None:
        hdu.header['EBVSFD98'] = (
            EBV_sfd98, 'MW reddening E(B-V) from SFD98')
    else:
        print 'Warning: SFD98 dust map not available.'
    EBV_planck = EBV( 'planck', theta, phi )
    if EBV_planck is not None:
        correction = MilkyWayDustCorrection( wl, EBV_planck )
        hdu.data = correction
        hdu.header['EBVPLNCK'] = (
            EBV_planck, 'MW reddening E(B-V) from Planck v1.20')
    else:
        print 'Warning: Planck dust map not available; no dust curve recorded.'
    hdulist.flush()
    hdulist.close()
    return 




    


def MilkyWayDustCorrection( wavelength, EBV, dustlaw='CCM89' ):
    # MW dust extinction law taken from Cardelli, Clayton & Mathis (1989)
    # my implementation of this follows Madusha's

    # C89 parameterise dust extinction ito a parameter R_v=A_v/E(B-V)
    # here i assume R_v = 3.1; this is Calzetti (2001)'s value for SFers
    # this is also the value given by C89 for the diffuse ISM

    Rv = 3.1

    # C89 give k(lam) normalized to 1 Av = 1. * R_v / E(B-V)
    # i do things in E(B-V) = Av/Rv
    
    # everything is parameterized according to x = 1 / lambda [um]
    # i assume <wavelength> is given in [Angstroms] = 10000 * [um]
    x = 1./(wavelength/1e4)

    infrared = ( ( 0.3 <= x ) & ( x <= 1.1 ) )
    optical =  ( ( 1.1 <  x ) & ( x <= 3.3 ) )

    a = np.where( infrared, +0.574 * x**1.61, 0. )
    b = np.where( infrared, -0.527 * x**1.61, 0. )

    y = x - 1.82

    # NB. np.polyval does p[0]*X**(N-1) + p[1]*X**(N-2) + ... + p[-2]*X + p[-1]
    # ie. polynmoial coefficients are for higher through to lower powers
    if dustlaw == 'CCM89' :
        acoeffs = ( +0.32999, -0.77530, +0.01979, +0.72085,
                    -0.02427, -0.50447, +0.17699, 1. )
        bcoeffs = ( -2.09002, +5.30260, -0.62251, -5.38434,
                    +1.07233, +2.28305, +1.41338, 0. )
    elif dustlaw == 'OD94' :
        acoeffs = ( -0.505, +1.647, -0.827, -1.718,
                    +1.137, +0.701, -0.609, +0.104, 1. )
        bcoeffs = ( +3.347,-10.805, +5.491,+11.102,
                    -7.985, -3.989, +2.908, +1.952, 0. )
    else :
        print 'Do not recognise the given dust law:', dustlaw
        print 'Recognised options are:'
        print '--- CCM89 (Cardelli, Clayton & Mathis, 1989, ApJ 345, 245)'
        print "--- OD94  (O'Donnell, 1994, ApJ 422, 1580"
        print 'No dust correction will be performed.'
        return 1.
        
    a = np.where( optical, np.polyval( acoeffs, y ), a )
    b = np.where( optical, np.polyval( bcoeffs, y ), b )

    attenuation = a + b / Rv
    # Rv is Av / E(B-V) ; correction is normalised to Av = 1
    # so to scale to E(B-V)=1, scale correction by Rv
    attenuation *= Rv 

    transmission = 10**( -0.4 * EBV * attenuation )
    # this is the fraction of light transmitted through the foreground
    correction = 1./transmission
    # this is the multiplicative scaling to correct for foreground dust

    return correction





def gamaTest( ):
    import atpy
    print 'requires InputCatA.fits; download from GAMA DR2 webpages.'
    gama = atpy.Table( '/Users/ent/data/gama/dr2/InputCatA.fits' )
    ebv0 = gama.EXTINCTION_R / 2.751
    
    plt.figure( 5 ) ; plt.clf()
    plt.xlabel( 'EBV from GAMA catalogues' )
    plt.ylabel( 'EBV from this code' )
    plt.title( 'red = Schlegel+98 dust map; black = Planck dust map' )
    
    for i, ( ra, dec ) in enumerate( zip( gama.RA, gama.DEC ) ):
        theta, phi = thetaPhiFromRaDec( ra, dec )
        dust = Planck_EBV( theta, phi )
        dust2 = Schlegel_EBV( theta, phi )

        plt.scatter( gama.EXTINCTION_R[ i ]/3.1, dust, 1, 'k', edgecolors='none' )
        plt.scatter( gama.EXTINCTION_R[ i ]/3.1, dust2, 1, 'r', edgecolors='none' )
        if i % 1000 == 999 :
            plt.draw()
            
