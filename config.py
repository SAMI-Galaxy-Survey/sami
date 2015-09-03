import astropy.units as u
import astropy.coordinates as coords
import os.path
from astropy import __version__ as ASTROPY_VERSION

# This script contains constants that are used in other SAMI packages.

ASTROPY_VERSION = tuple(int(x) for x in ASTROPY_VERSION.split('.'))

# ----------------------------------------------------------------------------------------

# Approximate plate scale
plate_scale=15.22

# Diameter of individual SAMI fibres in arcseconds
fibre_diameter_arcsec = 1.6

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# For Andy's dome windscreen position script.
# Distance between the polar and declination axes
polar_declination_dist=0.0625 # dome radii

# Distance between the declination axis & dome center on meridian
declination_dome_dist=0.0982  # dome radii

# Latitude of SSO (based on AAT Zenith Position, may not be most accurate)
latitude=coords.Angle(-31.3275, unit=u.degree)

# astropy version catch to be backwards compatible
if ASTROPY_VERSION[0] == 0 and ASTROPY_VERSION[1] <= 2:
    latitude_radians = latitude.radians
    latitude_degrees = latitude.degrees
else:
    latitude_radians = latitude.radian
    latitude_degrees = latitude.degree
# ----------------------------------------------------------------------------------------

# Pressure conversion factor from millibars to mm of Hg 
millibar_to_mmHg = 0.750061683

# Set the test data directory, assumed to be at the same level as the sami package.
test_data_dir = os.path.dirname(os.path.realpath(__file__)) + '/../test_data/'
