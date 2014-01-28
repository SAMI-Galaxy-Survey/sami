import astropy.units as u
import astropy.coordinates as coords
import os.path
from astropy import __version__ as astropy_version

# This script contains constants that are used in other SAMI packages.

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
if float(astropy_version[2]) >= 3.:
    latitude_radians=latitude.radian
else:
    latitude_radians=latitude.radians
# ----------------------------------------------------------------------------------------

# Pressure conversion factor from millibars to mm of Hg 
millibar_to_mmHg = 0.750061683

# Set the test data directory, assumed to be at the same level as the sami package.
test_data_dir = os.path.dirname(os.path.realpath(__file__)) + '/../test_data/'