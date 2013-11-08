import astropy.units as u
import astropy.coordinates as coords
import os.path

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
latitude_radians=latitude.radians
# ----------------------------------------------------------------------------------------
