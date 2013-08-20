import astropy.units as u
import astropy.coordinates as coords

# This script contains constants that are used in other SAMI packages.

# ----------------------------------------------------------------------------------------
# rough plate scale
plate_scale=15.22
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

# Pressure conversion factor from millibars to mm of Hg 
millibar_to_mmHg = 0.750061683