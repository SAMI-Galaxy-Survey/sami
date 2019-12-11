"""
Specific settings for instrument and fibre optics geometry, in this case for the SAMI AAO survey. 
Optical fibre geometry is assumed to be circular. 
For non-optical-fibre instruments, change response matrix calculation
"""

# Constants for the SAMI instrument
####################################
# Number of exposures(frames) and Fibres:
_Nfib = 61, #11
# Radius of individual SAMI fibres in arcseconds:
_Rfib_arcsec = 0.798
# Approximate plate scale:
_plate_scale = 15.22
# Approximate radius of Field-of_view covered by fibre bundle:
_fov_arcsec = 14.7
