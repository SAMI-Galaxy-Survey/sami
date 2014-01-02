"""
'export.py', module within 'sami.db'

Description: Export codes for the SAMI Galaxy Survey data archive. 

Written: 08.10.2013, Iraklis Konstantopoulos. Based on 'db.database'.

Contact: iraklis@aao.gov.au

Updates: .-

Table of Contents: 

.fetch_cube     Extract a cube or set of cubes. 
.export         Export any amount of data products of various types. 
"""

import numpy as np
import h5py as h5
import astropy.io.fits as pf
import astropy.io.ascii as ascii
import os
import sys
import sami

""" 
Commit message: 

Modularised fetch_cube() further. Bug fixes. 

-- moved makeHead() function outside fetch_cube(). 
-- writeto() had a clobber flag added, as it was creaed and written to separately. That didn't fly with the latest version of pyFITS. Should suppress warning from pyFITS ("Overwriting existing file"), as it is misleading. 
-- Introduced option to overwrite an existing file. 
-- fetch_cube() can now morph into the export() function by adding an RSS exporter segment. 
"""

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def checkSAMIformat(hdf):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Check that h5file is SAMI-formatted. """    
    if 'SAMI' not in hdf.keys():
        raise SystemExit('The nominated h5 file ('+h5file+') is not '+\
                         'SAMI-formatted')

    else: 
        SAMIformatted = True

    return SAMIformatted


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def getVersion(hdf, version):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Identify (latest) version in a given SAMI archive. """

    # Identify version sought, or the latest availabe (default). 
    if hdf['SAMI'].keys() != '':
        
        if version == '':
            numeric_versions = [float(str_in) for str_in in hdf['SAMI'].keys()]
            version = str(max(numeric_versions))
    else:
        SystemExit('The nominated h5 file ('+h5file+') does not '+\
                   'appear to contain any SAMI data releases. Try again!')

    return version
        

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def getObstype(hdf, name, version):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Identify observation type for a given SAMI target. """

    if (('Target' in hdf['SAMI/'+version].keys()) and 
        (name in hdf['SAMI/'+version+'/Target'].keys())):
        obstype = 'Target'
    if (('Calibrator' in hdf['SAMI/'+version].keys()) and 
        name in hdf['SAMI/'+version+'/Calibrator'].keys()):
        obstype = 'Calibrator'

    return obstype


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def getTargetGroup(hdf, name, version, obstype):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Determine target group of requested SAMI galaxy/star. """

    # If it isn't there, throw a fit. 
    if (name not in hdf['SAMI/'+version+'/Target']) and\
       (name not in hdf['SAMI/'+version+'/Calibrator']):
        raise SystemExit('The nominated h5 file ('+h5file+') does not '+\
                         'contain a target block for SAMI '+name)
    else:
        g_target = hdf['SAMI/'+version+'/'+obstype+'/'+name]
        return g_target


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def completeCube(hdf, colour, group):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Check that a requested cube is all there (data, var, wht). """
    
    if (colour+'_Cube_Data' not in group.keys()) or \
       (colour+'_Cube_Variance' not in group.keys()) or \
       (colour+'_Cube_Weight' not in group.keys()):
        
        hdf.close()
        raise SystemExit(
            'The target block is incomplete, please check archive.')
        
    else:
        cubeIsComplete = True
        return cubeIsComplete
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def defOutfile(hdf, name, colour, outfile, overwrite):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Define the output filename, if not set """

    if outfile == '': 
        outfile = 'SAMI_'+name+'_'+colour+'_cube.fits'
        
    # Check if outfile already exists
    if os.path.isfile(outfile) and (not overwrite):
        hdf.close()
        raise SystemExit("The nominated output .fits file ('" + 
                         outfile + "') already exists. Try again! ")

    return outfile


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def makeHead(dset):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Construct a header out of a set of HDF5 dataset attributes. """
    
    cards = []
    
    for ihdr in range(len(dset.attrs.keys())):
        aCard = pf.Card(keyword=dset.attrs.keys()[ihdr], 
                        value=dset.attrs.values()[ihdr])
        cards.append(aCard)
        
    hdr = pf.Header(cards=cards)
    return hdr


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def fetchCube(name, h5file, colour=''):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Run export() with simplest settings, fetch only cube. """


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def fetch_cube(name, h5file, version='', colour='', outfile='', 
               getCube=True, getRSS=False, getAll=False, overwrite=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ A tool to fetch a datacube in FITS format. 

    name      [str]  The name of the SAMI target required. 
    h5file    [str]  The SAMI archive file from which to export. 
    version   [str]  Data version sought. Latest is default. 
    colour    [str]  Colour-specific export. Set to 'B' or 'R'. 
    outfile   [str]  The name of the file to be output. Default: "col_name".
    overwrite [boo]  Overwrite output file as default. 
    """

    # Open HDF5 file and run a series of tests and diagnostics. 
    hdf = h5.File(h5file, 'r')

    SAMIformatted = checkSAMIformat(hdf)         # Check for SAMI formatting
    if version == '':                            # Get the data version
        version = getVersion(hdf, version)
    obstype = getObstype(hdf, name, version)     # Get observation type
    g_target = getTargetGroup(hdf, name,         # ID target group 
                              version, obstype)
    if colour == '':                             # Check for monochrome output
        colour = ['B','R']
    
    for col in range(len(colour)):

        # Look for cubes:
        cubeIsComplete = completeCube(hdf, colour[col], g_target)

        # Set name for output file (if not set): 
        if outfile == '': 
            outfile = defOutfile(hdf, name, colour[col], outfile, overwrite)

        # Data is primary HDU, VAR and WHT are extra HDUs. 
        data = g_target[colour[col]+'_Cube_Data']
        var =  g_target[colour[col]+'_Cube_Variance']
        wht =  g_target[colour[col]+'_Cube_Weight']
        
        # Construct headers. 
        hdr1 = makeHead(data)
        hdr2 = makeHead(var)
        hdr3 = makeHead(wht)
      
        # And now set up the Image Data Units. 
        hdu_c1 = pf.PrimaryHDU(np.array(data), hdr1)
        hdu_c2 = pf.ImageHDU(np.array(var), name='VARIANCE', header=hdr2)
        hdu_c3 = pf.ImageHDU(np.array(wht), name='WEIGHT', header=hdr3)
        
        hdulist = pf.HDUList([hdu_c1, hdu_c2, hdu_c3])
        hdulist.writeto(outfile, clobber=overwrite)
        
        hdulist.close()
        
    # close h5file and end process
    hdf.close()
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def export(name, h5file, get_cube=False, get_rss=False, 
           colour='', all_versions=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ The main SAMI_DB .fits export function 
 
    name       [str]  The name(s) of the SAMI target(s) required. 
    h5file     [str]  The SAMI archive file from which to export. 
    get_cube   [boo]  Export data-cube(s). 
    get_rss    [boo]  Export RSS file(s). 
    rss_centre [boo]  Export only central RSS file. *UNDER CONSTRUCTION*
    get_sdss   [boo]  Fetch an SDSS g' cutout. *UNDER CONSTRUCTION*
    get_meta   [boo]  Fetch emission line maps. *UNDER CONSTRUCTION*
    colour     [str]  Colour-specific export. Set to 'blue' or 'red'. 
    outfile    [str]  The name of the file to be output. Default: "col_name".
    """
    
    # Check that some data have been requested: 
    if (not get_cube) and (not get_rss): 
        raise SystemExit("Please raise at least one of the 'get_???' flags.")
    
    # Open HDF5 file and run a series of tests and diagnostics. 
    hdf = h5.File(h5file, 'r')

    SAMIformatted = checkSAMIformat(hdf)         # Check for SAMI formatting
    if version == '':                            # Get the data version
        version = getVersion(hdf, version)
    obstype = getObstype(hdf, name, version)     # Get observation type
    g_target = getTargetGroup(hdf, name,         # ID target group 
                              version, obstype)
    if colour == '':                             # Check for monochrome output
        colour = ['B','R']
    
    # Look for cubes:
    if ('Blue_cube_data' not in targ_group.keys()) or \
       ('Blue_cube_variance' not in targ_group.keys()) or \
       ('Blue_cube_weight' not in targ_group.keys()) or \
       ('Red_cube_data' not in targ_group.keys()) or \
       ('Red_cube_variance' not in targ_group.keys()) or \
       ('Red_cube_weight' not in targ_group.keys()):
        
        raise SystemExit(
            'The target block is incomplete, please check archive.')

    # Start the HDU and extension lists
    all_hdu = []
    all_ext = []

    # For now add a dummy Primary HDU
    phdu = pf.PrimaryHDU(np.arange(10))
    all_hdu.append(phdu)

    for col in range(len(colour)):

        # Figure out what will go in this multi-extension fits file. 
        # ***(Should also think about somple .h5 outputs)***
    
        if get_cube: 

            data = targ_group[colour[col]+'_Cube_Data']
            var =  targ_group[colour[col]+'_Cube_Variance']
            wht =  targ_group[colour[col]+'_Cube_Weight']

            all_ext.append(colour[col]+' Cube Data')
            all_hdu.append(pf.ImageHDU(np.array(data), 
                                       name=colour[col]+' DAT'))
            
            all_ext.append(colour[col]+' Cube Variance')
            all_hdu.append(pf.ImageHDU(np.array(var), 
                                       name=colour[col]+' VAR'))
            
            all_ext.append(colour[col]+' Cube Weight')
            all_hdu.append(pf.ImageHDU(np.array(wht), 
                                       name=colour[col]+' WHT'))
            
        if get_rss:
            for irss in range(7):

                rss_data = targ_group[colour[col]+'_RSS_data_'+str(irss+1)]
                rss_var = targ_group[colour[col]+'_RSS_variance_'+str(irss+1)]

                all_ext.append(colour[col]+' RSS Data '+str(irss+1))
                all_hdu.append(pf.ImageHDU(np.array(rss_data), 
                               name=colour[col]+' RSS DAT '+str(irss+1)))

                all_ext.append(colour[col]+' RSS Variance '+str(irss+1))
                all_hdu.append(pf.ImageHDU(np.array(rss_var), 
                               name=colour[col]+' RSS VAR '+str(irss+1)))

        """

        # Build Primary Header Unit as a table listing all extensions
        all_ext = []
        if get_cube: 
            n_ext_cube = len(colour)*3
            all_ext.append('Extensions 1-'+str(n_ext_cube)+' = Datacube')
            
        if get_rss:
            n_ext_rss = len(colour)*7
            all_ext.append('Extensions '+str(n_ext_cube+1)+'-'+
                           n_ext_cube+1+n_ext_rss+' = RSS strips')

        # Make primary unit, just that list of strings. 
        phu = pf.PrimaryHDU(all_ext)
        
        hdu_c1 = pf.ImageHDU(np.array(data), hdr1, name=colour[col]+' DATA')
        """

    # Write HDU list to and All_Hdu
    hdulist = pf.HDUList(all_hdu)
    hdulist.writeto('dummy_export.fits')

    # Close HDU, h5, and wrap it up. 
    hdulist.close()
    hdf.close()


""" THE END """
