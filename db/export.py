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
import os
import sys

""" 
Commit message: 

For commit message:

Wrote new, modular export() code. Fixed bugs in ingest.py. 

export.py: 

Wrote new export() code that calls copyH5() and optionally unpackFITS(). Very lightweight itself. WARNING: fits outputs do not work right now. There is a problem with HDF5 blocks full of external links. Can't be read by unpacking code, "can't open object". 

ingest.py: 

File access in create(overwrite=True) was changed to 'a' at the Nov'13 Busy Week. That isn't what we want though: we need to truncate not append. Also got rid of a little print statement that seems to have been left over from debugging.  

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
def getVersion(h5file, hdf, version):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Identify latest version in a given SAMI archive. """

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
def getTargetGroup(hdf, name, version):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Determine target group of requested SAMI galaxy/star. """

    # First determine the observation type. 
    if (('Target' in hdf['SAMI/'+version].keys()) and 
        (name in hdf['SAMI/'+version+'/Target'].keys())):
        obstype = 'Target'
    if (('Calibrator' in hdf['SAMI/'+version].keys()) and 
        name in hdf['SAMI/'+version+'/Calibrator'].keys()):
        obstype = 'Calibrator'

    # If it isn't there, throw a fit. 
    if (name not in hdf['SAMI/'+version+'/Target']) and\
       (name not in hdf['SAMI/'+version+'/Calibrator']):
        raise SystemExit('The nominated h5 file ('+h5file+') does not '+\
                         'contain a target block for SAMI '+name)
    else:
        g_target = hdf['SAMI/'+version+'/'+obstype+'/'+name]
        return (g_target, obstype)


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
def defOutfile(hdf, name, colour, overwrite):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Define the output filename, if not set """

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
def copyH5(name, h5archive, version, colour='', 
           outFile='SAMI_archive_file.h5', outType='fits', 
           getCube=True, getRSS=False, overwrite=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Copy or link datasets from the SAMI archive to a new h5 file """

    from sami.db.ingest import create
    import string
    import random
    import itertools

    # *** NB: This only copies Targets now, not Calibrators. Is that bad? 

    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        return ''.join(random.choice(chars) for x in range(size))
        
    # Interpert input, turn all names into strings. 
    if name is not isinstance(list, basestring):
        name = [name]
    if any(name) is not str:
        name = [str(s) for s in name]
    
    if colour == '':
        colour = ['B','R']

    # Generate an h5 file, open hdf, copy/link dsets.
    if outType == 'fits':
        fname = id_generator()+'.h5'
    if outType == 'h5':
        fname = outFile
    if (outType != 'fits') & (outType != 'h5'):
        raise SystemExit("Please define 'outType' as 'fits' or 'h5'.")

    
    """There is a slight, but intended inefficiency in the follwoing open-close
    -open process, written to take advantage of the i/o catch in create()."""
    create(fname, overwrite=overwrite)
    hdfOUT = h5.File(fname, 'a')

    # Copy the SAMI Target Table. 
    tMaster = '/SAMI/'+version+'/Table/SAMI_MASTER'
    if outType == 'fits':
        hdfOUT[tMaster] = h5.ExternalLink(h5archive, tMaster)
        
    if outType == 'h5':
        hdfOUT.copy(h5.File(h5archive, 'r')[tMaster], tMaster)

    # Commence name loop. 
    for thisTarg in range(len(name)):

        # Populate the list of dset to copy/link. 
        dsets = []
        dsetRSS = []
        extCube = ['Data', 'Variance', 'Weight']
        extRSS  = ['Data', 'Variance', 'FibreTable']
        
        if getCube:
            if 'B' in colour:
                dsets.append(['B_'+s for s in ['Cube_'+s for s in extCube]])
            if 'R' in colour:
                dsets.append(['R_'+s for s in ['Cube_'+s for s in extCube]])
                
        if getRSS:
            # Diagnose number of RSS parents, append all permutations to list. 
            myKeys = h5.File(h5archive, 'r')['/SAMI/'+version+'/Target/'+\
                                             name[thisTarg]].keys()
            if 'B' in colour:
                nRSS = sum(['B_RSS_Data' in s for s in myKeys])
                lRSS = ["{:01d}".format(s) for s in 1+np.arange(nRSS)]
                #dsets.append(['B_'+s for s in ['Cube_'+s for s in extRSS]])
                dsetB_RSS = map('_'.join, itertools.chain(\
                                            itertools.product(extRSS, lRSS)))
                dsets.append('B_RSS_'+s for s in dsetB_RSS)
            if 'R' in colour:
                nRSS = sum(['R_RSS_Data' in s for s in myKeys])
                lRSS = ["{:01d}".format(s) for s in 1+np.arange(nRSS)]
                #dsets.append(['R_'+s for s in ['Cube_'+s for s in extRSS]])
                dsetR_RSS = map('_'.join, itertools.chain(\
                                            itertools.product(extRSS, lRSS)))
                dsets.append('R_RSS_'+s for s in dsetR_RSS)
        
        # Flatten dsets into a single list (not a list of lists)
        dsets = [item for sublist in dsets for item in sublist]

        # Commence dset loop. 
        for allDsets in range(len(dsets)):
            thisDset = '/SAMI/'+version+'/Target/'+\
                     name[thisTarg]+'/'+dsets[allDsets]
            
            if outType == 'fits':
                hdfOUT[thisDset] = h5.ExternalLink(h5archive, thisDset)
                
            if outType == 'h5':
                hdfOUT.copy(h5.File(h5archive, 'r')[thisDset], thisDset)
                
    hdfOUT.close()
    return(fname)


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def unpackFITS(h5IN, h5archive, overwrite=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Package contents of an h5 block to multi-extention FITS files """

    """ 
    MAJOR BUG: This does not like the ExtendLinked HDF5 files one bit... Only 
    real blocks. No idea why. 
    
    (1) Read h5 baby block of symbolic links. 
    (2) Count number of target blocks. 
    (3) Begin loop over packaging function. 
      -- Write a single row of the SAMI Master as a primary HDU. 
      -- Write each dataset as a FITS extension with corresponding header. 
    (4) Inform user of FITS file screated, exit successfully. 
    """
    
    # Open h5 file. 
    hdf = h5.File(h5IN, 'r')

    # Count number of target blocks. 
    version = hdf['/SAMI'].keys()[0] # = getVersion(h5IN, hdf, '')
    # *** Assuming only one version of data available.  
    g_version = hdf['/SAMI/'+version]
    
    nTarget = 0
    nCalibrator = 0
    
    if 'Target' in g_version.keys():
        nTarget = len(g_version['Target'].keys())
        gTarget = g_version['Target']
        thereAreTargets = True
    if 'Calibrator' in g_version.keys():
        nCalibrator = len(g_version['Calibrator'].keys())
        gCalibrator = g_version['Calibrator']
        thereAreCalibrators = True
    
    nGroups = nTarget + nCalibrator

    def plural(nGroups):
        plural = '' 
        if nGroups > 1: plural == 's'
        return(plural)

    print("Identified "+str(nGroups)+" Target Block"+plural(nGroups)+\
          " in '"+h5IN+"'.")

    def stripTable(name, version, h5archive):
        #master = hdf['/SAMI/'+version+'/Table/SAMI_MASTER']
        h5archive = h5.File(h5archive, 'r')
        master = h5archive['/SAMI/'+version+'/Table/SAMI_MASTER']
        tabline = master[master["CATID"] == int(name)][0]
        # For now excluding all strings to make FITS-compatible
        # *** BUT HEADER will not know that.
        hdu = [v for v in tabline if not isinstance(v, str)]
        hdr = makeHead(master)
        h5archive.close()
        return(hdu, hdr)

    # Begin loop over all SAMI targets requested. 
    # *** CURRENTLY ONLY Targets, not Calibrators. Combine groups in a list?
    for thisG in range(nTarget):

        # What is the SAMI name of this target?
        name = gTarget.keys()[thisG]

        # Search for 'Cube' and 'RSS' among Dsets to define output filename
        areThereCubes = ['Cube' in s for s in gTarget[name].keys()]
        areThereRSS   = ['RSS'  in s for s in gTarget[name].keys()]
        sContents = []
        if sum(areThereCubes) > 0: sContents.append('cubes')
        if sum(areThereRSS)   > 0: sContents.append('RSS')
        if len(sContents) > 1: sContents = '_'.join(sContents)
        else: sContents = sContents[0]
    
        # Define output filename
        fname = 'SAMI_'+name+'_'+sContents+'.fits'

        # Primary HDU is a single row of the Master table. 
        hdu0, hdr0 = stripTable(name, version, h5archive)
        hdulist = pf.HDUList([pf.PrimaryHDU(hdu0, header=hdr0)])
        
        # Cycle through all dsets, make HDUs and headers with native names. 

        # Get number of datasets. 
        thisTarget = gTarget[name]
        nDsets = len(thisTarget.keys())

        # Begin loop through all datasets. 
        for thisDset in range(nDsets):
        #for thisDset in range(5):
            
            # Determine dataset. 
            dsetName = thisTarget.keys()[thisDset]
            print("Processing dataset '"+dsetName+"'...")

            # Create dataset and populate header. 
            data = thisTarget[dsetName]
            hdr = makeHead(data)

            # Add all this to an HDU.
            hdulist.append(
                pf.ImageHDU(np.array(thisTarget[dsetName]), 
                            name=dsetName, 
                            header=makeHead(data) ) )

        # Write to a new FITS file.
        hdulist.writeto(fname, clobber=overwrite)

    hdf.close()



# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def fetch_cube(name, h5file, version='', colour='', 
               getCube=True, getRSS=False, getAll=False, overwrite=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ A tool to fetch a datacube in FITS format. 

    name      [str]  The name of the SAMI target required. 
    h5file    [str]  The SAMI archive file from which to export. 
    version   [str]  Data version sought. Latest is default. 
    colour    [str]  Colour-specific export. Set to 'B' or 'R'. 
    overwrite [boo]  Overwrite output file as default. 
    """

    # Open HDF5 file and run a series of tests and diagnostics. 
    hdf = h5.File(h5file, 'r')

    # Check for SAMI formatting. 
    SAMIformatted = checkSAMIformat(hdf)

    # Convert SAMI ID to string
    if name is not str:
        name = str(name)

    # Get the data version. 
    if version == '': 
        version = getVersion(h5file, hdf, version)
    
    # Get target group and observation type. 
    g_target, obstype = getTargetGroup(hdf, name, version)

    # Check for monochrome output
    if colour == '':                              
        colour = ['B','R']
    
    for col in range(len(colour)):

        # Look for cubes:
        cubeIsComplete = completeCube(hdf, colour[col], g_target)

        # Set name for output file (if not set): 
        #if outfile == '': 
        outfile = defOutfile(hdf, name, colour[col], overwrite)

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

def validate_sami_id(candidate_sami_id):
    """Dummy validation function for SAMI IDs, always returns true."""
    return True

def interpret_id_list_from_file(filename):
    """Execute interpret_id_list on the contents of filename."""
    
    return interpret_id_list(open(filename).read())
    
def interpret_id_list(id_list_string):
    """Extract SAMI IDs and extra information from a string list.
    
    Arguments
    ---------
    
        id_list_string (string): A string containing a list of SAMI IDs and 
            additional information as a white space separated table.

    Returns: a tuple of two lists, the first being the list of SAMI IDs, and
    the second being the corresponding list of additional information.
            
    The input list is expected to be one SAMI ID per line, followed by any
    additional information provided by the user, such as a cross-
    identification. This is split up and returned as two lists, one of the
    SAMI IDs, and one of the corresponding additional information. SAMI IDs
    are validated with validate_sami_id, and invalid IDs are removed. Blank
    lines and lines starting with a hash character (#) are treated as comments
    and removed.

    In reality, this is all accomplished with a regular expression, so the
    syntax of the input is actually much more forgiving.

    """
    
    from re import compile, VERBOSE

    assert isinstance(id_list_string, str)
    
    split_re = compile(r"""
        \A          # Start of line
        \s*         # White space at start of line (optional)
        (\d+)       # Digits of SAMI ID (captured)
        (?:         # Start of non-capturing grouping
            [,\s+]  # Whitespace and/or comma separator from extra data
            (.*?)   # Extra data (captured)
        )?          # End of grouping, contents of group optional
        \s*         # Whitespace at end of line (optional)
        \Z          # End of line """,
                       VERBOSE)
    
    lines = id_list_string.splitlines()

    # Remove lines which do not match the pattern:
    lines = filter(split_re.match, lines)
        
    # Split into ID and extra_data, with the empty string for extra_data if
    # missing:
    id_info_list = [split_re.match(l).groups("") for l in lines]
    
    # Remove rows with invalid SAMI IDs
    id_info_list = filter(lambda l: validate_sami_id(l[0]), id_info_list)
    
    # Convert into separate lists
    id_list = [l[0] for l in id_info_list]
    info_list = [l[1] for l in id_info_list]

    assert isinstance(id_list, list)
    assert isinstance(info_list, list)
    
    return id_list, info_list

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def export(name, h5file, version='', getCube=True, getRSS=False, 
           colour='', outType='fits', overwrite=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ The main SAMI_DB .fits export function """

    # Check that some data have been requested. 
    if (not getCube) and (not getRSS): 
        raise SystemExit("Please raise at least one of the 'get???' flags.")

    # Open HDF5 archive. This accommodates some access issues, but should be 
    #  written better... It is only for getVersion(). 
    hdf = h5.File(h5file, 'r')
    
    # Get latest version of data. 
    if version == '': 
        version = getVersion(h5file, hdf, version)

    # Run copyH5 and optionally unpack FITS.
    babyH5 = copyH5(name, h5file, version, colour=colour, 
                    outType=outType, getCube=getCube, getRSS=getRSS, 
                    overwrite=overwrite)

    if outType == 'fits':
        unpackFITS(babyH5, h5file, overwrite=overwrite)


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def export_OLD(name, h5file, get_cube=False, get_rss=False, 
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
        raise SystemExit("Please raise at least one of the 'get???' flags.")
    
    # Open HDF5 file and run a series of tests and diagnostics. 
    hdf = h5.File(h5file, 'r')

    SAMIformatted = checkSAMIformat(hdf)         # Check for SAMI formatting
    if version == '':                            # Get the data version
        version = getVersion(h5file, hdf, version)
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
