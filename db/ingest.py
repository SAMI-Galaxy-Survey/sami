"""
'ingest.py', module within 'sami.db'

Description: Import codes for the SAMI Galaxy Survey data archive. 

Written: 08.10.2013, Iraklis Konstantopoulos. Based on 'db.database'.

Contact: iraklis@aao.gov.au

Updates: .-

Table of Contents (* to-be, **: under construction): 

.create         Creates an HDF5 file, not SAMI-specific. 
.format         Sets up the SAMI root and base (target-based) filestructure. 
.import_cube    Digests a SAMI datacube. Envokes 'import_rss' and 'fetch_SDSS' 
                from the SAMI_sdss module. 
.locate_rss     Queries a cube header, identifies parent RSS files, locates  
                them on the machine running the code. Returns filename listing. 
.make_list      Make a list for import_many from the contents of a directory. 
.import_table   Imports a SAMI target table. 
"""

import numpy as np
import h5py as h5
import astropy.io.fits as pf
import astropy.io.ascii as ascii
import os
import sys
import sami

""" 
For commit message: 

Continuing changes to data-import routines. 

Importing of cubes and RSS strips has been implemented. Fibre table not yet taken care of, its mix of datatypes require insertion as a compound dataset. Header information is being recorded, but comments are not supported by HDF5. This still needs to be sorted. And there are currently no variance headers accessible through the IFU object. I will return to this on Monday. 
"""

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def create(h5file, overwrite=False, verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Create a SAMI-formatted  HDF5 file """

    # Check that the 'h5file' string has a .h5 extension
    if not h5file[-3:] == '.h5': h5file = h5file + '.h5'

    # Check if file already exists, overwite if overwrite==True
    if os.path.isfile(h5file):
        file_already_exists = True
        print(file_already_exists)
        if not overwrite:
            raise SystemExit("The nominated h5 file ('"+h5file
                             +"') already exists. Please raise the overwrite "
                             +"flag if you wish to prodceed. Exiting.")
    else: file_already_exists = False
    
    # Create an h5 file
    if (not os.path.isfile(h5file)) or (overwrite==True):
        f = h5.File(h5file, 'w')

    # And require a SAMI root directory
    root = f.require_group("SAMI")
    
    # Close h5file
    f.close()

    # Screen output
    if verbose:
        if file_already_exists: prefix = 'Re-'
        else: prefix = ''
        print(prefix+"Initialised file '"+h5file+"'.")
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def import_cube(blue_cube, red_cube, h5file, version, safe_mode=False, 
                obstype='Target', ingest_rss=True, rss_only=False, 
                dataroot='./', verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Import a set of dataubes and parents to the SAMI Archive 
    
    Overhaul of original import code. The main change is the treatment of 
    versioning, which now comes from the DR pipeline (instead of being 
    automatically determined within import_cube). 
    
    The new rationale gives rise to the following blueprint: 
    
    (1) Check if the nominated h5 file (h5file) exists. 
    
    (2) Require a group for the nominated (/header-supplied) data version. 
    
    (3) Check h5file for the SAMI cubes in the data version to be imported*: 
      <> Do any data exist in this data version? EXIT with error. 
      <> No? Create target group. 
    
    (4) Check observation type by cross-ID on target/star catalogues*: 
      <> Found on star list? Add to "Calibrator" group (require group). 
      <> Found on target list? Add to "Target" group (require group). 
      <> Not found? EXIT with error. 
    
    (5) Start data import, deal with digest_rss in same way as before. 
    
    (6) Perform any QC tests and cleaning, EXIT successfully. 
    
    * Steps (2) and (3) will eventually rely on reading header tickets.
    
    [TODO] Every SAMI h5 file should contain the Target and Star catalogues as 
    tables, so that QC checks can be performed (many other reasons exist). This
    could be problematic when it comes to cluster fields. Do they all live in 
    the same archive? What is stopping us? Different Target tables... 

    [TODO] While two cubes should always be delivered, the functionality should 
    be available for monochrome importing. 
    
    [TODO] Set the default 'dataroot' to a header ticket stored by the DR 
    manager code. This ticket is not yet there. 
    
    [TODO] In export code, give the user the option to package the PSF star that
    corresponds to any Target cube being downloaded. 
     
    Arguments: 
    
    blue_cube   [str]  The name of a FITS file containing a blue SAMI cube. 
    red_cube    [str]  The name of a FITS file containing a red SAMI cube. 
    h5file      [str]  The name of the SAMI archive file onto which to save. 
    version     [str]  Version number (eventually a header item). 
    safe_mode   [boo]  Automatically creates a time-stamped backup archive. 
    ingest_rss  [boo]  Locate and import RSS parents as 'strips'. 
    rss_only    [boo]  Only import RSS parents, no cubes. 
    dataroot    [boo]  The root directory to scan for RSS parents. 
    verbose     [boo]  Toggles diagnostic and declarative verbosity. 
    """ 
    
    # Check if the nominated h5 file exists; prompt for creation if not, exit. 
    if not os.path.isfile(h5file):
        raise SystemExit("Cannot find the nominated HDF5 file ('"+h5file+"'). "+
                         "Please create a file using the 'create' function")
        
    # If file does exist, open (copy in safe_mode) and allow write privileges. 
    if safe_mode:
        import datetime
        import shutil
        if verbose: print("Safe Mode: beginning file copy.")
        date = datetime.datetime.now()
        datestamp = str(date.year).zfill(2)+str(date.month).zfill(2)+\
                    str(date.day).zfill(2)+'_'+\
                    str(date.hour).zfill(2)+str(date.minute).zfill(2)+\
                    str(date.second).zfill(2)
        bkp_file = h5file[:-3]+"_"+datestamp+".h5"
        shutil.copyfile(h5file,bkp_file)
        if verbose: print("Safe Mode: file successfully copied to '"+
                          bkp_file+"'.")

    hdf = h5.File(h5file, 'r+')

    # Check that the SAMI filesystem has been set up in this file. 
    if "SAMI" not in hdf.keys():
        hdf.close()
        raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                         "is not properly formatted. Please initialise the "+
                         "filesystem using 'SAMI_DB.format'")
    
    # First check the format of the version, If string, interpret. 
    if type(version) is str:  v_numeric = float(version)
    if type(version) is float: v_numeric = version
    
    # Require a group, leave file open
    if (verbose) and (version in hdf['SAMI'].keys()): 
        print
        print("NOTE: Some data may already be archived in this version. ")

    g_version = hdf.require_group("SAMI/"+version)
    
    # Version group in place, let's import some data! 
    """ 
    In the current version there is a built-in QC check to assign an observation
    type to the cube being imported. May change to reading a DR header ticket. 
    """
    # Read the header, find out the target name. 
    hduBLUE = pf.open(blue_cube)
    hduRED  = pf.open(red_cube)
    
    # Check that the nominated blue/red cubes are indeed of same target. 
    if hduBLUE[0].header['NAME'] == hduRED[0].header['NAME']:
        sami_name = hduBLUE[0].header['NAME']  # 'NAME' is the unique SAMI ID 
    else: 
        hdf.close()
        raise SystemExit("The two cube files are not matched according to "+
                         "their header-listed names. Please review the files "+
                         "and the validity of their headers. ")
    hduBLUE.close()
    hduRED.close()

    # What sort of observation this is: Target or Calibrator? Make group. 
    ### For now adding dev argument 'obstype' instead of cat cross-ID. 
    g_obstype = g_version.require_group(obstype)
    g_target = g_obstype.require_group(sami_name)

    # Check if data already exist in this version for this target. 
    if (g_target.keys() != []) and (not rss_only):
        raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                         "already contains data for the nominated SAMI "+
                         "target ("+sami_name+"). Please check archive "+
                         "and cubes. ")
    
    # Data import: begin big colour loop.
    colour = ['B', 'R']
    hdulist = [blue_cube, red_cube]
    
    for i in [0,1]:
        HDU = pf.open(hdulist[i])
        if verbose: 
            print
            print(HDU.info())

        # Identify RSS parents, locate in filesystem. 
        if (ingest_rss) or (rss_only):

            # First check if there are any already in here. 
            if colour[i]+'_RSS_data_1' in g_target.keys(): 
                raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                        "already contains RSS strips for the nominated SAMI "+
                        "target ("+sami_name+"). Please check archive "+
                        "and cubes. ")

            rss_list = locate_rss(hdulist[i], dataroot, verbose=verbose)
            n_rss = len(rss_list)
            
            if (n_rss != 0) and (verbose):
                print("")
                print("Located "+str(n_rss)+" RSS files for "+
                      "target "+sami_name+".")
                print("")

            else:
                hdf.close()
                raise SystemExit("No RSS files found inside the specified "+ 
                                 "root directory ("+dataroot+"). Please check "+
                                 "your 'dataroot' argument")
    
        # IMPORT CUBE
        # -----------
        """ 
        PROBLEM: attributes cannot have comments. Can think of two ways to 
         deal with this: 
         (a) store comments as string arrays, 
         (b) store a dictionary of header key meanings. 
        """ 

        # Cube: data: data -- require group (need to specify shape)
        cube_data = g_target.require_dataset(colour[i]+"_cube_data", 
                                             np.shape(HDU[0].data), 'f8', 
                                             exact=False,
                                             data=HDU[0].data, 
                                             chunks=True, compression='gzip')

        # Cube: data: header -- store all hdr fields as h5 attributes
        for n_hdr in range(len(HDU[0].header)):
            cube_data.attrs[HDU[0].header.keys()[n_hdr]] = \
                                HDU[0].header.values()[n_hdr]
        
        # Cube: variance: data
        cube_var = g_target.require_dataset(colour[i]+
                                              "_cube_variance", 
                                              np.shape(HDU[1].data), 
                                              'f8', data=HDU[1].data, 
                                              chunks=True, compression='gzip')
        # Cube: variance: header
        for n_hdr in range(len(HDU[1].header)):
            cube_var.attrs[HDU[1].header.keys()[n_hdr]] = \
                                    HDU[1].header.values()[n_hdr]
        
        # Cube: weight: data
        cube_wht = g_target.require_dataset(colour[i]+
                                              "_cube_weight", 
                                              np.shape(HDU[2].data), 
                                              'f8', data=HDU[2].data, 
                                              chunks=True, compression='gzip')
        # Cube: weight: header
        for n_hdr in range(len(HDU[2].header)):
            cube_wht.attrs[HDU[2].header.keys()[n_hdr]] = \
                                    HDU[2].header.values()[n_hdr]
                
        if ingest_rss or rss_only:
            for rss_loop in range(n_rss):
                rss_index = str(rss_loop+1)
                myIFU = sami.utils.IFU(rss_list[rss_loop], 
                                       sami_name, flag_name=True)
                # RSS: data: data
                rss_data = g_target.require_dataset(colour[i]+
                            "_RSS_data_"+rss_index, np.shape(myIFU.data), 'f8', 
                            data=myIFU.data, chunks=True, compression='gzip')
                
                # RSS: data: header
                for n_hdr in range(len(myIFU.primary_header)):
                    rss_data.attrs[myIFU.primary_header.keys()[n_hdr]] = \
                                        myIFU.primary_header.values()[n_hdr]

                # RSS: variance: data
                rss_var = g_target.require_dataset(colour[i]+
                                                "_RSS_variance_"+rss_index, 
                                                np.shape(myIFU.var), 'f8', 
                                                data=myIFU.var, chunks=True, 
                                                compression='gzip')

                # RSS: fibre table: data
                """ Needs to be imported as a compound dataset. """

                # RSS: fibre table: header
                """ Cannot be inserted before the CD has been sorted. """
                
        HDU.close()

    # Close h5file, exit successfully
    hdf.close()

    if verbose: 
        print
        print("Exiting successfully")
    


def import_cube_old(blue_cube, red_cube, h5file, 
                duplicate=False, overwrite=False, digest_rss=True, 
                rss_only=False, dataroot='./', 
                safe_mode=True, verbose=False):
    """ Original version, now superseded by immport_cube (in progress). 

    Import a SAMI datacube to the HDF5 data archive. 
 
    blue_cube:   BLUE datacube to be added to the SAMI DB. 
    red_cube:    RED datacube to be added to the SAMI DB. 
    h5file:      SAMI-formatted h5 file into which DB is written. 

    duplicate:   Duplicate any existing data, i.e. add a version. 
    overwrite:   Overwrite any existing data. 

    digest_rss:  Locate RSS parents, import strips related to input cubes.  
    rss_only:    Only ingest RSS strips, not cubes. 
    dataroot:    Data directory to be recursively searched by digest_rss. 
    safe_mode:   Gets rid of any unlinked groups/datasets. Slows things down. 
    verbose:     Toggle diagnostic and declarative verbosity. 
    
    This is the main 'add data' function for the SAMI DB. It ingests a set of
    blue and red SAMI datacubes and then locates and ingests their progenitor 
    row-stacked spectrum (RSS) files. RSS files are cut into 'strips' of 
    information that pertains only to the cubes beig imported (they contain 
    information on thirteen IFUs) before being imported. All other metadata 
    will be imported (in bulk) later using tailored codes. There is no need to 
    store SDSS cutouts, as those can be generated on-the-fly. 
    
    The main data archive will not include basic calibrations. These will live 
    in their own filesystem, the data-flow endpoint, as organised by the DRWG. 
    There is a "Calibrators" folder in the archive where we can store secondary
    stars (and aything else down the line). 
    """

    # First check if both the overwrite and duplicate flags are up. 
    if overwrite and duplicate:
        raise SystemExit("Both the 'duplicate' and 'overwrite' flags are up. "+
                         "Please choose which mode to follow. ")
    
    # Check if the nominated h5 file exists; prompt for creation if not, exit. 
    if not os.path.isfile(h5file):
        raise SystemExit("Cannot find the nominated HDF5 file ('"+h5file+"'). "+
                         "Please create a file using the 'create' function")
        
    # If file does exist, open (copy in safe_mode) and allow write privileges. 
    if safe_mode:
        import datetime
        import shutil
        if verbose: print("Safe Mode: beginning file copy.")
        date = datetime.datetime.now()
        datestamp = str(date.year).zfill(2)+str(date.month).zfill(2)+\
                    str(date.day).zfill(2)+'_'+\
                    str(date.hour).zfill(2)+str(date.minute).zfill(2)+\
                    str(date.second).zfill(2)
        bkp_file = h5file[:-3]+"_"+datestamp+".h5"
        shutil.copyfile(h5file,bkp_file)
        if verbose: print("Safe Mode: file successfully copied to '"+
                          bkp_file+"'.")

    hdf = h5.File(h5file, 'r+')

        
    # Check that the SAMI filesystem has been set up in this file. 
    if ("SAMI" not in hdf.keys()) or ("Targets" not in hdf['SAMI'].keys())\
       or ("Calibrators" not in hdf['SAMI'].keys()):
        hdf.close()
        raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                         "is not properly formatted. Please initialise the "+
                         "filesystem using 'SAMI_DB.format'")
    
    # If the "SAMI" group exists, let's make some data blocks
        
    # Read the header, find out the target name. 
    hduBLUE = pf.open(blue_cube)
    hduRED  = pf.open(red_cube)
    
    # Check that the nominated blue/red cubes are indeed of same target. 
    if hduBLUE[0].header['NAME'] == hduRED[0].header['NAME']:
        sami_name = hduBLUE[0].header['NAME']  # 'NAME' is the unique SAMI ID 
    else: 
        hdf.close()
        raise SystemExit("The two cube files are not matched according to "+
                         "their header-listed names. Please review the files "+
                         "and the validity of their headers. ")
    hduBLUE.close()
    hduRED.close()
    
    # Check if the file already contains a cube block for this sami ID
    """
    NOTE: 
    There is a subtlety here. While the contents of a database files should not 
    be manipulated post-creation, fine-tuning and other operations might be, on
    occassion, required. A manually deleted dataset will leave a trace that is 
    not picked up by h5dump or HDFView, but still listed as a key in the file 
    attributes list. 
    
    h5repack will copy over the file and not carry through any intentionally 
    broken links ('deleted' datasets). This is the preferred behaviour now. That
    is, every manual deletion should be followed by a file copy to lose those
    broken links. 
    
    Perhaps run a health check on the data archive after every import: attempt 
    a basic diagnostic (numeric or other) on every group in there, and flag bad
    groups or datasets for deletion automatically. 
    """
    
    if ("SAMI/Targets/"+sami_name in hdf) and \
            (not duplicate) and (not overwrite) and \
            (not rss_only):
        hdf.close()
        raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                         "already contains\n a target block called '" + 
                         sami_name+"'. If you wish to overwrite it, " + 
                         "please\n raise the 'overwrite' flag. If you wish " + 
                         "to import a duplicate\n observation, please " + 
                         "raise the 'duplicate' flag.")

    """
    NOTE: Need to check if OW or 2x =True but there are no data. The problem 
    here is that once the target group in question has been created, the code
    will always be tricked into thinking that this is the case -- the newly 
    created target group will be empty... This is tough to work around, as the
    group has to be defined within specific loops. Original code follows. 

    I can ignore the duplicate bit: the code will make a new version, but NOT 
    if this is the very first version, as the 2x loop is nested within the if 
    not v=01 loop. The OW loop does get confudsed though. 

    This snippet of code wasn't used, but might be handy someplace else: 
    # Perform a health check on the target block (delete phantom groups)
    targ_block = "SAMI/Targets/"+sami_name
    if "SAMI/Targets/"+sami_name in hdf:
        # OK, the node exists, now look for the version
        versions = hdf["SAMI/Targets/"+sami_name].keys()
        v_num = [int(versions[-1][-2:])]  # version numbers as integers
        v_latest = max(v_num)             # latest version as largest v_num
        v_str = '/v'+str(v_latest).zfill(2) # version number as string
        # Look in the latest version:
        if len(hdf["SAMI/Targets/"+sami_name+v_str].keys() == 0):
            del hdf["SAMI/Targets/"+sami_name]
    """

    # Create or define target block (h5 group). 
    # If this target is new to the archive, create v01.  
    if ("/SAMI/Targets/"+sami_name not in hdf) or rss_only:
        # OW should not be up. If it is, exit.  
        if overwrite: 
            hdf.close()
            raise SystemExit("The 'overwrite' flag is up, but "+
                             "the nominated h5 file ('"+h5file+"') contains no"+
                             " data related to the target selected for "+
                             "ingestion ("+sami_name+"). Exiting as a "+
                             "precaution.")
        # If OW is down, create the v01 folder. 
        else: 
            targ_group = hdf.require_group("SAMI/Targets/"+sami_name+"/v01")

    # If sami_name already on archive, determine latest version held. 
    else: 
        versions = hdf["SAMI/Targets/"+sami_name].keys()
        v_num = [int(versions[-1][-2:])]    # version numbers as integers
        v_latest = max(v_num)               # latest version as largest v_num
        v_str = '/v'+str(v_latest).zfill(2) # version number as string

        if overwrite:
            # Check if an empty group exists (a 'phantom')
            if len(hdf["SAMI/Targets/"+sami_name+v_str].keys()) == 0:
                hdf.close()
                raise SystemExit("The 'overwrite' flag is up, but "+
                             "the nominated h5 file ('"+h5file+"') contains no"+
                             " data related to the target selected for "+
                             "ingestion ("+sami_name+"). Exiting as a "+
                             "precaution.")
            else: 
                targ_group = hdf.require_group("SAMI/Targets/"+sami_name+v_str)

        if duplicate:
            # Create a new version group with appropriate index (latest + 1)
            v_new = v_latest + 1
            v_str = '/v'+str(v_new).zfill(2)
            targ_group = hdf.require_group("SAMI/Targets/"+sami_name+v_str)
    
    # Now that block has been created, write some datasets. 
    
    # Begin big colour loop.
    colour = ['Blue', 'Red']
    hdulist = [blue_cube, red_cube]
    
    for i in [0,1]:
        HDU = pf.open(hdulist[i])
        
        # If digest_rss is up, locate the RSS files using the locate function.  
        if (digest_rss) or (rss_only):
            rss_list = locate_rss(hdulist[i], dataroot, verbose=verbose)
            n_rss = len(rss_list)
            
            if n_rss == 0:
                hdf.close()
                raise SystemExit("No RSS files found inside the "+ 
                                 "specified root directory. Please check "+
                                 "your 'dataroot' input variable")
            else:
            #if n_rss < 7:
                if verbose: 
                    print("")
                    print("Located "+str(n_rss)+" RSS files for "+
                          "target "+sami_name+".")
                    print("")
                    
        """
        Comment on data types: integers are 'i2', 'i4', float are 'f', 
        strings are 'S10' etc. 
        """

        # (A) Setup/Duplication loop: Requires new datasets in new group.
        # ---------------------------------------------------------------
        if not overwrite:
            
            # Cube: data: data -- require group (need to specify shape)
            cube_data = targ_group.require_dataset(colour[i]+"_cube_data", 
                                                np.shape(HDU[0].data), 'f8', 
                                                exact=False,
                                                data=HDU[0].data, 
                                                chunks=True, compression='gzip')
            
            # Cube: data: header -- store all hdr fields as h5 attributes
            for n_hdr in range(len(HDU[0].header)):
                cube_data.attrs[HDU[0].header.keys()[n_hdr]] = \
                                    HDU[0].header.values()[n_hdr]
                
            # Cube: variance: data
            cube_var = targ_group.require_dataset(colour[i]+
                                              "_cube_variance", 
                                              np.shape(HDU[1].data), 
                                              'f8', data=HDU[1].data, 
                                              chunks=True, compression='gzip')
            # Cube: variance: header
            for n_hdr in range(len(HDU[1].header)):
                cube_var.attrs[HDU[1].header.keys()[n_hdr]] = \
                                    HDU[1].header.values()[n_hdr]
                
            # Cube: weight: data
            cube_wht = targ_group.require_dataset(colour[i]+
                                              "_cube_weight", 
                                              np.shape(HDU[2].data), 
                                              'f8', data=HDU[2].data, 
                                              chunks=True, compression='gzip')
            # Cube: weight: header
            for n_hdr in range(len(HDU[2].header)):
                cube_wht.attrs[HDU[2].header.keys()[n_hdr]] = \
                                    HDU[2].header.values()[n_hdr]
                
                
            # Now import the (nominally seven) parent RSS files. 
            if digest_rss or rss_only:
                
                # Loop over RSS files, use sami.utils.IFU class
                for rss_loop in range(n_rss):
                    rss_index = str(rss_loop+1)
                    myIFU = sami.utils.IFU(rss_list[rss_loop], 
                                           sami_name, flag_name=True)
                    # RSS: data: data
                    rss_data = targ_group.require_dataset(colour[i]+
                                                "_RSS_data_"+rss_index, 
                                                np.shape(myIFU.data), 'f8', 
                                                data=myIFU.data, chunks=True, 
                                                compression='gzip')
                    # RSS: data: header
                    """ 
                    -- RSS header input not possible yet, primary header 
                    not accessible through IFU object; need to make custom 
                    sami.utils that makes all extensions and headers 
                    available in their entirety. 
                    """
                    
                    # RSS: variance: data
                    rss_var = targ_group.require_dataset(colour[i]+
                                                "_RSS_variance_"+rss_index, 
                                                np.shape(myIFU.var), 'f8', 
                                                data=myIFU.var, chunks=True, 
                                                compression='gzip')
                    # RSS: fibre table: data
                    """
                    -- PROBLEM: the fibtab is a list, does not havea fixed 
                    data type... How do I input this here? I could turn 
                    everything into strings, but then when they are 
                    recovered they will need to be converted back to the 
                    original type. Tricky. 
                    
                    rss_fibtab = targ_group.require_dataset(colour[i]+
                    "_RSS_fibre_table", np.shape(myIFU.fibtab), 'f8', 
                    data=myIFU.fibtab, chunks=True, compression='gzip')
                    """
                    
        # (B) Overwriting loop: edits existing datasets. 
        # ----------------------------------------------
        else:
        # Edit cubes and their attributes
            cube_data = targ_group[colour[i]+'_cube_data']
            cube_data[:] = HDU[0].data
            for n_hdr in range(len(HDU[0].header)):
                cube_data.attrs[HDU[0].header.keys()[n_hdr]] = \
                                    HDU[0].header.values()[n_hdr]

            cube_var = targ_group[colour[i]+'_cube_variance']
            cube_var[:] = HDU[1].data
            for n_hdr in range(len(HDU[1].header)):
                cube_var.attrs[HDU[1].header.keys()[n_hdr]] = \
                                    HDU[1].header.values()[n_hdr]

            cube_wht = targ_group[colour[i]+'_cube_weight']
            cube_wht[:] = HDU[2].data
            for n_hdr in range(len(HDU[2].header)):
                cube_wht.attrs[HDU[2].header.keys()[n_hdr]] = \
                                    HDU[2].header.values()[n_hdr]
            
            # Edit RSS files:
            if digest_rss:
                
                # Overwrite RSS if they exist
                if colour[i]+'RSS_data' in targ_group:
                    for rss_loop in range(n_rss):
                        myIFU = sami.utils.IFU(rss_list[rss_loop], 
                                               sami_name, flag_name=True)
                        rss_data = targ_group[colour[i]+"_RSS_data_"+rss_index]
                        rss_data[:] = myIFU.data
                        rss_var = targ_group[colour[i]+
                                             "_RSS_variance_"+rss_index]
                        rss_var[:] = myIFU.var
                        
                # Create them if they do not exist
                else:
                    for rss_loop in range(n_rss):
                        myIFU = sami.utils.IFU(rss_list[rss_loop], 
                                               sami_name, flag_name=True)
                        # RSS: data: data
                        rss_data = targ_group.require_dataset(colour[i]+
                                                "_RSS_data", 
                                                np.shape(myIFU.data), 'f8', 
                                                data=myIFU.data, chunks=True, 
                                                compression='gzip')
                        # RSS: variance: data
                        rss_var = targ_group.require_dataset(colour[i]+
                                                "_RSS_variance", 
                                                np.shape(myIFU.var), 'f8', 
                                                data=myIFU.var, chunks=True, 
                                                compression='gzip')

        # Close the opened cube fits file
        HDU.close()
            
            
    # Close the HDF5 file and end process. 
    hdf.close()
    if verbose:
        if digest_rss: extra_str = " and its "+str(n_rss)+" RSS parents."
        else: extra_str = "."
        print("")
        print(">< Process complete, ingested cube for SAMI-"+
              sami_name+extra_str)
          
    
def import_many(tablein, h5file, duplicate=False, overwrite=False, 
                digest_rss=True, rss_only=False, dataroot='./', 
                verbose=True, timing=True):
    """ Wrapper for import_cube to digest any number of cubes
    
    'tablein' needs to list the blue and red cubes in each space-separated row. 
    All other arguments refer to the 'import_cube' kwargs. 
    """

    if timing: 
        import time
        timer_zero = time.time()

    tabdata = ascii.read(tablein, data_start=0, names=['blue', 'red'])
    for loop in range(len(tabdata)):
        if timing: timer_start = time.time()
        if verbose: print("Processing "+
                          os.path.basename(tabdata['blue'][loop])+", "+
                          os.path.basename(tabdata['red'][loop]))
        import_cube(tabdata['blue'][loop], tabdata['red'][loop], 
                    h5file, overwrite=overwrite, duplicate=duplicate, 
                    digest_rss=digest_rss, rss_only=rss_only, 
                    dataroot=dataroot, verbose=verbose)
        if timing: 
            timer_end = time.time()
            print(loop,timer_end-timer_start)
            time_elapsed = timer_end-timer_zero
            print(time_elapsed/(loop+1))


def import_table(h5file, tabin, cdf='', h5dir='/', verbose=False):
    """ Import a SAMI target table to an h5 archive. 

    1) Read in the table (all numeric), given preset columns (supply). 
    2) Attach as h5 table in the root directory. 

    Note that 'name' is the IAU name and the SAMI identifier is 'CATID'. 
    """
    import tables

    # Define column definitions for a standard SAMI table. 
    if cdf != '':
        raise SystemExit("Sorry, custom column definition files not yet "+
                         " supported. Exiting. ")
    else:
        coldef = ['name', 'RA', 'Dec', 
                  'r_petro', 'r_auto', 'z_tonry', 'z_spec', 
                  'M_r', 'r_e', '<mu(re)>', 'mu(re)', 'mu(2re)', 
                  'ellip', 'PA', 'M*', 'g-i', 'A_g', 
                  'CATID', 'SURV_SAMI', 'PRI_SAMI', 'BAD_CLASS']
    
    # Read in the ascii table. 
    table = ascii.read(tabin, names=coldef, data_start=0)
    if verbose:
        print("\nRead in table '"+tabin+"', beginning with:\n")
        print(table['name'][:15])

    # Define a user record as a PyTables class. 
    class sami_master(tables.IsDescription):
        name      = tables.StringCol(19,pos=1)
        RA        = tables.FloatCol(pos=2)
        Dec       = tables.FloatCol(pos=3)
        r_petro   = tables.Float32Col(pos=4)
        r_auto    = tables.Float32Col(pos=5)
        z_tonry   = tables.Float32Col(pos=6)
        z_spec    = tables.Float32Col(pos=7)
        M_r       = tables.Float32Col(pos=8)
        r_e       = tables.Float32Col(pos=9)
        med_mu_re = tables.Float32Col(pos=10)
        mu_re     = tables.Float32Col(pos=11)
        mu_2re    = tables.Float32Col(pos=12)
        ellip     = tables.Float32Col(pos=13)
        PA        = tables.Float32Col(pos=14)
        Mstar     = tables.Float32Col(pos=15)
        g_minus_i = tables.Float32Col(pos=16)
        A_g       = tables.Float32Col(pos=17)
        CATID     = tables.Int64Col(pos=0)
        SURV_SAMI = tables.UInt16Col(pos=18)
        PRI_SAMI  = tables.UInt16Col(pos=19)
        BAD_CLASS = tables.UInt16Col(pos=20)

    # Check if the nominated file exists
    if not os.path.isfile(h5file):
        print("The nominated HDF5 file ('"+h5file+"') does not exist. "+
              "Creating new file. ")
        
    # Open the nominated h5 file. 
    h5file = tables.openFile(h5file, mode = "a")
    
    # Check that the chosen directory exists. 
    ### FILL IN! 
    
    # Create a target table in the root directory of the h5file. 
    master = h5file.createTable(h5dir, 'SAMI_MASTER', sami_master)

    # Attributes can be tacked to the table here. 

    # Populate this master h5 table with 'tabin'. 
    galaxy = master.row
    for i in range(len(table['name'])):
        galaxy['name']      = table['name'][i]
        galaxy['RA']        = table['RA'][i]
        galaxy['Dec']       = table['Dec'][i]
        galaxy['r_petro']   = table['r_petro'][i]
        galaxy['r_auto']    = table['r_auto'][i]
        galaxy['z_tonry']   = table['z_tonry'][i]
        galaxy['z_spec']    = table['z_spec'][i]
        galaxy['M_r']       = table['M_r'][i]
        galaxy['r_e']       = table['r_e'][i]
        galaxy['med_mu_re'] = table['<mu(re)>'][i]
        galaxy['mu_re']     = table['mu(re)'][i]
        galaxy['mu_2re']    = table['mu(2re)'][i]
        galaxy['ellip']     = table['ellip'][i]
        galaxy['PA']        = table['PA'][i]
        galaxy['Mstar']     = table['M*'][i]
        galaxy['g_minus_i'] = table['g-i'][i]
        galaxy['A_g']       = table['A_g'][i]
        galaxy['CATID']     = table['CATID'][i]
        galaxy['SURV_SAMI'] = table['SURV_SAMI'][i]
        galaxy['PRI_SAMI']  = table['PRI_SAMI'][i]
        galaxy['BAD_CLASS'] = table['BAD_CLASS'][i]
        galaxy.append()

    master.flush()
    h5file.close()

    
def locate_rss(cubein, dataroot='/Users/iraklis/Data/SAMI/SAMI_13Apr', 
               verbose=True):
    """ Read in a SAMI cube, identify and locate its parent RSS files """ 
    
    import fnmatch
    
    hdulist = pf.open(cubein)
    hdr0 = hdulist[0].header
    hdulist.close()

    # Cycle through header and figure out how many keys contain 'RSS_FILE'. 
    n_rss = 0
    for i in range(len(hdr0)):
        if hdr0.keys()[i][0:8] == 'RSS_FILE': n_rss = n_rss+1

    if verbose: print('Identified '+str(n_rss)+' RSS files.')

    # Compile an array of RSS filenames
    rss_files = []
    for i in range(n_rss):
        makestr = 'RSS_FILE '+str(i+1)
        rss_files.append(hdr0[makestr])

    # Search for the RSS files on this machine. 
    rss_path = []
    filename = rss_files[0]
    if verbose: print("Locating the RSS files:")
    for root, dirnames, filenames in os.walk(dataroot):
        for name in filenames:
            for i in range(n_rss):
                if name == rss_files[i]:
                    rss_path.append(os.path.abspath(os.path.join(root,name)))
                    if verbose: print('   '+rss_path[i])

    # Check 
    if len(rss_files) == n_rss:
        if verbose: print('Located all RSS files.')
    else: 
        raise SystemExit("Sorry, I didn't find all "+str(n_rss)+" RSS files.")

    return rss_path
    

def make_list(basedir='', tableout='SAMI_input.lis', 
              overwrite=False, append=False, debug=True):
    """ Make an .import_many digestion list based on contents of basedir. """
    
    """ 
    This code relies on the existence of the standard ccd_1 and ccd_2 subdirs, 
    as deleivered by the good people at the Data Reduction Working Group. This 
    should not change (ever!), but major revision will be required if it does. 

    The code also relies on the naming convention for SAMI cubes as: 
      <SAMI_ID>_<ccd>_<N(dithers)>.fits
    """

    # Check if both the overwrite and append flags are up, exit of so.
    if (overwrite) and (append):
        raise SystemExit("Both the 'overwrite' and 'append' flags have " +
                         "been raised. That won't work, you have to choose " +
                         "just one! Exiting. ")

    # Check if basedir has been defined. If not, then it is the WD. 
    if basedir == '': basedir = os.getcwd()
    
    # Check for the ccd_1 and ccd_2 directories. 
    base_contents = os.listdir(basedir)
    
    if ('ccd_1' not in base_contents) or ('ccd_2' not in base_contents):
        raise SystemExit("The chosen basedir does not contain 'ccd_1' " +
                         "or/and 'ccd_2' directories. Exiting. ")
        
    # If those exist figure out their contents, isolate .fits files, and 
    #  then extract their SAMI IDs. Check that ccd_1 and ccd_2 contents match.
    contents_ccd1 = np.array(os.listdir(basedir+'/ccd_1/'))
    contents_ccd2 = np.array(os.listdir(basedir+'/ccd_2/'))

    # A little process that can recursively return filename extensions
    def isolator(str_in): 
        return os.path.splitext(str_in)[1]

    ext1 = np.array([isolator(str_in) for str_in in contents_ccd1])
    ext2 = np.array([isolator(str_in) for str_in in contents_ccd2])

    # Keep only .fits files
    contents_ccd1 = contents_ccd1[np.where(ext1 == '.fits')]
    contents_ccd2 = contents_ccd2[np.where(ext2 == '.fits')]

    # A little process to strip sami name as text preceding underscore: 
    def strip_name(str_in):
        uscore = str_in.index('_')
        return str_in[:uscore]

    sami_names1 = [strip_name(str_in) for str_in in contents_ccd1]
    sami_names2 = [strip_name(str_in) for str_in in contents_ccd2]

    if sami_names1 != sami_names2:
        raise SystemExit("The 'ccd_1' and 'ccd_2' lists are mismatched. "+
                         "Exiting. ")
        
    else: print("\nFound "+str(len(sami_names1))+
                " matched blue/red SAMI RSS files in 'basedir'.")
    
    # Now write the list file for digestion by import_many
    # First check if file exists and if overwrite flag is up: 
    if (os.path.isfile(tableout)) and (not overwrite) and (not append):
        raise SystemExit("The nominated list file already exists. Please "+
                         "use a different filename, or raise the "+
                         "'overwrite' flag.")
        
    if (os.path.isfile(tableout)) and (overwrite):
        print("Overwriting file '"+tableout+"'.")

    # Then create the file buffer, decide whether to overwrite or append: 
    if not append: f = open(tableout, 'w')
    if append: f = open(tableout, 'a')

    # Now write those tables. 
    for i in range(len(sami_names1)):
        f.write(basedir+'ccd_1/'+contents_ccd1[i]+' '+
                basedir+'ccd_2/'+contents_ccd2[i]+'\n')
        
    f.close()


""" THE END """
