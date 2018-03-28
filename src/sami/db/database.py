"""
SAMI_DB.py

6/2/13, Iraklis Konstantopoulos

Data archiving tools for the SAMI survey in the HDF5 filesystem. 

NOTES: 

  1/5/13: Naming convention for functions in the DB module: 
          -- 'import' to write something into the h5 archive, 
          -- 'export' to write a file (fits, jpg, etc). 
 
          Importing files: I need to get RSS and cubes together and then import
          The cubes know where they came from (RSS_X header fields), so those  
          files can be located once the cube has been imported. 

          Should the RSS files live in the target block? Probably. Import  
          single-IFU information block with the sami.utils.IFU class, store 
          appropriate parts of the datafile as a dataset in the target block. 
          
          The export_cube block can envoke a fetch_SDSS function and get a jpg. 
          
          Value-added information, such as line/flux maps can be imported 
          separately at any point in time, as long as they are tagged with the 
          target ID (GAMA name or unique identifier for cluster fields). 
          
          
  9/5/13: The fibre table of an RSS file is tricky to import. It does not have 
          a fixed-datatype, rather it is a sort of python list, whose contents 
          are mostly floats but some are strings. These will all have to be 
          imported as strings in the archive, and exported in their proper 
          format. It should be safe to let the code figure out what the type is:
          try to make a float out of it, keep it if it works. 

 10/5/13: Regarding the removal of objects from an h5 file. This isn't something
          that HDF5 is happy to do in a simple way. The officially sanctioned 
          wayis to unlink (delete) an object and then copy the file to a new 
          one. The unlinked object is not transferred over. See Section 5.5.2 
          of the HDF documentation: 
         
              www.hdfgroup.org/HDF5/doc/UG/UG_frame10Datasets.html
             
          Perhaps the best tactic is to unlink datasets when needed and then 
          copy the file every few weeks to save any unlinked space. 
          
          Therefore, there should be an option to overwrite datasets, rather 
          than 'requiring' by default. The h5py command to delete an object is 
          the 'del' command, as in: 
          
              >>> del f['MyDataset']

          Changing the contents of a dataset is a good alternative, as long as 
          the new dataset has the exact same dimensions. Make sure a good QC 
          check is in place. One issue is that require_dataset will not tell me 
          whether or not a set exists, if it was edited, or anything else. The 
          checks of whether a set exists are therefore still necessary. 

 14/5/13: The import_cube (and import_many) function has been fitted with an 
          'rss_only' flag, which ignores cube digestion and goes straight to 
          RSS files. Should write a script to prepare lists for import_many. 

 13/6/13: I haven't systematically been noting progress on this file, but a 
          bunch of backups do exist. Fixed a couple of bugs related to exception
          catching. 

          ** Remove 'calibrations' folder from h5file formatting. 

 17/6/13: Added some PyTbles functionality for ingesting the SAMI master tab.  

 18/6/13: Set up a couple of table query functions. 
 
 29/7/13: Need to introduce a data type header scan before importing a cube, 
          simply for the code to know whether to import into the 'Targets' or 
          'Calibrations' directories. 

          Also need to introduce a 'monochrome' switch to import_cube that will
          import a single cube, rather than a pair. 

 30/7/13: These phantom volumes are causing much trouble. The code should allow
          for human error and currently it does not becuase of these traces of
          deleted volumes. I am thinking of introducing a 'safe mode' that will
          copy the file to a new one, thus getting rid of any unlinked volumes. 
          Test how much time this would take. 

          Actually, the 'del' command does get rid of the trace, by the looks of
          it. No phantoms appear in the listed keys. The following 'health 
          check' shuold therefore be performed on opening an h5 file: 
          (1) detect phantoms as version keys that contain no datasets; 
          (2) delete all phantoms; 
          (3) proceed as normal. 

TABLE OF CONTENTS (* to-be, **: under construction): 

.create         Creates an HDF5 file, not SAMI-specific. 
.format         Sets up the SAMI root and base (target-based) filestructure. 
.import_cube    Digests a SAMI datacube. Envokes 'import_rss' and 'fetch_SDSS' 
                from the SAMI_sdss module. 
.locate_rss     Queries a cube header, identifies parent RSS files, locates  
                them on the machine running the code. Returns filename listing. 
.list_keys      List all the 'keys' (i.e. blocks and datasets) in an h5 file. 
.make_list      Make a list for import_many from the contents of a directory. 
.import_table   Imports a SAMI target table. 
.query_master** Query a SAMI target table. 
.query_multiple** Perform any number of queries on any number of SAMI tables. 
.search**       The query prototype.
.fetch_cube     Extract a cube or set of cubes. 
.export         Export any amount of data products of various types. 
.remove_block*  Unlink an entire target block (or version). 
.remove_rss*    Unlink a set of RSS 'strips'. 
.remove_cube*   Unlink a cube or set of cubes. 
.remove_misc*   Unlink value-added data. 
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import h5py as h5
import astropy.io.fits as pf
import astropy.io.ascii as ascii
import os
import sys

import sami

def create(h5file, overwrite=False, verbose=True):
    """ Create a generic HDF5 file. Will crash of type(h5file) is not str. """

    # Check that 'h5file' has a .h5 extension
    if not h5file[-3:] == '.h5': h5file = h5file + '.h5'

    # Initiate a little token used to control screen output
    screen_output = 0
    
    # Check if file already exists, overwite if overwrite==True
    if os.path.isfile(h5file):
        screen_output += 1
        if not overwrite:
            raise SystemExit("The nominated h5 file ('"+h5file
                             +"') already exists. Please raise the overwrite "
                             +"flag if you wish to prodceed. Exiting.")
    
    # Create an h5 file
    if (not os.path.isfile(h5file)) or (overwrite==True):
        f = h5.File(h5file, 'w')
        f.close()
    
    # Screen output
    if verbose:
        if screen_output > 0: prefix = 'Re-'
        else: prefix = ''
        print(prefix+"Initialised file '"+h5file+"'.")
    

def format(h5file):
    """ Set up an h5 file with the SAMI database hierarchy """
    
    import sys
    
    # Ad an extension to the file if it isn't already '.h5'
    if not h5file[-3:] == '.h5': h5file = h5file + '.h5'
    
    # Open the input file for reading and writing
    f = h5.File(h5file, 'r+')
    
    # Check if a SAMI root directory exists; create if not (require, not create)
    root = f.require_group("SAMI")
    
    # ...create the observations directory (science targets only)
    targ = root.require_group("Targets")
    
    # ...create the calibrations directory (stars etc.)
    calib = root.require_group("Calibrators")
    
    f.close()


def import_cube(blue_cube, red_cube, h5file, 
                duplicate=False, overwrite=False, digest_rss=True, 
                rss_only=False, colour='', dataroot='./', 
                safe_mode=True, verbose=False):
    """ Import a SAMI datacube to the HDF5 data archive. """
    
    """ 
    red_cube:    BLUE datacube to be added to the SAMI DB. 
    blue_cube:   RED datacube to be added to the SAMI DB. 
    h5file:      SAMI-formatted h5 file into which DB is written. 

    duplicate:   Duplicate any existing data, i.e. add a version. 
    overwrite:   Overwrite any existing data. 

    digest_rss:  Locate RSS parents, import strips related to input cubes.  
    rss_only:    Only ingest RSS strips, not cubes. 
    colour:      Enables 'monochromatic' input. Set to 'blue' or 'red'. 
    dataroot:    Data directory to be recursively searched by digest_rss. 
    safe_mode:   Gets rid of any unlinked groups/datasets. Slows things down. 
    verbose:     Toggle diagnostic and declarative verbosity. 
    
    This is the main 'add data' function for the SAMI DB. It ingests a set of
    blue and red SAMI datacubes and then optionally envokes a series of 
    functions to search for their progenitor row-stacked spectrum (RSS) files. 
    RSS files are cut into 'strips' of information that pertains only to the 
    cubes beig imported (they contain information on thirteen IFUs) beofre 
    being imported. All other metadata will be imported (in bulk) later using
    tailored codes. There is no need to store SDSS cutouts, as those can be 
    generated on-the-fly. 
    
    While two cubes should always be delivered, perhaps the functionality should
    be available for single-colour importing --> Add a 'None' input option for 
    the 'blue_cube' and 'red_cube' arguments and skip the relevant parts of the 
    code. Question has been posed to DRWG. 
    
    The main data archive will not include basic calibrations. These will live 
    in their own filesystem, the data-flow endpoint, as organised by the DRWG. 
    The "Calibrators" folder has, however, been maintained so as to store PSF
    and spectrophotometric standard stars. 

    This code has been tested on SAMI data from the April 2013 observing run. 
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
    """ 
    NOTE: There is a bit of a redundancy here. Shouldn't open and close a unit
          I will be opening and closing again further down. Keep open? 
    """
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
    created target gropu will be empty... This is tough to work around, as the
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
            if n_rss < 7:
                print("")
                print("WARNING: located only "+str(n_rss)+" RSS files for "+
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
    """ Wrapper for import_cube to digest any number of cubes """
    
    """
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
    """ Import a SAMI target table to an h5 archive. """
    
    """
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


"""
def search_master(s):
    # A PyTables Row Iterator for a 21-column SAMI target cataogue.

    import tables

    counter = 0
    for tables.row in s:
        name, z = tables.row['CATID'], tables.row['z_spec']
        if verbose:
            print("  Found SAMI galaxy %s at redshift z=%g" % (name, z))
        counter += 1
        idlist.append(name)
        if returnID: f.write(str(name)+'\n')
    print("\n  Found "+str(counter)+" galaxies satisfying query:\n  "+query)
"""


def query_master(h5in, query, idfile='sami_query.lis', 
                verbose=True, returnID=True, overwrite=True):
    """ Read a SAMI master table and perform a query """

    """ 
    The query should be passed either as a string argument or as an ascii file 
    containing such a string. Adding '@list' functionality will facilitate 
    browser-generated queries farther down the track. 
    """

    import tables

    # Interpret the 'query' argument (look for a filename). 
    if os.path.isfile(query):
        fq = open(query, 'r')
        query = fq.readlines()[0]

    # Open-read h5file. 
    h5file = tables.openFile(h5in, mode = "r")

    # Optionally open an ascii file to write IDs returned by the query. 
    if returnID:
        # Check if the file exists, check overwrite flag:
        if os.path.isfile(idfile):
            if not overwrite:
                raise SystemExit("The nominated output file ('"+idfile+"') "+
                                 "already exists. Please raise the 'overwrite'"+
                                 " flag or enter a different filename. ")
        f = open(idfile, 'w')
        idlist = []

    # Identify the SAMI master table -- assumed to live in the root directory
    master = h5file.root.SAMI_MASTER

    # Define a Row Iterator for screen output. 
    def print_sami(s):
        counter = 0
        for tables.row in s:
            name, z = tables.row['CATID'], tables.row['z_spec']
            if verbose:
                print("  Found SAMI galaxy %s at redshift z=%g" % (name, z))
            counter += 1
            idlist.append(name)
            if returnID: f.write(str(name)+'\n')
        print("\n  Found "+str(counter)+" galaxies satisfying query:\n  "+query)
        
    print_sami(master.where(query))

    h5file.close()
    if returnID: f.close()
    

def query_multiple(h5in, qfile, verbose=True):
    """ Query multiple tables within an h5 archive and combine results """

    import tables

    """ 
    This is modelled after query_master. One main difference is the query 
    argument, which has to be a file in this case, rather than a string. 
    
    """
    # Check that the input files exist
    if (not os.path.isfile(h5in)) or (not os.path.isfile(qfile)):
        raise System.Exit("One of the nominated files does not exist. Exiting.")

    # Open and read the query file line-per-line (even=table,odd=query) 
    counter = 0
    tabs, queries = [], []
    with open(qfile) as f:
        for i, l in enumerate(f):
            # Read the file in odd even lines
            if i%2==0:
                tabs.append(l.strip())
            else:
                queries.append(l.strip())
            counter += 1

    # Check if the length of the two lists is equal
    if len(tabs) != len(queries):
        raise SystemExit("Table-query mismatch. Please input an equal "+
                         "number of 'table' and 'query' lines. ")
    else: 
        print("Read %g queries" % (counter/2))
        if verbose: 
            for i in range(counter/2):
                print("In table '"+tabs[i]+"' query: "+queries[i])
            print("")

    f.close()

    # Now run all queries: 
    # Open h5 file with PyTabs: 
    h5file = tables.openFile(h5in, mode = "r")

    # Cannot provide any table name. They have to be defined here.
    # Need to write an 'identify table' function. 
    def id_tab(s):
        if s == 'SAMI_MASTER': return h5file.root.SAMI_MASTER
        if s == 'SAMI_MASTER_2': return h5file.root.SAMI_MASTER_2

    # and run that function on all tabs: 
    h5tabs = []
    for i in range(counter/2):
        h5tabs.append(id_tab(tabs[i]))

    h5file.close()
    
    # OK, have the tables defined as variables, now need to query them. 
    # I will copy-paste the print_sami code here, but need to do better.
    
    # Define a Row Iterator for screen output. 
    idlist = []
    def print_sami(s):
        count = 0
        for tables.row in s:
            name, z = tables.row['CATID'], tables.row['z_spec']
            #if verbose:
            #    print("  Found SAMI galaxy %s at redshift z=%g" % (name, z))
            count += 1
            idlist.append(name)
        return idlist
        
    # Run each query: 
    all_lists = []
    for i in range(counter/2):
        print_sami(h5tabs[i].where(queries[i]))
        print("Query "+str(i+1)+": Found "+str(len(idlist))+
              " galaxies satisfying "+queries[i])
        all_lists.append(idlist)

    # And now intersect all idlists within all_lists container
    final_idlist = set(all_lists[0]).intersection(*all_lists)
    print(final_idlist)

def fetch_cube(name, h5file, colour='', outfile=''):
    """ A tool to fetch a datacube in the standard format. """

    """ 
    name       [str]  The name of the SAMI target required. 
    h5file     [str]  The SAMI archive file from which to export. 
    colour     [str]  Colour-specific export. Set to 'blue' or 'red'. 
    outfile    [str]  The name of the file to be output. Default: "col_name".
    """
    
    # Digest SAMI name, search within h5file, identify block, write .fits.
    
    hdf = h5.File(h5file)
    
    # Check that h5file is SAMI-formatted.
    if 'SAMI' not in hdf.keys():
        raise SystemExit('The nominated h5 file ('+h5file+') is not '+\
                         'SAMI-formatted')
    
    # Check that h5file contains a target block for name.
    if name not in hdf['SAMI/Targets'].keys():
        raise SystemExit('The nominated h5 file ('+h5file+') does not '+\
                         'contain a target block for SAMI '+name)
    
    # Checks done, extract some cubes. 
    
    # Determine latest version, and therefore target block path.
    versions = hdf["SAMI/Targets/"+name].keys()
    v_num = [int(versions[-1][-2:])]  # version numbers as integers
    v_latest = max(v_num)             # latest version as largest v_num
    
    v_str = '/v'+str(v_latest).zfill(2) # version number as string
    targ_group = hdf.require_group("SAMI/Targets/"+name+v_str)
    
    # Look for cubes:
    if ('Blue_cube_data' not in targ_group.keys()) or \
       ('Blue_cube_variance' not in targ_group.keys()) or \
       ('Blue_cube_weight' not in targ_group.keys()) or \
       ('Red_cube_data' not in targ_group.keys()) or \
       ('Red_cube_variance' not in targ_group.keys()) or \
       ('Red_cube_weight' not in targ_group.keys()):
        
        raise SystemExit(
            'The target block is incomplete, please check archive.')
        
    # Check if only one colour is requested:
    if colour == '': colour = ['Blue','Red']
    
    for col in range(len(colour)):
        
        # Set name for output file (if not set): 
        if outfile == '': 
            this_outfile = 'SAMI_'+name+'_'+colour[col]+'_cube.fits'
        
        # Check if outfile already exists
        if os.path.isfile(outfile):
            hdf.close()
            raise SystemExit("The nominated output .fits file ('" + 
                             outfile + "') already exists. Try again! ")
            
        # Data is primary HDU, VAR and WHT are extra HDUs. 
        data = targ_group[colour[col]+'_cube_data']
        var =  targ_group[colour[col]+'_cube_variance']
        wht =  targ_group[colour[col]+'_cube_weight']
        
        # And construct headers -- 
        # -- just the data header for now, the VAR and WHT headers supplied by 
        #    the current cubes are dummies. 
        
        # Make data header: first set all keys...
        hdr1 = pf.Header.fromkeys(data.attrs.keys())
        
        # ...then set all values
        for ihdr in range(len(hdr1)):
            hdr1[ihdr] = data.attrs.values()[ihdr]
        
        # And now set up the Image Data Units...
        # (A) Cubes: 
        hdu_c1 = pf.PrimaryHDU(np.array(data), hdr1)
        hdu_c2 = pf.ImageHDU(np.array(var), name='VARIANCE')
        hdu_c3 = pf.ImageHDU(np.array(wht), name='WEIGHT')
        
        hdulist = pf.HDUList([hdu_c1, hdu_c2, hdu_c3])
        hdulist.writeto(this_outfile)
        
        hdulist.close()
        
    # close h5file and end process
    hdf.close()
    

def export(name, h5file, get_cube=False, get_rss=False, 
           colour='', all_versions=False):
    """ The main SAMI_DB .fits export function """

    """ 
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
    
    # Check that some data has been requested: 
    if (not get_cube) and (not get_rss): 
        raise SystemExit("Please raise at least one of the 'get_???' flags.")
    
    # Digest SAMI name, search h5file, identify blocks, write .fits.
    
    hdf = h5.File(h5file)
    
    # Check that h5file is SAMI-formatted.
    if 'SAMI' not in hdf.keys():
        raise SystemExit('The nominated h5 file ('+h5file+') is not '+\
                         'SAMI-formatted')
    
    # Check that h5file contains a target block for name.
    if name not in hdf['SAMI/Targets'].keys():
        raise SystemExit('The nominated h5 file ('+h5file+') does not '+\
                         'contain a target block for SAMI '+name)
    
    # Checks done, extract some cubes. 
    
    # ***Begin multi-target loop here*** (not yet implemented)

    # Determine latest version, and therefore target block path.
    versions = hdf["SAMI/Targets/"+name].keys()
    v_num = [int(versions[-1][-2:])]  # version numbers as integers
    v_latest = max(v_num)             # latest version as largest v_num
    
    v_str = '/v'+str(v_latest).zfill(2) # version number as string
    targ_group = hdf.require_group("SAMI/Targets/"+name+v_str)
    
    # Look for cubes:
    if ('Blue_cube_data' not in targ_group.keys()) or \
       ('Blue_cube_variance' not in targ_group.keys()) or \
       ('Blue_cube_weight' not in targ_group.keys()) or \
       ('Red_cube_data' not in targ_group.keys()) or \
       ('Red_cube_variance' not in targ_group.keys()) or \
       ('Red_cube_weight' not in targ_group.keys()):
        
        raise SystemExit(
            'The target block is incomplete, please check archive.')

    # Check if only one colour is requested:
    if colour == '': colour = ['Blue','Red']
    
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

            data = targ_group[colour[col]+'_cube_data']
            var =  targ_group[colour[col]+'_cube_variance']
            wht =  targ_group[colour[col]+'_cube_weight']

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


def test_contents(h5file):
    """ Check if h5file contains cubes and RSS datasets. """

    ### UNDER CONSTRUCTION ###

    f = h5.File(h5file)
    if 'SAMI' in f.keys():

        if ('Targets' in f['SAMI'].keys()):
            print('File is SAMI-formatted')
            
            targ_gr = f['SAMI/Targets']
            targets = targ_gr.keys()

        else: sys.exit('File is not SAMI-formatted')

        for i in range(len(targets)):

            # Check contents of each group:
            all_cube = ['Blue_cube_data', 'Blue_cube_variance', 
                        'Blue_cube_weight', 'Red_cube_data', 
                        'Red_cube_variance', 'Red_cube_weight']
            
            all_rss = ['Blue_RSS_data', 'Blue_RSS_variance', 
                       'Red_RSS_data', 'Red_RSS_variance'] 
                        
            if all_cube[:] in f['SAMI/Targets/'+targets[i]+'/v01'].keys():
                print(targets[i], 'is complete')

    else: sys.exit('File is not SAMI-formatted')


def list_keys(h5file):
    """ Read and list the contents of a SAMI (or any) HDF5 file. """
    
    f = h5.File(h5file)
    print(f.keys())
    if 'SAMI' in f.keys():
        print(f['SAMI'].keys())
        if 'Targets' in f['SAMI'].keys():
            print(f['SAMI/Targets'].keys())
    f.close()
    

def remove_block(h5file, dataset):
    """ Remove a dataset from an HDF5 archive. """
    
    """
    This is not trivial! The officially sanctioned way is to unlink (delete) 
    a set and then copy the file to a new one. The unlinked (deleted) dataset
    is not transferred over. See Section 5.5.2 of the HDF documentation: 
       
             www.hdfgroup.org/HDF5/doc/UG/UG_frame10Datasets.html
             
    Perhaps the best tactic is to unlink datasets when needed and then copy the
    file every few weeks to save any unlinked space. 

    Therefore, there should be an option to overwrite a dataset if required, 
    rather than 'requiring' the dataset. 

    """
    
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

        
def simple_reader(filein):
    """ Development code that reads in a cube and mines info """ 

    hdulist = pf.open(filein)

    print(hdulist.info())

    hdulist.close()
    

def search(h5file, query_item, query_value, verbose=True):
    """ A prototype query tool """
    
    # Need to search each group recursively
    hdf = h5.File(h5file)

    #*** Assuming all /v01 for now, determine version for full implementation

    # initialist lists to hold query results:
    id_targs = []
    id_dsets = []
    

    for idloop in range(len(hdf['SAMI/Targets/'])):

        # Determine SAMI name to access target block
        sami_name = hdf['SAMI/Targets/'].keys()[idloop]
        targ_group = hdf['SAMI/Targets/'+str(sami_name)+'/v01/']
        cube_data_blue = targ_group['Blue_cube_data']
        
        # Check that the name of the directory matches the respective attribute
        if cube_data_blue.attrs['NAME'] != str(sami_name):
            raise SystemExit("The SAMI names of the POSIX structure and "+
                             "related object attribute do not match. " )

        # Check all stored attributes for the query_item
        # *** RSS strips presetntly have no attributes, fill in later
        # *** just loop through all datasets, don't repeat! 

        for dsetloop in range(len(targ_group.keys())):
            this_key = targ_group.keys()[dsetloop]
            this_dset = targ_group[this_key]
            #if query_item in this_dset.attrs:
            #try: this_dset.attrs
            #finally: print(this_dset.attrs)
            if query_item in this_dset.attrs:

                for attloop in range(len(this_dset.attrs.keys())):
                    if this_dset.attrs.keys()[attloop] == query_item:
                        if this_dset.attrs.values()[attloop] == query_value:
                            att_index = attloop
                            if verbose: 
                                print(targ_group.name + 
                                      ": key '" + query_item +
                                "' = " + this_dset.attrs.values()[attloop])

            """

            if sami_name not in id_targs: 
            id_targs.append(str(sami_name))
            #id_dsets.append(targ_group.keys()[dsetloop])
                

            if (query_item in targ_group[dsetloop].attrs):
                identified_dset = targ_group[dsetloop]
                if verbose: print('Identified attribute named "'
                                  +query_item+'" in '+identified_dset.name)

            index = np.where(query_value == identified_dset.attrs)
            print(index)
            """
            
        #print(id_targs)
           
    hdf.close()
