"""
ingest.py
  Module within sami.db

Description: 
  Import codes for the SAMI Galaxy Survey data archive. 

Written: 
  08.10.2013, Iraklis Konstantopoulos. Based on 'db.database'.
  
Updated:
  03.04.2014, Nic Scott - added import_table() and import_manytables()
  05.08.2014, Nic Scott - modified to import_hlsp() and import_manyhlsps()

Contact: 
  iraklis@aao.gov.au

Dependencies:
  os, sys, numpy, astropy, h5py, tables, sami, 
  datetime, shutil [import_cube], 
  time [import_many], 
  fnmatch [locate_rss].

Table of Contents: 

  [create] --- Create a SAMI-formatted  HDF5 file. 

    <args>
    h5file     [str]  The name of the HDF5 file to be created. 
    overwrite  [boo]  Re-initialises a file. 
    verbose    [boo]  Toggles diagnostic and declarative verbosity. 
                  
    Start an HDF5 file in 'a' mode, that is, read/write if it exists, and create
    otherwise. Writes only the root HDF5 group "SAMI". The overwrite flag will 
    re-initialise an existing file, that is delete and create a "SAMI" group. 


  [import_cube] --- Import a set of data-cubes and parents to the SAMI Archive.

    <args> 
    blue_cube   [str]  FITS filename of blue SAMI cube ("" for none). 
    red_cube    [str]  FITS filename of red SAMI cube ("" for none). 
    h5file      [str]  The name of the SAMI archive file onto which to save. 
    version     [str]  Version number (eventually a header item). 
    safe_mode   [boo]  Automatically creates a time-stamped backup archive. 
    ingest_rss  [boo]  Locate and import RSS parents as 'strips'. 
    rss_only    [boo]  Only import RSS parents, no cubes. 
    dataroot    [boo]  The root directory to scan for RSS parents. 
    verbose     [boo]  Toggles diagnostic and declarative verbosity. 

    The main SAMI import code. It works only in one mode and is therefore not 
    modularised. There is the option to 'strip' RSS files, identify the arrays
    and segments pertaining to the cube being imported (there is a complex 
    relationship between the multiple dithers of RSS parents and the resultant
    cubes) and add them as datasets. 

    The code goes through a series of safety and quality control checks before
    writing any data to the nominated HDF5 file: 
      
      (1) Checks if the nominated h5 file (h5file) exists.

      (2) Requires a group for the nominated (/header-supplied) data version. 

      (3) Checks h5file for the SAMI cubes in the data version to be imported*:
        <> Do any data exist in this data version? EXIT with error. 
        <> No? Create target group. 

      (4) Checks observation type by identifier: 
        <> Star CATIDs in range [1e7, 2e7), 
        <> Galaxies in (0, 1e7) and (9e9, 1e10).
        <> Require "Calibrator" or "Target" group. 
        <> CATID not found? EXIT cleanly with error. 

    At that point data can be imported, a function is defined for this purpose,
    eat_data(). All header card comments are recorded and saved as attributes
    on the respective dataset. Some final QC checks are performed and the code
    exits successfully. 
 
    Note that step (2) will eventually rely on reading the data version as a 
    header ticket. At the moment it is passed manually as a required argument.

    Envokes locate_rss(). 


  [import_many] --- @list wrapper for import_cube. 

    <args>
    tablein  [str]  An input file listing blue/red cube filenames (+path).  
    * All other arguments inform import_cube(), see above for details. 

    This code is designed to receive output from make_list(), which lists pairs
    of blue/red SAMI cube filenamees (full path must precede, space-separated). 
    These are fed to import_cube() one line at a time, so the remainder of the 
    arguments for this function are fed directly to import_cube(), where the 
    user should look up their meanings. 

    A loop suppresses safe_mode to False for every one but the first iteration 
    of import_cube() the code envokes. The same goes for the version_confirm
    boolean. This is a temporary measure, as the check should be performed every
    time. Once the cubes contain the data release version as a header ticker 
    this can be abandoned. 


  [importMaster] --- Import a SAMI target table to an HDF5 archive. 

    <args>
    h5file   [str]  The name of the HDF archive into which to import the table.
    tabin    [str]  The ascii file containing the SAMI target table. 
    cdf      [str]  Column definitions, not yet supported. 
    version  [str]  Data version. Leave blank for latest. 
    verbose  [boo]  Toggles diagnostic and declarative verbosity. 

    This function reads in a SAMI target table as a compound dataset. The code 
    will check the data version to decide where the table should be inserted. 
    The names of the columns are supplied manually assuming that there will only
    ever be one format for SAMI target tables. 


  [locate_rss] --- Read in a SAMI cube, identify and locate its parent RSS files

    <args>
    cubein    [str]  The cube whose parents the code will seek. 
    dataroot  [str]  Path to data directory (relative or absolute).
    verbose   [boo]  Toggles diagnostic and declarative verbosity. 

    SAMI spectra are imprinted onto the AAOmega detector as 'row-stack spectra',
    commonly referred to as RSS files. Since one SAMI frame conveys information
    gathered by thirteen IFUs, we do not have a one-to-one relation between RSS
    and cubes. What's more, there are typically seven dithers, making this a 13-
    to-seven correspondence. 

    Since the SAMI archive is oriented along cubes, rather than observation 
    frames, the full complement of RSS parents needs to be found for each cube, 
    but not stored multiple times. 

    This code queries a cube header, identifies all parent RSS files (filenames
    stored as header tickets), and locates them on the local filesystem. The 
    dataroot argument allows the user to supply a base directory which the code
    can search recursively for the RSS frames, thus saving the time it would 
    take python to perform a 'locate' task. 
    
    Returns filename listing, including full paths. 


  [make_list] --- Make an .import_many import list based on contents of basedir.

    <args>
    dataroot   [str]  Path to data directory (relative or absolute).
    tableout   [str]  Filename of output table, default is 'SAMI_input.lis'.
    overwrite  [boo]  Delete and overwrite a list. 
    append     [boo]  Append results to end of existing buffer. 

    Simple function that populates a variable width, two-column table listing 
    pairs of filenames of blue/red SAMI cubes to be imported into an archive. 

    This is best used by being tasked with scanning a directory that pertains to
    a single SAMI observing run, listing the contents and combining the 
    filenames within eack sami-named folder into an import_many() input table.
    That way 'dataroot' can be set to the observing run path. 


Known bugs, issues, and fixes: 

  2013.10.31 -- PyTables cannot read attributes with empty value fields. We need
                to get that dictionary going right away. This has been contained
                by not including any header items without values (which is good 
                practise anyway) as [comm] attributes (the temporary measure), 
                and filling in blank header tickets with '[blank]'
"""

import numpy as np
import h5py as h5
import tables
import astropy.io.fits as pf
import astropy.io.ascii as ascii
import os
import sys
import astropy.table as astro_table

""" 
For commit message: 

Changed the overwrite process of create(). 

File access was changed to 'a' at the Nov'13 Busy Week. That isn't what we want though, truncation is the way to go, not appending, as the links are still there. Also got rid of a little print statement that seems to have been left over from debugging.  
"""

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def create(h5file, overwrite=False, verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Create a SAMI-formatted HDF5 file """

    # Check that the 'h5file' string has a .h5 extension
    if not h5file[-3:] == '.h5': h5file = h5file + '.h5'

    # Check if file already exists, overwite if overwrite==True
    if os.path.isfile(h5file):
        file_already_exists = True
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
                ingest_rss=True, rss_only=False, dataroot='./', verbose=True,
                version_confirm=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Import a set of data-cubes and parents to the SAMI Archive. """
    
    import sami
    
    # Check if the nominated h5 file exists; prompt for creation if not, exit. 
    if not os.path.isfile(h5file):
        raise SystemExit("Cannot find the nominated HDF5 file ('"+h5file+"'). "+
                         "Please create a file using the 'create' function")
        
    # Also check if the nominated cubes exist. 
    if (blue_cube != '') and (not os.path.isfile(blue_cube)):
        raise SystemExit("Cannot find the nominated blue cube. Exiting. ")

    if (red_cube != '') and (not os.path.isfile(red_cube)):
        raise SystemExit("Cannot find the nominated red cube. Exiting. ")
        
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

    # Confirm if version is not latest (cfoster, 8/11/2013)
    if hdf['SAMI'].keys() != [] and version < max(hdf['SAMI'].keys()) and \
        version_confirm:
        print
        usr_input=raw_input("The selected version is not the latest. Are "+
                            "you sure you want to continue? [Y/n]")
        if (usr_input == 'n'): 
            raise SystemExit("The wrong version number was entered. Please "+
                             "fix and rerun.")
        else:
            version_confirm=False

    g_version = hdf.require_group("SAMI/"+version)
    
    # Version group in place, let's import some data! 

    # Read the header, find out the target name. 
    if blue_cube != '': hduBLUE = pf.open(blue_cube)
    if red_cube != '': hduRED  = pf.open(red_cube)
    
    # Check that the nominated blue/red cubes are indeed of same target. 
    if (blue_cube != '') and (red_cube != ''): # if B+R cubes are nominated

        if hduBLUE[0].header['NAME'] == hduRED[0].header['NAME']:
            # Store SAMI ID as 'NAME'. 
            sami_name = hduBLUE[0].header['NAME']
            
            # And close the two HDUs. 
            hduBLUE.close()
            hduRED.close()

        else: 
            hdf.close()
            raise SystemExit("The two cube files are not matched according to "+
                        "their header-listed names. Please review the files "+
                         "(listed below) and the validity of their headers: \n"+
                         '\n  > blue cube: "'+blue_cube+
                         '"\n  > red cube:  "'+red_cube+'"')

    else:
        if (blue_cube == '') & (red_cube == ''):
            hdf.close()
            raise SystemExit("Oops! It looks like both the 'blue_cube' and "+
                             "'red_cube' are blank strings. ")
            
        if blue_cube != '': 
            sami_name = hduBLUE[0].header['NAME']
            hduBLUE.close() 

        if red_cube != '':  
            sami_name = hduRED[0].header['NAME']
            hduRED.close()

    # What sort of observation this is: Target or Calibrator? Make group. 
    """ Stars in range [1e7, 2e7), galaxies in (0, 1e7) \join (9e9, 1e10). """
    if (int(sami_name) >= 1e7) and (int(sami_name) < 2e7):
        obstype = 'Calibrator'
    else: 
        obstype = 'Target'

    g_obstype = g_version.require_group(obstype)
    g_target = g_obstype.require_group(sami_name)

    # Check if data already exist in this version for this target. 
    if (g_target.keys() != []) and (not rss_only):
        raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                         "already contains data for the nominated SAMI "+
                         "target ("+sami_name+"). Please check archive "+
                         "and cubes. ")
    
    # Data import: begin big colour loop.
    colour= []
    hdulist = []

    # Monochrome input: check if either cube input is an empty string. 
    if blue_cube != '': 
        colour.append('B')
        hdulist.append(blue_cube)

    if red_cube != '': 
        colour.append('R')
        hdulist.append(red_cube)
    
    for thisHDU in range(len(hdulist)):
        HDU = pf.open(hdulist[thisHDU])
        if verbose: 
            print
            print(HDU.info())

        # Identify RSS parents, locate in filesystem. 
        if (ingest_rss) or (rss_only):

            # First check if there are any already in here. 
            if colour[thisHDU]+'_RSS_data_1' in g_target.keys(): 
                raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                        "already contains RSS strips for the nominated SAMI "+
                        "target ("+sami_name+"). Please check archive "+
                        "and cubes. ")

            rss_list = locate_rss(hdulist[thisHDU], dataroot, verbose=verbose)
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

        # DEFINE DATA IMPORT FUNCTION
        # ---------------------------
        def eat_data(group, name, hdu, hdr='', importHdr=True, hdrItems=[]):
            """ Import datasets and headers in a consistent manner """

            the_array = group.create_dataset(name, data=hdu, 
                                             chunks=True, compression='gzip')

            if importHdr:

                # If importing selectively from the header: 
                if hdrItems != []:
                    # Create and populate a list of cards. 
                    cardList = []
                    for hcard in range(len(hdrItems)):
                        cardList.append(hdr.cards[hdrItems[hcard]])
                    
                    # Redefine header as subset of entire thing. 
                    hdr = pf.Header(cards=cardList)
                    
                for n_hdr in range(len(hdr)):
                    if hdr.values()[n_hdr] != '':
                        the_array.attrs[hdr.keys()[n_hdr]] = hdr.values()[n_hdr]
                    else:
                        the_array.attrs[hdr.keys()[n_hdr]] = '[blank]'
                    # Save header comments as separate attributes (temp). 
                    if hdr.comments[n_hdr] != '':
                        the_array.attrs\
                            ["[comm]"+hdr.keys()[n_hdr]] = hdr.comments[n_hdr]

            return the_array

        # IMPORT CUBE
        # -----------
        cube_data = eat_data(g_target, colour[thisHDU]+"_Cube_Data", 
                             HDU[0].data, hdr=HDU[0].header)
        cube_var  = eat_data(g_target, colour[thisHDU]+"_Cube_Variance", 
                             HDU[1].data, hdr=HDU[1].header)
        cube_wht  = eat_data(g_target, colour[thisHDU]+"_Cube_Weight", 
                             HDU[2].data, hdr=HDU[2].header)

        # IMPORT RSS
        # ----------
        if ingest_rss or rss_only:
            for rss_loop in range(n_rss):
                rss_index = str(rss_loop+1)
                myIFU = sami.utils.IFU(rss_list[rss_loop], 
                                       sami_name, flag_name=True)
                # RSS: Data, Variance. 
                rss_data = eat_data(g_target, 
                                    colour[thisHDU]+"_RSS_Data_"+rss_index,
                                    myIFU.data, hdr=myIFU.primary_header)
                rss_var = eat_data(g_target, colour[thisHDU]+"_RSS_Variance_"+\
                                   rss_index, myIFU.var, importHdr=False)

                # RSS: Fibre Table. Only wish to import bits an pieces. 
                tempTab = [ myIFU.fibtab['FIBNUM'], 
                            myIFU.fibtab['FIB_MRA'], myIFU.fibtab['FIB_MDEC'],
                            myIFU.fibtab['FIB_ARA'], myIFU.fibtab['FIB_ADEC'] ]
                fibComb = np.transpose(np.array(tempTab))

                rss_fibtab = eat_data(g_target, 
                                      colour[thisHDU]+'_RSS_FibreTable_'+\
                        rss_index, fibComb, hdr=myIFU.fibre_table_header, \
                        hdrItems=['CENRA', 'CENDEC', 'APPRA', 'APPDEC'])

        HDU.close()

    # Close h5file, exit successfully
    hdf.close()

    if verbose: 
        print
        print("Exiting successfully")
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def import_many(tablein, h5file, version, safe_mode=False, 
                ingest_rss=True, rss_only=False, dataroot='./', 
                verbose=True, timing=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ @list wrapper for import_cube. """
    
    if timing: 
        import time
        timer_zero = time.time()
        
    tabdata = ascii.read(tablein, data_start=0, names=['blue', 'red'])    
    
    for loop in range(len(tabdata)):
        if timing: timer_start = time.time()
        if verbose: print("Processing "+
                          os.path.basename(tabdata['blue'][loop])+", "+
                          os.path.basename(tabdata['red'][loop]))
        if loop>0:
            version_confirm=False
            safe_mode=True
	else:
	    version_confirm=True
	    safe_mode=False
            
        import_cube(tabdata['blue'][loop], tabdata['red'][loop], h5file, 
                    version, safe_mode=safe_mode, ingest_rss=ingest_rss, 
                    rss_only=rss_only, dataroot=dataroot, verbose=verbose,
                    version_confirm=version_confirm)
        if timing: 
            timer_end = time.time()
            print(loop,timer_end-timer_start)
            time_elapsed = timer_end-timer_zero
            print(time_elapsed/(loop+1))
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def importMaster(h5file, tabin, cdf='', version='', verbose=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Import a SAMI target table to an HDF5 archive. """

    import sami.db.export as export

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

    # Check if the nominated file exists. 
    if not os.path.isfile(h5file):
        print("The nominated HDF5 file ('"+h5file+"') does not exist. "+
              "Creating new file. ")
        
    # Get (latest) version. 
    """ It is difficult to do this with tables, so using h5py. """
    hdf = h5.File(h5file, 'r')
    version = export.getVersion(h5file, hdf, version)
    hdf.close()

    # Open the nominated h5 file -- use PyTables. 
    hdf = tables.openFile(h5file, 'r+')

    # Require the Table group (do not create). 
    g_table = hdf.createGroup("/SAMI/"+version, "Table", title='SAMI Tables')

    # Create a target table in the Target directory of the h5file. 
    master = hdf.createTable(g_table, 'SAMI_MASTER', sami_master)

    """ An array of attributes could be tacked to the table here. """

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
    hdf.close()

    
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def locate_rss(cubein, dataroot='./', verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
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
                    if verbose: print('   '+str(rss_path[-1:]))

    # Check 
    if len(rss_files) == n_rss:
        if verbose: print('Located all RSS files.')
    else: 
        raise SystemExit("Sorry, I didn't find all "+str(n_rss)+" RSS files.")

    return rss_path
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def make_list(dataroot='./', tableout='SAMI_input.lis', 
                  overwrite=False, append=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Make an .import_many import list based on contents of basedir. """

    # Check if both the overwrite and append flags are up, exit of so.
    if (overwrite) and (append):
        raise SystemExit("Both the 'overwrite' and 'append' flags have " +
                         "been raised. That won't work, you have to choose " +
                         "just one! Exiting. ")
    
    # Scan the dataroot/cubed subdirectory. 
    nameList = os.listdir(dataroot+'/cubed')

    # Create a file buffer, decide whether to overwrite or append: 
    if not append: f = open(tableout, 'w')
    if append: f = open(tableout, 'a')

    # Little function to compose a single list line. 
    def writeLine(name):
        base = dataroot+'cubed/'+name+'/'
        fnames = os.listdir(base) 

        # Sometimes directories have a '.DS_Store' file, remove it. 
        if '.DS_Store' in fnames:
            fnames.remove('.DS_Store')

        f.write(base+fnames[0]+" "+base+fnames[1]+"\n")

    writeAll = [writeLine(str_in) for str_in in nameList]

    f.close()


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def import_manyhlsps(inlist,indescription,h5file,version,
                 overwrite=False,confirm_version=True,verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Import many tables into the SAMI archive from a list of files."""


    # Inlist must contain a list of files and SAMI IDs

    f_list = open(inlist,'r')
    for line in f_list:
        infile,id = line.split()
        try:
            import_hlsp(infile,indescription,h5file,version,target=id,
                     overwrite=overwrite,confirm_version=confirm_version,
                     verbose=verbose)
        except Exception as e:
            print e
            print 'Import failed for '+infile
            print
            continue
        except SystemExit as e:
            print e
            print 'Import failed for '+infile
            print
            continue

    print
    print 'Import complete'

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def import_hlsp(infile,indescription,h5file,version,target='',
               overwrite=False,confirm_version=True,verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Import a table into the SAMI archive."""

    # Check the hdf5 file exists (else exit)
    if not os.path.isfile(h5file):
        raise Exception("Cannot find the nominated HDF5 file ('"+h5file+"'). "+
                     "Please create a file using the 'create' function")

    # Check both the infile and indescription exist.
    # Check that the indescription is a valid .txt file (else skip)
    if not os.path.isfile(infile):
        raise SystemExit("Cannot find the DMU file ('"+infile+"').")
    if not os.path.isfile(indescription):
        raise SystemExit("Cannot find the description file ('"
                         +indescription+"').")
    ext = os.path.splitext(indescription)[1]
    if ext != ".txt":
        raise SystemExit("Description file ('"+indescription+
                         "') is not a valid .txt file.")

    # Check h5file is a SAMI DB file

    hdf = h5.File(h5file,'r+')

    if "SAMI" not in hdf.keys():
        hdf.close()
        raise SystemExit("The nominated HDF5 file ('"+h5file+"') "+
                     "is not properly formatted. Please initialise the "+
                     "filesystem using 'SAMI_DB.format'")

    # Check whether a valid version ID has been specified

    if type(version) is str:  v_numeric = float(version)
    if type(version) is float: v_numeric = version
    if not version in hdf['SAMI'].keys():
        hdf.close()
        raise SystemExit("An invalid version was specified.")

    # Check whether a SAMI target ID has been specified, otherwise
    # assume a table of (sub)sample properties has been supplied.
    # If a SAMI target ID has been specified, check that a group
    # exists for this target already (else skip)
    # Set ingroup to the appropriate group

    if target != '':
        if not target in hdf['SAMI/'+version+'/Target/'].keys():
            hdf.close()
            raise SystemExit("No version " + version + " data exists for SAMI ID:"+target+
                             ". Target-specific data products must be associated with a"+
                             " target whose SAMI data already exists in the archive.")
        target_path = '/SAMI/'+version+'/Target/'+target+'/'
    else:
        target_path = '/SAMI/'+version+'/tables/'


    # Read in contents of indescription into a long
    # description string and a short description string

    f_desc = open(indescription,'r')
    title = f_desc.readline()
    title = title.strip()
    author = f_desc.readline()
    author = author.strip()
    contact = f_desc.readline()
    contact = contact.strip()
    reference = f_desc.readline()
    reference = reference.strip()
    short_description = f_desc.readline()
    short_description = short_description.strip()
    # NS Modify this to allow for formatting in long description?
    long_description = f_desc.readline()
    long_description = long_description.strip()


    # Check whether dataset exists.
    # Skip if exists and overwrite=False
    dataset,extension = os.path.splitext(infile)


    # Identify type of input data - fits image, fits table, ascii table
    # Check file to see if it's a .fits image, .fits table or ascii table
    if extension == '.fits':
        hdu = pf.open(infile)
        if ((len(hdu) == 1) and (hdu[0].is_image == True)):
            # File is a fits image
            map = hdu[0].data
            header = hdu[0].header
            hdf[target_path].create_dataset(dataset,data=map)
        
        elif ((len(hdu) == 2) and (hdu[1].is_image == False)):
            # File is a fits binary table
            table = astro_table.Table.read(infile,format='fits')
            header = hdu[1].header
            table.write(hdf,path=target_path+dataset)
        else:
            raise SystemExit("Please supply either a single-extension fits image "+
                             "or a two-extension fits binary table.")
    else:
        # File is an ascii table
        
        raise SystemExit("Ascii table import is not yet supported. Please supply a .fits file.")
        #table = astro_table.read(infile,format='ascii')
        #table_header =


    # Create new dataset of the appropriate size and type and add initial attributes

    dset = hdf[target_path+dataset]

    dset.attrs.create('Title',title)
    dset.attrs.create('Author',author)
    dset.attrs.create('Contact',contact)
    dset.attrs.create('Reference',reference)
    dset.attrs.create('Short description',short_description)
    dset.attrs.create('Long description',long_description)

    # Write header information as attributes
    # Each attribute is a 2 element array, with the attribute name
    # being the same as that of the corresponding fits header item.
    # The 1st element of the attribute is its value, the second
    # is the fits comment.

    if extension == '.fits':
        for n_hdr in range(len(header)):
            dset.attrs.create(header.keys()[n_hdr],(header[n_hdr],header.comments[n_hdr]))

    else:
        print "Ascii not supported, but you can't be here yet"


    hdf.close()


""" THE END """
