"""
'ingest.py', module within 'sami.db'

Description: Import codes for the SAMI Galaxy Survey data archive. 

Written: 08.10.2013, Iraklis Konstantopoulos. Based on 'db.database'.

Contact: iraklis@aao.gov.au

Table of Contents: 

.create         Creates an HDF5 file, not SAMI-specific. 
.format         Sets up the SAMI root and base (target-based) filestructure. 
.import_cube    Digests a SAMI datacube. Envokes 'import_rss' and 'fetch_SDSS' 
                from the SAMI_sdss module. 
.locate_rss     Queries a cube header, identifies parent RSS files, locates  
                them on the machine running the code. Returns filename listing. 
.make_list      Make a list for import_many from the contents of a directory. 
.importMaster   Imports a SAMI target table. 

Known bugs and fixes: 

2013.10.31 -- PyTables cannot read attributes with empty value fields. We need 
              to get that dictionary going right away. This has been contained 
              by not including any header items without values (which is good 
              practise anyway) as [comm] attributes (the temporary measure), and
              filling in blank header tickets with '[blank]'
 
"""

import numpy as np
import h5py as h5
import tables
import astropy.io.fits as pf
import astropy.io.ascii as ascii
import os
import sys
import sami

""" 
For commit message: 

Quick bug fix in make_list(). 

Discovered and fixed a bug in make_list() that assumed that the hidden file '.DS_Store' will always be present in a SAMI observing run directory. This had to be removed for the resulting list to work. Made this removel conditional on the file actually existing. 
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
                ingest_rss=True, rss_only=False, dataroot='./', verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Import a set of data-cubes and parents to the SAMI Archive. 
    
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
      <> Record all header card comments, insert into table (version-specific).

    (6) Perform any QC tests and cleaning, EXIT successfully. 
    
    * Steps (2) and (3) will eventually rely on reading header tickets.

    [TODO] Every SAMI h5 file should contain the Target and Star catalogues as 
    tables, so that QC checks can be performed (many other reasons exist). This
    could be problematic when it comes to cluster fields. Do they all live in 
    the same archive? What is stopping us? Different Target tables... 
    
    [TODO] Set the default 'dataroot' to a header ticket stored by the DR 
    manager code. This ticket is not yet there. 
    
    [TODO] In export code, give the user the option to package the PSF star that
    corresponds to any Target cube being downloaded. 
     
    Arguments: 
    
    blue_cube   [str]  FITS filename of blue SAMI cube ("" for none). 
    red_cube    [str]  FITS filename of red SAMI cube ("" for none). 
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
    
    for i in range(len(hdulist)):
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
        cube_data = eat_data(g_target, colour[i]+"_Cube_Data", 
                             HDU[0].data, hdr=HDU[0].header)
        cube_var  = eat_data(g_target, colour[i]+"_Cube_Variance", 
                             HDU[1].data, hdr=HDU[1].header)
        cube_wht  = eat_data(g_target, colour[i]+"_Cube_Weight", 
                             HDU[2].data, hdr=HDU[2].header)

        # IMPORT RSS
        # ----------
        if ingest_rss or rss_only:
            for rss_loop in range(n_rss):
                rss_index = str(rss_loop+1)
                myIFU = sami.utils.IFU(rss_list[rss_loop], 
                                       sami_name, flag_name=True)
                # RSS: Data, Variance. 
                rss_data = eat_data(g_target, colour[i]+"_RSS_Data_"+rss_index,
                                    myIFU.data, hdr=myIFU.primary_header)
                rss_var = eat_data(g_target, colour[i]+"_RSS_Variance_"+\
                                   rss_index, myIFU.var, importHdr=False)

                # RSS: Fibre Table. Only wish to import bits an pieces. 
                tempTab = [ myIFU.fibtab['FIBNUM'], 
                            myIFU.fibtab['FIB_MRA'], myIFU.fibtab['FIB_MDEC'],
                            myIFU.fibtab['FIB_ARA'], myIFU.fibtab['FIB_ADEC'] ]
                fibComb = np.transpose(np.array(tempTab))

                rss_fibtab = eat_data(g_target, colour[i]+'_RSS_FibreTable_'+\
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
    """ Wrapper for import_cube to digest any number of cubes. 
    
    'tablein' needs to list the blue and red cubes in each space-separated row. 
    All other arguments refer to the 'import_cube' args. 

    Safe Mode is tricky. It will produce an intermediate file for every cube it
    adds, rather than a single backup file. Problem. 
    """
    
    ### OBSTYPE NEEDS TO BE DEFINED
    
    if timing: 
        import time
        timer_zero = time.time()
        
    tabdata = ascii.read(tablein, data_start=0, names=['blue', 'red'])
    
    for loop in range(len(tabdata)):
        if timing: timer_start = time.time()
        if verbose: print("Processing "+
                          os.path.basename(tabdata['blue'][loop])+", "+
                          os.path.basename(tabdata['red'][loop]))
        
        import_cube(tabdata['blue'][loop], tabdata['red'][loop], h5file, 
                    version, safe_mode=safe_mode, ingest_rss=ingest_rss, 
                    rss_only=rss_only, dataroot=dataroot, verbose=verbose)
        if timing: 
            timer_end = time.time()
            print(loop,timer_end-timer_start)
            time_elapsed = timer_end-timer_zero
            print(time_elapsed/(loop+1))
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def importMaster(h5file, tabin, cdf='', version='', verbose=False):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Import a SAMI target table to an h5 archive. 

    1) Read in the table (all numeric), given preset columns (supply). 
    2) Attach as h5 table in the root directory. 

    Note that 'name' is the IAU name and the SAMI identifier is 'CATID'. 
    """
    import tables
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
    """ Doing this with h5py cause PyTables is retarded. Will then close and 
    re-open with PyTables... """
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
                    if verbose: print('   '+rss_path[i])

    # Check 
    if len(rss_files) == n_rss:
        if verbose: print('Located all RSS files.')
    else: 
        raise SystemExit("Sorry, I didn't find all "+str(n_rss)+" RSS files.")

    return rss_path
    

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def make_list(dataroot='./', tableout='SAMI_input.lis', 
                  overwrite=False, append=False, debug=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Make an .import_many import list based on contents of basedir. 

    There has been radical change in the structure of the data reduction 
    directory substructure. Data now organised in a <version>/<dates>/cubed/ 
    fashion, which makes the old make_list() code defunct. 

    The code should now scan a directory that pertains to a single observing
    run, list the contents and combine the filenames within eack sami-named 
    folder into an import_many() command. 

    The dataroot is now an observing run folder. 
    """

    # Check if both the overwrite and append flags are up, exit of so.
    if (overwrite) and (append):
        raise SystemExit("Both the 'overwrite' and 'append' flags have " +
                         "been raised. That won't work, you have to choose " +
                         "just one! Exiting. ")
    
    # Scan the dataroot/cubed subdirectory. 
    nameList = os.listdir(dataroot+'/cubed')

    # Sometimes directories have a '.DS_Store' file, remove it. 
    if '.DS_Store' in nameList:
        nameList.remove('.DS_Store')

    # Create a file buffer, decide whether to overwrite or append: 
    if not append: f = open(tableout, 'w')
    if append: f = open(tableout, 'a')

    # Little function to compose a single list line. 
    def writeLine(name):
        base = dataroot+'/cubed/'+name+'/'
        fnames = os.listdir(base)
        f.write(base+fnames[0]+" "+base+fnames[1]+"\n")

    writeAll = [writeLine(str_in) for str_in in nameList]

    f.close()



# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def make_list_old(basedir='', tableout='SAMI_input.lis', 
                  overwrite=False, append=False, debug=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Make an .import_many digestion list based on contents of basedir. 
 
    This code relies on the existence of the standard ccd_1 and ccd_2 subdirs, 
    as delivered by the good people in the Data Reduction Working Group. This 
    should not change (ever!). Major revision will be required if it does. 

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
