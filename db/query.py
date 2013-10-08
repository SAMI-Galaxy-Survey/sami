"""
'query.py', module within 'sami.db'

Description: Import codes for the SAMI Galaxy Survey data archive. 

Written: 08.10.2013, Iraklis Konstantopoulos. Based on 'db.database'.

Contact: iraklis@aao.gov.au

Updates: .-

Table of Contents (* to-be, **: under construction): 

.query_master**   Query a SAMI target table. 
.query_multiple** Perform any number of queries on any number of SAMI tables. 
.search**         The query prototype.
.test_contents**  ??
"""

import numpy as np
import h5py as h5
import astropy.io.fits as pf
import astropy.io.ascii as ascii
import os
import sys
import sami


def query_master(h5in, query, idfile='sami_query.lis', 
                verbose=True, returnID=True, overwrite=True):
    """ 
    ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- 
    Read a SAMI master table and perform a query

    The query should be passed either as a string argument or as an ascii file 
    containing such a string. Adding '@list' functionality will facilitate 
    browser-generated queries farther down the track. 
    ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- 
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
    """ 
    ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- 
    Query multiple tables within an h5 archive and combine results 
    ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- 
    """

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
