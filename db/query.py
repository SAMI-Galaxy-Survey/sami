"""
'query.py', module within 'sami.db'

Description: Query codes for the SAMI Galaxy Survey data archive. 

Written: 08.10.2013, Iraklis Konstantopoulos. Based on 'db.database'.

Contact: iraklis@aao.gov.au

Updates: .-

Table of Contents: 

.queryMaster    Query a SAMI target table. 
.query_multiple  Perform any number of queries on any number of SAMI tables. 
.search          The query prototype.
.test_contents   ??
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import h5py as h5
import tables
import os
import sys

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def print_sami(s, idfile, queryText, outFile=True, verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Define a Row Iterator for screen output """ 

    # Prepare some variables. 
    counter = 0
    idlist = []

    # Iterate over all supplied rows. 
    if outFile:
            f = open(idfile, 'w')
  
    for tables.row in s:
        name, z = tables.row['CATID'], tables.row['z_spec']
        if verbose:
            print("  Found SAMI galaxy %s at redshift z=%g" % (name, z))
        counter += 1
        idlist.append(name)
        if outFile: 
            f.write(str(name)+'\n')
    if verbose:
        print("\n  Found "+str(counter)+" galaxies satisfying query:\n  "+
              queryText)

    if outFile: 
        f.close()
    
    return(idlist)


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def makeTable(table, tabIndex, tableOut='sami_selection.html'):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Generate an html table for website output """

    print('TEST')
    print(tabIndex)
    print('TEST')

    # Where do the quick-look plots live? 
    baseURL ='file:///Users/iraklis/Data/SAMI/datasheets/GAMA/'

    # Write html preamble. 
    htmlTab = '<html><body><table>'

    # Write Row 0, column headings. 
    htmlTab = htmlTab+\
              "<tr>"+\
              "<td>Quicklook</td>"+\
              "".join(["<td>"+str(s)+"</td>" for s in table.colnames])+\
              "</tr>"

    # Populate the table. 
    for tables.row in table[tabIndex]:
        
        hlink = "<td><a href='"+baseURL+str(tables.row[0])+".pdf'>" +\
                "View</a></td>"
        try:
            htmlTab = htmlTab+\
                      "<tr>" + hlink+\
                      "".join(["<td>"+str(s)+"</td>" for s in tables.row])+\
                      "</tr>"
        except:
            print('Nah, mate [2]')

    # Wrap up html, return table.
    htmlTab = htmlTab +"</table></body></html>"
    return(htmlTab)


    """ This used to write an html file. Keeping old code here for now. 
    # Open a file to write the table, write html preamble. 
    f = open(tableOut, 'w')

    f.write('<html><body><table>')
    f.write("<tr>" + 
            "<td>Quicklook</td>" + 
            "".join(["<td>"+str(s)+"</td>" for s in table.colnames]) + 
            "</tr>")

    # Where do the quick-look plots live? 
    baseURL ='file:///Users/iraklis/Data/SAMI/datasheets/GAMA/'

    # Do the deed. 
    for tables.row in table[tabIndex]:

        hlink = "<td><a href='"+baseURL+str(tables.row[0])+".pdf'>" +\
                "View</a></td>"
        f.write("<tr>" + hlink + 
                "".join(["<td>"+str(s)+"</td>" for s in tables.row]) + 
                "</tr>")

    # Wrap up html. 
    f.write("</table></body></html>")
    f.close()
    """ 


# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def queryMaster(h5file, queryIn, version='', idfile='sami_query.lis', 
                verbose=False, returnID=False, tabulate=True, overwrite=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Read a SAMI master table and perform a query """

    import sami.db.export as export

    # Interpret the 'query' argument (look for a filename). 
    if os.path.isfile(queryIn):
        fq = open(queryIn, 'r')
        queryText = fq.readlines()[0]
    else:
        queryText = queryIn

    # Check that the nominated h5 file exists. 
    if not os.path.isfile(h5file):
        raise SystemExit("Cannot find nominated HDF5 file ('"+h5file+"').")
        
    # Get latest data version, if not supplied
    hdf0 = h5.File(h5file, 'r')
    version = export.getVersion(h5file, hdf0, version)
    hdf0.close()

    # Open-read h5file. 
    hdf = tables.openFile(h5file, 'r')

    # Optionally open an ascii file to write IDs returned by the query. 
    if returnID:
        # Check if the file exists, check overwrite flag:
        if os.path.isfile(idfile):
            if not overwrite:
                raise SystemExit("The nominated output file ('"+idfile+"') "+
                                 "already exists. Please raise the 'overwrite'"+
                                 " flag or enter a different filename. ")

    # Identify the SAMI master table, assumed to live in the Table directory
    g_table = hdf.getNode('/SAMI/'+version+'/Table/')
    master = g_table.SAMI_MASTER

    # Run the row iterator.
    try: 
        idlist = print_sami(master.where(queryText), idfile, 
                            queryText, outFile=returnID, verbose=verbose)
    except:
        hdf.close()
        raise SystemExit("Oops! Your query was not understood. Please "+
                         "check the spelling of the chosen variable.")

    # Generate a table, if requested, given a row index and the table.
    if tabulate:
        tabIndex = [row.nrow for row in master.where(queryText)]
        madeTab = makeTable(master, tabIndex)
        hdf.close()
        return(madeTab)

    # Otherwise close h5 file and return list of query results. 
    if not tabulate:
        hdf.close()
        return(idlist)



# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def queryMultiple(h5file, qfile, writeFile=False, outFile='multipleQuery.lis', 
                  overwrite=False, tabulate=True, verbose=True, version=''):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Query multiple tables within an h5 archive and combine results """

    import sami.db.export as export
    
    """ 
    This is modelled after query_master. One main difference is the query 
    argument, which has to be a file in this case, rather than a string. 
    """
    
    # Check that the input files exist
    if (not os.path.isfile(h5file)) or (not os.path.isfile(qfile)):
        raise SystemExit("One of the nominated files does not exist. Exiting.")

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

    # Get latest data version, if not supplied
    hdf0 = h5.File(h5file, 'r')
    version = export.getVersion(h5file, hdf0, version)
    hdf0.close()

    # Now run all queries: 

    # Open h5 file with PyTabs: 
    hdf = tables.openFile(h5file, 'r')

    # Read all tables, append to list
    h5tabs = []
    for i in range(counter/2):
        try:
            h5tabs.append(hdf.getNode('/SAMI/'+version+'/Table/', tabs[i]))
        except:
            hdf.close()
            raise SystemExit("Oops! Your query was not understood. Please "+
                             "check the spelling of table '"+tabs[i]+"'.")

    # OK, have the tables defined as variables, now need to query them. 

    # Run each query: 
    all_lists = []
    for i in range(counter/2):
        try:
            idlist = print_sami(h5tabs[i].where(queries[i]), outFile, 
                                queries[i], outFile=False, verbose=False)
            if verbose: 
                print("Query "+str(i+1)+": Found "+str(len(idlist))+
                      " galaxies satisfying "+queries[i])
            all_lists.append(idlist)
        except:
            hdf.close()
            raise SystemExit("Oops! Your query was not understood. Please "+
                             "check the spelling of query '"+queries[i]+"'.")

    # Finally, intersect all idlists within all_lists container return, exit.
    final_idlist = set(all_lists[0]).intersection(*all_lists)
    if verbose: 
            print(" ------- Found "+str(len(final_idlist))+
                  " galaxies satisfying all queries.")
    if writeFile:
        # Check if the file exists. 
        if (os.path.isfile(outFile)) and (not overwrite):
                print("\nFile already exists, please choose other filename or "+
                      "raise 'overwrite' flag.")

        if (not os.path.isfile(outFile)) or overwrite:
            f = open(outFile, 'w')
            [f.write(s) for s in str(list(final_idlist))]
            f.close()
    
    if tabulate:
        # Identify the SAMI master table, assumed to live in the Table directory
        g_table = hdf.getNode('/SAMI/'+version+'/Table/')
        master = g_table.SAMI_MASTER
        
        # Iterate over Master, locate CATID in final_idlist. 
        tabIndex = []
        for row in master:
            if row['CATID'] in final_idlist:
                tabIndex.append(row.nrow)

        madeTab = makeTable(master, tabIndex)
        hdf.close()
        return(madeTab)
    else:
        hdf.close()
        return(final_idlist)
        

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def test_contents(h5file):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
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



# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def search(h5file, query_item, query_value, verbose=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
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





# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def querycone(h5file, RAc, DECc, radius, version='', idfile='sami_query.lis', outFile=True, 
	      verbose=True, returnID=True, overwrite=True):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """ Performs a cone search and gives as output the idfile """
    """ Heavily based on the queryMaster function"""

    import sami.db.export as export

 
    # Get latest data version, if not supplied
    hdf0 = h5.File(h5file, 'r')
    version = export.getVersion(h5file, hdf0, version)
    hdf0.close()

    # Open-read h5file. 
    hdf = tables.openFile(h5file, 'r')

    # Optionally open an ascii file to write IDs returned by the query. 
    if returnID:
        # Check if the file exists, check overwrite flag:
        if os.path.isfile(idfile):
            if not overwrite:
                raise SystemExit("The nominated output file ('"+idfile+"') "+
                                 "already exists. Please raise the 'overwrite'"+
                                 " flag or enter a different filename. ")

    # Identify the SAMI master table, assumed to live in the Table directory
    g_table = hdf.getNode('/SAMI/'+version+'/Table/')
    master = g_table.SAMI_MASTER

    # Prepare some variables. 
    counter = 0
    idlist = []

    if outFile:
            f = open(idfile, 'w')

   
    for tables.row in master:
    			ra, dec = tables.row['RA'], tables.row['Dec']
			dist=sph_dist(ra,dec,RAc,DECc)
			 
			if (dist<radius):
			        name, z = tables.row['CATID'], tables.row['z_spec']
        			
				if verbose:
            				print("  Found SAMI galaxy %s at redshift z=%g at a distance of %f degrees" % (name, z, dist))
        			
				counter += 1
        			idlist.append(name)
        			
				if outFile: 
            				f.write(str(name)+'\n')
    
    print("\n  Found "+str(counter)+" galaxies satisfying the cone search: RA=%f, DEC=%f, radius=%f \n" % (RAc, DECc, radius))

    if outFile: 
        f.close()
    

    # Close h5 file 
    hdf.close()








# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
def sph_dist(ra1, dec1,ra2, dec2):
# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    """
    Compute the spherical distance between 2 pairs of coordinates
    using the Haversine formula
    Input coordinates are in decimal degrees
    Output: angular distance in decimal degrees
    """
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)
    ra2_rad = np.radians(ra2)
    dec2_rad = np.radians(dec2)

    d = np.sin((dec1_rad-dec2_rad)/2)**2;
    d += np.sin((ra1_rad-ra2_rad)/2)**2 * np.cos(dec1_rad)*np.cos(dec2_rad)

    return np.degrees(2*np.arcsin(np.sqrt(d)))



