"""
'utils.py', module within 'sami.db'

Description: Import codes for the SAMI Galaxy Survey data archive. 

Written: 08.10.2013, Iraklis Konstantopoulos. Based on 'db.database'.

Contact: iraklis@aao.gov.au

Updates: .-

Table of Contents (* to-be, **: under construction): 

.list_keys      List all the 'keys' (i.e. blocks and datasets) in an h5 file. 
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

""" THE END """
