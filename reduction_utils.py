'''
Created on 22nd May, 2017

@author: nscott

Helper functions for use when reducing sami data in bulk
'''
from __future__ import absolute_import, division, print_function, unicode_literals

import os,glob,shutil,re
import numpy as np

def remove_outdirs():
    # Remove the outdir directories created by 2dfDR.
    # This significantly reduces the size of reduced sami data
    
    counter = 0

    bias_dirs = glob.glob('*/reduced/bias/*/*/*_outdir')
    if len(bias_dirs) > 0:
        for dir in bias_dirs:
            shutil.rmtree(dir)
            counter = counter+1

    dark_dirs = glob.glob('*/reduced/dark/*/*/*/*_outdir')
    if len(dark_dirs) > 0:
        for dir in dark_dirs:
            shutil.rmtree(dir)
            counter = counter+1

    lflat_dirs = glob.glob('*/reduced/lflat/*/*/*_outdir')
    if len(lflat_dirs) > 0:
        for dir in lflat_dirs:
            shutil.rmtree(dir)
            counter = counter+1

    red_dirs = glob.glob('*/reduced/*/*/*/*/*/*_outdir')
    if len(red_dirs) > 0:
        for dir in red_dirs:
            shutil.rmtree(dir)
            counter = counter+1

    print('Number of outdir folders deleted: '+str(counter))

def locate_cross_fields():
    # Locate fields observed across multiple runs and return
    # the name of the folder containing that frame. Searches all
    # folders in the current directory.
    # ***DOES NOT LOCATE TARGETS OBSERVED ACROSS MULTIPLE FIELDS***


    # Method one - based on file system structure

    paths = glob.glob('*/reduced/*/*/Y*')
    fields = []
    plates = []
    runs = []
    
    valid_run_pattern = '\d{4}[_]\d{2}[_]\d{2}[-]\d{4}[_]\d{2}[_]\d{2}'

    for path in paths:
        tmp = path.split('/')
        if re.match(valid_run_pattern,tmp[0]):
            fields.append(tmp[-1])
            runs.append(tmp[0])
            plates.append(tmp[-2])

    fields = np.array(fields)
    plates = np.array(plates)
    runs = np.array(runs)

    unq_fields,unq_idx,unq_cnt = np.unique(fields,return_inverse=True,return_counts=True)
    cnt_mask = unq_cnt > 1
    dup_fields = unq_fields[cnt_mask]

    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_idx,cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx = np.argsort(unq_idx[idx_mask])
    dup_idx = np.split(idx_idx[srt_idx],np.cumsum(unq_cnt[cnt_mask])[:-1])

    crossed_fields = []
    crossed_runs = []

    for dup_id in dup_idx:
        if len(np.unique(runs[dup_id])) > 1:
            crossed_fields.append(fields[dup_id][0])
            crossed_runs.append(np.unique(runs[dup_id]))

    return crossed_fields,crossed_runs
