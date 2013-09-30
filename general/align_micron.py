try:
    from pyraf import iraf
except ImportError:
    print "pyraf not found! Can't do image alignment here."
import sami
import string
import sami.utils as utils
import numpy as np
import sami.observing.centroid
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import astropy.io.fits as pf


"""
This file contains some functions used for recovering the dither pattern of a set of SAMI observations. 
These revolve around finding the centroid for each source in each IFU and then computing the best cordinate transformation 
to bring each RSS frame on the coordinate system of a reference one. 

1) align_micron(inlist,reference,centroid=True,inter=False)

-inlist should be a ASCII file listing all the RSS exposures of the same plate.
-reference should be the RSS fits file to be used as a reference, i.e., all other frames will be aligned to this one. 
-centroid should be set to True if you are running the task for the first time. It calls the module necessary to compute 
 the centroid of each IFU. Once this is done, if adjustments need to be made on the coordinate transformation, this 
 should be put to False as the centroid estimate takes 90% of the time needed to run this module. 
-inter should be False if no interactive check of the astrometric solution is necessary. If set to True, this allows 
 the user to manually play with each coordinate transformation, change fit orders, rejects ifus, etc. All the 
 interaction is done within the IRAF task geomap. For info on how to work interactively, have a look at this link. 
 http://iraf.net/irafhelp.php?val=immatch.geomap&help=Help+Page

This task will produce a lot of output in forms of ASCII file, on terminal information and dither plot. 

- _dither_solution.txt  The main output of this module: contains for each RSS and ifus the x and y offset (in micron) relative to the reference RSS - the target galaxy name is also provided
- _centrFIB.txt     contains the coordinates in micron of the central fiber of each IFU 
- _dbsolution       contains the properties of the 2D function used to transform the coordinates system. 
                    This is the necessary input for the geoxytran part of this module
         
   For each RSS frame (excluding the reference one), the following files are produced
   
   - _mapin.txt        x,y centroid coordinates (in micron) for each ifu in the RSS frame and the reference one. Input to geomap
   - _xytrans          x,y coordinates in micron in the reference coordinate system of the central fiber in each ifu. Output of geoxytran
   - _fit          Results of the coordinate transformation. Including global rms and list of residuals for each ifu. The content 
                       of this file is also shown as output on the terminal while running this task like so:
               
               # Coordinate list: 12apr10042red_mapin.txt  Transform: 12apr10042red_mapin.txt
               #     Results file: 12apr10042red_fit
               # Coordinate mapping status
               #     X fit ok.  Y fit ok.
               #     Xin and Yin fit rms: 4.089513  2.86366
               # Coordinate mapping parameters
               #     Mean Xref and Yref: 27129.15  1583.458
               #     Mean Xin and Yin: 27095.31  1541.8610.7
               #     X and Y shift: -35.48904  -42.27051  (xin  yin)
               #     X and Y scale: 1.000134  1.000023  (xin / xref  yin / yref)
               #     X and Y axis rotation: 359.98999  359.96582  (degrees  degrees)

               # Input Coordinate Listing
               #     Column 1: X (reference) 
               #     Column 2: Y (reference)
               #     Column 3: X (input)
               #     Column 4: Y (input)
               #     Column 5: X (fit)
               #     Column 6: Y (fit)
               #     Column 7: X (residual)
               #     Column 8: Y (residual)

               -100670.7  52230.91  -100755.8  52170.43  -100756.1  52169.18     0.3125  1.242188
               -65639.65  9231.538  -65685.14  9177.396   -65686.4  9178.767   1.257813 -1.371094
               -6898.622   38964.6  -6961.655  38923.56  -6956.251  38924.12   -5.40332   -0.5625
                27066.43 -83685.31   27078.84 -83722.97   27085.75 -83724.05  -6.917969  1.085938
                26547.67  68438.74   26479.98  68401.43   26477.36  68405.06   2.615234 -3.632812
                29614.13  -105289.   29650.31 -105328.5   29646.49 -105327.9   3.820312  -0.59375
                32381.06 -48208.96   32381.26 -48246.88   32378.84 -48246.32   2.427734 -0.558594
                52606.15  69921.53   52537.65  69894.28   52538.56  69891.74  -0.914062  2.539062
                86881.55  70617.78    86816.3  70591.74   86817.25  70592.08  -0.945312 -0.335938
                59392.05 -6959.146   59370.64 -6988.889   59366.29 -6992.225   4.347656  3.335449
                75589.95  31111.39    75549.6  31086.51   75544.55  31081.93   5.046875  4.576172
                89784.94  9250.244   89745.29  9214.896   89750.94  9220.627  -5.648438 -5.730469
                46024.07 -85039.38   46031.81 -85128.83  INDEF     INDEF      INDEF INDEF
                     
               Two things need to be checked from this output while running this task
                    1) The rms (5th row from the top)
                2) How many fibers are given as INDEF in the residual columns. These are fibers 
                   that have been rejected because larger (>2.5sigma) outliers in the best solution.
                   If the numbers of outliers is too larger, it means there is a big problem with the data.
                   
            
Finally, the module produces a plot where the dither patterns are presented on the 2dF plate. 
Note that the size of each pattern has been magnified by x200 to make it visible. The dashed circles show the size 
of each fiber (with the same magnification) in order to give an idea of the size of the offset. 




2) get_centroid(infile) 

You should not touch this one, as it is called automatically, in case needed. 


"""

HG_CHANGESET = utils.hg_changeset(__file__)

ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]

def find_dither(RSSname,reference,centroid=True,inter=False,plot=False,remove_files=True):
      
      
      ## For each file in inlist call the get_cetroid module, computes centroid coordinates and stores them 
      ## into a txt file.  

      if centroid:
          for name in RSSname:
              get_centroid(name)
            
      nRSS=len(RSSname)
      
      
      ### For the reference frame extracts the position in micron for the central fibers of each IFU
      ### N.B. The huge assumption here is that the coordinates in micron of each central fibers on the focal 
      ### plane will remain exactly the same in the various exposures!  
      
      file_centralfib=string.join([string.strip(reference,'.fits'), "ref_centrFIB.txt"],'') 
      f=open(file_centralfib,'w')
      
      
      xcent=[] #x coordinates of central fiber of each bundle
      ycent=[]  #y coordinates of central fiber of each bundle
      galname=[] #name of the target galaxy
      ifu_good=[]
      for i, ifu in enumerate(ifus):
      
            try:
                ifu_data=utils.IFU(reference, ifu, flag_name=False) 
            except IndexError:
                # This probably means it's a dead hexabundle, just skip it
                continue
            x=np.float(-1*ifu_data.x_microns[np.where(ifu_data.n==1)]) #x coordinate of central fiber (-1x is to have coordinates back on focal plane referenceO)
            y=np.float(ifu_data.y_microns[np.where(ifu_data.n==1)]) #y coordinate of central fiber
            s= str(x)+'  '+str(y)+'\n'
            f.write(s)
            xcent.append(x)
            ycent.append(y)
            ifu_good.append(ifu)
            galname.append(ifu_data.name)
      f.close() 
      
      
      file_ref=string.join([string.strip(reference,'.fits'), "_centroid"],'')  # Name of the file containing the centroid coordinates for the RSS used as a reference
      
      xref=[]   #x coordinates of centroid in each ifu of the reference RSS
      yref=[]   #y coordinates of centroid in each ifu of the reference RSS
      
      ### Starting from the reference RSS, will start filling the next 4 arrays which will become the main output of this procedure
      ### Obviously for the reference RSS xshcol and yshcol are 0 by definition.
      
      RSScol=[]  # RSS name
      galID=[]   #galaxy ID
      ifscol=[]  # ifs
      xshcol=[]  # xshift in micron
      yshcol=[]  # yshif in micron
      n=0
      for line in open(file_ref):
         n=n+1
         cols=line.split()
         xref.append(float(cols[0]))
         yref.append(float(cols[1]))
         RSScol.append(reference)
         ifscol.append(n)
         xshcol.append(0)
         yshcol.append(0) 
         galID.append(galname[n-1])



      RSSmatch=RSSname
      RSSmatch.remove(reference)  ## Remove the reference RSS from the list of RSS files to align
     
      ## Run the loop where the dither solution is computed on the list of RSS files excluding the reference one.  
      
      ## Check if IRAF output db already exist. If yes, delete it
      
      file_geodb=string.join([string.strip(reference,'.fits'), "_dbsolution"],'') # File where the 2D solution of geomap is stored 
      if os.path.isfile(file_geodb):
                os.remove(file_geodb)

      results = []                
     
      for i in xrange(len(RSSmatch)):
         
             ## Define names of all the files used in this part of the module 
             
             name=string.strip(RSSmatch[i],'.fits')
             file_centroid=string.join([name, "_centroid"],'') # File containing the centroid coordinates. Produce by get_centroid
             file_geoin=string.join([name, "_mapin.txt"],'') # This is the input file of geomap. It includes 4 columns having the x,y coordinates of the centroid in the inupt RSS and the ones in the reference RSS
             file_stats=string.join([name, "_fit"],'') # File containing the detailed statistics for each fit. The content of each file is shown on the terminal.
             file_geoxy=string.join([name, "_xytrans"],'') # Output of geoxytrans containing the coordinates of each central fiber in the coordinate system of the reference frame. 
            
             
             ## Check if IRAF output files already exist. If yes, delete them otherwise IRAF will crash!!!
             
             if os.path.isfile(file_stats):
                    os.remove(file_stats)
             if os.path.isfile(file_geoxy):
                    os.remove(file_geoxy)
             
             ## The next two loops simply create file_geoin
             
             xin=[]
             yin=[]
             for line in open(file_centroid):
                 cols=line.split()
                 xin.append(float(cols[0]))
                 yin.append(float(cols[1]))
  
            
             f=open(file_geoin, 'w')
             for j in xrange(len(xin)):
                 s=str(xin[j])+' '+str(yin[j])+' '+str(xref[j])+' '+str(yref[j])+'\n' 
                 f.write(s)
  
             f.close()
    
             
             ## Run the IRAF geomap task 
            
             iraf.images.immatch.geomap(input=file_geoin,database=file_geodb,xmin="INDEF",ymin="INDEF",xmax="INDEF",ymax="INDEF",results=file_stats,xxorder=2.,yyorder=2.,xyorder=2.,yxorder=2.,
                         fitgeom='rscale', function='polynomial', interactive=inter, maxiter=10., reject=2.,verbose=0)
                         
             
             ## Show the statistics of each fit on the screen
             ## The parameters to check are the RMS in X and Y and make sure that not more than 1-2 objects have INDEF on the residual values
             ## INDEF values are present if the fiber has been rejected during the fit because too deviant from the best solution.  
             
             s='head -6'+' '+str(file_stats)
             os.system(s)            
        
             iraf.images.immatch.geoxytran(input=file_centralfib,output=file_geoxy, transform=file_geoin, database=file_geodb)
                 
             
             
             ## Append the results stored in file_geoxy on the RSScol,ifscol,xshcol,yshcol array so that they can be stored into a more user-friendly format
             n=0
             ## xshift and yshift are the same as xshcol, yshcol but for this frame only
             xshift = []
             yshift = []
             for line in open(file_geoxy):
                 n=n+1 
                 RSScol.append(RSSmatch[i])
                 ifscol.append(n)
                 cols=line.split()
                 x=-1*np.subtract(np.float(cols[0]),xcent[n-1]) #the -1 is to go back to on-sky positions
                 y=np.subtract(np.float(cols[1]),ycent[n-1])
                 xshcol.append(x)
                 yshcol.append(y) 
                 xshift.append(x)
                 yshift.append(y)
                 galID.append(galname[n-1])

             # Read back the RMS from one of IRAF's files
             xrms, yrms, n_good = read_rms(file_stats)

             # Store the results in a handy dictionary
             results.append({'filename': RSSmatch[i],
                             'ifus': np.array(ifu_good),
                             'xin': np.array(xin),
                             'yin': np.array(yin),
                             'xref': np.array(xref),
                             'yref': np.array(yref),
                             'xshift': np.array(xshift),
                             'yshift': np.array(yshift),
                             'xrms': xrms,
                             'yrms': yrms,
                             'n_good': n_good,
                             'reference': reference})

             if remove_files:
                 # Remove all the text files
                 os.remove(file_centroid)
                 os.remove(file_geoin)
                 os.remove(file_stats)
                 os.remove(file_geoxy)

      if remove_files:
          # Remove more text files
          os.remove(file_geodb)
          os.remove(file_centralfib)
          os.remove(string.join([string.strip(reference,'.fits'), "_centroid"],''))

      # Re-calculate the reference X and Y values
      recalculate_ref(results)

      # Save results for frames other than the reference frame in the FITS header
      for result in results:
          save_results(result)
                
      # Save results for the reference frame in the FITS header
      ref_results_dict = {'filename': reference,
                          'ifus': np.array(ifu_good),
                          'xin': np.array(xref),
                          'yin': np.array(yref),
                          'xref': np.array(xref),
                          'yref': np.array(yref),
                          'xshift': np.zeros(len(ifu_good)),
                          'yshift': np.zeros(len(ifu_good)),
                          'xrms': 0.0,
                          'yrms': 0.0,
                          'n_good': len(ifu_good),
                          'reference': reference,
                          'xref_median': results[0]['xref_median'],
                          'yref_median': results[0]['yref_median']}
      save_results(ref_results_dict)
      
      ## Save final dither solution
      #file_results=string.join([string.strip(reference,'.fits'), "_dither_solution.txt"],'')    
      #results=np.column_stack((RSScol,ifscol,galID,xshcol,yshcol))
      #np.savetxt(file_results, results, fmt='%15s')

      
      if plot:
          ## Plot the final dither patterns
          plt.ion()

          plt.subplot(1,1,1)
          
          ## Create 2dF FoV
          plt.rc("font", size=12)
          plt.ylim((-130000,130000))
          plt.xlim((-130000,130000))
          fov=plt.Circle((0,0),125000,fill=False,lw=0.5)
          plt.gca().add_patch(fov)
          
          for i in xrange(len(ifus)):
                    
                
                
                
                deltax=-1*np.multiply(np.extract((np.array(ifscol))==(i+1),xshcol),200)   #scale each offset by x200 in order to make it visible in the plot - the -1 is to go back in focal plane coordinates            
                deltay=np.multiply(np.extract((np.array(ifscol))==(i+1),yshcol),200)   #scale each offset by x200 in order to make it visible in the plot
                    
                x=np.add(deltax,xcent[i])
                y=np.add(deltay,ycent[i])
                
                fiber=plt.Circle((xcent[i],ycent[i]),(200.*(105./2.)),fill=False, ls='dashed',color=cm.winter(1.*i/len(ifus)))  #circle corresponding to the size of each fiber x200
                plt.gca().add_patch(fiber)  #plot central fiber
                
                #plt.title('IFS'+str(i+1))
                plt.plot(x,y,color=cm.winter(1.*i/len(ifus)),lw=2.) # plot offset pattern 
                plt.annotate('IFS'+str(i+1), xy=(xcent[i],np.add(ycent[i],(200.*(105./2.)))), xycoords='data',xytext=None, textcoords='data', arrowprops=None,color=cm.winter(1.*i/len(ifus)))  #plot IFS id



def get_centroid(infile):

    ## Create name of the file where centroid coordinates are stored 
    
    out_txt=string.join([string.strip(infile,'.fits'), "_centroid"],'')

    f=open(out_txt, 'w')
    
    ## Run centroid fit on each IFU
    
    for i, ifu in enumerate(ifus):

            try:
                ifu_data=utils.IFU(infile, ifu, flag_name=False)
            except IndexError:
                # Probably a broken hexabundle
                continue
                
            p_mic, data_mic, xlin_mic, ylin_mic, model_mic=sami.observing.centroid.centroid_fit(ifu_data.x_microns, ifu_data.y_microns,
                                                                                    ifu_data.data, circular=True)
            amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic=p_mic
            
            ##Get coordinates in micron. 
            ##Since centroid_fit currently inverts the x coordinates to have 'on-sky' coordinates, here 
            ##I need to re-multiply x coordinates by -1 to have them in the focal plane reference 
             
            x_out= -1*xout_mic
            y_out= yout_mic
            
            s=str(x_out)+' '+str(y_out)+'\n' # the data to write to file
                    
            f.write(s)
            
    f.close() # close the output file


def save_results(results):
    """Save the results in a new FITS header."""
    # Make the binary table HDU
    ifus_col = pf.Column(name='PROBENUM', format='I', array=results['ifus'])
    xin_col = pf.Column(name='X_CEN', format='E', array=results['xin'])
    yin_col = pf.Column(name='Y_CEN', format='E', array=results['yin'])
    xref_col = pf.Column(name='X_REF', format='E', array=results['xref'])
    yref_col = pf.Column(name='Y_REF', format='E', array=results['yref'])
    xshift_col = pf.Column(name='X_SHIFT', format='E', array=results['xshift'])
    yshift_col = pf.Column(name='Y_SHIFT', format='E', array=results['yshift'])
    xref_median_col = pf.Column(name='X_REFMED', format='E', 
                                array=results['xref_median'])
    yref_median_col = pf.Column(name='Y_REFMED', format='E', 
                                array=results['yref_median'])
    hdu = pf.new_table(pf.ColDefs([ifus_col, xin_col, yin_col, xref_col, 
                                   yref_col, xshift_col, yshift_col, 
                                   xref_median_col, yref_median_col]))
    hdu.header['X_RMS'] = (results['xrms'], 'RMS of X_SHIFT')
    hdu.header['Y_RMS'] = (results['yrms'], 'RMS of Y_SHIFT')
    hdu.header['N_GOOD'] = (results['n_good'], 'Number of galaxies used in fit')
    hdu.header['REF_FILE'] = (results['reference'], 'Reference filename')
    hdu.header['HGALIGN'] = (HG_CHANGESET, 'Hg changeset ID for alignment code')
    hdu.update_ext_name('ALIGNMENT')
    # Open up the file for editing
    hdulist = pf.open(results['filename'], 'update')
    # Remove the existing HDU, if it's there
    try:
        del hdulist['ALIGNMENT']
    except KeyError:
        pass
    hdulist.append(hdu)
    hdulist.close()
    return
    

def read_rms(filename):
    """Read back the RMS from one of IRAF's results files."""
    with open(filename) as f:
        n_bad = 0
        line = 'a'
        while line:
            line = f.readline()
            if line.startswith('#     Xin and Yin fit rms:'):
                linesplit = line[:-1].split()
                xrms = float(linesplit[-2])
                yrms = float(linesplit[-1])
            elif 'INDEF' in line:
                n_bad += 1
    n_good = len(ifus) - n_bad
    return xrms, yrms, n_good


def recalculate_ref(results_list):
    """Re-calculate the reference coordinates, taking the median."""
    n_obs = len(results_list)
    n_hexa = len(results_list[0]['xshift'])
    xref = np.zeros((n_hexa, n_obs))
    yref = np.zeros((n_hexa, n_obs))
    for index, results in enumerate(results_list):
        xref[:, index] = results['xin'] - results['xshift']
        yref[:, index] = results['yin'] + results['yshift']
    xref_median = np.median(xref, axis=1)
    yref_median = np.median(yref, axis=1)
    for results in results_list:
        results['xref_median'] = xref_median
        results['yref_median'] = yref_median
    return
