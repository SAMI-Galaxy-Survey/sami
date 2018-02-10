"""
This file contains some functions used for recovering the dither pattern of a set of SAMI observations. 
These revolve around finding the centroid for each source in each IFU and then computing the best cordinate 
transformation to bring each RSS frame on the coordinate system of a reference one. 

Frankly, the code is a mess. The original implementation used IRAF (via
pyraf) to do the fitting, so all the data went via ASCII files. Later, the
functionality was replicated without using IRAF, and this is now the
default, but for reasons of technical debt the system of ASCII files was
retained.

The alignment works by measuring the position of each galaxy in each
observation. The positions are then compared between the first frame and
each subsequent frame, in turn. The difference between two frames is
modelled as an x/y shift and a radial stretch; the offsets from this model
are then saved to the FITS file.

Advantages:

* Bad fits to individual galaxy positions are rejected during the model
  fitting, so the procedure is pretty robust.
* The overall accuracy is generally very good - see Allen et al (2015) for
  quality assessment.

Disadvantages:

* The shift+stretch model used is not strictly correct; a stretch in the
  zenith direction would be better than a radial stretch. This can cause
  occasional inaccuracies, particularly if one galaxy is a long way from
  the others in a field.
* The pairwise comparison between frames does not use all the available
  information. Additionally, if a galaxy has a poor fit in the first
  frame, that galaxy will never contribute to the model. A better method
  would use all frames simultaneously.
* If an IFU includes a second object (e.g. a foreground star) it can throw
  off the fit. This normally isn't a problem for the alignment step itself
  but because the same fits are used to decide where the centre of the
  datacube should be, it can leave the star in the middle of the cube and
  the galaxy off to the side. It would be useful to allow the user to
  override the positioning in some way.

1) find_dither(RSSname,reference,centroid=True,inter=False,plot=False,remove_files=True,do_dar_correct=True,max_shift=350.0)

---"RSSname" should be a list containing the names of the RSS fits files to be aligned 
---"reference" should be the RSS fits file to be used as a reference, i.e., all other frames will be aligned to this one. 
---"centroid" should be set to True if you are running the task for the first time. It calls the module necessary to compute 
   the centroid of each IFU. Once this is done, if adjustments need to be made on the coordinate transformation, this 
   should be put to False as the centroid estimate takes 90% of the time needed to run this module. 
---"inter" should be False if no interactive check of the astrometric solution is necessary. If set to True, this allows 
   the user to manually play with each coordinate transformation, change fit orders, rejects ifus, etc. All the 
   interaction is done within the IRAF task geomap. For info on how to work interactively, have a look at this link. 
   http://iraf.net/irafhelp.php?val=immatch.geomap&help=Help+Page
---"plot" should be False if no check of the final dither patterns is necessary.
    If set True, the module produces a plot where the dither patterns are presented on the 2dF plate. 
    Note that the size of each pattern has been magnified by x200 to make it visible. The dashed circles show the size 
    of each fiber (with the same magnification) in order to give an idea of the size of the offset. 
---"remove_files" should be set True, unless the intermediate steps of the procedure need to be saved (see below for a description 
    of these intermediate steps). 
---"do_dar" should be True to apply the DAR correction before the centroid fitting is performed.
---"max_shift" the maximum initial shift between the centroid position of the same bundle in two different exposures.
   This makes sure that wrong centroid positions do not contribute to the estimate of the coordinate transformation. 
   
   
The dithern pattern, as well as several useful information on the best fitting coordinate transformation, are 
saved into a new header of each RSS frame.     

In case "remove_files" is set to False, this task will produce a lot of output in forms of ASCII file. 

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
                     
            

2) get_centroid(infile) 

You should not touch this one, as it is called automatically, in case needed. 


"""

from .. import utils
from ..observing import centroid
from ..utils.mc_adr import DARCorrector

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import astropy.io.fits as pf
from scipy.optimize import leastsq


HG_CHANGESET = utils.hg_changeset(__file__)

# Multiply by this value to convert arcseconds to microns
ARCSEC_TO_MICRON = 1000.0 / 15.2

ifus=[1,2,3,4,5,6,7,8,9,10,11,12,13]

def find_dither(RSSname,reference,centroid=True,inter=False,plot=False,remove_files=True,
                do_dar_correct=True,max_shift=350.0,use_iraf=False):
      
      
      ## For each RSSname call the get_cetroid module, computes centroid coordinates and stores them 
      ## into a txt file.  

      if centroid:
          for name in RSSname:
              get_centroid(name, do_dar_correct=do_dar_correct)
            
      nRSS=len(RSSname)
      
      
      ### For the reference frame extracts the position in micron for the central fibers of each IFU
      ### These positions are stored into "central_data" and saved int the file "file_centralfib" to 
      ### be used later by the IRAF tasks. 
      ### N.B. The huge assumption here is that the coordinates in micron of each central fibers on the focal 
      ### plane will remain exactly the same in the various exposures!  
      
      file_centralfib=''.join([reference.strip('.fits'), "ref_centrFIB.txt"]) 
      f=open(file_centralfib,'w')
      
    
      galname=[] #name of the target galaxy
      central_data = []
      object_order = {}
      i = 0
      for ifu in ifus:
      
            try:
                ifu_data=utils.IFU(reference, ifu, flag_name=False) 
            except IndexError:
                # This probably means it's a dead hexabundle, just skip it
                continue
            x=np.float(-1*ifu_data.x_microns[np.where(ifu_data.n==1)]) #x coordinate of central fiber (-1x is to have coordinates back on focal plane referenceO)
            y=np.float(ifu_data.y_microns[np.where(ifu_data.n==1)])    #y coordinate of central fiber
            s= str(x)+'  '+str(y)+'\n'
            f.write(s)
            central_data.append({'name': ifu_data.name,
                                 'ifu': ifu,
                                 'xcent': x,
                                 'ycent': y})
            object_order[ifu_data.name] = i
            galname.append(ifu_data.name)
            i += 1
      f.close() 

      n_ifu = len(central_data)
      xcent = np.array([data['xcent'] for data in central_data])
      ycent = np.array([data['ycent'] for data in central_data])
      
      
      file_ref=''.join([reference.strip('.fits'), "_centroid"]) # Name of the file containing the centroid coordinates for the RSS used as a reference
      
      xref=np.zeros(n_ifu)   #x coordinates of centroid in each ifu of the reference RSS
      yref=np.zeros(n_ifu)   #y coordinates of centroid in each ifu of the reference RSS
      ifuref=np.zeros(n_ifu, dtype=int)

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
         index = object_order[cols[0]]
         ifuref[index] = int(cols[1])
         xref[index] = float(cols[2])
         yref[index] = float(cols[3])
         RSScol.append(reference)
         ifscol.append(int(cols[1]))
         xshcol.append(0)
         yshcol.append(0) 
         galID.append(cols[0])

      RSSmatch=list(RSSname)
      RSSmatch.remove(reference)  ## Remove the reference RSS from the list of RSS files to align
     
      ## Run the loop where the dither solution is computed on the list of RSS files excluding the reference one.  
      
      ## Check if IRAF output db already exist. If yes, delete it
      
      file_geodb=''.join([reference.strip('.fits'), "_dbsolution"]) # File where the 2D solution of geomap is stored 
      if os.path.isfile(file_geodb):
                os.remove(file_geodb)

      results = []                

      if use_iraf:
          from pyraf import iraf
          # Some pyraf installations appear to require this line to load the images module
          iraf.images()
     
      for i in range(len(RSSmatch)):
         
             ## Define names of all the files used in this part of the module 
             
             name=RSSmatch[i].strip('.fits')
             file_centroid=''.join([name, "_centroid"]) # File containing the centroid coordinates. Produce by get_centroid
             file_geoin=''.join([name, "_mapin.txt"]) # This is the input file of geomap. It includes 4 columns having the x,y coordinates of the centroid in the inupt RSS and the ones in the reference RSS
             file_stats=''.join([name, "_fit"]) # File containing the detailed statistics for each fit. The content of each file is shown on the terminal.
             file_geoxy=''.join([name, "_xytrans"]) # Output of geoxytrans containing the coordinates of each central fiber in the coordinate system of the reference frame. 
            
             
             ## Check if IRAF output files already exist. If yes, delete them otherwise IRAF will crash!!!
             
             if os.path.isfile(file_stats):
                    os.remove(file_stats)
             if os.path.isfile(file_geoxy):
                    os.remove(file_geoxy)
             
             ## The next two loops simply create file_geoin
             
             # xin=[]
             # yin=[]
             ifu_good = np.zeros(n_ifu, dtype=int)
             xin = np.zeros(n_ifu)
             yin = np.zeros(n_ifu)
             for line in open(file_centroid):
                 cols=line.split()
                 # xin.append(float(cols[2]))
                 # yin.append(float(cols[3]))
                 index = object_order[cols[0]]
                 ifu_good[index] = int(cols[1])
                 xin[index] = float(cols[2])
                 yin[index] = float(cols[3])
  
            
             f=open(file_geoin, 'w')
             good = []
             for j in range(n_ifu):
                 # Immediately censor any point that's moved by more than max_shift
                 # (Default value is 350um, about 5")
                 if np.sqrt((xin[j] - xref[j])**2 + (yin[j] - yref[j])**2) > max_shift:
                     good.append(False)
                     continue
                 good.append(True)
                 s=str(xin[j])+' '+str(yin[j])+' '+str(xref[j])+' '+str(yref[j])+'\n' 
                 f.write(s)
  
             good = np.array(good)
             f.close()
    
             
             ## Run the IRAF geomap task 
             ## geomap is run iteratively. 
             ## The main condition is that the RMS in the final solution cannot be larger than 50 micron in the X and Y directions.
             ## The starting point is a 2 sigma clippping. If the rms is still to high, the sigma clipping is reduced by 0.1 each time 
             ## until it reaches 1.5. This is to take into account very rare cases in which a large number of deviant bundles may 
             ## significantly affect the 2 sigma procedure. This happens in less than 1% of the cases. 
             
             sigma_clip= 2.
             
             while True: 

                if use_iraf:
                    iraf.images.immatch.geomap(input=file_geoin,database=file_geodb,xmin="INDEF",ymin="INDEF",xmax="INDEF",ymax="INDEF",results=file_stats,xxorder=2.,yyorder=2.,xyorder=2.,yxorder=2.,
                                fitgeom='rscale', function='polynomial', interactive=inter, maxiter=10., reject=sigma_clip,verbose=0)
              
                    ## Show the statistics of each fit on the screen
                    ## The parameters to check are the RMS in X and Y and make sure that not more than 1-2 objects have INDEF on the residual values
                    ## INDEF values are present if the fiber has been rejected during the fit because too deviant from the best solution.  
                 
                    s='head -6'+' '+str(file_stats)
                    os.system(s)    

                    # Read back the RMS from one of IRAF's files
                    xrms, yrms, n_good, good = read_rms(file_stats)
                    rms = np.sqrt(xrms**2 + yrms**2)

                else:
                    coords_in = np.array([xin,yin]).T
                    coords_ref = np.array([xref,yref]).T
                    fit, good, n_good = fit_transform([0, 0, 0, 0], coords_in, coords_ref, sigma_clip=sigma_clip, good=good)
                    coords_fit = plate_scale_model(fit[0], coords_ref)
                    delta = coords_fit - coords_in
                    xrms = np.sqrt(np.mean(delta[good, 0]**2))
                    yrms = np.sqrt(np.mean(delta[good, 1]**2))
                    rms = np.sqrt(xrms**2 + yrms**2)
                        
                ## Check if the rms is lower than 50 micron. 
                ## If YES ==> best solution is OK
                ## If NO ==> re-run GEOMAP with a smaller sigma clipping (i.e., at every loop the sigma clipping goes down by 0.1)
                
                if ((rms<50.) | (sigma_clip<=1.5)):
                    break
                sigma_clip=sigma_clip - 0.1
                if os.path.exists(file_stats):
                    os.remove(file_stats)
        
             if use_iraf:
                 iraf.images.immatch.geoxytran(input=file_centralfib,output=file_geoxy, transform=file_geoin, database=file_geodb)
                 ## Append the results stored in file_geoxy on the RSScol,ifscol,xshcol,yshcol array so that they can be stored into a more user-friendly format
                 n=0
                 ## xshift and yshift are the same as xshcol, yshcol but for this frame only
                 xshift = np.zeros(n_ifu)
                 yshift = np.zeros(n_ifu)
                 for index, line in enumerate(open(file_geoxy)):
                     n=n+1 
                     RSScol.append(RSSmatch[i])
                     ifscol.append(n)
                     cols=line.split()
                     x=-1*np.subtract(np.float(cols[0]),xcent[index]) #the -1 is to go back to on-sky positions
                     y=np.subtract(np.float(cols[1]),ycent[index])
                     xshcol.append(x)
                     yshcol.append(y) 
                     xshift[index] = x
                     yshift[index] = y
                     galID.append(galname[n-1])
             else:
                 coords_model = plate_scale_model(fit[0], np.array([xcent, ycent]).T)
                 xshift = coords_model[:, 0] - xcent
                 yshift = -1*(coords_model[:, 1] - ycent)

             # Store the results in a handy dictionary
             results.append({'filename': RSSmatch[i],
                             'ifus': ifu_good,
                             'xin': xin,
                             'yin': yin,
                             'xref': xref,
                             'yref': yref,
                             'xshift': xshift,
                             'yshift': yshift,
                             'xrms': xrms,
                             'yrms': yrms,
                             'sigma': sigma_clip,
                             'n_good': n_good,
                             'good': good,
                             'reference': reference})

             if remove_files:
                 # Remove all the text files
                 for filename in [file_centroid, file_geoin, file_stats, file_geoxy]:
                     if os.path.exists(filename):
                         os.remove(filename)

      if remove_files:
          # Remove more text files
          for filename in [file_geodb, file_centralfib, ''.join([reference.strip('.fits'), "_centroid"])]:
              if os.path.exists(filename):
                os.remove(filename)

      # Re-calculate the reference X and Y values
      recalculate_ref(results, central_data)

      # Save results for frames other than the reference frame in the FITS header
      for result in results:
          save_results(result)
                
      # Save results for the reference frame in the FITS header
      ref_results_dict = {'filename': reference,
                          'ifus': ifuref,
                          'xin': xref,
                          'yin': yref,
                          'xref': xref,
                          'yref': yref,
                          'xshift': np.zeros(n_ifu),
                          'yshift': np.zeros(n_ifu),
                          'xrms': 0.0,
                          'yrms': 0.0,
                          'sigma': 0.0,
                          'n_good': n_ifu,
                          'good': [True for i in range(n_ifu)],
                          'reference': reference,
                          'xref_median': results[0]['xref_median'],
                          'yref_median': results[0]['yref_median']}
      save_results(ref_results_dict)
      
      ## Save final dither solution
      #file_results=''.join([reference.strip('.fits'), "_dither_solution.txt"])
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
          
          for i in range(len(ifus)):
                    
                
                
                
                deltax=-1*np.multiply(np.extract((np.array(ifscol))==(i+1),xshcol),200)   #scale each offset by x200 in order to make it visible in the plot - the -1 is to go back in focal plane coordinates            
                deltay=np.multiply(np.extract((np.array(ifscol))==(i+1),yshcol),200)   #scale each offset by x200 in order to make it visible in the plot
                    
                x=np.add(deltax,xcent[i])
                y=np.add(deltay,ycent[i])
                
                fiber=plt.Circle((xcent[i],ycent[i]),(200.*(105./2.)),fill=False, ls='dashed',color=cm.winter(1.*i/len(ifus)))  #circle corresponding to the size of each fiber x200
                plt.gca().add_patch(fiber)  #plot central fiber
                
                #plt.title('IFS'+str(i+1))
                plt.plot(x,y,color=cm.winter(1.*i/len(ifus)),lw=2.) # plot offset pattern 
                plt.annotate('IFS'+str(i+1), xy=(xcent[i],np.add(ycent[i],(200.*(105./2.)))), xycoords='data',xytext=None, textcoords='data', arrowprops=None,color=cm.winter(1.*i/len(ifus)))  #plot IFS id



def plate_scale_model(p,ref):
    """Return the transformed coordinates for a simple scale, translation, rotation model.
    
    Parameters in p:
        p[0]: Angle (in arcseconds)
        p[1]: x-offset
        p[2]: y-offset
        p[3]: (scale - 1)*1e5
    
    ref (N x 2 array): coordinate pairs of galaxy positions in reference observation.
    
    """
    
    scale = 1.0 + p[3] / 1e5
    x_offset = p[1]
    y_offset = p[2]
    theta = np.deg2rad(p[0] / 3600.0)
    
    xout = scale * ((ref[:,0] + x_offset) * np.cos(theta) \
                                  - (ref[:,1] + y_offset) * np.sin(theta))
    yout = scale * ((ref[:,1] + y_offset) * np.cos(theta) \
                                  + (ref[:,0] + x_offset) * np.sin(theta))

    return np.array([xout, yout]).T

def plate_scale_model_residuals(p,off,ref):
    """Return the residual of a simple scale, translation, rotation model. 
    
    Parameters in p:
        p[0]: Angle (in arcseconds)
        p[1]: x-offset
        p[2]: y-offset
        p[3]: (scale - 1)*1e5
    
    off (N x 2 array): coordinate pairs of galaxy position in offset observation
    ref (N x 2 array): coordinate pairs of galaxy positions in reference observation.
    
    """

    residual = off - plate_scale_model(p,ref)
    
    return np.sqrt(np.sum(residual**2, 1))

def fit_transform(p0, coords_in, coords_ref, sigma_clip=None, good=None):
    """Fit a coordinate transform to get from coords_ref to coords_in."""
    fit = (p0, 0)
    if good is None:
        good = np.ones(len(coords_in), bool)
    while True:
        n_good = np.sum(good)
        if n_good == 0:
            break
        fit = leastsq(plate_scale_model_residuals, fit[0], 
                      args=(coords_in[good, :], coords_ref[good, :]))
        if sigma_clip is not None:
            residual = plate_scale_model_residuals(fit[0], coords_in, coords_ref)
            rms = np.sqrt(np.mean(residual[good]**2))
            new_good = residual < (sigma_clip * rms)
            if np.all(new_good == good):
                # Converged!
                break
            good = new_good
        else:
            break
    return fit, good, n_good

def get_centroid(infile, do_dar_correct=True):

    ## Create name of the file where centroid coordinates are stored 
    
    out_txt=''.join([infile.strip('.fits'), "_centroid"])

    f=open(out_txt, 'w')
    
    ## Run centroid fit on each IFU
    
    for i, ifu in enumerate(ifus):

            try:
                ifu_data=utils.IFU(infile, ifu, flag_name=False)
            except IndexError:
                # Probably a broken hexabundle
                continue
                
            p_mic, data_mic, xlin_mic, ylin_mic, model_mic=centroid.centroid_fit(ifu_data.x_microns, ifu_data.y_microns,
                                                                                    ifu_data.data, circular=True)
            amplitude_mic, xout_mic, yout_mic, sig_mic, bias_mic=p_mic
            
            ##Get coordinates in micron. 
            ##Since centroid_fit currently inverts the x coordinates to have 'on-sky' coordinates, here 
            ##I need to re-multiply x coordinates by -1 to have them in the focal plane reference 
             
            x_out= -1*xout_mic
            y_out= yout_mic
            
            # Adjust for DAR, which means that "true" galaxy position is offset from observed position
            if do_dar_correct:
                dar_calc = DARCorrector(method='simple')
                dar_calc.setup_for_ifu(ifu_data)
                dar_calc.wavelength = np.mean(ifu_data.lambda_range)
                x_out -= dar_calc.dar_east * ARCSEC_TO_MICRON
                y_out += dar_calc.dar_north * ARCSEC_TO_MICRON

            # the data to write to file
            s=ifu_data.name+' '+str(ifu_data.ifu)+' '+str(x_out)+' '+str(y_out)+'\n'
                    
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
    good_col = pf.Column(name='GOOD', format='B', array=results['good'])
    hdu = pf.BinTableHDU.from_columns(
        pf.ColDefs([ifus_col, xin_col, yin_col, xref_col, 
                    yref_col, xshift_col, yshift_col, 
                    xref_median_col, yref_median_col,
                    good_col]))
    hdu.header['X_RMS'] = (results['xrms'], 'RMS of X_SHIFT')
    hdu.header['Y_RMS'] = (results['yrms'], 'RMS of Y_SHIFT')
    hdu.header['SIGMA'] = (results['sigma'], 'Sigma clipping used in the fit')
    hdu.header['N_GOOD'] = (results['n_good'], 'Number of galaxies used in fit')
    hdu.header['REF_FILE'] = (results['reference'], 'Reference filename')
    hdu.header['HGALIGN'] = (HG_CHANGESET, 'Hg changeset ID for alignment code')

    hdu.header['EXTNAME'] = 'ALIGNMENT'
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
        good = []
        line = 'a'
        while line:
            line = f.readline()
            if line.startswith('#     Xin and Yin fit rms:'):
                linesplit = line[:-1].split()
                xrms = float(linesplit[-2])
                yrms = float(linesplit[-1])
            elif not (line.startswith('#') or len(line) <= 1):
                good.append('INDEF' not in line)
    n_good = np.sum(good)
    return xrms, yrms, n_good, good


def recalculate_ref(results_list, central_data):
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
    for index in range(n_hexa):
        delta_x = xref_median[index] - central_data[index]['xcent']
        delta_y = yref_median[index] - central_data[index]['ycent']
        if np.sqrt(delta_x**2 + delta_y**2) > 490.0:
            # The median position is outside the hexabundle
            # Reject it and just use the centre instead
            xref_median[index] = central_data[index]['xcent']
            yref_median[index] = central_data[index]['ycent']
    for results in results_list:
        results['xref_median'] = xref_median
        results['yref_median'] = yref_median
    return
