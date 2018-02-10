"""
Code for binning data in SAMI datacubes. Typically accessed via
bin_cube_pair(), which will calculate and save bins for a pair of cubes.
See sami.manager.bin_cubes_pair() for an example of calling this function.
"""

# -----------------------------------------------------------------------------
# NB Not all input keywords are fully implemented yet!!
# -----------------------------------------------------------------------------
#
# User calling sequence for Voronoi binning:
#       bin_mask = sami_binning.adaptive_bin_sami(sami_cube_file_name,targetSN=??)
#
#   bin_mask is a SAMI image (same spatial dimensions of the input cube) where the value of each
#   spaxel corresponds to the bin it was assigned to. NaNs in the input cube are assigned to bin 0.
#
#   sami_cube_file_name should be the full path name to a SAMI cube with covariance information
#   Default targetSN is 10. Can be set to any value, though will error is targetSN requires no
#   binning of the cube or is too high to be achieved by binning all spaxels (I think)
#
# Internal calling sequence:
#       bin = veronoi_2d_binning_wcovar.bin2d(xin,yin,datain,noise=noisein,covar=covarin,targetSN=10.0)
#       bin.bin_voronoi()
# -----------------------------------------------------------------------------
#
# User calling sequence for prescribed binning:
#       bin_mask = sami_binning.prescribed_bin_sami(cube_file_name,sectors={1,4,8,16},radial=??
#                   log={True,False},xmed=??,ymed=??,pa=??,eps=??)
#
# -----------------------------------------------------------------------------

import astropy.io.fits as pf
import numpy as np
from numpy import nanmedian
from scipy.ndimage.filters import median_filter
from scipy.ndimage.measurements import label
from . import voronoi_2d_binning_wcovar

def bin_cube_pair(path_blue, path_red, name=None, **kwargs):
    """Calculate bins, do binning and save results for a pair of cubes."""
    hdulist_blue = pf.open(path_blue, 'update')
    hdulist_red = pf.open(path_red, 'update')
    bin_mask = return_bin_mask(hdulist_blue, **kwargs)
    bin_and_save(hdulist_blue, bin_mask, name=name)
    bin_and_save(hdulist_red, bin_mask, name=name)
    hdulist_blue.close()
    hdulist_red.close()

def bin_and_save(hdulist, bin_mask, name=None):
    """Do binning and save results for an HDUList."""
    # TODO: Check if the extensions already exist. In most cases you would
    # want to either overwrite or just return without doing anything, but
    # occasionally the user might want to append duplicate extensions (the
    # current behaviour). (JTA 14/9/2015)

    # Default behaviour here is now to overwrite extensions. If extension exists
    # and overwrite=False this should have been caught by manager.bin_cubes()

    binned_cube, binned_var = bin_cube(hdulist, bin_mask)
    if name is None:
        suffix = ''
    else:
        suffix = '_' + name

    duplicate_extensions = []
    for ext in hdulist:
        if ((ext.name == 'BIN_MASK'+suffix.upper()) 
                or (ext.name == 'BINNED_FLUX'+suffix.upper())
                or (ext.name == 'BINNED_VARIANCE'+suffix.upper())):
            duplicate_extensions.append(ext.name)

    for ext in duplicate_extensions:
        del hdulist[ext]

    hdu_mask = pf.ImageHDU(bin_mask, name='BIN_MASK'+suffix)
    hdu_flux = pf.ImageHDU(binned_cube, name='BINNED_FLUX'+suffix)
    hdu_var = pf.ImageHDU(binned_var, name='BINNED_VARIANCE'+suffix)
    hdulist.append(hdu_mask)
    hdulist.append(hdu_flux)
    hdulist.append(hdu_var)
    hdulist.flush()
    return

def return_bin_mask(hdu, mode='adaptive', targetSN=10, minSN=None, sectors=8,radial=5,log=False):
    
    if mode == 'adaptive':
        bin_mask = adaptive_bin_sami(hdu,targetSN=targetSN, minSN=minSN)
        
    elif mode == 'prescriptive':
        bin_mask = prescribed_bin_sami(hdu,sectors=sectors,radial=radial,log=log)
        
    else:
        raise Exception('Invalid binning mode requested')

    return bin_mask

def bin_cube(hdu,bin_mask):
    #Produce a SAMI cube where each spaxel contains the
    #spectrum of the bin it is associated with
    
    # The variance in output correctly accounts for covariance, but the remaining covariance
    # between bins is not tracked (this may change if enough people request it)

    # Bin spectra and assign to spaxels

    cube = hdu[0].data
    var = hdu[1].data
    weight = hdu[2].data
    covar = reconstruct_covariance(hdu[3].data,hdu[3].header,n_wave=cube.shape[0])

    weighted_cube = cube*weight
    weighted_var = var*weight*weight

    binned_cube = np.zeros(np.shape(cube))
    binned_var = np.zeros(np.shape(cube))

    n_bins = int(np.max(bin_mask))

    for i in range(n_bins):
        spaxel_coords = np.array(np.where(bin_mask == i+1))
        binned_spectrum = np.nansum(cube[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)/len(spaxel_coords[0])
        binned_weighted_spectrum = np.nansum(weighted_cube[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)/len(spaxel_coords[0])
        binned_weight = np.nansum(weight[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)
        binned_weight2 = np.nansum(weight[:,spaxel_coords[0,:],spaxel_coords[1,:]]**2,axis=1)
        temp = np.tile(np.reshape(binned_spectrum,(len(binned_spectrum),1)),len(spaxel_coords[0,:]))
        binned_cube[:,spaxel_coords[0,:],spaxel_coords[1,:]] = temp
        binned_weighted_variance = np.nansum(weighted_var[:,spaxel_coords[0,:],spaxel_coords[1,:]]*
                                    np.nansum(np.nansum(covar[:,:,:,spaxel_coords[0,:],spaxel_coords[1,:]],
                                    axis=1)/2.0,axis=1),axis=1)
        binned_variance = binned_weighted_variance*((binned_spectrum/binned_weighted_spectrum)**2)/(len(spaxel_coords[0])**2)
        binned_var[:,spaxel_coords[0,:],spaxel_coords[1,:]] = np.tile(
                                np.reshape(binned_variance,(len(binned_variance),1)),len(spaxel_coords[0,:]))

    return binned_cube,binned_var

def reconstruct_covariance(covar_array_red,covar_header,n_wave=2048):
    #Reconstruct the full covariance array from the reduced covariance
    #information stored in a standard cube
    
    if covar_header['COVARMOD'] != 'optimal':
        raise Exception('This cube does not contain covariance information in the optimal format')
    
    n_spax = covar_array_red.shape[3]
    n_grid = covar_array_red.shape[1]

    #Create an empty full covariance cube
    covar_array_full = np.zeros([n_wave,n_grid,n_grid,n_spax,n_spax])
    
    #Fill full cube with values from reduced array
    n_covar = covar_header['COVAR_N']
    for i in range(n_covar):
        slice = covar_header['COVARLOC_'+str(i+1)]
        covar_array_full[slice,:,:,:,:] = covar_array_red[i,:,:,:,:]

    # Don't propagate NaNs
    covar_array_full[np.isfinite(covar_array_full) == False] = 0.0
    
    #Fill missing values (needs to be improved with median/interpolation filling)
    half_spax = (n_spax - 1) / 2
    half_covar = (n_grid - 1) / 2
    lowest_point = covar_header['COVARLOC_1']
    # lowest_point = np.min(np.where(
    #     (covar_array_full[1:,half_covar,half_covar,half_spax,half_spax] != 0.0) &
    #     (np.isfinite(covar_array_full[1:,half_covar,half_covar,half_spax,half_spax])))[0]) + 1
    for i in range(n_wave):
        if np.sum(np.abs(covar_array_full[i,:,:,:,:])) == 0:
            if i < lowest_point:
                covar_array_full[i,:,:,:,:] = covar_array_full[lowest_point,:,:,:,:]
            else:
                covar_array_full[i,:,:,:,:] = covar_array_full[i-1,:,:,:,:]
                                   
    return covar_array_full

def adaptive_bin_sami(hdu, targetSN=10.0, minSN=None):
    """
        Wrapper for handling SAMI data. Returns an 'image'
        where each spaxels' value indicates the bin it belongs
        to
        """
    
    # Open a SAMI cube
    data = hdu['PRIMARY'].data
    var = hdu['VARIANCE'].data
    wei = hdu['WEIGHT'].data
    covar = hdu['COVAR'].data
    covar_header = hdu['COVAR'].header
    
    # Create signal and noise images
    image = nanmedian(data*wei,axis=0)
    var_image = nanmedian(var*(wei*wei),axis=0)
    
    # Construct x and y index arrays
    s = np.shape(image)
    inds = np.indices(s)
    x = inds[0].ravel()
    y = inds[1].ravel()
    
    # Flatten arrays for input and mask nans
    signal = image.ravel()
    noise = np.sqrt(var_image.ravel())

    # Check if there is a minimum spaxel S/N to be included 
    if minSN == None:
        goodpixels = np.where((np.isfinite(signal) == True) &
                          (np.isfinite(noise) == True))

    else:
        goodpixels = np.where((np.isfinite(signal) == True) &
                        (np.isfinite(noise) == True) & (signal/noise > minSN))

    signal = signal[goodpixels]
    noise = noise[goodpixels]
                          
                          
    # Reconstruct then flatten covariance cube
    covar = reconstruct_covariance(covar,covar_header,n_wave=data.shape[0])
    covar_image = nanmedian(covar,axis=0)
    covar_matrix = np.rollaxis(covar_image[:,:,x[goodpixels[0]],y[goodpixels[0]]],2)
                          
    x = x[goodpixels]
    y = y[goodpixels]
                          
    # Initialise voronoi binning class
    bin = voronoi_2d_binning_wcovar.bin2D(x,y,signal,noise=noise,covar=covar_matrix,targetSN=targetSN)
                          
    # Perform the Voronoi tesselation
    if bin.lowSN == False:
        bin.bin_voronoi()

        # Reconstruct the bin mask image
                          
        bin_mask_image = np.zeros(s)
        n_bins = len(bin.listbins)
        for i in range(n_bins):
            bin_mask_image[x[bin.listbins[i]],y[bin.listbins[i]]] = i+1
    else:
        bin_mask_image = np.zeros(s)
        bin_mask_image[x,y] = 1

    return bin_mask_image

def second_moments(image,ind):
    img1 = image[ind]
    img1[np.where(img1 < 0.0)] = 0.0
    s = np.shape(image)
    
    #Compute coefficients of the moment of inertia tensor
    i = np.sum(img1)
    xmed = np.sum(img1*ind[1])/i
    ymed = np.sum(img1*ind[0])/i
    x2 = np.sum(img1*ind[1]**2)/i - xmed**2
    y2 = np.sum(img1*ind[0]**2)/i - ymed**2
    xy = np.sum(img1*ind[0]*ind[1])/i - xmed*ymed
    
    #Diagonalise the moment of inertia tensor
    theta = np.degrees(np.arctan2(2.0*xy,x2-y2)/2.0)+90.0
    a2 = (x2+y2)/2.0 + np.sqrt(((x2-y2)/2.0)**2 + xy**2)
    b2 = (x2+y2)/2.0 - np.sqrt(((x2-y2)/2.0)**2 + xy**2)
    eps = 1.0 - np.sqrt(b2/a2)
    maj = np.sqrt(a2)
    
    #Return the position of the peak intensity
    n = 20
    xmed1 = np.round(xmed,decimals=0)
    ymed1 = np.round(ymed,decimals=0)
    if (xmed1 - n > 0) & (xmed1+n < s[1]) & (ymed1 - n > 0) & (ymed1+n < s[0]):
        tmp = np.max(image[ymed1-n:ymed1+n+1,xmed1-n:xmed1+n+1])
        j = np.where(image == tmp)
        xpeak=j[1]
        ypeak=j[0]
    else:
        tmp = np.max(image)
        j = np.where(image == tmp)
        xpeak=j[1]
        ypeak=j[0]
    
    return maj,eps,theta,xpeak,ypeak,xmed,ymed

def find_galaxy(image,nblob=1,fraction=0.1,quiet=True):
    #Based on Michele Cappellari's IDL routine find_galaxy.pro
    #Derives basic galaxy parameters using the weighted 2nd moments
    #of the luminosity distribution
    #Makes use of 2nd_moments
    
    s = np.shape(image)
    a = median_filter(image,size=5,mode='constant')
    j = a.ravel().argsort()
    level = a.ravel()[j[np.size(a)*(1.0 - fraction)]]
    j = np.where(a > level)
    
    a = np.zeros(s)
    a[j] = 1
    a,n_regions = label(a)
    if (n_regions > 1) & (nblob <= n_regions):
        bins = range(0,n_regions+2)
        h,bins = np.histogram(a,bins=bins)
        gal = h.argsort()[-1*(nblob+1)]
    else: gal = 1
    ind = np.where(a == gal)
    
    maj,eps,pa,xpeak,ypeak,xmed,ymed = second_moments(image,ind)
    
    if quiet != True:
        print('Pixels used: %i' % len(ind[0]))
        print('Peak (x,y): %i %i' % (xpeak,ypeak))
        print('Mean (x,y): %f %f' % (xmed,ymed))
        print('Theta (deg): %f' % pa)
        print('Eps: %f' % eps)
        print('Sigma along major axis (pixels): %f' % maj)

    n_blobs = np.max(a)
    
    return maj,eps,pa,xpeak[0],ypeak[0],xmed,ymed,n_blobs

def prescribed_bin_sami(hdu,sectors=8,radial=5,log=False,
                        xmed='',ymed='',pa='',eps=''):

#Allocate spaxels to a bin, based on the standard SAMI binning scheme
#Returns a 50x50 array where each element contains the bin number to which
#a given spaxel is allocated. Bin ids spiral out from the centre.

#Users can select number of sectors (1, 4, 8, maybe 16?), number of radial bins and
#whether the radial progression is linear or logarithmic

#Users can provide centroid, pa and ellipticity information manually.
#The PA should be in degrees.

    cube = hdu['PRIMARY'].data
    n_spax = cube.shape[1]

    #Check if all the pa,eps,xc,yc information has been supplied. If not fill in missing info
    if (xmed == '') or (ymed == '') or (eps == '') or (pa == ''):
        image = np.nanmedian(cube,axis=0)
        image0 = np.copy(image)
        image0[np.isfinite(image) == False] = -1
        try:
            maj0,eps0,pa0,xpeak0,ypeak0,xmed0,ymed0,n_blobs = find_galaxy(image0,quiet=True,fraction=0.05)
        except:
            eps0,pa0,xmed0,ymed0 = 0.0,0.0,0.5*n_spax,0.5*n_spax
    n = 1

    while ((np.abs(xmed0 - 0.5*n_spax) > 3) or (np.abs(ymed0 - 0.5*n_spax) > 3)) and (n <= n_blobs) :
        try:
            maj0,eps0,pa0,xpeak0,ypeak0,xmed0,ymed0,junk = find_galaxy(image0,quiet=True,fraction=0.05,nblob=n)
            n+=1
        except:
            eps0,pa0,xmed0,ymed0 = 0.0,0.0,0.5*n_spax,0.5*n_spax
            n = n_blobs+1

    if xmed == '':
        xmed = xmed0
    if ymed == '':
        ymed = ymed0
    if eps == '':
        eps = eps0
    if pa == '':
        pa = pa0

    pa_rad = np.radians(pa)

    #Define the angular bins
    if sectors == 4:
        angles = [0.0,90.,180.,270.,360.]
    elif sectors == 8:
        ratio = 1. - eps
        sub_angle = np.degrees(np.arctan(ratio))
        angles = [0.0,sub_angle,90.,180.-sub_angle,180.,
              180.+sub_angle,270.,360.-sub_angle,360.]
    elif sectors == 16:
        raise Exception('Not yet implemented. Sorry!')
    elif sectors == 1:
        angles = [0.0,360.0]
    else:
        raise Exception('Pick either 1, 4, 8 or 16 sectors')

    #Shift and rotate the spaxel coordinates so that the galaxy centre
    #is at (0,0) and the major axis is aligned with the x axis
    spax_pos = np.indices(np.shape(image),dtype=np.float)
    spax_pos[0,:] = spax_pos[0,:] - round(xmed)
    spax_pos[1,:] = spax_pos[1,:] - round(ymed)
    spax_pos_rot = np.zeros(np.shape(spax_pos))
    spax_pos_rot[0,:] = spax_pos[0,:,:]*np.cos(pa_rad) - spax_pos[1,:,:]*np.sin(pa_rad)
    spax_pos_rot[1,:] = spax_pos[0,:,:]*np.sin(pa_rad) + spax_pos[1,:,:]*np.cos(pa_rad)

    #Determine the elliptical distance of each spaxel to the origin
    dist_ellipse = np.sqrt(spax_pos_rot[0,:,:]**2 + (spax_pos_rot[1,:,:]/(1. - eps))**2)

    #Determine the angle of each spaxel relative to the major axis
    ang_ellipse = np.zeros(np.shape(dist_ellipse))
    for i in range(n_spax):
        for j in range(n_spax):
            if spax_pos_rot[0,i,j] >= 0:
                if spax_pos_rot[1,i,j] >= 0:
                    ang_ellipse[i,j] = np.degrees(np.arctan(spax_pos_rot[1,i,j]/spax_pos_rot[0,i,j]))
                else:
                    ang_ellipse[i,j] = 360.+ np.degrees(np.arctan(spax_pos_rot[1,i,j]/spax_pos_rot[0,i,j]))
            else:
                ang_ellipse[i,j] = 180. + np.degrees(np.arctan(spax_pos_rot[1,i,j]/spax_pos_rot[0,i,j]))

    #Define the radial binning scheme
    max_rad = np.max(dist_ellipse[np.isfinite(image) == True])
    if log == True:
        radii = 10.0**np.linspace(np.log10(0.5),np.log10(max_rad),num=radial+1)
        radii[0] = 0.0
    else:
        radii = np.linspace(0.0,max_rad,num=radial+1)

    #Assign each spaxel to a different radial and angular bin
    rad_bins = np.digitize(np.ravel(dist_ellipse),radii).reshape(n_spax,n_spax)
    ang_bins = np.digitize(np.ravel(ang_ellipse),angles).reshape(n_spax,n_spax)

    #Rationalize the radial and angular binning and create the bin mask

    bin_mask = np.ones(np.shape(image),dtype=np.int32)
    bin_mask[np.isfinite(image) == False] = 0

    ang_bins[np.where(ang_bins == np.max(ang_bins))] = 0
    temp = (rad_bins*10) + ang_bins
    temp[np.where(rad_bins == 9)] = 0
    bin_nums = np.unique(temp)
    for i in range(len(bin_nums)):
        ind = np.where(temp == bin_nums[i])
        bin_mask[ind] = i+1
    bin_mask[np.isfinite(image) == False] = 0

    return bin_mask

