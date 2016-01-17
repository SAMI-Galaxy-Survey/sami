"""
Functions for measuring and recording WCS information.

In particular, wcs_position_coords is supposed to determine the WCS for
a file based on cross-correlating a collapsed image from the datacube with
an external photometric image. However, this was never shown to work
properly (the results were clustering around particular values, for
unknown reasons), so it was put to one side and never finished. Instead,
the SAMI Galaxy Survey has been using the 'nominal' WCS, which assumes
that the catalogued object is in the centre of the data.
"""

import numpy as np
import scipy as sp
import astropy.io.ascii as ascii
from scipy.interpolate import griddata
import astropy.io.fits as pf
import os
import urllib

from .. import samifitting as fitting
from ..sdss import sdss

#########################

def wcs_solve(myIFU, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, plot=False, write=False, nominal=False, remove_thput_file=True):
    
    """Wrapper for wcs_position_coords, extracting coords from IFU.
        
        This function cross-correlates a g or r-band convolved SAMI cube with its
        respective SDSS g-band image and pins down the positional WCS for the
        central spaxel of the cube.
        """
    
    # Get Object RA + DEC from fibre table (this is the input catalogues RA+DEC in deg)
    object_RA = np.around(myIFU.obj_ra[myIFU.n == 1][0], decimals=6)
    object_DEC = np.around(myIFU.obj_dec[myIFU.n == 1][0], decimals=6)
    
    # Build wavelength axis.
    CRVAL3 = myIFU.crval1
    CDELT3 = myIFU.cdelt1
    Nwave  = np.shape(object_flux_cube)[0]
    
    # -- crval3 is middle of range and indexing starts at 0.
    # -- this wave-axis agrees with QFitsView interpretation.
    CRVAL3a = CRVAL3 - ((Nwave-1)/2)*CDELT3
    wave = CRVAL3a + CDELT3*np.arange(Nwave)
    
    object_flux_cube = np.transpose(object_flux_cube, (2,0,1))
    
    return wcs_position_coords(object_RA, object_DEC, wave, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, plot=plot, write=write, nominal=nominal)

def wcs_position_coords(object_RA, object_DEC, wave, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, plot=False, write=False, nominal=False, remove_thput_file=True):
    """Equate the WCS position information from a cross-correlation between a
        g-band SAMI cube and a g-band SDSS image."""
    
    if nominal:
        img_crval1 = object_RA
        img_crval2 = object_DEC
        xcube = size_of_grid
        ycube = size_of_grid
        img_cdelt1 = -1.0 * output_pix_size_arcsec / 3600.0
        img_cdelt2 = output_pix_size_arcsec / 3600.0
    
    else:
        
        # Get SDSS g-band throughput curve
        if not os.path.isfile("sdss_"+str(band)+".dat"):
            urllib.urlretrieve("http://www.sdss.org/dr3/instruments/imager/filters/"+str(band)+".dat", "sdss_"+str(band)+".dat")
        
        # and convolve with the SDSS throughput
        sdss_filter = ascii.read("SDSS_"+str(band)+".dat", comment="#", names=["wave", "pt_secz=1.3", "ext_secz=1.3", "ext_secz=0.0", "extinction"])
        
        # re-grid g["wave"] -> wave
        thru_regrid = griddata(sdss_filter["wave"], sdss_filter["ext_secz=1.3"], wave, method="cubic", fill_value=0.0)
        
        # initialise a 2D simulated g' band flux array.
        len_axis = np.shape(object_flux_cube)[1]
        Nwave = len(wave)
        reconstruct = np.zeros((len_axis,len_axis))
        tester = np.zeros((len_axis,len_axis))
        data_bit = np.zeros((Nwave,len_axis,len_axis))
        
        # Sum convolved flux:
        for i in range(Nwave):
            data_bit[i] = object_flux_cube[i]*thru_regrid[i]
        
        reconstruct = np.nansum(data_bit,axis=0) # not absolute right now
        reconstruct[np.isnan(reconstruct)] = 0. # replacing nan with 0.0
        reconstruct[reconstruct < 0] = 0.       # replacing negative fluxes with 0.0
        
        cube_image = reconstruct
        xcube = len(cube_image[0])
        ycube = len(cube_image[1])
        cube_image_crop = cube_image[(len(cube_image[0])/2)-10:(len(cube_image[0])/2)+10,(len(cube_image[1])/2)-10:(len(cube_image[1])/2)+10]
        cube_image_crop = sp.ndimage.zoom(cube_image_crop, 5, order=3)
        cube_image_crop_norm = (cube_image_crop - np.min(cube_image_crop))/np.max(cube_image_crop - np.min(cube_image_crop))
        
        # Check if the user supplied a red RSS file, throw exception.
        if np.array_equal(cube_image, tester):
            raise SystemExit("All values are zero: please provide the cube corresponding to the requested spectral band of the image!")
        
        ##########
        
        cube_size = np.around((size_of_grid*output_pix_size_arcsec)/3600, decimals=6)
        
        # Get SDSS Image
        if not os.path.isfile(str(object_name)+"_SDSS_"+str(band)+".fits"):
            sdss.getSDSSimage(object_name=object_name, RA=object_RA, DEC=object_DEC,
                         band=str(band), size=cube_size, number_of_pixels=size_of_grid)
        
        # Open SDSS image and extract data & header information
        image_file = pf.open(str(object_name)+"_SDSS_"+str(band)+".fits")
        image_data = image_file['Primary'].data
        
        
        image_header = image_file['Primary'].header
        img_crval1 = float(image_header['CRVAL1']) #RA
        img_crval2 = float(image_header['CRVAL2']) #DEC
        img_crpix1 = float(image_header['CRPIX1']) #Reference x-pixel
        img_crpix2 = float(image_header['CRPIX2']) #Reference y-pixel
        img_cdelt1 = float(image_header['CDELT1']) #Delta RA
        img_cdelt2 = float(image_header['CDELT2']) #Delta DEC
        
        SDSS_image = image_data
        SDSS_image_crop = SDSS_image[(len(SDSS_image[0])/2)-10:(len(SDSS_image[0])/2)+10,(len(SDSS_image[1])/2)-10:(len(SDSS_image[1])/2)+10]
        SDSS_image_crop_norm = (SDSS_image_crop - np.min(SDSS_image_crop))/np.max(SDSS_image_crop - np.min(SDSS_image_crop))
    
    ##########
    
    if (not nominal) and np.size(np.where(image_data == 0.0)) != 2*np.size(image_data):
        # Cross-correlate normalised SAMI-cube g-band image and SDSS g-band image
        WCS_flag = 'SDSS'
        crosscorr_image = sp.signal.correlate2d(SDSS_image_crop_norm, cube_image_crop_norm)
        
        # 2D Gauss Fit the cross-correlated cropped image
        crosscorr_image_1d = np.ravel(crosscorr_image)
        #use for loops to recover indicies in x and y positions of flux values
        x_pos = []
        y_pos = []
        for i in xrange(np.shape(crosscorr_image)[0]):
            for j in xrange(np.shape(crosscorr_image)[1]):
                x_pos.append(i)
                y_pos.append(j)
        x_pos=np.array(x_pos)
        y_pos=np.array(y_pos)
        
        #define guess parameters for TwoDGaussFitter:
        amplitude = max(crosscorr_image_1d)
        mean_x = (np.shape(crosscorr_image)[0])/2
        mean_y = (np.shape(crosscorr_image)[1])/2
        sigma_x = 5.0
        sigma_y = 6.0
        rotation = 60.0
        offset = 4.0
        p0 = [amplitude, mean_x, mean_y, sigma_x, sigma_y, rotation, offset]
        
        # call SAMI TwoDGaussFitter
        GF2d = fitting.TwoDGaussFitter(p0, x_pos, y_pos, crosscorr_image_1d)
        # execute gauss fit using
        GF2d.fit()
        GF2d_xpos = GF2d.p[2]
        GF2d_ypos = GF2d.p[1]
        
        # reconstruct the fit
        GF2d_reconstruct=GF2d(x_pos, y_pos)
        
        x_shape = len(crosscorr_image[0])
        y_shape = len(crosscorr_image[1])
        x_offset_pix = GF2d_xpos - x_shape/2
        y_offset_pix = GF2d_ypos - y_shape/2
        x_offset_arcsec = -x_offset_pix * output_pix_size_arcsec/5
        y_offset_arcsec = y_offset_pix * output_pix_size_arcsec/5
        x_offset_degree = ((x_offset_arcsec/3600)/24)*360
        y_offset_degree = (y_offset_arcsec/3600)
    
    else:
        WCS_flag = 'Nominal'
        y_offset_degree = 0.0
        x_offset_degree = 0.0
    
    # Create dictionary of positional WCS
    if isinstance(xcube/2, int):
        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2 + 0.5),
            "CRPIX2":(ycube/2 + 0.5), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"DEGREE", "CTYPE2":"DEGREE"}
    else:
        WCS_pos={"CRVAL1":(img_crval1 + x_offset_degree), "CRVAL2":(img_crval2 + y_offset_degree), "CRPIX1":(xcube/2),
            "CRPIX2":(ycube/2), "CDELT1":(img_cdelt1), "CDELT2":(img_cdelt2), "CTYPE1":"DEGREE", "CTYPE2":"DEGREE"}
    
    ##########
    
    # Remove temporary files
    if remove_thput_file and os.path.exists("sdss_"+str(band)+".dat"):
        os.remove("sdss_"+str(band)+".dat")
    if os.path.exists(str(object_name)+"_SDSS_"+str(band)+".fits"):
        os.remove(str(object_name)+"_SDSS_"+str(band)+".fits")
    
    return WCS_pos,WCS_flag


def update_wcs_coords(filename, nominal=False, remove_thput_file=True):
    """Recalculate the WCS data in a SAMI datacube."""
    
    # Pick out the relevant data
    header = pf.getheader(filename)
    ra = (header['CRVAL1'] + (1 + np.arange(header['NAXIS1']) - header['CRPIX1']) * header['CDELT1'])
    dec = (header['CRVAL2'] + (1 + np.arange(header['NAXIS2']) - header['CRPIX2']) * header['CDELT2'])
    wave = (header['CRVAL3'] + (1 + np.arange(header['NAXIS3']) - header['CRPIX3']) * header['CDELT3'])
    object_RA = np.mean(ra)
    object_DEC = np.mean(dec)
    object_flux_cube = pf.getdata(filename)
    object_name = header['NAME']
    if header['GRATID'] == '580V':
        band = 'g'
    elif header['GRATID'] == '1000R':
        band = 'r'
    else:
        raise ValueError('Could not identify band. Exiting')

    size_of_grid = np.shape(object_flux_cube)[0] #should be = 50
    output_pix_size_arcsec = header['CDELT1'] #should be = 0.5

    # Calculate the WCS
    WCS_pos, WCS_flag = wcs_position_coords(object_RA, object_DEC, wave, object_flux_cube, object_name, band, size_of_grid, output_pix_size_arcsec, nominal=nominal, remove_thput_file=remove_thput_file)

    # Update the file
    hdulist = pf.open(filename, 'update', do_not_scale_image_data=True)
    header = hdulist[0].header
    for key, value in WCS_pos.items():
        header[key] = value
        header['WCS_SRC'] = WCS_flag
    hdulist.close()

    return

############### END OF FILE ###############