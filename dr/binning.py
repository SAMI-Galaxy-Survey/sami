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
import os.path

from glob import glob

from .. import slogging
log = slogging.getLogger(__name__)
log.setLevel(slogging.INFO)

import astropy.io.fits as pf
import numpy as np
from numpy import nanmedian
from scipy.ndimage.filters import median_filter
from scipy.ndimage.measurements import label
from . import voronoi_2d_binning_wcovar
from ..utils.other import hg_changeset

import code

def bin_cube_pair(path_blue, path_red, name=None, **kwargs):
    """Calculate bins, do binning and save results for a pair of cubes."""
    hdulist_blue = pf.open(path_blue, 'update')
    hdulist_red = pf.open(path_red, 'update')
    bin_mask = return_bin_mask(hdulist_blue, **kwargs)
    bin_and_save(hdulist_blue, bin_mask, name=name, **kwargs)
    bin_and_save(hdulist_red, bin_mask, name=name, **kwargs)
    hdulist_blue.close()
    hdulist_red.close()

def is_id_in_catalogs(sami_id, catalogs):
    sami_id = int(sami_id)
    for cat in catalogs:
        if sami_id not in catalogs[cat]['CATAID']:
            print "SAMI ID %s not in GAMA catalog %s" % (sami_id, cat)
            return False
    # Found in all catalogs
    return True


class CatalogAccessor(object):
    """Class to handle accessing GAMA catalogs stored in FITS files."""

    def __init__(self, path_to_catalogs, catalog_descriptions):
        self.path_to_catalogs = path_to_catalogs
        self.catalog_names = catalog_descriptions.keys()
        self.catalog_descriptions = catalog_descriptions
        self.catalogs = dict()
        self.catalog_filenames = dict()

        self.load_catalogs()
        log.debug("Catalog data is loaded and available.")


    def load_catalogs(self):
        """Find, load, and check the catalogs.

        For each, search in the path_to_catalogs directory for a likely looking
        file, open it, find the data, and then check that the expected columns
        are present.

        """

        for cat in self.catalog_names:
            try:
                files_found = glob(self.path_to_catalogs + "/*" + cat + "*.fits")
                if len(files_found) > 1:
                    log.warning("Multiple potential catalogs found for %s!", cat)
                self.catalog_filenames[cat] = files_found[0]
                with pf.open(self.catalog_filenames[cat]) as f:
                    self.catalogs[cat] = f[1].data
                for col in self.catalog_descriptions[cat]:
                    assert col in self.catalogs[cat].columns.dtype.names
            except Exception as e:
                print("Original error: %s" % e.message)
                raise ValueError("Invalid or missing GAMA Catalog %s in directory %s" %
                                 (cat, os.path.abspath(self.path_to_catalogs)))

    def cataid_available(self, sami_id):
        sami_id = int(sami_id)
        for cat in self.catalog_names:
            if sami_id not in self.catalogs[cat]['CATAID']:
                print "SAMI ID %s not in GAMA catalog %s" % (sami_id, cat)
                return False
        # Found in all catalogs
        return True

    def retrieve(self, catalog_name, column, cataid):
        """Return the value from the catalog for the given CATAID"""
        cataid = int(cataid)
        catalog = self.catalogs[catalog_name]
        if cataid not in catalog['CATAID']:
            # print "SAMI ID %s not in GAMA catalogs - no aperture spectra produced" % cataid
            raise ValueError("CATAID %s not in GAMA Catalog" % cataid)
        else:
            # Cut down the catalog to only contain the row for this SAMI ID.
            return catalog[catalog['CATAID'] == cataid][column][0]

def aperture_spectra_pair(path_blue, path_red, path_to_catalogs):
    """Calculate binned spectra and save as new file for each pair of cubes."""

    if log.isEnabledFor(slogging.INFO):
        log.info("Running aperture_spectra_pair HG version %s", hg_changeset(__file__))
    log.debug("Starting aperture_spectra_pair: %s, %s, %s", path_blue, path_red, path_to_catalogs)

    # A dictionary of required catalogs and the columns required in each catalog.
    catalogs_required = {
        'ApMatchedCat': ['THETA_J2000', 'THETA_IMAGE'],
        'SersicCatAll': [
            'GAL_RE_R',
            'GAL_PA_R',
            'GAL_R90_R',
            'GAL_ELLIP_R'],
        # Note spelling of Distance(s)Frames different from that used by GAMA
        'DistanceFrames': ['Z_TONRY_2']
    }

    gama_catalogs = CatalogAccessor(path_to_catalogs, catalogs_required)

    # Open the two cubes
    with pf.open(path_blue, memmap=True) as hdulist_blue, pf.open(path_red, memmap=True) as hdulist_red:

        # Work out the standard apertures to extract
        #
        #     This step requires some details from the header, so it must be done
        #     after the files have been opened.

        from astropy.cosmology import WMAP9 as cosmo
        from astropy import units as u
        from astropy.wcs import WCS
        standard_apertures = dict()

        sami_id = hdulist_blue[0].header['NAME']

        if gama_catalogs.cataid_available(sami_id):
            log.info("Constructing apertures using GAMA data for SAMI ID %s", sami_id)
        else:
            print("No aperture spectra produced for {} because it is not in the GAMA catalogs".format(sami_id))
            return None


        # The position angle must be adjusted to get PA on the sky.
        # See http://www.gama-survey.org/dr2/schema/dmu.php?id=36
        pos_angle_adjust = (gama_catalogs.retrieve('ApMatchedCat', 'THETA_J2000', sami_id) -
                            gama_catalogs.retrieve('ApMatchedCat', 'THETA_IMAGE', sami_id))


        # size of a pixel in angular units. CDELT1 is the WCS pixel size, and CTYPE1 is "DEGREE"
        pix_size = np.abs((hdulist_blue[0].header['CDELT1'] * u.deg).to(u.arcsec).value)
        # confirm that both files are the same!
        assert hdulist_blue[0].header['CTYPE1'] == 'DEGREE'
        assert hdulist_blue[0].header['CTYPE1'] == hdulist_red[0].header['CTYPE1']
        assert hdulist_blue[0].header['CDELT1'] == hdulist_red[0].header['CDELT1']


        standard_apertures['re'] = {
            'aperture_radius': gama_catalogs.retrieve('SersicCatAll', 'GAL_RE_R', sami_id)/pix_size,
            'pa': gama_catalogs.retrieve('SersicCatAll', 'GAL_PA_R', sami_id) + pos_angle_adjust,
            'ellipticity': gama_catalogs.retrieve('SersicCatAll', 'GAL_ELLIP_R', sami_id)
        }

#        standard_apertures['r90'] = {
#            'aperture_radius': gama_catalogs.retrieve('SersicCatAll', 'GAL_R90_R', sami_id)/pix_size,
#            'pa': gama_catalogs.retrieve('SersicCatAll', 'GAL_PA_R', sami_id) + pos_angle_adjust,
#            'ellipticity': gama_catalogs.retrieve('SersicCatAll', 'GAL_ELLIP_R', sami_id)
#        }

        redshift = gama_catalogs.retrieve('DistanceFrames', 'Z_TONRY_2', sami_id)
        ang_size_kpc = (1*u.kpc / cosmo.kpc_proper_per_arcmin(redshift)).to(u.arcsec).value / pix_size

        standard_apertures['3kpc_round'] = {
            'aperture_radius': 1.5*ang_size_kpc,
            'pa': 0,
            'ellipticity': 0
        }

        standard_apertures['1.4_arcsecond'] = {
            'aperture_radius': 1.4/pix_size,
            'pa': 0,
            'ellipticity': 0
        }

        standard_apertures['2_arcsecond'] = {
            'aperture_radius': 2.0/pix_size,
            'pa': 0,
            'ellipticity': 0
        }

        standard_apertures['3_arcsecond'] = {
            'aperture_radius': 3.0/pix_size,
            'pa': 0,
            'ellipticity': 0
            }

        standard_apertures['4_arcsecond'] = {
            'aperture_radius': 4.0/pix_size,
            'pa': 0,
            'ellipticity': 0
            }

#        try:
#            seeing = get_seeing(hdulist_blue)
#        except:
#            print "Unable to determine seeing for %s, seeing aperture not included" % path_blue
#        else:
#            standard_apertures['seeing'] = {
#                'aperture_radius': seeing/pix_size,
#                'pa': 0,
#                'ellipticity': 0
#            }

        bin_mask = None

        for hdulist in (hdulist_blue, hdulist_red):

            # Construct a path name for the output spectra:
            path = hdulist.filename()
            out_dir = os.path.dirname(path)
            if out_dir == "":
                out_dir = '.'
            out_file_base = os.path.basename(path).split(".")[0]
            output_filename = out_dir + "/" + out_file_base + "_aperture_spec.fits"

            # Create a new output FITS file:
            aperture_hdulist = pf.HDUList([pf.PrimaryHDU()])

            # Copy the header from the fits CUBE to primary HDU
            aperture_hdulist[0].header = hdulist[0].header

            aperture_hdulist[0].header['HGAPER'] = (hg_changeset(__file__), "Hg changeset ID for aperture code")

            # Calculate the aperture bins based on first file only.
            if bin_mask is None:
                bin_mask = dict()
                for aper in standard_apertures:
                    bin_mask[aper] = aperture_bin_sami(hdulist, **standard_apertures[aper])
                    standard_apertures[aper]['mask'] = (bin_mask[aper] == 1)
                    standard_apertures[aper]['n_pix_included'] = int(np.sum(standard_apertures[aper]['mask']))
                log_aperture_data(standard_apertures, sami_id)

            for aper in standard_apertures:
                aperture_data = standard_apertures[aper]

                binned_cube, binned_var = bin_cube(hdulist, aperture_data['mask'])

                if log.isEnabledFor(slogging.DEBUG):
                    print "Bins: ", np.unique(bin_mask[aper]).tolist()

                n_spax_included = aperture_data['n_pix_included']

                # Calculate area correction:
                #
                #     The minimum quantum for binning spectra is whole spaxels.
                #     So the area of the binned aperture spectra will generally
                #     not exactly match the area of the aperture. We compute a
                #     scaling that will standardize this, so that comparing
                #     aperture spectra will not introduce any systematics.
                spaxel_area = n_spax_included * pix_size**2
                # (remember aperture_radius is in pix_size, so the ellipse is initially pix_size)
                aperture_area = (2 * np.pi *
                                 (aperture_data['aperture_radius'])**2 *
                                 (1 - aperture_data['ellipticity'])) * pix_size**2
                area_correction = aperture_area / spaxel_area

                if n_spax_included > 0:
                    # Find the x, y index of a spectrum inside the first (only) bin:
                    x, y = np.transpose(np.where(bin_mask[aper] == 1))[0]
                    aperture_spectrum = binned_cube[:, x, y] * n_spax_included
                    aperture_variance = binned_var[:, x, y] * n_spax_included
                else:
                    aperture_spectrum = np.zeros_like(binned_cube[:, 0, 0])
                    aperture_variance = np.zeros_like(binned_var[:, 0, 0])

                aperture_hdulist.extend([
                    pf.ImageHDU(aperture_spectrum, name=aper.upper()),
                    pf.ImageHDU(aperture_variance, name=aper.upper() + "_VAR"),
                    pf.ImageHDU((bin_mask[aper] == 1).astype(int), name=aper.upper() + "_MASK")])

                output_header = aperture_hdulist[aper.upper()].header
                output_header['RADIUS'] = (
                    aperture_data['aperture_radius'],
                    "Radius of the aperture in spaxels")
                output_header['ELLIP'] = (
                    aperture_data['ellipticity'],
                    "Ellipticity of the aperture (1-b/a)")
                output_header['POS_ANG'] = (
                    aperture_data['pa'],
                    "Position angle of the major axis, N->E")
                output_header['KPC_SIZE'] = (
                    ang_size_kpc,
                    "Size of 1 kpc at galaxy distance in pixels")
                output_header['Z_TONRY'] = (
                    redshift,
                    "Redshift used to calculate galaxy distance")
                output_header['N_SPAX'] = (
                    n_spax_included,
                    "Number of spaxels included in mask")
                output_header['AREACORR'] = (
                    area_correction,
                    "Ratio of included spaxel area to aper area")

                # Copy the wavelength axis WCS information into the new header.
                # This is done by creating a new WCS for the cube header,
                # dropping the first two axes (which are spatial coordinates),
                # and then appending the remaining header keywords.
                output_header.extend(WCS(hdulist[0].header).dropaxis(0).dropaxis(0).to_header())

                log.debug("Aperture %s completed", aper)

            aperture_hdulist.writeto(output_filename, clobber=True)
            log.info("Aperture spectra written to %s", output_filename)


def bin_and_save(hdulist, bin_mask, name=None, **kwargs):
    """Do binning and save results for an HDUList."""
    # TODO: Check if the extensions already exist. In most cases you would
    # want to either overwrite or just return without doing anything, but
    # occasionally the user might want to append duplicate extensions (the
    # current behaviour). (JTA 14/9/2015)

    # Default behaviour here is now to overwrite extensions. If extension exists
    # and overwrite=False this should have been caught by manager.bin_cubes()

    binned_cube, binned_var = bin_cube(hdulist, bin_mask, **kwargs)
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

    """
def bin_cube(hdu,bin_mask, mode='', **kwargs):
    #Produce a SAMI cube where each spaxel contains the
    #spectrum of the bin it is associated with
    
    Parameters

        bin_mask is a 2D array of integers. Spaxels with the same integer
        value will be combined into the same binned spectrum. Spaxels with a
        bin "id" of 0 will not be binned.

        hdu is an open SAMI FITS Cube file.

    Notes:

        The variance in output correctly accounts for covariance, but the
        remaining covariance between bins is not tracked (this may change if
        enough people request it)

    """

    cube = hdu[0].data
    var = hdu[1].data
    weight = hdu[2].data
    covar = reconstruct_covariance(hdu[3].data,hdu[3].header,n_wave=cube.shape[0])

    weighted_cube = cube*weight
    weighted_var = var*weight*weight

    binned_cube = np.ones(np.shape(cube))*np.nan
    binned_var = np.ones(np.shape(var))*np.nan

    n_bins = int(np.max(bin_mask))

    for i in range(n_bins):
        spaxel_coords = np.array(np.where(bin_mask == i+1))
        n_spaxels = len(spaxel_coords[0])
        if n_spaxels == 1:
            binned_cube[:,spaxel_coords[0,:],spaxel_coords[1,:]] = cube[:,spaxel_coords[0,:],spaxel_coords[1,:]]
            binned_var[:,spaxel_coords[0,:],spaxel_coords[1,:]] = var[:,spaxel_coords[0,:],spaxel_coords[1,:]]
        elif n_spaxels > 1:
            binned_spectrum = np.nansum(cube[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)/n_spaxels
            binned_weighted_spectrum = np.nansum(weighted_cube[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)#/n_spaxels
            binned_weight = np.nansum(weight[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)
            binned_weight2 = np.nansum(weight[:,spaxel_coords[0,:],spaxel_coords[1,:]]**2,axis=1)

            if mode == 'adaptive':
                temp = np.tile(np.reshape(binned_weighted_spectrum/binned_weight,(len(binned_spectrum),1)),n_spaxels)
            else:
                temp = np.tile(np.reshape(binned_spectrum,(len(binned_spectrum),1)),n_spaxels)

            binned_cube[:,spaxel_coords[0,:],spaxel_coords[1,:]] = temp
            #covar_factor = np.nansum(np.nansum(covar[:,:,:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)/2.0,axis=1) #This needs to be an accurate calculation of the covar factor
            order = np.argsort(np.nanmedian(weight[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=0))[::-1]
            covar_factor = return_covar_factor(spaxel_coords[0,:],spaxel_coords[1,:],covar,order)
            #binned_weighted_variance = np.nansum(weighted_var[:,spaxel_coords[0,:],spaxel_coords[1,:]],axis=1)*covar_factor
            binned_weighted_variance = np.nansum(weighted_var[:,spaxel_coords[0,:],spaxel_coords[1,:]]*covar_factor,axis=1)
            binned_variance = binned_weighted_variance*((binned_spectrum/binned_weighted_spectrum)**2)#/(n_spaxels**2)
            if mode == 'adaptive':
                temp_var = np.tile(np.reshape(binned_weighted_variance/(binned_weight**2),(len(binned_variance),1)),n_spaxels)
            else:
                temp_var = np.tile(np.reshape(binned_variance,(len(binned_variance),1)),n_spaxels)
            
            binned_var[:,spaxel_coords[0,:],spaxel_coords[1,:]] = temp_var

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

def return_covar_factor(xin,yin,covar,order):

    xin = xin[order]
    yin = yin[order]

    n_grid = covar.shape[1]
    xin2 = np.transpose(np.tile(xin,(n_grid**2,1)))
    yin2 = np.transpose(np.tile(yin,(n_grid**2,1)))
    
    ximprint = np.repeat(np.arange(n_grid)-(n_grid-1)/2,n_grid)
    yimprint = np.tile(np.arange(n_grid)-(n_grid-1)/2,n_grid)
    
    covar_factor = np.zeros((covar.shape[0],len(xin)))
    covar_factor[:,0] = np.ones(covar.shape[0])
    #covar_image = np.nanmedian(covar,axis=0)
    #covar_matrix = np.rollaxis(covar_image[:,:,xin,yin],2)
    #covar_flat = np.reshape(covar_matrix,(len(xin),n_grid**2))
    
    covar_matrix = np.rollaxis(covar[:,:,:,xin,yin],3)
    covar_flat = np.reshape(covar_matrix,(len(xin),covar.shape[0],n_grid**2))
    
    for i in range(1,covar_factor.shape[1]):
        #w = np.where((abs(xin - xin[i]) < 2) & (abs(yin-yin[i]) < 2))
        xoverlap = xin2[:i,:] - (ximprint + xin[i])
        yoverlap = yin2[:i,:] - (yimprint + yin[i])
        w = np.where((xoverlap == 0) & (yoverlap == 0))[1]
        #covar_factor[i] = np.nansum(covar_flat[i,w]+1)
        covar_factor[:,i] = np.nansum(covar_flat[i,:,w]+1,axis=0)
    
    covar_factor = covar_factor[:,np.argsort(order)]
    
    return covar_factor

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
    covar_image = np.nanmedian(covar,axis=0)
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
    # Based on Michele Cappellari's IDL routine find_galaxy.pro
    # Derives basic galaxy parameters using the weighted 2nd moments
    # of the luminosity distribution
    # Makes use of 2nd_moments
    
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
        print 'Pixels used: %i' % len(ind[0])
        print 'Peak (x,y): %i %i' % (xpeak,ypeak)
        print 'Mean (x,y): %f %f' % (xmed,ymed)
        print 'Theta (deg): %f' % pa
        print 'Eps: %f' % eps
        print 'Sigma along major axis (pixels): %f' % maj

    n_blobs = np.max(a)
    
    return maj,eps,pa,xpeak[0],ypeak[0],xmed,ymed,n_blobs

def prescribed_bin_sami(hdu,sectors=8,radial=5,log=False,
                        xmed='',ymed='',pa='',eps=''):
    """Allocate spaxels to a bin, based on the standard SAMI binning scheme.

    Returns a 50x50 array where each element contains the bin number to which
    a given spaxel is allocated. Bin ids spiral out from the centre.

    Users can select number of sectors (1, 4, 8, maybe 16?), number of radial bins and
    whether the radial progression is linear or logarithmic

    Users can provide centroid, pa and ellipticity information manually.
    The PA should be in degrees.

    NOTE: This code will break if the SAMI cubes are ever not square.

    """

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

    ang_ellipse[np.where(np.isfinite(ang_ellipse) == False)] = 0.0

    #Define the radial binning scheme
    max_rad = np.max(dist_ellipse[np.isfinite(image) == True])
    max_rad_maj = max_rad*eps
    if log == True:
        radii = 10.0**np.linspace(np.log10(0.5),np.log10(max_rad),num=radial+1)
        radii[0] = 0.0
    else:
        radii = np.linspace(0.0,max_rad,num=radial+1)

    radii[0] = -1

    #Assign each spaxel to a different radial and angular bin
    rad_bins = np.digitize(np.ravel(dist_ellipse),radii,right=True).reshape(n_spax,n_spax)
    ang_bins = np.digitize(np.ravel(ang_ellipse),angles,right=True).reshape(n_spax,n_spax)

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

def aperture_bin_sami(hdu, aperture_radius=1, ellipticity=0, pa=0):
    """Produce an aperture bin (inside and outside) for the aperture given."""
    log.debug("Arguments: %s, %s, %s", aperture_radius, ellipticity, pa)

    # Note, pyfits transposes axes compared to python, so we un-do that below.
    # `cube` will now have RA on the first axis, DEC on the second axis, and
    # WAVELENGTH on the 3rd.
    cube = np.transpose(hdu['PRIMARY'].data)
    n_spax = cube.shape[1]

    # Use the centre of the cube...
    xmed = float(cube.shape[0] - 1)/2.0
    ymed = float(cube.shape[1] - 1)/2.0

    pa_rad = np.radians(pa)

    # Compute coordinates of input spaxels in the coordinate system of the
    # galaxy, i.e., distance from the major axis in the first coordinate and
    # distance from the minor axis in the second coordinate

    spax_pos = np.indices((n_spax, n_spax), dtype=np.float)
    # Shift centre to be centre of cube.
    spax_pos[0, :] = spax_pos[0, :, :] - xmed
    spax_pos[1, :] = spax_pos[1, :, :] - ymed
    spax_pos_rot = np.zeros_like(spax_pos)

    # Here, for SAMI, the first axis is RA and the second axis -DEC.
    # (This gives for north up, east to the left.)
    #
    # Below is the standard form of the rotation matrix formula, which rotates
    # counter-clockwise in the x-y plane.
    #
    # However, we are rotating the input coordinates, before applying the
    # elipticity. This reverses the sense of the rotation.
    #
    # Therefore, we rotate by `-pa_rad` so that the rotation is North  through
    # east, or counter-clockwise.
    spax_pos_rot[0, :] = spax_pos[0, :, :] * np.cos(-pa_rad) - spax_pos[1, :, :] * np.sin(-pa_rad)
    spax_pos_rot[1, :] = spax_pos[0, :, :] * np.sin(-pa_rad) + spax_pos[1, :, :] * np.cos(-pa_rad)

    # Determine the elliptical distance of each spaxel to the origin
    dist_ellipse = np.sqrt((spax_pos_rot[0, :, :] / (1. - ellipticity)) ** 2 + spax_pos_rot[1, :, :] ** 2)

    log.debug("Range of distances: %s to %s", np.min(dist_ellipse), np.max(dist_ellipse))
    log.debug("Pixels within radius: %s", np.sum(dist_ellipse < aperture_radius))

    log.debug(dist_ellipse)

    # Finally, we transpose back to the coordinate system y, x so that the
    # output of this code will match the expectation of `bin_cube`, which does
    # not transpose the FITS data.
    dist_ellipse = np.transpose(dist_ellipse)

    # Assign each spaxel to a different radial bin
    rad_bins = np.digitize(np.ravel(dist_ellipse), (0, aperture_radius)).reshape(n_spax, n_spax)

    return rad_bins


def get_seeing(hdulist):

    sami_id = hdulist[0].header['NAME']
    std_id = hdulist[0].header['STDNAME']
    obj_cube_path = hdulist.filename()

    id_path_section = "{0}/{0}".format(sami_id)

    start = obj_cube_path.rfind(id_path_section)
    star_cube_path = (obj_cube_path[:start] +
                      "{0}/{0}".format(std_id) +
                      obj_cube_path[start+len(id_path_section):])

    return pf.getval(star_cube_path, 'PSFFWHM')


def log_aperture_data(standard_apertures, sami_id):
    """Log aperture information in a readable format"""
    if log.isEnabledFor(slogging.INFO):
        aperture_info = "Aperture Information for %s:\n" % sami_id
        aperture_info += ("   {:<11s} {:>8s} {:>8s} {:>8s} {:>8s}\n".format(
            'Aperture', 'n_pix', 'radius', 'ellip', 'PA'))
        for aper in standard_apertures:
            aperture_info += ("   {:11s} {:8.0f} {:8.2f} {:8.2f} {:8.2f}\n".format(
                aper,
                standard_apertures[aper]['n_pix_included'],
                standard_apertures[aper]['aperture_radius'],
                standard_apertures[aper]['ellipticity'],
                standard_apertures[aper]['pa']))
        log.info(aperture_info)
