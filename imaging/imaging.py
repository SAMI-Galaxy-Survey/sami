from __future__ import print_function, division

import os
import re
import sys
import reproject

import numpy as np

from scipy import signal

import astropy.wcs, astropy.table
from astropy.io import fits

from .. import samifitting

from .. import config
CATS, PHOT, EXTN = (
    config.__CATALOGUES__, config.__PHOTOMETRY__, config.__PHOTEXTNUM__)

try:
    import find_galaxy
    __DEFAULT_FITTING_METHOD__ = 'find_galaxy'
except ImportError:
    warn_mess = ('Module ``find_galaxy`` is optional, but highly recommended.'
        + ' You can set the keyword ``method`` to \'gaussian\' as a workaround,'
        + ' but the Gaussian fit function provided in ``samifitting`` is not as'
        + ' robust as ``find_galaxy``.'
        + ' The ``find_galaxy`` is distributed as part of the MGE software and'
        + ' can be obtained from M. Cappellari at '
        + ' http://www-astro.physics.ox.ac.uk/~mxc/software/#mge')
    warnings.warn(warn_mess, DeprecationWarning)
    __DEFAULT_FITTING_METHOD__ = 'gaussian'



class sami_image(object):

    """
    name : string
    data : 2d float array
    band : string, optional
        At the moment ``band`` in only descriptive. Too time-consuming to 
        implement given the incoherent structure of the photometry for SAMI
        (i.e. different file names, different headers, extensions, etc).

    """

    def __init__(self, name, data, band=None):

       self.name = name
       self.data = data
       self.band = band





            


class sami_wcs_image(sami_image):

    """
    name : string
    data : 2d float array
    band : string, optional
        At the moment ``band`` in only descriptive. Too time-consuming to 
        implement given the incoherent structure of the photometry for SAMI
        (i.e. different file names, different headers, extensions, etc).
    wcs  : astropy.wcs.WCS instance, optional

    """

    def __init__(self, name, data, wcs=astropy.wcs.WCS(), band=None):

       super(sami_wcs_image, self).__init__(name, data, band=band)

       self.wcs  = wcs



    @classmethod
    def from_file(cls, name, band=None):

        filename, extension =  find_file(name, band=band)

        hdu = fits.open(filename)[extension]
        data, header = hdu.data, hdu.header
        wcs = astropy.wcs.WCS(header)

        ra, dec = find_coordinates(name)

        sami_wcs_image_inst = cls(name, data, wcs=wcs, band=band)

        sami_wcs_image_inst.recentre_wcs(ra, dec)

        return sami_wcs_image_inst



    def writeto(self, output_filename, **kwargs):
        _head = self.wcs.to_header()

        for pc_key, cd_key in zip(['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'],
                                  ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']):
            _head[cd_key] = _head.pop(pc_key, 0.)
        
        fits.writeto(output_filename, self.data, _head, **kwargs)




    def circular_mask(self, radius, mode='arcsec', centre=None, fill_value=0.0):

        if mode=='arcsec':
            _current_scale = self.get_pixel_scale() * 3600.
            # Assume scale is returned in [deg], convert to [arcsec].
            radius /= _current_scale 
        elif mode=='pixel':
            pass           
        else:
            raise ValueError(
                'Keyword ``mode`` set to an invalid value ({})'.format(mode))

        centre = self.wcs.wcs.crpix if centre is None else centre
        shape_x, shape_y = self.data.shape
        ii,jj = np.meshgrid(np.arange(shape_x), np.arange(shape_y), indexing='ij')

        mask = np.where((ii-centre[0])**2 + (jj-centre[1])**2 > radius**2)
        self.data[mask] = fill_value



    def get_pixel_scale(self, use_units=False):

        cdelt = self.wcs.wcs.get_cdelt()
        pc = self.wcs.wcs.get_pc()
        scale = np.array(cdelt * pc)

        # Get the eigenvalues of the WCS matrix.
        eigs_pc = np.linalg.eig(pc)[0]
        eigs_scale = np.linalg.eig(scale)[0]

        try:
            np.testing.assert_almost_equal(abs(eigs_pc[0]), abs(eigs_pc[1]))
            np.testing.assert_almost_equal(abs(cdelt[0]), abs(cdelt[1]))
            np.testing.assert_almost_equal(abs(eigs_scale[0]), abs(eigs_scale[1]))
        except AssertionError:
            raise ValueError("Non-square pixels.  Please resample data.")

        if use_units:
            return abs(eigs_scale[0]) * u.Unit(mywcs.wcs.cunit[0])
        else:
            return abs(eigs_scale[0]) 



    def is_wcs_aligned(self):

        """Check if the WCS is aligned to have North up, East left. In WCS
        terms, this corresponds to the following transformation matrix:
        CTYPE1  = 'RA---TAN'           / x scaling                                      
        CTYPE2  = 'DEC---TAN'          / x scaling                                      
        CD1_1   =          -some_value / pixel size x                                   
        CD1_2   =                    0 / xy rotation                                    
        CD2_2   =           some_value / pixel size y                                   
        CD2_1   =                    0 / yx rotation   

        Return
        ------

        """

        pc = self.wcs.wcs.get_pc()

        return (pc[0, 1] == 0. and pc[1, 0] == 0.)




    def align_wcs(self):

        """Rotate the image to have North up, East left. In WCS terms, this
        corresponds to the following transformation matrix:
        CTYPE1  = 'RA---TAN'           / x scaling                                      
        CTYPE2  = 'DEC---TAN'          / x scaling                                      
        CD1_1   =          -some_value / pixel size x                                   
        CD1_2   =                    0 / xy rotation                                    
        CD2_2   =           some_value / pixel size y                                   
        CD2_1   =                    0 / yx rotation   
        """

        #raise NotImplementedError('This method is buggy and cannot be used')

        __current_scale = self.get_pixel_scale(use_units=False)

        # Create the output WCS instance.
        _new_wcs = self.wcs.deepcopy()
        _new_wcs.wcs.pc = np.array([[-1., 0.],
                                    [ 0., 1.]])
        _new_wcs.wcs.cdelt = np.array([__current_scale, __current_scale])

        # Determine the shape of the output array. Take the vector pointing to
        # the corner of the original image, convert it to sky coordinates, then
        # convert the sky coordinates to the pixel coordinates of the rotated
        # image.
        _corner_pix_bl = np.array([1., 1.])
        _corner_sky = self.wcs.all_pix2world(_corner_pix_bl[0], _corner_pix_bl[1], 1)
        _corner_pix_bl = _new_wcs.all_world2pix(_corner_sky[0], _corner_sky[1], 1)
        _max_len_bl = np.max(np.abs(_corner_pix_bl))

        _corner_pix_tr = np.array(self.data.shape)
        _corner_sky = self.wcs.all_pix2world(_corner_pix_tr[0], _corner_pix_tr[1], 1)
        _corner_pix_tr = _new_wcs.all_world2pix(_corner_sky[0], _corner_sky[1], 1)
        _max_len_tr = np.max(np.abs(_corner_pix_tr))

        _max_len = np.max([_max_len_bl, _max_len_tr])
        _max_len_old = np.max(self.data.shape)
        _out_shape = _max_len_old + 2. * (_max_len - _max_len_old)
        _out_shape = np.repeat(_out_shape, 2)
        _out_shape = _out_shape.round().astype(np.int)

        _out_centre = _out_shape / 2.

        _new_wcs.wcs.crpix = _out_centre

        # Use headers because WCS instances do not allow to change ``NAXIS1``
        # and ``NAXIS2``.
        _new_head = _new_wcs.to_header()
        _new_head['NAXIS1'], _new_head['NAXIS2'] = _out_shape[1], _out_shape[0]
        _new_head['NAXIS'] = 2

        _repr_data, _repr_foot = reproject.reproject_exact(
            (self.data, self.wcs), _new_head)#, shape_out=_out_shape_)
        # The areas where the new image had no coverage in the old image are
        # set to numpy.nan by default. Replace them with the median.
        _repr_data[np.where(_repr_foot < 1.)] = np.nanmedian(_repr_data)
     
        _new_wcs = astropy.wcs.WCS(_new_head)

        self.data, self.wcs = _repr_data, _new_wcs



    def resample(self, new_scale):

        """
        new_scale : float or ``astropy.units.Quantity`` instance
            new pixel scale in [arcsec / pixel]. If an instance of
            ``astropy.units.Quantity`` is passed, then the units must be the
            same as: ``astropy.wcs.WCS.wcs.cunit``.
        """

        use_units = isinstance(new_scale, astropy.units.Quantity)

        __current_scale = self.get_pixel_scale(use_units=use_units)

        # Determine the shape of the output array.
        _out_shape = np.array(self.data.shape) * __current_scale / new_scale
        _out_shape = _out_shape.round().astype(np.int)

        # Create the output WCS instance.
        _new_wcs = self.wcs.deepcopy()
        _new_wcs.wcs.pc = self.wcs.wcs.get_pc() * new_scale / __current_scale
        _out_centre = self.wcs.wcs.crpix * __current_scale / new_scale
        _out_centre = _out_centre.round().astype(np.int)

        _new_wcs.wcs.crpix = _out_centre

        # Use headers because WCS instances do not allow to change ``NAXIS1``
        # and ``NAXIS2``.
        _new_head = _new_wcs.to_header()
        _new_head['NAXIS1'], _new_head['NAXIS2'] = _out_shape[1], _out_shape[0]
        _new_head['NAXIS'] = 2

        _repr_data, _repr_foot = reproject.reproject_exact(
            (self.data, self.wcs), _new_head, parallel=False)#, shape_out=_out_shape_)
        _new_wcs = astropy.wcs.WCS(_new_head)

        self.data, self.wcs = _repr_data, _new_wcs



    def resize(self, new_size, mode='arcsec'):

        """
        new_size : float or int
            The size of the new image, expressed in pixels (int), in arcsec
            (float), or as a fraction of the old size (float, positive). The
            method interprets ``new_size`` according to the value of the keyword
            ``mode`` (see relevant documentation).
        mode : ['arcsec', 'frac', 'pixel'], optional
            Determines if ``new_size`` is expressed in units of arcsec, pixels
            or as a fraction of the old image size.
        """

        if mode=='arcsec':
            _current_scale = self.get_pixel_scale()
            # Assume scale is returned in [deg], convert to [arcsec].
            _current_scale *= 3600. 
            _out_shape = np.repeat(new_size/_current_scale, 2)
        elif mode=='frac':
            _out_shape = np.array(self.data.shape) * new_size
        elif mode=='pixel':
            _out_shape = np.array([new_size, new_size])
        else:
            raise ValueError(
                'Keyword ``mode`` set to an invalid value ({})'.format(mode))

        _out_shape = _out_shape.round().astype(np.int)

        # Create the output WCS instance.
        _new_wcs = self.wcs.deepcopy()
        _out_centre = _out_shape // 2
        _new_wcs.wcs.crpix = _out_centre.round().astype(np.int)

        # Use headers because WCS instances do not allow to change ``NAXIS1``
        # and ``NAXIS2``.
        _new_head = _new_wcs.to_header()
        _new_head['NAXIS1'], _new_head['NAXIS2'] = _out_shape[1], _out_shape[0]
        _new_head['NAXIS'] = 2

        _repr_data, _repr_foot = reproject.reproject_exact(
            (self.data, self.wcs), _new_head)#, shape_out=_out_shape_)
        _new_wcs = astropy.wcs.WCS(_new_head)

        self.data, self.wcs = _repr_data, _new_wcs



    def recentre_wcs(self, ra, dec):

        # Identify the pixels closest to the centre and convert the fractional
        # pixels to the closest integer value.
        crpix1, crpix2 = [
            np.int(x + 0.5) for x in self.wcs.all_world2pix(ra, dec, 1)]

        # Check that the pixel values are inside the image!
        if (crpix1 > self.data.shape[1]) or (crpix2 > self.data.shape[0]):
            warnings.warn(
                'The galaxy {} appears to be **outside** the image!'.format(
                    self.name))

        crval1, crval2 = [
            float(x) for x in self.wcs.all_pix2world(crpix1, crpix2, 1)]

        self.wcs.wcs.crpix = crpix1, crpix2
        self.wcs.wcs.crval = crval1, crval2



    def cross_correlate(self, template_wcs_image,
                        method=__DEFAULT_FITTING_METHOD__,
                        fraction=0.05):

        _templ_img = template_wcs_image.data.copy()
        _templ_img -= _templ_img.mean()

        _targt_img = self.data.copy()
        _targt_img -= _targt_img.mean()

        # Cross-correlate the target and template images.
        crosscorr_image = signal.correlate2d(
            _targt_img, _templ_img, mode='same', boundary='fill')
        _x_, _y_ = (
            _find_galaxy_fit(crosscorr_image, fraction=fraction) 
            if method=='find_galaxy' else
            _gauss_fit(crosscorr_image))

        _new_wcs = template_wcs_image.wcs.deepcopy()
        _template_centre_pix = (np.array(_templ_img.shape)-1.) / 2.
        _template_centre_sky = template_wcs_image.wcs.all_pix2world(
            _template_centre_pix[0], _template_centre_pix[1], 0)

        # Notice that we swap ``_x_`` and ``_y_`` to follow the FITS convention.
        _new_wcs.wcs.crpix = np.array([_y_+1., _x_+1.])
        _new_wcs.wcs.crval = _template_centre_sky

        return _new_wcs



# +---+------------------------------------------------------------------------+
# | N.| Nothing to see here, just utilities. Please move on.                   |
# +---+------------------------------------------------------------------------+

def _find_galaxy_fit(image, fraction=0.05):

    __fg_inst__ = find_galaxy.find_galaxy(image, fraction=fraction)
    x_med, y_med = __fg_inst__.xmed, __fg_inst__.ymed
    x_peak, y_peak = __fg_inst__.xpeak, __fg_inst__.ypeak

    return x_med, y_med



def _gauss_fit(image):

    # 2D Gauss Fit the cross-correlated cropped image
    x_pos, y_pos = image.shape
    x_pos, y_pos = np.arange(x_pos), np.arange(y_pos)
    x_pos, y_pos = np.meshgrid(x_pos, y_pos, indexing='xy')
    x_pos, y_pos = np.ravel(x_pos), np.ravel(y_pos)

    #define guess parameters for TwoDGaussFitter:
    amplitude = np.nanmax(image)
    mean_x, mean_y = np.unravel_index(np.nanargmax(image), image.shape)
    sigma_x = 2
    sigma_y = 2
    rotation = 60.0
    offset = 4.0
    p0 = [amplitude, mean_x, mean_y, sigma_x, sigma_y, rotation, offset]

    # call SAMI TwoDGaussFitter
    GF2d = samifitting.TwoDGaussFitter(p0, x_pos, y_pos, np.ravel(image))
    # execute gauss fit using
    GF2d.fit()
    GF2d_xpos = GF2d.p[2]
    GF2d_ypos = GF2d.p[1]

    # reconstruct the fit
    #GF2d_reconstruct=GF2d(x_pos, y_pos)

    return GF2d_xpos, GF2d_ypos


def find_coordinates(name):
    """Return the file name of the ``band``-band photometry for galaxy
    ``self.name``.
    """

    for phot in CATS.keys():

        try:
            cat = astropy.table.Table.read(CATS[phot])

            index = find_in_cat(name, cat)

            try:
                return cat[index]['RA'], cat[index]['dec']
            except KeyError:
                return cat[index]['RA'], cat[index]['Dec']
            finally:
                del cat

        except CatalogueMatchError:
            pass

    raise CatalogueMatchError(
        'No match found for galaxy name {} in catalogues {}'.format(
             name, CATS))




def find_in_cat(name, catalogue):
    """Find the argument of the table ``catalogue`` entry corresponding to the
    galaxy with id equal to ``name``.

    """

    # Default is appropriate for the SAMI input catalogues.
    if 'CATID' in catalogue.colnames:
        colname = 'CATID'
    elif 'CATAID' in catalogue.colnames:
        colname = 'CATAID'
    elif 'name' in catalogue.colnames:
        colname = 'name'
    elif 'id' in catalogue.colnames:
        colname = 'id'
    elif 'ID' in catalogue.colnames:
        colname = 'ID'
    else:
        raise ValueError('Could not find the keyword for galaxy id in {}'.format(
            catalogue))

    index = np.where(catalogue[colname].astype(str)==str(name))

    if len(index[0]) == 0:
        raise CatalogueMatchError(
            'No matches found for galaxy {} in catalogue {}'.format(
                 name, catalogue))
    elif len(index[0]) > 1:
        raise CatalogueMatchError(
            'More than one match found for galaxy {} in catalogue {}'.format(
                 name, catalogue))
    else: # Only one match found
        return index



def find_file(name, band=None):

    for phot in PHOT.keys():

        try:
            regex = PHOT[phot]
            return find_file_from_regex(name, regex, band=band), EXTN[phot]
        except FITSFileMatchError:
            pass

    raise FITSFileMatchError(
        'No match found for galaxy name {} in paths {}'.format(
        name, PHOT))



def find_file_from_regex(name, regex, band=None):

    regex = regex.replace('[galaxy_id]', str(name))
    if band: regex.replace('[band]', str(band))

    match_files = re.compile(regex)

    files_path = os.path.realpath(os.path.dirname(regex))

    full_filenames = []

    full_filenames += [dirpath + '/' + filename
                       for dirpath, dirnames, filenames
                       in os.walk(files_path)
                       for filename in filenames
                       if match_files.search(dirpath + '/' + filename)]

    if len(full_filenames) == 0:
        raise FITSFileMatchError(
            'No matches found for galaxy {} using regex {}'.format(
                 name, regex))
    elif len(full_filenames) > 1:
        raise FITSFileMatchError(
            'More than one match found for galaxy {} using regex {}'.format(
                 name, regex))
    else: # Only one match found
        return full_filenames[0]



class CatalogueMatchError(Exception):
    pass

class FITSFileMatchError(Exception):
    pass


if __name__=="__main__":

    print("Usage examples:\n")
    print(">>> from sami import imaging")
    print(">>> swi = imaging.sami_wcs_image.from_file(551488)")
    print(">>> swi.resize(15., mode='arcsec')")
    print(">>> swi.writeto('testing_resize_delete_this_file.fits')")

    print("Usage examples:\n")
    print(">>> from sami import imaging")
    print(">>> swi = imaging.sami_wcs_image.from_file(551488)")
    print(">>> swi.resample(2./3600.)")
    print(">>> swi.writeto('testing_resample_delete_this_file.fits')")
