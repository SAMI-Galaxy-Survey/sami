"""
Refined wavelength calibration code that derives an improved wavecal solution from
blue twilight frames then applies this back to the arcs

This code runs on blue reduced twilight sky frames and is called as part
of the reduce_sky() pipeline command. For each blue twilight frame it derives
an offset in angstroms between the nominal wavelength solution for that frame
and the 'true' wavelenght solution determined by correlating with a high
resolution solar spectrum.

After individual solutions are calculated for each twilight, this code determines
an overall fibre-to-fibre wavecal correction by removing a linear shape term then
averaging over all twilights in a run.

This fibre-to-fibre wavecal correction is then applied to the wavelength solution
stored in an extension in each arc frame, to be propagated by 2dfDR through the
object frames.

NB This is most successful when an additional wavecal refinement step based on the
5577 sky line is applied by 2dfDR

"""

import astropy.io.fits as pf
import numpy as np
from astrop.io import Table
import os

def wavecorr_frame(fits):

    reduced_path = fits.reduced_path
    with pf.open(reduced_path,mode='update') as twilight_hdulist:
        offsets = calculate_wavelength_offsets(twilight_hdulist)
        offsets = remove_slope(offsets)
        record_wavelength_offsets(twilight_hdulist,offsets)
    
def remove_slope(offsets):

    x = np.arange(len(offsets))
    p = np.poly1d(np.polyfit(x,offsets,1))
    
    offsets_flat = offsets - p(x) + 0.1
    
    return offsets_flat
        
    
def calculate_wavelength_offsets(twilight_hdu):

	hdulist_solar = pt.open('/Users/nscott/Data/fts-atlas-interp-sami.fits')
	solar_flux = hdulist_solar[0].data
	sh = hdulist_solar[0].header
	solar_wav = np.arange(sh['NAXIS1'])*sh['CDELT1'] + sh['CRVAL1']
	
	twi_head = twilight_hdu[0].header
	twi_wav = (np.arange(twi_head['NAXIS1']) - twi_head['CRPIX1'])*twi_head['CDELT1'] + twi_head['CRVAL1']
	
	solar_shifted = np.roll(solar_flux,-100)
	
	twilight_frame = twilight_hdu[0].data
	
	offsets = []
	for i in range(twilight_frame.shape[0]):
		fibre_spec, fibre_wav = prepare_fibre_spectrum(twilight_hdu[0].data[i,:],twi_wav,solar_wav)
		offset = calculate_wavelength_offset_fibre(fibre_spec,np.copy(solar_shifted))
		offset = offset*(solar_wav[1]-solar_wav[0])
		offsets.append(offset)
		
	return offsets
	
def calculate_wavelength_offset_fibre(fib,sol):
	
	diffs = []
	for i in range(201):
		diff = fib - sol
		sol = np.roll(sol,1)
		diffs.append(np.sqrt(np.nanmean(diff**2)))
		
	best_diff = np.argmin(diffs)
	best_diff = best_diff - 100
	return best_diff
	
def prepare_fibre_spectrum(fibre_spec, fibre_wav, solar_wav):
	# Pre-process a SAMI twilight sky fibre spectrum. This involves
	# removing the shape of the spectrum using a polynomial normalisation
	# then interpolating onto the same wavelength scale as the solar spectrum
	
	tmp_spec = fibre_spec[np.where(np.isfinite(fibre_spec))]
	tmp_wav = fibre_wav[np.where(np.isfinite(fibre_spec))]
	p = np.polyfit(tmp_wav,tmp_spec,20)
	f = np.poly1d(p)
	spec_fit = f(fibre_wav)
	norm_spec = fibre_spec/spec_fit
	
	hr_spec = np.interp(solar_wav,fibre_wav,norm_spec)
	hr_wav = np.copy(solar_wav)
	
	return hr_spec, hr_wav

	
def record_wavelength_offsets(twilight_hdulist,offsets):

    # Save the derived wavelength offsets for a given twilight frame to
    # a new 'WAVECORR' extension or replace values if extension already exists

    if 'WAVECORR' in twilight_hdulist:
        twilight_hdulist['WAVECORR'].data = offsets
    else:
        h = pf.ImageHDU(offsets)
        h.header['EXTNAME'] = 'WAVECORR'
        h.header['CUNIT1'] = ('Angstroms','Units for axis 1')
        h.header['CTYPE1'] = ('Delta Wavelength','Wavelength offset for fibre')

    twilight_hdulist.append(h)
    twilight_hdulist.flush()
    
def wavecorr_av(path_list,root_dir,overwrite=True)

    # For all reduced twilight sky frames with 'WAVECORR' extensions:
    # 1) Read in their 'WAVECORR' offsets array
    # 2) Fit and subtract a linear shape term to 
    #   leave only the fibre-to-fibre variations
    # 3) Median over all offset arrays to derive the median fibre-to-fibre 
    #   wavelength variation
    # 4) Write this to a new file in relevant calibration folders (CHECK THIS)
    
    hdu = fits.open(path_list[0])
    
    offsets = np.zeros((len(hdu['WAVECORR'].data),len(path_list)))
    hdu.close()
    for i,path in enumerate(path_list):
        offset = fits.getdata(path,'WAVECORR')
        offsets[:,i] = offset
    
    offsets_av = np.nanmedian(offsets,axis=1)
    
    tb = Table(offsets_av,names='Offset')
    tb.write(os.path.join(root_dir,'average_blue_wavelength_offset.dat'),format='ascii.commented_header')
    
def apply_wavecorr(path,root_dir):

    if not os.path.isfile(os.path.join(root_dir,'average_blue_wavelength_offset.dat')):
        print('No average wavelength correction file found.') 
        print('Wavelength correction not applied')
        return
        
    tb = Table.read(os.path.join(root_dir,'average_blue_wavelength_offset.dat'))
    offsets = tb['Offset'].data
    
    hdulist = fits.open(path,'update')
    if 'MNGRTWCR' in hdu[0].header:
        if hdulist[0].header['MNGRTWCR'] != 'T':
            hdulist['SHIFTS'].data[0] = hdulist['SHIFTS'].data[0] + offsets
            hdulist[0].header['MNGRTWCR'] = 'T'
    else:
        hdulist['SHIFTS'].data[0] = hdulist['SHIFTS'].data[0] + offsets
        hdulist[0].header['MNGRTWCR'] = 'T'        
    
    
    
	