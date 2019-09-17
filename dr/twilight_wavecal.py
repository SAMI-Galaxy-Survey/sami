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
from astropy.table import Table
import os,code,warnings

warnings.simplefilter('ignore',np.RankWarning)

def wavecorr_frame(inputs):

    fits,overwrite = inputs
    reduced_path = fits.reduced_path
    with pf.open(reduced_path,mode='update') as twilight_hdulist:
        if ('WAVECORR' not in twilight_hdulist) | (overwrite == True):
            offsets = calculate_wavelength_offsets(twilight_hdulist)
            offsets = remove_slope(offsets)
            record_wavelength_offsets(twilight_hdulist,offsets)
    
def remove_slope(offsets):
    # Remove the linear shape of the wavelength offset function, leaving the
    # fibre-to-fibre variation and the mean offset

    x = np.arange(len(offsets))
    p = np.poly1d(np.polyfit(x,offsets,1))
    
    offsets_flat = offsets - p(x) + offsets[int((len(offsets)+1)/2)]
    
    return offsets_flat

def vac_to_air(wav):
    
    wav = np.asarray(wav)
    sigma2 = (1e4/wav)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
    
    return wav/fact
        

def calculate_wavelength_offsets(twilight_hdu):
    # Wrapper function to apply the offset measurement to all fibres in a frame
    
    # Offset is determined wrt a high-resolution solar spectrum, and has unit Angstoms

    hdulist_solar = pf.open('./standards/solar/fts-atlas-interp-sami.fits')
    solar_flux = hdulist_solar[0].data
    sh = hdulist_solar[0].header
    solar_wav = np.arange(sh['NAXIS1'])*sh['CDELT1'] + sh['CRVAL1']
    solar_wav = vac_to_air(solar_wav)
	
    twi_head = twilight_hdu[0].header
    twi_wav = (np.arange(twi_head['NAXIS1']) - twi_head['CRPIX1'])*twi_head['CDELT1'] + twi_head['CRVAL1']
	
    solar_shifted = np.roll(solar_flux,-500)
	
    twilight_frame = twilight_hdu[0].data

    good_range = [4500,5700]
    good_sol = np.where((solar_wav > good_range[0]) & (solar_wav < good_range[1]))  

    offsets = []
    for i in range(twilight_frame.shape[0]):
        fibre_spec, fibre_wav = prepare_fibre_spectrum(twilight_hdu[0].data[i,:],twi_wav,solar_wav)
        good_fib = np.where((fibre_wav > good_range[0]) & (fibre_wav < good_range[1]))
        offset = calculate_wavelength_offset_fibre(fibre_spec[good_fib],np.copy(solar_shifted)[good_sol])
        offset = offset*(solar_wav[1]-solar_wav[0])
        offsets.append(offset)
    return offsets
	
def calculate_wavelength_offset_fibre(fib,sol):
    # Determine the offset in pixels (of the high-res spectrum) between an input
    # single-fibre spectrum and a high resolution solar spectrum
    
    fib = fib/np.nanmedian(fib)
    sol = sol/np.nanmedian(sol)
    
    #diffs = []
    #for i in range(1001):
    #    diff = fib - sol
    #    sol = np.roll(sol,1)
    #    diffs.append(np.sqrt(np.nanmean(diff**2)))
        
    #best_diff = np.argmin(diffs)
    #best_diff = best_diff - 500
    
    sol_new = np.roll(sol,50)
    diffs = []
    for i in range(10):
        diff = fib - sol_new
        sol_new = np.roll(sol_new,100)
        diffs.append(np.sqrt(np.nanmean(diff**2)))
        
    best_diff0 = np.argmin(diffs)
    
    sol_new = np.roll(sol,best_diff0*100)
    diffs = []
    for i in range(101):
        diff = fib-sol_new
        sol_new = np.roll(sol_new,1)
        diffs.append(np.sqrt(np.nanmean(diff**2)))
        
    best_diff = np.argmin(diffs)
    best_diff = best_diff + best_diff0*100 - 500

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
    
def wavecorr_av(file_list,root_dir):

    # For all reduced twilight sky frames with 'WAVECORR' extensions:
    # 1) Read in their 'WAVECORR' offsets array
    # 2) Fit and subtract a linear shape term to 
    #   leave only the fibre-to-fibre variations
    # 3) Median over all offset arrays to derive the median fibre-to-fibre 
    #   wavelength variation
    # 4) Write this to a new file in relevant calibration folders
    
    hdu = pf.open(file_list[0].reduced_path)
    
    offsets = np.zeros((len(hdu['WAVECORR'].data),len(file_list)))
    hdu.close()
    for i,file in enumerate(file_list):
        offset = pf.getdata(file.reduced_path,'WAVECORR')
        offsets[:,i] = offset
    
    offsets_av = np.nanmedian(offsets,axis=1)
    offsets_av = np.reshape(offsets_av,(len(offsets_av),1))

    tb = Table(offsets_av,names=['Offset'])
    tb.write(os.path.join(root_dir,'average_blue_wavelength_offset.dat'),format='ascii.commented_header',overwrite=True)
    
def apply_wavecorr(path,root_dir):

    # Uses a stored average wavelength offset derived from multiple twilight sky frames
    # and corrects all blue arc frames by adjusting the 'SHIFTS' array
    
    # Offsets are ADDED I think (NS)

    if not os.path.isfile(os.path.join(root_dir,'average_blue_wavelength_offset.dat')):
        print('No average wavelength correction file found.') 
        print('Wavelength correction not applied')
        return
        
    tb = Table.read(os.path.join(root_dir,'average_blue_wavelength_offset.dat'),format='ascii.commented_header')
    offsets = tb['Offset'].data
    
    hdulist = pf.open(path,'update')
    if 'MNGRTWCR' in hdulist[0].header:
        if hdulist[0].header['MNGRTWCR'] != 'T':
            hdulist['SHIFTS'].data[0] = hdulist['SHIFTS'].data[0] + offsets
            hdulist[0].header['MNGRTWCR'] = 'T'
    else:
        hdulist['SHIFTS'].data[0] = hdulist['SHIFTS'].data[0] + offsets
        hdulist[0].header['MNGRTWCR'] = 'T'

    hdulist.close()
    
    
    
	
