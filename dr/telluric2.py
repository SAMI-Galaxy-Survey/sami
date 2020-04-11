import numpy as np
import subprocess, os, shutil
from astropy.io    import fits, ascii as ap_ascii
from astropy.table import Table

def TelluricCorrectPrimary(path_list,probenum,molecfit_dir=''):

    hdulist = fits.open(path_list[1])
    fibre = hdulist['FIBRES_IFU']
       
    # Load in SS flux data
    ww = np.where((fibre.data['PROBENUM'] == probenum) & (fibre.data['TYPE'] == 'P'))
    flux_data_raw = np.sum(hdulist[0].data[ww,:].squeeze(),axis=0)
    sigma_flux = np.sqrt(np.sum(hdulist[1].data[ww,:].squeeze(),axis=0))
    # Might put in an interpolation over NaNs; for now just taking a straight copy
    flux_data = flux_data_raw.copy()
    header = hdulist[0].header
    crval1 = header['CRVAL1']
    cdelt1 = header['CDELT1']
    naxis1 = header['NAXIS1']
    crpix1 = header['CRPIX1']
    wave_axis = crval1 + cdelt1 * (np.arange(naxis1) + 1 - crpix1)
    
    transfer_function, sigma_transfer, corrected_flux = TelluricCorrect(path_list[1],flux_data,
                sigma_flux,wave_axis,mf_bin_dir=molecfit_dir,primary=True)
                
    uncorrected_flux = hdulist[0].data.copy()
    hdulist[0].data*= transfer_function
    hdulist[1].data = hdulist[0].data**2 * ((sigma_transfer/transfer_function)**2 +
                                            hdulist[1].data/uncorrected_flux**2)    
    
    path_telluric_corrected = path_list[1].replace('red.fits','fcal.fits')
    hdulist.writeto(path_telluric_corrected,overwrite=True)
    
    return [path_list[0],path_telluric_corrected]


def TelluricCorrect(fcal_fname, star_flux, star_flux_err, wave, mf_bin_dir = '', 
                    wrange_include='none', delete_files=False, quiet=True, primary=False):

    print('Running molecfit on {}'.format(fcal_fname))

    """ Perform a telluric correction for every each fiber spectum using ESO's molecfit software """
    # Using the model correction determined by the molecfit software for the secondary standard 
    # star, correct all the fiber spectra in the fcal primary data extension.
    
    #_______________________________________ OUTPUT FILES _________________________________________#
    
    if not primary:
        obs_name = fcal_fname.replace('fcal.fits', '') # Name of observation (i.e "06mar20040")
    else:
        obs_name = fcal_fname.replace('red.fits','')
    obs_name_root = os.path.basename(obs_name)
    gal_list_fname  = obs_name + '/galaxy_list.txt' # File containing list of fnames of gal spec
    gal_spec_dir    = obs_name + '/spec_files/'     # directory to hold galaxy spectra table files
    param_fname     = obs_name + '/param_file.par'  # parameter file (Molecfit Input file)
    star_spec_fname = obs_name + '/star.dat'        # ascii table containing star spectrum
    mf_output_dir   = obs_name + '/molecfit_output/'# molecfit output directory name
    mf_script_fname = obs_name + '/molecfit.sh'     # script name which calls molecfit 
    #mf_bin_dir      = '/Users/nscott/Reduction/molecfit/bin' # directory for molecfit binary files
    
    # Check that the above directories exists, making them if they arent
    for directory in [gal_spec_dir, mf_output_dir]:
        if not os.path.isdir(directory):
            os.makedirs(os.path.dirname(directory))

    #----------------------------------------------------------------------------------------------#        
    #_________________________ EXTRACT SPECTRAL INFO FROM FCAL.FITS FILE __________________________#
    # Need to find the ID of the fiber at the centre of the secondary standard star bundle.

    with fits.open(fcal_fname) as hdu:
        # extract headers
        h0         = hdu['PRIMARY'].header  
        h1         = hdu['FIBRES_IFU'].header   
        # extract tables - not needed when getting flux from parent function
        #primary    = hdu['PRIMARY'].data
        #variance   = hdu['VARIANCE'].data
        #fibers_ifu = hdu['FIBRES_IFU'].data
        # extract name of secondary standard star
        if not primary:
            star_name  = hdu['FLUX_CALIBRATION'].header['STDNAME']
        else:
            star_name = hdu[0].header['MNGRNAME']
        
    # Air wavelength in microns
    #wave           = (h0['CRVAL1'] + h0['CDELT1'] * (np.arange(h0['NAXIS1']) - h0['CRPIX1']))*10**-4
    wave = wave*(10**-4)
    
    # identify rows of central 19 fibers in star fiber bundle
    #centrals      = np.where((fibers_ifu['NAME'] == star_name) & (fibers_ifu['FIBNUM'] <= 19))[0]
    # extract star flux and flux error
    #star_flux      = np.nansum([primary[i]          for i in centrals], axis=0)
    #star_flux_err  = np.sqrt(np.nansum([variance[i] for i in centrals], axis=0))   
    
    # convert 0.0 flux error to np.inf ? and 0.0 flux to np.nan
    star_flux[np.where(star_flux         == 0.0)] = np.nan
    star_flux_err[np.where(star_flux_err == 0.0)] = np.inf

    # Write star info to ascii file in required Molecfit format
    star_table     = Table([wave, star_flux, star_flux_err, np.isfinite(star_flux)],
                                    names=[r'#Wavelength', 'Flux', 'Flux_Err', 'Mask'])
    star_table.write(star_spec_fname, format='ascii', overwrite=True)
    #gal_fnames        = [None] * h0['NAXIS2'] # initialise list of galaxy parameter filenames

    # Need to arrange all galaxy spectra into individual files, and then provide a list of these 
    # filenames to the dictionary below. For now, include ALL fiber spectra, including the secondary
    # standard fibers and the sky fibers
    
    #for i in range(h0['NAXIS2']): # iterate through rows
    #   gal_name      = fibers_ifu['NAME'][i]   # name
    #   gal_fiber_id  = fibers_ifu['FIBNUM'][i] # fiber number
    #   gal_flux      = primary[i]              # flux
    
    #   with np.errstate(invalid='ignore'):     # ignore RuntimeWarning in sqrt for NaNs
    #       gal_flux_err = np.sqrt(variance[i]) # flux_error (noise spectrum)
    
    #   gal_fnames[i] = f"{gal_spec_dir}{gal_name}_{gal_fiber_id}.dat"      
    #   gal_table     = Table([wave, gal_flux, gal_flux_err, ~np.isnan(gal_flux)], 
    #                               names=[r'#Wavelength', 'Flux', 'Flux_Err', 'Mask'])
    #
    #   gal_table.write(gal_fnames[i], format='ascii', overwrite=True)  # write table to file
   
    #----------------------------------------------------------------------------------------------#
    #___________________________ WRITE LIST OF FILENAMES TO A TXT FILE ____________________________#
    # write the list of all the galaxy parameters filenames to a .txt file  
    
    #with open(gal_list_fname, 'w') as file:
    #   [file.write(f + '\n') for f in gal_fnames]
    
    #----------------------------------------------------------------------------------------------#
    #________________________ WRITE DICTIONARY OF PARAMATER FILE KEYWORDS _________________________#
    
    if not primary:
        cont_n = 3
    else:
        cont_n = 5

    dic = { ## INPUT DATA
       'filename'       : star_spec_fname,
       #'listname'      : gal_list_fname, # List of additional files to be corrected
       'trans'          : 1,              # type of input spectrum (transmission=1, emission=0)
       'columns'        : 'Wavelength Flux Flux_Err Mask', # input table column names
       'default_error'  : '',
       'wlgtomicron'    : 1.0,            # wavelength already converted to microns
       'vac_air'        : 'air',          # wavelength in air
       'wrange_include' : wrange_include, # wavelength range to include in fit
       'wrange_exclude' : 'none',         # wavelength range to exclude in fit
       'prange_exclude' : 'none',         # pixel      range to exclude in fit

       ## RESULTS
       'output_dir'     : mf_output_dir,  # directory of molecfit output files
       'output_name'    : obs_name_root,       # use observation name to label molecfit output files
       'plot_creation'  : ' ',            # create postscript plots
       'plot_range'     : 0,              # create plots for fit ranges (0 = nah dont)

       ## FIT PRECISION
       'ftol'           : 0.01,           # Relative chi2      convergence criterion
       'xtol'           : 0.01,           # Relative parameter convergence criterion

       ## MOLECULAR COLUMNS
       'list_molec'     : 'H2O O2',       # List of molecules to be included in the model
       'fit_molec'      : '1 1',          # Fit flags for molecules (1 = yes)
       'relcol'         : '1.0 1.0',      # Molecular column values, expressed wrt to input 
                                          # ATM profile
       ## BACKGROUND AND CONTINUUM
       'flux_unit'      : 0,
       'fit_back'       : 0,              # Fit of telescope background [bool]
       'telback'        : 0.1,            # Initial value for telescope background fit
       'fit_cont'       : 1,              # Polynomial fit of continuum --> degree: cont_n
       'cont_n'         : cont_n,              # Degree of coefficients for continuum fit
       'cont_const'     : 1.0,            # Initial constant term  for continuum fit

       ## WAVELENGTH SOLUTION
       'fit_wlc'        : 1,              # Refine wavelength solution using polynomal deg wcl_n 
       'wlc_n'          : 3,              # Degree of polynomial of refined wavelength solution
       'wlc_const'      : 0.0,            # Initial constant term for wavelength correction

       ## RESOLUTION
       'fit_res_box'    : 0,              # Fit resolution by boxcar function (0 = no)
       'relres_box'     : 0.0,            # Initial value for FWHM of boxcar wrt  slit width
       'kernmode'       : 0,              # Voigt profile instead of Gaussian & Lorentzaian (nah)
       'fit_res_gauss'  : 1,              # Fit resolution by Gaussian (1 = yes)
       'res_gauss'      : 1.0,            # Initial value for FWHM of Gaussian in pixels
       'fit_res_lorentz': 0,              # Fit resolution by Lorentzian (0 = no)
       'res_lorentz'    : 0.0,            # Initial value for FWHM of Lorentzian in pixels
       'kernfac'        : 30.0,           # Size of Gaussian/Lorentzian/Voigtian kernal in FWHM
       'varkern'        : 1,              # Variable kernel (1 = yes)
       'kernel_file'    : 'none',         # Ascii file for user defined kernal elements (optional)

       ## AMBIENT PARAMETERS
       'obsdate'        : int(np.floor(h0['UTMJD'])), # Observing date in [years] or MJD in [days]
       'utc'            : int(np.float('0.'+str(h0['UTMJD']).split('.')[1]) *24*60*60), # UTC in [s]
       'telalt'         : h0['ZDSTART'],  # Starting zeneth distance (Telescope altitude angle)[deg]
       'rhum'           : h1['ATMRHUM'] * 100, # Humidity in [%]
       'pres'           : h1['ATMPRES'],  # Pressure in hPa [millibar -> hPa is 1-1]
       'temp'           : h1['ATMTEMP'],  # Ambient temperature in [deg C]
       'm1temp'         : h1['MIRRTEMP'], # Mirror  temperature in [deg C]
       'geoelev'        : h0['ALT_OBS'],  # Elevation above sea level in [m]
       'longitude'      : h0['LONG_OBS'], # Telescope longitude
       'latitude'       : h0['LAT_OBS'],  # Telescope latitude

       ## INSTRUMENTAL PARAMETERS
       'slitw'          : 1.6,            # Fiber width in arcsec.
       'pixsc'          : 1.6 / 2.5,      # FWHM of fiber projected onto ccd = 2.5 [pix], 
                                          # hence Pixel scale in arcsec      = 1.6/2.5 [arcsec /pix]
       ## ATMOSPHERIC PROFILES
       'ref_atm'        : 'equ.atm',      # Reference atmospheric profile
       'gdas_prof'      : 'auto',         # Specific GDAS-like input profile (auto = auto retrieval)
       'layers'         : 1,              # Grid of layer heights for merging reg_atm and GDAS prof
       'emix'           : 5.0,            # Upper mixing heights in kms (5 is default)
       'pwv'            : -1.,            # input water vapour profile in mm (-1 = no scaling)
       'clean_mflux'    : 1,              # internal GUI specific parameter
        }

    # Write the above information to the parameter file
    # for each key in the dictionary, write a new line using the format
    # key : dict[key]
    with open(param_fname, 'w') as file:
        for key in dic.keys():
            file.write(key + ': ' + str(dic[key]) + '\n')
            file.write('\n')
        file.write('end \n')
    
    #----------------------------------------------------------------------------------------------#
    #__________________________ EXECUTE BASH COMMANDS TO CALL MOLECFIT  ___________________________#
    if quiet == True:
        with open(os.devnull,'w') as devnull:
            [subprocess.run([f"{mf_bin_dir}/{func}", f"{param_fname}"],stdout=devnull)
             for func in ['molecfit', 'calctrans']]#, 'corrfilelist']]  
    else:
        [subprocess.run([f"{mf_bin_dir}/{func}", f"{param_fname}"],stdout=devnull)
             for func in ['molecfit', 'calctrans']]
    #----------------------------------------------------------------------------------------------#
    #_______________________ SAVE TELLURIC CORRECTED SPECTRUM TO SCI.FITS  ________________________#
        
    #with fits.open(fcal_fname) as hdu:
    #   for i in range(hdu['PRIMARY'].header['NAXIS2']):    
    #       gal_name     = hdu['FIBRES_IFU'].data['NAME'][i]   # name
    #       gal_fiber_id = hdu['FIBRES_IFU'].data['FIBNUM'][i] # fiber number           
    #       filename     = f"{mf_output_dir}{gal_name}_{gal_fiber_id}_TAC.dat"  
            # extract the telluric corrected flux from the appropriate _TAC.dat file
    #       hdu['PRIMARY'].data[i] = ap_ascii.read(filename)['tacflux']
        
        # add a line to the fits primary header to indicate Molecfit has done the correction
    #   hdu['PRIMARY'].header['TELLURIC'] = ('Molecfit', 'ESO molecfit software used')
    #   hdu.writeto(fcal_fname.replace('fcal', 'sci'), overwrite=True) # save to sci file.      
            
    #----------------------------------------------------------------------------------------------#
    #_________________________ CLEANUP! DELETE INTERMEDIARY FILES  ________________________________#        
    # Need to delete all the intermediary files (i.e all the Molecfit specific input and output).
    # They are all found in the directory called obs_name. The sci.fits file is saved above this 
    # directory, so the whole directory can be removed  
    
    transfer_table = fits.open(f"{mf_output_dir}{obs_name_root}_tac.fits")
    transfer_data = transfer_table[1].data
    model_flux = transfer_data['cflux']
    transfer_function = 1./transfer_data['mtrans']
    sigma_transfer = star_flux_err/star_flux*transfer_function  #np.zeros(len(transfer_function))
    sigma_transfer[transfer_function == 1.] = 0.0

    # Possibly not multi-processing safe. NEED TO CHECK THIS - ignore_errors=True is a possibly dangerous fudge
    
    if delete_files:
        shutil.rmtree(obs_name,ignore_errors=True)
    
    return transfer_function, sigma_transfer, model_flux




