"""
This module contains the IFU class, used extensively throughout sami.

An IFU instance contains data from a single IFU's observation. As well as
the observed flux, it stores the variance and a lot of metadata. See the
code for everything that's copied.

One quirk to be aware of: the data on disk are stored in terms of total
counts, but the IFU object automatically scales this by exposure time to
get a flux.
"""

import numpy as np

# astropy fits file io (replacement for pyfits)
import astropy.io.fits as pf
# extra astropy bits to calculate airmass
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from .fluxcal2_io import read_model_parameters

class IFU:

    def __init__(self, rss_filename, probe_identifier, flag_name=True):
        """A class containing data and other information from a single file pertaining to a particular object or
        probe."""
        
        self.infile=rss_filename

        # Open the file (should I really be doing this here?)
        hdulist=pf.open(rss_filename)

        data_in=hdulist['PRIMARY'].data
        variance_in=hdulist['VARIANCE'].data
        
        # Load all necessary metadata
        self.primary_header = hdulist['PRIMARY'].header
        self.fibre_table_header = hdulist['FIBRES_IFU'].header
        self.reduction_arguments = hdulist['REDUCTION_ARGS'].data 

        #TEMP - store full headers (Nic)
        self.primary_header = hdulist['PRIMARY'].header
        self.fibre_table_header = hdulist['FIBRES_IFU'].header
        self.reduction_arguments = hdulist['REDUCTION_ARGS'].header

        fibre_table=hdulist['FIBRES_IFU'].data

        # Some useful stuff from the header
        self.exptime = self.primary_header['EXPOSED']
        self.crval1 = self.primary_header['CRVAL1']
        self.cdelt1 = self.primary_header['CDELT1']
        self.crpix1 = self.primary_header['CRPIX1']
        self.naxis1 = self.primary_header['NAXIS1']

        # Field centre values (not bundle values!)
        self.meanra = self.primary_header['MEANRA']
        self.meandec = self.primary_header['MEANDEC']


        # Determine and store which spectrograph ARM this is from (red/blue)

        if (self.primary_header['SPECTID'] == 'BL'):
            self.spectrograph_arm = 'blue'
        elif (self.primary_header['SPECTID'] == 'RD'):
            self.spectrograph_arm = 'red'

        self.gratid = self.primary_header['GRATID']
        self.gain = self.primary_header['RO_GAIN']

        self.zdstart=self.primary_header['ZDSTART']
        self.zdend=self.primary_header['ZDEND']

        # get header keywords needed to calculate airmass, seperately from ZD in header.
        # this is because ZDEND is sometimes not present, or not correct.  The current
        # airmass calc (3/10/19) later in the pipeline assumes ZDEND=ZDSTART.  Instead,
        # here we will calc the ZD and airmass based on times and coords.
        self.utdate = self.primary_header['UTDATE']
        self.utstart = self.primary_header['UTSTART']
        self.utend = self.primary_header['UTEND']
        self.lat_obs = self.primary_header['LAT_OBS']
        self.long_obs = self.primary_header['LONG_OBS']
        self.alt_obs = self.primary_header['ALT_OBS']

        # define observatory location:
        obs_loc = EarthLocation(lat=self.lat_obs*u.deg, lon=self.long_obs*u.deg, height=self.alt_obs*u.m)

        # Convert to the correct time format:
        date_formatted = self.utdate.replace(':','-')
        time_start = date_formatted+' '+self.utstart
        # note that here we assume UT date start is the same as UT date end.  This works for
        # the AAT, given the time difference from UT at night, but will not for other observatories.
        time_end = date_formatted+' '+self.utend
        time1 = Time(time_start) 
        time2 = Time(time_end) 
        time_diff = time2-time1
        time_mid = time1 + time_diff/2.0

        # define coordinates using astropy coordinates object:
        coords = SkyCoord(self.meanra*u.deg,self.meandec*u.deg) 

        # calculate alt/az using astropy coordinate transformations:
        altazpos1 = coords.transform_to(AltAz(obstime=time1,location=obs_loc))   
        altazpos2 = coords.transform_to(AltAz(obstime=time2,location=obs_loc))
        altazpos_mid = coords.transform_to(AltAz(obstime=time_mid,location=obs_loc))   

        # convert to ZD at start, end and midpoint, removing the degrees units
        # put in by astropy:
        zd1 = 90.0-altazpos1.alt/u.deg
        zd2 = 90.0-altazpos2.alt/u.deg
        zd_mid = 90.0-altazpos_mid.alt/u.deg

        # convert back to altitude and use float() so that this is not an object with
        # (dimensionless units), but actually a simple float:
        alt1 = float(90.0 - zd1) 
        alt2 = float(90.0 - zd2)
        alt_mid = float(90.0 - zd_mid)

        # calc airmass at the start, end and midpoint:
        airmass1 = 1./ ( np.sin( ( alt1 + 244. / ( 165. + 47 * alt1**1.1 )
                            ) / 180. * np.pi ) )
        airmass2 = 1./ ( np.sin( ( alt2 + 244. / ( 165. + 47 * alt2**1.1 )
                            ) / 180. * np.pi ) )
        airmass_mid = 1./ ( np.sin( ( alt_mid + 244. / ( 165. + 47 * alt_mid**1.1 )
                            ) / 180. * np.pi ) )

        # get effective airmass by simpsons rule integration:
        self.airmass_eff = ( airmass1 + 4. * airmass_mid + airmass2 ) / 6.

        #print('effective airmass:',self.airmass_eff)
        #print('ZD start:',self.zdstart)
        #print('ZD start (calculated):',zd1)
        
        # check that the ZD calculated actually agrees with the ZDSTART in the header
        d_zd = abs(zd1-self.zdstart)
        if (d_zd>0.1):
            print('WARNING: calculated ZD different from ZDSTART.  Difference:',d_zd)
            # if we have this problem, assume that the ZDSTART header keyword is correct
            # and that one or more of the other keywords has a problem.  Then set
            # the effective airmass to be based on ZDSTART:
            alt1 = 90.0-self.zdstart
            self.airmass_eff = 1./ ( np.sin( ( alt1 + 244. / ( 165. + 47 * alt1**1.1 )
                                ) / 180. * np.pi ) )
            
            
        
        # Wavelength range
        x=np.arange(self.naxis1)+1
        
        L0=self.crval1-self.crpix1*self.cdelt1 #Lc-pix*dL
        
        self.lambda_range=L0+x*self.cdelt1

        # Based on the given information (probe number or object name) find the other piece of information. NOTE - this
        # will fail for unassigned probes which will have empty strings as a name.
        if flag_name==True:
            if len(probe_identifier)>0:
                self.name=probe_identifier # Flag is true so we're selecting on object name.
                msk0=fibre_table.field('NAME')==self.name # First mask on name.
                table_find=fibre_table[msk0] 

                # Find the IFU name from the find table.
                self.ifu=np.unique(table_find.field('PROBENUM'))[0]

            else:
                # Write an exception error in here?
                pass
            
        else:
            self.ifu=probe_identifier # Flag is not true so we're selecting on probe (IFU) number.
            
            msk0=fibre_table.field('PROBENUM')==self.ifu # First mask on probe number.
            table_find=fibre_table[msk0]

            # Pick out the place in the table with object names, rejecting SKY and empty strings.
            object_names_nonsky = [s for s in table_find.field('NAME') if s.startswith('SKY')==False and s.startswith('Sky')==False and len(s)>0]
            #print np.shape(object_names_nonsky)

            self.name=list(set(object_names_nonsky))[0]
            
        mask=np.logical_and(fibre_table.field('PROBENUM')==self.ifu, fibre_table.field('NAME')==self.name)
        table_new=fibre_table[mask]

        # Mean RA of probe centre, degrees
        self.ra = table_new.field('GRP_MRA')[0]
        # Mean Dec of probe centre, degrees
        self.dec = table_new.field('GRP_MDEC')[0]


        #X and Y positions of fibres in absolute degrees.
        self.xpos=table_new.field('FIB_MRA') #RA add -1*
        self.ypos=table_new.field('FIB_MDEC') #Dec

        # Positions in arcseconds relative to the field centre
        self.xpos_rel=table_new.field('XPOS')
        self.ypos_rel=table_new.field('YPOS')
 
        # Fibre number - used for tests.
        self.n=table_new.field('FIBNUM')
    
        # Fibre designation.
        self.fib_type=table_new.field('TYPE')
        
        # Probe Name
        self.hexabundle_name=table_new.field('PROBENAME')
        
        # Adding for tests only - LF 05/04/2012
        self.x_microns=-1*table_new.field('FIBPOS_X') # To put into on-sky frame
        self.y_microns=table_new.field('FIBPOS_Y')
        
        # Name of object
        name_tab=table_new.field('NAME')
        self.name=name_tab[0]
        
        # indices of the corresponding spectra (SPEC_ID counts from 1, image counts from 0)
        ind=table_new.field('SPEC_ID')-1
        
        self.data=data_in[ind,:]/self.exptime
        self.var=variance_in[ind,:]/(self.exptime*self.exptime)

        # Master sky spectrum:
        self.sky_spectra = hdulist['SKY'].data
        #    TODO: It would be more useful to have the sky spectrum subtracted from
        #    each fibre which requires the RWSS file/option in 2dfdr

        # 2dfdr determined fibre througput corrections
        try:
            self.fibre_throughputs = hdulist['THPUT'].data[ind]
        except KeyError:
            # None available; never mind.
            pass

        # Added for Iraklis, might need to check this.
        self.fibtab=table_new

        # TEMP -  object RA & DEC (Nic)
        self.obj_ra=table_new.field('GRP_MRA')
        self.obj_dec=table_new.field('GRP_MDEC')

        # Pre-measured offsets, if available
        try:
            offsets_table = hdulist['ALIGNMENT'].data
        except KeyError:
            # Haven't been measured yet; never mind
            pass
        else:
            line_number = np.where(offsets_table['PROBENUM'] == self.ifu)[0][0]
            offsets = offsets_table[line_number]
            self.x_cen = -1 * offsets['X_CEN'] # Following sign convention for x_microns above
            self.y_cen = offsets['Y_CEN']
            self.x_refmed = -1 * offsets['X_REFMED']
            self.y_refmed = offsets['Y_REFMED']
            self.x_shift = -1 * offsets['X_SHIFT']
            self.y_shift = offsets['Y_SHIFT']

        # Fitted DAR parameters, if available
        try:
            hdu_fluxcal = hdulist['FLUX_CALIBRATION']
        except KeyError:
            # Haven't been measured yet; never mind
            pass
        else:
            self.atmosphere = read_model_parameters(hdu_fluxcal)[0]
            del self.atmosphere['flux']
            del self.atmosphere['background']

        # Object RA & DEC
        self.obj_ra=table_new.field('GRP_MRA')
        self.obj_dec=table_new.field('GRP_MDEC')

        # Cleanup to reduce memory footprint
        del hdulist
