'''
Created on Oct 20, 2014

@author: agreen
'''

from datetime import datetime

import numpy as np

# Set up logging
from .. import slogging
log = slogging.getLogger(__name__)
log.setLevel(slogging.DEBUG)



# astropy fits file io (replacement for pyfits)
import astropy.io.fits as pf

class KoalaIFU(object):

    def __init__(self, rss_filename):
        """A class containing data and other information from a single file pertaining to a particular object or
        probe."""
        
        self.rss_file = rss_filename

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

        fibre_table = hdulist['FIBRES_IFU'].data

        # Some useful stuff from the header
        self.exptime = self.primary_header['EXPOSED']
        self.crval1 = self.primary_header['CRVAL1']
        self.cdelt1 = self.primary_header['CDELT1']
        self.crpix1 = self.primary_header['CRPIX1']
        self.naxis1 = self.primary_header['NAXIS1']

        self.meanra = self.primary_header['MEANRA']
        self.meandec = self.primary_header['MEANDEC']

        # datetime object representing the date of the observation. See
        # https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior
        self.obs_date = datetime.strptime(
            self.primary_header['UTDATE'],
            "%Y:%m:%d")

        # Determine and store which spectrograph ARM this is from (red/blue)

        if (self.primary_header['SPECTID'] == 'BL'):
            self.spectrograph_arm = 'blue'
        elif (self.primary_header['SPECTID'] == 'RD'):
            self.spectrograph_arm = 'red'

        self.gratid = self.primary_header['GRATID']
        self.gain = self.primary_header['RO_GAIN']

        self.zdstart=self.primary_header['ZDSTART']
        self.zdend=self.primary_header['ZDEND']
        
        # Wavelength range
        x=np.arange(self.naxis1)+1
        L0=self.crval1-self.crpix1*self.cdelt1 #Lc-pix*dL
        self.lambda_range = L0+x*self.cdelt1

        # Koala has only one IFU
        self.ifu = 0
                    
        #self.fibre_offsets_arcsec_ra = fibre_table.field("XPOS")
        #self.fibre_offsets_arcsec_dec = fibre_table.field("YPOS")
        
        # Details about the KOALA IFU for drizzling:

        # Plate scale in arcseconds per millimeter (1000.0 means the input
        # positions are in arcseconds instead of microns)
        self.plate_scale = 1000.0

        if self.fibre_table_header['FOV'].strip() == "Normal":
            # Fibre diameter for Small field of view
            self.fibre_diameter_arcsec = 0.7
        else:
            # Fibre diameter for Small field of view
            self.fibre_diameter_arcsec = 1.25



        #X and Y positions of fibres in absolute degrees.
        #self.xpos=table_new.field('FIB_MRA') #RA add -1*
        #self.ypos=table_new.field('FIB_MDEC') #Dec

        # Positions in arcseconds relative to the field centre
        # Rotate RA and Decs from fibre tables to get the orientation right for early KOALA data:
        # This was fixed before the run starting 26 March 2015

        if self.obs_date < datetime(2015, 3, 26):
            print("Rotating KOALA field to correct for fibre table issues.")
            if self.fibre_table_header['INST_ROT'] > 91 or self.fibre_table_header['INST_ROT'] < 89:
                print("WARNING: Instrument rotation not in expected range. Output orientation may be wrong.")
            theta = np.radians(90.0)
            self.fibre_ra_offset_arcsec = np.cos(theta) * fibre_table.field('XPOS') - np.sin(theta) * fibre_table.field('YPOS')
            self.fibre_dec_offset_arcsec = np.sin(theta) * fibre_table.field('XPOS') + np.cos(theta) * fibre_table.field('YPOS')
            del theta
        elif True:
            if self.fibre_table_header['INST_ROT'] < -91 or self.fibre_table_header['INST_ROT'] > -89:
                print("WARNING: Instrument rotation not in expected range. Output orientation may be wrong.")
            theta = np.radians(-90.0)
            # Note, the RA offset has its sign changed: I believe this is
            # because RA increases to the left, but pixels increase to the
            # right. The original cubing code for SAMI is designed to work in
            # microns, so SAMI does not need to be reversed in this way.
            self.fibre_ra_offset_arcsec = -(np.cos(theta) * fibre_table.field('XPOS') - np.sin(theta) * fibre_table.field('YPOS'))
            self.fibre_dec_offset_arcsec = np.sin(theta) * fibre_table.field('XPOS') + np.cos(theta) * fibre_table.field('YPOS')
            del theta
        else:
            self.fibre_ra_offset_arcsec = fibre_table.field('XPOS')
            self.fibre_dec_offset_arcsec = fibre_table.field('YPOS')

        # Correct for errors in the KOALA Fibre Mapping (koala_fibres.txt):
        # The positions of fibres 305 and 306 were swapped in data prior to 29
        # March 2015, and must be corrected:
        if self.obs_date < datetime(2015, 3, 29):
            print("Swapping fibre 305 and 306.")
            # Remember python arrays are zero-indexed, so we must subtract one from the fibre number
            tmp = (self.fibre_ra_offset_arcsec[304], self.fibre_dec_offset_arcsec[304])
            self.fibre_ra_offset_arcsec[304] = self.fibre_ra_offset_arcsec[305]
            self.fibre_dec_offset_arcsec[304] = self.fibre_dec_offset_arcsec[305]
            self.fibre_ra_offset_arcsec[305] = tmp[0]
            self.fibre_dec_offset_arcsec[305] = tmp[1]
            del tmp
            
        # Fibre number - used for tests.
        #self.n=fibre_table.field('FIBNUM')
    
        # Fibre designation.
        self.fib_type = fibre_table.field('TYPE')
        
        # Probe Name
        #self.hexabundle_name = table_new.field('PROBENAME')
        
        # Adding for tests only - LF 05/04/2012
        #self.x_microns=-1*table_new.field('FIBPOS_X') # To put into on-sky frame
        #self.y_microns=table_new.field('FIBPOS_Y')
        
        # Name of object
        #name_tab=table_new.field('NAME')
        #self.name=name_tab[0]
        
        # indices of the corresponding spectra (SPEC_ID counts from 1, image counts from 0)
        #ind=table_new.field('SPEC_ID')-1
        
        self.data = data_in/self.exptime
        self.var = variance_in/(self.exptime*self.exptime)

        # Master sky spectrum:
        # try:
        #     self.sky_spectra = hdulist['RWSS'].data
        # except KeyError:
        #     pass
        #    TODO: It would be more useful to have the sky spectrum subtracted from
        #    each fibre which requires the RWSS file/option in 2dfdr

        # 2dfdr determined fibre througput corrections
        try:
            self.fibre_throughputs = hdulist['THPUT'].data
        except KeyError:
            # not available, never mind
            pass

        # Added for Iraklis, might need to check this.
#        self.fibtab=table_new

        # TEMP -  object RA & DEC (Nic)
#        self.obj_ra=table_new.field('GRP_MRA')
#        self.obj_dec=table_new.field('GRP_MDEC')

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

        # Object RA & DEC
        self.obj_ra = self.primary_header["MEANRA"]
        self.obj_dec = self.primary_header["MEANDEC"]

        # Cleanup to reduce memory footprint
        del hdulist
        
    @property
    def xpos_rel(self):
        log.warn("xpos_rel attribute is deprecated.")
        return self.fibre_ra_offset_arcsec

    @property
    def ypos_rel(self):
        log.warn("ypos_rel attribute is deprecated.")
        return self.fibre_dec_offset_arcsec