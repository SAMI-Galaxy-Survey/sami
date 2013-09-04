
import numpy as np

# astropy fits file io (replacement for pyfits)
import astropy.io.fits as pf

class IFU:

    def __init__(self, infile, pick, flag_name=True):
        """A class containing data and other information from a single file pertaining to a particular object or
        probe."""
        
        self.infile=infile

        # Open the file (should I really be doing this here?)
        hdulist=pf.open(infile)

        data_in=hdulist['PRIMARY'].data
        variance_in=hdulist['VARIANCE'].data
        primary_header=hdulist['PRIMARY'].header

        #TEMP - store full headers (Nic)
        self.primary_header = hdulist['PRIMARY'].header
        self.fibre_table_header = hdulist['FIBRES_IFU'].header
        self.reduction_arguments = hdulist['REDUCTION_ARGS'].header

        fibre_table=hdulist['FIBRES_IFU'].data

        # Some useful stuff from the header
        self.exptime=primary_header['EXPOSED']
        self.crval1=primary_header['CRVAL1']
        self.cdelt1=primary_header['CDELT1']
        self.crpix1=primary_header['CRPIX1']
        self.naxis1=primary_header['NAXIS1']

        self.meanra=primary_header['MEANRA']
        self.meandec=primary_header['MEANDEC']

        # TEMP - determine and store which spectrograph ARM this is from (Nic)
        if (self.primary_header['SPECTID'] == 'BL'):
            self.spectrograph_arm = 'blue'
        elif (self.primary_header['SPECTID'] == 'RD'):
            self.spectrograph_arm = 'red'

        self.gratid=primary_header['GRATID']
        self.gain=primary_header['RO_GAIN']
        
        # Wavelength range
        x=np.arange(self.naxis1)+1
        
        L0=self.crval1-self.crpix1*self.cdelt1 #Lc-pix*dL
        
        self.lambda_range=L0+x*self.cdelt1

        # Based on the given information (probe number or object name) find the other piece of information. NOTE - this
        # will fail for unassigned probes which will have empty strings as a name.
        if flag_name==True:
            if len(pick)>0:
                self.name=pick # Flag is true so we're selecting on object name.
                msk0=fibre_table.field('NAME')==self.name # First mask on name.
                table_find=fibre_table[msk0] 

                # Find the IFU name from the find table.
                self.ifu=np.unique(table_find.field('PROBENUM'))[0]

            else:
                # Write an exception error in here?
                pass
            
        else:
            self.ifu=pick # Flag is not true so we're selecting on probe (IFU) number.
            
            msk0=fibre_table.field('PROBENUM')==self.ifu # First mask on probe number.
            table_find=fibre_table[msk0]

            # Pick out the place in the table with object names, rejecting SKY and empty strings.
            object_names_nonsky = [s for s in table_find.field('NAME') if s.startswith('SKY')==False and s.startswith('Sky')==False and len(s)>0]
            #print np.shape(object_names_nonsky)

            self.name=list(set(object_names_nonsky))[0]
            
        mask=np.logical_and(fibre_table.field('PROBENUM')==self.ifu, fibre_table.field('NAME')==self.name)
        table_new=fibre_table[mask]

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
        
        # Object RA & DEC
        self.fibtab=table_new

        # TEMP -  object RA & DEC (Nic)
        self.obj_ra=table_new.field('GRP_MRA')
        self.obj_dec=table_new.field('GRP_MDEC')

        del hdulist
