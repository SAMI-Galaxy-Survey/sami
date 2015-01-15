import matplotlib
import pylab as py
import numpy as np
import astropy.io.fits as pf
from scipy.ndimage.filters import median_filter

from .fluxcal import get_coords

##############################################################################
# script to test sky subtraction accuracy.  Runs on a single RSS frame.  

def sky_residuals(infile, fibstart=1, fibend=2000, allfib=False, verbose=False,
                  plot=False):
    """
    Function to check sky subtraction accuracy in SAMI data.
    
    It returns the average and median continuum and line sky flux and the
    residuals for these.
    
    the optional flags are:
    fibstart  - first fibre to use, starting at fibre 1 (i.e. not zero indexed).
    fibend    - last fibre to use.
    allfib    - use all fibres, not just sky.
    verbose   - output to the screen
    plot      - make plots
    """
    
    # half-width of window around the sky lines:
    hwidth = 10
    
    # open file:
    hdulist = pf.open(infile)

    # get primary header: 
    primary_header=hdulist['PRIMARY'].header

    # check file is an object frame:
    obstype=primary_header['OBSTYPE']
    # raise exception if not an object frame:
    if (obstype != 'OBJECT'):
        print 'OBSTYPE keyword = ',obstype
        raise IOError('Input file was not an OBJECT frame')
    
    # get data and variance
    im = hdulist[0].data
    var = hdulist['VARIANCE'].data

    # get array sizes:
    (ys,xs) = im.shape

    # try and get the sky spectum, raise an exception if no 
    # sky spectrum found:
    try:
        sky = hdulist['SKY'].data
    except KeyError:
        print("SKY extension not found!")
        raise IOError('No sky extension found')

    # get wavelength info:
    # crval1=primary_header['CRVAL1']
    # cdelt1=primary_header['CDELT1']
    # crpix1=primary_header['CRPIX1']
    # naxis1=primary_header['NAXIS1']
    # x=np.arange(naxis1)+1
    # L0=crval1-crpix1*cdelt1 #Lc-pix*dL        
    # lam=L0+x*cdelt1
    lam = get_coords(primary_header, 1)

    # get binary table info:
    fib_tab_hdu=find_fibre_table(hdulist)
    table_data = hdulist[fib_tab_hdu].data
    types=table_data.field('TYPE')
    
    # next add back in the sky spectrum to the data:
    im_sky = im + sky

    # define sky lines:
    #skylines=[5577.346680,6300.308594,6363.782715,6533.049805,6553.625977,6923.192383,7316.289551,7340.900879,7358.680176]
    # use a shorter version as some of the reddest lines are not always in all fibres:
    skylines=[5577.346680,6300.308594,6363.782715,6533.049805,6553.625977,6923.192383]

    # calculate some basic statistics for each sky spectrum.
    # what do we want: 
    # 1) summed residual flux as a fraction of total sky
    # 2) summed residual flux as fraction for strong emission lines

    ns = 0 
    fracs=np.zeros(ys)
    skyflux=np.zeros(ys)
    fibs=np.zeros(ys)
    line_fracs=np.zeros(ys)
    line_skyflux=np.zeros(ys)
    if (verbose):
        print '   |      Cont flux     |      Line flux        '
        print 'fib|f(sky)  f(res)  frac| f(sky)   f(res)  frac ' 
    for i in xrange(ys):
        if ((types[i] == 'S' or allfib) and (i >= fibstart-1 and i<= fibend-1)):
            spec_test = im[i,:]
            spec_test = spec_test[~np.isnan(spec_test)]
            sub_med = np.median(spec_test)

            spec_test = im_sky[i,:]
            spec_test = spec_test[~np.isnan(spec_test)]
            sky_med = np.median(spec_test)

            frac = sub_med/sky_med

            skyflux[ns]=sky_med
            fracs[ns]=frac
            
            fibs[ns]=i+1

            line_res=0
            line_flux=0
            line_res_s=0
            line_flux_s=0
            ibad=0

            nlines_used=0
            
            for line in skylines:
                # only use lines in range:
                if (line > lam[0] and line < lam[xs-1]):

                    #print 'testing...',line,lam[0],lam[xs-1]
                    ll = lam
                    ff = im
                    ss = im_sky

                    nlines_used=nlines_used+1
                
                    # get the index of the pixel nearest the sky line:

                    iloc = min(range(len(ll)), key=lambda i: abs(ll[i]-line))

                    # get the data around the sky line
                    xx = ll[iloc-hwidth:iloc+hwidth+1]
                    yy = ff[i,iloc-hwidth:iloc+hwidth+1]

                    # get median filtered continuum near the line:
                    cont = median_filter(ff[i,:],size=51)
                    cont_sky = median_filter(ss[i,:],size=51)
                    cc = cont[iloc-hwidth:iloc+hwidth+1]
                    cc_sky = cont_sky[iloc-hwidth:iloc+hwidth+1]
                
                    #sig = np.sqrt(yy)
                    # sum the flux over the line
                    line_res = np.sum(ff[i,iloc-hwidth:iloc+hwidth+1]-cont[iloc-hwidth:iloc+hwidth+1])
                    line_flux = np.sum(ss[i,iloc-hwidth:iloc+hwidth+1]-cont_sky[iloc-hwidth:iloc+hwidth+1])

                    # get the residual line flux 
                    if (np.isnan(line_res) or np.isnan(line_flux)):
                        ibad=ibad+1
                    else:
                        line_res_s = line_res_s + line_res
                        line_flux_s = line_flux_s + line_flux


                        #print 'test:',line_res_s,line_flux_s,ibad
            if (line_flux_s > 0):
                line_fracs[ns] = line_res_s/line_flux_s
                line_skyflux[ns]=line_flux_s
            else:
                line_fracs[ns] = 0.0
                line_skyflux[ns] = 0.0
                
            if (verbose):
                print '{0:3d} {1:6.2f} {2:6.2f} {3:6.3f} {4:8.2f} {5:7.2f} {6:6.3f}'.format(i+1,sky_med,sub_med,frac,line_flux_s,line_res_s,line_fracs[ns])
            #print i+1,sky_med_r,sub_med_r,frac_r,line_flux_r,line_res_r,line_fracs_r[ns]

            ns=ns+1
            #            print 'number of lines used:',nlines_used

    # get the median/mean fractional sky residuals:
    medsky_cont=np.median(abs(fracs[0:ns]))
    medsky_line=np.median(abs(line_fracs[0:ns]))
    meansky_cont=np.mean(abs(fracs[0:ns]))
    meansky_line=np.mean(abs(line_fracs[0:ns]))

    # get the median/mean fluxes:
    medskyflux_cont = np.median(abs(skyflux[0:ns]))
    medskyflux_line = np.median(abs(line_skyflux[0:ns]))
    meanskyflux_cont = np.mean(abs(skyflux[0:ns]))
    meanskyflux_line = np.mean(abs(line_skyflux[0:ns]))
    
    if (verbose):
        print 'median absolute continuum residuals:',medsky_cont
        print 'median absolute line residuals:',medsky_line
        print 'mean absolute continuum residuals:',meansky_cont
        print 'mean absolute line residuals:',meansky_line

    if (plot):
        py.figure(1)            
        lab = infile+' cont residual'
        py.plot(fibs[0:ns],fracs[0:ns],'-',color='r',label=lab)
        for i in xrange(ys):
            if (types[i] == 'S'):
                py.plot(fibs[i],fracs[i],'x',color='g')
            else:
                py.plot(fibs[i],fracs[i],'.',color='r')

                
        lab = infile+' line residual'
        py.plot(fibs[0:ns],line_fracs[0:ns],'-',color='b',label=lab)
        for i in xrange(ys):
            if (types[i] == 'S'):
                py.plot(fibs[i],line_fracs[i],'x',color='m')
            else:
                py.plot(fibs[i],line_fracs[i],'.',color='b')
                
                
        py.axhline(0.0,color='k',linestyle='--')
        py.xlabel('Fibre')
        py.ylabel('fractional sky residual')
        py.title('Fractional sky residuals')
        py.legend(prop={'size':10})
        
        # put results into a dictonary:
    sky_sub_res = {
        'med_frac_skyres_cont':medsky_cont,
        'med_frac_skyres_line':medsky_line,
        'med_skyflux_cont':medskyflux_cont,
        'med_skyflux_line':medskyflux_line,
        'mean_frac_skyres_cont':meansky_cont,
        'mean_frac_skyres_line':meansky_line,
        'mean_skyflux_cont':meanskyflux_cont,
        'mean_skyflux_line':meanskyflux_line}
        
    return sky_sub_res


#############################################################################

def find_fibre_table(hdulist):
    """Returns the extension number for FIBRES_IFU, MORE.FIBRES_IFU FIBRES or MORE.FIBRES,
    whichever is found. Modified from SAMI versiuon that only uses FIBRES_IFU.
    Raises KeyError if neither is found."""

    extno = None
    try:
        extno = hdulist.index_of('FIBRES')
    except KeyError:
        pass

    if extno is None:
        try:
            extno = hdulist.index_of('MORE.FIBRES')
        except KeyError:
            pass

    if extno is None:            
        try:
            extno = hdulist.index_of('FIBRES_IFU')
        except KeyError:
            pass
        
    if extno is None:
        try:
            extno = hdulist.index_of('MORE.FIBRES_IFU')
        except KeyError:
            raise KeyError("Extensions 'FIBRES_IFU' and "
                           "'MORE.FIBRES_IFU' both not found")
    return extno
