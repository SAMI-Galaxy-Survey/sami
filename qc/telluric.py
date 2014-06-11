from ..dr import telluric

import astropy.io.fits as pf
import numpy as np

def compare_star_field(path_pair, output=None):
    """Compare measurements between all stars in a field."""
    n_hexa = 13
    n_pixel = pf.getval(path_pair[1], 'NAXIS1')
    result = np.zeros((n_hexa, 7, n_pixel))
    for i_hexa in xrange(n_hexa):
        telluric.derive_transfer_function(path_pair, use_probe=i_hexa+1)
        result[i_hexa, :, :] = pf.getdata(path_pair[1], 'FLUX_CALIBRATION')
    if output is not None:
        pf.PrimaryHDU(result).writeto(output)
    return result

def snr_in_all_tellurics(mngr_list):
    """Return array of S/N in telluric region in star in each frame."""
    telluric_limits = [[6850, 6960],
                       [7130, 7360],
                       [7560, 7770],
                       [8100, 8360]]
    snr_telluric = []
    snr_input = []
    snr_output = []
    central = 31 + 63*np.arange(13)
    for mngr in mngr_list:
        for fits in mngr.files(ndf_class='MFOBJECT', telluric_corrected=True,
                               min_exposure=900.0, ccd='ccd_2', name='main',
                               do_not_use=False):
            telluric_data = pf.getdata(fits.telluric_path, 'FLUX_CALIBRATION')
            corrected_data = pf.getdata(fits.telluric_path)
            corrected_noise = np.sqrt(
                pf.getdata(fits.telluric_path, 'VARIANCE'))
            uncorrected_data = pf.getdata(fits.fluxcal_path)
            uncorrected_noise = np.sqrt(
                pf.getdata(fits.fluxcal_path, 'VARIANCE'))
            header = pf.getheader(fits.telluric_path)
            wavelength = header['CRVAL1'] + header['CDELT1'] * (
                1 + np.arange(header['NAXIS1']) - header['CRPIX1'])
            in_telluric = np.zeros(header['NAXIS1'], dtype=bool)
            for telluric_limits_single in telluric_limits:
                in_telluric[(wavelength >= telluric_limits_single[0]) &
                            (wavelength <= telluric_limits_single[1])] = True
                # If there are only a few non-telluric pixels at the end of the
                # spectrum, mark them as telluric anyway, in case the primary flux
                # calibration screwed them up.
                minimum_end_pixels = 50
                n_blue_end = np.sum(wavelength < telluric_limits_single[0])
                if n_blue_end > 0 and n_blue_end < minimum_end_pixels:
                    in_telluric[wavelength < telluric_limits_single[0]] = True
                n_red_end = np.sum(wavelength > telluric_limits_single[1])
                if n_red_end > 0 and n_red_end < minimum_end_pixels:
                    in_telluric[wavelength > telluric_limits_single[1]] = True
                in_telluric[~(np.isfinite(telluric_data[0, :]) & 
                              np.isfinite(telluric_data[2, :]))] = False
            snr_telluric.append(np.median(telluric_data[5, in_telluric] / 
                                          telluric_data[6, in_telluric]))
            corrected_data = corrected_data[central, :][:, in_telluric]
            corrected_noise = corrected_noise[central, :][:, in_telluric]
            uncorrected_data = uncorrected_data[central, :][:, in_telluric]
            uncorrected_noise = uncorrected_noise[central, :][:, in_telluric]
            snr_input.append(np.median(uncorrected_data / uncorrected_noise, 
                                       axis=1))
            snr_output.append(np.median(corrected_data / corrected_noise,
                                        axis=1))
    return np.array(snr_telluric), np.array(snr_input), np.array(snr_output)
