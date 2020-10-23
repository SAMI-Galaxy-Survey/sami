"""
Quality control for reduced arc frames.

bad_fibres() looks for arc lines that appear in the wrong place. The
results are saved to the FITS file, although currently no real use is
made of them.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import astropy.io.fits as pf
import numpy as np
from .fluxcal import get_coords

LINE_INFO_RED = {
    'windows': (
        # (6305.9, 6309.8),
        (6413.9, 6419.4),
        (6536.2, 6540.2),
        (6602.7, 6607.4),
        (6697.3, 6700.6),
        (6764.9, 6768.3),
        (7105.6, 7109.6),
        (7228.5, 7235.3),
        (7309.2, 7318.3),
        (7349.0, 7355.8),
        ),
    'continuum_windows': (
        # ((6300.0, 6305.0), (6311.5, 6316.5)),
        ((6407.5, 6412.5), (6421.0, 6426.0)),
        ((6530.0, 6535.0), (6548.5, 6553.5)),
        ((6588.7, 6593.7), (6607.4, 6612.3)),
        ((6691.0, 6696.0), (6701.5, 6706.5)),
        ((6759.3, 6764.3), (6769.0, 6774.0)),
        ((7100.0, 7105.0), (7110.0, 7115.0)),
        ((7222.5, 7227.5), (7235.8, 7240.8)),
        ((7303.5, 7308.5), (7319.5, 7324.5)),
        ((7344.0, 7349.0), (7355.8, 7360.4)),
        ),
}

LINE_INFO_BLUE = {
    'windows': (
        (3763.0, 3768.1),
        (3942.4, 3951.4),
        (4127.8, 4136.0),
        (4541.0, 4549.2),
        (4760.8, 4768.9),
        (4961.7, 4968.8),
        (5100.4, 5110.7),
        (5214.5, 5223.5),
        (5604.1, 5610.1),
        (5697.3, 5703.4),
        ),
    'continuum_windows': (
        ((3755.0, 3760.0), (3773.0, 3778.0)),
        ((3936.0, 3941.0), (3953.0, 3958.0)),
        ((4120.0, 4125.0), (4138.0, 4143.0)),
        ((4535.0, 4540.0), (4553.0, 4558.0)),
        ((4754.0, 4759.0), (4771.0, 4776.0)),
        ((4955.0, 4960.0), (4976.0, 4981.0)),
        ((5095.0, 5100.0), (5112.0, 5117.0)),
        ((5207.0, 5212.0), (5226.0, 5231.0)),
        ((5596.0, 5601.0), (5612.0, 5617.0)),
        ((5665.0, 5670.0), (5705.0, 5710.0)),
        ),
}

def measure_lines(path, line_info=None):
    """Return the flux in each fibre in each requested line."""
    # Load the data
    hdulist = pf.open(path)
    header = hdulist[0].header
    wavelength = get_coords(header, 1)
    data = hdulist[0].data
    if line_info is None:
        # Check which wavelength range we're in
        if header['SPECTID'] == 'RD':
            line_info = LINE_INFO_RED
        else:
            line_info = LINE_INFO_BLUE
    n_line = len(line_info['windows'])
    n_fibre = data.shape[0]
    flux = np.zeros((n_fibre, n_line))
    centre = np.zeros((n_fibre, n_line))
    for i_line, (window, continuum_windows) in enumerate(zip(
            line_info['windows'], line_info['continuum_windows'])):
        # Pick out the interesting data
        in_window = (wavelength >= window[0]) & (wavelength < window[1])
        data_sub = data[:, in_window].copy()
        # Subtract off the continuum
        window_low, window_high = continuum_windows
        flux_low = np.median(data[:, (wavelength >= window_low[0]) &
                                     (wavelength < window_low[1])], 1)
        flux_high = np.median(data[:, (wavelength >= window_high[0]) &
                                      (wavelength < window_high[1])], 1)
        wavelength_low = np.mean(window_low)
        wavelength_high = np.mean(window_high)
        continuum = (
            np.outer(flux_high, (wavelength[in_window] - wavelength_low) /
                     (wavelength_high - wavelength_low)) +
            np.outer(flux_low, (wavelength_high - wavelength[in_window]) /
                     (wavelength_high - wavelength_low))
        )
        data_sub = data_sub - continuum

        flux[:, i_line] = np.sum(data_sub, 1)
        centre[:, i_line] = (
            np.sum(data_sub * wavelength[in_window], 1) / flux[:, i_line])
    return flux, centre, line_info

def bad_fibres(path, n_sigma_centre=5.0, ratio_flux=2.0, save=False):
    """Return an array of fibres that appear to have bad wavelengths."""
    # Get the fluxes and positions of the arc lines
    flux, centre, line_info = measure_lines(path)
    n_fibre, n_line = flux.shape
    # These fibres have lines with poorly positioned fibres
    bad_centre = (
        np.abs(centre - np.outer(np.ones(n_fibre), np.median(centre, 0))) >
        n_sigma_centre * np.outer(np.ones(n_fibre), np.nanstd(centre, 0))
    )
    # # These fibres have discrepant fluxes
    # bad_flux = np.zeros(n_fibre, bool)
    # for i_line in range(n_line):
    #     positive_flux = (flux[:, i_line] > 0.0)
    #     bad_flux[~positive_flux] = True
    #     median_flux = np.median(flux[positive_flux, i_line])
    #     bad_flux[positive_flux] |= (
    #         np.abs(np.log(flux[positive_flux, i_line] / median_flux)) >
    #         np.log(ratio_flux)
    #     )
    # bad = bad_centre | bad_flux
    bad = bad_centre
    if save:
        hdu = pf.ImageHDU(bad.astype(int).T)
        hdu.name = 'QC_ARC'
        header = hdu.header
        for i_line, window in enumerate(line_info['windows']):
            header['WAVEL_{}'.format(i_line)] = np.mean(window)
        hdulist = pf.open(path, 'update')
        try:
            hdulist['QC_ARC']
        except KeyError:
            # No existing HDU
            hdulist.append(hdu)
        else:
            # Existing HDU; overwrite it
            hdulist['QC_ARC'] = hdu
        hdulist.flush()
        hdulist.close()
    return bad



