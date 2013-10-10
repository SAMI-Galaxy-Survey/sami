""" SDSS throughput library for SNR and surface brightness codes. """
import numpy as np

# g' wavelength and throughput
sdss_wave_g = np.arange(3630, 5830)
sdss_thru_g = np.array([\
        0.000e+00, 3.000e-04, 8.000e-04, 1.300e-03, 1.900e-03, 2.400e-03,
        3.400e-03, 5.500e-03, 1.030e-02, 1.940e-02, 3.260e-02, 4.920e-02,
        6.860e-02, 9.000e-02, 1.123e-01, 1.342e-01, 1.545e-01, 1.722e-01,
        1.873e-01, 2.003e-01, 2.116e-01, 2.214e-01, 2.301e-01, 2.378e-01,
        2.448e-01, 2.513e-01, 2.574e-01, 2.633e-01, 2.691e-01, 2.747e-01,
        2.801e-01, 2.852e-01, 2.899e-01, 2.940e-01, 2.979e-01, 3.016e-01,
        3.055e-01, 3.097e-01, 3.141e-01, 3.184e-01, 3.224e-01, 3.257e-01,
        3.284e-01, 3.307e-01, 3.327e-01, 3.346e-01, 3.364e-01, 3.383e-01,
        3.403e-01, 3.425e-01, 3.448e-01, 3.472e-01, 3.495e-01, 3.519e-01,
        3.541e-01, 3.562e-01, 3.581e-01, 3.597e-01, 3.609e-01, 3.613e-01,
        3.609e-01, 3.595e-01, 3.581e-01, 3.558e-01, 3.452e-01, 3.194e-01,
        2.807e-01, 2.339e-01, 1.839e-01, 1.352e-01, 9.110e-02, 5.480e-02,
        2.950e-02, 1.660e-02, 1.120e-02, 7.700e-03, 5.000e-03, 3.200e-03,
        2.100e-03, 1.500e-03, 1.200e-03, 1.000e-03, 9.000e-04, 8.000e-04,
        6.000e-04, 5.000e-04, 3.000e-04, 1.000e-04, 0.000e+00])

# r' wavelength and throughput
sdss_wave_r = np.arange(5380, 7231, 25)
sdss_thru_g = np.array([\
        0.000e+00, 1.400e-03, 9.900e-03, 2.600e-02, 4.980e-02, 8.090e-02,
         1.190e-01, 1.630e-01, 2.100e-01, 2.564e-01, 2.986e-01, 3.339e-01,
         3.623e-01, 3.849e-01, 4.027e-01, 4.165e-01, 4.271e-01, 4.353e-01,
         4.416e-01, 4.467e-01, 4.511e-01, 4.550e-01, 4.587e-01, 4.624e-01,
         4.660e-01, 4.692e-01, 4.716e-01, 4.731e-01, 4.740e-01, 4.747e-01,
         4.758e-01, 4.776e-01, 4.800e-01, 4.827e-01, 4.854e-01, 4.881e-01,
         4.905e-01, 4.926e-01, 4.942e-01, 4.951e-01, 4.955e-01, 4.956e-01,
         4.958e-01, 4.961e-01, 4.964e-01, 4.962e-01, 4.953e-01, 4.931e-01,
         4.906e-01, 4.873e-01, 4.752e-01, 4.474e-01, 4.059e-01, 3.544e-01,
         2.963e-01, 2.350e-01, 1.739e-01, 1.168e-01, 6.970e-02, 3.860e-02,
         2.150e-02, 1.360e-02, 1.010e-02, 7.700e-03, 5.600e-03, 3.900e-03,
         2.800e-03, 2.000e-03, 1.600e-03, 1.300e-03, 1.000e-03, 7.000e-04,
         4.000e-04, 2.000e-04, 0.000e+00])
