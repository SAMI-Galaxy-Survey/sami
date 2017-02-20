#include <stdio.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#ifdef NAN
#endif

/*
 *
 * This module provides a C++ implementation of the python function
 * `create_covar_matrix` from the module `covar`. This implementation is faster.
 *
 * There is one main function `create_covar_matrix` which calculates the
 * covariance between the SAMI spaxels, given an input weights array
 * `overlap_array` and an array of variances `variances`. For each point (x, y)
 * in the SAMI grid, the covariance is  calculated only within a square of side
 * 2 * `covarRad` + 1, centred on the point (x, y). This solution was envisaged
 * to save storage space on the covariance arrays.
 *
 * Callable from C or python (via cCovar.py):
 *
 * extern "C" int create_covar_matrix(int covarRad, int nx, int ny,
 *                                    int n_fibres, double* overlap_array,
 *                                    double* variances, double* output)
 *
 *     Parameters
 *     ----------
 *
 *     covarRad : int
 *         defines the area around each spaxel where the covariance is
 *         calculated
 *     nx, ny, n_fibers : int
 *         the (virtual) shape of the array `overlap_array`. Typically in SAMI we
 *         have `nx`, `ny` and `n_fibers` = 50, 50, n_rss x n_fibres_bundle,
 *          where n_rss is typically 7, and n_fibres_bundle is typically 61.
 *     overlap_array : double pointer
 *         1D array of the contribution of each of the `n_fibres` fibres to the
 *         each of the `nx` x `ny` spaxels in the SAMI grid.
 *     variances : double pointer
 *         1D array of the variance of each of the `n_fibres` fibres.
 *     output : double pointer
 *         1D array of the covariances, with shape:
 *             `nx` x `ny` x (2 * `covarRad` + 1) x (2 * `covarRad` + 1)
 *
 *   For any question, please NULL
 *   Francesco D'Eugenio <fdeugenio@gmail.com>
 */



/* This function calculates the covariance between the SAMI spaxels, given an
 * input weights array `overlap_array` and an array of variances `variances`.
 * For each point (x, y) in the SAMI grid, the covariance is  calculated only
 * within a square of side 2 * `covarRad` + 1, centred on the point (x, y). This
 * solution was envisaged to save storage space on the covariance arrays.
 *
 * Parameters
 * ----------
 *
 * covarRad : int
 *     defines the area around each spaxel where the covariance is
 *     calculated
 * nx, ny, n_fibers : int
 *     the (virtual) shape of the array `overlap_array`. Typically in SAMI we
 *     have `nx`, `ny` and `n_fibers` = 50, 50, n_rss x n_fibres_bundle,
 *      where n_rss is typically 7, and n_fibres_bundle is typically 61.
 * overlap_array : double pointer
 *     1D array of the contribution of each of the `n_fibres` fibres to the
 *     each of the `nx` x `ny` spaxels in the SAMI grid.
 * variances : double pointer
 *     1D array of the variance of each of the `n_fibres` fibres.
 *
 * Return
 * ------
 *
 * output : double pointer
 *     1D array of the covariances, with shape:
 *         `nx` x `ny` x (2 * `covarRad` + 1) x (2 * `covarRad` + 1)
 *
 */
extern "C" int create_covar_matrix(int covarRad, int nx, int ny, int n_fibres,
    double* overlap_array, double* variances, double* output) {


    //covarS = 2 # Radius of sub-region to record covariance information - probably
    //           # shouldn't be hard coded, but scaled to drop size in some way
    double *poutput = output;
    double *pvars = variances;

    // Set up coordinate arrays for the covariance sub-arrays
    int nc = 2 * covarRad + 1;
    int xB[nc*nc], yB[nc*nc];

    for (int i=0; i<nc; i++) {
        for (int j=0; j<nc; j++) {
            xB[j+i*nc] = i - covarRad;
            yB[j+i*nc] = j - covarRad;
        }
    }

    // Pre-compute all the `n_fibres` calls to `sqrt` (this saves time!).
    double sqrtvars[n_fibres];
    for (int k=0; k<n_fibres; k++) {*(sqrtvars + k) = sqrt(*(pvars + k)); }

    double a, b, norm, sqrtvar;
    int xC, yC;

    // This triple loop is optimised by the compiler and therefore does not
    // need switching to a single loop. The end of this module contains an
    // alternative implementation with a single loop, but that is not faster
    // and less clear, therefore we stick to the current implementation.
    for (int i=0; i<nx; i++) {
        for (int j=0; j<ny; j++) {

            for (int k=0; k<n_fibres; k++) {

                a = *(overlap_array + (ny * i + j) * n_fibres + k);

                if (a==a) // If a is finite...
                {
                     sqrtvar = *(sqrtvars+k);
                     // ...process the data.
                     a *= sqrtvar;
                     if (a!=a) {a = 1.0;}

                     for (int l=0; l<nc; l++) {
                         for (int m=0; m<nc; m++) {
                             xC = i + *(xB + m*nc + l);
                             yC = j + *(yB + m*nc + l);

                             if ((xC < 0) or (xC >= nx) or (yC < 0) or (yC >= ny)) {
                                 b = 0. * sqrtvar;}
                             else {
                                 b = *(overlap_array + (ny * xC + yC) * n_fibres + k) * sqrtvar;
                             }

                             if (b!=b) {b = 0.;}

                             *(poutput + ((ny*i + j) * nc + m)*nc + l) += (a * b);
                         }
                     }

                }

            }

            /*
            norm = *(poutput + ((ny*i + j) * nc + covarRad)*nc + covarRad);
            if (norm==0.) { norm = NAN; }

            for (int l=0; l<nc; l++) {
                for (int m=0; m<nc; m++) {
                    *(poutput + ((ny*i + j) * nc + m)*nc + l) /= norm;
                }
            }
            */
            norm = *(poutput + ((ny*i + j) * nc + covarRad)*nc + covarRad);
            if (norm==0.) {
                for (int l=0; l<nc; l++) {
                    for (int m=0; m<nc; m++) {
                        *(poutput + ((ny*i + j) * nc + m)*nc + l) = NAN;
                    }
                }
            } else {
                for (int l=0; l<nc; l++) {
                    for (int m=0; m<nc; m++) {
                        *(poutput + ((ny*i + j) * nc + m)*nc + l) /= norm;
                    }
                }
            }
        }
    }

    return 0;
}
    


// +----------------------------------------------------------------------------+
// | End of the implementation. What follows is a failed experiment.            |
// +----------------------------------------------------------------------------+

// As slow as the main function but a less clear implementation. Turned down.
/*
extern "C" int create_covar_matrix2(int covarRad, int nx, int ny, int n_fibres,
    double* overlap_array, double* variances, double* output) {


    //covarS = 2 # Radius of sub-region to record covariance information - probably
    //           # shouldn't be hard coded, but scaled to drop size in some way
    double *poutput = output;
    double *poverlap = overlap_array;
    double *pvars = variances;
    
    // Set up coordinate arrays for the covariance sub-arrays
    int nc = 2 * covarRad + 1;
    int xB[nc*nc], yB[nc*nc];

    for (int i=0; i<nc; i++) {
        for (int j=0; j<nc; j++) {
            xB[j+i*nc] = i - covarRad;
            yB[j+i*nc] = j - covarRad;
        }
    }

    double a, b, norm, var;
    int xC, yC;
    int i, j, k;

    // Main loop.
    for (int w=0; w<(nx*ny*n_fibres); w++) {

        // Unravel the indices.
        k = w % n_fibres;
        j = (w / n_fibres) % ny;
        i = (w / n_fibres) / ny;

        a = *(overlap_array + w);

        if (a==a) // If a is finite...
        {
             // ...process the data.
             var = sqrt(*(pvars + k));
             a *= var;
             if (a!=a) {a = 1.0;}

             for (int l=0; l<nc; l++) {
                 for (int m=0; m<nc; m++) {
                     // xC = i + xB[m*nc + l];
                     // yC = j + yB[m*nc + l];
                     xC = i + *(xB + m*nc + l);
                     yC = j + *(yB + m*nc + l);

                     if ((xC < 0) or (xC >= nx) or (yC < 0) or (yC >= ny)) {
                         b = 0. * var; }
                     else {
                         b = *(overlap_array + (ny * xC + yC) * n_fibres + k) * var;
                     }

                     if (b!=b) {b = 0.;}

                     *(poutput + ((ny*i + j) * nc + m)*nc + l) += (a * b);
                 }
             }

        }

    }



    for (int i=0; i<nx; i++) {
        for (int j=0; j<ny; j++) {
            norm = *(poutput + ((ny*i + j) * nc + covarRad)*nc + covarRad);

            if (norm==0.) { norm = NAN; }

            for (int l=0; l<nc; l++) {
                for (int m=0; m<nc; m++) {
                    *(poutput + ((ny*i + j) * nc + m)*nc + l) /= norm;
                 }
            }
        }
    }
    


    return 0;
}
*/
