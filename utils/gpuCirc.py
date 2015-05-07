#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pyopencl as cl

#GPU_PROGRAM = None

#def get_gpu_program():


def resample_circle(xpix, ypix, xc, yc, r):
    """Resample a circle/drop onto an output grid.
    
    Parameters
    ----------
    xpix: (int) Number of pixels in the x-dimension of the output grid
    ypix: (int) Number of pixels in the y-dimension of the output grid
    xc: (float) x-position of the centre of the circle.
    yc: (float) y-position of the centre of the circle.
    r: (float) radius of the circle
    
    Output
    ------
    2D array of floats. Note that the zeroth axis is for the y-dimension and the
    first axis is for the x-dimension. i.e., out.shape -> (ypix, xpix)
    This can be VERY CONFUSING, particularly when one remembers that imshow's
    behaviour is to plot the zeroth axis as the vertical coordinate and the first
    axis as the horizontal coordinate.
    
    """

    (x_pixel_grid, y_pixel_grid) = np.meshgrid(
        np.arange(xpix, dtype=np.float32), 
        np.arange(ypix, dtype=np.float32),
        sparse=False)


    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    xgrid_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x_pixel_grid)
    ygrid_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=y_pixel_grid)

    prg = cl.Program(ctx, """
        __kernel void pixel(__global const float *xgrid_g, __global const float *ygrid_g, __global float *res_g) {
          int gid = get_global_id(0);

          //__local int n_tests;
          //__local int n_included;

          int n_tests = 0;
          int n_included = 0;

          for (float i = 0; i < 1; i = i + 0.001) {
            for (float j = 0; j < 1; j = j + 0.001) {
              n_tests++;
              if (
                  ((xgrid_g[gid] + i) * (xgrid_g[gid] + i) + 
                  (ygrid_g[gid] + i) * (ygrid_g[gid] + i)) < 3.0
                ) {
                n_included++;
              }
            }
          }

          res_g[gid] = float(n_included)/float(n_tests);
        }
        """).build()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, x_pixel_grid.nbytes)

    pixel = prg.pixel
    pixel.set_scalar_arg_dtypes([None, None, np.float32, np.float32,  np.float32, None])

    event = pixel(queue, (40*60,), None, xgrid_g, ygrid_g, 5, 10, 3, res_g)

    res_np = np.empty_like(x_pixel_grid)

    cl.enqueue_copy(queue, res_np, res_g, wait_for=[event])

    return res_np
