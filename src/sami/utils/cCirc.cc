/*
 *
 * Functions to calculate the:
 * 1) drizzle overlap between a square and a circle (function `resample_circle`
 *   by Jon Nielsen 2012)
 * 2) Gaussian overlap between a circular Gaussian and a square grid (function
 *   `inteGrauss2d` by Francesco D'Eugenio 2017)
 *
 *  These calculations are the crucial step in drizzling, to know how much each
 * input fibre (circle) contributes to each output spaxel (square).
 *
 * Callable from C or python (via cCirc.py):
 *   extern "C" int weight_map(int nx, int ny, double xc, double yc, double r,
 *                             double* output)
 *     nx,ny defines the size of the square grid
 *     xc,yc is the centre of the circl
 *     r is the radius of the circle
 *     output is the weight map, as a 1D array (which cm.py reshapes to 2D)
 *
 *   Jon Nielsen <jon.nielsen@anu.edu.au>
 *
 * extern "C" int weight_map_Gaussian(int nx, int ny, double xc, double yc,
 *                                    double sigma, double n_sigma, 
 *                                    double* output)
 *     nx,ny defines the size of the square grid
 *     xc,yc is the centre of the circl
 *     sigma is the standard deviation of the Gaussian, in units of pixels
 *     support is the number of standard deviations beyond which the Gaussian
 *         is truncated.
 *     output is the weight map, as a 1D array (which cm.py reshapes to 2D)
 *
 *   Francesco D'Eugenio <fdeugenio@gmail.com>
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <limits>

static const double eps = std::numeric_limits<double>::epsilon();
static const double SQRT2 = sqrt(2.);

// +---+------------------------------------------------------------------------+
// | 1.| Drizziling functions by Jon Nielsen.                                   |
// +---+------------------------------------------------------------------------+

/* A structure that contains relevant information about the intersection points
 * between the circle and the grid */
struct Point {
  double x;
  double y;
  double theta;
  bool on_x;
  bool on_y;
  Point() : x(0.0), y(0.0), theta(0.0), on_x(false), on_y(false) {};
  Point(double x_, double y_, double theta_, bool on_x_, bool on_y_)
    : x(x_), y(y_), theta(theta_), on_x(on_x_), on_y(on_y_) {};
  bool operator<(const Point& rhs) const { return theta < rhs.theta; };
};

/* Find all the intersections between the circle and the grid */
std::vector<Point> find_intersections(double xc, double yc, double r)
{
  std::vector<Point> points;
  int xmin, xmax, ymin, ymax, i, j;
  double x, x2, y, y2, theta;
  bool on_x, on_y;

  xmin = (int)ceil(xc-r);
  xmax = (int)floor(xc+r);
  ymin = (int)ceil(yc-r);
  ymax = (int)floor(yc+r);

  for (i=xmin; i<=xmax; i++) {
    x = i - xc;
    y2 = r*r - x*x;
    if (fabs(y2) < eps) {
      y = 0.0;
    } else {
      y = sqrt(y2);
    }
    if (fabs(y+yc-round(y+yc)) < eps) {
      on_y = true;
    } else {
      on_y = false;
    }
    // If this is a tangent, then ignore it
    if (y > eps) {
      theta = atan2(y,x);
      if (theta < 0) {
	theta += 2*M_PI;
      }
      points.push_back(Point(i,y+yc,theta,true,on_y));
      theta = atan2(-y,x);
      if (theta < 0) {
	theta += 2*M_PI;
      }
      points.push_back(Point(i,-y+yc,theta,true,on_y));
    }
  }

  for (j=ymin; j<=ymax; j++) {
    y = j - yc;
    x2 = r*r - y*y;
    if (fabs(x2) < eps) {
      x = 0.0;
    } else {
      x = sqrt(x2);
    }
    if (fabs(x+xc-round(x+xc)) < eps) {
      on_x = true;
    } else {
      on_x = false;
    }
    // If this is a tangent, then ignore it
    if (x > eps) {
      theta = atan2(y,x);
      if (theta < 0) {
	theta += 2*M_PI;
      }
      points.push_back(Point(x+xc,j,theta,on_x,true));
      theta = atan2(y,-x);
      if (theta < 0) {
	theta += 2*M_PI;
      }
      points.push_back(Point(-x+xc,j,theta,on_x,true));
    }
  }

  return points;
}

void print_point(const Point& point)
{
  std::cout << point.x << "\t" << point.y << "\t" << point.theta << "\t" <<
    point.on_x << "\t" << point.on_y << std::endl;
}

void print_points(const std::vector<Point>& points)
{
  std::vector<Point>::const_iterator i;
  for (i=points.begin(); i!=points.end(); i++) {
    print_point(*i);
  }
}

bool equiv(const Point& a, const Point& b)
{
  if (sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y)) < eps) {
    return true;
  }
  return false;
}

void area_contribution(Point a, Point b, double xc, double yc, double r, int& i, int& j, double& area)
{
  double delta_x, delta_y, delta_theta, mid_theta;
  double tmpx, tmpy;
  Point c, d, xp, yp;
  double x, y;

  area = 0.0;

  // We assume that a.theta < b.theta
  delta_theta = b.theta - a.theta;

  // Work out which square we are dealing with here
  mid_theta = (a.theta+b.theta)/2.0;
  x = r * cos(mid_theta) + xc;
  i = int(floor(x));
  y = r * sin(mid_theta) + yc;
  j = int(floor(y));

  // First get the circle segment area
  area = 0.5*r*r*(delta_theta - sin(delta_theta));
  if (area > 1) {
    std::cout << "area > 1" << std::endl;
  }

  // Next get the polygonal area
  
  if (a.on_x && b.on_x) {
    // Both points are on an x gridline
    delta_x = fabs(a.x-b.x);
    if (delta_x < eps) {
      // Both points are on the same x gridline
      // No polygonal contribution at all.
    } else {
      // Points are on different x gridlines.  Note that they must both
      // have the same upper and lower y grid bounds, or else we would have
      // a point on a y gridline between them.
      delta_y = fabs(a.y-b.y);
      if (y < yc) {
	tmpy = std::max(a.y,b.y);
	// rectangular area
	area += ceil(tmpy) - tmpy;
      } else {
	tmpy = std::min(a.y,b.y);
	// rectangular area
	area += tmpy - floor(tmpy);
      }
      // triangular area
      area += 0.5*delta_y;
    }
  } else if (a.on_y && b.on_y) {
    // Both points are on a y gridline
    delta_y = fabs(a.y-b.y);
    if (delta_y < eps) {
      // Both points are on the same y gridline
      // No polygonal contribution at all.
    } else {
      // Points are on different y gridlines.  Note that they must both
      // have the same upper and lower x grid bounds, or else we would have
      // a point on an x gridline between them.
      delta_x = fabs(a.x-b.x);
      if (x < xc) {
	tmpx = std::max(a.x,b.x);
	// rectangular area
	area += ceil(tmpx) - tmpx;
      } else {
	tmpx = std::min(a.x,b.x);
	// rectangular area
	area += tmpx - floor(tmpx);
      }
      // triangular area
      area += 0.5*delta_x;
    }
  } else {
    // One is on x, the other on y
    // Call the point on x xp, and the point on y yp
    if (a.on_x && b.on_y) {
      xp = a;
      yp = b;
    } else {
      xp = b;
      yp = a;
    }
    // Now we know which is which, construct point c, which is the
    // point on the same x gridline as xp, but also on the next y gridline
    // closer to the centre of the circle
    if (xp.y < yc) {
      c.y = ceil(xp.y);
    } else {
      c.y = floor(xp.y);
    }
    c.x = xp.x;
    // Now also point d, which is on the same y gridline as yp,
    // but also on the next x gridline closer to the centre of the circle
    if (yp.x < xc) {
      d.x = ceil(yp.x);
    } else {
      d.x = floor(yp.x);
    }
    d.y = yp.y;
    if (equiv(c,d)) {
      // The same point, so it's a triangle
      area += fabs(0.5*(xp.y-c.y)*(yp.x-c.x));
    } else {
      // Not the same point - it's a pentagon
      // Note that we ignore any intersections of the circle with other edges
      // of this square.  This is handled by the calling function, which uses
      // 1-area as a subtractive term.
      area += fabs(xp.y-c.y) + fabs((xp.y-d.y)*(yp.x-d.x)) + fabs(0.5*(xp.x-yp.x)*(xp.y-yp.y));
    }
  }
}

extern "C" int weight_map(int nx, int ny, double xc, double yc, double r, double* output)
{
  int xmin, xmax, ymin, ymax;
  double x, x1, x2, y;
  std::vector<Point> points;
  int i, j, ii, jj;
  Point start;
  double area;
  double offset;
  double total_area = 0.0;

  // Work out which grid elements are entirely within the circle.
  ymin=int(ceil(yc-r));
  ymax=int(floor(yc+r));
  for (j=ymin; j<ymax; j++) {
    y = j - yc;
    x1 = sqrt(r*r - y*y);
    x2 = sqrt(r*r - (y+1)*(y+1));
    x = std::min(x1,x2);
    xmin=int(ceil(-x+xc));
    xmax=int(floor(x+xc));
    for (i=xmin; i<xmax; i++) {
      if ((i < 0) || (i >= nx) || (j < 0) || (j >= ny)) {
	std::cout << "point out of range!" << std::endl;
      } else {
	output[j*nx+i] = 1;
      }
    }
  }

  points = find_intersections(xc, yc, r);
  if (points.size() == 0) {
    i = int(floor(xc));
    j = int(floor(yc));
    if ((i < 0) || (i >= nx) || (j < 0) || (j >= ny)) {
      std::cout << "point out of range!" << std::endl;
    } else {
      output[j*nx+i] = M_PI*r*r;
    }
  } else {
    std::sort(points.begin(), points.end());
    start = *points.begin();
    start.theta += 2*M_PI;
    points.push_back(start);

    i = 0;
    while (i < points.size()-1) {
      j = i+1;
      while (equiv(points[i],points[j])) {
	i++;
	j++;
      }
      area_contribution(points[i], points[j], xc, yc, r, ii, jj, area);
      if ((ii < 0) || (ii >= nx) || (jj < 0) || (jj >= ny)) {
	std::cout << "point out of range!" << std::endl;
      } else if (output[jj*nx+ii] != 0.0) {
	area = 1-area;
	output[jj*nx+ii] -= area;
      } else {
	output[jj*nx+ii] = area;
      }
      i = j;
    }
  }
  /*
  for (j=0; j<ny; j++) {
    for (i=0; i<nx; i++) {
      printf("%d %d %.16f\n", i, j, out[j*nx+i]);
      total_area += out[j*nx+i];
    }
  }

  printf("%.8f %.8f\n", total_area, M_PI*r*r);
  */
  return 0;
}



// +---+-----------------------------------------------------------------------+
// | 2.| Gaussian integration methods by Francesco D'Eugenio.                  |
// +---+-----------------------------------------------------------------------+

// +---------------------------------------------------------------------------+
// | Written by Francesco D'Eugenio on the Couch to Canberra. A slow but       |
// | comfortable ride.                                                         |
// | 15/02/2017 - but would you trust the date from Windows OS?                |
// |                                                                           |
// +---------------------------------------------------------------------------+



double _inteGRaussian(double a, double b) {
    /*Computes the integral of the univariate normal distribution between the
    extremes `a` and `b`. In terms of the Error Function `erf` the cumulative
    distribution of the univariate Normal Distribution is:
        1/2 (1 + erf( z/sqrt(2)) )
    */
    return .5 * (erf(b/SQRT2) - erf(a/SQRT2));
}



/* This function calculates the integral of the 2D Gaussian of mean
 *  (`xc`, `yc`) and standard deviation (`sigma`, `sigma`) on a `xpix` by `ypix`
 *  grid. The Gaussian is truncated to zero beyond the box of half side
 *  `support` times `sigma`. We call the truncated function `Gc`, because it
 *  has compact support.
 *
 * Parameters
 * ----------
 *
 * xpix: int
 *     Number of pixels in the x-dimension of the output grid
 * ypix: int
 *     Number of pixels in the y-dimension of the output grid
 * xc: float
 *     x-position of the centre of the circle.
 * yc: float
 *     y-position of the centre of the circle.
 * r: float
 *     standard deviation of the Gaussian, in units of pixels.
 * support : float
 *     number of standard deviations beyond which the Gaussian is truncated.
 *
 * Return
 * ------
 *
 * 2D array of floats, containing the integral of the function Gc over each
 * pixel.
 *
 * Notes
 * -----
 *
 * The zeroth axis of the output array is for the y-dimension and the
 * first axis is for the x-dimension. i.e., out.shape -> (ypix, xpix)
 * This can be VERY CONFUSING, particularly when one remembers that imshow's
 * behaviour is to plot the zeroth axis as the vertical coordinate and the first
 * axis as the horizontal coordinate.
 *
 * Here we treat function Gc as a Gaussian within the set `U` and zero outside.
 * Therefore the set `U` is effectively the support of the function `f`. `U` is
 * defined as the set of points (x,y) in the grid such that:
 * U := {(x,y) in Grid: |x-xc| <= support * sigma; |y-yc| <= support * sigma}
 * The reason why the support `U` has a square shape is that this is a
 * convenient grid where to compute the factorizable integral of f(x,y).
 * In the future the algorithm might use the circle Uc of centre (xc, yc) and
 * radius
 *  `support * sigma` as support. This does not make a difference in most cases,
 *  provided that the light profile under study does not fall off as fast as a
 *  Gaussian and `support` is a reasonable number (here ``reasonable`` is short
 *  for: five is safe, three is risky, one is crazy and seven is too cautios).
 */
extern "C" int weight_map_Gaussian(int nx, int ny, double xc, double yc, double sigma, double n_sigma, double* output)
{
    double supp_radius = n_sigma * sigma;
    int x_supp_min = floor(xc - supp_radius) > 0 ? int(floor(xc - supp_radius)) : 0;
    int x_supp_max = ceil(xc + supp_radius) < nx ? int(ceil(xc + supp_radius)) : nx;
    int y_supp_min = floor(yc - supp_radius) > 0 ? int(floor(yc - supp_radius)) : 0;
    int y_supp_max = ceil(yc + supp_radius) < ny ? int(ceil(yc + supp_radius)) : ny;

    // Before performing the integrals, we operate a change of variables. This is
    // elegant, efficient and exquisitely stable.
    double xNorm_supp_min = (double(x_supp_min) - xc)/sigma;
    double yNorm_supp_min = (double(y_supp_min) - yc)/sigma;
    double xNorm_supp_max = (double(x_supp_max) - xc)/sigma;
    double yNorm_supp_max = (double(y_supp_max) - yc)/sigma;

    int x_supp_size = x_supp_max - x_supp_min;
    int y_supp_size = y_supp_max - y_supp_min;

    // Arrays to store the integrals of the Gaussian.
    double integrx[x_supp_size], integry[y_supp_size];
    double *pintegrx = integrx, *pintegry = integry;
    //integrx = np.array([_inteGRaussian(xk, xk + 1.) for xk in supp_x])
    //integry = np.array([_inteGRaussian(yk, yk + 1.) for yk in supp_y])
    double delta = 1./sigma;

    //std::cout << "Size of supp is (" << x_supp_size << ", " << y_supp_size << ")" << std::endl;
    //std::cout << "The support is given by " << x_supp_min << ", " << x_supp_max << ";  " << y_supp_min << ", " << y_supp_max << std::endl;
    //std::cout << "Centre is (" << xc << ", " << yc <<  ")" << std::endl;

    for (int i=0; i<x_supp_size; i++) {
        *(pintegrx+i) = _inteGRaussian(
             xNorm_supp_min+double(i)*delta,
             xNorm_supp_min+double(i+1)*delta);
    }
    for (int i=0; i<y_supp_size; i++) {
        *(pintegry+i) = _inteGRaussian(
             yNorm_supp_min+double(i)*delta,
             yNorm_supp_min+double(i+1)*delta);
    }

    // Assign zero to all the elements of the array outside the circle...
    // TODO

    // Now assign the values of the 2D-integral.
    double *poutput = output;
    //for (int j=y_supp_min; j<=y_supp_max; j++) {
    for (int j=0; j<y_supp_size; j++) {
        //for (int i=x_supp_min; i<=x_supp_max; i++) {
        for (int i=0; i<x_supp_size; i++) {
            //std::cout << "Assigning 2D Integral at step (" << i << ", " << j << ")" << std::endl;
            //*(poutput + j*nx+i) = *(pintegrx + i) * *(pintegry + j);
            *(poutput + (j+y_supp_min)*nx+i+x_supp_min) = *(pintegrx + i) * *(pintegry + j);
        }
    }

  return 0;
}
