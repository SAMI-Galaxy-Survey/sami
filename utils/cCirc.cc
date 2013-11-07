/*
 *
 * A set of routines to generate a weight map for the intersection of a circle
 * with a square grid.  Squares on the grid that are completely within the circle
 * receive a weight of 1.  Square that intersect the circle are given a weight
 * that is proportional to the area of the square inside the circle.
 *
 * Callable from C or python (via cm.py):
 *   extern "C" int weight_map(int nx, int ny, double xc, double yc, double r, double* output)
 *     nx,ny defines the size of the square grid
 *     xc,yc is the centre of the circl
 *     r is the radius of the circle
 *     output is the weight map, as a 1D array (which cm.py reshapes to 2D)
 *
 * Jon Nielsen <jon.nielsen@anu.edu.au>
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <limits>

static const double eps = std::numeric_limits<double>::epsilon();

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
