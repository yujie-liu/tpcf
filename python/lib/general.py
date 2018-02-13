""" Module with general methods """

# Python modules
import numpy

def hist2point(hist, bins_x, bins_y, exclude_zeros=True):
    """ Convert 2D histogram into a set of weighted data points.
        Use bincenter as coordinate.
        Inputs:
        + hist: ndarray
            2D histogram
        + bins_x: ndarray
            Binedges along the  x-axis
        + bins_y: ndarray
            Binedges along the y-axis
        + exclude_zeros: bool (default=True)
            If True, return non-zero weight points only.
        Outputs:
        + catalog: ndarrays
            Catalog of weighted data points. Format: [X, Y, Weight]"""

    # Get bins center and create a grid
    center_x = 0.5*(bins_x[:-1]+bins_x[1:])
    center_y = 0.5*(bins_y[:-1]+bins_y[1:])
    grid_x, grid_y = numpy.meshgrid(center_x, center_y)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()

    # Convert the grid into a catalog
    hist = hist.T.flatten()
    catalog = numpy.array([grid_x, grid_y, hist]).T

    # Return catalog
    if exclude_zeros:
        return catalog[hist != 0]
    return catalog


def distance(theta, r1, r2):
    """ Calculate distance between two points at radius r1, r2 separated by angle theta """
    return numpy.sqrt(r1**2+r2**2-2*r1*r2*numpy.cos(theta))


def prob_convolution(maps, bins, dist, bins_s):
    """ Probability convolution of three PDFs and a given distance pairing function
    Given f(x), g(y), h(z) and the pairing function d(x,y,z), calculate:
        D(s) = Integrate[f(x)*g(y)*h(z)*dirac_delta[d(x,y,z)-s]] over the parameter space.
    Inputs:
    + maps: list, tuple, ndarray
        PDF along the three dimensions.
    + bins: list, tuple, ndarray
        Binedges along the three dimensions.
    + dist: function
        Pairing function that takes argument x, y, z and compute the separation.
    + bins_s: list, tuple, ndarray
        Binedges of resulting distribution.
    Outputs:
    + separation: ndarray
        Pairwise separation distribution """

    # Check if dimension is compatible
    for i in range(3):
        if len(bins[i]) != len(maps[i])+1:
            raise ValueError("Bins {0} must have len(Maps {0})+1".format(i))

    # Define data points on the x-y surface for integration
    map_xy = numpy.zeros((bins[0].size-1, bins[1].size-1))
    for i in range(bins[0].size-1):
        for j in range(bins[1].size-1):
            map_xy[i, j] = maps[0][i]*maps[1][j]
    return prob_convolution2d(map_xy, maps[2], bins, dist, bins_s)


def prob_convolution2d(map_xy, map_z, bins, dist, bins_s):
    """ Probability convolution given a 2d probability density and a 1d
    and a distance pairing distribution.
    Given f(x,y), g(z) and function d(x,y,z),  calculate:
        D[s] = Integrate[f(x,y)*g(z)*dirac_delta[d(x,y,z) - s]] over the parameter space.
    Inputs:
    + map_xy: ndarray
        Probability along the first and second dimensions.
    + map_z: list, tuple, ndarray
        Probability along the third dimension.
    + bins: list, tuple, ndarray
        Binedges along the three dimensions.
    + dist: function
        Pairing function that takes argument x, y, z and compute the separation.
    + bins_s: list, tuple, ndarray
        Binedges of the resulting distribution.
    Output:
    + separation: ndarray
        Pairwise separation distribution """

    # Check if dimension is compatible
    if len(bins[0]) != map_xy.shape[0]+1:
        raise ValueError("Bins X must have len(Map XY col)+1.")
    if len(bins[1]) != map_xy.shape[1]+1:
        raise ValueError("Bins Y must have len(Map XY row)+1.")
    if len(bins[2]) != len(map_z)+1:
        raise ValueError("Bins Z must have len(Map Z)+1.")

    # Define data points on the x-y surface for integration
    points_xy = hist2point(map_xy, bins[0], bins[1])

    # Define data points on z axis for integration
    # exclude zeros bins
    cut = map_z > 0
    center_z = 0.5*(bins[2][:-1] + bins[2][1:])
    center_z = center_z[cut]
    weight_z = map_z[cut]

    # Define separation histogram
    separation = numpy.zeros(bins_s.size-1)

    # Integration
    for i, point_z in enumerate(center_z):
        if i % 100 is 0:
            print(i)
        s = dist(points_xy[:, 0], points_xy[:, 1], point_z)

        # Fill separation distribution
        w = weight_z[i]*points_xy[:, 2]
        hist, _ = numpy.histogram(s, bins=bins_s, weights=w)
        separation += hist

    return separation
