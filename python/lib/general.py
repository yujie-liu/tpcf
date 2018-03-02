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
