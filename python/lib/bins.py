""" Module with class for binnings """

import numpy


class Bins(object):
    """ Class to handle uniform binnings """
    def __init__(self, limit, num_bins):
        """ Constructor sets up number of bins and bins range
        Inputs:
        + limit: dict
            Dictionary with bin range
            Key: dec_min, dec_max, ra_min, ra_max, z_min, z_max
        + binw: dict
            Dictionary with bin number
            Key: s, theta, dec, ra, z """

        # Set bin range
        limit_d = {}
        for key, val in limit.items():
            # convert angular limit radian and
            # redshift limit to comoving distance
            if key in ['dec_min', 'dec_max', 'ra_min', 'ra_max', 'theta_max']:
                limit_d[key] = numpy.deg2rad(float(val))
            else:
                limit_d[key] = float(val)
        self.limit = {}
        self.limit['s'] = [0., limit_d['s_max']]
        self.limit['theta'] = [0., limit_d['theta_max']]
        self.limit['dec'] = [limit_d['dec_min'], limit_d['dec_max']]
        self.limit['ra'] = [limit_d['ra_min'], limit_d['ra_max']]
        self.limit['z'] = [limit_d['z_min'], limit_d['z_max']]

        # Set number of bins
        self.num_bins = {}
        for key, val in num_bins.items():
            self.num_bins[key] = int(val)

    def __eq__(self, other):
        """ Comparing one bins with other """
        for key, val in self.limit.items():
            if val != other.limit[key]:
                return False
        for key, val in self.num_bins.items():
            if val != other.num_bins[key]:
                return False
        return True

    def min(self, key):
        """ Return binning lower bound """
        return self.limit[key][0]

    def max(self, key):
        """ Return binning upper bound """
        return self.limit[key][1]

    def nbins(self, key):
        """ Return number of bins """
        return self.num_bins[key]

    def binw(self, key):
        """ Return binwidth """
        return (self.max(key)-self.min(key))/self.nbins(key)

    def bins(self, key, cosmo=None):
        """ Return uniform binning """
        bins_min = self.min(key)
        bins_max = self.max(key)
        nbins = self.nbins(key)

        # Uniformly over 'r'
        if key == 'z' and cosmo is not None:
            bins_min = cosmo.z2r(bins_min)
            bins_max = cosmo.z2r(bins_max)

        return numpy.linspace(bins_min, bins_max, nbins+1)
