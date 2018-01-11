""" Module to handle correlation function """

import numpy

class Correlation(object):
    """ Class to handle correlation function """
    def __init__(self, rand_rand, data_data, data_rand, bins):
        """ Constructor """
        self.rand_rand = rand_rand
        self.data_data = data_data
        self.data_rand = data_rand
        self.bins = bins

    def error(self, w_dist, uw_dist):
        """ Get bin error of separation distribution
        Inputs:
        + w_dist: ndarray
            Weighted distribution
        + uw_dist: ndarray
            Unweighted distribution
        Outputs:
        + error: ndarray
            Bin errors """
        uw_error = numpy.sqrt(uw_dist)
        w_error = numpy.where(uw_error != 0, w_dist/uw_error, 0.)
        error = numpy.array([w_error, uw_error]).T
        return error

    def tpcf(self):
        """ Calculate two-point correlation
        Outputs:
        + correlation: ndarray
            Two-point correlation function
            Equation: (DD-2DR+RR)/RR """

        tpcf = self.data_data-2*self.data_rand+self.rand_rand
        tpcf = numpy.where(self.rand_rand != 0, tpcf/self.rand_rand, 0.)
        return tpcf

    def tpcfss(self):
        """ Calculate two-point correlation s^2
        Outputs:
        + correlation: ndarray
            Two-point correlation function s^2
            Equation: s^2*(DD-2DR+RR)/RR """
        s = (self.bins[1:]+self.bins[:-1])/2.
        return self.tpcf()*s**2
