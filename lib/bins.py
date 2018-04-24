""" Module with class for binnings """

import numpy
from lib.cosmology import Cosmology


class Bins(object):
    """ Class to handle uniform binnings """
    def __init__(self, limit, num_bins=None, islice=0, nslice=1, cosmo=None, auto=None,
                 binw_s=4.00):
        """ Constructor sets up number of bins and bins range """

        # Convert single cosmology input to list of length 1
        if not isinstance(cosmo, (list, tuple, numpy.ndarray)):
            cosmo = [cosmo]

        # Set bin range
        limit_d = {}
        for key, val in limit.items():
            # convert angular limit to radian
            if key in ['dec_min', 'dec_max', 'ra_min', 'ra_max']:
                limit_d[key] = numpy.deg2rad(float(val))
            elif key in ['z_min', 'z_max', 's_max']:
                limit_d[key] = float(val)
        self.limit = {}
        self.limit['s'] = [0., limit_d['s_max']]
        self.limit['dec'] = [limit_d['dec_min'], limit_d['dec_max']]
        self.limit['ra'] = [limit_d['ra_min'], limit_d['ra_max']]

        # Calculate maximum angular separation
        r_min = Cosmology.max_cosmo(cosmo).z2r(limit_d['z_min'])
        theta_max = numpy.arccos(1.-limit_d['s_max']**2/(2*r_min**2))
        self.limit['theta'] = [0., theta_max]

        # Apply z-slicing
        diff = (limit_d['z_max']-limit_d['z_min'])/nslice
        z_min = limit_d['z_min'] + diff*islice
        z_max = z_min + diff + Cosmology.max_cosmo(cosmo).dels_to_delz(limit_d['s_max'], z_min)
        z_max = min(z_max, limit_d['z_max'])
        self.limit['z'] = [z_min, z_max]

        # Set number of bins
        self.num_bins = {}
        if not auto:
            # Manual binnings
            print('- Setting manual binnings')
            for key, val in num_bins.items():
                if key in ['ra', 'dec', 'z', 'theta', 's']:
                    self.num_bins[key] = int(val)
        elif auto:
            # Auto binnings
            print('- Setting auto binnings')
            self._set_auto_nbins(Cosmology.min_cosmo(cosmo), binw_s)

        # Print out number of bins
        print('- Bin range: ')
        for key in sorted(self.limit.keys()):
            print('    + {0:6s}: [{1:.5f}, {2:.5f}]'.format(key, self.min(key), self.max(key)))

        print('- Number of bins:')
        for key in sorted(self.num_bins.keys()):
            print('    + {0:6s}: {1:4d}'.format(key, self.nbins(key)))

    def __eq__(self, other):
        """ Comparing one bins with other """
        for key, val in self.limit.items():
            if val != other.limit[key]:
                return False
        for key, val in self.num_bins.items():
            if val != other.num_bins[key]:
                return False
        return True

    def _set_auto_nbins(self, cosmo, binw_s):
        """ Set number of bins based on binwidths """

        # Separation
        self.num_bins['s'], binw_s = self._find_nbins('s', binw_s)

        # Redshift/Comoving distance distribution
        binw_r = binw_s/2.
        self.num_bins['z'], binw_r = self._find_nbins('r', binw_r, cosmo)

        # Angular variable
        binw_angl = binw_r/cosmo.z2r(self.max('z'))
        for key in ['dec', 'ra', 'theta']:
            self.num_bins[key], _ = self._find_nbins(key, binw_angl)

    def _find_nbins(self, key, binw, cosmo=None):
        """ Return the number of bins given key and binwidth """
        if key == 'r' and cosmo is not None:
            low, high = cosmo.z2r([self.min('z'), self.max('z')])
        else:
            low = self.min(key)
            high = self.max(key)

        nbins = int(numpy.ceil((high-low)/binw))
        binw = (high-low)/nbins

        return nbins, binw

    def find_zslice(self, index, total, cosmo):
        """ Find zslice """
        diff = (self.max('z')-self.min('z'))/total
        z_low = index*diff+self.min('z')
        z_high = z_low+diff+cosmo.dels_to_delz(self.max('s'))
        return z_low, z_high

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
