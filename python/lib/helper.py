""" Module with helper class """

import numpy

import lib.special as special

class JobHelper(object):
    """ Class to handle multiprocess job """
    def __init__(self, total_jobs):
        """ Constructor """
        if total_jobs <= 0:
            raise ValueError('Total jobs must be at least 1.')
        self.total_jobs = total_jobs
        self.current_job = 0

    def increment(self):
        """ Increment current job index by 1 """
        if self.current_job != self.total_jobs-1:
            self.current_job += 1
        else:
            print('Already at last job index.')
        print("Job number: {}. Total jobs: {}.".format(self.current_job, self.total_jobs))

    def set_current_job(self, current_job):
        """ Set current job index to input """
        if current_job < 0 or current_job >= self.total_jobs:
            raise ValueError('Job must be at least 0 and less than total job.')
        self.current_job = current_job
        print("Job number: {}. Total jobs: {}.".format(self.current_job, self.total_jobs))

    def get_index_range(self, size):
        """ Calculate the start and end indices given job size
        Inputs:
        + size: int
            Size of job.
        Outputs:
        + job_range: tuple
            Return the start and end indices """
        job_index = numpy.floor(numpy.linspace(0, size, self.total_jobs+1))
        job_index = job_index.astype(int)
        job_range = (job_index[self.current_job], job_index[self.current_job+1])
        return job_range

class Bins(object):
    """ Class to handle uniform binnings """
    def __init__(self, limit, binw, cosmo):
        """ Constructor sets up number of bins and bins range
        Inputs:
        + limit: dict
            Dictionary with bin range
            Key: dec_min, dec_max, ra_min, ra_max, z_min, z_max
        + binw: dict
            Dictionary with binwidth
            Key: path, dec, ra, z
        + cosmo: cosmology.Cosmology
            Cosmology object to convert redshift to comoving distance"""

        # Set bin range
        limit_d = {}
        for key, val in limit.items():
            # convert angular limit radian and
            # redshift limit to comoving distance
            if key in ['dec_min', 'dec_max', 'ra_min', 'ra_max', 'theta_max']:
                limit_d[key] = numpy.deg2rad(float(val))
            elif key in ['z_min', 'z_max']:
                key = 'r' + key[1:]
                limit_d[key] = cosmo.z2r(float(val))
            else:
                limit_d[key] = float(val)
        self.limit = {}
        self.limit['s'] = [0., limit_d['s_max']]
        self.limit['theta'] = [0., limit_d['theta_max']]
        self.limit['dec'] = [limit_d['dec_min'], limit_d['dec_max']]
        self.limit['ra'] = [limit_d['ra_min'], limit_d['ra_max']]
        self.limit['r'] = [limit_d['r_min'], limit_d['r_max']]

        # Set number of bins
        self.num_bins = {}
        self.set_nbins(binw)

    def set_nbins(self, binw):
        """ Set up number of bins """

        # separation (Mpc/h)
        binw_s = float(binw['s'])
        nbins, binw_s = self._get_nbins(self.limit['s'][0], self.limit['s'][1], binw_s)
        self.num_bins['s'] = nbins

        # comoving distance (Mpc/h)
        binw_r = binw['r']
        if binw_r == 'auto':
            binw_r = self.default_binw('r', binw_s=binw_s)
        else:
            binw_r = float(binw_r)
        nbins, binw_r = self._get_nbins(self.limit['r'][0], self.limit['r'][1], binw_r)
        self.num_bins['r'] = nbins

        # angular variables (rad)
        for key, val in binw.items():
            if key in ['dec', 'ra', 'theta']:
                if val == 'auto':
                    binw_angle = self.default_binw(key, binw_r=binw_r)
                else:
                    binw_angle = numpy.deg2rad(float(val))
                nbins, _ = self._get_nbins(self.limit[key][0], self.limit[key][1], binw_angle)
                self.num_bins[key] = nbins

    def _get_nbins(self, x_min, x_max, binwidth):
        """ Return number of bins given min, max and binw.
        Round binw such that (max-min)/binw is an integer and return the new binw."""
        nbins = int(numpy.ceil((x_max-x_min)/binwidth))
        binw = (x_max-x_min)/nbins
        return nbins, binw

    def default_binw(self, key, binw_r=None, binw_s=None):
        """ Return the default binwidth """
        if key in ['dec', 'ra', 'theta']:
            binw = 1.*binw_r/self.limit['r'][1]
        elif key == 'r':
            binw = 0.25*binw_s
        else:
            raise ValueError('Valid key: "dec", "ra", "theta", "r"')
        return binw

    def min(self, key):
        """ Return binning lower bound """
        return self.limit[key][0]

    def max(self, key):
        """ Return binning upper bound """
        return self.limit[key][1]

    def nbins(self, key):
        """ Return number of bins """
        return self.num_bins[key]

    def bins(self, key):
        """ Return uniform binning """
        return numpy.linspace(self.limit[key][0], self.limit[key][1], self.num_bins[key]+1)


class CorrelationHelper(object):
    """ Class to handle multiprocess correlation function """
    def __init__(self, ntotal):
        """ Constructor set up attributes
        Inputs:
        + ntotal:
            Number of instances to be expected """
        self.n = 1
        self.ntotal = ntotal

        # Initialize histogram
        self.data_data = None
        self.theta_distr = None
        self.r_theta_distr = None
        self.r_distr = None

        # Initialize binnings and normalization
        self.bins = None
        self.norm = {"dd": None, "dr": None, "rr": None}

    def set_dd(self, data_data):
        """ Setting up data data """
        self.data_data = data_data

    def set_theta_distr(self, theta_distr):
        """ Setting up f(theta) """
        self.theta_distr = theta_distr

    def set_r_theta_distr(self, r_theta_distr):
        """ Setting up g(r, theta) """
        self.r_theta_distr = r_theta_distr

    def set_r_distr(self, r_distr):
        """ Setting up P(r) """
        self.r_distr = r_distr

    def set_bins(self, bins):
        """ Seeting up binnings """
        if not isinstance(bins, Bins):
            raise ValueError("Must be instance of helper.Bins")
        self.bins = bins

    def set_norm(self, norm_dd, norm_dr, norm_rr):
        """ Setting up normalization """
        self.norm["dd"] = norm_dd
        self.norm["dr"] = norm_dr
        self.norm["rr"] = norm_rr

    def add(self, other):
        """ Add another part """
        self.n += 1
        self.data_data = self.data_data+other.data_data
        self.theta_distr = self.theta_distr+other.theta_distr
        self.r_theta_distr = self.r_theta_distr+other.r_theta_distr

    def get_dd(self):
        """ Return DD(s) """
        print("Calculate DD(s)")
        return self.data_data

    def get_rr(self):
        """ Calculate and return RR(s) """
        print("Calculate RR(s)")

        # Initialize separation distribution and binning
        rand_rand = numpy.zeros((2, self.bins.nbins('s')))

        # Set up PDF maps and bins
        bins_theta = self.bins.bins('theta')
        bins_r = self.bins.bins('r')
        bins = [bins_theta, bins_r, bins_r]

        w_maps = [self.theta_distr, self.r_distr[0], self.r_distr[0]]
        uw_maps = [self.theta_distr, self.r_distr[1], self.r_distr[1]]

        # Calculate weighted distribution
        rand_rand[0] += special.prob_convolution(
            w_maps, bins, special.distance, self.bins.bins('s'))

        # Calculate unweighted istribution
        rand_rand[1] += special.prob_convolution(
            uw_maps, bins, special.distance, self.bins.bins('s'))

        return rand_rand

    def get_dr(self):
        """ Calculate and return DR(s) """
        print("Calculate DR(s)")

        # Initialize separation distribution and binning
        data_rand = numpy.zeros((2, self.bins.nbins('s')))

        # Set up PDF maps and bins
        bins_theta = self.bins.bins('theta')
        bins_r = self.bins.bins('r')
        bins = [bins_theta, bins_r, bins_r]

        # Calculate weighted distribution
        data_rand[0] += special.prob_convolution2d(
            self.r_theta_distr[0], self.r_distr[0], bins, special.distance, self.bins.bins('s'))

        # Calculate unweighted distribution
        data_rand[1] += special.prob_convolution2d(
            self.r_theta_distr[1], self.r_distr[1], bins, special.distance, self.bins.bins('s'))

        return data_rand
