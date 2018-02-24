""" Module with helper class """

import numpy

import lib.general as general

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
        print("Job number: {}. Total jobs: {}".format(self.current_job, self.total_jobs))

    def set_current_job(self, current_job):
        """ Set current job index to input """
        if current_job < 0 or current_job >= self.total_jobs:
            raise ValueError('Job must be at least 0 and less than total job.')
        self.current_job = current_job
        print("Job number: {}. Total jobs: {}".format(self.current_job, self.total_jobs))

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
    def __init__(self, limit, binw):
        """ Constructor sets up number of bins and bins range
        Inputs:
        + limit: dict
            Dictionary with bin range
            Key: dec_min, dec_max, ra_min, ra_max, z_min, z_max
        + binw: dict
            Dictionary with binwidth
            Key: path, dec, ra, z """

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
        self.set_nbins(binw)

    def __eq__(self, other):
        """ Comparing one bins with other """
        for key, val in self.limit.items():
            if val != other.limit[key]:
                return False
        for key, val in self.num_bins.items():
            if val != other.num_bins[key]:
                return False
        return True

    def set_nbins(self, binw):
        """ Set up number of bins """
        for key, val in binw.items():
            binw = float(val)
            if key in ['dec', 'ra', 'theta']:
                binw = numpy.deg2rad(binw)
            nbins, _ = self._get_nbins(self.limit[key][0], self.limit[key][1], binw)
            self.num_bins[key] = nbins

    def _get_nbins(self, x_min, x_max, binwidth):
        """ Return number of bins given min, max and binw.
        Round binw such that (max-min)/binw is an integer and return the new binw."""
        nbins = int(numpy.ceil((x_max-x_min)/binwidth))
        binw = (x_max-x_min)/nbins
        return nbins, binw

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

    def bins(self, key):
        """ Return uniform binning """
        return numpy.linspace(self.limit[key][0], self.limit[key][1], self.num_bins[key]+1)


class CorrelationHelper(object):
    """ Class to handle multiprocess correlation function calculation """
    def __init__(self, bins):
        """ Constructor set up attributes """

        # Initialize binnings
        self.bins = bins

        # Initalize norm factor
        self.norm = {"dd": None, "dr": None, "rr": None}

        # Initialize histogram
        self.data_data = numpy.zeros((2, self.bins.nbins('s')))
        self.theta_distr = numpy.zeros(self.bins.nbins('theta'))
        self.z_theta_distr = numpy.zeros((2, self.bins.nbins('theta'), self.bins.nbins('z')))
        self.z_distr = None

    def set_z_distr(self, z_distr):
        """ Setting up P(r) """
        self.z_distr = z_distr

    def set_norm(self, norm_dd, norm_dr, norm_rr):
        """ Setting up normalization """
        self.norm["dd"] = norm_dd
        self.norm["dr"] = norm_dr
        self.norm["rr"] = norm_rr

    def set_data_data(self, tree, catalog, job_helper=None):
        """ Calculate DD(s) using a modified nearest-neighbors kd-tree search.
        Metric: Euclidean
        Inputs:
        + tree: sklearn.neighbors.BallTree or sklearn.neighbors.KDTree
            KD-tree built from data catalog
        + catalog: numpy.ndarray
            Data catalog in KD-tree (with weights)
        + job_helper: helper.JobHelper (default=None)
            Job manager to handle multiprocess indices. If None, assume one job. """

        # If job_helper is None, assume one job
        if job_helper is None:
            job_helper = JobHelper(1)
            job_helper.set_current_job(0)

        # Get start and end indices
        start, end = job_helper.get_index_range(catalog.shape[0])

        s_max = self.bins.max('s')
        nbins_s = self.bins.nbins('s')
        print("Calculate DD from index {} to {}".format(start, end-1))
        for i, point in enumerate(catalog[start:end]):
            if i % 10000 is 0:
                print(i)
            index, s = tree.query_radius(point[: 3].reshape(1, -1), r=s_max,
                                         return_distance=True)

            # Fill weighted distribution
            # weight is the product of the weights of two points
            weights = catalog[:, 3][index[0]]*point[3]
            hist, _ = numpy.histogram(s[0], bins=nbins_s, range=(0., s_max), weights=weights)
            self.data_data[0] += hist

            # Fill unweighted distribution
            hist, _ = numpy.histogram(s[0], bins=nbins_s, range=(0., s_max))
            self.data_data[1] += hist

        # Correction for double counting in the first bin from pairing a galaxy
        # with itself
        self.data_data[0][0] -= numpy.sum(catalog[start:end, 3]**2)
        self.data_data[1][0] -= end-start

        # Correction for double counting
        self.data_data = self.data_data/2.

    def set_theta_distr(self, tree, catalog, job_helper=None):
        """ Calculate f(theta) using a modified nearest-neighbors search BallTree algorithm.
        Metric = 'haversine'.
        Inputs:
        + tree: sklearn.neighbors.BallTree
            Balltree built from data catalog
        + catalog: numpy.ndarray
            Angular catalog in Balltree (with weights)
        + job_helper: helper.JobHelper (default=None)
            Job manager to handle multiprocess indices. If None, assume one job."""

        # If job_helper is None, assume one job
        if job_helper is None:
            job_helper = JobHelper(1)
            job_helper.set_current_job(0)

        # Get start and end indices
        start, end = job_helper.get_index_range(catalog.shape[0])

        theta_max = self.bins.max('theta')
        nbins_theta = self.bins.nbins('theta')
        print("Construct f(theta) from index {} to {}".format(start, end-1))
        for i, point in enumerate(catalog[start:end]):
            if i % 10000 is 0:
                print(i)
            index, theta = tree.query_radius(point[:2].reshape(1, -1),
                                             r=theta_max,
                                             return_distance=True)
            # weight is the product of the weights of each point
            weights = point[2]*catalog[:, 2][index[0]]
            hist, _ = numpy.histogram(theta[0], bins=nbins_theta,
                                      range=(0., theta_max), weights=weights)
            self.theta_distr += hist

        # Correction for double counting
        self.theta_distr = self.theta_distr/2.

    def set_z_theta_distr(self, tree, data_catalog, angular_catalog, mode, job_helper=None):
        """ Calculate g(theta, z) using a modified nearest-neighbors BallTree search
        Metric = 'haversine'.
        NOTE: assume uniformed comoving bins
        Inputs:
        + tree: sklearn.neighbors.BallTree
            Balltree built from DEC and RA coordinates
        + data_catalog: numpy.ndarray
            Catalog from galaxy data (with weights)
        + angular_catalog: numpy.ndarray
            Angular catalog from random data (with weights)
        + job_helper: helper.JobHelper (default=None)
            Job manager to handle multiprocess indices. If None, assume one job. """

        # Get start and end indices
        if mode == "angular_tree":
            start, end = job_helper.get_index_range(data_catalog.ntotal)
        elif mode == "data_tree":
            start, end = job_helper.get_index_range(angular_catalog.shape[0])

        # Initialize some binning variables
        nbins_z = self.bins.nbins('z')
        nbins_theta = self.bins.nbins('theta')
        theta_max = self.bins.max('theta')
        limit = ((0., theta_max), (self.bins.min('z'), self.bins.max('z')))

        print("Construct angular-comoving from index {} to {}".format(start, end-1))
        if mode == "angular_tree":
            for i, point in enumerate(data_catalog[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = tree.query_radius(
                    point[:2].reshape(1, -1), r=theta_max, return_distance=True)
                z = numpy.repeat(point[2], index[0].size)

                # Fill unweighted histogram
                weights = angular_catalog[:, 2][index[0]]
                hist, _, _ = numpy.histogram2d(
                    theta[0], z, bins=(nbins_theta, nbins_z), range=limit, weights=weights)
                self.z_theta_distr[1] += hist

                # Fill weighted histogram
                # weight is the product of the weight of the data point and the weight
                # of the angular point.
                weights = weights*point[3]
                hist, _, _ = numpy.histogram2d(
                    theta[0], z, bins=(nbins_theta, nbins_z), range=limit, weights=weights)
                self.z_theta_distr[0] += hist

        elif mode == "data_tree":
            for i, point in enumerate(angular_catalog[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = tree.query_radius(
                    point[:2].reshape(1, -1), r=theta_max, return_distance=True)
                z = data_catalog[:, 2][index[0]]

                # Fill weighted histogram
                # weight is the product of the weight of the data point and the weight of
                # the angular point
                weights = point[2]*data_catalog[:, 3][index[0]]
                hist, _, _ = numpy.histogram2d(
                    theta[0], z, bins=(nbins_theta, nbins_z), range=limit, weights=weights)
                self.z_theta_distr[0] += hist

                # Fill unweighted histogram
                # weight is the weight of the angular point
                hist, _, _ = numpy.histogram2d(
                    theta[0], z, bins=(nbins_theta, nbins_z), range=limit)
                self.z_theta_distr[1] += hist*point[2]

    def add(self, other):
        """ Add another part """
        self.data_data = self.data_data+other.data_data
        self.theta_distr = self.theta_distr+other.theta_distr
        self.z_theta_distr = self.z_theta_distr+other.z_theta_distr

    def get_data_data(self):
        """ Return DD(s) """
        print("Calculate DD(s)")
        return self.data_data

    def get_rand_rand(self, cosmo):
        """ Calculate and return RR(s) """
        if self.z_distr is None:
            return RuntimeError("Comoving distribution is None.")

        print("Calculate RR(s)")

        # Initialize separation distribution and binning
        rand_rand = numpy.zeros((2, self.bins.nbins('s')))

        # Set up PDF maps and bins
        bins_theta = self.bins.bins('theta')
        bins_z = self.bins.bins('z')
        bins_r = cosmo.z2r(bins_z)
        bins = [bins_theta, bins_r, bins_r]

        w_maps = [self.theta_distr, self.z_distr[0], self.z_distr[0]]
        uw_maps = [self.theta_distr, self.z_distr[1], self.z_distr[1]]

        # Calculate weighted distribution
        rand_rand[0] += general.prob_convolution(
            w_maps, bins, general.distance, self.bins.bins('s'))

        # Calculate unweighted istribution
        rand_rand[1] += general.prob_convolution(
            uw_maps, bins, general.distance, self.bins.bins('s'))

        return rand_rand

    def get_data_rand(self, cosmo):
        """ Calculate and return DR(s) """
        if self.z_distr is None:
            raise RuntimeError("Comoving distribution is None.")

        print("Calculate DR(s)")

        # Initialize separation distribution and binning
        data_rand = numpy.zeros((2, self.bins.nbins('s')))

        # Set up PDF maps and bins
        bins_theta = self.bins.bins('theta')
        bins_z = self.bins.bins('z')
        bins_r = cosmo.z2r(bins_z)
        bins = [bins_theta, bins_r, bins_r]

        # Calculate weighted distribution
        data_rand[0] += general.prob_convolution2d(
            self.z_theta_distr[0], self.z_distr[0], bins, general.distance, self.bins.bins('s'))

        # Calculate unweighted distribution
        data_rand[1] += general.prob_convolution2d(
            self.z_theta_distr[1], self.z_distr[1], bins, general.distance, self.bins.bins('s'))

        return data_rand
