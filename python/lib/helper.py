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

class CorrelationHelper(object):
    """ Class to handle multiprocess correlation function calculation """
    def __init__(self, bins, cosmo=None):
        """ Constructor set up attributes """

        # Initialize binnings
        self.bins = bins

        # Initalize norm factor
        self.norm = {"dd": None, "dr": None, "rr": None}

        # Initialize histogram
        self.data_data = numpy.zeros((2, self.bins.nbins('s')))
        self.theta_distr = numpy.zeros(self.bins.nbins('theta'))
        self.rz_theta_distr = numpy.zeros((2, self.bins.nbins('theta'), self.bins.nbins('z')))
        self.rz_distr = None

        # Initalize cosmology
        self.cosmo = cosmo

    def set_rz_distr(self, rz_distr):
        """ Setting up P(r) """
        self.rz_distr = rz_distr

    def set_norm(self, norm_dd, norm_dr, norm_rr):
        """ Setting up normalization """
        self.norm["dd"] = norm_dd
        self.norm["dr"] = norm_dr
        self.norm["rr"] = norm_rr

    def set_dd(self, tree, catalog, job_helper=None):
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

    def set_rz_theta_distr(self, tree, data_catalog, angular_catalog, mode, job_helper=None):
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
        nbins = (self.bins.nbins('theta'), self.bins.nbins('z'))
        theta_max = self.bins.max('theta')
        rz_min = self.bins.min('z')
        rz_max = self.bins.max('z')
        if self.cosmo is not None:
            rz_min = self.cosmo.z2r(rz_min)
            rz_max = self.cosmo.z2r(rz_max)
        limit = ((0., theta_max), (rz_min, rz_max))

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
                    theta[0], z, bins=nbins, range=limit, weights=weights)
                self.rz_theta_distr[1] += hist

                # Fill weighted histogram
                # weight is the product of the weight of the data point and the weight
                # of the angular point.
                weights = weights*point[3]
                hist, _, _ = numpy.histogram2d(
                    theta[0], z, bins=nbins, range=limit, weights=weights)
                self.rz_theta_distr[0] += hist

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
                    theta[0], z, bins=nbins, range=limit, weights=weights)
                self.rz_theta_distr[0] += hist

                # Fill unweighted histogram
                # weight is the weight of the angular point
                hist, _, _ = numpy.histogram2d(
                    theta[0], z, bins=nbins, range=limit)
                self.rz_theta_distr[1] += hist*point[2]

    def add(self, other):
        """ Add another part """
        self.data_data = self.data_data+other.data_data
        self.theta_distr = self.theta_distr+other.theta_distr
        self.rz_theta_distr = self.rz_theta_distr+other.rz_theta_distr

    def get_dd(self):
        """ Return DD(s) """
        print("Calculate DD(s)")
        return self.data_data

    def get_rr(self, cosmo=None):
        """ Calculate and return RR(s) """
        if self.rz_distr is None:
            return RuntimeError("Redshift/comoving distribution is None.")

        print("Calculate RR(s)")

        # Initialize separation distribution and binning
        rand_rand = numpy.zeros((2, self.bins.nbins('s')))

        # Set up binnings
        # Angular separation bins
        bins_theta = self.bins.bins('theta')

        # Apply cosmology to calculate comoving bins
        if self.cosmo is None and cosmo is not None:
            bins_r = cosmo.z2r(self.bins.bins('z'))  # Uniform over 'z', NOT 'r'
        else:
            bins_r = self.bins.bins('z', self.cosmo) # Uniform over 'r'

        # Take the bins center
        bins_theta = 0.5*(bins_theta[:-1]+bins_theta[1:])
        bins_r = 0.5*(bins_r[:-1]+bins_r[1:])

        # Calculate 4-dimensional weights matrix
        weights = (self.rz_distr[:, None, :, None]*self.rz_distr[:, None, None, :]
                   *self.theta_distr[None, :, None, None])

        # Calculate 3-dimensional separation matrix
        dist = general.distance(
            bins_theta[:, None, None], bins_r[None, :, None], bins_r[None, None, :])

        # Calculate RR(s) by histograming
        for i in range(2):
            rand_rand[i], _ = numpy.histogram(dist, bins=self.bins.bins('s'), weights=weights[i])

        return rand_rand

    def get_dr(self, cosmo=None):
        """ Calculate and return DR(s) """
        if self.rz_distr is None:
            raise RuntimeError("Comoving distribution is None.")

        print("Calculate DR(s)")

        # Initialize separation distribution and binning
        data_rand = numpy.zeros((2, self.bins.nbins('s')))

        # Set up binnings
        # Angular separation bins
        bins_theta = self.bins.bins('theta')

        # Apply cosmology to calculate comoving bins
        if self.cosmo is None and cosmo is not None:
            bins_r = cosmo.z2r(self.bins.bins('z'))  # Uniform over 'z', NOT 'r'
        else:
            bins_r = self.bins.bins('z', self.cosmo) # Uniform over 'r'

        # Take the bins center
        bins_theta = 0.5*(bins_theta[:-1]+bins_theta[1:])
        bins_r = 0.5*(bins_r[:-1]+bins_r[1:])

        # Construct a 4d matrix for weights
        weights = self.rz_distr[:, None, None, :]*self.rz_theta_distr[:, :, :, None]

        # Construct a 3d matrix for pairing
        dist = general.distance(
            bins_theta[:, None, None], bins_r[None, :, None], bins_r[None, None, :])

        # Construct DR(s)
        for i in range(2):
            data_rand[i], _ = numpy.histogram(dist, bins=self.bins.bins('s'), weights=weights[i])

        return data_rand

    def get_rr_dr(self, cosmo=None):
        """ Calculate and return RR(s) and DR(s) """
        if self.rz_distr is None:
            return RuntimeError("Redshift/comoving distribution is None.")

        # Initialize separation distribution and binning
        rand_rand = numpy.zeros((2, self.bins.nbins('s')))
        data_rand = numpy.zeros((2, self.bins.nbins('s')))

        # Set up binnings
        # Angular separation bins
        bins_theta = self.bins.bins('theta')

        # Apply cosmology to calculate comoving bins
        if self.cosmo is None and cosmo is not None:
            bins_r = cosmo.z2r(self.bins.bins('z'))  # Uniform over 'z', NOT 'r'
        else:
            bins_r = self.bins.bins('z', self.cosmo) # Uniform over 'r'

        # Take the bins center
        bins_theta = 0.5*(bins_theta[:-1]+bins_theta[1:])
        bins_r = 0.5*(bins_r[:-1]+bins_r[1:])

        # Calculate 4-dimensional weights matrix
        weights_rr = (self.rz_distr[:, None, :, None]*self.rz_distr[:, None, None, :]
                      *self.theta_distr[None, :, None, None])
        weights_dr = self.rz_distr[:, None, None, :]*self.rz_theta_distr[:, :, :, None]

        # Calculate 3-dimensional separation matrix
        dist = general.distance(
            bins_theta[:, None, None], bins_r[None, :, None], bins_r[None, None, :])

        # Calculate RR(s) and DR(s) by histograming
        print("Calculate RR(s)")
        print("Calculate DR(s)")
        for i in range(2):
            rand_rand[i], _ = numpy.histogram(
                dist, bins=self.bins.bins('s'), weights=weights_rr[i])
            data_rand[i], _ = numpy.histogram(
                dist, bins=self.bins.bins('s'), weights=weights_dr[i])

        return rand_rand, data_rand
