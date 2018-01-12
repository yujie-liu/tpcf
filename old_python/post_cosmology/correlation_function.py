""" Modules to construct two-point correlation function using KD-Tree
Estimators used: (DD-2*DR+RR)/RR,
where D and R are data and randoms catalogs respectively """
__version__ = 1.00

import os
import configparser
import numpy
from sklearn.neighbors import KDTree, BallTree
from astropy.io import fits

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi


def import_fits(fname_key, fits_reader, region):
    """ Import data into 2-d array
    Inputs:
    + fname_key: string
        Key for data filename in reader.
    + fits_reader: dict
        Must have attributes: "INDEX"=index of headers. "RA", "DEC", "Z",
        "WEIGHT"=corresponded variable names in header.
    + region: dict
        Region of galaxies to import. Must have attributes: "dec_max", "dec_min",
        "ra_max", "ra_min", "z_min", "z_max"
    Outputs:
    + catalog: ndarray or tuple of ndarrays
        Return catalog format in each row [DEC, RA, Z, WEIGHT].
    """
    print("Importing from: {}".format(fits_reader[fname_key]))

    header_index = int(fits_reader["index"])
    hdulist = fits.open(fits_reader[fname_key])
    tbdata = hdulist[header_index].data
    temp_dec = DEG2RAD*tbdata[fits_reader["dec"]]
    temp_ra = DEG2RAD*tbdata[fits_reader["ra"]]
    temp_z = tbdata[fits_reader["z"]]
    try:
        temp_weight_fkp = tbdata[fits_reader["weight_fkp"]]
        temp_weight_noz = tbdata[fits_reader["weight_noz"]]
        temp_weight_cp = tbdata[fits_reader["weight_cp"]]
        temp_weight_sdc = tbdata[fits_reader["weight_sdc"]]
        temp_weight = (temp_weight_sdc * temp_weight_fkp *
                       (temp_weight_noz + temp_weight_cp - 1))
    except KeyError:
        temp_weight = tbdata[fits_reader["weight"]]

    catalog = numpy.array([temp_dec, temp_ra, temp_z, temp_weight]).T
    hdulist.close()

    # cut by region
    dec_min = DEG2RAD*float(region["dec_min"])
    dec_max = DEG2RAD*float(region["dec_max"])
    ra_min = DEG2RAD*float(region["ra_min"])
    ra_max = DEG2RAD*float(region["ra_max"])
    z_min = float(region["z_min"])
    z_max = float(region["z_max"])
    cut = ((dec_min <= temp_dec) & (temp_dec <= dec_max)
           &(ra_min <= temp_ra) & (temp_ra <= ra_max)
           &(z_min <= temp_z) & (temp_z <= z_max))

    print("Data size: {}".format(numpy.sum(cut)))
    return catalog[cut]


def celestial2cart(declination, right_ascension, radius):
    """ Convert Celestial coordiante into Cartesian coordinate system
        Conversion equation is:
         -x = r*cos(dec)*cos(ra)
         -y = r*cos(dec)*sin(ra)
         -z = r*sin(dec)
        Inputs:
        + declination: array
            Values of DEC, declination, in radian.
        + right_ascension: array
            Values of RA, right ascension, in radian.
        + radius: array
            Values of the radius. Should have dimension of length.
        Outputs:
        + cart_x: array
            Values of X, same unit with radius.
        + cart_y: array
            Values of Y, same unit with radius.
        + cart_z: array
            Values of Z. same unit with radius.
        """
    cart_x = radius*numpy.cos(declination)*numpy.cos(right_ascension)
    cart_y = radius*numpy.cos(declination)*numpy.sin(right_ascension)
    cart_z = radius*numpy.sin(declination)
    return cart_x, cart_y, cart_z


def hist2point(hist, bins_x, bins_y, exclude_zeros=True):
    """ Convert 2d histogram into data points with weight. Take bincenter as
        value of points.
        Inputs:
        + hist: 2-d array
            Values of 2d histogram (length(bins_x)-1, length(bins_y-1)).
        + bins_x: array
            Binedges in x-axis.
        + bins_y: array
            Binedges in y-axis.
        + exclude_zeros: bool, default=True
            If True, return non-zero weight bins only.
        Outputs:
        + catalog: ndarray or tuple of ndarrays
            Array of data points with weights. Format in each row [X, Y, Weight]
        """
    center_x = 0.5*(bins_x[:-1]+bins_x[1:])
    center_y = 0.5*(bins_y[:-1]+bins_y[1:])
    grid_x, grid_y = numpy.meshgrid(center_x, center_y)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    hist = hist.T.flatten()
    catalog = numpy.array([grid_x, grid_y, hist]).T
    if exclude_zeros:
        return catalog[hist != 0]
    return catalog


def get_distance(theta, radius1, radius2):
    """ Given two points at radius1 and radius2 with angular separation
    theta (in rad), calulate distance between points"""
    return numpy.sqrt(radius1**2+radius2**2-2*radius1*radius2*numpy.cos(theta))


def get_bins(x_min, x_max, binwidth):
    """ Return the binnings given min, max and width. Apply roundings to
    binwidth such that (max-min)/binwidth is an integer. Also return the new
    binwidth."""
    nbins = int(numpy.ceil((x_max-x_min)/binwidth))
    if nbins <= 0:
        raise Exception("Max must be greater than Min.")
    bins = numpy.linspace(x_min, x_max, nbins+1)
    binw = bins[1]-bins[0]
    return bins, binw


def get_job_index(no_job, total_jobs, job_size):
    """ Calculate start and end index based on job number, total number of jobs,
    and the size of the jobs
    Inputs:
    + no_job: int
        Job number. Must be greater or equal to 0 and less than total_jobs
        (0 <= no_job < total_jobs).
    + total_jobs: int
        Total number of jobs. Must be a positive integer.
    + job_size: int
        Size of the jobs. Must be a positive integer.
    Outputs:
    + job_range: tuple
        Return range index of the job
    """
    # Calculate start and end index based on job number and total number of
    # jobs.
    job_index = numpy.floor(numpy.linspace(0, job_size, total_jobs+1))
    job_index = job_index.astype(int)
    job_range = (job_index[no_job], job_index[no_job+1])
    return job_range


def prob_convolution(map_x, map_y, map_z, bins_x, bins_y, bins_z, dist, bins_s):
    """ Probability convolution given 3 probability density of each dimension
    and a distance pairing distribution.
    Given f(x), g(y), h(z), and function d(x, y, z), will calculate the
    separation function:
        D[s] = Integration[f(x)*g(y)*h(z)*dirac_delta[d(x, y, z) - s]]
    over the parameter space.
    Input:
    + map_x: list, array-like
        Probability along the first dimension.
    + map_y: list, array-like
        Probability along the second dimension.
    + map_z: list, array-like
        Probaility along the third dimension.
    + bins_x: list, array-like
        Binedges along the first dimension. Must have length(map_x)+1.
    + bins_y: list, array-like
        Binedges along the second dimension. Must have length(map_y)+1.
    + bins_z: list, array-like
        Binedges along the third dimension. Must have length(map_z)+1.
    + dist: function
        Pairing function that takes argument x, y, z and compute the separation.
    + bins_s: list, array-like
        Binedges of the output separation distribution.
    Output:
    + pairs_separation: list, array-like
        Values of pairs separation distribution along bins_s. Has length of
        length(bins_s)-1."""
    if len(bins_x) != len(map_x)+1:
        raise ValueError("bins_x must have length(map_x)+1.")
    if len(bins_y) != len(map_y)+1:
        raise ValueError("bins_y must have length(map_y)+1.")
    if len(bins_z) != len(map_z)+1:
        raise ValueError("bins_z must have length(map_z)+1.")
    # Define data points on the x-y surface for integration
    map_xy = numpy.zeros((bins_x.size-1, bins_y.size-1))
    for i in range(bins_x.size-1):
        for j in range(bins_y.size-1):
            map_xy[i, j] = map_x[i]*map_y[j]
    return prob_convolution2d(map_xy, map_z, bins_x, bins_y, bins_z, dist,
                              bins_s)


def prob_convolution2d(map_xy, map_z, bins_x, bins_y, bins_z, dist, bins_s):
    """ Probability convolution given a 2d probability density and a 1d
    and a distance pairing distribution.
    Given f(x, y), g(z), and function d(x, y, z), will calculate the
    separation function:
        D[s] = Integration[f(x, y)*g(z)*dirac_delta[d(x, y, z) - s]]
    over the parameter space.
    Input:
    + map_xy: list, array-like
    + map_z: list, array-like
        Probaility along the third dimension.
    + bins_x: list, array-like
        Binedges along the first dimension. Must have length(map_xy col)+1.
    + bins_y: list, array-like
        Binedges along the second dimension. Must have length(map_xy row)+1.
    + bins_z: list, array-like
        Binedges along the third dimension. Must have length(map_z)+1.
    + dist: function
        Pairing function that takes argument x, y, z and compute the separation.
    + bins_s: list, array-like
        Binedges of the output separation distribution.
    Output:
    + pairs_separation: list, array-like
        Values of pairs separation distribution along bins_s. Has length of
        length(bins_s)-1."""
    if len(bins_x) != map_xy.shape[0]+1:
        raise ValueError("bins_x must have length(map_xy col)+1.")
    if len(bins_y) != map_xy.shape[1]+1:
        raise ValueError("bins_y must have length(map_xy row)+1.")
    if len(bins_z) != len(map_z)+1:
        raise ValueError("bins_z must have length(map_z)+1.")
    # Define data points on the x-y surface for integration
    points_xy = hist2point(map_xy, bins_x, bins_y)

    # Define data points on z axis for integration
    # exclude zeros bins
    cut = map_z > 0
    center_z = 0.5*(bins_z[:-1] + bins_z[1:])
    center_z = center_z[cut]
    weight_z = map_z[cut]

    # Define separation histogram
    pairs_separation = numpy.zeros(bins_s.size-1)

    # Integration
    for i, point_z in enumerate(center_z):
        if i % 100 is 0:
            print(i)
        temp_s = dist(points_xy[:, 0], points_xy[:, 1], point_z)

        # Fill histogram
        temp_weight = weight_z[i]*points_xy[:, 2]
        temp_hist, _ = numpy.histogram(temp_s, bins=bins_s,
                                       weights=temp_weight)
        pairs_separation += temp_hist

    return pairs_separation


def correlation(rand_rand, data_rand, data_data, bins):
    """ Construct two-point correlation function.
    Inputs:
    + rand_rand: array
        Values of separation distribution between pairs of random galaxies
        RR(s).
    + data_rand: array
        Values of separation distribution between pairs of a random point
        and a galaxy DR(s).
    + data_data: array
        Valus of separation distribution between pairs of galaxies DD(s).
    Note: rand_rand, data_rand, data_data must have the same size.
    + bins: array
        Binedges of rand_rand, data_rand, data_data.
    Output:
    + correlation_function: array
        Two-point correlation function computed using equations:
        f = [DD(s)-2*DR(s)+RR(s)]/RR(s)
        If RR(s) = 0, then f = 0
    + correlation_function_ss: array
        Two-point correlation function multiplied by s^2: f(s)s^2
        """
    correlation_function = data_data-2*data_rand+rand_rand
    correlation_function = numpy.divide(correlation_function, rand_rand,
                                        out=numpy.zeros_like(rand_rand),
                                        where=rand_rand != 0)
    center_s = 0.5*(bins[:-1]+bins[1:])
    correlation_function_ss = correlation_function*center_s**2
    return correlation_function, correlation_function_ss


def get_error(hist_w, hist_u):
    """ Get the bin error of weighted and unweighted pairs separation
    histogram. Bin error in unweighted pairs separation is assumed to be
    Poisson. Bin error in weighted pairs separation is given by equation
    error_w = n_w/sqrt(n_u).
    Inputs:
    + hist_w: array
        Values of weighted histogram.
    + hist_u: array
        Values of unweighted histogram.
    Outputs:
    + error: ndarray or tuples of ndarray
        The bun error in the weighted and unweighted histograms respectively.
    """
    error_u = numpy.sqrt(hist_u)
    error_w = numpy.divide(hist_w, numpy.sqrt(hist_u),
                           out=numpy.zeros_like(hist_w),
                           where=hist_u != 0)
    error = numpy.array([error_w, error_u]).T
    return error


class CorrelationFunction():
    """ Class to construct two-point correlation function """
    def __init__(self, config_fname, import_catalog=True):
        """ Constructor takes in configuration file and sets up binning
        variables """
        if not os.path.isfile(config_fname):
            raise IOError("Configuration file not found.")
        # Initialize variables
        self.data_cat = None
        self.rand_cat = None
        self.z_hist = None
        self.angular_points = None
        self.bins_s = None
        self.bins_theta = None
        self.__n_rand = None
        self.__w_sum_rand = None
        self.__w2_sum_rand = None
        self.set_configuration(config_fname, import_catalog)

    def __angular_distance_thread(self, arc_tree, start, end):
        """ Thread function to calculate angular distance distribution f(theta)
        as one-dimensional histogram.
        Inputs:
        + arc_tree: ball tree
            Ball tree fill with data in angular catalog. For more details,
            refer to sklearn.neighbors.BallTree
        + start: int
            Starting index of galaxies in galaxies catalog: index_start = start.
        + end: int
            Ending index of galaxies in galaxies catalog: index_end = end-1.
        Outputs:
        + theta_hist: ndarrays or tuples of ndarrays
            Return values of f(theta).
        """
        # Define angular distance distribution f(theta) as histogram
        nbins_theta = self.bins_theta.size-1
        theta_max = self.bins_theta.max()
        theta_hist = numpy.zeros(nbins_theta)

        print("Construct f(theta) from index {} to {}".format(start, end-1))
        for i, point in enumerate(self.angular_points[start:end]):
            if i % 10000 is 0:
                print(i)
            index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                 r=theta_max,
                                                 return_distance=True)
            temp_weight = point[2]*self.angular_points[:, 2][index[0]]
            temp_hist, _ = numpy.histogram(theta[0], bins=nbins_theta,
                                           range=(0., theta_max),
                                           weights=temp_weight)
            theta_hist += temp_hist

        # Correction for double counting
        theta_hist = theta_hist/2.

        return theta_hist

    def __angular_redshift_thread(self, angular_points, arc_tree,
                                  start, end, mode):
        """ Thread function for calculating angular-redshift distribution
        g(theta, z) as two-dimensional histogram.
        Inputs:
        + angular_points: ndarray
            Angular distribution R(ra, dec) breaks into data points with
            proper weights. Format of each row must be [DEC, RA, WEIGHT].
        + arc_tree: ball tree
            Ball tree fill with data in angular catalog. For more details,
            refer to sklearn.neighbors.BallTree
        + start: int
            Starting index of galaxies in galaxies catalog: index_start = start.
        + end: int
            Ending index of galaxies in galaxies catalog: index_end = end-1.
        + mode: string
            Must be either "data" or "angular". If "data", loop over galaxies
            catalog. If "angular", loop over angular distribution points.
        Outputs:
        + theta_z_hist: ndarray or tuple of ndarrays
            Return values of weighted and unweighted g(theta, z) respectively.
            Each has dimension (length(self.__bins_theta)-1, length(self.__bins_z)-1).
        """
        # Define angular-radial distribution g(theta, z) as weighted and
        # unweighted 2-d histogram respectively
        nbins_theta = self.bins_theta.size-1
        nbins_z = self.z_hist[1].size-1
        theta_max = self.bins_theta.max()
        bins_range = ((0., theta_max),
                      (self.z_hist[1].min(), self.z_hist[1].max()))
        theta_z_hist = numpy.zeros((2, nbins_theta, nbins_z))

        print("Construct g(theta, z) from index {} to {}".format(start, end-1))
        if mode == "data":
            for i, point in enumerate(self.data_cat[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                     r=theta_max,
                                                     return_distance=True)
                temp_z = numpy.repeat(point[2], index[0].size)
                # Fill unweighted histogram
                temp_weight = angular_points[:, 2][index[0]]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_z,
                                                    bins=(nbins_theta, nbins_z),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_z_hist[1] += temp_hist
                # Fill weighted histogram
                temp_weight = temp_weight*point[3]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_z,
                                                    bins=(nbins_theta, nbins_z),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_z_hist[0] += temp_hist
        elif mode == "angular":
            for i, point in enumerate(angular_points[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                     r=theta_max,
                                                     return_distance=True)
                temp_z = self.data_cat[:, 2][index[0]]
                # Fill weighted histogram
                temp_weight = point[2]*self.data_cat[:, 3][index[0]]
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_z,
                                                    bins=(nbins_theta, nbins_z),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_z_hist[0] += temp_hist
                # Fill unweighted histogram
                temp_weight = numpy.repeat(point[2], index[0].size)
                temp_hist, _, _ = numpy.histogram2d(theta[0], temp_z,
                                                    bins=(nbins_theta, nbins_z),
                                                    range=bins_range,
                                                    weights=temp_weight)
                theta_z_hist[1] += temp_hist

        return theta_z_hist

    def __pairs_separation_thread(self, point_cat, tree_cat, tree, start, end):
        """ Thread function for calculating separation distribution.
        Inputs:
        + point_cat: ndarray or tupless of ndarray
            Point catalog in Cartesian coordinates outside tree.
            Each row format [X,Y,Z].
        + tree_cat: ndarray or tupless of ndarray
            Galaxies catalog in Cartesian coordinates inside tree.
            Each row format [X,Y,Z].
        + tree: kd-tree
            K-D tree filled with points from tree_cat.
        + start: int
            Starting index of galaxies in galaxies catalog: index_start = start.
        + end: int
            Ending index of galaxies in galaxies catalog: index_end = end-1.
        Outputs:
        + pairs_seprtion: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted DD(s) respectively.
        """
        # Define weighted and unweighted DD(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.bins_s.size-1
        s_max = self.bins_s.max()
        pairs_separation = numpy.zeros((2, nbins_s))

        print("Calculate DD(s) from index {} to {}".format(start, end-1))
        for i, point in enumerate(point_cat[start:end]):
            if i % 10000 is 0:
                print(i)
            index, dist = tree.query_radius(point[: 3].reshape(1, -1),
                                            r=s_max,
                                            return_distance=True)
            # Fill weighted histogram
            temp_weight = tree_cat[:, 3][index[0]]*point[3]
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            pairs_separation[0] += temp_hist
            # Fill unweighted histogram
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max))
            pairs_separation[1] += temp_hist

        # Correction for double counting
        if numpy.array_equal(point_cat, tree_cat):
            # sum of w^2
            pairs_separation[0][0] -= numpy.sum(point_cat[start:end, 3]**2)
            pairs_separation[1][0] -= end-start
            pairs_separation = pairs_separation/2.

        return pairs_separation

    def set_configuration(self, config_fname, import_catalog=True):
        """ Sets up binning variable based on input configuration file.
            If import_catalog is True, will import catalog data."""
        config = configparser.ConfigParser(os.environ)
        config.read(config_fname)
        binnings = config['BINNING']
        region = config['REGION']

        # Setting up binning variables
        # spatial separation (Mpc/h)
        s_max = float(region["s_max"])
        binwidth_s = float(binnings['binwidth_s'])
        self.bins_s, binwidth_s = get_bins(0., s_max, binwidth_s)

        # comoving distance distribution(Mpc/h)
        z_min = float(region["z_min"])
        z_max = float(region["z_max"])
        binwidth_z = float(binnings['binwidth_z'])
        bins_z, binwidth_z = get_bins(z_min, z_max, binwidth_z)

        theta_max = DEG2RAD*float(region["theta_max"])
        binwidth_theta = DEG2RAD*float(binnings['binwidth_theta'])
        self.bins_theta, binwidth_theta = get_bins(0., theta_max,
                                                   binwidth_theta)

        # Import random and data catalogs
        if import_catalog:
            self.data_cat = import_fits('data_filename', config['INPUT'],
                                        region)
            if config['INPUT']['random_format'] == "catalog":
                self.rand_cat = import_fits('random_filename', config['INPUT'],
                                            region)

                # Setting up some bin variables
                # declination, right ascension, and angular separation (rad)
                dec_min = DEG2RAD*float(region["dec_min"])
                dec_max = DEG2RAD*float(region["dec_max"])
                binwidth_dec = DEG2RAD*float(binnings['binwidth_dec'])
                bins_dec, binwidth_dec = get_bins(dec_min, dec_max,
                                                  binwidth_dec)

                ra_min = DEG2RAD*float(region["ra_min"])
                ra_max = DEG2RAD*float(region["ra_max"])
                binwidth_ra = DEG2RAD*float(binnings['binwidth_ra'])
                bins_ra, binwidth_ra = get_bins(ra_min, ra_max, binwidth_ra)

                # Calculate P(r) and R(ra, dec)
                # Calculate weighted and unweighted radial distribution P(r) as
                # two one-dimensional histograms respectively
                z_hist = numpy.zeros((2, bins_z.size-1))
                z_hist[0] += numpy.histogram(self.rand_cat[:, 2], bins=bins_z,
                                             weights=self.rand_cat[:, 3])[0]
                z_hist[1] += numpy.histogram(self.rand_cat[:, 2],
                                             bins=bins_z)[0]
                z_hist = 1.*z_hist/self.rand_cat.shape[0]
                self.z_hist = [z_hist, bins_z]

                # Calculate the angular distribution R(ra, dec) and breaks into
                # data points with proper weights
                angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                                 self.rand_cat[:, 1],
                                                 bins=(bins_dec, bins_ra))

                self.angular_points = hist2point(*angular_hist)

            elif config['INPUT']['random_format'] == "hist":
                print("Importing distribution from: {}".format(
                    config['INPUT']['random_distribution']))
                with numpy.load(config['INPUT']['random_distribution']) as temp_file:
                    z_hist = temp_file['Z_HIST']
                    bins_z = temp_file['BINS_Z']
                    bins_dec = temp_file['BINS_DEC']
                    bins_ra = temp_file['BINS_RA']
                    self.angular_points = temp_file['ANGULAR_POINTS']
                    self.__n_rand = temp_file['N_DATA']
                    self.__w_sum_rand = temp_file['W_SUM']
                    self.__w2_sum_rand = temp_file['W2_SUM']
                self.z_hist = [z_hist, bins_z]
            else:
                raise ValueError("Argument random_format must be either "
                                 "\"hist\" or \"catalog\"")

        # Print out new configuration information
        print("Configuration:")
        print("Spatial separation (Mpc/h):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(0., s_max,
                                                            binwidth_s))
        print("Comoving distribution (Mpc/h):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(z_min, z_max,
                                                            binwidth_z))
        print("Angular separation (rad):")
        print("-[Min, Max, Binwidth] = [{}, {}, {}]".format(0., theta_max,
                                                            binwidth_theta))
    def redshift_distribution(self):
        """ Return the comoving distance distribution and binedges """
        return self.z_hist

    def angular_distance(self, no_job, total_jobs, leaf=40):
        """ Calculate the angular distance distribution f(theta) as an
        one-dimensional histogram. Binnings are defined in config file.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given radius defined in config file.
        Inputs:
        + no_job: int
            Job number. Must be greater or equal to 0 and less than total_jobs
            (0 <= no_job < total_jobs).
        + total_jobs: int
            Total number of jobs. Must be a positive integer.
        + leaf: int (default=40)
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + theta_hist: array
            Values of f(theta).
        + bins: array
            Binedges of histogram  (length(theta_hist)-1).
        """
        if self.angular_points is None:
            raise TypeError("Angular distribution are not imported.")

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, self.angular_points.shape[0])

        # Create a BallTree and use modified nearest-neighbors algorithm to
        # calculate angular distance up to a given radius.
        arc_tree = BallTree(self.angular_points[:, :2], leaf_size=leaf,
                            metric='haversine')
        theta_hist = self.__angular_distance_thread(arc_tree, *job_range)

        return theta_hist, self.bins_theta

    def angular_redshift(self, no_job, total_jobs, leaf=40):
        """ Calculate g(theta, z), angular-redshift distribution, as a
        two-dimensional histogram. Binnings are defined in config file.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given radius defined in config file.
        Inputs:
        + no_job: int
            Job number. Must be greater or equal to 0 and less than total_jobs
            (0 <= no_job < total_jobs).
        + total_jobs: int
            Total number of jobs. Must be a positive integer.
        + leaf: int (default=40)
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + theta_z_hist: ndarray or tuple of ndarrays
            Return values of weighted and unweighted g(theta, z) respectively.
            Each has dimension (length(bins_theta)-1, length(bins_z)-1).
        + bins_theta: array
            Binedges along x-axis.
        + bins_z: array
            Binedges along y-axis.
        """
        if self.data_cat is None:
            raise TypeError("Catalogs are not imported.")
        if self.angular_points is None:
            raise TypeError("Angular distribution are not imported")

        # Optimizing: Runtime of BallTree modified nearest-neighbors is O(NlogM)
        # where M is the number of points in Tree and N is the number of points
        # for pairings. Thus, the BallTree is created using the size of the
        # smaller catalog, galaxies catalog vs. angular catalog.
        if self.angular_points.shape[0] >= self.data_cat.shape[0]:
            job_size = self.data_cat.shape[0]
            mode = "data"
            arc_tree = BallTree(self.angular_points[:, :2], leaf_size=leaf,
                                metric='haversine')
        else:
            job_size = self.angular_points.shape[0]
            mode = "angular"
            arc_tree = BallTree(self.data_cat[:, :2], leaf_size=leaf,
                                metric='haversine')

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, job_size)
        theta_z_hist = self.__angular_redshift_thread(
            self.angular_points, arc_tree, job_range[0], job_range[1], mode)

        return theta_z_hist, self.bins_theta, self.z_hist[1]

    def rand_rand(self, theta_hist, cosmo):
        """ Calculate separation distribution RR(s) between pairs of randoms.
        Inputs:
        + theta_hist: array
            Values of angular distance distribution f(theta).
        + cosmo: cosmology.Cosmology
            Cosmology parameters to convert redshift z to comoving distance r.
            See cosmology.Cosmology for more detail.
        Outputs:
        + rand_rand: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted RR(s) respectively.
        + bins: array
            Binedges of RR(s) (length(rand_rand_hist)+1).
        """
        if self.z_hist is None:
            raise TypeError("Comoving distribution is not imported")
        if len(theta_hist) != self.bins_theta.size-1:
            raise ValueError("f(theta) must have the length of {}".format(
                self.bins_theta.size-1))

        # Convert redshift distribution P(z) to comoving distance distribution
        # P(r) using input cosmology. Note that only binedges change while
        # the value is unchanged.
        r_hist, bins_r = self.z_hist
        bins_r = cosmo.z2r(bins_r)

        print("Construct RR(s)")
        rand_rand = numpy.zeros((2, self.bins_s.size-1))
        rand_rand[0] += prob_convolution(theta_hist, r_hist[0], r_hist[0],
                                         self.bins_theta, bins_r, bins_r,
                                         get_distance, self.bins_s)
        rand_rand[1] += prob_convolution(theta_hist, r_hist[1], r_hist[1],
                                         self.bins_theta, bins_r, bins_r,
                                         get_distance, self.bins_s)

        print("Construct RR(s)")

        return rand_rand, self.bins_s

    def data_rand(self, theta_z_hist, cosmo):
        """ Calculate separation distribution DR(s) between pairs of a random
        point and a galaxy.
        Inputs:
        + theta_z_hist: ndarrays or tuples of ndarrays
            Values of weighted and unweighted g(theta, z) respectively.
            Dimension must be (2, length(self.__bins_theta)-1, length(self.__bins_z)-1).
        + cosmo: cosmology.Cosmology
            Cosmology parameters to convert redshift z to comoving distance r.
            See cosmology.Cosmology for more detail.
        Outputs:
        + data_rand: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted DR(s) respectively.
        + bins: array
            Binedges of DR(s) (length(data_rand_hist)+1).
        """
        if self.z_hist is None:
            raise TypeError("Comoving distribution is not imported")
        if theta_z_hist.shape != (2, self.bins_theta.size-1,
                                  self.z_hist[1].size-1):
            print(theta_z_hist.shape)
            raise ValueError(
                ("g(theta, z) must have the dimension of (2, {}, {})").format(
                    self.bins_theta.size-1, self.z_hist[1].size-1))

        # Convert redshift distribution P(z) to comoving distance distribution
        # P(r) using input cosmology. Note that only binedges change while
        # the value is unchanged.
        r_hist, bins_r = self.z_hist
        bins_r = cosmo.z2r(bins_r)

        print("Construct DR(s)")
        data_rand = numpy.zeros((2, self.bins_s.size-1))
        data_rand[0] += prob_convolution2d(theta_z_hist[0], r_hist[0],
                                           self.bins_theta, bins_r, bins_r,
                                           get_distance, self.bins_s)
        data_rand[1] += prob_convolution2d(theta_z_hist[1], r_hist[1],
                                           self.bins_theta, bins_r, bins_r,
                                           get_distance, self.bins_s)
        return data_rand, self.bins_s

    def pairs_separation(self, no_job, total_jobs, cosmo, out="DD", leaf=40):
        """ Calculate separation distribution between pairs of galaxies.
        Use a modfied nearest-neighbors k-d tree algorithm to calculate distance
        up to a given radius defined in config file.
        Metric: Euclidean (or Minkowski with p=2).
        Inputs:
        + no_job: int
            Job number. Must be greater or equal to 0 and less than total_jobs
            (0 <= no_job < total_jobs).
        + total_jobs: int
            Total number of jobs.
        + cosmo: cosmology.Cosmology
            Cosmology parameters to convert redshift z to comoving distance r.
            See cosmology.Cosmology for more detail.
        + out: string (default="DD")
            Valid argument are "RR", "DR", DD". Distribution to calculate.
        + leaf: int (default=40)
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.BallTree.
        Outputs:
        + pairs_separation: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted pairs_separation
            respectively.
        + bins: array
            Binedges of DD(s) (length(data_data_hist)+1).
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Choose catalogs based on input
        if out == "DD":
            tree_cat = self.data_cat
            point_cat = self.data_cat
        elif out == "RR":
            tree_cat = self.rand_cat
            point_cat = self.rand_cat
        elif out == "DR":
            # Optimizing: Run time of k-d tree modified nearest-neighbors is
            # O(NlogM) where M is the number of points in Tree and N is the
            # number of points for pairings. Thus, the kd-tree is created using
            # the smaller of galaxies catalog and randoms catalog.
            if self.data_cat.shape[0] > self.rand_cat.shape[0]:
                tree_cat = self.data_cat
                point_cat = self.rand_cat
            else:
                tree_cat = self.rand_cat
                point_cat = self.data_cat

        # Convert Celestial coordinate into Cartesian coordinate and create
        # k-d tree and point cartesian catalog.
        # Point cartesian catalog
        temp = celestial2cart(point_cat[:, 0], point_cat[:, 1],
                              cosmo.z2r(point_cat[:, 2]))
        cart_point_cat = numpy.array([temp[0], temp[1], temp[2],
                                      point_cat[:, 3]]).T

        # k-d tree
        temp = celestial2cart(tree_cat[:, 0], tree_cat[:, 1],
                              cosmo.z2r(tree_cat[:, 2]))
        cart_tree_cat = numpy.array([temp[0], temp[1], temp[2],
                                     tree_cat[:, 3]]).T
        tree = KDTree(cart_tree_cat[:, :3], leaf_size=leaf, metric='euclidean')

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, cart_point_cat.shape[0])

        # Compute pairs separation using modified nearest-neighbors algorithm
        # to calculate angular distance up to a given radius.
        pairs_separation = self.__pairs_separation_thread(cart_point_cat,
                                                          cart_tree_cat, tree,
                                                          *job_range)

        return pairs_separation, self.bins_s


    def normalization(self, weighted=True):
        """ Calculate and return normalization factor
        Inputs:
        + weighted: bool (default=True)
            If True, return weighted normalization constants.
            If False, return unweighted normalization constants.
        Outputs:
        + norm_rr: int, float
            Normalization constant for RR(s)
        + norm_dr: int, float
            Normalization constant for DR(s)
        + norm_rr: int, float
            Normalization constant for DD(s)
        Note:
        Equations for unweighted normalization constant:
        - norm_rr = 0.5*n_rand*(n_rand-1)
        - norm_dd = 0.5*n_data*(n_data-1)
        - norm_dr = n_rand*n_data
        where n_rand and n_data are the size of the randoms and galaxies
        catalog respectively.
        Equations for weighted normalization constant:
        - norm_rr = 0.5*(sum_w_rand^2-sum_w2_rand)
        - norm_dd = 0.5*(sum_w_data^2-sum_w2_rand)
        - norm_dr = sum_w_data*sum_w_rand
        where sum_w_rand and sum_w_data is the sum of weights, sum_w2_rand
        and sum_w2_data is the sum of weights squared in randoms and galaxies
        catalog respectively.
        """
        if self.data_cat is None:
            raise TypeError("Catalogs are not imported.")

        if weighted:
            # Calculate weighted normalization constant
            if self.rand_cat is None:
                w_sum_rand = self.__w_sum_rand
                w2_sum_rand = self.__w2_sum_rand
            else:
                w_sum_rand = numpy.sum(self.rand_cat[:, 3])
                w2_sum_rand = numpy.sum(self.rand_cat[:, 3]**2)
            w_sum_data = numpy.sum(self.data_cat[:, 3])
            w2_sum_data = numpy.sum(self.data_cat[:, 3]**2)

            norm_rr = 0.5*(w_sum_rand**2-w2_sum_rand)
            norm_dd = 0.5*(w_sum_data**2-w2_sum_data)
            norm_dr = w_sum_rand*w_sum_data

            return norm_rr, norm_dr, norm_dd

        # Calculate unweighted normalization constant
        if self.rand_cat is None:
            n_rand = self.__n_rand
        else:
            n_rand = self.rand_cat.shape[0]
        n_data = self.data_cat.shape[0]

        norm_rr = 0.5*n_rand*(n_rand-1)
        norm_dd = 0.5*n_data*(n_data-1)
        norm_dr = n_data*n_rand

        return norm_rr, norm_dr, norm_dd
