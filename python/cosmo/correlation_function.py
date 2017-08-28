""" Modules to construct two-point correlation function using KD-Tree
Estimators used: (DD-2*DR+RR)/RR,
where D and R are data and randoms catalogs respectively """
__version__ = 1.40

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
        temp_weight = (temp_weight_sdc*temp_weight_fkp*
                       (temp_weight_noz + temp_weight_cp -1))
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

def get_distance(radius1, radius2, theta):
    """ Given two points at radius1 and radius2 with angular separation
    theta (in rad), calulate distance between points"""
    return numpy.sqrt(radius1**2+radius2**2-2*radius1*radius2*numpy.cos(theta))


def get_bins(x_min, x_max, binwidth):
    """ Return the binnings given min, max and width """
    nbins = int(numpy.ceil((x_max-x_min)/binwidth))
    return numpy.linspace(x_min, x_max, nbins+1)

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


class CorrelationFunction():
    """ Class to construct two-point correlation function """
    def __init__(self, config_fname, analysis_mode=False):
        """ Constructor takes in configuration file and sets up binning
        variables."""
        config = configparser.ConfigParser()
        config.read(config_fname)
        binnings = config['BINNING']
        region = config['REGION']

        # Import random and data catalogs
        self.data_cat = None
        self.rand_cat = None
        if not analysis_mode:
            reader = config['FITS']
            self.data_cat = import_fits('data_filename', reader, region)
            self.rand_cat = import_fits('random_filename', reader, region)

        # Setting up binning variables
        binwidth_ra = DEG2RAD*float(binnings['binwidth_ra'])
        binwidth_dec = DEG2RAD*float(binnings['binwidth_dec'])
        binwidth_theta = DEG2RAD*float(binnings['binwidth_theta'])
        binwidth_z = float(binnings['binwidth_z'])
        binwidth_s = float(binnings['binwidth_s'])
        dec_min = DEG2RAD*float(region["dec_min"])
        dec_max = DEG2RAD*float(region["dec_max"])
        ra_min = DEG2RAD*float(region["ra_min"])
        ra_max = DEG2RAD*float(region["ra_max"])
        z_min = float(region["z_min"])
        z_max = float(region["z_max"])
        s_max = float(region["s_max"])
        theta_max = DEG2RAD*float(region["theta_max"])

        self.__bins_ra = get_bins(ra_min, ra_max, binwidth_ra)
        self.__bins_dec = get_bins(dec_min, dec_max, binwidth_dec)
        self.__bins_z = get_bins(z_min, z_max, binwidth_z)
        self.__bins_s = get_bins(0., s_max, binwidth_s)
        self.__bins_theta = get_bins(0., theta_max, binwidth_theta)

    def __angular_distance_thread(self, angular_points, arc_tree, start, end):
        """ Thread function to calculate angular distance distribution f(theta)
        as one-dimensional histogram.
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
        Outputs:
        + theta_hist: ndarrays or tuples of ndarrays
            Return values of f(theta).
        """
        # Define angular distance distribution f(theta) as histogram
        nbins_theta = self.__bins_theta.size-1
        theta_max = self.__bins_theta.max()
        theta_hist = numpy.zeros(nbins_theta)

        print("Construct f(theta) from index {} to {}".format(start, end-1))
        for i, point in enumerate(angular_points[start:end]):
            if i % 10000 is 0:
                print(i)
            index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                 r=theta_max,
                                                 return_distance=True)
            temp_weight = point[2]*angular_points[:, 2][index[0]]
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
        # Define angular-radial distribution g(theta, r) as weighted and
        # unweighted 2-d histogram respectively
        nbins_theta = self.__bins_theta.size-1
        nbins_z = self.__bins_z.size-1
        theta_max = self.__bins_theta.max()
        bins_range = ((0., theta_max),
                      (self.__bins_z.min(), self.__bins_z.max()))
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

    def __data_data_thread(self, data_cat, data_tree, start, end):
        """ Thread function for calculating separation distribution DD(s)
        between pairs of galaxies.
        Inputs:
        + data_cat: ndarray or tupless of ndarray
            Galaxies catalog in Cartesian coordinates. Each row format [X,Y,Z].
        + data_tree: k-d tree
            K-D tree fill with data in galaxies catalog. For more details,
            refer to sklearn.neighbors.KDTree
        + start: int
            Starting index of galaxies in galaxies catalog: index_start = start.
        + end: int
            Ending index of galaxies in galaxies catalog: index_end = end-1.
        Outputs:
        + data_data: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted DD(s) respectively.
        """
        # Define weighted and unweighted DD(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        data_data = numpy.zeros((2, nbins_s))

        print("Calculate DD(s) from index {} to {}".format(start, end-1))
        for i, point in enumerate(data_cat[start:end]):
            if i % 10000 is 0:
                print(i)
            index, dist = data_tree.query_radius(point[: 3].reshape(1, -1),
                                                 r=s_max,
                                                 return_distance=True)
            # Fill weighted histogram
            temp_weight = data_cat[:, 3][index[0]]*point[3]
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_data[0] += temp_hist
            # Fill unweighted histogram
            temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                           range=(0., s_max))
            data_data[1] += temp_hist

        # Correction for double counting
        data_data[0][0] -= numpy.sum(data_cat[start:end, 3]**2) # sum of w^2
        data_data[1][0] -= end-start
        data_data = data_data/2.

        return data_data

    def redshift_distribution(self, cosmo=None):
        """ Calculate weighted and unweighted redshift distribution P(z) as
        two one-dimensional histograms. If cosmology is given, return weighted
        and unweighted comoving distance distribution P(r) instead. Note that
        even binnings in P(z) will result in uneven binnings in P(r).
        Inputs:
        + cosmo: cosmology.Cosmology
            Cosmology parameters to convert redshift z to comoving distance r.
            See cosmology.Cosmology for more detail.
        Outputs:
        + hist: array
            Values of weighted and unweighted P(z).
        + binedges: array
            Return binedges (length(hist)+1)
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Calculate weighted and unweighted radial distribution P(r) as two
        # one-dimensional histograms respectively.
        z_hist = numpy.zeros((2, self.__bins_z.size-1))
        z_hist[0] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_z,
                                     weights=self.rand_cat[:, 3])[0]
        z_hist[1] += numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_z)[0]
        z_hist = 1.*z_hist/self.rand_cat.shape[0]
        if cosmo is None:
            return z_hist, self.__bins_z
        return z_hist, cosmo.z2r(self.__bins_z)

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
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Compute 2d angular distribution R(ra, dec) and breaks them into data
        # points with proper weights.
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular_points = hist2point(*angular_hist)

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, angular_points.shape[0])

        # Create a BallTree and use modified nearest-neighbors algorithm to
        # calculate angular distance up to a given radius.
        arc_tree = BallTree(angular_points[:, :2], leaf_size=leaf,
                            metric='haversine')
        theta_hist = self.__angular_distance_thread(angular_points, arc_tree,
                                                    *job_range)

        return theta_hist, self.__bins_theta

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
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Compute 2d angular distribution R(ra, dec) and breaks them into data
        # points with proper weights.
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular_points = hist2point(*angular_hist)

        # Optimizing: Runtime of BallTree modified nearest-neighbors is O(NlogM)
        # where M is the number of points in Tree and N is the number of points
        # for pairings. Thus, the BallTree is created using the size of the
        # smaller catalog, galaxies catalog vs. angular catalog.
        if angular_points.shape[0] >= self.data_cat.shape[0]:
            job_size = self.data_cat.shape[0]
            mode = "data"
            arc_tree = BallTree(angular_points[:, :2], leaf_size=leaf,
                                metric='haversine')
        else:
            job_size = angular_points.shape[0]
            mode = "angular"
            arc_tree = BallTree(self.data_cat[:, :2], leaf_size=leaf,
                                metric='haversine')

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, job_size)
        theta_z_hist = self.__angular_redshift_thread(angular_points,
                                                      arc_tree,
                                                      job_range[0],
                                                      job_range[1], mode)

        return theta_z_hist, self.__bins_theta, self.__bins_z

    def rand_rand(self, theta_hist, z_hist, cosmo):
        """ Calculate separation distribution RR(s) between pairs of randoms.
        Inputs:
        + theta_hist: array
            Values of angular distance distribution f(theta).
        + z_hist: ndarrays or tuples of ndarrays
            Values of the weighted and unweighted of the comoving distance
            distribution P(r) respective. Dimension must be
            (2, length(self.__bins_z)-1).
        + cosmo: cosmology.Cosmology
            Cosmology parameters to convert redshift z to comoving distance r.
            See cosmology.Cosmology for more detail.
        Outputs:
        + rand_rand: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted RR(s) respectively.
        + bins: array
            Binedges of RR(s) (length(rand_rand_hist)+1).
        """
        # Convert redshift distribution P(z) to comoving distance distribution
        # P(r) using input cosmology. Note that only binedges change while
        # the value is unchanged.
        bins_r = cosmo.z2r(self.__bins_z)

        # Convert f(theta) and P(r) into data points, weighted and unweighted
        weight = numpy.zeros((2, theta_hist.size, z_hist.shape[1]))
        for i in range(theta_hist.size):
            for j in range(z_hist.shape[1]):
                weight[0][i, j] = theta_hist[i]*z_hist[0][j]
                weight[1][i, j] = theta_hist[i]*z_hist[1][j]
        temp_points = [hist2point(weight[0], self.__bins_theta, bins_r),
                       hist2point(weight[1], self.__bins_theta, bins_r)]
        center_r = 0.5*(bins_r[:-1]+bins_r[1:])
        center_r = center_r[z_hist[1] != 0]

        # Define weighted and unweighted RR(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        rand_rand = numpy.zeros((2, nbins_s))

        # Integration
        print("Construct RR(s)")
        for i, temp_r in enumerate(center_r):
            if i % 100 is 0:
                print(i)
            temp_s = get_distance(temp_r, temp_points[0][:, 1],
                                  temp_points[0][:, 0])

            # Fill weighted histogram
            temp_weight = z_hist[0][i]*temp_points[0][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            rand_rand[0] += temp_hist
            # Fill unweighted histogram
            temp_weight = z_hist[1][i]*temp_points[1][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, bins=nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            rand_rand[1] += temp_hist

        return rand_rand, self.__bins_s

    def data_rand(self, theta_z_hist, z_hist, cosmo):
        """ Calculate separation distribution DR(s) between pairs of a random
        point and a galaxy.
        Inputs:
        + theta_z_hist: ndarrays or tuples of ndarrays
            Values of weighted and unweighted g(theta, r) respectively.
            Dimension must be (2, length(self.__bins_theta)-1, length(self.__bins_r)-1).
        + z_hist: ndarrays or tuples of ndarrays
            Values of the weighted and unweighted of the comoving distance
            distribution P(r) respective. Dimension must be
            (2, length(self.__bins_z)-1).
        + cosmo: cosmology.Cosmology
            Cosmology parameters to convert redshift z to comoving distance r.
            See cosmology.Cosmology for more detail.
        Outputs:
        + data_rand: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted DR(s) respectively.
        + bins: array
            Binedges of DR(s) (length(data_rand_hist)+1).
        """
        # Convert redshift distribution P(z) to comoving distance distribution
        # P(r) using input cosmology. Note that only binedges change while
        # the value is unchanged.
        bins_r = cosmo.z2r(self.__bins_z)

        # Convert g(theta, r) into data points, weighted and unweighted
        temp_points = (
            (hist2point(theta_z_hist[0], self.__bins_theta, bins_r),
             hist2point(theta_z_hist[1], self.__bins_theta, bins_r)))

        # Define weighted and unweighted DR(s) as two one-dimensional
        # histograms respectively.
        nbins_s = self.__bins_s.size-1
        s_max = self.__bins_s.max()
        data_rand = numpy.zeros((2, nbins_s))

        # Integration
        print("Construct DR(s)")
        center_r = 0.5*(bins_r[:-1]+bins_r[1:])
        for i, temp_r in enumerate(center_r[z_hist[1] != 0]):
            if i % 100 is 0:
                print(i)
            temp_s = get_distance(temp_r, temp_points[0][:, 1],
                                  temp_points[0][:, 0])
            # Fill weighted histogram
            temp_weight = z_hist[0][i]*temp_points[0][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_rand[0] += temp_hist
            # Fill unweighted histogram
            temp_weight = z_hist[1][i]*temp_points[1][:, 2]
            temp_hist, _ = numpy.histogram(temp_s, nbins_s,
                                           range=(0., s_max),
                                           weights=temp_weight)
            data_rand[1] += temp_hist

        return data_rand, self.__bins_s

    def data_data(self, no_job, total_jobs, cosmo, leaf=40):
        """ Calculate separation distribution DD(s) between pairs of galaxies.
        Use a modified nearest-neighbors KDTree algorithm to calculate distance
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
        + leaf: int (default=40)
            Number of points at which to switch to brute-force. For a specified
            leaf_size, a leaf node is guaranteed to satisfy
            leaf_size <= n_points <= 2*leaf_size, except in the case that
            n_samples < leaf_size. More details in sklearn.neighbors.KDTree.
        Outputs:
        + data_data: ndarrays or tuples of ndarrays
            Return values of weighted and unweighted DD(s) respectively.
        + bins: array
            Binedges of DD(s) (length(data_data_hist)+1).
        """
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        # Convert redshift z into comoving distance r
        temp_r = cosmo.z2r(self.data_cat[:, 2])

        # Convert Celestial coordinate into Cartesian coordinate
        # Conversion equation is:
        # x = r*cos(dec)*cos(ra)
        # y = r*cos(dec)*sin(ra)
        # z = r*sin(dec
        temp_dec = self.data_cat[:, 0]
        temp_ra = self.data_cat[:, 1]
        temp_weight = self.data_cat[:, 3]
        cart_x = temp_r*numpy.cos(temp_dec)*numpy.cos(temp_ra)
        cart_y = temp_r*numpy.cos(temp_dec)*numpy.sin(temp_ra)
        cart_z = temp_r*numpy.sin(temp_dec)
        cart_cat = numpy.array([cart_x, cart_y, cart_z, temp_weight]).T

        # Calculate start and end index based on job number and total number of
        # jobs.
        job_range = get_job_index(no_job, total_jobs, cart_cat.shape[0])

        # Create KD-tree and compute DD(s) using modified nearest-neighbors
        # algorithm to calculate angular distance up to a given radius.
        cart_tree = KDTree(cart_cat[:, :3], leaf_size=leaf, metric='euclidean')
        data_data = self.__data_data_thread(cart_cat, cart_tree, *job_range)

        return data_data, self.__bins_s

    def correlation(self, rand_rand, data_rand, data_data, bins):
        """ Construct two-point correlation function.
        Inputs:
        + rand_rand: array
            Values of separation distribution between pairs of random galaxies
            RR(s).
        + data_rand: array
            Values of separation distribution between pairs of a random point
            and a galaxy DR(s).
        + data_data: array
            Valus of separation distribution between paris of galaxies DD(s).
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

    # Compute normalization constant
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
        if self.data_cat is None or self.rand_cat is None:
            raise TypeError("Catalogs are not imported.")

        if weighted:
            # Calculate weighted normalization constant
            w_sum_rand = numpy.sum(self.rand_cat[:, 3])
            w_sum_data = numpy.sum(self.data_cat[:, 3])
            w2_sum_rand = numpy.sum(self.rand_cat[:, 3]**2)
            w2_sum_data = numpy.sum(self.data_cat[:, 3]**2)

            norm_rr = 0.5*(w_sum_rand**2-w2_sum_rand)
            norm_dd = 0.5*(w_sum_data**2-w2_sum_data)
            norm_dr = w_sum_rand*w_sum_data

            return norm_rr, norm_dr, norm_dd

        # Calculate unweighted normalization constant
        n_rand = self.rand_cat.shape[0]
        n_data = self.data_cat.shape[0]

        norm_rr = 0.5*n_rand*(n_rand-1)
        norm_dd = 0.5*n_data*(n_data-1)
        norm_dr = n_data*n_rand

        return norm_rr, norm_dr, norm_dd
