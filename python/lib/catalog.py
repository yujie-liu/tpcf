""" Module to handle galaxy survey catalogs """

# Python modules
import numpy
from astropy.table import Table
from sklearn.neighbors import KDTree, BallTree

# User-defined modules
import lib.general as general
from lib.helper import JobHelper

class DataCatalog(object):
    """ Class to handle data point catalogs. Data point catalogs are catalogs that have
    the coordinates (dec, ra, z) of each data point. """

    def __init__(self, reader, limit, cosmo):
        """ Initialize data point catalog.
        Inputs:
        + reader: dict
            Dictionary with path and properties of catalog file.
            Keys: path, dec, ra, z. Values format: str
        + limit: dict
            Dictionary with boundaries (inclusive) of catalog. If None, use full catalog.
            Keys: dec, ra, r. Values format (min, max)
        + cosmo: cosmology.Cosmology
            Cosmology object to convert redshift to comoving distance."""
        self.ntotal = 0
        self.catalog = None
        self.set_catalog(reader, cosmo)
        if limit is not None:
            self.set_limit(limit)

    def set_catalog(self, reader, cosmo):
        """ Read in data point catalog from FITS file. Convert redshift to
        comoving distance.
        Inputs:
        + reader: dict
            Dictionary with path and properties of catalog file.
            Keys: path, dec, ra, z. Values format: str
        + cosmo: cosmology.Cosmology
            Cosmology class to convert redshift to comoving distance."""

        # Read in configuration file
        print("Import catalog from: {}".format(reader['path']))

        # Read in table
        table = Table.read(reader['path'])
        dec = numpy.deg2rad(table[reader['dec']].data)
        ra = numpy.deg2rad(table[reader['ra']].data)
        r = cosmo.z2r(table[reader['z']].data)
        try:
            w = table[reader['weight']]
        except KeyError:
            w_fkp = table[reader['weight_fkp']].data
            w_noz = table[reader['weight_noz']].data
            w_cp = table[reader['weight_cp']].data
            w_sdc = table[reader['weight_sdc']].data
            w = w_sdc*w_fkp*(w_noz+w_cp-1)

        self.catalog = numpy.array([dec, ra, r, w]).T
        self.ntotal = self.catalog.shape[0]

    def set_limit(self, limit):
        """ Set boundaries of catalog
        Inputs:
        + limit: dict
            Dictionary with boundaries (inclusive) of catalog
            Keys: dec, ra, r. Values format: (min, max) """

        # Read limit and perform array slicing
        dec, ra, r = self.catalog[:, :3].T
        index = ((limit['dec'][0] <= dec) & (dec <= limit['dec'][1])
                 & (limit['ra'][0] <= ra) & (ra <= limit['ra'][1])
                 & (limit['r'][0] <= r) & (r <= limit['r'][1]))

        self.catalog = self.catalog[index]
        self.ntotal = self.catalog.shape[0]

    def to_distr(self, limit, nbins):
        """ Convert into DistrCatalog and return.
        Inputs:
        + limit: list, tuple, ndarray, dict
            Bins range in order 'dec', 'ra', 'r'.
            If dict, use values from keys 'dec', 'ra', 'r'
        + nbins: list, tuple, ndarray, dict
            Number of bins in order 'dec', 'ra', 'r'.
            If dict, use values from keys 'dec', 'ra', 'r'
        Outputs:
        + distr_catalog: DistrCatalog
            Distribution catalog """

        # Initialize bins range and number of bins
        num_bins = nbins
        bins_range = limit
        if isinstance(bins_range, dict):
            bins_range = (limit['dec'], limit['ra'], limit['r'])
        if isinstance(num_bins, dict):
            num_bins = (nbins['dec'], nbins['ra'], nbins['r'])

        # Calculate comoving distribution
        r_distr_w, bins_r = self.comoving_distr(
            bins_range[2], num_bins[2], weighted=True, normed=True)
        r_distr_uw, _ = self.comoving_distr(
            bins_range[2], num_bins[2], weighted=False, normed=True)

        # Calculate angular distribution as a set of weighted data points
        angular_distr, bins_dec, bins_ra = self.angular_distr(
            bins_range[:2], num_bins[:2], weighted=False, normed=False)

        # Set up DistrCatalog attributes
        distr_catalog = DistrCatalog()
        distr_catalog.r_distr = numpy.array([r_distr_w, r_distr_uw])
        distr_catalog.angular_distr = general.hist2point(angular_distr, bins_dec, bins_ra)
        distr_catalog.bins_r = bins_r
        distr_catalog.bins_dec = bins_dec
        distr_catalog.bins_ra = bins_ra
        distr_catalog.norm_vars['ntotal'] = self.ntotal
        distr_catalog.norm_vars['sum_w'] = numpy.sum(self.catalog[:, 3])
        distr_catalog.norm_vars['sum_w2'] = numpy.sum(self.catalog[:, 3]**2)

        return distr_catalog

    def comoving_distr(self, limit, nbins, weighted=False, normed=False):
        """ Calculate comoving distance distribution
        Inputs:
        + limit: list, tuple, ndarray, dict
            Binning range for comoving histogram. If dict, use value of key 'r'.
        + nbins: int
            Number of bins for comoving histogram
        + weighted: bool (default=False)
            If True, return weighted histogram. Else return unweighted.
        + normed: bool (default=False)
            If True, normalized by number of galaxies.
        Outputs:
        + r_distr: ndarray
            Comoving distance distribution
        + bins_r: ndarray
            Comoving distance bins """
        bins_range = limit['r'] if isinstance(limit, dict) else limit

        weights = self.catalog[:, 3] if weighted else None
        r_distr, bins_r = numpy.histogram(
            self.catalog[:, 2], bins=nbins, range=bins_range, weights=weights)

        # Normalized by number of galaxies
        if normed:
            r_distr = 1.*r_distr/self.ntotal

        return r_distr, bins_r

    def angular_distr(self, limit, nbins, weighted=False, normed=False):
        """ Calculate angular distribution.
        Inputs:
        + limit: list, tupple, ndarray, dict
            Bins range in order 'dec', 'ra'.
            If dict, use value from keys 'dec' and 'ra'
        + nbins: list, tupple, ndarray, dict
            Number of bins in order 'dec', 'ra'.
            If dict, use value from keys 'dec' and 'ra'
        + weighted: bool (default=False)
            If True, return weighted histogram. Else return unweighted.
        + normed: bool (default=False)
            If True, normalized by number of galaxies.
        Outputs:
        + angular_distr: ndarray
            Angular distribution
        + bins_dec: ndarray
            Angular bins for declination
        + bins_ra: ndarray
            Angular bins for right ascension """

        # Initialize bins range and number of bins
        num_bins = nbins
        bins_range = limit
        if isinstance(bins_range, dict):
            bins_range = (limit['dec'], limit['ra'])
        if isinstance(num_bins, dict):
            num_bins = (nbins['dec'], nbins['ra'])

        # Calculate distribution
        weights = self.catalog[:, 3] if weighted else None
        angular_distr, bins_dec, bins_ra = numpy.histogram2d(
            self.catalog[:, 0], self.catalog[:, 1],
            bins=num_bins, range=bins_range, weights=weights)

        # Normalized by number of galaxies
        if normed:
            angular_distr = 1.*angular_distr/self.catalog.shape[0]

        return angular_distr, bins_dec, bins_ra

    def norm(self):
        """ Return unweighted and weighted normalization factor
        for pairwise separation distribution
        Unweighted equation:
        - norm = 0.5(ntotal^2-ntotal); ntotal is the size of catalog
       Weighted equation:
        - norm = 0.5(sum_w^2-sum_w2); sum_w and sum_w2 are the sum of weights and weights squared
        Outputs:
        + w_norm: float
            Weighted normalization factor
        + uw_norm: float
            Unweighted normalization factor """

        sum_w = numpy.sum(self.catalog[:, 3])
        sum_w2 = numpy.sum(self.catalog[:, 3]**2)

        w_norm = 0.5*(sum_w**2-sum_w2)
        uw_norm = 0.5*(self.ntotal**2-self.ntotal)

        return w_norm, uw_norm

    def _s_distr_thread(self, s_max, nbins_s, catalog, kdtree, job_helper):
        """ Thread function of self.s_distr
        Outputs:
        + s_distr: ndarray
            Return weighted and unweighted pairwise separation distribution."""

        # Initialize weighted (i=0) and unweighted (i=1) distribution
        s_distr = numpy.zeros((2, nbins_s))

        # Get start and end indices
        start, end = job_helper.get_index_range(self.ntotal)

        print("Calculate galaxy pairwise separation from index {} to {}".format(start, end-1))
        for i, point in enumerate(catalog[start:end]):
            if i % 10000 is 0:
                print(i)
            index, s = kdtree.query_radius(point[: 3].reshape(1, -1), r=s_max,
                                           return_distance=True)

            # Fill weighted distribution
            # weight is the product of the weights of two points
            weights = catalog[:, 3][index[0]]*point[3]
            hist, _ = numpy.histogram(s[0], bins=nbins_s, range=(0., s_max), weights=weights)
            s_distr[0] += hist

            # Fill unweighted distribution
            hist, _ = numpy.histogram(s[0], bins=nbins_s, range=(0., s_max))
            s_distr[1] += hist

        # Correction for double counting in the first bin from pairing a galaxy
        # with itself
        s_distr[0][0] -= numpy.sum(catalog[start:end, 3]**2)
        s_distr[1][0] -= end-start

        # Correction for double counting
        s_distr = s_distr/2.

        return s_distr

    def s_distr(self, s_max, nbins_s, job_helper=None, leaf=40):
        """ Calculate pairwise separation distribution.
        Use a modified nearest-neighbors KD-tree search. Metric: Euclidean
        (or Minknowski with p=2).
        Inputs:
        + s_max: float
            Maximum separation
        + nbins_s: int
            Number of bins
        + job_helper: helper.JobHelper (default=None)
            Job manager to handle multiprocess indices. If None, assume one job.
        + leaf: int (default=40)
            Number of points at which KD-tree switches to brute-force. A leaf
            node is guaranteed to satisfy leaf_size <= n_points <= 2*leaf_size,
            except in the case that n_samples < leaf_size.
            More details can be found at sklearn.neightbors.KDTree.
        Outputs:
        + s_distr = ndarray
            Return weighted and unweighted pairwise separation distribution.
        + bins: ndarray
            Binedges of distribution.
        """
        # Convert Celestial coord into Cartesian coord.
        dec, ra, r = self.catalog[:, :3].T
        cart_catalog = numpy.array([r*numpy.cos(dec)*numpy.cos(ra),
                                    r*numpy.cos(dec)*numpy.sin(ra),
                                    r*numpy.sin(dec),
                                    self.catalog[:, 3]]).T

        # Build a KD-tree with Euclidean metric
        kdtree = KDTree(cart_catalog[:, :3], leaf_size=leaf, metric='euclidean')

        # If job_helper is None, assume one job
        if job_helper is None:
            job_helper = JobHelper(1)
            job_helper.set_current_job(0)

        # Calculate the pairwise separation distribution
        s_distr = self._s_distr_thread(s_max, nbins_s, cart_catalog, kdtree, job_helper)

        return s_distr, nbins_s


class DistrCatalog(object):
    """ Class to handle distribution catalog. Distribution catalogs are catalogs
    that does not have the coordinates of each data points, but have the angular and
    redshift (comoving distribution). """

    def __init__(self, reader=None):
        """ Initialize angular and comoving distribution
        Inputs:
        + reader: dict (default=None)
            Dictionary with properties and path to NPZ catalog. If None, only
            initialize attributes """
        self.r_distr = None
        self.angular_distr = None
        self.bins_r = None
        self.bins_ra = None
        self.bins_dec = None
        self.norm_vars = {'ntotal': None, 'sum_w': None, 'sum_w2': None}

        # Read from NPZ catalog file
        if reader is not None:
            print('Import catalog from NPZ: {}'.format(reader['path']))
            with numpy.load(reader['path']) as f:
                self.r_distr = f[reader['r_distr']]
                self.angular_distr = f[reader['angular_distr']]
                self.bins_r = f[reader['bins_r']]
                self.bins_ra = f[reader['bins_ra']]
                self.bins_dec = f[reader['bins_dec']]
                self.norm_vars['ntotal'] = f[reader['ntotal']]
                self.norm_vars['sum_w'] = f[reader['sum_w']]
                self.norm_vars['sum_w2'] = f[reader['sum_w2']]

    def norm(self, data_catalog=None):
        """ Return unweighted and weighted normalization factor
        for pairwise separation distribution
        Unweighted equation:
        - norm = 0.5(ntotal^2-ntotal); ntotal is the size of catalog
        Weighted equation:
        - norm = 0.5(sum_w^2-sum_w2); sum_w and sum_w2 are the sum of weights and weights squared
        Inputs:
        + data_catalog: DataCatalog (default=None)
            If None, calculate the normalization factor for itself (i.e. RR).
            Otherwise, calculate the normalization factor for correlation distribution
            with the input catalog (i.e. DR).
        Outputs:
        + w_norm: float
            Weighted normalization factor
        + uw_norm: float
            Unweighted normalization factor """

        if data_catalog is not None:
            w_norm = numpy.sum(data_catalog.catalog[:, 3])*self.norm_vars['sum_w']
            uw_norm = data_catalog.ntotal*self.norm_vars['ntotal']
            return w_norm, uw_norm

        w_norm = 0.5*(self.norm_vars['sum_w']**2-self.norm_vars['sum_w2'])
        uw_norm = 0.5*(self.norm_vars['ntotal']**2-self.norm_vars['ntotal'])
        return w_norm, uw_norm


    def _theta_distr_thread(self, theta_max, nbins, balltree, job_helper):
        """ Thread function of self.angular_separation.
        Outputs:
        + theta_distr: ndarray
            Angular separation distribution. """

        # Initialize angular separation distribution f(theta)
        theta_distr = numpy.zeros(nbins)

        # Get start and end indices
        start, end = job_helper.get_index_range(self.angular_distr.shape[0])

        print("Construct angular separation from index {} to {}".format(start, end-1))
        for i, point in enumerate(self.angular_distr[start:end]):
            if i % 10000 is 0:
                print(i)
            index, theta = balltree.query_radius(point[:2].reshape(1, -1),
                                                 r=theta_max,
                                                 return_distance=True)
            # weight is the product of the weights of each point
            weights = point[2]*self.angular_distr[:, 2][index[0]]
            hist, _ = numpy.histogram(theta[0], bins=nbins, range=(0., theta_max), weights=weights)
            theta_distr += hist

        # Correction for double counting
        theta_distr = theta_distr/2.

        return theta_distr

    def theta_distr(self, theta_max, nbins, job_helper=None, leaf=40):
        """ Calculate pairwise angular separation distribution.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given angle. Metric = 'haversine'.
        Inputs:
        + theta_max: float
            Maximum angular separation
        + nbins: int
            Number of bins
        + job_helper: helper.JobHelper (default=None)
            Job manager to handle multiprocess indices. If None, assume one job.
        + leaf: int (default=40)
            Number of points at which KD-tree switches to brute-force. A leaf
            node is guaranteed to satisfy leaf_size <= n_points <= 2*leaf_size,
            except in the case that n_samples < leaf_size.
            More details can be found at sklearn.neightbors.BallTree.
        Outputs:
        + theta_distr: ndarray
            Angular separation distribution
        + bins: ndarray
            Binedges of distribution. """

        # Create a BallTree
        balltree = BallTree(self.angular_distr[:, :2], leaf_size=leaf, metric='haversine')

        # If job_helper is None, assume one job
        if job_helper is None:
            job_helper = JobHelper(1)
            job_helper.set_current_job(0)

        # Calculate the pairwise angular separation distribution
        theta_distr = self._theta_distr_thread(theta_max, nbins, balltree, job_helper)
        bins = numpy.linspace(0., theta_max, nbins+1)

        return theta_distr, bins

    def _r_theta_distr_thread(self, theta_max, nbins_theta, data_catalog, balltree,
                              job_helper, mode):
        """ Thread function of self.r_theta_distr
        Outputs:
        + r_theta_distr: ndarray """

        # Initialize comoving-angular distribution
        nbins_r = self.bins_r.size-1
        limit = ((0., theta_max), (self.bins_r.min(), self.bins_r.max()))
        r_theta_distr = numpy.zeros((2, nbins_theta, nbins_r))

        # Get start and end indices
        if mode == "angular_tree":
            start, end = job_helper.get_index_range(data_catalog.ntotal)
        elif mode == "data_tree":
            start, end = job_helper.get_index_range(self.angular_distr.shape[0])

        print("Construct angular-comoving from index {} to {}".format(start, end-1))
        if mode == "angular_tree":
            for i, point in enumerate(data_catalog[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = balltree.query_radius(
                    point[:2].reshape(1, -1), r=theta_max, return_distance=True)
                r = numpy.repeat(point[2], index[0].size)

                # Fill unweighted histogram
                weights = self.angular_distr[:, 2][index[0]]
                hist, _, _ = numpy.histogram2d(
                    theta[0], r, bins=(nbins_theta, nbins_r), range=limit, weights=weights)
                r_theta_distr[1] += hist

                # Fill weighted histogram
                # weight is the product of the weight of the data point and the weight
                # of the angular point.
                weights = weights*point[3]
                hist, _, _ = numpy.histogram2d(
                    theta[0], r, bins=(nbins_theta, nbins_r), range=limit, weights=weights)
                r_theta_distr[0] += hist

        elif mode == "data_tree":
            for i, point in enumerate(self.angular_distr[start:end]):
                if i % 10000 is 0:
                    print(i)
                index, theta = balltree.query_radius(
                    point[:2].reshape(1, -1), r=theta_max, return_distance=True)
                r = data_catalog[:, 2][index[0]]

                # Fill weighted histogram
                # weight is the product of the weight of the data point and the weight of
                # the angular point
                weights = point[2]*data_catalog[:, 3][index[0]]
                hist, _, _ = numpy.histogram2d(
                    theta[0], r, bins=(nbins_theta, nbins_r), range=limit, weights=weights)
                r_theta_distr[0] += hist

                # Fill unweighted histogram
                # weight is the weight of the angular point
                hist, _, _ = numpy.histogram2d(
                    theta[0], r, bins=(nbins_theta, nbins_r), range=limit)
                r_theta_distr[1] += hist*point[2]

        return r_theta_distr

    def r_theta_distr(self, data, theta_max, nbins_theta, job_helper=None, mode=None, leaf=40):
        """ Calculate comoving-angular distribution.
        Use a modified nearest-neighbors BallTree algorithm to calculate
        angular distance up to a given angle. Metric = 'haversine'.
        NOTE: assume self.bins_r is uniformed
        Inputs:
        + data: DataCatalog
            Data catalog object to pair with
        + theta_max: float
            Maximum angular separation
        + nbins_theta: int
            Number of angular bins
        + job_helper: helper.JobHelper (default=None)
            Job manager to handle multiprocess indices. If None, assume one job.
        + leaf: int (default=40)
            Number of points at which KD-tree switches to brute-force. A leaf
            node is guaranteed to satisfy leaf_size <= n_points <= 2*leaf_size,
            except in the case that n_samples < leaf_size.
            More details can be found at sklearn.neightbors.BallTree.
        Outputs:
        + r_theta_distr: ndarray
            Angular-comoving distribution
        + bins: ndarray
            Binedges of angular-comoving distribution (bins_theta, bins_r). """

        # Runtime is O(N*log(M)); M = size of tree; N = size of catalog
        # Create tree from catalog with smaller size
        if mode is None:
            if self.angular_distr.shape[0] >= data.ntotal:
                mode = "angular_tree"
                balltree = BallTree(self.angular_distr[:, :2], leaf_size=leaf, metric='haversine')
            else:
                mode = "data_tree"
                balltree = BallTree(data.catalog[:, :2], leaf_size=leaf, metric='haversine')
        elif mode == "data_tree":
            balltree = BallTree(data.catalog[:, :2], leaf_size=leaf, metric='haversine')
        elif mode == "angular_tree":
            balltree = BallTree(self.angular_distr[:, :2], leaf_size=leaf, metric='haversine')

        # If job_helper is None, assume one job
        if job_helper is None:
            job_helper = JobHelper(1)
            job_helper.set_current_job(0)

        r_theta_distr = self._r_theta_distr_thread(
            theta_max, nbins_theta, data.catalog, balltree, job_helper, mode)
        bins_theta = numpy.linspace(0., theta_max, nbins_theta)

        return r_theta_distr, bins_theta, self.bins_r
