""" Module to handle galaxy survey catalogs """

# Python modules
import numpy
from astropy.table import Table
from sklearn.neighbors import BallTree, KDTree

# User-defined modules
import lib.general as general

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

    def build_balltree(self, metric, return_catalog=False, leaf=40):
        """ Build a balltree from catalog
        Inputs:
        + metric: str
            Metric must be either 'haversine' or 'euclidean'.
            If metric is 'haversine', build a balltree from DEC and RA coordinates of galaxies.
            If metric is 'euclidean', build a 3-dimensional kd-tree
        + return_catalog: bool (default=False)
            If True, return the catalog in balltree
        + leaf: int (default=40)
            Number of points at which KD-tree switches to brute-force. A leaf
            node is guaranteed to satisfy leaf_size <= n_points <= 2*leaf_size,
            except in the case that n_samples < leaf_size.
            More details can be found at sklearn.neightbors.BallTree. """
        if metric == 'euclidean':
            # Convert Celestial coordinate into Cartesian coordinate
            dec, ra, r = self.catalog[:, :3].T
            catalog = numpy.array([r*numpy.cos(dec)*numpy.cos(ra),
                                   r*numpy.cos(dec)*numpy.sin(ra),
                                   r*numpy.sin(dec),
                                   self.catalog[:, 3]]).T
            tree = KDTree(catalog[:, :3], leaf_size=leaf, metric='euclidean')
        elif metric == 'haversine':
            catalog = self.catalog[:, :-2]
            tree = BallTree(catalog, leaf_size=leaf, metric=metric)
        else:
            raise ValueError('Metric must be either "haversine" or "euclidean".')

        print("Creating BallTree with metric {}".format(metric))

        # Return KD-tree and the catalog
        if return_catalog:
            return tree, catalog
        return tree


class DistrCatalog(object):
    """ Class to handle distribution catalog. Distribution catalogs are catalogs
    that does not have the coordinates of each data points, but have the angular and
    redshift (comoving distribution). """

    def __init__(self):
        """ Initialize angular and comoving distribution """
        self.r_distr = None
        self.angular_distr = None
        self.bins_r = None
        self.bins_ra = None
        self.bins_dec = None
        self.norm_vars = {'ntotal': None, 'sum_w': None, 'sum_w2': None}

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

    def build_balltree(self, return_catalog=False, leaf=40):
        """ Build a balltree using DEC and RA from angular distributions.
        Metric: haversine.
        + return_catalog: bool (default=False)
            If True, return the angular distribution catalog.
        + leaf: int (default=40)
            Number of points at which KD-tree switches to brute-force. A leaf
            node is guaranteed to satisfy leaf_size <= n_points <= 2*leaf_size,
            except in the case that n_samples < leaf_size.
            More details can be found at sklearn.neightbors.BallTree."""
        print("Creating BallTree")
        balltree = BallTree(self.angular_distr[:, :2], leaf_size=leaf, metric='haversine')
        if return_catalog:
            return balltree, self.angular_distr
        return balltree
