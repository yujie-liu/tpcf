""" Module to handle galaxy survey catalogs """

# Python modules
import numpy
from astropy.table import Table
from sklearn.neighbors import BallTree, KDTree

# User-defined modules
import lib.general as general

class GalaxyCatalog(object):
    """ Class to handle galaxy catalogs. """

    def __init__(self, reader, limit):
        """ Initialize galaxy catalog.
        Inputs:
        + reader: dict
            Dictionary with path and properties of catalog file.
            Keys: path, dec, ra, z. Values format: str
        + limit: dict
            Dictionary with boundaries (inclusive) of catalog. If None, use full catalog.
            Keys: dec, ra, z. Values format (min, max) """
        self.ntotal = 0
        self.catalog = None
        self.set_catalog(reader)
        if limit is not None:
            self.set_limit(limit)

    def set_catalog(self, reader):
        """ Read in galaxy catalog from FITS file. Convert redshift to
        comoving distance.
        Inputs:
        + reader: dict
            Dictionary with path and properties of catalog file.
            Keys: path, dec, ra, z. Values format: str """

        # Read in configuration file
        print("Import catalog from: {}".format(reader['path']))

        # Read in table
        table = Table.read(reader['path'])
        dec = numpy.deg2rad(table[reader['dec']].data)
        ra = numpy.deg2rad(table[reader['ra']].data)
        z = table[reader['z']].data
        try:
            w = table[reader['weight']]
        except KeyError:
            w_fkp = table[reader['weight_fkp']].data
            w_noz = table[reader['weight_noz']].data
            w_cp = table[reader['weight_cp']].data
            w_sdc = table[reader['weight_sdc']].data
            w = w_sdc*w_fkp*(w_noz+w_cp-1)

        self.catalog = numpy.array([dec, ra, z, w]).T
        self.ntotal = self.catalog.shape[0]

    def get_catalog(self, cosmo=None):
        """ Get catalog. If cosmology is given, convert redshift to comoving distance
        Inputs:
        + cosmo: cosmology.Cosmology (default=None)
            Cosmology model to convert comoving to redshift.
        Outputs:
        + catalog: ndarray
            Data catalog. """
        # If cosmology is given, convert redshift to comoving distance
        catalog = numpy.copy(self.catalog)
        if cosmo is not None:
            catalog[:, 2] = cosmo.z2r(catalog[:, 2])
        return catalog

    def set_limit(self, limit):
        """ Set boundaries of catalog
        Inputs:
        + limit: dict
            Dictionary with boundaries (inclusive) of catalog
            Keys: dec, ra, z. Values format: (min, max) """

        # Read limit and perform array slicing
        dec, ra, z = self.catalog[:, :3].T
        index = ((limit['dec'][0] <= dec) & (dec <= limit['dec'][1])
                 & (limit['ra'][0] <= ra) & (ra <= limit['ra'][1])
                 & (limit['z'][0] <= z) & (z <= limit['z'][1]))

        self.catalog = self.catalog[index]
        self.ntotal = self.catalog.shape[0]

    def to_rand(self, limit, nbins, cosmo=None):
        """ Convert into RandomCatalog and return. If given cosmology, convert
        redshift distribution to comoving distribution.
        Inputs:
        + limit: list, tuple, ndarray, dict
            Bins range in order 'dec', 'ra', 'z'.
            If dict, use values from keys 'dec', 'ra', 'z'
        + nbins: list, tuple, ndarray, dict
            Number of bins in order 'dec', 'ra', 'z'.
            If dict, use values from keys 'dec', 'ra', 'z'
        + cosmo: cosmology.Cosmology (default=None)
            Cosmology model to convert comoving to redshift.
        Outputs:
        + rand: RandomCatalog
            Distribution catalog """

        # Initialize bins range and number of bins
        num_bins = nbins
        bins_range = limit
        if isinstance(bins_range, dict):
            bins_range = (limit['dec'], limit['ra'], limit['z'])
        if isinstance(num_bins, dict):
            num_bins = (nbins['dec'], nbins['ra'], nbins['z'])

        # Calculate comoving distribution
        rz_distr_w, bins_rz = self.redshift_distr(
            bins_range[2], num_bins[2], cosmo=cosmo, weighted=True, normed=True)
        rz_distr_uw, _ = self.redshift_distr(
            bins_range[2], num_bins[2], cosmo=cosmo, weighted=False, normed=True)

        # Calculate angular distribution as a set of weighted data points
        angular_distr, bins_dec, bins_ra = self.angular_distr(
            bins_range[:2], num_bins[:2], weighted=False, normed=False)

        # Set up DistrCatalog attributes
        rand = RandomCatalog()
        rand.rz_distr = numpy.array([rz_distr_w, rz_distr_uw])
        rand.angular_distr = general.hist2point(angular_distr, bins_dec, bins_ra)
        rand.bins_rz = bins_rz
        rand.bins_dec = bins_dec
        rand.bins_ra = bins_ra
        rand.norm_vars['ntotal'] = self.ntotal
        rand.norm_vars['sum_w'] = numpy.sum(self.catalog[:, 3])
        rand.norm_vars['sum_w2'] = numpy.sum(self.catalog[:, 3]**2)

        return rand

    def redshift_distr(self, limit, nbins, cosmo=None, weighted=False, normed=False):
        """ Calculate redshift distribution. If cosmology is given, convert redshift
        to comoving distribution.
        Inputs:
        + limit: list, tuple, ndarray, dict
            Binning range for redshift histogram. If dict, use value of key 'z'.
        + nbins: int
            Number of bins for comoving histogram.
        + cosmo: cosmology.Cosmology (default=None)
            Cosmology model to convert comoving to redshift.
        + weighted: bool (default=False)
            If True, return weighted histogram. Else return unweighted histogram.
        + normed: bool (default=False)
            If True, normalized by number of galaxies.
        Outputs:
        + z_distr: ndarray
            Redshift (comoving) distribution.
        + bins_z: ndarray
            Redshift (comoving) binedges. """
        bins_range = limit['z'] if isinstance(limit, dict) else limit
        weights = self.catalog[:, 3] if weighted else None
        z = self.catalog[:, 2]

        # If given cosmological model, calculate comoving distribution
        if cosmo is not None:
            bins_range = cosmo.z2r(bins_range)
            z = cosmo.z2r(z)

        z_distr, bins_z = numpy.histogram(z, bins=nbins, range=bins_range, weights=weights)

        # Normalized by number of galaxies
        if normed:
            z_distr = z_distr/self.ntotal

        return z_distr, bins_z

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

    def build_balltree(self, metric, cosmo=None, return_catalog=False, leaf=40):
        """ Build a balltree from catalog. If metric is 'euclidean', cosmology is required.
        Inputs:
        + metric: str
            Metric must be either 'haversine' or 'euclidean'.
            If metric is 'haversine', build a balltree from DEC and RA coordinates of galaxies.
            If metric is 'euclidean', build a 3-dimensional kd-tree
        + return_catalog: bool (default=False)
            If True, return the catalog in balltree
        + cosmo: cosmology.Cosmology (default=None)
            Cosmology model to convert comoving to redshift.
        + leaf: int (default=40)
            Number of points at which KD-tree switches to brute-force. A leaf
            node is guaranteed to satisfy leaf_size <= n_points <= 2*leaf_size,
            except in the case that n_samples < leaf_size.
            More details can be found at sklearn.neightbors.BallTree. """
        if metric == 'euclidean':
            if cosmo is None:
                raise TypeError('Cosmology must be given if metric is "euclidean".')
            # Convert Celestial coordinate into Cartesian coordinate
            dec, ra, z = self.catalog[:, :3].T
            r = cosmo.z2r(z)
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


class RandomCatalog(object):
    """ Class to handle random catalog. Random catalog has the angular and redshift
    (comoving) distribution, but not the coordinates of each galaxy. """

    def __init__(self):
        """ Initialize angular, redshift (comoving) distribution, and normalization variables """
        self.rz_distr = None
        self.angular_distr = None
        self.bins_rz = None
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
