""" Module for multiprocessing code on calculating DD(s)-galaxies separation
distribution """

import sys
import numpy
from sklearn.neighbors import KDTree
from astropy.io import fits
from cosmology import Cosmology


DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi


def import_fits(fname_key, fits_reader, cosmo):
    """ Import data into 2-d array
    Inputs:
    + fname_key: string
        Key for data filename in reader.
    + fits_readers: dict
        Must have attributes: "INDEX"=index of headers. RA", "DEC", "Z",
        "WEIGHT"=corresponded variable names in header.
    + cosmol: cosmology.Cosmology
        Cosmological parameters to convert redshift to comoving distance.
    Outputs:
    + catalog: ndarray or tuple of ndarrays
        Return catalog format in each row [DEC, RA, R, WEIGHT].
    """
    header_index = int(fits_reader["INDEX"])
    hdulist = fits.open(fits_reader[fname_key])
    tbdata = hdulist[header_index].data
    temp_dec = DEG2RAD*tbdata[fits_reader["DEC"]]
    temp_ra = DEG2RAD*tbdata[fits_reader["RA"]]
    temp_r = cosmo.z2r(tbdata[fits_reader["Z"]])
    try:
        temp_weight_fkp = tbdata[fits_reader["WEIGHT_FKP"]]
        temp_weight_noz = tbdata[fits_reader["WEIGHT_NOZ"]]
        temp_weight_cp = tbdata[fits_reader["WEIGHT_CP"]]
        temp_weight_sdc = tbdata[fits_reader["WEIGHT_SDC"]]
        temp_weight = (temp_weight_sdc*temp_weight_fkp*
                       (temp_weight_noz + temp_weight_cp -1))
    except:
        temp_weight = tbdata[fits_reader["WEIGHT"]]
    catalog = numpy.array([temp_dec, temp_ra, temp_r, temp_weight]).T
    hdulist.close()
    return catalog


def data_data_thread(data_cat, data_tree, bins, start, end):
    """ Thread function for calculating separation distribution DD(s) between
    pairs of galaxies. Use a modified nearest-neighbors KDTree algorithm to
    calculate distance up to a given radius defined in config file.
    Metric: Euclidean (or Minkowski with p=2).
    Inputs:
    + data_cat: array
        Data catalog in format [X,Y,Z].
    + data_tree: kd-tree
        KD-tree fill with data in data catalog.
    + bins: array
        Binedges of DD(s).
    + start: int
        Starting index of galaxies in data catalog: index_start = start.
    + end: int
        Ending index of galaxies in data catalog: index_end = end-1.
    + leaf: int (default=40)
        Number of points at which to switch to brute-force. For a specified
        leaf_size, a leaf node is guaranteed to satisfy
        leaf_size <= n_points <= 2*leaf_size, except in the case that
        n_samples < leaf_size. More details in sklearn.neighbors.KDTree.
    Outputs:
    + data_data: ndarrays or tupple of ndarrays
        Return values of weighted and unweighted DD(s) respectively.
    """

    # Define weighted and unweighted DD(s) as two one-dimensional
    # histograms respectively.
    nbins_s = bins.size-1
    s_max = bins.max()
    data_data = numpy.zeros((2, nbins_s))

    print("Calculate DD(s) from {} to {}".format(start, end))
    for i, point in enumerate(data_cat[start:end]):
        if i % 10000 is 0:
            print(i)
        index, dist = data_tree.query_radius(point[: 3].reshape(1, -1),
                                             r=s_max,
                                             return_distance=True)
        # Fill weighted histogram
        temp_weight = data_cat[:, 3][index[0]]*data_cat[i, 3]
        temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                       range=(0., s_max),
                                       weights=temp_weight)
        data_data[0] += temp_hist
        # Fill unweighted histogram
        temp_hist, _ = numpy.histogram(dist[0], bins=nbins_s,
                                       range=(0., s_max))
        data_data[1] += temp_hist

    return data_data


def main():
    """ Main """

    # define cosmological parameters
    # H0=Hubble constant at z=0
    # Om0=Omega matter at z=0
    # Ob0=Omega baryonic matter at z=0
    # Ode0=Omega dark energy at z=0
    cosmo = Cosmology()
    cosmo.set_model(H0=100, Om0=0.307, Ob0=0.0486, Ode0=0.693)

    # import data
    fname = "/home/chris/project/bao/correlation/catalog/boss/galaxies_DR9_CMASS_Combined.fits"
    fits_reader = {"INDEX":1, "RA":"ra", "DEC":"dec", "Z":"z",
                   "WEIGHT_FKP":"weight_fkp", "WEIGHT_NOZ":"weight_noz",
                   "WEIGHT_CP":"weight_cp", "WEIGHT_SDC":"weight_sdc",
                   "data_fname":fname}
    data_cat = import_fits("data_fname", fits_reader, cosmo)

    # define histogram and bins
    bins = numpy.linspace(0., 200., 51)
    data_data = numpy.zeros((2, 50))

    # Convert Celestial coordinate into Cartesian coordinate
    # Conversion equation is:
    # x = r*cos(dec)*cos(ra)
    # y = r*cos(dec)*sin(ra)
    # z = r*sin(dec
    temp_dec = data_cat[:, 0]
    temp_ra = data_cat[:, 1]
    temp_r = data_cat[:,2]
    temp_weight = data_cat[:, 3]
    cart_x = temp_r*numpy.cos(temp_dec)*numpy.cos(temp_ra)
    cart_y = temp_r*numpy.cos(temp_dec)*numpy.sin(temp_ra)
    cart_z = temp_r*numpy.sin(temp_dec)
    cart_cat = numpy.array([cart_x, cart_y, cart_z, temp_weight]).T

    # Calculate based on command line input
    no_job = int(sys.argv[1])
    total_job = int(sys.argv[2])
    prefix = sys.argv[3]
    job_index = numpy.floor(numpy.linspace(0, cart_cat.shape[0]-1, total_job+1))
    job_index = job_index.astype(int)

    # Create KD-tree and compute DD(s) using modified nearest-neighbors
    # algorithm to calculate angular distance up to a given radius.
    cart_tree = KDTree(cart_cat[:, :3], metric='euclidean')
    data_data = data_data_thread(cart_cat, cart_tree, bins,
                                 job_index[no_job], job_index[no_job+1])

    numpy.save("out/{}_{}".format(prefix, no_job), data_data)


if __name__ == "__main__":
    main()
