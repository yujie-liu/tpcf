''' Modules to construct two-point correlation function using KD-Tree
    Estimators used: (DD-2*DR+RR)/RR,
    where D and R are data and randoms catalogs respectively
    '''

import numpy
from sklearn.neighbors import KDTree, BallTree


def separation(r1, r2, theta):
    return numpy.sqrt(r1**2+r2**2-2*r1*r2*numpy.cos(theta))


def import_data(fname):
    ''' Import data into 2-d numpy array
        Format: declination, right ascension, comoving distance, weight
        '''
    catalog = numpy.genfromtxt(fname, skip_header=1)
    col_temp = numpy.copy(catalog[:, 0])
    catalog[:, 0] = numpy.deg2rad(catalog[:, 1])
    catalog[:, 1] = numpy.deg2rad(col_temp)
    return catalog


def data_data_histogram(data_cat, distance_max, num_bins):
    ''' Construct data-data separation DD(s) using data catalog.
        Parameters:
        + data_cat:  2-d numpy array: N rows, 4 columns
            Data catalog with N points used to compute DD(s)
            Format: declination, right ascension, comoving distance, weight
        + distance_max: float
            Maximum separations to be considered into DD(s)
        + num_bins: int
            Number of bins for DD(s)
        Outputs:
        + data_data: numpy array length num_bins
            Values of DD(s)
        + bins: numpy array length num_bins+1
            Binedges of DD(s)
        '''
    # First convert into Cartesian coordinates
    # Conversion equation is
    # x = r*cos(dec)*cos(ra)
    # y = r*cos(dec)*sin(ra)
    # z = r*sin(dec)
    tempx = data_cat[:, 2]*numpy.cos(data_cat[:, 0])*numpy.cos(data_cat[:, 1])
    tempy = data_cat[:, 2]*numpy.cos(data_cat[:, 0])*numpy.sin(data_cat[:, 1])
    tempz = data_cat[:, 2]*numpy.sin(data_cat[:, 0])
    cart_cat = numpy.array([tempx, tempy, tempz, data_cat[:, 3]]).T

    # Create KD-tree and compute DD(s)
    cart_tree = KDTree(cart_cat[:, : 3], leaf_size=1, metric='euclidean')
    data_data = numpy.zeros(num_bins)  # DD(s)
    print("Start constructing DD(s)...")
    for i, point in enumerate(cart_cat):
        if i % 10000 is 0:
            print(i)
        _, dist = cart_tree.query_radius(point[: 3].reshape(1, -1),
                                         r=distance_max, return_distance=True)
        temphist, bins_s = numpy.histogram(dist[0], num_bins,
                                           range=(0, distance_max))
        data_data += temphist
    data_data = data_data/2  # account for double counting
    return data_data, bins_s


def histogram2points(hist, bins_x, bins_y):
    ''' Convert 2-d histogram into data points with weight.
        Take bincenters as data points. Zero bins are ignored.
        Parameters:
        + hist:  2-d numpy array
            The values of of the histogram.
            Dimension must be [Nx, Ny], the number of bins in X and Y
        + bins_x:  numpy array
            Binedges in X, length must be Nx+1
        + bins_y:  numpy array
            Binedges in Y, length must be Nx+1
        Outputs:
        + data: 2-d numpy array
            Array of data points with weight. Format [X,Y,Weight]
        '''
    center_x = 0.5*(bins_x[:-1]+bins_x[1:])
    center_y = 0.5*(bins_y[:-1]+bins_y[1:])
    grid_x, grid_y = numpy.meshgrid(center_x, center_y)
    grid_x = grid_x.flatten()
    grid_y = grid_y.flatten()
    hist = hist.T.flatten()

    # Create data array with non-zero bins
    data = numpy.array([grid_x, grid_y, hist]).T
    data = data[hist > 0]
    return data


def angular_separation_histogram(angular_distr, angle_max, num_bins):
    ''' Construct f(theta) angular separation histogram
        Parameters:
        + angular_distr: numpy array
            Angular distribution of random data points
            Data must be broken into one dimensional array of points
            Format: [DEC, RA, Weight]
        + angle_max: float
            Maximum angular separation (in rad) to be considered into f(theta)
        + num_bins: int
            Number of bins for f(theta)
        Outputs:
        + theta_hist: numpy array length num_bins
            Values of f(theta)
        + bins: numpy array length num_bins+1
            Binedges of f(theta)
        '''
    # create a BallTree and compute f(theta)
    arc_tree = BallTree(angular_distr[:, :2], leaf_size=1,
                        metric='haversine')
    theta_hist = numpy.zeros(num_bins)
    print("Start constructing f(theta)")
    for i, point in enumerate(angular_distr):
        if i % 10000 is 0:
            print(i)
        index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                             r=angle_max,
                                             return_distance=True)
        tempw = point[2]*angular_distr[:, 2][index[0]]
        temphist, bins_theta = numpy.histogram(theta[0], num_bins,
                                               range=(0., angle_max),
                                               weights=tempw)
        theta_hist += temphist
    theta_hist = theta_hist/2.  # account for double counting
    return theta_hist, bins_theta


def get_rand_rand(theta_hist, r_hist, distance_max, num_bins):
    ''' Construct data-data separation DD(s) using data catalog.
        Parameters:
        + theta_hist:  list, tuples
            Values and binedges for angular separation histogram
            Format: values = theta_hist[0], bins = theta_hist[1]
        + r_hist:  list, tuples
            Values and binedges of radial histogram
            Format: values = r_hist[0], bins = r_hist[1]
        + distance_max: float
            Maximum separations to be considered into RR(s)
        + num_bins: int
            Number of bins for RR(s)
        Outputs:
        + rand_rand: numpy array length num_bins
            Values of RR(s)
        + bins: numpy array length num_bins+1
            Binedges of RR(s)
        '''
    # Compute weight matrix
    weight = numpy.zeros((theta_hist[0].size, r_hist[0].size))
    for i in range(theta_hist[0].size):
        for j in range(r_hist[0].size):
            weight[i, j] = theta_hist[1][i]*r_hist[1][j]
    temp = histogram2points(weight, theta_hist[1], r_hist[1])

    # Construct RR(s)
    rand_rand =  numpy.zeros(num_bins)
    center_r = 0.5*(r_hist[1][:-1]+r_hist[1][1:])
    for i, tempr in enumerate(center_r[r_hist[0]!=0]):
        if i % 100 is 0:
            print(i)
        temps = separation(rtemp, temp[:, 0], temp[:, 1])
        temphist, bins_s = numpy.histogram(temps, num_bins,
                                           range=(0, distance_max))


        bins_s = np.linspace(0., s_max, nbin_s+1)
for i, rtemp in enumerate(r_centers[radial_hist_uw!=0]):
    if i%100 is 0:
        print i
    s = separation(rtemp, c[:,0], c[:,1])
    temp_hist, _ = np.histogram(s, bins_s, weights = radial_hist_uw[i]*c[:,2])
    rr += temp_hist

def main():
    ''' For testing purpose '''
    cat_dir = "/home/chris/project/bao/correlation/catalog/"
    # data_fname = cat_dir+"boss/galaxies_DR9_CMASS_North_DC.dat"
    rand_fname = cat_dir+"boss/randoms_DR9_CMASS_North_DC.dat"
    # data_cat = import_data(data_fname)
    rand_cat = import_data(rand_fname)
    angular = numpy.histogram2d(rand_cat[:, 0], rand_cat[:, 1], (240, 620))
    temp = histogram2points(*angular)
    angular_separation_histogram(temp, numpy.deg2rad(10.), 100)


if __name__ == "__main__":
    main()
