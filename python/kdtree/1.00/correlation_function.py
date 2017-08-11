''' Modules to construct two-point correlation function using KD-Tree
Estimators used: (DD-2*DR+RR)/RR,
where D and R are data and randoms catalogs respectively'''
__version__ = 1.00

import configparser
import numpy
import matplotlib.pyplot
from sklearn.neighbors import KDTree, BallTree

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi


def import_data(fname):
    ''' Import data into 2-d numpy array
        Format: declination, right ascension, comoving distance, weight
        '''
    catalog = numpy.genfromtxt(fname, skip_header=1)
    col_temp = numpy.copy(catalog[:, 0])
    catalog[:, 0] = DEG2RAD*catalog[:, 1]
    catalog[:, 1] = DEG2RAD*col_temp
    return catalog


def histogram2points(hist, bins_x, bins_y, exclude_zeros=True):
    ''' Convert 2-d histogram into data points with weight.
        Take bincenters as data points.
        Parameters:
        + hist:  2-d numpy array
            The values of of the histogram.
            Dimension must be [Nx, Ny], the number of bins in X and Y
        + bins_x:  numpy array
            Binedges in X, length must be Nx+1
        + bins_y:  numpy array
            Binedges in Y, length must be Nx+1
        + exclude_zeros: bool (default = True)
            Excluded zeros bins
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
    if exclude_zeros:
        return data[hist > 0]
    return data


def get_distance(radius1, radius2, theta):
    ''' Given two points at radius1 and radius2 with angular separation
    theta (in rad), calulate distance between points'''
    return numpy.sqrt(radius1**2+radius2**2-2*radius1*radius2*numpy.cos(theta))


def get_binnings(x_min, x_max, binwidth):
    ''' Return the binnings given min, max and width '''
    nbins = int(numpy.ceil((x_max-x_min)/binwidth))
    return numpy.linspace(x_min, x_max, nbins)


class CorrelationFunction():
    ''' Class to construct two-point correlation function
    Instance Variables:
      + data_cat: galaxies catalog in format [DEC, RA, RADIUS, WEIGHT]
      + rand_cat: randoms catalog in format [DEC, RA, RADIUS, WEIGHT]
    Functions:
        + angular_distance(): return angular distance distribution f(theta)
        + radial_angular_distance(): return radial angular
                                     distribution g(theta, r)
        + rand_rand(): return separation distribution between randoms RR(s)
        + data_data(): return separation distribution between galaxies DD(s)
        + data_rand(): return separation distribution cross samples DR(s)
        + correlation(rand_rand, data_rand, data_data):
            - compute two-point correlation function for any RR(s), DR(s), DD(s)
    '''
    def __init__(self, config_fname):
        ''' Class contructor parses configuration files '''
        # Parse parameters from configuration files
        config = configparser.ConfigParser()
        config.read(config_fname)

        # Import random and data catalogs
        self.data_cat = import_data(config['FILE'].get('data_filename'))
        self.rand_cat = import_data(config['FILE'].get('random_filename'))

        # Setting up binning
        binnings = config['BINNING']
        binwidth_ra = DEG2RAD*float(binnings.get('binwidth_ra'))
        binwidth_dec = DEG2RAD*float(binnings.get('binwidth_dec'))
        binwidth_theta = DEG2RAD*float(binnings.get('binwidth_theta'))
        binwidth_r = float(binnings.get('binwidth_r'))
        binwidth_s = float(binnings.get('binwidth_s'))

        ra_min = min(self.rand_cat[:, 1].min(), self.data_cat[:, 1].min())
        ra_max = max(self.rand_cat[:, 1].max(), self.data_cat[:, 1].max())
        dec_min = min(self.rand_cat[:, 0].min(), self.data_cat[:, 0].min())
        dec_max = max(self.rand_cat[:, 0].max(), self.data_cat[:, 0].max())
        r_min = min(self.rand_cat[:, 2].min(), self.data_cat[:, 2].min())
        r_max = max(self.rand_cat[:, 2].max(), self.data_cat[:, 2].max())
        s_max = float(binnings.get('s_max'))
        theta_max = numpy.arccos(1.-0.5*s_max**2/r_min**2)

        self.__bins_ra = get_binnings(ra_min, ra_max, binwidth_ra)
        self.__bins_dec = get_binnings(dec_min, dec_max, binwidth_dec)
        self.__bins_r = get_binnings(r_min, r_max, binwidth_r)
        self.__bins_s = get_binnings(0., s_max, binwidth_s)
        self.__bins_theta = get_binnings(0., theta_max, binwidth_theta)

    # Construct f(theta)
    def angular_distance(self):
        ''' Construct f(theta) angular separation histogram
        Outputs:
        + theta_hist: numpy array
            Values of f(theta)
        + bins: numpy array
            Binedges of f(theta)
            '''
        # Compute R(ra, dec)
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular = histogram2points(*angular_hist)

        # create a BallTree and compute f(theta) up to self.__theta_max
        arc_tree = BallTree(angular[:, :2], leaf_size=40, metric='haversine')
        theta_hist = numpy.zeros(self.__bins_theta.size-1)

        print("Start constructing f(theta)")
        for i, point in enumerate(angular):
            if i % 10000 is 0:
                print(i)
            index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                 r=self.__bins_theta.max(),
                                                 return_distance=True)
            temp_weight = point[2]*angular[:, 2][index[0]]
            temp_hist, _ = numpy.histogram(theta[0], self.__bins_theta,
                                           weights=temp_weight)
            theta_hist += temp_hist
        # correction for double counting
        theta_hist = theta_hist/2.

        return theta_hist, self.__bins_theta

    # Construct g(theta, r)
    def radial_angular_distance(self):
        ''' Construct g(theta, r) radial angular separation histogram
        X axis is theta, Y axis is r
        Outputs:
        + r_theta_hist: 2d numpy array
            Dimension: (self.__nbins_theta, self.__nbins_r)
            Values of g(theta, r)
        + bins_theta: numpy array
            Binedges X of g(theta, r)
        + bins_r: numpy array
            Binedges Y of g(theta, r)
            '''
        # Compute R(ra, dec)
        angular_hist = numpy.histogram2d(self.rand_cat[:, 0],
                                         self.rand_cat[:, 1],
                                         bins=(self.__bins_dec, self.__bins_ra))
        angular = histogram2points(*angular_hist)

        # create a BallTree and compute g(theta, r) up to self.__theta_max
        arc_tree = BallTree(angular[:, :2], leaf_size=40, metric='haversine')

        r_theta_hist = numpy.zeros((self.__bins_theta.size-1,
                                    self.__bins_r.size-1))
        for i, point in enumerate(self.data_cat):
            if i % 10000 is 0:
                print(i)
            index, theta = arc_tree.query_radius(point[:2].reshape(1, -1),
                                                 r=self.__bins_theta.max(),
                                                 return_distance=True)
            temp_r = numpy.ones_like(theta[0])*point[2]
            temp_weight = angular[:, 2][index[0]]
            temp_hist = numpy.histogram2d(theta[0], temp_r,
                                          bins=(self.__bins_theta, self.__bins_r),
                                          weights=temp_weight)[0]
            r_theta_hist += temp_hist

        return r_theta_hist, self.__bins_theta, self.__bins_r

    # Construct DD(s)
    def data_data(self):
        ''' Construct data-data separation DD(s) using data catalog.
        Outputs:
        + data_data: numpy array length self.__nbins_s
            Values of DD(s)
        + bins: numpy array length self.__nbins_s+1
            Binedges of DD(s)
            '''
        # First convert into Cartesian coordinates
        # Conversion equation is
        # x = r*cos(dec)*cos(ra)
        # y = r*cos(dec)*sin(ra)
        # z = r*sin(dec)
        temp_ra = self.data_cat[:, 1]
        temp_dec = self.data_cat[:, 0]
        temp_r = self.data_cat[:, 2]
        temp_weight = self.data_cat[:, 3]
        temp_x = temp_r*numpy.cos(temp_dec)*numpy.cos(temp_ra)
        temp_y = temp_r*numpy.cos(temp_dec)*numpy.sin(temp_ra)
        temp_z = temp_r*numpy.sin(temp_dec)
        cart_cat = numpy.array([temp_x, temp_y, temp_z, temp_weight]).T

        # Create KD-tree and compute DD(s)
        cart_tree = KDTree(cart_cat[:, :3], leaf_size=40, metric='euclidean')
        data_data = numpy.zeros(self.__bins_s.size-1)  # DD(s)
        print("Start constructing DD(s)...")
        for i, point in enumerate(cart_cat):
            if i % 10000 is 0:
                print(i)
            _, dist = cart_tree.query_radius(point[: 3].reshape(1, -1),
                                             r=self.__bins_s.max(),
                                             return_distance=True)
            temp_hist, _ = numpy.histogram(dist[0], self.__bins_s)
            data_data += temp_hist

        # correction for double counting
        data_data[0] -= self.data_cat.shape[0]
        data_data = data_data/2.

        return data_data, self.__bins_s

    # Construct RR(s)
    def rand_rand(self):
        ''' Construct random-random separation RR(s) using random catalog.
        Outputs:
        + data_data: numpy array length self.__nbins_s
            Values of RR(s)
        + bins: numpy array length self.__nbins_s+1
            Binedges of RR(s)
            '''
        # Construct radial distribution P(r)
        r_hist, _ = numpy.histogram(self.rand_cat[:, 2], bins=self.__bins_r)
        r_hist = 1.*r_hist/self.rand_cat.shape[0]

        # Construct angular separation distribuion f(theta)
        theta_hist, _ = self.angular_distance()

        # Integration
        # convert f(theta) and P(r) into data points
        weight = numpy.zeros((theta_hist.size, r_hist.size))
        for i in range(theta_hist.size):
            for j in range(r_hist.size):
                weight[i, j] = theta_hist[i]*r_hist[j]
        temp = histogram2points(weight, self.__bins_theta, self.__bins_r)
        centers_r = 0.5*(self.__bins_r[:-1]+self.__bins_r[1:])

        rand_rand = numpy.zeros(self.__bins_s.size-1)
        for i, temp_r in enumerate(centers_r[r_hist != 0]):
            if i % 100 is 0:
                print(i)
            temp_s = get_distance(temp_r, temp[:, 1], temp[:, 0])
            temp_weight = r_hist[i]*temp[:, 2]
            temp_hist, _ = numpy.histogram(temp_s, self.__bins_s,
                                           weights=temp_weight)
            rand_rand += temp_hist

        return rand_rand, self.__bins_s

    # Construct DR(s)
    def data_rand(self):
        ''' Construct data-random separation DR(s) using random and
        data catalogs.
        Outputs:
        + data_data: numpy array length self.__nbins_s
            Values of DR(s)
        + bins: numpy array length self.__nbins_s+1
            Binedges of DR(s)
            '''

        # Construct radial distribution P(r)
        # and radial angular separation distribution g(theta, r)
        r_hist, _ = numpy.histogram(self.rand_cat[:, 2], self.__bins_r)
        r_hist = 1.*r_hist/self.rand_cat.shape[0]
        r_theta_hist = self.radial_angular_distance()[0]

        # Integration over P(r) and g(theta, r)
        temp = histogram2points(r_theta_hist, self.__bins_theta, self.__bins_r)
        centers_r = 0.5*(self.__bins_r[:-1]+self.__bins_r[1:])
        data_rand = numpy.zeros(self.__bins_s.size-1)

        for i, temp_r in enumerate(centers_r[r_hist != 0]):
            if i % 100 is 0:
                print(i)
            temp_s = get_distance(temp_r, temp[:, 1], temp[:, 0])
            temp_weight = r_hist[i]*temp[:, 2]
            temp_hist, _ = numpy.histogram(temp_s, self.__bins_s,
                                           weights=temp_weight)
            data_rand += temp_hist

        return data_rand, self.__bins_s

    # Construct tpcf(s) and tpcf(s)*s^2
    def correlation(self, rand_rand, data_rand, data_data, bins_s):
        ''' Construct two-point correlation function.
        Parameters:
        + rand_rand, data_rand, data_data: numpy array
            Values of RR(s), DR(s), and DD(s) respectively. All arrays must
            have the same size
        + bins_s:   numpy array
            Binedges of RR(s), DR(s), DD(s)
        Output:
        + correlation_function: numpy array
            Two-point correlation function computed using equations:
            f = [DD(s)-2*DR(s)+RR(s)]/RR(s)
            If RR(s) = 0, then f = 0
        + correlation_function: numpy array
            Two-point correlation function multiplied by s^2: f(s)s^2
            '''
        correlation_function = data_data-2*data_rand+rand_rand
        correlation_function = numpy.divide(correlation_function, rand_rand,
                                            out=numpy.zeros_like(rand_rand),
                                            where=rand_rand != 0)
        centers_s = 0.5*(bins_s[:-1]+bins_s[1:])
        correlation_function_ss = correlation_function*centers_s**2
        return correlation_function, correlation_function_ss

    # Compute normalization constant
    def normalization(self, weighted=True):
        ''' Compute and return normalization factor
        Parameters:
        + weighted: bool (default=True)
        Outputs:
         + norm_rr: float(weighted), int(unweighted)
            Normalization constant for RR(s)
            - If weighted is False, then: norm_rr = 0.5*n_rand*(n_rand-1),
            where n_rand is size of randoms catalog
            - If weighted is True, then: norm_rr = 0.5*(sum_w^2-sum_w2)
            where sum_w is the sum of weight, sum_w2 is the sum of weight square
        + norm_dr: float(weighted), int(unweighted)
            Normalization constant for DR(s)
           - If weighted is False, then: norm_dr = n_data*(n_rand-1),
            where n_rand, n_data is size of randoms, galaxies catalog
            - If weighted is True, then: norm_dd = sum_w_data*sum_w_rand
            where sum_w_rand, sum_w_data is the sum of weight of randoms,
            galaxies catalog
        + norm_dd: float(weighted), int(unweighted)
             Normalization constant for DD(s)
            - If weighted is False, then: norm_dd = 0.5*n_data*(n_data-1),
            where n_data is size of galaxies catalog
            - If weighted is True, then: norm_dd = 0.5*(sum_w^2-sum_w2)
            where sum_w is the sum of weight, sum_w2 is the sum of weight square
            '''
        if weighted:
            # Compute weighted normalization constant
            w_sum_rand = numpy.sum(self.rand_cat[:, 3])
            w_sum_data = numpy.sum(self.data_cat[:, 3])
            w2_sum_rand = numpy.sum(self.rand_cat[:, 3]**2)
            w2_sum_data = numpy.sum(self.data_cat[:, 3]**2)

            norm_rr = 0.5*(w_sum_rand**2-w2_sum_rand)
            norm_dd = 0.5*(w_sum_data**2-w2_sum_data)
            norm_dr = w_sum_rand*w_sum_data

            return norm_rr, norm_dr, norm_dd

        # Compute unweighted normalization constant
        n_rand = self.rand_cat.shape[0]
        n_data = self.data_cat.shape[0]

        norm_rr = 0.5*n_rand*(n_rand-1)
        norm_dd = 0.5*n_data*(n_data-1)
        norm_dr = n_data*n_rand

        return norm_rr, norm_dr, norm_dd

def main():
    ''' Main '''
    tpcf = CorrelationFunction('config.cfg')

    # Calculating normalized factor
    norm_w = tpcf.normalization()
    norm_uw = tpcf.normalization(weighted=False)
    print(norm_w, norm_uw)

    rand_rand, bins_s = tpcf.rand_rand()
    data_rand, _ = tpcf.data_rand()
    data_data, _ = tpcf.data_data()

    rand_rand = rand_rand/norm_uw[0]
    data_rand = data_rand/norm_uw[1]
    data_data = data_data/norm_uw[2]

    correlation_function = tpcf.correlation(rand_rand, data_rand, data_data,
                                            bins_s)


    matplotlib.pyplot.plot(bins_s[:-1], correlation_function[1])
    matplotlib.pyplot.savefig("test.png")


if __name__ == "__main__":
    main()
