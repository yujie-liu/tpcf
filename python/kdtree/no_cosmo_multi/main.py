""" Sample script to run and plot two-point correlation function using
module correlation_function """

import numpy
import matplotlib
import matplotlib.pyplot
from correlation_function import CorrelationFunction

matplotlib.rc('font', size=15)


def main():
    """ Test """
    # Constructor takes in configuration file names
    tpcf = CorrelationFunction("sample_config.cfg")

    # Get normalization factor (both weighted and unweighted)
    norm = numpy.array([tpcf.normalization(weighted=True),
                        tpcf.normalization(weighted=False)])

    # Calculate RR(s)
    # theta_hist, bins_theta = tpcf.angular_distance(0, 3)
    # theta_hist += tpcf.angular_distance(1, 3)[0]
    # theta_hist += tpcf.angular_distance(2, 3)[0]
    # rand_rand, bins_s = tpcf.rand_rand(theta_hist)
    # rand_rand[0] = rand_rand[0]/norm[0][0]
    # rand_rand[1] = rand_rand[1]/norm[1][0]

    # Calculate RR(s)
    r_theta_hist, bins_theta, bins_r = tpcf.r_angular_distance(0, 3)
    r_theta_hist += tpcf.r_angular_distance(1, 3)[0]
    r_theta_hist += tpcf.r_angular_distance(2, 3)[0]
    data_rand, bins_s = tpcf.data_rand(r_theta_hist)
    data_rand[0] = data_rand[0]/norm[0][1]
    data_rand[1] = data_rand[1]/norm[1][1]

    # Calculate DD(s)
    # data_data, bins_s = tpcf.data_data(0, 3)
    # data_data += tpcf.data_data(1, 3)[0]
    # data_data += tpcf.data_data(2, 3)[0]
    # data_data[0] = data_data[0]/norm[0][2]
    # data_data[1] = data_data[1]/norm[1][2]


    # Create plot figure for RR(s), DR(s), DD(s) and tpcf
    figure, axis = matplotlib.pyplot.subplots(1, figsize=(12, 8))

    bins_center = 0.5*(bins_s[:-1]+bins_s[1:])
    axis.plot(bins_center, data_rand[0], label='Weighted')
    axis.plot(bins_center, data_rand[1], label='Unweighted')
    axis.legend()
    figure.tight_layout()
    matplotlib.pyplot.show()


if __name__ == "__main__":
    main()
