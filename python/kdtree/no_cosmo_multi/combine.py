""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

import sys
import glob
import numpy
from correlation_function import CorrelationFunction

def main():
    """ Main """
    # Cmd argument are configurationfile and output prefix
    config_fname = sys.argv[1]
    prefix = sys.argv[2]  # prefix can include directory name

    # Combining histogram by simply taking the sum in each bin
    fname_list = glob.glob("{}*".format(prefix))
    for i, fname in enumerate(fname_list):
        print("Reading from {}".format(fname))
        temp_file = numpy.load(fname)
        if i is 0:
            data_data = temp_file["DD"]
            theta_hist = temp_file["ANGULAR_D"]
            r_theta_hist = temp_file["ANGULAR_R"]
        else:
            data_data += temp_file["DD"]
            theta_hist += temp_file["ANGULAR_D"]
            r_theta_hist += temp_file["ANGULAR_R"]

    # Create an instance of two-point correlation function that reads in
    # configuration file
    # IMPORTANT: setting should be the same with job config file.
    tpcf = CorrelationFunction(config_fname)

    # Calculate RR(s) and DR(s)
    rand_rand, bins_s = tpcf.rand_rand(theta_hist)
    data_rand, _ = tpcf.data_rand(r_theta_hist)

    # Normalization
    norm = numpy.array([tpcf.normalization(weighted=True),
                        tpcf.normalization(weighted=False)])
    for i in range(2):
        rand_rand[i] = rand_rand[i]/norm[i][0]
        data_rand[i] = data_rand[i]/norm[i][1]
        data_data[i] = data_data[i]/norm[i][2]

    # Construct two-point correlation function, both weighted and unweighted
    correlation = numpy.zeros((2, 2, bins_s.size-1))
    correlation[0] = tpcf.correlation(rand_rand[0], data_rand[0], data_data[0],
                                      bins_s)
    correlation[1] = tpcf.correlation(rand_rand[1], data_rand[1], data_data[1],
                                      bins_s)

    # Save results
    numpy.savez("{}_final".format(prefix),
                DD=data_data, RR=rand_rand, DR=data_rand,
                TPCF=correlation[:, 0], TPCFSS=correlation[:, 1],
                BINS=bins_s)


if __name__ == "__main__":
    main()