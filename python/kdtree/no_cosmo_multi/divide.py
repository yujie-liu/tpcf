""" Script for submitting jobs to calculate DD(s), DR(s), and RR(s) """

import sys
import numpy
from correlation_function import CorrelationFunction

def main():
    """ Main """

    # Cmd arguments are job number, total number of jobs, configuration file,
    # and output prefix
    # Job number must be a nonnegative integer less than total number of jobs
    # Total number of jobs must be at least 1
    no_job = int(sys.argv[1])
    total_jobs = int(sys.argv[2])
    config_fname = sys.argv[3]
    prefix = sys.argv[4]

    # Create an instance of two-point correlation function that reads in
    # configuration file
    tpcf = CorrelationFunction(config_fname)

    # Calculate child-process histogram
    print("Job number: {}. Total jobs: {}.".format(no_job, total_jobs))
    # Angular distance distribution f(theta)
    theta_hist, _ = tpcf.angular_distance(no_job, total_jobs)
    # Radial angular distribution g(theta, r)
    r_theta_hist, _, _ = tpcf.r_angular_distance(no_job, total_jobs)
    # Galaxies separation distribution DD(s)
    data_data, _ = tpcf.data_data(no_job, total_jobs)

    # Save with prefix
    numpy.savez("{}_{:03d}".format(prefix, no_job),
                DD=data_data, ANGULAR_D=theta_hist, ANGULAR_R=r_theta_hist)



if __name__ == "__main__":
    main()
