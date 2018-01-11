""" Script for submitting jobs to calculate DD(s), DR(s), and RR(s) """

# Python modules
import sys
import configparser
import pickle

# User-defined modules
from lib.catalog import DataCatalog
from lib.cosmology import Cosmology
from lib.helper import Bins, JobHelper, CorrelationHelper


def initialize_catalog(config_fname):
    """ Initialize catalog and binning """
    # Read from configuration file
    config = configparser.SafeConfigParser()
    config.read(config_fname)

    # Set cosmology
    cosmo = Cosmology(config['COSMOLOGY'])

    # Set binnings
    bins = Bins(config['LIMIT'], config['BINWIDTH'], cosmo)

    # Initialize catalog
    data_catalog = DataCatalog(config['DATA'], bins.limit, cosmo)
    rand_catalog = DataCatalog(config['RANDOM'], bins.limit, cosmo)
    rand_catalog = rand_catalog.to_distr(bins.limit, bins.num_bins)

    return data_catalog, rand_catalog, bins


def main():
    """ Main
    Cmd arguments are job number, total number of jobs, configuration file,
    and output prefix
      + Job number: Job number must be an integer from 0 to total_job-1.
      + Total jobs: Total number of jobs that will be submitted.
      + Configuration: Setting for correlation function. See below.
      + Prefix: Prefix of the output files (can include folder path, folder must
      exist).
    If job number is 0, will also save comoving distribution P(r) and
    normalization factor."""

    # Read in cmd argument
    no_job = int(sys.argv[1])
    total_jobs = int(sys.argv[2])
    config_fname = sys.argv[3]
    prefix = sys.argv[4]

    # Set job helper
    job_helper = JobHelper(total_jobs)
    job_helper.set_current_job(no_job)

    # Initialize catalog and binnings
    data, rand, bins = initialize_catalog(config_fname)

    # Calculate DD(s), f(theta), and g(theta, r)
    data_data, _ = data.s_distr(bins.max('s'), bins.nbins('s'), job_helper)
    theta_distr, _ = rand.theta_distr(bins.max('theta'), bins.nbins('theta'), job_helper)
    r_theta_distr, _, _ = rand.r_theta_distr(
        data, bins.max('theta'), bins.nbins('theta'), job_helper)

    # Initialize save object
    save_object = CorrelationHelper(total_jobs)
    save_object.set_dd(data_data)
    save_object.set_theta_distr(theta_distr)
    save_object.set_r_theta_distr(r_theta_distr)
    if no_job == 0:
        save_object.set_r_distr(rand.r_distr)
        save_object.set_bins(bins)
        save_object.set_norm(data.norm(), rand.norm(data), rand.norm())

    pickle_out = open("{}_{:03d}.pickle".format(prefix, no_job), "wb")
    pickle.dump(save_object, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    main()
