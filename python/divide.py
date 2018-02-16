""" Script for submitting jobs to calculate DD(s), DR(s), and RR(s) """

# Standard Python modules
import sys
import configparser
import pickle

# User-defined modules
from lib.cosmology import Cosmology
from lib.catalog import DataCatalog, DistrCatalog
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
    # Random catalog
    rand_type = config['RANDOM']['type'].lower()
    if rand_type == 'data_catalog':
        # Import randoms catalog as FITS catalog
        rand_catalog = DataCatalog(config['RANDOM'], bins.limit, cosmo)
        rand_catalog = rand_catalog.to_distr(bins.limit, bins.num_bins)

    elif rand_type == 'distr_catalog':
        # Import randoms catalog as NPZ distribution
        rand_catalog = DistrCatalog(config['RANDOM'])

    elif rand_type == 'pickle':
        # Import randoms catalog as pickles
        pickle_in = open(config['RANDOM']['path'], 'rb')

        # Comparing two bins value
        test_bins = pickle.load(pickle_in)
        if not bins == test_bins:
            raise RuntimeError("Bins do not match.")

        print("Import catalog from Pickle: {}".format(config['RANDOM']['path']))
        rand_catalog = pickle.load(pickle_in)
        pickle_in.close()

    else:
        raise ValueError('TYPE must be "data_catalog", "distr_catalog", or "pickle"')

    # Data catalog
    data_catalog = DataCatalog(config['GALAXY'], bins.limit, cosmo)

    return data_catalog, rand_catalog, bins


def main():
    """ Main
    Cmd arguments are job number, total number of jobs, configuration file,
    and output prefix
      + --ijob: Job number must be an integer from 0 to total_job-1.
      + --njob: Total number of jobs that will be submitted.
      + --config: Setting for correlation function. See below.
      + --prefix: Prefix of the output files (can include folder path, folder must
      exist).
    If job number is 0, will also save comoving distribution P(r) and
    normalization factor."""
    print("DIVIDE module")

    # Read in cmd argument
    args = {'--config': None, '--njob': None, '--ijob': None, '--prefix': None}
    for i in range(len(sys.argv)):
        arg = sys.argv[i]
        if arg in args.keys():
            args[arg] = sys.argv[i+1]
            print(arg, sys.argv[i+1])
    args['--njob'] = int(args['--njob'])
    args['--ijob'] = int(args['--ijob'])

    # Set job helper
    job_helper = JobHelper(args['--njob'])
    job_helper.set_current_job(args['--ijob'])

    # Initialize catalog and binnings
    data, rand, bins = initialize_catalog(args['--config'])

    # Calculate DD(s), f(theta), and g(theta, r)
    data_data, _ = data.s_distr(bins.max('s'), bins.nbins('s'), job_helper)
    theta_distr, _ = rand.theta_distr(bins.max('theta'), bins.nbins('theta'), job_helper)
    r_theta_distr, _, _ = rand.r_theta_distr(
        data, bins.max('theta'), bins.nbins('theta'), job_helper)

    # Initialize save object
    save_object = CorrelationHelper(args['--njob'])
    save_object.set_dd(data_data)
    save_object.set_theta_distr(theta_distr)
    save_object.set_r_theta_distr(r_theta_distr)
    if args['--ijob'] == 0:
        save_object.set_r_distr(rand.r_distr)
        save_object.set_bins(bins)
        save_object.set_norm(data.norm(), rand.norm(data), rand.norm())

    pickle_out = open("{}_{:03d}.pkl".format(args['--prefix'], args['--ijob']), "wb")
    pickle.dump(save_object, pickle_out, protocol=-1)
    pickle_out.close()


if __name__ == "__main__":
    main()
