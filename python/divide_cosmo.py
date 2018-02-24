""" Script for submitting jobs to calculate DD(s), DR(s), and RR(s) """

# Standard Python modules
import argparse
import pickle
import time

# Python modules
import numpy

# User-defined modules
from lib.cosmolib.helper import JobHelper

def load(fname):
    """ Load pickle """
    with open(fname, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

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
    # Read in cmd argument
    parser = argparse.ArgumentParser(description='Divide calculation into multiple parts.')
    parser.add_argument('-p', '-P', '--prefix', type=str, help='Output prefix.')
    parser.add_argument('-i', '-I', '--index', type=int, help='Index of job. Start from 0 to N-1.')
    parser.add_argument('-n', '-N', '--total', type=int, help='Total number of jobs.')
    parser.add_argument('-t', '-T', '--time', type=bool, default=False, help='Enable save runtime.')
    parser.add_argument('--version', action='version', version='KITCAT 1.0')
    args = parser.parse_args()

    # Set job helper
    job_helper = JobHelper(args.n)
    job_helper.set_current_job(args.i)

    # Set timer
    timer = {'dd': None, 'dr': None, 'rr': None}

    # Load pickle file
    loader = next(load('{}_preprocess.pkl'.format(args.prefix)))

    # Set correlation helper
    correlation = loader['helper']

    # Calculate f(theta)
    start_time = time.time()
    tree = loader['rr']['tree']
    catalog = loader['rr']['catalog']
    correlation.set_theta_distr(tree, catalog, job_helper)
    timer['rr'] = time.time()-start_time
    print("--- {} seconds ---".format(timer['rr']))

    # Calculate g(theta, r)
    start_time = time.time()
    tree = loader['dr']['tree']
    data_catalog = loader['dr']['data_catalog']
    angular_catalog = loader['dr']['angular_catalog']
    mode = loader['dr']['mode']
    correlation.set_z_theta_distr(tree, data_catalog, angular_catalog, mode, job_helper)
    timer['dr'] = time.time()-start_time
    print("--- {} seconds ---".format(timer['dr']))

    # Save object and timer
    if args.time is not None:
        runtime = [timer['rr'], timer['dr'], timer['dd']]
        numpy.savetxt('{}_timer.txt'.format(args.prefix), runtime, header="DR(s) DD(s)")

    pickle_out = open("{}_divide_{:03d}.pkl".format(args.prefix, args.i), "wb")
    pickle.dump(correlation, pickle_out, protocol=-1)
    if args.i == 0:
        pickle.dump(loader['dd'], pickle_out, protocol=-1)
    pickle_out.close()


if __name__ == "__main__":
    main()
