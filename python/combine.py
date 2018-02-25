""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

import argparse
import glob
import pickle
import configparser

import numpy

from lib.cosmology import Cosmology
from lib.correlation import Correlation

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
    Args: output prefix """
    print("COMBINE module")

    # Read in cmd argument
    parser = argparse.ArgumentParser(
        description='Combine child calculation to produce two-point correlation function')
    parser.add_argument('-c', '-C', '--cosmology', type=str, default=None,
                        help='Path to cosmology configuration file.')
    parser.add_argument('-p', '-P', '--prefix', type=str, help='Output prefix.')
    parser.add_argument('--version', action='version', version='KITCAT 1.10')
    args = parser.parse_args()

    # Read in helper from pickles
    fname_list = sorted(glob.glob("{}_divide*.pkl".format(args.prefix)))
    helper = None
    for i, fname in enumerate(fname_list):
        print("Reading from {}".format(fname))
        if i != 0:
            helper.add(next(load(fname)))
            continue
        helper = next(load(fname))

    # If cosmology is not set previously
    cosmo = None
    if helper.cosmo is None:
        # Read in cosmology configuration
        config = configparser.SafeConfigParser()
        config.read(args.cosmology)
        cosmo = Cosmology(config['COSMOLOGY'])

        # Calculate DD
        data = next(load('{}_preprocess.pkl'.format(args.prefix)))['dd']
        tree, catalog = data.build_balltree('euclidean', cosmo=cosmo, return_catalog=True)
        helper.set_data_data(tree, catalog)

    data_data = helper.get_data_data()
    data_rand = helper.get_data_rand(cosmo)
    rand_rand = helper.get_rand_rand(cosmo)

    # Set up and calculate correlation function
    tpcf = Correlation(data_data, data_rand, rand_rand, helper.bins.bins('s'), helper.norm)

    # Save into NPZ files
    numpy.savez("{}_weighted.npz".format(args.prefix),
                bins=tpcf.bins, tpcf=tpcf.tpcf(), tpcfss=tpcf.tpcfss(),
                dd=tpcf.w_distr['dd'], rr=tpcf.w_distr['rr'], dr=tpcf.w_distr['dr'])
    numpy.savez("{}_unweighted.npz".format(args.prefix),
                bins=tpcf.bins, tpcf=tpcf.tpcf(False), tpcfss=tpcf.tpcfss(False),
                dd=tpcf.u_distr['dd'], rr=tpcf.u_distr['rr'], dr=tpcf.u_distr['dr'])

if __name__ == "__main__":
    main()
