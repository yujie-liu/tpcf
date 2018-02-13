""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

import sys
import glob
import pickle
import configparser

import numpy

from lib.cosmology import Cosmology
from lib.correlation import Correlation


def main():
    """ Main
    Args: output prefix """
    print("COMBINE module")
    config_fname = sys.argv[1]
    prefix = sys.argv[2]

    # Read in helper from pickles
    fname_list = sorted(glob.glob("{}*.pkl".format(prefix)))
    helper = None
    for i, fname in enumerate(fname_list):
        print("Reading from {}".format(fname))
        pickle_in = open(fname, "rb")
        if i != 0:
            helper.add(pickle.load(pickle_in))
            pickle_in.close()
            continue
        helper = pickle.load(pickle_in)
        data  = pickle.load(pickle_in)
        pickle_in.close()

    # Set cosmology
    config = configparser.SafeConfigParser()
    config.read(config_fname)
    cosmo = Cosmology(config['COSMOLOGY'])

    # Calculate DD, DR, RR
    data_data, _ = data.s_distr(helper.bins.max('s'), helper.bins.nbins('s'), cosmo)
    data_rand = helper.get_dr(cosmo)
    rand_rand = helper.get_rr(cosmo)

    # Set up and calculate correlation function
    tpcf = Correlation(data_data, data_rand, rand_rand, helper.bins.bins('s'), helper.norm)

    # Save into NPZ files
    numpy.savez("{}_weighted.npz".format(prefix),
                bins=tpcf.bins, tpcf=tpcf.tpcf(), tpcfss=tpcf.tpcfss(),
                dd=tpcf.w_distr['dd'], rr=tpcf.w_distr['rr'], dr=tpcf.w_distr['dr'])
    numpy.savez("{}_unweighted.npz".format(prefix),
                bins=tpcf.bins, tpcf=tpcf.tpcf(False), tpcfss=tpcf.tpcfss(False),
                dd=tpcf.u_distr['dd'], rr=tpcf.u_distr['rr'], dr=tpcf.u_distr['dr'])

if __name__ == "__main__":
    main()
