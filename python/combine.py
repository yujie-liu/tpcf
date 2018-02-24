""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

# Standard Python modules
import glob
import pickle
import argparse

# Python modules
import numpy

# User-defined modules
from lib.correlation import Correlation


def main():
    """ Main
    Args: output prefix """
    print("COMBINE module")

    parser = argparse.ArgumentParser(
        description='Combine child calculation to produce two-point correlation function')
    parser.add_argument('-p', '-P', '--prefix', type=str, help='Output prefix.')
    parser.add_argument('--version', action='version', version='KITCAT 1.0')
    args = parser.parse_args()

    # Read in helper from pickles
    fname_list = sorted(glob.glob("{}_divide*.pkl".format(args.prefix)))
    helper = None
    for i, fname in enumerate(fname_list):
        print("Reading from {}".format(fname))
        pickle_in = open(fname, "rb")
        if i != 0:
            helper.add(pickle.load(pickle_in))
            pickle_in.close()
            continue
        helper = pickle.load(pickle_in)
        pickle_in.close()

    # Calculate DD, DR, RR
    data_data = helper.get_data_data()
    data_rand = helper.get_data_rand()
    rand_rand = helper.get_rand_rand()

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
