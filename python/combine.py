""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

import sys
import glob

import pickle

def main():
    """ Main
    Args: output prefix """
    prefix = sys.argv[1]

    # Read in helper from pickles
    fname_list = sorted(glob.glob("{}*.pickle".format(prefix)))
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
    data_data = helper.get_dd()
    data_rand = helper.get_dr()
    rand_rand = helper.get_rr()
    bins_s = helper.bins.bins('s')
    bins_s = 0.5*(bins_s[:-1]+bins_s[1:])
    norm = helper.norm

    for i in range(2):
        data_data[i] = data_data[i]/norm['dd'][i]
        data_rand[i] = data_rand[i]/norm['dr'][i]
        rand_rand[i] = rand_rand[i]/norm['rr'][i]



if __name__ == "__main__":
    main()
