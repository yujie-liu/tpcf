""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

# Standard Python module
import argparse
import glob
import configparser

# User-defined module
from lib.myio import save, load, read_cosmology
from lib.correlation import Correlation


def main():
    """ Combine jobs and calculate DD, DR, RR, and tpcf """

    # Read in cmd argument
    parser = argparse.ArgumentParser(
        description='Combine child calculation to produce two-point correlation function')
    parser.add_argument('-c', '-C', '--cosmology', type=str, default=None,
                        help='Path to cosmology configuration file.')
    parser.add_argument('-p', '-P', '--prefix', type=str, help='Run prefix.')
    parser.add_argument('-o', '-O', '--output', type=str, default=None,
                        help='Name of output .pkl. If not specific, use PREFIX_output.pkl')
    parser.add_argument('--version', action='version', version='KITCAT 1.10')
    args = parser.parse_args()

    print("COMBINE module")

    # Read in helper from pickles
    fname_list = sorted(glob.glob("{}_divide_*.pkl".format(args.prefix)))
    helper = None
    for i, fname in enumerate(fname_list):
        print("Reading from {}".format(fname))
        if i != 0:
            helper.add(next(load(fname)))
            continue
        helper = next(load(fname))

    if helper.n_cosmo == 1:
        # Calculate DD, DR, RR, and tpcf
        data_data = helper.get_dd()
        rand_rand, data_rand = helper.get_rr_dr()
        tpcf = Correlation(data_data, data_rand, rand_rand, helper.bins.bins('s'), helper.norm)
        tpcf_list = [tpcf]
    else:
        data = next(load('{}_preprocess.pkl'.format(args.prefix)))['dd']
        rand_rand, data_rand = helper.get_rr_dr()

        # Calculate DD, DR, RR, and tpcf given cosmology
        tpcf_list = [None]*helper.n_cosmo
        for i, cosmo in enumerate(helper.cosmo_list):
            print('- Cosmology Model {}'.format(i+1))

            # Calculate DD
            tree, catalog = data.build_balltree('euclidean', cosmo=cosmo, return_catalog=True)
            helper.set_dd(tree, catalog)
            data_data = helper.get_dd()

            # Set up and calculate correlation function
            tpcf = Correlation(data_data, data_rand[i], rand_rand[i],
                               helper.bins.bins('s'), helper.norm)
            tpcf_list[i] = tpcf

    # Save output
    if args.output is None:
        save('{}_output.pkl'.format(args.prefix), tpcf_list)
    else:
        if not args.output.endswith('.pkl'):
            args.output += '.pkl'
        save(args.output, tpcf_list)


if __name__ == "__main__":
    main()
