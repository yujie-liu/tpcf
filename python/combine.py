""" Script for combining job results and calculate DD(s), DR(s), and RR(s) """

import argparse
import glob
import pickle
import configparser

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

def save(fname, *save_list):
    """ Pickle a list of objects """
    pickle_out = open(fname, 'wb')
    for save_object in save_list:
        pickle.dump(save_object, pickle_out, protocol=-1)
    pickle_out.close()

def read_cosmology(cosmo):
    """ Read in multiple cosmological models """
    cosmo_list = []
    cosmo_dict = {'hubble0': [], 'omega_m0': [], 'omega_de0': []}

    # Read value from configuration section
    for key, val in cosmo.items():
        if key in cosmo_dict.keys():
            cosmo_dict[key] = [float(pars) for pars in val.split(',')]

    # Initialize list of cosmological model
    for i in range(len(cosmo_dict['hubble0'])):
        cosmo_list.append(Cosmology({'hubble0': cosmo_dict['hubble0'][i],
                                     'omega_m0': cosmo_dict['omega_m0'][i],
                                     'omega_de0': cosmo_dict['omega_de0'][i]}))
    return cosmo_list

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
    fname_list = sorted(glob.glob("{}_divide_*.pkl".format(args.prefix)))
    helper = None
    for i, fname in enumerate(fname_list):
        print("Reading from {}".format(fname))
        if i != 0:
            helper.add(next(load(fname)))
            continue
        helper = next(load(fname))

    if helper.cosmo is not None:
        # If cosmology is set previously

        # Calculate DD, DR, RR, and tpcf
        data_data = helper.get_dd()
        rand_rand, data_rand = helper.get_rr_dr()
        tpcf = Correlation(data_data, data_rand, rand_rand, helper.bins.bins('s'), helper.norm)
        save('{}_output.pkl'.format(args.prefix), tpcf)

    else:
        # If cosmology is not set previously
        loader = next(load('{}_preprocess.pkl'.format(args.prefix)))
        data = loader['dd']

        # Cosmological configuration parameter
        if args.cosmology is None:
            print('- Reading cosmological model from preprocess')
            # Read in cosmology from preprocess
            cosmo_list = loader['cosmo_list']
        else:
            print('- Reading cosmological model from configuration file {}'.format(args.cosmology))
            # Read in cosmology from configuration file
            config = configparser.SafeConfigParser()
            config.read(args.cosmology)
            cosmo_list = read_cosmology(config['COSMOLOGY'])

        # Calculate DD, DR, RR, and tpcf given cosmology
        tpcf_list = [None]*len(cosmo_list)
        for i, cosmo in enumerate(cosmo_list):
            print('- Cosmology Model {}'.format(i+1))

            tree, catalog = data.build_balltree('euclidean', cosmo=cosmo, return_catalog=True)
            helper.set_dd(tree, catalog)

            data_data = helper.get_dd()
            rand_rand, data_rand = helper.get_rr_dr(cosmo)

            # Set up and calculate correlation function
            tpcf = Correlation(data_data, data_rand, rand_rand, helper.bins.bins('s'), helper.norm)
            tpcf_list[i] = tpcf
        save('{}_output.pkl'.format(args.prefix), tpcf_list)


if __name__ == "__main__":
    main()
