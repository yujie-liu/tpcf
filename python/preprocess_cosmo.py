""" Module to create preprocessed catalog object"""

# Standard Python module
import pickle
import configparser
import argparse

# User-defined module
from lib.cosmolib.catalog import DataCatalog
from lib.cosmolib.helper import Bins, CorrelationHelper

def save(fname, *save_list):
    """ Pickle a list of objects """
    pickle_out = open(fname, 'wb')
    for save_object in save_list:
        pickle.dump(save_object, pickle_out, protocol=-1)
    pickle_out.close()

def main():
    """ Convert FITS into Catalog object
    Args:
    + config_fname: string
        Path to configuration file"""
    print("PREPROCESS module")

    # Read in cmd argument
    # Read in cmd argument
    parser = argparse.ArgumentParser(description='Preprocess galaxy and random catalogs.')
    parser.add_argument('-c', '--config', type=str, help='Path to configuration file.')
    parser.add_argument('-p', '--prefix', type=str, help='Output prefix.')
    parser.add_argument('--version', action='version', version='KITCAT 1.0')
    args = parser.parse_args()

    # Read from configuration file
    config = configparser.SafeConfigParser()
    config.read(args.config)

    # Set binning schemes
    bins = Bins(config['LIMIT'], config['BINWIDTH'])

    # Initialize catalog and save dictionary
    print('Initialize catalog')
    data = DataCatalog(config['GALAXY'], bins.limit) # data catalog
    rand = DataCatalog(config['RANDOM'], bins.limit) # random catalog
    rand = rand.to_distr(bins.limit, bins.num_bins)
    save_dict = {'dd': None, 'dr': None, 'rr': None, 'helper': None}
    save_dict['dd'] = data

    # Create a balltree for f(theta), RR calculation and pickle
    print('Setting up RR')
    tree, catalog = rand.build_balltree(return_catalog=True)
    save_dict['rr'] = {'tree': tree, 'catalog': catalog}

    # Create a balltree for g(theta, r), DR calculation and pickle
    # Runtime is O(N*log(M)); M = size of tree; N = size of catalog
    # Create tree from catalog with smaller size
    print('Setting up DR')
    mode = 'angular_tree'
    if rand.angular_distr.shape[0] <= data.ntotal:
        # Tree from Data, catalog from Random
        tree = data.build_balltree(metric='haversine', return_catalog=False)
        mode = 'data_tree'
    save_dict['dr'] = {
        'tree': tree, 'data_catalog': data.catalog, 'angular_catalog': catalog, 'mode': mode}

    # Save binning
    helper = CorrelationHelper(bins)
    helper.set_z_distr(rand.z_distr)
    helper.set_norm(norm_dd=data.norm(), norm_dr=rand.norm(data), norm_rr=rand.norm())
    save_dict['helper'] = helper

    # Save
    save('{}_preprocess.pkl'.format(args.prefix), save_dict)


if __name__ == "__main__":
    main()
