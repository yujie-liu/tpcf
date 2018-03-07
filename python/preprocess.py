""" Module to create preprocessed catalog object"""

# Standard Python module
import pickle
import configparser
import argparse

# User-defined module
from lib.catalog import GalaxyCatalog
from lib.cosmology import Cosmology
from lib.helper import CorrelationHelper
from lib.bins import Bins

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
    """ Convert FITS into Catalog object """
    print("PREPROCESS module")

    # Read in cmd argument
    parser = argparse.ArgumentParser(description='Preprocess galaxy and random catalogs.')
    parser.add_argument('-c', '-C', '--config', type=str, help='Path to configuration file.')
    parser.add_argument('-p', '-P', '--prefix', type=str, help='Output prefix.')
    parser.add_argument('-iz', '--index_z', type=int, default=0, help='Index of Z-slice.')
    parser.add_argument('-nz', '--total_z', type=int, default=1, help='Total number of Z-slices.')
    parser.add_argument('--version', action='version', version='KITCAT 1.10')
    args = parser.parse_args()

    # Read from configuration file
    print('- Reading configuration file from {}'.format(args.config))
    config = configparser.SafeConfigParser()
    config.read(args.config)

    # Read in cosmology
    print('- Setting up cosmological models')
    cosmo_list = read_cosmology(config['COSMOLOGY'])
    cosmo = None if len(cosmo_list) > 1 else cosmo_list[0]
    print('- Number of models: {}'.format(len(cosmo_list)))

    # Read in binning scheme
    print('- Setting up binning schemes')
    bins = Bins(config['LIMIT'], num_bins=config['NBINS'])

    # Initialize catalog and save dictionary
    print('- Initialize catalog')
    data = GalaxyCatalog(config['GALAXY'], bins.limit) # data catalog
    rand = GalaxyCatalog(config['RANDOM'], bins.limit) # random catalog
    rand = rand.to_rand(bins.limit, bins.num_bins, cosmo)
    save_dict = {'dd': None, 'dr': None, 'rr': None, 'helper': None}
    if cosmo is None:
        save_dict['cosmo_list'] = cosmo_list

    # Create a kd-tree for DD calculation and pickle
    print('- Setting up DD')
    if cosmo is None:
        save_dict['dd'] = data
    else:
        tree, catalog = data.build_balltree(metric='euclidean', cosmo=cosmo, return_catalog=True)
        save_dict['dd'] = {'tree': tree, 'catalog': catalog}

    # Create a balltree for f(theta), RR calculation and pickle
    print('- Setting up RR')
    tree, catalog = rand.build_balltree(return_catalog=True)
    save_dict['rr'] = {'tree': tree, 'catalog': catalog}

    # Create a balltree for g(theta, r), DR calculation and pickle
    # Runtime is O(N*log(M)); M = size of tree; N = size of catalog
    # Create tree from catalog with smaller size
    print('- Setting up DR')
    mode = 'angular_tree'
    if rand.angular_distr.shape[0] <= data.ntotal:
        # Tree from Data, catalog from Random
        tree = data.build_balltree(metric='haversine', return_catalog=False)
        mode = 'data_tree'
    save_dict['dr'] = {'tree': tree,
                       'data_catalog': data.get_catalog(cosmo),
                       'angular_catalog': catalog,
                       'mode': mode}

    # Save helper object
    helper = CorrelationHelper(bins, cosmo)
    helper.set_rz_distr(rand.rz_distr)
    helper.set_norm(norm_dd=data.norm(), norm_dr=rand.norm(data), norm_rr=rand.norm())
    save_dict['helper'] = helper

    # Save
    save('{}_preprocess.pkl'.format(args.prefix), save_dict)


if __name__ == "__main__":
    main()
