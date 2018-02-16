""" Module to create preprocessed Catalog object"""

# Standard Python module
import sys
import pickle
import configparser

# User-defined module
from lib.catalog import DataCatalog
from lib.cosmology import Cosmology
from lib.helper import Bins

def main():
    """ Convert FITS into Catalog object
    Args:
    + config_fname: string
        Path to configuration file"""

    # Read from configuration file
    config_fname = sys.argv[1]
    config = configparser.SafeConfigParser()
    config.read(config_fname)

    # Set cosmology
    cosmo = Cosmology(config['COSMOLOGY'])

    # Set binnings
    bins = Bins(config['LIMIT'], config['BINWIDTH'], cosmo)

    # Initialize catalog
    catalog = DataCatalog(config['PREPROCESS'], bins.limit, cosmo)
    catalog = catalog.to_distr(bins.limit, bins.num_bins)

    # Save pickle
    pickle_out = open(config['PREPROCESS']['output'], 'wb')
    pickle.dump(bins, pickle_out, protocol=-1)
    pickle.dump(catalog, pickle_out, protocol=-1)
    pickle_out.close()


if __name__ == "__main__":
    main()
