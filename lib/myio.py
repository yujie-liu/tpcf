""" Modules for handling simple I/O functions"""

import pickle

from lib.cosmology import Cosmology

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
    i = 0
    for key, val in cosmo.items():
        if key in cosmo_dict.keys():
            # Get parameters and convert to float
            pars = [float(p) for p in val.split(',')]

            # Check length of each arguments
            if i == 0:
                n = len(pars)
            if n != len(pars):
                raise ValueError('All cosmological parameters must have the same length')

            # Add into a temporary dictionary
            cosmo_dict[key] = pars

    # Initialize list of cosmological model
    for i in range(n):
        temp = {}
        for key, val in cosmo_dict.items():
            temp[key] = val[i]
        cosmo_list.append(Cosmology(temp))

    return cosmo_list
