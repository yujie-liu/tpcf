""" Script for calculating angular distribution and comoving distribution of
a catalog """

import sys
import configparser
import numpy
from cosmology import Cosmology
from correlation_function import import_fits, get_bins

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi


def main():
    """ Calculating angular distribution and comoving distribution of a catalog
    Arguments:
      + config filename: path to catalog
      """
    config_fname = sys.argv[1]

    config = configparser.ConfigParser()
    config.read(config_fname)
    binnings = config['BINNING']
    region = config['REGION']

    # Create cosmology
    cosmo_params = config["COSMOLOGY"]
    cosmo = Cosmology()
    cosmo.set_model(float(cosmo_params["hubble0"]),
                    float(cosmo_params["omega_m0"]),
                    float(cosmo_params["omega_b0"]),
                    float(cosmo_params["omega_de0"]),
                    float(cosmo_params["temp_cmb"]),
                    float(cosmo_params["nu_eff"]),
                    list(map(float, cosmo_params["m_nu"].split(","))))

    # Import catalog
    catalog = import_fits('random_filename', config['PREPROCESS'], region, cosmo)

    # Setting up some bin variables
    s_max = float(region["s_max"])
    binwidth_s = float(binnings['binwidth_s'])
    _, binwidth_s = get_bins(0., s_max, binwidth_s)

    # comoving distance distribution(Mpc/h)
    r_min = cosmo.z2r(float(region["z_min"]))
    r_max = cosmo.z2r(float(region["z_max"]))
    if binnings['binwidth_r'] == 'auto':
        binwidth_r = binwidth_s/2.
    else:
        binwidth_r = float(binnings['binwidth_r'])
    bins_r, binwidth_r = get_bins(r_min, r_max, binwidth_r)

    # declination, right ascension, and angular separation (rad)
    dec_min = DEG2RAD*float(region["dec_min"])
    dec_max = DEG2RAD*float(region["dec_max"])
    if binnings['binwidth_dec'] == 'auto':
        binwidth_dec = 1.*binwidth_r/r_max
    else:
        binwidth_dec = DEG2RAD*float(binnings['binwidth_dec'])
    bins_dec, binwidth_dec = get_bins(dec_min, dec_max,
                                      binwidth_dec)

    ra_min = DEG2RAD*float(region["ra_min"])
    ra_max = DEG2RAD*float(region["ra_max"])
    if binnings['binwidth_ra'] == 'auto':
        binwidth_ra = 1.*binwidth_r/r_max
    else:
        binwidth_ra = DEG2RAD*float(binnings['binwidth_ra'])
    bins_ra, binwidth_ra = get_bins(ra_min, ra_max, binwidth_ra)

    # Calculate P(r) and R(ra, dec)
    # Calculate weighted and unweighted radial distribution P(r) as
    # two one-dimensional histograms respectively
    r_hist = numpy.zeros((2, bins_r.size-1))
    r_hist[0] += numpy.histogram(catalog[:, 2], bins=bins_r,
                                 weights=catalog[:, 3])[0]
    r_hist[1] += numpy.histogram(catalog[:, 2], bins=bins_r)[0]
    r_hist = 1.*r_hist/catalog.shape[0]

    # Calculate the angular distribution R(ra, dec)
    angular_hist, _, _ = numpy.histogram2d(catalog[:, 0], catalog[:, 1],
                                           bins=(bins_dec, bins_ra))

    # Calculate the component for normalization constant
    w_sum = numpy.sum(catalog[:, 3])
    w2_sum = numpy.sum(catalog[:, 3]**2)

    # Save results
    numpy.savez(config['PREPROCESS']["output_filename"],
                R_HIST=r_hist, ANGULAR_HIST=angular_hist,
                BINS_R=bins_r, BINS_DEC=bins_dec, BINS_RA=bins_ra,
                N_DATA=catalog.shape[0], W_SUM=w_sum, W2_SUM=w2_sum)

if __name__ == "__main__":
    main()
