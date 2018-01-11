""" Script to generat unittest data """

import configparser
import numpy
from correlation_function import CorrelationFunction
from cosmology import Cosmology

def main():
    """ Run and plot two-point correlation function using
    class correlation_function.CorrelationFunction """

    # Define cosmology
    config = configparser.ConfigParser()
    config.read("test/test_config.cfg")
    cosmo_params = config["COSMOLOGY"]
    cosmo = Cosmology()
    cosmo.set_model(float(cosmo_params["hubble0"]),
                    float(cosmo_params["omega_m0"]),
                    float(cosmo_params["omega_b0"]),
                    float(cosmo_params["omega_de0"]),
                    float(cosmo_params["temp_cmb"]),
                    float(cosmo_params["nu_eff"]),
                    list(map(float, cosmo_params["m_nu"].split(","))))

    # Generate data for unittest
    tpcf = CorrelationFunction("test/test_config.cfg")
    norm = numpy.array([tpcf.normalization(weighted=True),
                        tpcf.normalization(weighted=False)])
    z_hist, bins_pz = tpcf.redshift_distribution()
    theta_hist, bins_ftheta = tpcf.angular_distance(0, 1)
    theta_z_hist, bins_gtheta, bins_gz = tpcf.angular_redshift(0, 1)
    rand_rand, bins_rr = tpcf.rand_rand(theta_hist, z_hist, cosmo)
    data_rand, bins_dr = tpcf.data_rand(theta_z_hist, z_hist, cosmo)
    data_data, bins_dd = tpcf.data_data(0, 1, cosmo)

    # Construct two-point correlation function, both weighted and unweighted
    correlation = tpcf.correlation(rand_rand[0], data_rand[0], data_data[0],
                                   bins_dd)

    # Save RR(s), DR(s), DD(s) and tpcf into .npz format
    numpy.savez("test/tpcf_test",
                RR=rand_rand, DR=data_rand, DD=data_data,
                BINS_RR=bins_rr, BINS_DR=bins_dr, BINS_DD=bins_dd,
                Z_HIST=z_hist, BINS_PZ=bins_pz,
                ANGULAR_D=theta_hist, BINS_FTHETA=bins_ftheta,
                ANGULAR_Z=theta_z_hist, BINS_GTHETA=bins_gtheta, BINS_GZ=bins_gz,
                TPCF=correlation, NORM=norm)


if __name__ == "__main__":
    main()
