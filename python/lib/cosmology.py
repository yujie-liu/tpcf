""" Convert redshift into comoving distance given cosmology

AstroPy has internally defined cosmologies based on recent CMB missions
(see http://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies).
We'll use those parameters here to define comoving distance, but with
two changes:

1) Define H0 = 100h km/s/Mpc; astropy default uses the measured value of H0.
2) Allow for the possibile definition of nonzero curvature (not by default).
"""


import numpy
from astropy import cosmology, units
from scipy import interpolate

class Cosmology():
    """ Class to manage cosmological parameters and convert redshift z into
        comoving distance using linear interpolation technique. """
    def __init__(self, cosmo):
        """ Initialize a cosmological model. Note that H0 = 100 h km/s/Mpc
        is used. For now the cosmological parameters measured by Planck
        (P.A.R. Ade et al., Paper XIII, A&A 594:A13, 2016) are used.
        """
        self.model = 0
        self.__comoving_table = 0
        self.set_model(cosmo)

    def __set_comoving_table(self):
        """ Set redshift-comoving table """
        # Default parameters
        z_min = 0.
        z_max = 3.0
        step = 0.00001

        # Set table
        n =  int(numpy.ceil((z_max-z_min)/step))
        z = numpy.linspace(z_min, z_max, n)
        r = self.model.comoving_distance(z)
        self.__comoving_table = numpy.array([z, r]).T

    def set_model(self, cosmo):
        """ Read cosmologies from configuration file and reset table
        Inputs:
        + cosmo: dictionary
            Dictionary with comsology paramaters
            Key: hubble0, omega_m0, omega_b0, omega_de0, temp_cmb, nu_eff
        """
        # Set astropy cosmology model
        params = {}
        for key, val in cosmo.items():
            params[key] = float(val)
        m_nu1 = params['m_nu1']
        m_nu2 = params['m_nu2']
        m_nu3 = params['m_nu3']
        self.model = cosmology.LambdaCDM(H0=params['hubble0']*units.km/(units.Mpc*units.s),
                                         Om0=params['omega_m0'],
                                         Ob0=params['omega_b0'],
                                         Ode0=params['omega_de0'],
                                         Tcmb0=params['temp_cmb'],
                                         Neff=params['nu_eff'],
                                         m_nu=[m_nu1, m_nu2, m_nu3]*units.eV)

        # Set up redshift-comoving table
        self.__set_comoving_table()

    def z2r(self, z):
        """Convert redshift to comoving distance by linear interpolating from table.
        Inputs:
        + z: float
            Redshift 0 < z < 3.0.
        Outputs:
        + r: float
            Return comoving distance given set cosmology.
        """
        r = interpolate.PchipInterpolator(*self.__comoving_table.T)(z)
        if isinstance(z, (list, tuple, numpy.ndarray)):
            return r
        # Avoid 0-dimensional array
        return float(r)


    @property
    def comoving_table(self):
        """ Return a copy of the comoving distance table """
        return numpy.copy(self.__comoving_table)
