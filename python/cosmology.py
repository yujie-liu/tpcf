# coding: utf-8
"""Quick conversion of positions in (z,alpha,delta) to 3D positions using
a given cosmology.

AstroPy has internally defined cosmologies based on recent CMB missions
(see http://docs.astropy.org/en/stable/cosmology/#built-in-cosmologies).
We'll use those parameters here to define comoving distance, but with
two changes:

1) Define H0 = 100h km/s/Mpc; astropy default uses the measured value of H0.
2) Allow for the possibile definition of nonzero curvature (not by default).
"""

from astropy import cosmology
from astropy import units as u
import numpy as np
import scipy.interpolate as interpolate

class Cosmology:
    """Define a cosmology.

    Attributes:
        z (:obj:`numpy.array` of :obj:`float`): redshift table.
        r (:obj:`numpy.array` of :obj:`float`): comoving distance table.
        r_vs_z (:obj:`scipy.interpolate.PchipInterpolator`): interpolation.
    """

    def __init__(self):
        """Initialize a cosmological model and arrays to interpolate
        redshift to comoving distance. Note that H0 = 100 h km/s/Mpc is
        used. For now the cosmological parameters measured by Planck
        (P.A.R. Ade et al., Paper XIII, A&A 594:A13, 2016) are used.
        """
        h = 0.677
        pl15 = cosmology.LambdaCDM(name='pl15',
                                   H0=100 * u.km/(u.Mpc*u.s),
                                   Om0=0.307,
                                   Ob0=0.0486,
                                   Ode0=0.693,
                                   Tcmb0=2.725*u.K,
                                   Neff=3.05,
                                   m_nu=np.asarray([0., 0., 0.06])*u.eV)

        z0 = 0.
        z1 = 3.
        nz = int((z1-z0)/0.001) + 1
        self.z = np.linspace(z0, z1, nz)
        self.r = np.zeros(nz, dtype=float)
        self.r[1:] = pl15.comoving_distance(self.z[1:])

        # Define an interpolation. Use the 1D cubic monotonic interpolator
        # from scipy.
        self.r_vs_z = interpolate.PchipInterpolator(self.z, self.r)

    def z2r(self, z):
        """Convert redshift to comoving line-of-sight distance.

        Args:
            z: redshift.

        Returns:
            r: comoving distance along the line of sight, in units.Mpc.
        """
        return self.r_vs_z(z)

