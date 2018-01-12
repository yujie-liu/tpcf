""" Script to convert .fits to .root"""

import numpy
import rootpy.compiled
from cosmology import Cosmology
from astropy.io import fits
from rootpy.io import root_open
from rootpy.tree import Tree, TreeModel, FloatCol, ObjectCol

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi

class Galaxy(TreeModel):
    phi = FloatCol()
    theta = FloatCol()
    r = FloatCol()
    w = FloatCol()

def main():
    """Main"""
    # Define cosmology
    # parameters
    hubble0 = 100
    omega_m0 = 0.307
    omega_b0 = 0.0486
    omega_de0 = 0.693
    temp_cmb = 2.725
    nu_eff = 3.05
    m_nu = [0.,0.,0.06]
    cosmo = Cosmology()
    cosmo.set_model(hubble0, omega_m0, omega_b0, omega_de0, temp_cmb, nu_eff,
                    m_nu)

    # Open .FITS
    fname = "/home/chris/project/bao/correlation/catalog/boss/randoms_DR9_CMASS_North.fits"
    hdulist = fits.open(fname)
    tbdata = hdulist[1].data

    ra = tbdata["ra"]
    dec = tbdata["dec"]
    z = tbdata["z"]
    r = cosmo.z2r(z)
    try:
        weight_fkp = tbdata["weight_fkp"]
        weight_noz = tbdata["weight_noz"]
        weight_cp = tbdata["weight_cp"]
        weight_sdc = tbdata["weight_sdc"]
        weight = weight_fkp*weight_sdc*(weight_noz+weight_cp-1)
    except KeyError:
        weight = tbdata["weight"]
    hdulist.close()

    # Write to ROOT ntupple
    root_file = root_open("randoms_DR9_CMASS_North.root", "recreate")
    tree = Tree("data_pol", model=Galaxy)

    for i in xrange(ra.size):
        tree.phi = 0.5*numpy.pi-DEG2RAD*dec[i]
        tree.theta = DEG2RAD*ra[i]
        tree.r = r[i]
        tree.w = weight[i]
        tree.fill()

    tree.Write()
    root_file.Close()


if __name__ == "__main__":
    main()