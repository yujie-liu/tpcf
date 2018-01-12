""" Script to convert .root to .fits"""

import numpy
from astropy.io import fits
from rootpy.io import root_open

DEG2RAD = numpy.pi/180.
RAD2DEG = 180./numpy.pi

def main():
    """Main"""

    # Open .ROOT
    root_file = root_open("catalog/randoms_DESI.root", "read")
    root_tree = root_file.Get("data_pol")
    root_catalog = root_tree.to_array()

    n = root_catalog.shape[0]
    ra = numpy.zeros(n)
    dec = numpy.zeros(n)
    z = numpy.zeros(n)
    weight = numpy.zeros(n)
    for i, point in enumerate(root_catalog):
        ra[i] = RAD2DEG*point[0]
        dec[i] = 90.-RAD2DEG*point[1]
        z[i] = point[2]
        weight[i] = point[3]

    # Open .FITS
    cols1 = fits.Column(name="ra", array=ra, format="D")
    cols2 = fits.Column(name="dec", array=dec, format="D")
    cols3 = fits.Column(name="z", array=z, format="D")
    cols4 = fits.Column(name="weight", array=weight, format="D")
    table = fits.BinTableHDU.from_columns([cols1, cols2, cols3, cols4])
    table.writeto("catalog/randoms_DESI.fits")

    root_file.Close()


if __name__ == "__main__":
    main()