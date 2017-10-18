""" Get boundaries of given catalogs """

import sys
import numpy
from astropy.io import fits


def main():
    """ Main """
    fname_list = sys.argv[1:]

    # open .fits file
    ra_min = 100000
    ra_max = -100000
    dec_min = 100000
    dec_max = -100000
    z_min = 100000
    z_max = -100000
    for i, fname in enumerate(fname_list):

        hdulist = fits.open(fname)
        tbdata = hdulist[1].data
        ra = tbdata['ra']
        dec = tbdata['dec']
        z = tbdata['z']
        hdulist.close()

        print("Catalog {}".format(i))
        print("RA:  [{}, {}]".format(ra.min(), ra.max()))
        print("DEC: [{}, {}]".format(dec.min(), dec.max()))
        print("Z:   [{}, {}]".format(z.min(), z.max()))

        ra_min = min(ra.min(), ra_min)
        ra_max = max(ra.max(), ra_max)
        dec_min = min(dec.min(), dec_min)
        dec_max = max(dec.max(), dec_max)
        z_min = min(z.min(), z_min)
        z_max = max(z.max(), z_max)

    print("All:")
    print("RA:  [{}, {}]".format(ra_min, ra_max))
    print("DEC: [{}, {}]".format(dec_min, dec_max))
    print("Z:   [{}, {}]".format(z_min, z_max))


if __name__ == "__main__":
    main()