import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
deg = np.pi / 180.0

ddata = fits.open('data.fits')
ralist = ddata[1].data['RA']
declist = ddata[1].data['DEC']
zlist = np.array(ddata[1].data['Z'])
ndata = len(zlist)
thlist = np.array([np.pi/2.0 - dec*deg for dec in declist])
phlist = np.array([ra*deg for ra in ralist])

dveclist = np.array([hp.ang2vec(thlist[i],phlist[i]) for i in range(0,ndata)])

seplist = np.array([])
for i in range(0,ndata):
    for j in range(0,i):
        cth12 = np.dot(dveclist[i],dveclist[j])
        th12 = np.arccos(cth12)
        nsep = [th12]
        seplist = np.append(seplist,nsep)