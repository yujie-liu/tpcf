import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import healpy as hp
deg = np.pi / 180.0
ntarg = 5000
nside = 1024
npix = hp.nside2npix(nside)
parea = hp.nside2pixarea(nside)
tarea = 2.0*np.pi*(1.0-np.cos(1.4*deg))
ptarg = np.maximum(int(ntarg * parea/tarea),1)

pixids = np.arange(npix)
cdata = np.load('cmplarr' + str(nside) +'.npz')
cmap = cdata.f.CMPLTNSS
angmap = cmap*ptarg
ddata = fits.open('data.fits')
ralist = ddata[1].data['RA']
declist = ddata[1].data['DEC']
zlist = np.array(ddata[1].data['Z'])
ndata = len(zlist)
thlist = np.array([np.pi/2.0 - dec*deg for dec in declist])
phlist = np.array([ra*deg for ra in ralist])

nzcut = (cmap != 0.)
cangmap = angmap[nzcut]
cpixids = pixids[nzcut]
ncpix = len(cpixids)
veclist = np.array([hp.pix2vec(nside,pixid) for pixid in cpixids])
dveclist = np.array([hp.ang2vec(thlist[i],phlist[i]) for i in range(0,ndata)])

seplist = np.array([])
for i in range(0,ncpix):
    for j in range(0,ndata):
        cth12 = np.dot(veclist[i],dveclist[j])
        th12 = np.arccos(cth12)
        nsep = [th12] * int(cangmap[i])
        seplist = np.append(seplist,nsep)