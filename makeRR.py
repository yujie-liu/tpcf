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
ptarg = int(ntarg * parea/tarea)
if ptarg == 0:
    ptarg = 1

pixids = np.arange(npix)
cdata = np.load('cmplarr' + str(nside) +'.npz')
cmap = cdata.f.CMPLTNSS
angmap = cmap*ptarg

nzcut = (cmap != 0.)
cangmap = angmap[nzcut]
cpixids = pixids[nzcut]
ncpix = len(cpixids)
veclist = np.array([hp.pix2vec(nside,pixid) for pixid in cpixids])

seplist = np.array([])
for i in range(0,ncpix):
    for j in range(0,i):
        cth12 = np.dot(veclist[i],veclist[j])
        th12 = np.arccos(cth12)
        nsep = [th12]*(int(cangmap[i])*int(cangmap[j])
        seplist = np.append(seplist,nsep)