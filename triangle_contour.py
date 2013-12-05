# -*- coding: utf-8 -*-
"""
pl_bathy_contour_triangular_grid.py

@author: vsheremet
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import *
import sys
from get_fvcom_gom3_grid import get_fvcom_gom3_grid

Grid=get_fvcom_gom3_grid('disk2')    
# FVCOM GOM3 triangular grid
#Grid={'x':x,'y':y,'xc':xc,'yc':yc,'lon':lon,'lat':lat,'lonc':lonc,'latc':latc,'coslat':coslat,'coslatc':coslatc,'kvf':nv,'kff':nbe,'kvv':nbsn,'nvv':ntsn,'kfv':nbve,'nfv':ntve}
lon=Grid['lon'];lat=Grid['lat'];lonc=Grid['lonc'];latc=Grid['latc']
kvf=Grid['kvf']

# bathymetry
h=np.load('gom3.h.npy')

hc=100. # plot countour 100m

NF=len(lonc) # number of triangles (grid cells) 
NV=len(lon)  # number of vertices (grid nodes)

#plt.figure()
fc=hc
ns=0
xa=np.array(NF);ya=np.array(NF);
xb=np.array(NF);yb=np.array(NF);
for kf in [1021]: #range(NF):
    kv=kvf[:,kf] # three vertices of a triangle
    xv=lon[kv];yv=lat[kv];fv=h[kv] # positions and values at the three vertices
    if (fc >= fv.min()) and (fc <= fv.max()): # contour passes through the triangle
        
        
    
