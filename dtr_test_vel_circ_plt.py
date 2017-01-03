# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:36:52 2014

@author: vsheremet
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import *
import multiprocessing as mp
import sys

from get_fvcom_gom3_grid import get_fvcom_gom3_grid
Grid=get_fvcom_gom3_grid('disk')

def get_uvcircle(Grid):
    """
defines velocity field for circular motion test

    u,v=get_uv_circle(Grid)

    Grid - parameters of the triangular grid
    u,v - velocity field defined at the triangle baricenters
    
    """
    
# center of circulation
    loc=-67.5;lac=41.5; 
    dx=(Grid['lonc']-loc)*Grid['coslatc']
    dy=(Grid['latc']-lac)
    di=np.sqrt(dx*dx+dy*dy)
    an=np.angle(dx+1j*dy)
# velocity is linearly increasing with distance 
# 0.1 m/s at 1 deg distance away from center   
# cyclonic gyre    
    u=-0.1*di*np.sin(an)
    v= 0.1*di*np.cos(an)
# adjust the velocity so that the rotation will be perfect 
# on lon-lat plane
    u=u*Grid['coslatc']/np.cos(lac*np.pi/180)    
    
    return u,v

#u,v=get_uvcircle(Grid)
u=np.load('u_circ.npy')
v=np.load('v_circ.npy')

plt.figure()
MS=100 # subsampling
plt.quiver(Grid['lonc'][::MS],Grid['latc'][::MS],u[::MS],v[::MS]);
plt.title('u,v')
plt.show()
