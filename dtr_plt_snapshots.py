# -*- coding: utf-8 -*-
"""
plot drifter tracks output from dtr
@author: Vitalii Sheremet, FATE Project, 2012-2013
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import *

# www.ngdc.noaa.gov
# world vector shoreline ascii
FNCL='necscoast_worldvec.dat'
# lon lat pairs
# segments separated by nans
"""
nan nan
-77.953942	34.000067
-77.953949	34.000000
nan nan
-77.941035	34.000067
-77.939568	34.001241
-77.939275	34.002121
-77.938688	34.003001
-77.938688	34.003881
"""
CL=np.genfromtxt(FNCL,names=['lon','lat'])

from get_fvcom_gom3_grid import get_fvcom_gom3_grid
Grid=get_fvcom_gom3_grid('disk')

"""
DATADIR='DriftTrackData/GOM3init/0212/'
FTS='20040212'
FVS='0'
#lont=np.load('drifttrack.lont.npy')
#latt=np.load('drifttrack.latt.npy')
lont=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_lont.npy')
latt=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_latt.npy')
"""

DATADIR='/home/vsheremet/u/'
FTS='20050212'
FVS='0'

"""
RUNCODE='_ShBr_4d_1h'
lont=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_lont'+RUNCODE+'.npy')
latt=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_latt'+RUNCODE+'.npy')

RUNCODE='_ShBr_4d_12min'
'dtr_20050212_0_latt_ShBr_4d_12min.py'
lont=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_lont'+RUNCODE+'.npy')
latt=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_latt'+RUNCODE+'.npy')
"""
FTS='20050202'
FVS='0'
RUNCODE='_ShBr_14d'
lont=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_lont'+RUNCODE+'.npy')
latt=np.load(DATADIR+'dtr_'+FTS+'_'+FVS+'_latt'+RUNCODE+'.npy')

(NT,ND)=lont.shape

"""
skip this - the output is already daily
lont=lont[::24,:]
latt=latt[::24,:]
(NT,ND)=lont.shape
"""

# plot drifter positions every 1 days
for kt in range(0,NT,7):
    plt.figure();
    plt.plot(Grid['lon'],Grid['lat'],'g.',Grid['lonc'],Grid['latc'],'c+',lont[kt,:],latt[kt,:],'r.');
    plt.plot(CL['lon'],CL['lat'])
    plt.title(FTS+' '+FVS+str(kt))
   
plt.show()    

