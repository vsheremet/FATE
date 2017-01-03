# -*- coding: utf-8 -*-
"""
FATE Project 2012-2015
@author: Vitalii Sheremet, vsheremet@whoi.edu
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import *
import multiprocessing as mp
import sys

def tri_trace_lonlat(lonp,latp,Grid,u,v,dt):
    """
Trace a drifter track over a triangular grid.    
Use constant velocity in each triangle to trace the drifter track
during one time step. Mostly applicable for monthly or daily velocity fields
when a drifter traverses several triangles during a time step.

usage: lonq,latq=tri_trace_lonlat(lonp,latp,Grid,u,v,dt)

lonp,latp - lon, lat of a drifter at the beginning of time step 
Grid - arrays specifying the triangular grid (dictionary):
    Grid{'lon'},Grid{'lat'} - grid nodes (vertices of triangles)
    Grid{'lonc'},Grid{'latc'} - grid velocity nodes (centers of triangles),
        strictly speaking, circumcenters of Voronoi-Delaunay triangulation.
    Gid{'kvf'},Grid{'kfv'},etc grid connectivity information.

u,v - velocity fields defined on a triangular grid
dt - time step during which to trace the drifter.

lonq,latq - lon,lat of the drifter at the end of time step returned.     
    

This method is approximate, valid only for small spherical triangles.
The metric (Lame) coefficient is evaluated at the triangle center.
    
    """
    # find the triangle to which the current drifter position belongs
    kf,lamb0,lamb1,lamb2=find_kf_lonlat(Grid,lonp,latp)
    
    print 'kf',kf
# coordinates of the vertices of the kf triangle
    kvf=Grid['kvf'][:,kf]
# introduce local coordinate system with reference to the triangle center lonc,latc
    cc=Grid['coslatc'][kf]
    dpm=90./10000000. # degrees per meter equator to pole 10000km
# velocity field in degrees
    ud=u[kf]/cc*dpm
    vd=v[kf]*dpm
    print 'ud,vd',ud,vd
    
# track within the triangle is given parametrically
# lon = lonp + ud*tau
# lat = latp + vd*tau
    
    taus=np.zeros(3)
# find intersection with each triangle edge
# edge 2
    lona=Grid['lon'][kvf[0]]
    lata=Grid['lat'][kvf[0]]
    lonb=Grid['lon'][kvf[1]]
    latb=Grid['lat'][kvf[1]]
    tau=((latp-lata)/(latb-lata)-(lonp-lona)/(lonb-lona))/(ud/(lonb-lona)-vd/(latb-lata))
    print 'tau',tau
    taus[2]=tau
    
# edge 0
    lona=Grid['lon'][kvf[1]]
    lata=Grid['lat'][kvf[1]]
    lonb=Grid['lon'][kvf[2]]
    latb=Grid['lat'][kvf[2]]
    tau=((latp-lata)/(latb-lata)-(lonp-lona)/(lonb-lona))/(ud/(lonb-lona)-vd/(latb-lata))
    print 'tau',tau
    taus[0]=tau
# edge 1
    lona=Grid['lon'][kvf[2]]
    lata=Grid['lat'][kvf[2]]
    lonb=Grid['lon'][kvf[0]]
    latb=Grid['lat'][kvf[0]]
    tau=((latp-lata)/(latb-lata)-(lonp-lona)/(lonb-lona))/(ud/(lonb-lona)-vd/(latb-lata))
    print 'tau',tau
    taus[1]=tau

#    kf,=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    
    plt.figure(1)
    plt.plot(Grid['lon'][kvf],Grid['lat'][kvf],'r.')
    plt.plot(Grid['lonc'][kf],Grid['latc'][kf],'r+')
    plt.plot(lonp,latp,'k*')
    
    tau=taus[0]
    lonq=lonp+ud*tau
    latq=latp+vd*tau 
    plt.plot(lonq,latq,'ro')
    tau=taus[1]
    lonq=lonp+ud*tau
    latq=latp+vd*tau 
    plt.plot(lonq,latq,'go')
    tau=taus[2]
    lonq=lonp+ud*tau
    latq=latp+vd*tau 
    plt.plot(lonq,latq,'bo')
    
    plt.show()    
    

    
    return lonq,latq

def find_kf_lonlat(Grid,lonp,latp):
    """
kf,lamb0,lamb1,lamb2=find_kf_lonlat(Grid,lonp,latp)

find to which triangle a point (lonp,latp) belongs
input:
Grid - triangular grid info
lonp,latp - point on a plane
output:
kf - index of the the triangle
lamb0,lamb1,lamb2 - barycentric coordinates of P in the triangle

This method is approximate, valid only for small spherical triangles.
The metric coefficient is evaluated at P.

derived from find_kf

Vitalii Sheremet, FATE Project
    """
    cp=np.cos(latp*np.pi/180.)
    xp=lonp*cp;yp=latp
# coordinates of the vertices
    kvf=Grid['kvf']
    x=Grid['lon'][kvf]*cp;y=Grid['lat'][kvf]  
# calculate baricentric trilinear coordinates
    A012=((x[1,:]-x[0,:])*(y[2,:]-y[0,:])-(x[2,:]-x[0,:])*(y[1,:]-y[0,:])) 
# A012 is twice the area of the whole triangle,
# or the determinant of the linear system above.
# When xc,yc is the baricenter, the three terms in the sum are equal.
# Note the cyclic permutation of the indices
    lamb0=((x[1,:]-xp)*(y[2,:]-yp)-(x[2,:]-xp)*(y[1,:]-yp))/A012
    lamb1=((x[2,:]-xp)*(y[0,:]-yp)-(x[0,:]-xp)*(y[2,:]-yp))/A012
    lamb2=((x[0,:]-xp)*(y[1,:]-yp)-(x[1,:]-xp)*(y[0,:]-yp))/A012
    kf,=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
#    kf=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.)).flatten()
#    kf,=np.where((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    return kf,lamb0[kf],lamb1[kf],lamb2[kf]

##########################################################
from get_fvcom_gom3_grid import get_fvcom_gom3_grid
Grid=get_fvcom_gom3_grid('disk')

lonp=-69.
latp=42.
dt=60*60*24

FNU='/home/vsheremet/FATE/GOM3_DATA/GOM3_monthly/ua/201004m_ua.npy'
FNV='/home/vsheremet/FATE/GOM3_DATA/GOM3_monthly/va/201004m_va.npy'
u=np.load(FNU).flatten()
v=np.load(FNV).flatten()

lonq,latq=tri_trace_lonlat(lonp,latp,Grid,u,v,dt)