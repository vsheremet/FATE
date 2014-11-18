# -*- coding: utf-8 -*-
"""
FATE Project 2012-2015
@author: Vitalii Sheremet, vsheremet@whoi.edu

v2: issue with acuracy drifter at edge,
    tau can be close to zero, resulting in stepping
    back to the previous triangle and next in all negative taus
    
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import *
import multiprocessing as mp
import sys

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

def find_kf_lonlat2(Grid,lonp,latp):
    """
kf,lamb0,lamb1,lamb2=find_kf_lonlat2(Grid,lonp,latp)

find to which triangle a point (lonp,latp) belongs
input:
Grid - triangular grid info
lonp,latp - point on a plane
output:
kf - index of the the triangle
lamb0,lamb1,lamb2 - barycentric coordinates of P in the triangle

This method is approximate, valid only for small spherical triangles.
The metric coefficient is set to unity.
lambs are not strictly correct, triangles are stretched in longitude


derived from find_kf
drived from find_kf_lonlat

Vitalii Sheremet, FATE Project
    """
#    cp=np.cos(latp*np.pi/180.)
    cp=1.
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

lonp=-68. # two triangles then negative taus
lonp=-67.8 # infinite loop (vonvergence front)
latp= 41.7

# example: frontal convergence
# check R4 and R8 precision issues: position on the edge
 
lonp=-67.85 
latp= 41.69

lonp=-68.596
latp= 41.973


dt=60*60*24
dt=200000.
print 'dt',dt

FNU='201004m_ua.npy'
FNV='201004m_va.npy'
u=np.load(FNU).flatten()
v=np.load(FNV).flatten()

#lonq,latq,kfq=tri_trace_lonlat(lonp,latp,Grid,u,v,dt)

#def tri_trace_lonlat(lonp,latp,Grid,u,v,dt):
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
kf,lamb0,lamb1,lamb2=find_kf_lonlat2(Grid,lonp,latp)
 
t=0. 
print 't',t
plt.figure()

while t<dt:

    print 'kf',kf
# introduce local coordinate system with reference to the triangle center lonc,latc
    cc=Grid['coslatc'][kf]
    dpm=90./10000000. # degrees per meter equator to pole 10000km
# velocity field in degrees
    ud=u[kf]/cc*dpm
    vd=v[kf]*dpm
#        print 'ud,vd',ud,vd

    print 'lonp,latp',lonp,latp
    print 'ud,vd',ud,vd
    
# track within the triangle is given parametrically
# lon = lonp + ud*tau
# lat = latp + vd*tau

    
    taus=np.zeros(3)

# neighboring triangles    
    kff=Grid['kff'][:,kf].flatten() 
    print 'kff',kff
# vertices of the kf triangle
    kvf=Grid['kvf'][:,kf].flatten()
# one of 3 neightbor triangles
    i=0    
    kff1=kff[i]
    # vertices of the neighbor triangle
    kvf1=Grid['kvf'][:,kff1].flatten()
    # common vertices of the common edge
    kve=list(set(kvf) & set(kvf1))
    # coordinates of the common vertices
    lona=Grid['lon'][kve[0]]
    lata=Grid['lat'][kve[0]]
    lonb=Grid['lon'][kve[1]]
    latb=Grid['lat'][kve[1]]
    print 'lona,lata,lonb,latb',lona,lata,lonb,latb
    tau=((latp-lata)/(latb-lata)-(lonp-lona)/(lonb-lona))/(ud/(lonb-lona)-vd/(latb-lata))
    taus[i]=tau
# one of 3 neightbor triangles
    i=1    
    kff1=kff[i]
    # vertices of the neighbor triangle
    kvf1=Grid['kvf'][:,kff1].flatten()
    # common vertices of the common edge
    kve=list(set(kvf) & set(kvf1))
    # coordinates of the common vertices
    lona=Grid['lon'][kve[0]]
    lata=Grid['lat'][kve[0]]
    lonb=Grid['lon'][kve[1]]
    latb=Grid['lat'][kve[1]]
    print 'lona,lata,lonb,latb',lona,lata,lonb,latb
    tau=((latp-lata)/(latb-lata)-(lonp-lona)/(lonb-lona))/(ud/(lonb-lona)-vd/(latb-lata))
    taus[i]=tau
# one of 3 neightbor triangles
    i=2    
    kff1=kff[i]
    # vertices of the neighbor triangle
    kvf1=Grid['kvf'][:,kff1].flatten()
    # common vertices of the common edge
    kve=list(set(kvf) & set(kvf1))
    # coordinates of the common vertices
    lona=Grid['lon'][kve[0]]
    lata=Grid['lat'][kve[0]]
    lonb=Grid['lon'][kve[1]]
    latb=Grid['lat'][kve[1]]
    print 'lona,lata,lonb,latb',lona,lata,lonb,latb
    tau=((latp-lata)/(latb-lata)-(lonp-lona)/(lonb-lona))/(ud/(lonb-lona)-vd/(latb-lata))
    taus[i]=tau
  
#    kf,=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
    i=[0,1,2,0] # to plot a closed contour
    plt.plot(Grid['lon'][kvf][i],Grid['lat'][kvf][i],'r.-')
    plt.plot(Grid['lonc'][kf],Grid['latc'][kf],'r+')
    plt.plot(lonp,latp,'g^')
    
    print 'taus',taus
# among three intersections,  pick the first positive.
# ie - index of the edge 0,1,2 in the triangle
    ip=np.argwhere(taus>0.)
    try:
        ie=ip[np.argmin(taus[ip])]
    except:
        print 'all taus negative'
        plt.show()    
        sys.exit()
            
#    tau=taus[taus>0.].min()
    tau=taus[ie]
#        tau=taus[ie].flatten()
    print 'tau',tau
    kfq=kff[ie]
    lonq=lonp+ud*tau
    latq=latp+vd*tau 
    plt.plot(lonq,latq,'bv')
    print 'lonq,latq',lonq,latq
            
    # drifter enters triangle kfq
    print 'kfq',kfq
    print 'kvfq',Grid['kvf'][:,kfq]     
    kfq2,lamb0,lamb1,lamb2=find_kf_lonlat2(Grid,lonq,latq)
    print 'kfq2',kfq2

    t=t+tau
    print 't',t
    print '--'
    lonp=lonq*1.
    latp=latq*1.
    kf=kfq*1
 

plt.show()    



#return lonq,latq,kfq

