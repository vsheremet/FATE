# -*- coding: utf-8 -*-
"""
Drifter Tracking using velocity field from FVCOM GOM3 model
2013-04-11 ver1 Runge-Kutta scheme for 2D field
2013-04-12 ver2 xy coordinates
           ver3 time dependent vel from local files

2013-05-01 ver7 curvilinear coordinates lon,lat
           RungeKutta4_lonlat, VelInterp_lonlat
2013-05-02 ver 8 multiple drifters 
2013-05-03 ver 9 added check if point is inside polygon in VelInterp_lonlat
2013-05-06 ver10 VelInterp_lonlat vel=0 if point is outside mesh
                 drifter array init position at nodes of GOM3R grid
                 NCPU= 1,ND=644   timing [s] per step:  3.5996 0.000
                 NCPU= 1,ND=10276 timing [s] per step:  53.8676 0.0048
                                  timing [s] per step:  73.2635294118 0.00176470588235
2013-05-07 ver11 with multiprocessing 
                 NCPU=16,ND=644   timing [s] per step:  0.608 1.0416   speedup 5.92
                 NCPU=16,ND=10276 timing [s] per step:  1.182 1.188    speedup 45.57
                                             per step:  1.272 1.671

2013-05-08 ver12 RungeKutta4_lonlat_opt imized with numexpr                                             
                 timing [s] per step:  1.2592 1.142 numexpr makes timing worse                                            
                 timing [s] per step:  1.2084 1.438 nonoptimized
                 timing [s] per step:  1.1976 1.478 tau2,tau6 without numexpr
                 list comprehension does not affect speed
                                  
2013-05-09 ver13 inconvexpolygon launch only drifters inside a given polygon
NCPU=16 ND=7085 62days
timing [s] per step:  1.11063801209 1.30794492948 
2h17m

NCPU=16 ND=1170 62days
timing [s] per step:  0.795285426461 1.37378777703
2013-05-10 12:26:04.821766 2013-05-10 13:38:30.361342
1:12:25.539576

dtr.py version for simultaneous multiple runs of 12 cases: 
6 contrasting years
1980,1981 2002,2003 2009,2010
2 velocity fields

12 concurrent jobs on 1CPU each: 
timing [s] per step:  65.6294157152 0.00295500335796
2013-05-10 15:06:49.091239 2013-05-11 18:20:28.857209
1 day, 3:13:39.765970
timing [s] per step:  62.9699328408 0.00302216252518
2013-05-10 15:06:51.415552 2013-05-11 17:14:25.505476
1 day, 2:07:34.089924
timing [s] per step:  65.818791135 0.00310275352586
2013-05-10 15:06:55.508860 2013-05-11 18:25:16.557422
1 day, 3:18:21.048562
timing [s] per step:  63.4765211551 0.00275352585628
2013-05-10 15:06:54.049473 2013-05-11 17:27:01.678387
1 day, 2:20:07.628914
timing [s] per step:  69.7303693754 0.00333781061115
2013-05-10 15:06:48.992175 2013-05-11 20:02:24.555273
1 day, 4:55:35.563098
timing [s] per step:  62.9926259234 0.00342511752854
2013-05-10 15:07:04.325761 2013-05-11 17:15:08.679972
1 day, 2:08:04.354211
timing [s] per step:  73.4529952989 0.00344526527871
2013-05-10 15:06:49.847238 2013-05-11 21:35:01.926681
1 day, 6:28:12.079443
timing [s] per step:  63.2800873069 0.00313633310947
2013-05-10 15:06:54.983735 2013-05-11 17:22:09.683246
1 day, 2:15:14.699511
timing [s] per step:  74.8396977837 0.00401611820013
2013-05-10 15:07:08.148741 2013-05-11 22:09:47.323444
1 day, 7:02:39.174703
timing [s] per step:  63.8027803895 0.00314304902619
2013-05-10 15:06:57.911297 2013-05-11 17:35:10.960885
1 day, 2:28:13.049588
timing [s] per step:  84.7493485561 0.00351242444594
2013-05-10 15:07:10.545832 2013-05-12 02:16:14.600824
1 day, 11:09:04.054992
timing [s] per step:  66.0839019476 0.00312290127602
2013-05-10 15:07:18.475696 2013-05-11 18:32:17.513308
1 day, 3:24:59.037612

2013-05-13 ver 14 RungeKutta4: fixed weights in virtual time steps: 1. 0.5 0.5 1.
timing [s] per step:  65.2499865682 0.00362659503022
2013-05-13 12:06:35.573750 2013-05-14 15:10:57.229217
1 day, 3:04:21.655467

best
timing [s] per step:  63.5352988583 0.00263935527199
2013-05-17 07:33:57.615238 2013-05-18 09:55:33.357433
1 day, 2:21:35.742195
worst
timing [s] per step:  70.6055809268 0.00303559435863
2013-05-17 07:33:48.580930 2013-05-18 12:51:20.168564
1 day, 5:17:31.587634

2013-05-24 dtr_v15.py RungeKutta4_lonlattime - interpolate vel fields in time

timing [s] per step:  63.8381061115 0.00327065144392
2013-05-24 12:36:37.531473 2013-05-25 15:05:43.669979
1 day, 2:29:06.138506
timing [s] per step:  79.2464607119 0.00389523169913
2013-05-24 12:37:17.609162 2013-05-25 21:29:41.319480
1 day, 8:52:23.710318


                 
@author: Vitalii Sheremet, FATE Project, 2012-2013
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import *
import multiprocessing as mp
import sys

def RungeKutta4_lonlattime(lon,lat,Grid,ua,va,uc,vc,ub,vb,tau):
    """
Use classical 4th order 4-stage Runge-Kutta algorithm 
to track particles one time step
 
    lon,lat=RungeKutta4_lonlattime(lon,lat,Grid,ua,va,ui,vi,ub,vb,tau)

    lon,lat - coordinates of an array of particles, degE, degN
    Grid - triangular grid info
    u,v  - E,N velocity field defined on the grid
    ua,va - beginning of time step
    uc,vc - interpolated at the middle of time step
    ub,vb -end of time step (next time level)
    tau - nondim time step, deg per (velocityunits*dt), in other words, v*tau -> deg
          if dt in sec, v in m/s, then tau=dt/111111.

    VelInterp_lonlat - velocity field interpolating function
           u,v=VelInterp_lonlat(lon,lat,Grid,u,v)

Vitalii Sheremet, FATE Project, 2012-2013
    """
    """    
    lon1=lon*1.;          lat1=lat*1.;        urc1,v1=VelInterp_lonlat(lon1,lat1,Grid,u,v);  
    lon2=lon+0.5*tau*urc1;lat2=lat+0.5*tau*v1;urc2,v2=VelInterp_lonlat(lon2,lat2,Grid,u,v);
    lon3=lon+0.5*tau*urc2;lat3=lat+0.5*tau*v2;urc3,v3=VelInterp_lonlat(lon3,lat3,Grid,u,v);
    lon4=lon+0.5*tau*urc3;lat4=lat+0.5*tau*v3;urc4,v4=VelInterp_lonlat(lon4,lat4,Grid,u,v);
    lon=lon+tau/6.*(urc1+2.*urc2+2.*urc3+urc4);
    lat=lat+tau/6.*(v1+2.*v2+2.*v3+v4);
    """
        
    urc1,v1=VelInterp_lonlat(lon,lat,Grid,ua,va);
    tau2 = tau*0.5
    lon2=lon+tau2*urc1;lat2=lat+tau2*v1;urc2,v2=VelInterp_lonlat(lon2,lat2,Grid,uc,vc);
    lon3=lon+tau2*urc2;lat3=lat+tau2*v2;urc3,v3=VelInterp_lonlat(lon3,lat3,Grid,uc,vc);
    lon4=lon+tau *urc3;lat4=lat+tau *v3;urc4,v4=VelInterp_lonlat(lon4,lat4,Grid,ub,vb);
    tau6 = tau/6.0
    lon=lon+tau6*(urc1+2.*urc2+2.*urc3+urc4);
    lat=lat+tau6*(v1+2.*v2+2.*v3+v4);
    return lon,lat

def RungeKutta4_lonlat(lon,lat,Grid,u,v,tau):
    """
Use classical 4th order 4-stage Runge-Kutta algorithm 
to track particles one time step
 
 
    lon,lat=RungeKutta4_lonlat(lon,lat,Grid,u,v,tau)
     
    lon,lat - coordinates of an array of particles, degE, degN
    Grid - triangular grid info
    u,v  - E,N velocity field defined on the grid
    tau - nondim time step, deg per (velocityunits*dt), in other words, v*tau -> deg
          if dt in sec, v in m/s, then tau=dt/111111.

    VelInterp_lonlat - velocity field interpolating function
           u,v=VelInterp_lonlat(lon,lat,Grid,u,v)

Vitalii Sheremet, FATE Project, 2012-2013
    """
    """    
    lon1=lon*1.;          lat1=lat*1.;        urc1,v1=VelInterp_lonlat(lon1,lat1,Grid,u,v);  
    lon2=lon+0.5*tau*urc1;lat2=lat+0.5*tau*v1;urc2,v2=VelInterp_lonlat(lon2,lat2,Grid,u,v);
    lon3=lon+0.5*tau*urc2;lat3=lat+0.5*tau*v2;urc3,v3=VelInterp_lonlat(lon3,lat3,Grid,u,v);
    lon4=lon+0.5*tau*urc3;lat4=lat+0.5*tau*v3;urc4,v4=VelInterp_lonlat(lon4,lat4,Grid,u,v);
    lon=lon+tau/6.*(urc1+2.*urc2+2.*urc3+urc4);
    lat=lat+tau/6.*(v1+2.*v2+2.*v3+v4);
    """
        
    urc1,v1=VelInterp_lonlat(lon,lat,Grid,u,v);
    tau2 = tau*0.5
    lon2=lon+tau2*urc1;lat2=lat+tau2*v1;urc2,v2=VelInterp_lonlat(lon2,lat2,Grid,u,v);
    lon3=lon+tau2*urc2;lat3=lat+tau2*v2;urc3,v3=VelInterp_lonlat(lon3,lat3,Grid,u,v);
    lon4=lon+tau *urc3;lat4=lat+tau *v3;urc4,v4=VelInterp_lonlat(lon4,lat4,Grid,u,v);
    tau6 = tau/6.0
    lon=lon+tau6*(urc1+2.*urc2+2.*urc3+urc4);
    lat=lat+tau6*(v1+2.*v2+2.*v3+v4);
    return lon,lat
    
def step(args):
    lo=args['lo'];la=args['la'];Grid=args['Grid'];u=args['u'];v=args['v'];tau=args['tau']
    lo1,la1=RungeKutta4_lonlat(lo,la,Grid,u,v,tau)
    return [lo1,la1]
    
def gen_args(los,las,Grid,u,v,tau):
    for k in range(len(los)):
        lo=los[k];la=las[k]
        yield {'lo':lo,'la':la,'Grid':Grid,'u':u,'v':v,'tau':tau}

    
def nearxy(x,y,xp,yp):
    """
i=nearxy(x,y,xp,yp)
find the closest node in the array (x,y) to a point (xp,yp)
input:
x,y - np.arrays of the grid nodes, cartesian coordinates
xp,yp - point on a plane
output:
i - index of the closest node
min_dist - the distance to the closest node
For coordinates on a sphere use function nearlonlat

Vitalii Sheremet, FATE Project
    """
    dx=x-xp
    dy=y-yp
    dist2=dx*dx+dy*dy
# dist1=np.abs(dx)+np.abs(dy)
    i=np.argmin(dist2)
    return i

def nearlonlat(lon,lat,lonp,latp):
    """
i=nearlonlat(lon,lat,lonp,latp)
find the closest node in the array (lon,lat) to a point (lonp,latp)
input:
lon,lat - np.arrays of the grid nodes, spherical coordinates, degrees
lonp,latp - point on a sphere
output:
i - index of the closest node
min_dist - the distance to the closest node, degrees
For coordinates on a plane use function nearxy

Vitalii Sheremet, FATE Project
"""
    cp=np.cos(latp*np.pi/180.)
# approximation for small distance
    dx=(lon-lonp)*cp
    dy=lat-latp
    dist2=dx*dx+dy*dy
# dist1=np.abs(dx)+np.abs(dy)
    i=np.argmin(dist2)
#    min_dist=np.sqrt(dist2[i])
    return i 

def find_kf(Grid,xp,yp):
    """
kf,lamb0,lamb1,lamb2=find_kf(Grid,xp,yp)

find to which triangle a point (xp,yp) belongs
input:
Grid - triangular grid info
xp,yp - point on a plane
output:
kf - index of the the triangle
lamb0,lamb1,lamb2 - barycentric coordinates of P in the triangle

Vitalii Sheremet, FATE Project
    """

# coordinates of the vertices
    kvf=Grid['kvf']
    x=Grid['x'][kvf];y=Grid['y'][kvf]  
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

def find_kf2(Grid,xp,yp):
    """
kf,lamb0,lamb1,lamb2=find_kf(Grid,xp,yp)

find to which triangle a point (xp,yp) belongs
input:
Grid - triangular grid info
xp,yp - point on a plane
output:
kf - index of the the triangle
lamb0,lamb1,lamb2 - barycentric coordinates of P in the triangle

Faster version than find_kf. Find the closest vertex first
and then check lamb condition only for neighboring triangles.

Vitalii Sheremet, FATE Project
    """
# find the nearest vertex    
    kv=nearxy(Grid['x'],Grid['y'],xp,yp)
# list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]

# sometimes this fails
# append the list with the nearest barycenter 
    kf=nearxy(Grid['xc'],Grid['yc'],xp,yp)
#    kkf=np.concatenate((kfv,np.array([kf])))
# and the triangles surrounding the nearest barycenter    
    kff=Grid['kff'][:,kf]
    kkf=np.concatenate((kfv,np.array([kf]),kff))

# coordinates of the vertices
    kvf=Grid['kvf'][:,kkf]
    x=Grid['x'][kvf];y=Grid['y'][kvf]  
# calculate baricentric trilinear coordinates
    A012=((x[1,:]-x[0,:])*(y[2,:]-y[0,:])-(x[2,:]-x[0,:])*(y[1,:]-y[0,:])) 
# A012 is twice the area of the whole triangle,
# or the determinant of the linear system above.
# When xc,yc is the baricenter, the three terms in the sum are equal.
# Note the cyclic permutation of the indices
    lamb0=((x[1,:]-xp)*(y[2,:]-yp)-(x[2,:]-xp)*(y[1,:]-yp))/A012
    lamb1=((x[2,:]-xp)*(y[0,:]-yp)-(x[0,:]-xp)*(y[2,:]-yp))/A012
    lamb2=((x[0,:]-xp)*(y[1,:]-yp)-(x[1,:]-xp)*(y[0,:]-yp))/A012
#    kf,=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
#    kf=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.)).flatten()
#    kf,=np.where((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.))
# kf is an index in the short list of triangles surrounding the vertex
    kf=np.argwhere((lamb0>=0.)*(lamb1>=0.)*(lamb2>=0.)).flatten()
# select only the first entry, the same triangle may enter twice
# since we appended the closest barycenter triangle
    kf=kf[0]
# return the index in the full grid     
    return kkf[kf],lamb0[kf],lamb1[kf],lamb2[kf]
    
    
def polygonal_barycentric_coordinates(xp,yp,xv,yv):
    """
Calculate generalized barycentric coordinates within an N-sided polygon.

    w=polygonal_barycentric_coordinates(xp,yp,xv,yv)
    
    xp,yp - a point within an N-sided polygon
    xv,yv - vertices of the N-sided polygon, length N
    w     - polygonal baricentric coordinates, length N,
            normalized w.sum()=1
   
Used for function interpolation:
    fp=(fv*w).sum()
    where fv - function values at vertices,
    fp the interpolated function at the point (xp,yp)
    
Vitalii Sheremet, FATE Project    
    """
    N=len(xv)   
    j=np.arange(N)
    ja=(j+1)%N # next vertex in the sequence 
    jb=(j-1)%N # previous vertex in the sequence
# area of the chord triangle j-1,j,j+1
    Ajab=np.cross(np.array([xv[ja]-xv[j],yv[ja]-yv[j]]).T,np.array([xv[jb]-xv[j],yv[jb]-yv[j]]).T) 
# area of triangle p,j,j+1
    Aj=np.cross(np.array([xv[j]-xp,yv[j]-yp]).T,np.array([xv[ja]-xp,yv[ja]-yp]).T)  

# In FVCOM A is O(1.e7 m2) .prod() may result in inf
# to avoid this scale A
    Aj=Aj/max(abs(Aj))
    Ajab=Ajab/max(abs(Ajab))
    
    w=xv*0.
    j2=np.arange(N-2)
    
    for j in range(N):
# (j2+j+1)%N - list of triangles except the two adjacent to the edge pj
# For hexagon N=6 j2=0,1,2,3; if j=3  (j2+j+1)%N=4,5,0,1
        w[j]=Ajab[j]*Aj[(j2+j+1)%N].prod()
# timing [s] per step:  1.1976 1.478
# timing [s] per step:  1.2048 1.4508 
        
    
#    w=np.array([Ajab[j]*Aj[(j2+j+1)%N].prod() for j in range(N)])
# timing [s] per step:  1.2192 1.4572
# list comprehension does not affect speed

# normalize w so that sum(w)=1       
    w=w/w.sum() 
       
    return w,Aj

    
def Veli(x,y,Grid,u,v):
    """
Velocity interpolatin function

    ui,vi=Veli(x,y,Grid,u,v)
    
    x,y - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid
    u,v - velocity field defined at the triangle baricenters
    
    """
# 1 fastest, 
# find nearest barycenter
    kf=nearxy(Grid['xc'],Grid['yc'],x,y)
# but the point may be in the neighboring triangle 
#timing [s] per step:  0.0493136494444 0.0309618651389

# 2 slower     
# find the triangle to which point x,y truely belongs
#    kf,lamb0,lamb1,lamb2=find_kf(Grid,x,y)
# by means of calculating baricentric coordinates for all triangles in the grid
#timing [s] per step:  0.482606426944 0.148569285694

# 3 fasterthan 2
# find the closest vertex and closest barycenter
# and calculate barycentric coordinates 
# in the small neighborhood of those points
#    kf,lamb0,lamb1,lamb2=find_kf2(Grid,x,y)
#timing [s] per step:  0.0725187981944 0.0322402066667


# nearest neighbor interpolation    
    ui=u[kf]
    vi=v[kf]
    
    return ui,vi
    
def Veli2(xp,yp,Grid,u,v):
    """
Velocity interpolatin function

    ui,vi=Veli(x,y,Grid,u,v)
    
    xp,yp - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid
    u,v - velocity field defined at the triangle baricenters
    
    """
    
# find the nearest vertex    
    kv=nearxy(Grid['x'],Grid['y'],xp,yp)
#    print kv
# list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
#    print kfv
    xv=Grid['xc'][kfv];yv=Grid['yc'][kfv]
    w=polygonal_barycentric_coordinates(xp,yp,xv,yv)
#    print w

# interpolation within polygon, w - normalized weights: w.sum()=1.    
    ui=(u[kfv]*w).sum()
    vi=(v[kfv]*w).sum()
        
    return ui,vi 

def VelInterp_lonlat(lonp,latp,Grid,u,v):
    """
Velocity interpolating function

    urci,vi=VelInterp_lonlat(lonp,latp,Grid,u,v)
    
    lonp,latp - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid
    u,v - velocity field defined at the triangle baricenters
    
    urci - interpolated u/cos(lat)
    vi   - interpolated v
    The Lame coefficient cos(lat) of the spherical coordinate system
    is needed for RungeKutta4_lonlat: dlon = u/cos(lat)*tau, dlat = vi*tau

    
    """
    
# find the nearest vertex    
    kv=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
#    print kv
# list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
#    print kfv
# coordinates of the (dual mesh) polygon vertices: the centers of triangle faces
    lonv=Grid['lonc'][kfv];latv=Grid['latc'][kfv] 
    w,Aj=polygonal_barycentric_coordinates(lonp,latp,lonv,latv)
# baricentric coordinates are invariant wrt coordinate transformation (xy - lonlat), check! 

# Check whether any Aj are negative, which would mean that a point is outside the polygon.
# Otherwise, the polygonal interpolation will not be continous.
# This check is not needed if the triangular mesh and its dual polygonal mesh
# are Delaunay - Voronoi. 

# normalize subareas by the total area 
# because the area sign depends on the mesh orientation.    
    Aj=Aj/Aj.sum()
    if np.argwhere(Aj<0).flatten().size>0:
# if point is outside the polygon try neighboring polygons
#        print kv,kfv,Aj
        for kv1 in Grid['kvv'][0:Grid['nvv'][kv],kv]:
            kfv1=Grid['kfv'][0:Grid['nfv'][kv1],kv1]
            lonv1=Grid['lonc'][kfv1];latv1=Grid['latc'][kfv1] 
            w1,Aj1=polygonal_barycentric_coordinates(lonp,latp,lonv1,latv1)
            Aj1=Aj1/Aj1.sum()
            if np.argwhere(Aj1<0).flatten().size==0:
                w=w1;kfv=kfv1;kv=kv1;Aj=Aj1
#                print kv,kfv,Aj

# Now there should be no negative w
# unless the point is outside the triangular mesh
    if np.argwhere(w<0).flatten().size>0:
#        print kv,kfv,w
        
# set w=0 -> velocity=0 for points outside 
        w=w*0.        

# interpolation within polygon, w - normalized weights: w.sum()=1.    
# use precalculated Lame coefficients for the spherical coordinates
# coslatc[kfv] at the polygon vertices
# essentially interpolate u/cos(latitude)
# this is needed for RungeKutta_lonlat: dlon = u/cos(lat)*tau, dlat = vi*tau

# In this version the resulting interpolated field is continuous, C0.
    cv=Grid['coslatc'][kfv]    
    urci=(u[kfv]/cv*w).sum()
    vi=(v[kfv]*w).sum()
        
    return urci,vi
    
def ingom3(lonp,latp,Grid):
    """
check if point is inside GOM3 mesh

    i=ingom3(lonp,latp,Grid)
    
    lonp,latp - arrays of points where the interpolated velocity is desired
    Grid - parameters of the triangular grid

    i - boolean, True if lonp,latp inside GOM3, False otherwise
    
    """

# find the nearest vertex    
    kv=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
#    print kv
# list of triangles surrounding the vertex kv    
    kfv=Grid['kfv'][0:Grid['nfv'][kv],kv]
#    print kfv
# coordinates of the (dual mesh) polygon vertices: the centers of triangle faces
    lonv=Grid['lonc'][kfv];latv=Grid['latc'][kfv] 
    w,Aj=polygonal_barycentric_coordinates(lonp,latp,lonv,latv)
# baricentric coordinates are invariant wrt coordinate transformation (xy - lonlat), check! 

# Check whether any Aj are negative, which would mean that a point is outside the polygon.
# Otherwise, the polygonal interpolation will not be continous.
# This check is not needed if the triangular mesh and its dual polygonal mesh
# are Delaunay - Voronoi. 

# normalize subareas by the total area 
# because the area sign depends on the mesh orientation.    
    Aj=Aj/Aj.sum()
    if np.argwhere(Aj<0).flatten().size>0:
# if point is outside the polygon try neighboring polygons
#        print kv,kfv,Aj
        for kv1 in Grid['kvv'][0:Grid['nvv'][kv],kv]:
            kfv1=Grid['kfv'][0:Grid['nfv'][kv1],kv1]
            lonv1=Grid['lonc'][kfv1];latv1=Grid['latc'][kfv1] 
            w1,Aj1=polygonal_barycentric_coordinates(lonp,latp,lonv1,latv1)
            Aj1=Aj1/Aj1.sum()
            if np.argwhere(Aj1<0).flatten().size==0:
                w=w1;kfv=kfv1;kv=kv1;Aj=Aj1
#                print kv,kfv,Aj

# Now there should be no negative w
# unless the point is outside the triangular mesh
    i=(w>=0.).all()
        
    return i
  
def inconvexpolygon(xp,yp,xv,yv):
    """
check if point is inside a convex polygon

    i=inconvexpolygon(xp,yp,xv,yv)
    
    xp,yp - arrays of points to be tested
    xv,yv - vertices of the convex polygon

    i - boolean, True if xp,yp inside the polygon, False otherwise
    
    """
    
    
    N=len(xv)   
    j=np.arange(N)
    ja=(j+1)%N # next vertex in the sequence 
#    jb=(j-1)%N # previous vertex in the sequence
    
    NP=len(xp)
    i=np.zeros(NP,dtype=bool)
    for k in range(NP):
        # area of triangle p,j,j+1
        Aj=np.cross(np.array([xv[j]-xp[k],yv[j]-yp[k]]).T,np.array([xv[ja]-xp[k],yv[ja]-yp[k]]).T) 
    # if a point is inside the convect polygon all these Areas should be positive 
    # (assuming the area of polygon is positive, counterclockwise contour)
        Aj /= Aj.sum()
    # Now there should be no negative Aj
    # unless the point is outside the triangular mesh
        i[k]=(Aj>0.).all()
        
    return i
  
    
def RataDie(yr,mo=1,da=1,hr=0,mi=0,se=0):
    """

RD = RataDie(yr,mo=1,da=1,hr=0,mi=0,se=0)

returns the serial day number in the (proleptic) Gregorian calendar
or elapsed time in days since 0001-01-00.

Vitalii Sheremet, SeaHorse Project, 2008-2013.
"""
#
#    yr+=(mo-1)//12;mo=(mo-1)%12+1; # this extends mo values beyond the formal range 1-12
    RD=367*yr-(7*(yr+((mo+9)//12))//4)-(3*(((yr+(mo-9)//7)//100)+1)//4)+(275*mo//9)+da-396+(hr*3600+mi*60+se)/86400.;
    return RD    
############################################################################
# Drifter Tracking Script
############################################################################
    
# FVCOM GOM3 triangular grid
"""
from pydap.client import open_url
from netCDF4 import Dataset
URL='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3'
#http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3?
#a1u[0:1:3][0:1:90414],a2u[0:1:3][0:1:90414],art1[0:1:48450],art2[0:1:48450],
#aw0[0:1:2][0:1:90414],awx[0:1:2][0:1:90414],awy[0:1:2][0:1:90414],cc_hvc[0:1:90414],
#h[0:1:48450],lat[0:1:48450],latc[0:1:90414],lon[0:1:48450],lonc[0:1:90414],
#nbe[0:1:2][0:1:90414],nbsn[0:1:10][0:1:48450],nbve[0:1:8][0:1:48450],
#nn_hvc[0:1:48450],nprocs,ntsn[0:1:48450],ntve[0:1:48450],nv[0:1:2][0:1:90414],
#partition[0:1:90414],siglay[0:1:44][0:1:48450],siglev[0:1:45][0:1:48450],
#x[0:1:48450],xc[0:1:90414],y[0:1:48450],yc[0:1:90414],z0b[0:1:90414],
#Itime[0:1:171882],Itime2[0:1:171882],Times[0:1:171882],file_date[0:1:171882],
#iint[0:1:171882],kh[0:1:171882][0:1:45][0:1:48450],
#km[0:1:171882][0:1:45][0:1:48450],kq[0:1:171882][0:1:45][0:1:48450],
#l[0:1:171882][0:1:45][0:1:48450],net_heat_flux[0:1:171882][0:1:48450],
#omega[0:1:171882][0:1:45][0:1:48450],q2[0:1:171882][0:1:45][0:1:48450],
#q2l[0:1:171882][0:1:45][0:1:48450],salinity[0:1:171882][0:1:44][0:1:48450],
#short_wave[0:1:171882][0:1:48450],temp[0:1:171882][0:1:44][0:1:48450],
#time[0:1:171882],u[0:1:171882][0:1:44][0:1:90414],ua[0:1:171882][0:1:90414],
#uwind_stress[0:1:171882][0:1:90414],v[0:1:171882][0:1:44][0:1:90414],
#va[0:1:171882][0:1:90414],vwind_stress[0:1:171882][0:1:90414],
#ww[0:1:171882][0:1:44][0:1:90414],zeta[0:1:171882][0:1:48450]

#ds=open_url(URL)                 # pydap version 
ds = Dataset(URL,'r').variables   # netCDF4 version

#xxx=ds['xxx']; np.save('gom3.xxx.npy',np.array(xxx))
a1u=ds['a1u']; np.save('gom3.a1u.npy',np.array(a1u))
a2u=ds['a2u']; np.save('gom3.a2u.npy',np.array(a2u))
art1=ds['art1']; np.save('gom3.art1.npy',np.array(art1))
art2=ds['art2']; np.save('gom3.art2.npy',np.array(art2))
aw0=ds['aw0']; np.save('gom3.aw0.npy',np.array(aw0))
awx=ds['awx']; np.save('gom3.awx.npy',np.array(awx))
awy=ds['awy']; np.save('gom3.awy.npy',np.array(awy))
cc_hvc=ds['cc_hvc']; np.save('gom3.cc_hvc.npy',np.array(cc_hvc))
    
h=ds['h']; np.save('gom3.h.npy',np.array(h))

lat=ds['lat']; np.save('gom3.lat.npy',np.array(lat))
lon=ds['lon']; np.save('gom3.lon.npy',np.array(lon))
latc=ds['latc']; np.save('gom3.latc.npy',np.array(latc))
lonc=ds['lonc']; np.save('gom3.lonc.npy',np.array(lonc))

nbe=ds['nbe']; np.save('gom3.nbe.npy',np.array(nbe))
nbsn=ds['nbsn']; np.save('gom3.nbsn.npy',np.array(nbsn))
nbve=ds['nbve']; np.save('gom3.nbve.npy',np.array(nbve))
nn_hvc=ds['nn_hvc']; np.save('gom3.nn_hvc.npy',np.array(nn_hvc))
nprocs=ds['nprocs']; np.save('gom3.nprocs.npy',np.array(nprocs))
ntsn=ds['ntsn']; np.save('gom3.ntsn.npy',np.array(ntsn))
ntve=ds['ntve']; np.save('gom3.ntve.npy',np.array(ntve))
nv=ds['nv']; np.save('gom3.nv.npy',np.array(nv))
partition=ds['partition']; np.save('gom3.partition.npy',np.array(partition))
siglay=ds['siglay']; np.save('gom3.siglay.npy',np.array(siglay))
siglev=ds['siglev']; np.save('gom3.siglev.npy',np.array(siglev))

x=ds['x']; np.save('gom3.x.npy',np.array(x))
xc=ds['xc']; np.save('gom3.xc.npy',np.array(xc))
y=ds['y']; np.save('gom3.y.npy',np.array(y))
yc=ds['yc']; np.save('gom3.yc.npy',np.array(yc))
"""    
    
x=np.load('gom3.x.npy')
y=np.load('gom3.y.npy')
xc=np.load('gom3.xc.npy')
yc=np.load('gom3.yc.npy')

lon=np.load('gom3.lon.npy')
lat=np.load('gom3.lat.npy')
lonc=np.load('gom3.lonc.npy')
latc=np.load('gom3.latc.npy')

# precalculate Lame coefficients for the spherical coordinates
coslat=np.cos(lat*np.pi/180.)
coslatc=np.cos(latc*np.pi/180.)


#nv: Array of 32 bit Integers [three = 0..2][nele = 0..90414] 
#long_name: nodes surrounding element
#standard_name: face_node_connectivity
#start_index: 1
nv=np.load('gom3.nv.npy')
nv-=1 # convert from FORTRAN to python 0-based indexing
#kvf=nv

#nbe: Array of 32 bit Integers [three = 0..2][nele = 0..90414] 
# long_name: elements surrounding each element
nbe=np.load('gom3.nbe.npy')
nbe-=1 # convert from FORTRAN to python 0-based indexing
#kff=nbe

#nbsn: Array of 32 bit Integers [maxnode = 0..10][node = 0..48450]
#long_name: nodes surrounding each node
 # list of nodes surrounding a given node, 1st and last entries identical to make a closed loop
nbsn=np.load('gom3.nbsn.npy')
nbsn-=1 # convert from FORTRAN to python 0-based indexing
#kvv=nbsn

#ntsn: Array of 32 bit Integers [node = 0..48450]
#long_name: #nodes surrounding each node
 # the number of nodes surrounding a given node + 1, because 1st and last entries identical to make a closed loop
ntsn=np.load('gom3.ntsn.npy')
#nvv=ntsn

#nbve: Array of 32 bit Integers [maxelem = 0..8][node = 0..48450] 
#long_name: elems surrounding each node
# list of elements surrounding a given node, 1st and last entries identical to make a closed loop
nbve=np.load('gom3.nbve.npy')
nbve-=1 # convert from FORTRAN to python 0-based indexing
#kfv=nbve

#ntve: Array of 32 bit Integers [node = 0..48450] 
#long_name: #elems surrounding each node
# the number of elements surrounding a given node + 1, because 1st and last entries identical to make a closed loop
ntve=np.load('gom3.ntve.npy')
#nfv=ntve

Grid={'x':x,'y':y,'xc':xc,'yc':yc,'lon':lon,'lat':lat,'lonc':lonc,'latc':latc,'coslat':coslat,'coslatc':coslatc,'kvf':nv,'kff':nbe,'kvv':nbsn,'nvv':ntsn,'kfv':nbve,'nfv':ntve}

#u=np.load('20010101000000_ua.npy').flatten()
#v=np.load('20010101000000_va.npy').flatten()
#u=np.load('19970401000000_ua.npy').flatten()
#v=np.load('19970401000000_va.npy').flatten()

FN0='/home/vsheremet/FATE/GOM3_DATA/'

"""
lond0=np.arange(-69.,-66.,0.25)
latd0=np.arange(41.,43.,0.25)

lond0=np.arange(-70.4,-70.1,0.2)
latd0=np.arange(41.4,41.55,0.2)

#lond0=np.array([-66.])
#latd0=np.array([ 42.])

llond0,llatd0=np.meshgrid(lond0,latd0)
llond0=llond0.flatten()
llatd0=llatd0.flatten()
ND=llond0.size
"""

# rectangular grid that completely covers gom3
# and is aligned with 32x32 front detection subwindows
gom3r_lon=np.load('gom3r.lon.npy')
gom3r_lat=np.load('gom3r.lat.npy')
MS=2
#subsample every 2nd node
#gom3r_llon,gom3r_llat=np.meshgrid(gom3r_lon[::2],gom3r_lat[::2])
gom3r_llon,gom3r_llat=np.meshgrid(gom3r_lon[::MS],gom3r_lat[::MS])
RSHAPE=gom3r_llon.shape
# deploy drifters at nodes of GOM3R grid
llond0=gom3r_llon.flatten()
llatd0=gom3r_llat.flatten()
ND=llond0.size
print 'ND =',ND

LArea='GB150m'
if LArea=='GOM3':
#select only points inside GOM3 mesh
    i=np.zeros(ND,dtype=bool)
    for k in range(ND):
        i[k]=ingom3(llond0[k],llatd0[k],Grid)
    llondx=llond0[np.argwhere(i==False).flatten()]
    llatdx=llatd0[np.argwhere(i==False).flatten()]
    llond0=llond0[np.argwhere(i==True).flatten()]
    llatd0=llatd0[np.argwhere(i==True).flatten()]
    ND=llond0.size
    print 'ND =',ND

elif LArea=='GB150m':
# select only points inside GB150m Area
    GB150m=np.load('GB150m.npy')
    i=inconvexpolygon(llond0,llatd0,GB150m[:,0],GB150m[:,1])
    llond0=llond0[np.argwhere(i).flatten()]
    llatd0=llatd0[np.argwhere(i).flatten()]
    ND=llond0.size
    print 'ND =',ND


#> 1977, 1978
#> 1980, 1981
#> 2002, 2003
#> 2009, 2010

# 1986,1987


YEAR=sys.argv[1]
year=int(YEAR)
t0=RataDie(year,2,12)
FTS=YEAR+'0212'
#t1=RataDie(1980,4,2)

# velocity depth code: 0 - surface; a - depth averaged
D=sys.argv[2]

t1=t0+62
dtday=1./24.
tt=np.arange(t0,t1+dtday,dtday)
NT=len(tt)
#NT=175
lont=np.zeros((NT,ND))
latt=np.zeros((NT,ND))
dt=60*60.
tau=dt/111111. # deg per (velocityunits*dt)
# dt in seconds
# vel units m/s
# in other words v*tau -> deg 

# initial positions
lont[0,:]=llond0
latt[0,:]=llatd0

NCPU=1
#NCPU=4
 
tic1=datetime.now()
tic=os.times()


#time dependent u,v
kt=0
tRD=tt[kt]
tn=np.round(tRD*24.)/24.
ti=datetime.fromordinal(int(tn))
YEAR=str(ti.year)
MO=str(ti.month).zfill(2)
DA=str(ti.day).zfill(2)
hr=(tn-int(tn))*24
HR=str(int(np.round(hr))).zfill(2)            
TS=YEAR+MO+DA+HR+'0000'
print TS

FNU=FN0+'GOM3_'+YEAR+'/'+'u'+D+'/'+TS+'_u'+D+'.npy'
FNV=FN0+'GOM3_'+YEAR+'/'+'v'+D+'/'+TS+'_v'+D+'.npy'
u1=np.load(FNU).flatten()
v1=np.load(FNV).flatten()

for kt in range(NT-1):
#    print kt
    
# time dependent u,v at current time level (from previous step)    
    u0=u1*1.0
    v0=v1*1.0    
    
#time dependent u,v at next time level
    tRD=tt[kt+1]
    tn=np.round(tRD*24.)/24.
    ti=datetime.fromordinal(int(tn))
    YEAR=str(ti.year)
    MO=str(ti.month).zfill(2)
    DA=str(ti.day).zfill(2)
    hr=(tn-int(tn))*24
    HR=str(int(np.round(hr))).zfill(2)            
    TS=YEAR+MO+DA+HR+'0000'
    print TS
    
    FNU=FN0+'GOM3_'+YEAR+'/'+'u'+D+'/'+TS+'_u'+D+'.npy'
    FNV=FN0+'GOM3_'+YEAR+'/'+'v'+D+'/'+TS+'_v'+D+'.npy'
    u1=np.load(FNU).flatten()
    v1=np.load(FNV).flatten()

# velocity at the middle of time step, linear interpolation
    ui=(u0+u1)*0.5
    vi=(v0+v1)*0.5

#    if NCPU==1:    
    for kd in range(ND):
# for each drifter make one time step using classic 4th order Runge-Kutta method        
        lont[kt+1,kd],latt[kt+1,kd]=RungeKutta4_lonlattime(lont[kt,kd],latt[kt,kd],Grid,u0,v0,ui,vi,u1,v1,tau)
#    else:
#        los=lont[kt,:];las=latt[kt,:]
#        p = mp.Pool(processes=NCPU)
#        lolas1=p.map(step,gen_args(los,las,Grid,u,v,tau))
#        p.close()
#        p.join()
#        lolas1=np.array(lolas1)
#        lont[kt+1,:]=lolas1[:,0];latt[kt+1,:]=lolas1[:,1]
     
toc=os.times()
print 'timing [s] per step: ', (toc[0]-tic[0])/NT,(toc[1]-tic[1])/NT
toc1=datetime.now()
print tic1,toc1
print toc1-tic1

#np.save('drifttrack.lont.npy',lont)
#np.save('drifttrack.latt.npy',latt)
np.save('dtr_'+FTS+'_'+D+'_lont.npy',lont)
np.save('dtr_'+FTS+'_'+D+'_latt.npy',latt)


