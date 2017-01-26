# -*- coding: utf-8 -*-
"""
test dtr_v25.py

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

2013-06-11 dtr_v16.py save temp data along the track (trilinear interpolation)

2013-06-13 dtr_v17.py RungeKutta4_lonlattime for multiple processors
temperature output disabled (needs to be parallelized)
           
           dtr_v18.py get_uv(tRD) put into a function
           
timing [s] per step:  9.58470114171 2.85089321692
2013-06-14 16:10:37.085921 2013-06-17 06:12:23.835650
2 days, 14:01:46.749729

2013-06-18 dtr_v19.py input init YEARMODA

timing [s] per step:  9.55428475487 2.76058428475
2013-07-12 09:27:10.222188 2013-07-15 04:57:26.887217
2 days, 19:30:16.665029

timing [s] per step:  9.59444593687 2.71593015447
2013-08-08 20:45:29.628681 2013-08-11 17:57:50.630853
2 days, 21:12:21.002172

2013-08-16 dtr_v20.py added GBext area
2013-09-05 17:58:42.340595 2013-09-05 20:26:48.248594
2:28:05.907999

to do: fix division by zero 
  Aj1=Aj1/Aj1.sum()
/home/vsheremet/u/dtr.py:419: RuntimeWarning: invalid value encountered in divide
  Ajab=Ajab/max(abs(Ajab))
/home/vsheremet/u/dtr.py:551: RuntimeWarning: divide by zero encountered in divide
  Aj1=Aj1/Aj1.sum()
timing [s] per step:  1.13877098724 1.66352585628
2013-09-11 19:21:37.367011 2013-09-11 22:11:29.328551
2:49:51.961540

timing [s] per step:  1.25442578912 1.41126930826
2013-10-13 13:00:29.877484 2013-10-13 14:57:55.880056
1:57:26.002572

timing [s] per step:  1.19682337139 1.58029550034
2013-12-01 05:04:31.306893 2013-12-01 06:59:15.397789
1:54:44.090896
         
2013-12-04 added Shelf Break region                 
@author: Vitalii Sheremet, FATE Project, 2012-2013

2014-01-28 dtr_v21.py
moved init cond to dtr_init_positions.py
moved Grid to
from get_fvcom_gom3_grid import get_fvcom_gom3_grid
Grid=get_fvcom_gom3_grid('disk')

NCPU=16 GOM3R grid MS=4 ingom3 init positions
timing [s] per step:  2.74891966759 1.39639889197
2014-01-28 10:35:58.462827 2014-01-28 14:07:48.412039
3:31:49.949212

2014-12-30 dtr_v22.py
u,v=get_uv2(tRD,D)
for time steps smaller than hour
interpolate between two hourly fields
2014-01-30 dtr_v23.py
added tt output

2014-03-11 dtr_v24.py
polygonal_barycentric_coordinates       N=2,N=1 for areas close to boundary
VelInterp_lonlat(lonp,latp,Grid,u,v)    Aj.sum()==0

see timing info at the end of the file
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import *
import multiprocessing as mp
import sys
#from pydap.client import open_url
from netCDF4 import Dataset        # netCDF4 version



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
    ub,vb - end of time step (next time level)
    tau - nondim time step, deg per (velocityunits*dt), in other words, v*tau -> deg
          if dt in sec, v in m/s, then tau=dt/111111.

    VelInterp_lonlat - velocity field interpolating function
           u,v=VelInterp_lonlat(lon,lat,Grid,u,v)

Vitalii Sheremet, FATE Project, 2012-2014
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
    lon4=lon+    tau*urc3;lat4=lat+    tau*v3;urc4,v4=VelInterp_lonlat(lon4,lat4,Grid,u,v);
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
    lo=args['lo'];la=args['la'];Grid=args['Grid'];ua=args['ua'];va=args['va'];uc=args['uc'];vc=args['vc'];ub=args['ub'];vb=args['vb'];tau=args['tau']
    lo1,la1=RungeKutta4_lonlattime(lo,la,Grid,ua,va,uc,vc,ub,vb,tau)
    return [lo1,la1]
    
def gen_args(los,las,Grid,ua,va,uc,vc,ub,vb,tau):
    for k in range(len(los)):
        lo=los[k];la=las[k]
        yield {'lo':lo,'la':la,'Grid':Grid,'ua':ua,'va':va,'uc':uc,'vc':vc,'ub':ub,'vb':vb,'tau':tau}
    
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

def find_kf_lonlat(Grid,lonp,latp):
    """
kf,lamb0,lamb1,lamb2=find_kf(Grid,lonp,latp)

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
    
    
def polygonal_barycentric_coordinates_old(xp,yp,xv,yv):
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
    AScale=max(abs(Aj))
    Aj=Aj/AScale
    Ajab=Ajab/AScale
    
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
    
    N=2 -> lenear interpolation
    N=1 -> fixed value w=1
    
Vitalii Sheremet, FATE Project    
    """
    N=len(xv)
    if N>2:
        j=np.arange(N)
        ja=(j+1)%N # next vertex in the sequence 
        jb=(j-1)%N # previous vertex in the sequence
    # area of the chord triangle j-1,j,j+1
        Ajab=np.cross(np.array([xv[ja]-xv[j],yv[ja]-yv[j]]).T,np.array([xv[jb]-xv[j],yv[jb]-yv[j]]).T) 
    # area of triangle p,j,j+1
        Aj=np.cross(np.array([xv[j]-xp,yv[j]-yp]).T,np.array([xv[ja]-xp,yv[ja]-yp]).T)  
    
    # In FVCOM A is O(1.e7 m2) .prod() may result in inf
    # to avoid this scale A
        AScale=max(abs(Aj))
        Aj=Aj/AScale
        Ajab=Ajab/AScale
        
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
        w=w/w.sum() 

# for areas close to boundary
    elif N==2:
        w=xv*0.
        w[0]=np.dot(np.array([xv[1]-xp,yv[1]-yp]).T,np.array([xv[1]-xv[0],yv[1]-yv[0]]).T)    
        w[1]=np.dot(np.array([xp-xv[0],yp-yv[0]]).T,np.array([xv[1]-xv[0],yv[1]-yv[0]]).T)
    # normalize w so that sum(w)=1       
        w=w/w.sum()
        Aj=w*0.

    elif N==1:
        w=xv*0.+1.
        Aj=w*0.
       
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

    if Aj.sum()==0.:
        w=w*0.
    else:    
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

def inpolygon(xp,yp,xv,yv):
    """
check if point is inside a polygon

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

def sh_parse_timestamp(TIMESTAMP):
    """

sh_parse_timestamp(TIMESTAMP) -> yr,mo,da,hr,mi,se

parse TIMESTAMP string and convert to yr,mo,da,hr,mi,se

Acceptable formats:
'YYYY-MM-DD HR:MI:SE'
'YYYY-MM-DDTHR:MI:SE'
'YYYY-MM-DDTHR:MI:SEZ'
'YYYY/MM/DD HR:MI:SE'
'YYYY/MM/DDTHR:MI:SE'    
'YYYY/MM/DDTHR:MI:SEZ'

If MM,DD,HR,MI,SE are omited, 
then default values are assumed 
   01,01,00,00,00, respectively.
DD,HR,MI,SE may be fractional and outside formal ranges, e.g., 
TIMESTAMP='2001-01-121.25 24.5:120.4:90.1234'
 - 2001, yearday 121 00:00:00 + 0.25d + 24.5h + 120.4m +90.1234s
TIMESTAMP='2001-01 24.5'
 - 2001-01-01 00:00:00 + 24.5h

Vitalii Sheremet, SeaHorse Project, 2008-2012
    """
    YR='0000';MO='01';DA='01';HR='00';MI='00';SE='00';TIME='00:00:00'
    TIMESTAMP=TIMESTAMP.strip()
    if TIMESTAMP[-1]=='Z':
        TIMESTAMP=TIMESTAMP[0:-1]
        
    if TIMESTAMP.find(' ')>-1:
        DATE,TIME=TIMESTAMP.split(' ')
    elif TIMESTAMP.find('T')>-1:
        DATE,TIME=TIMESTAMP.split('T')
    else:
        DATE=TIMESTAMP
            
    if DATE.find('/')>-1:    
        CS='/'
    else:
        CS='-'
        
    DATE=DATE.split(CS)
    if len(DATE)==3:
        YR=DATE[0];MO=DATE[1];DA=DATE[2]
    elif len(DATE)==2:
        YR=DATE[0];MO=DATE[1]
    elif len(DATE)==1:
        YR=DATE[0]
    else:
        print('sh_parse_timestamp: error: unknown date format')
    
    TIME=TIME.split(':')
    if len(TIME)==3:
        HR=TIME[0];MI=TIME[1];SE=TIME[2]
    elif len(TIME)==2:
        HR=TIME[0];MI=TIME[1]
    elif len(TIME)==1:
        HR=TIME[0]
    else:
        print('sh_parse_timestamp: error: unknown time format')
             
    yr=int(YR);mo=int(MO);da=float(DA);hr=float(HR);mi=float(MI);se=float(SE)
    return yr,mo,da,hr,mi,se    

def get_uv1(tRD,D):
    """
get velocity fields either from a local file or from internet

    u,v=get_uv1(tRD,D)

    tRD - time RataDie, python ordinal
    u,v - velocity fields
    D - depth code: a - avg; 0 - surf
    """

# location of velocity fields on a local disk
    FN0='../GOM3_DATA/'
    tn=np.round(tRD*24.)/24.
    ti=datetime.fromordinal(int(tn))
    YEAR=str(ti.year)
    MO=str(ti.month).zfill(2)
    DA=str(ti.day).zfill(2)
    hr=(tn-int(tn))*24
    HR=str(int(np.round(hr))).zfill(2)            
    TS=YEAR+MO+DA+HR+'0000'
    #print TS
    
    FNU=FN0+'GOM3_'+YEAR+'/'+'u'+D+'/'+TS+'_u'+D+'.npy'
    FNV=FN0+'GOM3_'+YEAR+'/'+'v'+D+'/'+TS+'_v'+D+'.npy'
    print FNU
    #print FNV
    u=np.load(FNU).flatten()
    v=np.load(FNV).flatten()
    return u,v

def get_uv2(tRD,D):
    """
get velocity fields either from a local file or from internet
interpolates linearly between two hourly fields

    u,v=get_uv2(tRD,D)

    tRD - time RataDie, python ordinal
    u,v - velocity fields
    D - depth code: a - avg; 0 - surf
    """

# location of velocity fields on a local disk
    FN0='../GOM3_DATA/'

#    tn=np.round(tRD*24.)/24.
    tn=np.floor(tRD*24.)/24.
    ti=datetime.fromordinal(int(tn))
    YEAR=str(ti.year)
    MO=str(ti.month).zfill(2)
    DA=str(ti.day).zfill(2)
    hr=(tn-int(tn))*24
    HR=str(int(np.round(hr))).zfill(2)            
    TS=YEAR+MO+DA+HR+'0000'
    #print TS
    a=(tRD-tn)*24.
    #print a
    FNU=FN0+'GOM3_'+YEAR+'/'+'u'+D+'/'+TS+'_u'+D+'.npy'
    FNV=FN0+'GOM3_'+YEAR+'/'+'v'+D+'/'+TS+'_v'+D+'.npy'
    print FNU
    #print FNV
    u0=np.load(FNU).flatten()
    v0=np.load(FNV).flatten()
   
    
    
#    tn=np.round(tRD*24.)/24.
    tn=np.floor(tRD*24.+1.)/24.
    ti=datetime.fromordinal(int(tn))
    YEAR=str(ti.year)
    MO=str(ti.month).zfill(2)
    DA=str(ti.day).zfill(2)
    hr=(tn-int(tn))*24
    HR=str(int(np.round(hr))).zfill(2)            
    TS=YEAR+MO+DA+HR+'0000'
    #print TS
    b=1.0-a
    #print b
    FNU=FN0+'GOM3_'+YEAR+'/'+'u'+D+'/'+TS+'_u'+D+'.npy'
    FNV=FN0+'GOM3_'+YEAR+'/'+'v'+D+'/'+TS+'_v'+D+'.npy'
    #print FNU
    #print FNV
    u1=np.load(FNU).flatten()
    v1=np.load(FNV).flatten()

    u=u0*b+u1*a
    v=v0*b+v1*a
    return u,v

def get_fvcom_gom3_1(tsec_MJD,lonp,latp,Grid):
    """
    D=get_fvcom_gom3_1(tsec_MJD,lon,lat,Grid)
    
    input:
    tsec_MJD - time seconds since 1858-11-17T00:00:00 (MJD=JD-2400000.5)
    lonp - longitude, degrees East
    latp - latitude, degrees North
    Grid - Grid information
    
    output:
    D - dictionary: 
    D['u']
    D['v']
    """        
    #http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3.html
    URL='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3'
    ds = Dataset(URL,'r').variables   # netCDF4 version
    #tsecFVCOM=Grid['time'][:]*86400
    tsecFVCOM=np.array(ds['time'][:])*86400
    kt=np.argmin(np.abs(tsecFVCOM-tsec_MJD))
    # find time layers 1 hour earlier and 1 hour later
    # note that some layers may have identical times (start and end of each month)
    kta=np.argmin(np.abs(tsecFVCOM-tsec_MJD+3600))
    ktb=np.argmin(np.abs(tsecFVCOM-tsec_MJD-3600))
    print 'kt,kta,ktb',kt,kta,ktb
    print 'closest Times'
    T1=ds['Times'][kt];T1=T1.tostring()
    print T1


    #find nearest triangle center
    kf,lamb0,lamb1,lamb2=find_kf_lonlat(Grid,lonp,latp)
    print 'kf',kf
    print Grid['lonc'][kf],Grid['latc'][kf]
    # find the nearest vertex    
    kv=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
    print 'kv',kv
    
    h=Grid['h'][kv] # bottom depth, m
    siglay=Grid['siglay'][:,kv] # nondim layer depth, -1.<siglay[:]<0.   45 
    siglev=Grid['siglev'][:,kv] # nondim level depth, -1.<siglev[:]<0.   46
    
    ksiglay=44 # bottom layer
    ksiglay=0 # surface layer
    D={}
    D['u']=np.array(ds['u'][kt,ksiglay,kf]) # 
    D['v']=np.array(ds['v'][kt,ksiglay,kf]) # 
    return D

def get_fvcom_gom3_ver2(tsec_MJD,lonp,latp,Grid):
    """
    D=get_fvcom_gom3_1(tsec_MJD,lon,lat,Grid)
    
    input:
    tsec_MJD - time seconds since 1858-11-17T00:00:00 (MJD=JD-2400000.5)
    lonp - longitude, degrees East
    latp - latitude, degrees North
    Grid - Grid information
    
    output:
    D - dictionary: 
    D['u']
    D['v']
    """        
    #http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3.html
    URL='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3'
    ds = Dataset(URL,'r').variables   # netCDF4 version
    tsecFVCOM=Grid['time'][:]*86400
    t#secFVCOM=np.array(ds['time'][:])*86400
    kt=np.argmin(np.abs(tsecFVCOM-tsec_MJD))
    # find time layers 1 hour earlier and 1 hour later
    # note that some layers may have identical times (start and end of each month)
    kta=np.argmin(np.abs(tsecFVCOM-tsec_MJD+3600))
    ktb=np.argmin(np.abs(tsecFVCOM-tsec_MJD-3600))
    print 'kt,kta,ktb',kt,kta,ktb
    print 'closest Times'
    T1=ds['Times'][kt];T1=T1.tostring()
    print T1


    #find nearest triangle center
    kf,lamb0,lamb1,lamb2=find_kf_lonlat(Grid,lonp,latp)
    print 'kf',kf
    print Grid['lonc'][kf],Grid['latc'][kf]
    # find the nearest vertex    
    kv=nearlonlat(Grid['lon'],Grid['lat'],lonp,latp)
    print 'kv',kv
    
    h=Grid['h'][kv] # bottom depth, m
    siglay=Grid['siglay'][:,kv] # nondim layer depth, -1.<siglay[:]<0.   45 
    siglev=Grid['siglev'][:,kv] # nondim level depth, -1.<siglev[:]<0.   46
    
    ksiglay=44 # bottom layer
    ksiglay=0 # surface layer
    D={}
    D['u']=np.array(ds['u'][kt,ksiglay,kf]) # 
    D['v']=np.array(ds['v'][kt,ksiglay,kf]) # 
    return D    

def get_uv1_server(tsec_MJD,Grid):
    """
get velocity fields from server
    get_uv1_server(tsec_MJD,Grid):
    """
    #http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3.html
    URL='http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3'
    ds = Dataset(URL,'r').variables   # netCDF4 version
    tsecFVCOM=Grid['time'][:]*86400
    #secFVCOM=np.array(ds['time'][:])*86400
    kt=np.argmin(np.abs(tsecFVCOM-tsec_MJD))
    T1=ds['Times'][kt];T1=T1.tostring()
    print T1
    
    #ksiglay=44 # bottom layer
    ksiglay=0 # surface layer
    u=np.array(ds['u'][kt,ksiglay,:]).flatten() # 
    v=np.array(ds['v'][kt,ksiglay,:]).flatten() # 
    return u,v    
    
############################################################################
NCPU=1
NCPU=1# number of CPUs to use

# FVCOM GOM3 triangular grid
############################################################
from get_fvcom_gom3_grid import get_fvcom_gom3_grid
#Grid=get_fvcom_gom3_grid('server') # latest grid info from server
Grid=get_fvcom_gom3_grid('time')   # load from dict and update only time from server

Grid=get_fvcom_gom3_grid('dict') # quick load from local dict
# same as 
#Grid=np.load('Grid.npy').item()


t=datetime(1985,7,1,12,0,0)
#t=datetime(1978,1,1,0,0,0)
print 'selected time and point'
print t
t0=datetime(1858,11,17) # MJD=JD-2400000.5
tsec_MJD=(t-t0).total_seconds()

#lon=-(68.+9./60);lat=43.+49./60; # midway between two Yarmouths
#lon=-68.1;lat=43.8
lon=-68.1202774;lat=43.80513
print lon,lat

print ''
print 'FVCOM GOM3 Dataset returns'
D=get_fvcom_gom3_1(tsec_MJD,lon,lat,Grid)
print 'u,v', D['u'],D['v']


############################################################################

lond0=-(68.+9./60);latd0=43.+49./60; # midway between two Yarmouths
td0=datetime(1985,7,1,12,0,0)

# specify the length of calculation, typically 62 days
#t1=t0+62
t1sec_MJD=(td0-t0).total_seconds()
t2sec_MJD=t1sec_MJD+1*86400


#t1=t0+1.
#t1=t0+15.
#t1=t0+14.


# specify time step: standard 1h
# number of time steps per hour
MH=5 # 1h/10=6min step
MH=1  # 1h step

if MH==1:
    get_uv=get_uv1_server
else:
    get_uv=get_uv2_server
    
dtsec=60*60./MH
tau=dtsec/111111. # deg per (velocityunits*dt)
# dt in seconds
# vel units m/s
# in other words v*tau -> deg 

tt=np.arange(t1sec_MJD,t2sec_MJD+dtsec,dtsec)
NT=len(tt)
lont=np.zeros((NT,1))
latt=np.zeros((NT,1))
#tempt=np.zeros((NT,ND))

# initial positions
lont[0]=lond0
latt[0]=latd0
 
tic1=datetime.now()
tic=os.times()


#time dependent u,v
kt=0
tsec_MJD=tt[kt]

u1,v1=get_uv(tsec_MJD,Grid)

#FNT=FN0+'GOM3_'+YEAR+'/'+'temp'+D+'/'+TS+'_temp'+D+'.npy'
#temp1=np.load(FNT).flatten()



for kt in range(NT-1):
# time dependent u,v at current time level (from previous step)    
    u0=u1*1.0;v0=v1*1.0
#    temp0=temp1*1.0    
#time dependent u,v at next time level
    tsec_MJD=tt[kt+1]
    u1,v1=get_uv(tsec_MJD,Grid)

#    FNT=FN0+'GOM3_'+YEAR+'/'+'temp'+D+'/'+TS+'_temp'+D+'.npy'
#    temp1=np.load(FNT).flatten()
# velocity at the middle of time step, linear interpolation
    ui=(u0+u1)*0.5;vi=(v0+v1)*0.5

    lont[kt+1],latt[kt+1]=RungeKutta4_lonlattime(lont[kt],latt[kt],Grid,u0,v0,ui,vi,u1,v1,tau)
     
toc=os.times()
print 'MH,NT'
print MH,NT
print 'timing [s] per step: ', (toc[0]-tic[0])/NT,(toc[1]-tic[1])/NT
toc1=datetime.now()
print tic1,toc1
print toc1-tic1

D={}
D['lont']=lont;D['latt']=latt;D['tt']=tt
#save positions
FN='dtr_fvcom.npy'
np.save(FN,D)
