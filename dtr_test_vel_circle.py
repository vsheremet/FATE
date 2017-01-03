# -*- coding: utf-8 -*-
"""
Drifter Tracking using velocity field from FVCOM GOM3 model
based on dtr_v24
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

def get_uv1(tRD,D):
    """
get velocity fields either from a local file or from internet

    u,v=get_uv1(tRD,D)

    tRD - time RataDie, python ordinal
    u,v - velocity fields
    D - depth code: a - avg; 0 - surf
    """

# location of velocity fields on a local disk
    FN0='/home/vsheremet/FATE/GOM3_DATA/'
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
    print FNU
    print FNV
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
    FN0='/home/vsheremet/FATE/GOM3_DATA/'

#    tn=np.round(tRD*24.)/24.
    tn=np.floor(tRD*24.)/24.
    ti=datetime.fromordinal(int(tn))
    YEAR=str(ti.year)
    MO=str(ti.month).zfill(2)
    DA=str(ti.day).zfill(2)
    hr=(tn-int(tn))*24
    HR=str(int(np.round(hr))).zfill(2)            
    TS=YEAR+MO+DA+HR+'0000'
    print TS
    a=(tRD-tn)*24.
    print a
    FNU=FN0+'GOM3_'+YEAR+'/'+'u'+D+'/'+TS+'_u'+D+'.npy'
    FNV=FN0+'GOM3_'+YEAR+'/'+'v'+D+'/'+TS+'_v'+D+'.npy'
    print FNU
    print FNV
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
    print TS
    b=1.0-a
    print b
    FNU=FN0+'GOM3_'+YEAR+'/'+'u'+D+'/'+TS+'_u'+D+'.npy'
    FNV=FN0+'GOM3_'+YEAR+'/'+'v'+D+'/'+TS+'_v'+D+'.npy'
    print FNU
    print FNV
    u1=np.load(FNU).flatten()
    v1=np.load(FNV).flatten()

    u=u0*b+u1*a
    v=v0*b+v1*a
    return u,v

def get_uvmonthly(tRD,D):
    """
get velocity fields either from a local file or from internet

    u,v=get_uvavg(tRD,D)

    tRD - time RataDie, python ordinal
    u,v - velocity fields
    D - depth code: a - avg; 0 - surf
    """

# location of velocity fields on a local disk
    FN0='/home/vsheremet/FATE/GOM3_DATA/GOM3_monthly/'
    tn=np.round(tRD*24.)/24.
    ti=datetime.fromordinal(int(tn))
    YEAR=str(ti.year)
    MO=str(ti.month).zfill(2)
    TS=YEAR+MO
    print TS
    
    FNU=FN0+'u'+D+'/'+TS+'m_u'+D+'.npy'
    FNV=FN0+'v'+D+'/'+TS+'m_v'+D+'.npy'
    print FNU
    print FNV
    u=np.load(FNU).flatten()
    v=np.load(FNV).flatten()
    return u,v

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
        
# Drifter Tracking Script
############################################################################
    
# FVCOM GOM3 triangular grid
############################################################
from get_fvcom_gom3_grid import get_fvcom_gom3_grid
Grid=get_fvcom_gom3_grid('disk')
############################################################
"""
# test values of initial conditions
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
############################################################################
# delete files if you want new initial conditions
try:
    llond0=np.load('llond0.npy')
    llatd0=np.load('llatd0.npy')
    print 'loading init positions from file'
    ND=llond0.size
    print 'ND =',ND

except:
# calculate intial conditions for some cases    
    print 'ipython run dtr_init_positions.py to make llond0.npy llatd0.npy'
    sys.exit()
    
#############################################################################
# example of running code
# ipython: run dtr_monthly.py 1984
# run each year on a separate CPU
############################################################################
NCPU=1
#NCPU=16 # number of CPUs to use

# velocity depth code: 0 - surface; a - depth averaged
D='a'

#ARG1=sys.argv[1]
#YEAR=ARG1[0:4];year=int(YEAR)
#months=range(1,7)
year=2010
mo=2
da=1
ti=datetime(year,mo,da)
print ti.isoformat()
YEAR=str(ti.year)
MO=str(ti.month).zfill(2)
DA=str(ti.day).zfill(2)

print YEAR,MO,DA
print year,mo,da
print D

t0=RataDie(year,mo,da)
FTS=YEAR+MO+DA

# specify the length of calculation, typically 62 days
#t1=t0+62
#t1=t0+30
#t1=t0+1.
#t1=t0+15.
t1=t0+1.

# specify time step: standard 1h
# number of time steps per hour
MH=5 # 1h/10=6min step
MH=1  # 1h step
    
dtsec=60*60./MH
tau=dtsec/111111. # deg per (velocityunits*dt)
# dt in seconds
# vel units m/s
# in other words v*tau -> deg 

dtday=1./24./MH
tt=np.arange(t0,t1+dtday,dtday)
NT=len(tt)
lont=np.zeros((NT,ND))
latt=np.zeros((NT,ND))
#tempt=np.zeros((NT,ND))

# initial positions
lont[0,:]=llond0
latt[0,:]=llatd0
 
tic1=datetime.now()
tic=os.times()

#time dependent u,v
kt=0
tRD=tt[kt]

#    u1,v1=get_uvmonthly(tRD,D)
u1,v1=get_uvcircle(Grid)
u0=u1*1.0;v0=v1*1.0
ui=(u0+u1)*0.5;vi=(v0+v1)*0.5

#FNT=FN0+'GOM3_'+YEAR+'/'+'temp'+D+'/'+TS+'_temp'+D+'.npy'
#temp1=np.load(FNT).flatten()
#plt.figure()
#plt.quiver(Grid['lonc'],Grid['latc'],u0,v0);
#plt.title('u,v')
#plt.show()
np.save('u_circ.npy',u0)
np.save('v_circ.npy',v0)


if NCPU==1:    
    for kt in range(NT-1):
        print kt,NT
        for kd in range(ND):
# for each drifter make one time step using classic 4th order Runge-Kutta method        
            lont[kt+1,kd],latt[kt+1,kd]=RungeKutta4_lonlattime(lont[kt,kd],latt[kt,kd],Grid,u0,v0,ui,vi,u1,v1,tau)
# save temp data
#            kf,lamb0,lamb1,lamb2=find_kf_lonlat(Grid,lont[kt,kd],latt[kt,kd])
#            kvf=Grid['kvf'][:,kf]# coordinates of the vertices
#            tempt[kt,kd]=temp1[kvf][0]*lamb0+temp1[kvf][1]*lamb1+temp1[kvf][2]*lamb2
       
else:
    for kt in range(NT-1):
        print kt,NT
        los=lont[kt,:];las=latt[kt,:]
        p = mp.Pool(processes=NCPU)
        lolas1=p.map(step,gen_args(los,las,Grid,u0,v0,ui,vi,u1,v1,tau))
        p.close()
        p.join()
        lolas1=np.array(lolas1)
        lont[kt+1,:]=lolas1[:,0];latt[kt+1,:]=lolas1[:,1]
     
toc=os.times()
print 'MH,NT,ND,NCPU'
print MH,NT,ND,NCPU
print 'timing [s] per step: ', (toc[0]-tic[0])/NT,(toc[1]-tic[1])/NT
toc1=datetime.now()
print tic1,toc1
print toc1-tic1


#save positions once per day
np.save('dtr_test_circ_lont.npy',lont[::24*MH,:])
np.save('dtr_test_circ_latt.npy',latt[::24*MH,:])
np.save('dtr_test_circ_tt.npy',tt[::24*MH])
#np.save('dtr_'+FTS+'_'+D+'_tempt.npy',tempt)

# bathymetry from FVCOM
bathy=np.load('gom3.h.npy')

# boundary FVCOM
kff=Grid['kff']*1
NF=kff.shape[1]
kfb=np.argwhere(Grid['kff']==-1)[:,1]

#for kt in range(0,NT,NT-1):
for kt in range(0,NT,NT-1):
    plt.figure();
    plt.plot(lont[kt,:],latt[kt,:],'r.');
    plt.tricontour(Grid['lon'],Grid['lat'],bathy,[70.,100.,200.],colors='b') 
    plt.plot(Grid['lonc'][kfb],Grid['latc'][kfb],'k.')
#    plt.axis([-72.,-65.,39.,43.])
    plt.title('vel circle'+str(kt))
   
plt.show()    

