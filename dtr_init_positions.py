# -*- coding: utf-8 -*-
"""
Initial position for Drifter Tracking using velocity field from FVCOM GOM3 model
"""

import numpy as np
import matplotlib.pyplot as plt

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


############################################################################
# calculate intial conditions for some cases    
print 'calculating init positions'
    
from get_fvcom_gom3_grid import get_fvcom_gom3_grid
Grid=get_fvcom_gom3_grid('disk')

# GOM3R - rectangular grid that completely covers GOM3 the triangular grid
# and is aligned with 32x32 front detection subwindows
gom3r_lon=np.load('gom3r.lon.npy')
gom3r_lat=np.load('gom3r.lat.npy')
MS=2
MS=4
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
LArea='GOM3'
LArea='GBext'
LArea='ShBr'
LArea='CodSpawn1'
LArea='GOM3'

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

elif LArea=='GBext':
# GB extended area    
# select only points inside GBext Area
    GBextlon=np.array([-69.5,-67.,-64.,-67.,-69.5])
    GBextlat=np.array([ 39. , 39., 42., 43., 42. ])
    i=inconvexpolygon(llond0,llatd0,GBextlon,GBextlat)
    llond0=llond0[np.argwhere(i).flatten()]
    llatd0=llatd0[np.argwhere(i).flatten()]
    ND=llond0.size
    print 'ND =',ND

elif LArea=='ShBr':
# select only points along ShBr Area around 100m

    Linelon=np.array([-72.,-71.,-70.,-69.,-68.,-67.,-66.,-65.])
    Linelat=np.array([39.7,40.0,40.0,40.1,40.3,40.5,41.3,42.4])
    from scipy import interpolate
    finterp1d = interpolate.interp1d(Linelon, Linelat)
    llond0=np.arange(-72.,-65.,0.005)
    llatd0=finterp1d(llond0)
    ND=llond0.size
    print 'ND =',ND
    
    plt.figure()
    plt.plot(Grid['lon'],Grid['lat'],'g.',Grid['lonc'],Grid['latc'],'c+',llond0,llatd0,'r.')
    plt.show()

elif LArea=='CodSpawn1':        
    Spawn=np.array([
    [-66.10, 42.30],
    [-68.00, 42.30],
    [-68.00, 41.75],
    [-66.10, 41.75],
    ])
    i=inconvexpolygon(llond0,llatd0,Spawn[:,0],Spawn[:,1])
    llond0=llond0[np.argwhere(i).flatten()]
    llatd0=llatd0[np.argwhere(i).flatten()]
    ND=llond0.size
    print 'ND =',ND

np.save('llond0.npy',llond0)
np.save('llatd0.npy',llatd0)


