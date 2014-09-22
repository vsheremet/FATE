# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:11:50 2013

@author: vsheremet
"""
import numpy as np
import matplotlib.delaunay as triang
import matplotlib.pyplot as plt



def triangle_meshcoeff(x,y):
    """
    Triangular Mesh Coefficients
    input x,y - array of Vertices
    """
    coef=1.    
    return coef

# a*cos(30deg)=1
# a - unilateral triangle side
# 2*pi*R/n*cos(30deg)=dR
# R - radius of the shell
# n - number of nodes in the shell
# dR=1 - step in radius between shells
# n=2*pi*R*sqrt(3)/2

pi=np.pi*1.

# circle center
x0=np.array([0.])
y0=np.array([0.])

# surrounding hexagon
# 2*pi*1=6.2831
r=1.;n=6
a=np.array(range(n))*2*np.pi/n
x1=r*np.cos(a)
y1=r*np.sin(a)
x=np.concatenate((x0,x1))
y=np.concatenate((y0,y1))

# 2nd shell
# 2*pi*2=12.5663
r=2.;n=12
a=np.array(range(n))*2*np.pi/n
#a=(np.array(range(n))+0.5)*2*np.pi/n
xb=r*np.cos(a)
yb=r*np.sin(a)
x=np.concatenate((x,xb))
y=np.concatenate((y,yb))


# 3rd shell
# 2*pi*3=18.8495
r=3.;n=18
a=np.array(range(n))*2*np.pi/n
a=(np.array(range(n))+0.5)*2*np.pi/n
xb=r*np.cos(a)
yb=r*np.sin(a)
x=np.concatenate((x,xb))
y=np.concatenate((y,yb))

"""
# 3rd shell
# 2*pi*3=18.8495
r=3.;n=18
a=np.array(range(n))*2*np.pi/n
xb=r*np.cos(a)
yb=r*np.sin(a)
x=np.concatenate((x,xb))
y=np.concatenate((y,yb))
"""

# 4th shell
# 2*pi*4=25.1327
r=4.;n=24
a=np.array(range(n))*2*np.pi/n
#a=(np.array(range(n))+0.5)*2*np.pi/n
xb=r*np.cos(a)
yb=r*np.sin(a)
x=np.concatenate((x,xb))
y=np.concatenate((y,yb))


# 5th shell
# 2*pi*5=31.4159
r=5.;n=24
a=(np.array(range(n))+0.5)*2*np.pi/n
xb=r*np.cos(a)
yb=r*np.sin(a)
x=np.concatenate((x,xb))
y=np.concatenate((y,yb))

# 6th shell
# 2*pi*5=37.6991
r=6.;n=36
a=np.array(range(n))*2*np.pi/n
xb=r*np.cos(a)
yb=r*np.sin(a)
x=np.concatenate((x,xb))
y=np.concatenate((y,yb))


F,E,V,neig = triang.delaunay(x,y)

i=[0,1,2,0]
for t in V:
    tc = t[i]
    plt.plot(x[tc],y[tc])

plt.plot(x,y,'o')
plt.axis('equal')
plt.show()
