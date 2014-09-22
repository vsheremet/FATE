# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:11:50 2013

@author: vsheremet
"""
import numpy as np
import matplotlib.delaunay as triang
import matplotlib.pyplot as plt

# Fibonacci spiral

phi=(np.sqrt(5)+1.)/2.
a0=2.*np.pi/(phi*phi)

n=7
F_n=np.round( (phi**n-1./(-phi)**n)/np.sqrt(5) )
F_15=610

N=1+1+2+3+5+8+13+21+34+55+89+144+233+377+610+987+1597+2584+4181+6765
N=1+1+2+3+5+8+13+21+34+55+89+144+233+377+610+987+1597
n=np.array(range(N+1))
r=np.sqrt(n)
a=a0*n

# circle center
x=r*np.cos(a)
y=r*np.sin(a)


F,E,V,neig = triang.delaunay(x,y)


i=[0,1,2,0]
for t in V:
    tc = t[i]
    plt.plot(x[tc],y[tc])
    

plt.plot(x,y,'o')
plt.axis('equal')
plt.show()
