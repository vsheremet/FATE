# -*- coding: utf-8 -*-
"""
test simple multiprocessing
"""

import numpy as np
import multiprocessing as mp

def f(x):
    return x*x

x = np.linspace(0,20,10000)
p = mp.Pool(processes=16)
p.map(f, x)

"""
p = mp.Process(target=f, args=(x,))
p.start()
p.join()
"""



"""
ND=640000
x=np.zeros(ND,dtype=float)
for k in range(ND):
    x[k]=np.sin(k/127.*np.pi)
"""

"""
lont=np.zeros((NT,ND))
latt=np.zeros((NT,ND))
# initial positions
lont[0,:]=llond0
latt[0,:]=llatd0
for kt in range(NT-1):
     for kd in range(ND):
# for each drifter make one time step using classic 4th order Runge-Kutta method
lont[kt+1,kd],latt[kt+1,kd]=RungeKutta4(lont[kt,kd],latt[kt,kd],Grid,u,v,tau)
"""

