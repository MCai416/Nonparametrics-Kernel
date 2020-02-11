# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:44:28 2020

@author: Ming Cai
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family':'DejaVu Sans',
        'weight':'normal',
        'size'   : 30}

matplotlib.rc('font', **font)


# space 

x_min = -4
x_max = 4
grid_size = 100 * (x_max - x_min) + 1
#plt.xlim([x_min,x_max])

# bandwidth init 
cons = 1
n = 100
a = cons*np.around(n**(-1/5), decimals = 2)

# space init
x = np.linspace(x_min, x_max, num = grid_size)

# random variable monte carlo 
rv = np.random.normal(0, 1, n)

# functions

def k_emp(rv, a, x):  
    # here, rv is a one dim scalar
    # one dimensional kernel
    return 0.5*(np.where(np.abs((x-rv))/a<=1, 1, 0))

def fnonpar(rv, a, x, kernel):
    # here, rv is a one dimensional random variable array
    n = rv.shape[0]
    f = np.zeros(x.shape[0])
    for i in range(n):
        f = f+kernel(rv[i], a, x)
    return f/(n*a)

def cdf(x):
    n = x.shape[0]
    cdf = np.zeros(n)
    cdf[0] = x[0]
    cml = 0 
    for i in range(1,n):
        cml = cml + x[i]
        cdf[i] = cml
    return cdf/100

def npdf(x, m=0, s=1):
    a = 1/(s*np.sqrt(2*np.pi))
    b = np.exp((-1/2)*((x-m)/s)**2)
    return a*b

def k_normal(rv, a, x):
    z = (x-rv)/a
    return npdf(z)

# estimation

n0 = npdf(x, 0, 1)
ncdf0 = cdf(n0)
f = fnonpar(rv, a, x, k_emp)
fn = fnonpar(rv, a, x, k_normal)

# Plot
fig, axs = plt.subplots(2, 2)
fig.suptitle('Draws = %d, Bandwidth = %.2f, Underlying: Normal'%(n, a),fontsize = 50)

axs[0, 0].plot(x, n0, linewidth = 3, color = 'blue')
axs[0, 0].plot(x, f, linewidth = 3, color = 'red')
axs[0, 0].set_title('Identity Kernel, PDF')

axs[1, 0].plot(x, n0, linewidth = 3, color = 'blue')
axs[1, 0].plot(x, fn, linewidth = 3, color = 'red')
axs[1, 0].set_title('Normal Kernel, PDF')

axs[0, 1].plot(x, ncdf0, linewidth = 3, color = 'blue')
axs[0, 1].plot(x, cdf(f), linewidth = 3, color = 'red')
axs[0, 1].set_title('Identity Kernel, CDF')

axs[1, 1].plot(x, ncdf0, linewidth = 3, color = 'blue')
axs[1, 1].plot(x, cdf(fn), linewidth = 3, color = 'red')
axs[1, 1].set_title('Normal Kernel, CDF')

