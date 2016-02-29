# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 13:25:50 2016

@author: AKononov
"""
import numpy as np
import matplotlib.pyplot as plt
from mlearning import *
from tools import *

'''
plt.close()
samples = 50*np.random.rand(1000,1)
target = 50.45 - 41.7 * samples
noise = np.random.normal(0,1000,(1000,1))
target_n = target + noise			
weights = linreg(samples, target_n, iterations=500, rate=0.001)
print(weights)
plt.show()
plt.close()
plt.scatter(samples, target_n)
samples_n = add_ones(samples)
estimation = np.dot(samples_n, weights)
plt.plot(samples, estimation, 'r')
weights = linreg(samples, target_n, method='exact')
print(weights)
estimation = np.dot(samples_n, weights)
plt.plot(samples, estimation, 'y')
plt.plot(samples, target, 'g')
plt.show()
'''

samples = 50*np.random.rand(3,4)
print(samples)
print(rescale(samples))