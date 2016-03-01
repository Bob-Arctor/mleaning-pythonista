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

#samples = 50*np.random.rand(18,5)
#samples = np.array([[1,2,3],[2,2,2],[4,4,4],[6,4,1]])
#print(samples)
print('-'*20)
plt.close()
#loading data
data = np.genfromtxt('logreg.txt', dtype=None, delimiter='/t', skip_header=1)
data = [x.split('   ') for x in data]
data = [ [s.strip() for s in x] for x in data]
data = [ [np.nan if s=='.' else float(s) for s in x] for x in data]
data = np.array(data)
data = delnans(data)
target = np.atleast_2d(data[:,3])
samples = np.concatenate([data[:,:3],data[:,4:]],axis=1)
samples_t = samples[:100,:]
samples_p = samples[100:,:]
target_t = target.T[:100,:]
target_p = target.T[100:,:]
print(target_t.shape)
print(samples_t.shape)
#print(target.T)
weights = logreg(samples_t, target_t, iterations=50, rate=0.000001)
print(weights)
plt.show()
print(np.dot(add_ones(samples_p),weights))
print(target_p)