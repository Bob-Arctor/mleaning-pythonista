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

data = np.genfromtxt('logreg2.txt', dtype=str, delimiter='\n',skip_header=1)
#print(data)
data = [x.split('\t') for x in data]
#data = [ [s.strip() for s in x] for x in data]
#data = [ [np.nan if s=='.' else float(s) for s in x] for x in data]
data = np.asfarray(data)
#data = delnans(data)

target = np.atleast_2d(data[:,2]).T
#samples = np.concatenate([data[:,:3],data[:,4:]],axis=1)
samples = data[:,:2]
print(samples.shape)
print(target.shape)
#samples_t = samples[:100,:]
#s_t_n = rescale(samples_t)
#s_p_n = rescale(samples_p)
#samples_p = samples[100:,:]
#target_t = target.T[:100,:]
#target_p = target.T[100:,:]
#print(samples_t.shape)

weights = logreg(samples, target, iterations=7000, rate=0.001)
#print(weights)
plt.show()
print('-'*50)
print(np.concatenate([logistic(np.dot(add_ones(samples),weights)),target],axis=1))
print('-'*50)
#print(np.concatenate([logistic(np.dot(add_ones(s_t_n),weights)),target_t],axis=1))
print(weights)
#colors = plt.cm.coolwarm(list(target_t.T))
#plt.close()
plt.scatter(samples[:,0], samples[:,1], c=target, s=50)
x1 = np.linspace(samples[:,0].min(),samples[:,0].max(),5)
x2 = (-weights[2,0] - weights[0,0]*x1) / weights[1,0]
plt.plot(x1,x2)
plt.gray()
plt.show()
