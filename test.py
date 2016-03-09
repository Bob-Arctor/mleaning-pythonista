# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:18:54 2016

@author: AKononov
"""

import numpy as np
import matplotlib.pyplot as plt
from mlearning import *
from tools import *
import logreg

data = np.genfromtxt('D:\\Documents\\Coding\\Datasets\\BNP\\train.csv', dtype=str, delimiter='\n',skip_header=1)
data = data[:1000]
data = np.array([x.split(',') for x in data])
data = array_to_float(data)
#data = [ [np.nan if s=='.' else float(s) for s in x] for x in data]

data = np.asfarray(data)
data = delnans_all(data,ax=0)
data = delnans_any(data,ax=1)
target = np.atleast_2d(data[:,1]).T
# samples = standardize(rescale(data[:,:2]))
# samples = np.hstack((data[:,:1],  data[:,4:]))
samples = data[:,2:]
#plt.gray()
logr = logreg.LogReg()
logr.fit(samples, target, iterations=100, rate=0.00001, precision=2, regularization=5, order=2, draw_result=False)

def pairplots(X, labels=None, target=None):
	nVariables = data.shape[1]
	if labels is None:
		labels = ['var%d'%i for i in range(nVariables)]
	fig = plt.figure()
	for i in range(nVariables):
		for j in range(nVariables):
			nSub = i * nVariables + j + 1
			ax = fig.add_subplot(nVariables, nVariables, nSub)
			if i == j:
				ax.hist(data[:,i])
				ax.set_title(labels[i])
			else:
				ax.plot(data[:,i], data[:,j], '.k')
	return fig
	
	
def pairplots_logreg(X, logr, labels=None, target=None, maximum=5):
	nVariables = min(maximum, X.shape[1])
	if labels is None:
		labels = ['var%d'%i for i in range(nVariables)]
	fig = plt.figure()
	for i in range(nVariables):
		for j in range(nVariables):
			print('making plot at %d , %d'%(i,j))
			nSub = i * nVariables + j + 1
			ax = fig.add_subplot(nVariables, nVariables, nSub)
			if i == j:
				ax.hist(X[:,i], color='orange')
				ax.set_title(labels[i])
			else:
				logr.fitplot(X, target, ax, x1=i, x2=j)
				#ax.plot(data[:,i], data[:,j], '.k')
	return fig
	
#manager = plt.get_current_fig_manager()
#manager.window.showMaximized()

#f = pairplots_logreg(samples, logr, target=target, maximum=10)#samples.shape[0])
#f.set_size_inches(30,20)
#plt.show()