# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:07:55 2016

@author: AKononov

logistic regression class
"""

import numpy as np
import matplotlib.pyplot as plt
from tools import *
import learner

class LogReg(learner.Learner):
	def __init__(self):
		super(LogReg, self).__init__()
		self.weights = np.atleast_2d([])
		self.fitting_err = []
		self.decision_bound = None
		self.X_mean = None
		self.X_std = None
		self.order = 0
		
	def fit(self, 
			samples,					# sample points, rows-samples,cols-features
			target, 				# target array to fit, atleast2d
			method='BGD', 		# BGD, SGD
			iterations=20, 
			rate=0.01, 			# learning rate
			precision=4, 			# used to compare realtive error
			regularization=0, 
			order=1, 
			scale=True,
			draw_error=True,
			draw_result=True):			# draw scatterplot and decision boundary
		print('-'*50)
		print('Fitting logistic regression')		
		# non linear logistic refression
		if scale:
			X = standardize(rescale(samples))
		# need to add columns to samples with multiple degrees of it
		X = raise_order(X, order)
		# first init weights matrix - dimention m + 1 where m is dimention of samples
		# adding extra element for xn+1 = 1
		weights = np.zeros([X.shape[1] + 1, 1])
		X = add_ones(X)
		# calculate h theta
		h = np.dot(X, weights)
		h = logistic(h)
		# calculate current error
		self.fitting_err = []
		err = (-1. / X.shape[0]) * (np.dot(np.log(h).T,target) + np.dot(np.log(1-h).T,(1 - target)))
		self.fitting_err.append(err[0,0])
		for i in range(iterations):
			# save old error
			old_err = err
			# weight subarray with 0 for theta n+1
			weightsR = weights.copy()
			weightsR[-1,:] = 0
			# calculate gradient with regularization
			grad = ( np.dot(np.transpose(h - target), X) + regularization * weightsR.T) * float(rate) 
			# adjust weights
			weights = weights - np.transpose(grad)
			# calculate h theta
			h = np.dot(X, weights)
			h = logistic(h)
			# calculate regularisation value
			regul = regularization * np.dot(weights[:-1,:].T,weights[:-1,:]) / (2 * X.shape[0])
			# calculate current error
			err = (-1. / X.shape[0]) * (np.dot(np.log(h).T,target) + np.dot(np.log(1-h).T,(1 - target))) - regul
			self.fitting_err.append(err[0,0])			
			# calculate relativ error
			relative_err = err / old_err
			# print('%d iteration: error %.4f relative error %.4f'%(i, err, relative_err))
			if round(relative_err[0][0], precision) == 1.:
				#break
				rate *=2
			if np.abs(old_err[0][0]) < np.abs(err[0][0]) or np.isnan(err[0][0]):
				#print('Error increasing, try changing learning rate. Currently: %10f'%(rate))
				#weights = 0*weights
				#break
				rate /= 2
				weights = self.weights.copy()
			else:
				self.weights = weights.copy()		
		print('-'*50)
		print('Converged')
		# saving weights
		self.weights = weights.copy()
		self.order = order
		# drawing
		if draw_error:
			fig1 = plt.figure()
			ax1 = fig1.add_subplot(111)
			self.errorplot(ax1)
			plt.show()
		if draw_result:
			fig2 = plt.figure()
			ax2 = fig2.add_subplot(111)
			self.fitplot(samples,target,ax2)
			plt.show()
		
	def fitplot(self, X, target, ax, x1=0, x2=1):
		ax.scatter(X[:,x1], X[:,x2], c=target, s=50)
		u = np.linspace(X[:,x1].min(),X[:,x1].max(),50)
		v = np.linspace(X[:,x2].min(),X[:,x2].max(),50)
		z= np.zeros(shape=(len(u),len(v)))
		for i in range(len(u)):
			for j in range(len(v)):
				z[i,j] = np.dot(add_ones(raise_order(np.atleast_2d([u[i],v[j]]),self.order)),self.weights)
		z = z.T
		ax.contour(u,v,z, c='r', levels=[0])
		
	def errorplot(self, ax):
		for i in range(len(self.fitting_err)):
			ax.scatter(i, self.fitting_err[i])