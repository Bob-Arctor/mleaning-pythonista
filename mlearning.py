# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from tools import *

def linreg(samples, target, method='BGD', iterations=20, rate=0.01, precision=6):
	# regression
	# linear with gradient decent or exact formula
	# or logistic
	# param method:  'BGD' - batch gradient descent, 'Norm' - using formula
	# 'SGD' stocastic gradient descent
	# first init weights matrix - dimention m + 1 where m is dimention of samples
	# adding extra element for x0 = 1
	weights = np.zeros([samples.shape[1] + 1, 1])
	# add bias column to samples
	samples = add_ones(samples)
	# if method linear exact use formula
	if method=='Norm':
		weights = np.linalg.inv(np.dot(np.transpose(samples), samples))
		weights = np.dot(weights, np.transpose(samples))
		weights = np.dot(weights, target)
	elif method in ('BDG', 'SGD'):
		# calculate h theta
		h = np.dot(samples, weights)
		# calculate current error
		err = (1. / samples.shape[0]) * np.dot(np.transpose(h - target), h - target)
		# perform decent
		for i in range(iterations):
			# plot error
			plt.scatter(i,err)
			# save old error
			old_err = err
			# calculate gradient
			grad = np.dot(np.transpose(h - target), samples) * float(rate) / samples.shape[0]
			# adjust weights
			weights = weights - np.transpose(grad)
			# calculate h theta
			h = np.dot(samples, weights)
			if method == 'logistic':
				h = logistic(h)
			# calculate current error
			err = (1. / samples.shape[0]) * np.dot(np.transpose(h - target), h - target)
			# calculate relativ error
			relative_err = err / old_err
			# print results
			# print('%d iteration: error %.4f relative error %.4f'%(i, err, relative_err))
			if round(relative_err, precision) == 1.:
				break
			elif old_err[0][0] < err[0][0]:
				print('Error increasing, try changing learning rate. Currently: %f'%(rate))
				weights = 0*weights
				break 
	# print('relative error: %.6f'%(relative_err)) 
	return weights
	
	
def logreg(samples, target, method='BGD', iterations=20, rate=0.01, precision=4):
	# logistic regression
	# first init weights matrix - dimention m + 1 where m is dimention of samples
	# adding extra element for x0 = 1
	weights = np.zeros([samples.shape[1] + 1, 1])
	# add bias column to samples
	samples = add_ones(samples)
	# if method linear exact use formula
	# calculate h theta
	h = np.dot(samples, weights)
	h = logistic(h)
	# calculate current error
	err = (-1. / samples.shape[0]) * (np.dot(np.log(h).T,target) + np.dot(np.log(1-h).T,(1 - target)))
	# perform decent 
	
	for i in range(iterations):
		# plot error
		plt.scatter(i,err)
		# save old error
		old_err = err
		# calculate gradient
		grad = np.dot(np.transpose(h - target), samples) * float(rate)
		# adjust weights
		weights = weights - np.transpose(grad)
		# calculate h theta
		h = np.dot(samples, weights)
		h = logistic(h)
		# calculate current error
		err = (-1. / samples.shape[0]) * (np.dot(np.log(h).T,target) + np.dot(np.log(1-h).T,(1 - target)))
		# calculate relativ error
		relative_err = err / old_err
		# print results
		# print('%d iteration: error %.4f relative error %.4f'%(i, err, relative_err))
		if round(relative_err[0][0], precision) == 1.:
			pass #break
		elif np.abs(old_err[0][0]) < np.abs(err[0][0]):
			print('Error increasing, try changing learning rate. Currently: %f'%(rate))
			weights = 0*weights
			break
		elif np.isnan(err[0][0]):
			print('Error increased to infinity, try changing learning rate')
			weights = 0*weights
			break 
	print('relative error: %.6f'%(relative_err)) 
	return weights
	
	