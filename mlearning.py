# coding: utf-8
import numpy as np

def linreg(samples, target, method='SGD', iterations=20, rate=1):
	# liniar regression with gradient decent
	# types can be liniar gradient d or stochastic gc
	# first init weights matrix - dimention m + 1 where m is dimention of samples
	# adding extra element for x0 = 1
	weights = np.zeros([1, samples.shape[1] + 1])
	# add bias column to samples
	temp = np.ones([samples.shape[0], samples.shape[1] + 1])
	temp[:,0:-1] = samples
	samples = temp
	# transpose target array to vector
	# target = np.transpose(target)
	# calculate h theta
	h = np.dot(samples, np.transpose(weights))
	# calculate current error
	err = (1. / samples.shape[0]) * np.dot(np.transpose(h - target), h - target)
	# perform decent
	for i in range(iterations):!=
		# save old error
		old_err = err
		# calculate gradient
		grad = np.dot(np.transpose(h - target), samples) * float(rate) / samples.shape[0]
		# adjust weights
		weights = weights - grad
		# calculate h theta
		h = np.dot(samples, np.transpose(weights))
		# calculate current error
		err = (1. / samples.shape[0]) * np.dot(np.transpose(h - target), h - target)
		# calculate relativ error
		relative_err = err / old_err
		# print results
		print('%d iteration: error %.4f relative error %.4f'%(i, err, relative_err))
	return weights
		

samples = 5*np.random.rand(100,1)
target = 3 - 4.1 * samples
noise = np.random.normal(0,1,(100,1))
target_n = target + noise			
weights = linreg(samples, target_n)
print(weights)