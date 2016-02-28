# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def add_ones(samples):
	# adds column of ones to the end
	samples = np.array(samples)
	temp = np.ones([samples.shape[0], samples.shape[1] + 1])
	temp[:,0:-1] = samples
	samples = temp
	return samples

def linreg(samples, target, method='GD', iterations=20, rate=0.01, precision=6):
	# liniar regression with gradient decent or exact formula
	# param method: 'GD' - gradient descent, 'exact' - using formula
	# first init weights matrix - dimention m + 1 where m is dimention of samples
	# adding extra element for x0 = 1
	weights = np.zeros([samples.shape[1] + 1, 1])
	# add bias column to samples
	samples = add_ones(samples)
	# if method exact use formula
	if method=='exact':
		weights = np.linalg.inv(np.dot(np.transpose(samples), samples))
		weights = np.dot(weights, np.transpose(samples))
		weights = np.dot(weights, target)
	elif method=='GD':
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
		
def logreg():
	return


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