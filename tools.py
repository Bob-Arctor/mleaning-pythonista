import numpy as np
import itertools

def tanh(x):
    return np.tanh(x)


def tanh_deriv(x):
    return 1.0 - np.tanh(x) ** 2


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    return logistic(x) * (1 - logistic(x))
    
    
def add_ones(samples):
	# adds column of ones to the end
	samples = np.array(samples)
	temp = np.ones([samples.shape[0], samples.shape[1] + 1])
	temp[:,0:-1] = samples
	samples = temp
	return samples


def add_zeros(samples):
	# adds column of ones to the end
	samples = np.array(samples)
	temp = np.zeros([samples.shape[0], samples.shape[1] + 1])
	temp[:,0:-1] = samples
	samples = temp
	return samples


def add_column(samples, col=1):
	# adds column of ones to the end
	samples = np.array(samples)
	temp = np.empty([samples.shape[0], samples.shape[1] + 1])
	temp.fill(col)
	temp[:,0:-1] = samples
	samples = temp
	return samples
 
 
def categorize(features, ignore_nans=True):
     # returns new array where every unique feature is labeled
    uniq = np.unique(features)
    if ignore_nans and np.nan in uniq:
        uniq = np.delete(list(uniq).index(np.nan))
    uniq = list(uniq)
    labeled_features = []
    for cat in features:
        if ignore_nans and np.isnan(cat):
            labeled_features.append(cat)
        else:
            labeled_features.append(uniq.index(cat))
    return np.array(labeled_features)
    
    
def rescale(samples, low=-1, high=1):
    # returns rescaled column from -1 to 1
	mins = np.min(samples, axis=0)
	maxs = np.max(samples, axis=0)
	fs = np.asfarray(samples)
	rng = maxs - mins
	return high - (((high - low) * (maxs - fs)) / rng)


def scale_back(scaled, original, low=-1, high=1):
	mins = np.min(original, axis=0)
	maxs = np.max(original, axis=0)
	rng = maxs - mins
	return maxs - (high - scaled)*rng / (high-low)

    
def standardize(samples):
	mean = np.mean(samples, axis=0)
	std = np.std(samples, axis=0)
	return (samples - mean) / std
	
	
def stand_back(samples, original):
	mean = np.mean(original, axis=0)
	std = np.std(original, axis=0)	
	return samples * std + mean
	

def getstats(samples):
	mins = np.min(samples, axis=0)
	maxs = np.max(samples, axis=0)
	scaled_x = rescale(samples)
	mean = np.mean(scaled_x, axis=0)
	std = np.std(scaled_x, axis=0)
	stats = {'min':mins,
			'max':maxs,
			'mean':mean,
			'std':std}
	return stats
	
	
def delnans(s):
	return s[~np.isnan(s).any(axis=1)]
	

#raises order of a sample set by n, returns new columns
def raise_order(samples, n):
	if n==1:
		return samples
	elif n<1:
		raise AttributeError('n must be greater or equal to 1')
	else:
		s = samples.copy()
		# make iterations array
		iter = []
		for i in range(2, n+1):
			iter += list(itertools.combinations_with_replacement(range(s.shape[1]),i))
		# for every combination
		for el in iter:
			# for every element in combination
			f = None
			for i in el:
				if f is None:
					f = samples[:,i].copy()
				else:
					f *= samples[:,i]
			s = np.concatenate([s,np.atleast_2d(f).T], axis=1)
	return s
		