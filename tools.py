import numpy as np

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
    fs = np.asfarray(samples)
    for col in fs.T:
        col = (2 * col - col.max() - col.min())/(col.max() - col.min())
    return fs
    
    
def scale_linear_bycolumn(rawpoints, high=100.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / rng)