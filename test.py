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

data = np.genfromtxt('logreg4.txt', dtype=str, delimiter='\n',skip_header=1)
data = [x.split('\t') for x in data]
data = np.asfarray(data)

target = np.atleast_2d(data[:,3]).T
# samples = standardize(rescale(data[:,:2]))
samples = data[:,:2]

plt.gray()

logr = logreg.LogReg()
logr.fit(samples, target, iterations=500, rate=0.00001, precision=2, regularization=5, order=4)

# p = logr.predict(samples)

#print(logr.accuracy)