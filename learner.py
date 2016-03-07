# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 12:01:37 2016

@author: AKononov

abstract class for all learners
"""
from abc import ABCMeta, abstractmethod

class Learner(object):
	__metaclass_ = ABCMeta
	
	@abstractmethod
	def fit(X, target):
		pass
	
	@abstractmethod
	def predict(X):
		pass
	
	@abstractmethod
	def draw():
		pass