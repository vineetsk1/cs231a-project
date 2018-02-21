# coding=utf-8
'''
Various Gaussian Process kernel functions.

Created on Apr 5, 2014

@author: ntraft
'''
from __future__ import division
import numpy as np

################################################################################
# COVARIANCE FUNCTIONS
################################################################################

def sq_dist(a, b):
	return (a**2).reshape(-1, 1) + b**2 - 2*np.outer(a, b)

def sq_exp(a, b, _, l, sigma2=1):
	return sigma2 * np.exp(-.5 * sq_dist(a/l, b/l))

def sq_exp_iso(a, b, kind, l, sigma2=1):
	if kind == 'test':
		return sigma2 * np.ones((len(a), 1))
	else:
		return sigma2 * np.exp(-.5 * sq_dist(a/l, b/l))

def matern(a, b, kind, l, sigma2=1):
	if kind == 'test' and False:
		return sigma2
	else:
		a = np.sqrt(5) * a/l;
		b = np.sqrt(5) * b/l;
		r = sq_dist(a, b)
		return sigma2 * np.exp(-np.sqrt(r)) * (1 + np.sqrt(r) + r/3);

def linear(a, b, kind, sigma2=1):
	if kind == 'test' and False:
		r = np.sum((a*b).reshape(-1,1), 1).reshape(-1,1)
	else:
		r = np.outer(a, b)
	return (1 + r) * sigma2

def noise(a, b, kind, sigma2=0):
	if kind == 'train':
		return sigma2 * np.eye(len(a))
	elif kind == 'test':
		return sigma2
	else:
# 		return sigma2
		return 0

################################################################################
# KERNEL GENERATION FUNCTIONS
################################################################################

def sq_exp_kernel(l=1, sigma2=1):
	''' Squared exponential kernel. '''
	def f(a, b, kind):
		return sq_exp(a, b, kind, l, sigma2)
	return f

def matern_kernel(l, sigma2=1):
	'''
	Matérn kernel. See "Gaussian Processes for Machine Learning", by
	Rasmussen and Williams, Chapter 4.
	
	Specifically, this kernel is Matérn class with v=5/2, multiplied by an
	optional signal variance sigma2.
	'''
	def f(a, b, kind):
		return matern(a, b, kind, l, sigma2)
	return f

def linear_kernel(sigma2=1):
	''' Linear kernel, obtained from linear regression. '''
	def f(a, b, kind):
		return linear(a, b, kind, sigma2)
	return f

def noise_kernel(sigma2=0):
	'''
	Standard noise kernel. Adds a small amount of variance at every point
	in the covariance matrix where i=j.
	'''
	def f(a, b, kind):
		return noise(a, b, kind, sigma2)
	return f

def summed_kernel(*args):
	'''
	Forms a kernel consisting of the sum of the given kernels.
	'''
	def f(a, b, kind):
		K = np.zeros((len(a), len(b)))
		for k in args:
			K += k(a, b, kind)
		return K
	return f
