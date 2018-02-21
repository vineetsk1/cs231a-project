'''
Created on Apr 1, 2014

@author: ntraft
'''
from __future__ import division
import numpy as np
import covariance as cov

#import com.ntraft.covariance as cov

class GaussianProcess:
	'''
	Represents a Gaussian process that can be sampled from. Samples are taken
	at each test point, given the supplied observations.
	'''

	def __init__(self, zx, zy, testpoints, kernel=cov.sq_exp_kernel()):
		''' Creates a new Gaussian process from the given observations. '''
		self.timepoints = testpoints
		self.kernel = kernel
		
		# covariance of observations
		self.K = kernel(zx, zx, 'train')
		self.K += 1e-9*np.eye(self.K.shape[0])
		Ltrain = np.linalg.cholesky(self.K)
		
		# compute the predictive mean at our test points
		self.Kstar = kernel(zx, testpoints, 'cross')
		v = np.linalg.solve(Ltrain, self.Kstar)
		self.mu = np.dot(v.T, np.linalg.solve(Ltrain, zy))
		
		# compute the predictive variance at our test points
		self.Kss = kernel(testpoints, testpoints, 'test')
		self.Kss += 1e-9*np.eye(self.Kss.shape[0])
		self.prior_L = np.linalg.cholesky(self.Kss)
		
		self.Kss = self.kernel(testpoints, testpoints, 'train')
# 		self.Kss += 1e-3*np.eye(self.Kss.shape[0])
		self.L = np.linalg.cholesky(self.Kss - np.dot(v.T, v))
	
	def sample(self, n=1):
		'''
		Draw n samples from the gaussian process posterior.
		
		Returns a timepoints x n matrix, with each sample being a column.
		'''
		sz = (len(self.timepoints), n)
		return self.mu.reshape(-1,1) + np.dot(self.L, np.random.normal(size=sz))
	
	def sample_prior(self, n=1):
		'''
		Draw n samples from the gaussian process prior.
		
		Returns a timepoints x n matrix, with each sample being a column.
		'''
		sz = (len(self.timepoints), n)
		return np.dot(self.prior_L, np.random.normal(size=sz))


class ParametricGaussianProcess:
	'''
	Represents a Gaussian process of a parametric function. This is actually
	implemented as two separate GPs, one for x and one for y. The processes can
	be sampled from to predict x,y = f(t). Samples are taken at each test
	point, given the supplied observations.
	'''

	def __init__(self, observations, timepoints, xkernel=cov.sq_exp_kernel(), ykernel=cov.sq_exp_kernel()):
		zt = observations[:,0]
		zx = observations[:,1]
		zy = observations[:,2]

		self.xgp = GaussianProcess(zt, zx, timepoints, xkernel)
		self.ygp = GaussianProcess(zt, zy, timepoints, ykernel)
	
	def sample(self, n=1):
		'''
		Draw n samples from the gaussian process posterior.
		
		Returns a timepoints x n x 2 matrix. The first dimension is time, the
		second dimension is samples, and the third dimension is x,y.
		'''
		x_post = self.xgp.sample(n)
		y_post = self.ygp.sample(n)
		return np.dstack((x_post, y_post))
	
	def sample_prior(self, n=1):
		'''
		Draw n samples from the gaussian process prior.
		
		Returns a timepoints x n x 2 matrix. The first dimension is time, the
		second dimension is samples, and the third dimension is x,y.
		'''
		x_post = self.xgp.sample_prior(n)
		y_post = self.ygp.sample_prior(n)
		return np.dstack((x_post, y_post))

class ParametricGPModel:
	'''
	Models a parametric Gaussian process. The process retains its prior information as
	given by its covariance functions, but the posterior can periodically be recomputed 
	to incorporate new data.
	'''

	def __init__(self, xkernel=cov.sq_exp_kernel(), ykernel=cov.sq_exp_kernel()):
		self.xkernel = xkernel
		self.ykernel = ykernel
	
	def recompute(self, observations, timepoints):
		'''
		Recompute the posterior at the given evaluation points.
		'''
		self.gp = ParametricGaussianProcess(observations, timepoints, self.xkernel, self.ykernel)
	
	def sample(self, n=1):
		'''
		Draw n samples from the gaussian process posterior. See
		ParametricGaussianProcess.sample for complete details.
		'''
		return self.gp.sample(n)
	
	def sample_prior(self, n=1):
		'''
		Draw n samples from the gaussian process prior. See
		ParametricGaussianProcess.sample_prior for complete details.
		'''
		return self.gp.sample_prior(n)
