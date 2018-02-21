from __future__ import division
import numpy as np
import covariance as cov
from gp import GaussianProcess
#import com.ntraft.covariance as cov
#from com.ntraft.gp import GaussianProcess
import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
import sys
if sys.platform.startswith('darwin'):
	matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

# This is the true unknown function we are trying to approximate
x1 = lambda x: x.flatten()
x2 = lambda x: x.flatten() # y = x
# x2 = lambda x: 2*np.ones_like(x) # constant
# x2 = lambda x: np.sin(0.9*x).flatten() # sin





# Sample some input points and noisy versions of the function evaluated at
# these points.

N = 20		# number of training points
n = 40		# number of test points
s = 0.00000	# noise variance
# T = np.random.uniform(-5, 0, size=(N,))
T = np.linspace(-10, -5, N)
# T = np.linspace(-90, 0, N)
T[-1] = 19.6 # set a goal point
# T[-1] = 175 # set a goal point
x = x1(T) + s*np.random.randn(N)
y = x2(T) + s*np.random.randn(N)




# points we're going to make predictions at.
Ttest = np.linspace(-5, 20, n)
#Ttest = np.linspace(0, 180, n)

axis = [-20, 35, -10, 25]
#axis = [-200, 400, -90, 200]

# Build our Gaussian process.
# xkernel = cov.sq_exp_kernel(2.5, 1)
# ykernel = cov.sq_exp_kernel(2.5, 1)
# kernel = cov.matern_kernel(2.28388, 2.52288)
# kernel = cov.linear_kernel(-2.87701)
# xkernel = cov.summed_kernel(cov.sq_exp_kernel(2.5, 1), cov.noise_kernel(0.01))
# ykernel = cov.summed_kernel(cov.sq_exp_kernel(2.5, 1), cov.noise_kernel(0.01))
# Cafeteria Hyperparams (pre-evaluated)
# xkernel = cov.summed_kernel(
# 	cov.matern_kernel(33.542, 47517),
# 	cov.linear_kernel(315.46),
# 	cov.noise_kernel(0.53043)
# )
# ykernel = cov.summed_kernel(
# 	cov.matern_kernel(9.8147, 155.36),
# 	cov.linear_kernel(17299),
# 	cov.noise_kernel(0.61790)
# )
# Cafeteria Hyperparams

xkernel = cov.summed_kernel(
	#cov.sq_exp_kernel(-1),
	cov.matern_kernel(np.exp(1.9128), np.exp(2*5.3844)),
	cov.linear_kernel(np.exp(-.5*-2.8770)),
	cov.noise_kernel(np.exp(2*-0.3170))
)
ykernel = cov.summed_kernel(
	#cov.sq_exp_kernel(-1),
	cov.matern_kernel(np.exp(1.2839), np.exp(2*2.5229)),
	cov.linear_kernel(np.exp(-3.2*-4.8792)),
	cov.noise_kernel(np.exp(2*-0.2407))
)
xgp = GaussianProcess(T, x, Ttest, xkernel)
ygp = GaussianProcess(T, y, Ttest, ykernel)

# PLOTS:

# draw samples from the prior at our test points.
xs = xgp.sample_prior(10)
ys = ygp.sample_prior(10)
pl.figure(1)
pl.plot(xs, ys)
pl.title('Ten samples from the GP prior')

# draw samples from the posterior
ns = 100
xs = xgp.sample(ns)
ys = ygp.sample(ns)

# illustrate the possible paths.
'''pl.figure(2)
pl.subplots_adjust(0.05, 0.1, 0.95, 0.9)

pl.subplot(2,2,1)
pl.plot(x, y, 'yo', ms=8)
ne = 10
pl.plot(xs[:,0:ne], ys[:,0:ne], 'g-')
pl.title('{} samples from the GP posterior'.format(ne))
pl.axis(axis)

pl.subplot(2,2,2)
pl.plot(x, y, 'yo', ms=8)
pl.plot(xs, ys, 'g-')
pl.title('{} samples from the GP posterior'.format(ns))
pl.axis(axis)

pl.subplot(2,2,3)
pl.plot(x, y, 'yo', ms=8)
pl.plot(x1(Ttest), x2(Ttest), 'b-')
pl.plot(xgp.mu, ygp.mu, 'r--', lw=2)
pl.title('Predictive mean and ground truth')
pl.axis(axis)

pl.subplot(2,2,4)
pl.plot(x, y, 'yo', ms=8)
xmean = np.mean(xs, 1)
ymean = np.mean(ys, 1)
pl.plot(xmean, ymean, 'r--', lw=2)
pl.title('Mean of {} samples'.format(ns))
pl.axis(axis)'''

pl.show()
