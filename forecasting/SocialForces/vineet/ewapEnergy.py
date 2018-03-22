import numpy as np
import numpy.matlib

# EWAPENERGY
# Energy function of individual pedestrians

# E = ewapEnergy(vhat, p, v, ui, zi, params)
# E, dE = ewapEnergy(vhat, p, v, ui, zi, params)

# Input:
# 	vhat:   (1,2) vector of velocity in next time step
# 	p:      (n,2) vectors of pedestrian positions (xi, yi)
#		      p[0,:] is subject of index
# 		      p[1:, :] is other pedestrians
# 	v:	    (n,2) vectors of pedestrian velocities (dxi/dt, dyi/dt)
# 	ui:     comfortable speed for pedestrian i
#   zi:     (1,2) vector of destination for pedestrian i
#   params: parameters of energy function
# Output:
#    E: energy of the input
#   dE: gradient of energy to the input

def ewapEnergy(vhat, p, v, ui, zi, params):
	vhat[np.isnan(vhat)] = 0 # treat NaN as zero
	vq, vr = cart2pol(vhat[0], vhat[1]) # Magnitude and normalization of vhat
	
	# Free flow term
	S = (ui - vr)**2
	
	# Destination term
	z = cart2pol(zi[0] - p[0,0], zi[1] - p[0,1]) # normalized dir to dest
	if vr == 0: # if dir undefined
		vq = z  # set to goal dir
	D = -np.dot(np.asarray([[np.cos(z), np.sin(z)]]), np.asarray([[np.cos(vq)], [np.sin(vq)]]))

	# Interaction term
	I = 0
	if p.shape[0] > 1:
		
		# Initial terms
		k = np.matlib.repmat(p[0, :], p.shape[0]-1, 1) - p[1:, :]
		q = np.matlib.repmat(vhat, v.shape[0]-1, 1) - v[1:, :]

		# Energy Eij
		dstar = (k - (kq / (qr ** 2))) * np.ones((1,2)) * q
		dstar[np.isnan(dstar)] = np.inf
		eij = np.exp(-0.5 * (np.sum(dstar ** 2, axis=1)) / np.dot(params[0], params[0]) )

		# Coefficients wd and wf
		phi = kphi - np.arctan2(v[0,1], v[1,1]) - pi
		wd = np.exp(-0.5 * (kr ** 2) / np.dot(params[1], params[1]))
		wf = (0.5 * (1 + np.cos(phi))) ** params[2]
		wf[2 * np.abs(phi) > pi] = 0

	E = 6*I + np.dot(params[3],S) + np.dot(params[4],D)
	
	dE = None # TODO

	return E, dE