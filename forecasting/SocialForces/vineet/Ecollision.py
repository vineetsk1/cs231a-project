import numpy as np

def Ecollision(sr, params, vhat, p, v):

	vhat[np.isnan(vhat)] = 0 # Treat NaN as 0

	I = 0

	if p.shape[0] > 1:
		k = np.repmat(p[0,:], p.shape[0]-1, 1) - p[1:, :]
		q = np.repmat(vhat, v.shape[0]-1, 1) - v[1:, :]

		kphi, kr = cart2pol(k[:, 0], k[:, 1])
		_, qr = cart2pol(q[:, 0], q[:, 1])
		kq = np.sum(k * q, axis=1)

		# Energy Eij
		dstar = (k - (kq / (qr ** 2)) * np.ones((1,2)) * q)
		dstar[np.isnan(dstar)] = np.inf
		eij = np.exp(-0.5 * (np.sum(dstar ** 2, 2)) / (np.dot(sr, sr)))

		# Coefficients wd and wf
		phi = kphi - np.atan2(v[0,1], v[1,1]) - pi
		wd = np.exp(-0.5 * (kr ** 2) / (np.dot(params[0], params[0])) )
		wf = (0.5 * (1 + np.cos(phi))) ** params[1]

		I = np.sum(wd * wf * eij)

	return I