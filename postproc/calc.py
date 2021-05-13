# -*- coding: utf-8 -*-
"""
@author: B. Font Garcia
@description: Module containing functions to calculate derivatives, averages, decompositions, vorticity.
@contact: b.fontgarcia@soton.ac.uk
"""
# Imports
import numpy as np
import warnings

# Functions
def avg_z(u):
	"""
	Return the span-wise spatial average of a three-dimensional field.
	If the passed u is spanwise periodic use trapz()/(n-1). Otherwise mean().
	:param u: The field to be spanwise-averaged.
	:return: Return the span-wise spatial average of a three-dimensional field.
	"""
	from scipy.integrate import trapz, simps
	if not len(u.shape)==3:
		warnings.warn("Field not 3D. Returning same array.")
		return u
	else:
		if np.array_equal(u[..., 0], u[..., -1]): # Periodic on last axis
			return trapz(u, axis=2)/(u.shape[2]-1)
			# return simps(u, axis=2)/(u.shape[2]-1)
		else:
			print('hi')
			return u.mean(axis=2)


def avg_z2(u):
	"""
	Return the span-wise spatial average of a three-dimensional field using Simpson rule.
	If the passed u is spanwise periodic use trapz()/(n-1). Otherwise mean().
	:param u: The field to be spanwise-averaged.
	:return: Return the span-wise spatial average of a three-dimensional field.
	"""
	from scipy.integrate import trapz, simps
	if not len(u.shape)==3:
		warnings.warn("Field not 3D. Returning same array.")
		return u
	else:
		sum = np.zeros(u.shape[:2])
		for i in np.arange(u.shape[0]):
			print(i)
			for j in np.arange(u.shape[1]):
				for k in np.arange(4, u.shape[2]-4):
					sum[i,j] = sum[i,j] + u[i,j,k]
				sum[i,j] = sum[i,j] + 1/48*(17*u[i,j,0]+59*u[i,j,1]+43*u[i,j,2]+49*u[i,j,3]+
											49*u[i,j,-4]+43*u[i,j,-3]+59*u[i,j,-2]+17*u[i,j,-1])
				sum[i,j] = sum[i,j]/(u.shape[2]-1)
		return sum


def avg_z_1D(u):
	"""
	Return the span-wise spatial average of a three-dimensional field.
	If the passed u is spanwise periodic use trapz()/(n-1). Otherwise mean().
	:param u: The field to be spanwise-averaged.
	:return: Return the span-wise spatial average of a three-dimensional field.
	"""
	from scipy.integrate import trapz, simps, quadrature
	from scipy.interpolate import Rbf,InterpolatedUnivariateSpline
	N = len(u)
	print(N)
	# Mean
	a1 = np.mean(u)
	# Rectangular rule
	a2 = 0
	for i in range(0,len(u)-1):
		a2 = a2 + (u[i]+u[i+1])/2
	a2 = a2/(N-1)
	# Trapz
	a3 = trapz(u)/(u.size-1)
	# Simps
	a4 = simps(u)/(u.size-1)
	# Simps2
	a5 = 0
	for i in range(4,len(u)-4):
		a5 = a5 + u[i]
	a5 = a5 + 1/48*(17*u[0]+59*u[1]+43*u[2]+49*u[3]+49*u[-4]+43*u[-3]+59*u[-2]+17*u[-1])
	a5 = a5/(N-1)
	# Gaussian quad with rbf
	f = InterpolatedUnivariateSpline(np.arange(N),u)
	a6, err = quadrature(f, 0, N-1)
	a6 = a6/(N-1)
	return a1, a2, a3, a4, a5, a6


def make_periodicZ(u, **kwargs): # Methods t = True (add layer), t = False (substitue last layer with 1st layer)
	add = kwargs.get('add', True)
	if add:
		u_temp = np.zeros((u.shape[0], u.shape[1], u.shape[2] + 1))
		u_temp[..., :-1], u_temp[..., -1] = u, u[..., 0]
		u = u_temp
	else:
		u[..., -1] = u[..., 0]
	return u


def make_periodic(a, axis):
	b = np.expand_dims(np.take(a, indices=0, axis=axis), axis=axis)
	return np.concatenate((a, b), axis=axis)


def decomp_z(u):
	"""
	:param u: field to be decomposed in z direction.
	:return: the average and the fluctuating parts of a three-dimensional field spatially averaged
	in the span-wise direction.
	"""
	if not len(u.shape)==3:
		raise ValueError("Fields must be three-dimensional")
	else:
		u_avg = avg_z(u)
		u_f = u-u_avg[:, :, None]
		return u_avg, u_f


def ddx(u, x=None, acc=1):
	"""
	:param u: n-dimensional field.
	:return: the first- or second-acc derivative in the i direction of (n>=1 dimensional) field.
	"""
	if x is not None:
		if acc == 2:
			return np.gradient(u, x, axis=0, edge_order=2)
		else:
			raise ValueError('Only 2nd accuracy acc considered when x is included')
	else:
		if acc == 2:
			return np.gradient(u, axis=0, edge_order=2)
		elif acc == 1:
			a = np.diff(u, n=1, axis=0)
			if u.ndim == 1:
				return np.append(0, a)
			elif u.ndim == 2:
				return np.concatenate((np.zeros(u.shape[1])[None, :,], a), axis=0)
			elif u.ndim == 3:
				return np.concatenate((np.zeros(u.shape[1:])[None, :, :], a), axis=0)
			else:
				raise ValueError('Only arrays with dimensions <=3 considered.')
		else:
			raise ValueError('Only 1st or 2nd accuracy acc considered.')


def ddy(u, y=None, acc=1):
	"""
	:param u: n-dimensional field.
	:return: the first-acc derivative in the j direction of (n>=2 dimensional) field.
	"""
	if y is not None:
		if acc == 2:
			return np.gradient(u, y, axis=1, edge_order=2)
		else:
			raise ValueError('Only 2nd accuracy acc considered when y is included')
	else:
		if acc == 2:
			return np.gradient(u, axis=1, edge_order=2)
		elif acc == 1:
			a = np.diff(u, n=1, axis=1)
			if u.ndim == 2:
				return np.concatenate((np.zeros(u.shape[0])[:, None], a), axis=1)
			elif u.ndim == 3:
				return np.concatenate((np.zeros((u.shape[0], u.shape[2]))[:, None, :], a), axis=1)
			else:
				raise ValueError('Only arrays with dimensions >=2 considered.')
		else:
			raise ValueError('Only 1st or 2nd accuracy acc considered.')


def ddz(u, z=None, acc=1):
	"""
	:param u: n-dimensional field.
	:return: the first-acc derivative in the j direction of (n>=2 dimensional) field.

	"""
	if not np.array_equal(u[..., 0], u[..., -1]):  # Field is not periodic
		if z is not None:
			if acc == 2:
				return np.gradient(u, z, axis=2, edge_order=2)
			else:
				raise ValueError('Only 2nd accuracy acc considered when z is included')
		else:
			if acc == 2:
				return np.gradient(u, axis=2, edge_order=2)
			elif acc == 1:
				a = np.diff(u, n=1, axis=2)
				if u.ndim == 3:
					return np.concatenate((np.zeros((u.shape[0], u.shape[1]))[:, :, None], a), axis=2) # Return same shape with zeros in k=0
				else:
					raise ValueError('Only arrays with dimensions =3 considered.')
			else:
				raise ValueError('Only 1st or 2nd accuracy order considered.')
	else: # Field is periodic
		if z is not None:
			if acc == 2:
				return np.gradient(u, z, axis=2, edge_order=2)
			else:
				raise ValueError('Only 2nd accuracy acc considered when z is included')
		else:
			u_temp = np.zeros((u.shape[0], u.shape[1], u.shape[2] + 2))
			u_temp[:, :, 1:-1], u_temp[:, :, 0], u_temp[:, :, -1] = u, u[:, :, -2], u[:, :, 1]
			del u
			if z is not None:
				if acc == 2:
					dudz = np.gradient(u_temp, z, axis=2, edge_order=2)
					return dudz[:, :, 1:-1]
				else:
					raise ValueError('Only 2nd accuracy acc considered when z is included')
			else:
				if acc == 1:
					dudz = np.diff(u_temp, n=1, axis=2)
					return dudz[..., :-1]
				elif acc == 2:
					dudz = np.gradient(u_temp, axis=2, edge_order=2)
				else:
					raise ValueError('Only 1st or 2nd accuracy order considered.')
				return dudz[:, :, 1:-1]

# def ddz(u, z=None):
#	 """
#	 :param u: n-dimensional field.
#	 :return: the first-order derivative in the k direction of (n>=3 dimensional) field.
#	 """
#	 if np.array_equal(u[..., 0], u[..., -1]):  # Periodic on last axis
#		 u_temp = np.zeros((u.shape[0], u.shape[1], u.shape[2]+2))
#		 u_temp[:, :, 1:-1], u_temp[:, :, 0], u_temp[:, :, -1] = u, u[:, :, -1], u[:, :, 0]
#		 del u
#		 if z is not None:
#			 dudz = np.gradient(u_temp, z, axis=2, edge_order=2)
#		 else:
#			 dudz = np.gradient(u_temp, axis=2, edge_order=2)
#		 del u_temp
#		 return dudz[:, :, 1:-1]
#	 else:
#		 if z is not None:
#			 return np.gradient(u, z, axis=2, edge_order=2)
#		 else:
#			 return np.gradient(u, axis=2, edge_order=2)


def vortZ(u, v, **kwargs):
	"""
	:param u: x component of the velocity vector field.
	:param v: y component of the velocity vector field.
	:return: The z-vorticity of a two-dimensional velocity vector field
	"""
	# if not (len(u.shape)==2 and len(v.shape)==2):
	#	 raise ValueError("Fields must be two-dimensional")
	# else:
	x = kwargs.get('x', None)
	y = kwargs.get('y', None)
	acc = kwargs.get('acc', 1)
	return ddx(v, x, acc)-ddy(u, y, acc)


def vort(u, v, w, **kwargs):
	"""
	:param u: x component of the velocity vector field.
	:param v: y component of the velocity vector field.
	:param w: z component of the velocity vector field.

	:return: the three components of the vorticity of a three-dimensional velocity vector field.
	"""
	if not (len(u.shape)==3 and len(v.shape)==3 and len(w.shape)==3):
		raise ValueError("Fields must be three-dimensional")
	else:
		x = kwargs.get('x', None)
		y = kwargs.get('y', None)
		z = kwargs.get('z', None)
		return ddy(w, y) - ddz(v, z), ddz(u, z) - ddx(w, x), ddx(v, x) - ddy(u, y)

def grad(u):
	"""
	Return the gradient of a n-dimensinal array (scalar field)
	:param u: input scalar array
	:return: vector field array, gradient of u
	"""
	return np.array(np.gradient(u, edge_order=2))


def div(*args):
	dd = [ddx, ddy, ddz]
	res = np.zeros(args[0].shape)
	for i, a in enumerate(args):
		res += dd[i](a)
	return res


def _J(u, v, w):
	"""
	Calculate the velocity gradient tensor
	:param u: Horizontal velocity component
	:param v: Vertical velocity component
	:param w: Spanwise velocity component
	:return: np.array (tensor) of the velocity gradient
	"""
	a11, a12, a13 = ddx(u), ddy(u), ddz(u)
	a21, a22, a23 = ddx(v), ddy(v), ddz(v)
	a31, a32, a33 = ddx(w), ddy(w), ddz(w)
	return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])


def S(u, v, w):
	J = _J(u, v, w)
	return 0.5*(J + np.transpose(J, (1,0,2,3,4)))


def R(u, v, w):
	J = _J(u, v, w)
	return 0.5*(J - np.transpose(J, (1,0,2,3,4)))


def Q(u, v, w):
	J = _J(u, v, w)
	S = 0.5*(J + np.transpose(J, (1,0,2,3,4)))
	R = 0.5*(J - np.transpose(J, (1,0,2,3,4)))
	S_mag = np.linalg.norm(S, ord='fro', axis=(0,1))
	S_mag2 = np.sqrt(np.trace(np.dot(S, S.T)))
	print(np.sum(S_mag))
	print(np.sum(S_mag2))
	R_mag = np.linalg.norm(R, ord='fro', axis=(0,1))

	print(R_mag.shape)

	Q = 0.5*(R_mag**2-S_mag**2)
	# Q = np.clip(Q, 0, None)
	return Q

def crop(u, window, **kwargs):
	scaling = kwargs.get('scaling', 1)
	shift = kwargs.get('shift', (0, 0))
	N, M = u.shape[0], u.shape[1]

	# Create uniform grid
	if 'grid' in kwargs:
		grid = kwargs['grid']
		x, y = grid[0] / scaling + shift[0], grid[1] / scaling + shift[1]
	elif 'x' in kwargs and 'y' in kwargs:
		x = kwargs.get('x') / scaling + shift[0]
		y = kwargs.get('y') / scaling + shift[1]
		x, y = np.meshgrid(x, y)
	elif 'x_lims' in kwargs and 'y_lims' in kwargs:
		xlims = kwargs.get('x_lims')
		ylims = kwargs.get('y_lims')
		x, y = np.linspace(xlims[0] / scaling, xlims[1] / scaling, N), np.linspace(ylims[0] / scaling, ylims[1] / scaling, M)
		x, y = x + shift[0], y + shift[1]
		x, y = np.meshgrid(x, y)
	else:
		xmin, xmax = 0, N - 1
		ymin, ymax = -M / 2, M / 2 - 1
		x, y = np.linspace(xmin / scaling, xmax / scaling, N), np.linspace(ymin / scaling, ymax / scaling, M)
		x, y = x + shift[0], y + shift[1]
		x, y = np.meshgrid(x, y)

	x_args = np.where(np.logical_and(np.any(x > window[0][0], axis=0), np.any(x < window[0][1], axis=0)))[0]
	y_args = np.where(np.logical_and(np.any(y > window[1][0], axis=1), np.any(y < window[1][1], axis=1)))[0]
	return u[x_args][:, y_args]


def map_cyl(grid2D, r, eps):
	x, y = grid2D[0], grid2D[1]
	r_grid = np.sqrt(x**2+y**2)

	indices = np.where((r_grid>=r*(1+0.5*eps)) & (r_grid<=r*(1+1.5*eps)))
	r_grid = np.zeros(x.shape)
	r_grid[indices] = 1
	return r_grid, indices


def map_normal(grid2D, r, r_max, angle):
	x, y = grid2D[0], grid2D[1]
	r_grid = np.sqrt(x**2+y**2)
	atan_grid = np.arctan2(y, x)*180/np.pi

	angle_eps = 0.5

	# indices = np.where((r_grid<=r_max) & (np.abs(atan_grid-angle)<angle_eps))
	indices = np.where((r_grid<=r_max) & close(atan_grid, angle, angle_eps))
	result = np.zeros(x.shape)
	result[indices] = 1
	return result, indices


def separation_points(q, alphas, eps=0.0005):
	"""
	:param q: 2D scalar field (quantity) we analyse
	:param l: list of tuples. Each tuples is: ((i,j), alpha))
	:return: upper and lower separation points
	"""
	alpha_u, alpha_l = None, None
	# find upper separation point
	for (idx, alpha) in sorted(alphas, key=lambda tup: tup[1])[80:]:
		if -eps<=q[idx] and q[idx]<=eps:
			alpha_u = alpha
			break
	# find lower separation point
	for (idx, alpha) in sorted(alphas, key=lambda tup: tup[1], reverse=True)[80:]:
		if -eps<=q[idx] and q[idx]<=eps:
			alpha_l = alpha
			break
	return alpha_u, alpha_l-360


def separation_points2(q, alphas, eps=0.1):
	"""
	:param q: 2D scalar field (quantity) we analyse
	:param l: list of tuples. Each tuples is: ((i,j), alpha))
	:return: upper and lower separation points
	"""
	alpha_u, alpha_l = None, None
	l_upper = sorted(alphas, key=lambda tup: tup[1])[100:]
	l_lower = sorted(alphas, key=lambda tup: tup[1], reverse=True)[100:]

	# find upper separation point
	for i, (idx, alpha) in enumerate(l_upper):
		if q[idx]<eps:
			alpha_u = alpha
			break

	# find lower separation point
	for i, (idx, alpha) in enumerate(l_lower):
		if q[idx]<eps:
			alpha_l = alpha
			break
	return alpha_u, alpha_l-360


def corr(a,b):
	am = np.mean(a)
	bm = np.mean(b)
	ac = a - am  # a centered
	bc = b - bm  # b centered

	cov_ab = np.mean(ac * bc)
	cov_aa = np.mean(ac ** 2)
	cov_bb = np.mean(bc ** 2)
	return cov_ab / np.sqrt(cov_aa * cov_bb)


def close(a, b, eps):
	return np.abs(a - b) < eps