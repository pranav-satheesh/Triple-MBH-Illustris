import numpy as np

#From Bonetti et al. 2018
############################################################################################
############################################################################################

# grid points, i.e. points (m1, qout, qin) at which we performed simulations
# qin = m2/m1, qout = m3/(m1+m2), where m1 and m2 are part of the original binary

m1_bonetti = np.array([5,6,7,8,9,10]) #log10 m1

qout_bonetti = np.array([-1.5,-1.,-0.5,0.0]) #log10 qout
qout_bigp_bonetti = np.array([0.5,1.0]) #log10 qout, !simulations performed only for m1 = 10^9!

qin_bonetti = np.array([-1.5,-1.,-0.5,0.0]) #log10 qin

# 3d arrays containing merger fractions (as %) for prompt mergers
# indexing is (m1, qout, qin), e.g. prompt_merger_frac12[i_m1][i_qout][i_qin]

prompt_merger_frac12 = np.array( 
[ [ [1.9,0.6,1.9,0.0],  [19.2,8.3,2.6,5.8],   [34.6,35.3,28.2,19.2], [23.1,40.4,30.8,16.7] ], 
  [ [4.5,1.3,0.6,0.6],  [11.5,9.0,3.2,1.9],   [25.6,38.5,19.9,17.9], [39.7,43.6,26.3,14.7] ], 
  [ [5.1,3.2,0.0,1.3],  [23.1,9.6,10.3,6.4],  [23.7,32.1,23.1,24.4], [23.7,22.4,23.1,14.7] ], 
  [ [9.6,4.5,1.3,0.0],  [14.1,9.6,10.3,5.8],  [28.2,22.4,25.6,25.0], [14.7,21.8,23.1,19.2] ], 
  [ [9.0,4.5,2.6,1.3],  [23.7,16.0,12.8,5.8], [9.0,24.4,26.3,26.9],  [25.0,25.6,17.3,13.5] ], 
  [ [21.2,7.7,4.5,0.6], [33.3,21.8,16.7,9.6], [19.9,36.5,37.2,32.7], [32.1,18.6,19.2,25.6] ] ] )

prompt_merger_frac13 = np.array( 
[ [ [1.9,0.0,0.0,0.0], [1.3,0.6,0.6,0.0],  [0.0,0.6,0.6,3.8],   [0.0,0.0,0.6,4.5] ], 
  [ [0.6,0.6,0.6,0.0], [0.0,1.3,0.6,0.6],  [0.0,0.0,1.3,6.4],   [0.0,0.0,1.3,8.3] ], 
  [ [4.5,0.6,0.6,0.0], [1.9,2.6,0.0,1.3],  [3.2,1.3,6.4,3.8],   [5.8,1.9,0.0,6.4] ], 
  [ [8.3,3.2,1.3,0.6], [2.6,6.4,1.3,0.0],  [0.6,6.4,9.0,9.6],   [1.3,0.0,3.2,10.3] ], 
  [ [5.1,1.9,0.6,0.0], [3.2,5.8,2.6,0.0],  [1.3,3.2,14.1,12.2], [0.0,0.0,1.9,14.1] ], 
  [ [9.6,7.7,1.3,1.3], [6.4,13.5,3.8,1.3], [4.5,8.3,14.7,16.7], [1.3,4.5,7.7,18.6] ] ] )

prompt_merger_frac23 = np.array(
[ [ [0.0,0.0,0.0,0.0], [0.0,0.6,0.0,0.0], [0.0,0.0,0.6,5.1],  [1.3,0.0,3.8,1.9] ], 
  [ [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.6], [0.0,0.0,0.0,4.5],  [1.9,0.6,2.6,5.8] ], 
  [ [0.0,0.0,0.0,0.0], [0.0,0.0,1.3,0.6], [0.6,0.6,0.6,5.1],  [0.0,0.6,7.1,6.4] ], 
  [ [0.0,0.0,0.0,0.0], [0.0,0.0,0.6,0.0], [0.6,1.3,1.9,10.9], [2.6,1.9,8.3,11.5] ], 
  [ [0.0,0.0,0.0,0.0], [0.0,0.6,0.6,0.6], [0.6,1.3,3.8,14.1], [1.9,7.1,4.5,15.4] ], 
  [ [0.0,0.0,0.0,0.6], [0.0,1.9,1.3,0.0], [0.0,3.2,1.9,10.9], [5.8,4.5,7.1,15.4] ] ] )

# 2d arrays containing merger fractions (as %) for prompt mergers in case of a massive perturber, i.e. qout>1
# indexing is (qout, qin), e.g. prompt_merger_frac_bigp12[i_qout][i_qin]

prompt_merger_frac_bigp12 = np.array(
[ [27.6,19.9,24.4,21.2], [26.3,39.1,34.0,26.9] ] )

prompt_merger_frac_bigp13 = np.array(
[ [1.9,2.6,3.2,7.7], [0.0,0.6,1.3,1.9] ] )

prompt_merger_frac_bigp23 = np.array(
[ [2.6,1.3,4.5,7.7], [0.6,4.5,5.1,5.1] ] )

############################################################################################

# 3d arrays containing merger fractions (as %) for delayed mergers
# indexing is (m1, qout, qin), e.g. delayed_merger_frac12[i_m1][i_qout][i_qin]

delayed_merger_frac12 = np.array(
[ [ [0.6,1.3,0.0,0.6], [0.0,1.9,3.8,1.3], [0.0,0.0,1.3,2.6], [0.0,0.0,0.0,0.0] ], 
  [ [1.3,1.3,1.3,0.6], [0.0,3.8,0.0,1.3], [0.0,1.3,0.0,2.6], [0.0,0.0,0.0,0.0] ],
  [ [3.8,2.6,0.6,0.6], [1.3,4.5,5.1,2.6], [1.9,0.6,1.9,7.1], [0.0,0.0,0.0,0.0] ],
  [ [3.2,5.1,3.2,3.2], [5.8,7.1,7.1,5.1], [0.0,0.6,5.8,7.1], [0.0,0.0,0.0,0.0] ],
  [ [17.3,16.0,7.1,5.1], [10.3,7.7,12.2,8.3], [0.0,5.8,5.1,12.8], [0.0,0.0,0.0,0.6] ],
  [ [26.3,13.5,10.9,5.1], [5.8,11.5,11.5,12.2], [2.6,1.9,8.3,6.4], [0.0,0.6,0.0,0.6] ],
] )

delayed_merger_frac13 = np.array(
[ [ [1.9,0.0,0.0,0.0], [0.0,1.3,0.0,0.0], [1.3,1.3,1.9,0.0], [3.8,0.0,0.0,0.0] ],
  [ [0.0,0.6,0.0,0.0], [0.0,2.6,0.0,0.0], [2.6,1.3,1.3,0.6], [0.0,0.6,1.9,1.3] ],
  [ [2.6,0.0,0.0,0.0], [0.6,1.9,0.6,0.0], [5.8,4.5,1.9,1.3], [1.9,3.2,0.6,5.1] ],
  [ [1.9,0.0,0.0,0.0], [2.6,5.8,0.0,0.0], [1.9,4.5,5.8,1.3], [2.6,1.9,4.5,4.5] ],
  [ [3.2,0.6,0.0,0.0], [3.2,9.0,1.9,0.0], [3.8,5.8,10.9,2.6], [0.0,0.6,5.1,9.6] ],
  [ [5.1,1.9,0.0,0.0], [5.8,11.5,0.6,0.0], [5.1,8.3,7.1,1.3], [7.7,5.8,5.8,7.7] ], 
] )

delayed_merger_frac23 = np.array(
[ [ [0.6,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,1.3] ], 
  [ [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,1.9], [0.6,0.0,0.0,0.6] ],
  [ [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.6,0.0], [0.0,0.0,0.0,6.4] ],
  [ [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,3.2], [0.0,0.6,0.6,4.5] ],
  [ [0.0,0.0,0.0,0.0], [0.0,0.6,0.0,0.0], [0.0,0.0,0.0,1.3], [0.0,0.6,0.6,11.5] ],
  [ [0.0,0.0,0.0,0.0], [0.0,0.0,0.6,0.0], [0.0,0.0,0.6,1.3], [0.0,0.6,1.3,6.4] ],
] )

# 2d arrays containing merger fractions (as %) for delayed mergers in case of a massive perturber, i.e. qout>1
# indexing is (qout, qin), e.g. delayed_merger_frac_bigp12[i_qout][i_qin]

delayed_merger_frac_bigp12 = np.array(
[ [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0] ] )

delayed_merger_frac_bigp13 = np.array(
[ [0.9,1.3,0.9,3.4], [0.0,0.0,1.7,2.1] ] )

delayed_merger_frac_bigp23 = np.array(
[ [1.3,2.1,0.4,3.8], [3.4,3.8,0.9,4.3] ] )

############################################################################################
############################################################################################
def linear_interp(val_to_int, X_val, values):
	'''
	Function that performs linear interpolation.

	input parameters:
		val_to_int -> point (float) at which interpolation is needed, Xi
		X_val -> 1d array of floats (X grid points, Xg)
		values -> 1d array of floats (values at grid points, F(Xg))

	return:
		value at Xi, i.e. F(Xi) 
	'''

	len1 = len(X_val)
	I = len1-1

	####################################
	# find closest grid point
	for i in range(len1):
		if val_to_int < X_val[i]:
			I = i-1
			break  
	
	if I == -1: # if Xi < X_val[0], then Xi = X_val[0]
		I = 0
		xd = 0.
	elif I == len1-1: # if Xi > X_val[-1] then Xi = X_val[-1]
		I = I-1
		xd = 1.
	else:
		xd = (val_to_int-X_val[I])/(X_val[I+1]-X_val[I])

	####################################
	
	c = values[I]*(1.-xd) + values[I+1]*xd

	return c

############################################################################################
def bilinear_interp(val_to_int, X_val, Y_val, values):
	'''
	Function that performs bilinear interpolation.

	input parameters:
		val_to_int -> 1d array of floats at which interpolation is needed, (Xi, Yi)
		X_val -> 1d array of floats (X grid points, Xg)
		Y_val -> 1d array of floats (Y grid points, Yg)
		values -> 2d array of floats (values at grid points, F(Xg, Yg))

	return:
		value (float) at (Xi, Yi), i.e. F(Xi, Yi)
	'''

	len1 = len(X_val)
	I = len1-1
	len2 = len(Y_val)
	J = len2-1

	####################################
	for i in range(len1):
		if val_to_int[0] < X_val[i]:
			I = i-1
			break

	if I == -1:
		I = 0
		xd = 0.
	elif I == len1-1:
		I = I-1
		xd = 1.  
	else:
		xd = (val_to_int[0]-X_val[I])/(X_val[I+1]-X_val[I])

	####################################
	for j in range(len2):
		if val_to_int[1] < Y_val[j]:
			J = j-1
			break
	  
	if J == -1:
		J = 0
		yd = 0.
	elif J == len2-1:
		J = J-1
		yd = 1.
	else:
		yd = (val_to_int[1]-Y_val[J])/(Y_val[J+1]-Y_val[J])

	####################################

	c0 = values[I][J]  *(1.-xd) + values[I+1][J]  *xd
	c1 = values[I][J+1]*(1.-xd) + values[I+1][J+1]*xd
	
	c = c0*(1.-yd) + c1*yd

	return c

############################################################################################
def trilinear_interp(val_to_int, X_val, Y_val, Z_val, values):
	'''
	Function that performs trilinear interpolation.

	input parameters:
		val_to_int -> 1d array at which interpolation is needed, (Xi, Yi, Zi)
		X_val -> 1d array of floats (X grid points, Xg)
		Y_val -> 1d array of floats (Y grid points, Yg)
		Z_val -> 1d array of floats (Z grid points, Zg)
		values -> 3d array of floats (values at grid points, F(Xg, Yg, Zg))

	return:
		value (float) at (Xi, Yi, Zi), i.e. F(Xi, Yi, Zi)
	'''

	len1 = len(X_val)
	I = len1-1
	len2 = len(Y_val)
	J = len2-1
	len3 = len(Z_val)
	K = len3-1

	####################################
	for i in range(len1):
		if val_to_int[0] < X_val[i]:
			I = i-1
			break

	if I == -1:
		I = 0
		xd = 0.
	elif I == len1-1:
		I = I-1
		xd = 1.  
	else:
		xd = (val_to_int[0]-X_val[I])/(X_val[I+1]-X_val[I])

	####################################
	for j in range(len2):
		if val_to_int[1] < Y_val[j]:
			J = j-1
			break
	  
	if J == -1:
		J = 0
		yd = 0.
	elif J == len2-1:
		J = J-1
		yd = 1.
	else:
		yd = (val_to_int[1]-Y_val[J])/(Y_val[J+1]-Y_val[J])

	####################################
	for k in range(len3):
		if val_to_int[2] < Z_val[k]:
			K = k-1
			break
	  
	if K == -1:
		K = 0
		zd = 0.
	elif K == len3-1:
		K = J-1
		zd = 1.
	else:
		zd = (val_to_int[2]-Z_val[K])/(Z_val[K+1]-Z_val[K])

	####################################

	c00 = values[I][J][K]     *(1.-xd) + values[I+1][J][K]     *xd
	c01 = values[I][J][K+1]   *(1.-xd) + values[I+1][J][K+1]   *xd
	c10 = values[I][J+1][K]   *(1.-xd) + values[I+1][J+1][K]   *xd
	c11 = values[I][J+1][K+1] *(1.-xd) + values[I+1][J+1][K+1] *xd

	c0 = c00*(1.-yd) + c10*yd
	c1 = c01*(1.-yd) + c11*yd
	
	c = c0*(1.-zd) + c1*zd

	return c

############################################################################################

# examples:

# (m1, qout, qin) must be in this order to match that of prompt_merger_frac, log values used
#print trilinear_interp([6.,-1.1,-0.25], m1, qout, qin, prompt_merger_frac12) 

# (qout_bigp, qin) must be in this order to match that of prompt_merger_frac_bigp, log values used
#print bilinear_interp([0.5,-1], qout_bigp, qin, prompt_merger_frac_bigp12)



'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig2=plt.figure()
ax2 = fig2.gca(projection='3d')

ax2.plot([-1.5],[-0.5],[trilinear_interp([6.,-1.5,-0.5], m1, qout, qin, prompt_merger_frac12)],"ko" )
ax2.plot([-1],[-0.5],[trilinear_interp([6.,-1.,-0.5], m1, qout, qin, prompt_merger_frac12)],"ko" )
ax2.plot([-1.5],[0.0],[trilinear_interp([6.,-1.5,0.0], m1, qout, qin, prompt_merger_frac12)],"ko" )
ax2.plot([-1],[0.0],[trilinear_interp([6.,-1,0.0], m1, qout, qin, prompt_merger_frac12)],"ko" )

ax2.plot([-1.1],[-0.25],[trilinear_interp([6.,-1.1,-0.25], m1, qout, qin, prompt_merger_frac12)],"ro" )

plt.show()
'''



