import numpy as np
from scipy.interpolate import RegularGridInterpolator
import random
import matplotlib.pyplot as plt
import scienceplots

# import os
# import sys
# print(os.path.dirname( __file__ ))
# print(sys.path)

filename = "Data/paper2_sorted_data.txt"
M1,qin,qout,a,b,c,d = np.loadtxt(filename,ußnpack=True)

qin = [0.03162,0.1,0.3162,1]
qout = [0.03162,0.1,0.3162,1]
M1 = [10**10,10**9,10**8,10**7,10**6,10**5]

A = np.zeros((np.size(M1),np.size(qin),np.size(qout)))
B = np.zeros((np.size(M1),np.size(qin),np.size(qout)))
C = np.zeros((np.size(M1),np.size(qin),np.size(qout)))
D = np.zeros((np.size(M1),np.size(qin),np.size(qout)))

for i in range(np.size(M1)):
    a_M = a[16*i:16*(i+1)]
    b_M = b[16*i:16*(i+1)]
    c_M = c[16*i:16*(i+1)]
    d_M = d[16*i:16*(i+1)]
    for j in range(np.size(qin)):
        for k in range(np.size(qout)):
            A[i,j,k] = a_M[j*4+k]
            B[i,j,k] = b_M[j*4+k]
            C[i,j,k] = c_M[j*4+k]
            D[i,j,k] = d_M[j*4+k]
            
fA = RegularGridInterpolator((M1,qin,qout), A)
fB = RegularGridInterpolator((M1,qin,qout), B)
fC = RegularGridInterpolator((M1,qin,qout), C)
fD = RegularGridInterpolator((M1,qin,qout), D)

M1_ill,qin_ill,qout_ill = np.loadtxt("Data/triple-masses-from-illustris.txt",unpack=True)

def interp_result(M1,qin,qout,f_choice):
    #fchoice is between a,b,c,d
    
    if(f_choice == 'a' or f_choice == 'A'):
        if(M1 > 10**10):
            M1 = 10**10
        if(qout > 1):
            qout = 1
        if(qout < 0.03162):
            qout = 0.03162
        if(qin < 0.03162):
            qin = 0.03162
        return fA([M1,qin,qout])[0]
    
    if(f_choice == 'b' or f_choice == 'B'):
        if(M1 > 10**10):
            M1 = 10**10
        if(qout > 1):
            qout = 1
        if(qout < 0.03162):
            qout = 0.03162
        if(qin < 0.03162):
            qin = 0.03162
        return fB([M1,qin,qout])[0]

    if(f_choice == 'c' or f_choice == 'C'):
        if(M1 > 10**10):
            M1 = 10**10
        if(qout > 1):
            qout = 1
        if(qout < 0.03162):
            qout = 0.03162
        if(qin < 0.03162):
            qin = 0.03162
        return fC([M1,qin,qout])[0]

    if(f_choice == 'd' or f_choice == 'D'):
        if(M1 > 10**10):
            M1 = 10**10
        if(qout > 1):
            qout = 1
        if(qout < 0.03162):
            qout = 0.03162
        if(qin < 0.03162):
            qin = 0.03162
        return fD([M1,qin,qout])[0]

    
    #to plot the meshgrid and matching Bonetti et. al 2
merger_p = np.zeros((len(qin),len(qout)))

for i in range(len(qin)):
    for j in range(len(qout)):
        merger_p[i,j] = A[0,i,j] + B[0,i,j] + C[0,i,j]

X,Y = np.meshgrid(np.log10(qin),np.log10(qout))
plt.style.use('science')
plt.contour(Y,X, merger_p, colors='black');
plt.contourf(Y,X, merger_p, 20, cmap='viridis')
plt.xlabel("$\log q_{in}$")
plt.ylabel("$\log q_{out}$")
plt.colorbar()
plt.savefig("Figures/meshplot.pdf")
        
     
            
            