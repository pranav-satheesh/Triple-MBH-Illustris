import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def hot_angle():

    alpha = 2.018
    beta = 5.244
    theta = stats.beta(alpha,beta).rvs()

    return theta

def cold_angle():

    alpha = 2.544
    beta = 19.527
    theta = stats.beta(alpha,beta).rvs()

    return theta

def random_high():

    #random spin orientation with a high spin magnitued (fixed)

    a1 = 0.9
    a2 = 0.9

    theta1 = np.arccos(stats.uniform(-1,2).rvs())
    theta2 = np.arccos(stats.uniform(-1,2).rvs())

    phi1 = 2 * np.pi * np.random.uniform(0,1)
    phi2 = 2 * np.pi * np.random.uniform(0,1)

    S1x = a1 * np.sin(theta1) * np.cos(phi1)
    S1y =  a1 * np.sin(theta1) * np.sin(phi1)
    S1z = a1 * np.cos(theta1)

    S2x = a2 * np.sin(theta2) * np.cos(phi2)
    S2y = a2 * np.sin(theta2) * np.sin(phi2)
    S2z = a2 * np.cos(theta2)

    S1_vec = np.array([S1x,S1y,S1z])
    S2_vec = np.array([S2x,S2y,S2z])

    return S1_vec,S2_vec

def random_dry():

    # random spin orientation with a dry spin magnitude dictated by a beta distribution

    alpha = 8
    beta = 4

    a1 = stats.beta(alpha,beta).rvs()
    a2 = stats.beta(alpha,beta).rvs()
    
    theta1 = np.arccos(stats.uniform(-1,2).rvs())
    theta2 = np.arccos(stats.uniform(-1,2).rvs())

    phi1 = 2 * np.pi * np.random.uniform(0,1)
    phi2 = 2 * np.pi * np.random.uniform(0,1)

    S1x = a1 * np.sin(theta1) * np.cos(phi1)
    S1y =  a1 * np.sin(theta1) * np.sin(phi1)
    S1z = a1 * np.cos(theta1)

    S2x = a2 * np.sin(theta2) * np.cos(phi2)
    S2y = a2 * np.sin(theta2) * np.sin(phi2)
    S2z = a2 * np.cos(theta2)

    S1_vec = np.array([S1x,S1y,S1z])
    S2_vec = np.array([S2x,S2y,S2z])

    return S1_vec,S2_vec


def hot():

    #"Hot" accretion model (partially aligned spins)
    #Based on fits in Lousto et al. 2012

    alpha = 3.212
    beta = 1.563
    a1 = stats.beta(alpha,beta).rvs()
    a2 = stats.beta(alpha,beta).rvs()

    theta1 = hot_angle()
    theta2 = hot_angle()

    phi1 = 2 * np.pi * np.random.uniform(0,1)
    phi2 = 2 * np.pi * np.random.uniform(0,1)

    S1x = a1 * np.sin(theta1) * np.cos(phi1)
    S1y =  a1 * np.sin(theta1) * np.sin(phi1)
    S1z = a1 * np.cos(theta1)

    S2x = a2 * np.sin(theta2) * np.cos(phi2)
    S2y = a2 * np.sin(theta2) * np.sin(phi2)
    S2z = a2 * np.cos(theta2)

    S1_vec = np.array([S1x,S1y,S1z])
    S2_vec = np.array([S2x,S2y,S2z])

    return S1_vec,S2_vec

def cold():

    #"Cold" accretion model (mostly aligned spins)
    #Based on fits in Lousto et al. 2012
    
    alpha = 5.935
    beta = 1.856
    a1 = stats.beta(alpha,beta).rvs()
    a2 = stats.beta(alpha,beta).rvs()

    theta1 = cold_angle()
    theta2 = cold_angle()

    phi1 = 2 * np.pi * np.random.uniform(0,1)
    phi2 = 2 * np.pi * np.random.uniform(0,1)

    S1x = a1 * np.sin(theta1) * np.cos(phi1)
    S1y =  a1 * np.sin(theta1) * np.sin(phi1)
    S1z = a1 * np.cos(theta1)

    S2x = a2 * np.sin(theta2) * np.cos(phi2)
    S2y = a2 * np.sin(theta2) * np.sin(phi2)
    S2z = a2 * np.cos(theta2)

    S1_vec = np.array([S1x,S1y,S1z])
    S2_vec = np.array([S2x,S2y,S2z])

    return S1_vec,S2_vec

def deg5_high():

    #the spins are mostly aligned with the misalignment angle about 0-5 degrees from the orbital angular momentum

    a1 = 0.9
    a2 = 0.9


    loc = np.cos(5*np.pi/180)
    scale = np.cos(0) - np.cos(5*np.pi/180)
    # loc + scale = cos 5 degree 

    theta1 = np.arccos(stats.uniform(loc,scale).rvs())
    theta2 = np.arccos(stats.uniform(loc,scale).rvs())

    phi1 = 2 * np.pi * np.random.uniform(0,1)
    phi2 = 2 * np.pi * np.random.uniform(0,1)

    S1x = a1 * np.sin(theta1) * np.cos(phi1)
    S1y =  a1 * np.sin(theta1) * np.sin(phi1)
    S1z = a1 * np.cos(theta1)

    S2x = a2 * np.sin(theta2) * np.cos(phi2)
    S2y = a2 * np.sin(theta2) * np.sin(phi2)
    S2z = a2 * np.cos(theta2)

    S1_vec = np.array([S1x,S1y,S1z])
    S2_vec = np.array([S2x,S2y,S2z])



    return S1_vec,S2_vec
    

def gw_kick(q,S1_vec,S2_vec):

    #symmetric mass ratio
    eta = q/(1+q)**2

    a1mag = np.linalg.norm(S1_vec)
    a2mag = np.linalg.norm(S2_vec)


    ### Constants for kick equation:
    ### From Lousto & Zlochower 2009, Lousto et al. 2010:
    A = 1.2e+4 # km/s  
    B = -0.93  
    H = 6.9e+3 # km/s 
    xi = 145.0 * np.pi / 180.0
    
    ### Lousto et al. 2012 (incl. hangup kicks)
    V11 = 3677.76
    VA = 2481.21
    VB = 1792.45
    VC = 1506.52

    

    Vm = A * eta**2 * np.sqrt(1 - 4 * eta) * (1 + B * eta)
    a1_pll = S1_vec[2]
    a2_pll = S2_vec[2]
    a1_prp = S1_vec[:-1]
    a2_prp = S2_vec[:-1]
    Vs_prp = H * eta*eta / (1 + q) * ( a2_pll - q*a1_pll )

    if np.linalg.norm(a1_prp) > 0 or np.linalg.norm(a2_prp) > 0:
        ## spin is not aligned with z-axis; need to calculate Vs_pll:

        # S-tilde from Lousto et al. 2012        
        St_pll = 2 * (a2_pll + q*q*a1_pll) / ((1+q)*(1+q))

        # This quantity is Delta_prp from Lousto et al 2012 divided by m2^2
        Delta_prp = (1+q) * (a2_prp - q*a1_prp)

        cos_Delta_prp = Delta_prp[0] / np.linalg.norm(Delta_prp)
            
        # these quantities must always be randomized:
        phi1 =  2 * np.pi * np.random.random()
        sign = -1 if np.random.rand()<0.1 else 1

        Vs_pll = ( 16*eta*eta / (1+q) *                
                       (V11 + VA*St_pll + VB*St_pll*St_pll + VC*St_pll*St_pll*St_pll) *
                       np.sqrt( (a2_prp[0] - q*a1_prp[0])*(a2_prp[0] - q*a1_prp[0]) +     
                                (a2_prp[1] - q*a1_prp[1])*(a2_prp[1] - q*a1_prp[1]) ) *   
                       np.cos( np.pi + sign*np.arccos(cos_Delta_prp) - phi1 ) )
    else:                                                              
        ### spin is aligned along z-axis; no kick along z-axis
        Vs_pll = 0.0

    Vk = np.empty(3)
    Vk[0] = Vm + Vs_prp * np.cos(xi)
    Vk[1] = Vs_prp * np.sin(xi)
    Vk[2] = Vs_pll

    return Vk