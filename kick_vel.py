import numpy as np

def calc_kick(q,a1,a2):

    # q should be m1/m2 where m2>m1
    # a1 and a2 are vectors. The first two components are perpendicular to orbital angular momentum and the last one
    # parallel to angular momentum

    if q > 1.0:
        print("Warning: redefining entered value q={} as 1/q".format(q))
    
    a1mag = np.linalg.norm(a1)
    a2mag = np.linalg.norm(a2)

    A = 1.2e+4 # km/s  
    B = -0.93  
    H = 6.9e+3 # km/s 
    xi = 145.0 * np.pi / 180.0
    
    ### Lousto et al. 2012 (incl. hangup kicks)
    V11 = 3677.76
    VA = 2481.21
    VB = 1792.45
    VC = 1506.52
    
    ### symmetric mass ratio:
    eta = q / ((1+q) * (1+q))

    

