from astropy import constants as const
from astropy import units as u
import numpy as np
import spin_models as spin
from tqdm import tqdm
#convertions and constants
G_value = const.G.value
Msun_to_kg = (1*u.M_sun).to(u.kg).value #Msun to kg
pc_to_m = (1*u.pc).to(u.m).value #parsecs to m


def a_hard(M1, qin):
    ahard = 0.80 * (4*qin)/(1+qin)**2 * ((M1*(1+qin))/10**8)**(1/2)
    return ahard

def v_and_a_after_slingshot(m1,m2,mint,aini,m1_scatt_key):
    '''Finds the velocity and seperation of the inner binary after a triple interaction.
    Keys:

    '''
    f1 = 0.4
    f2 = 0.9
    f3 = 0.53
    
    qout = mint/(m1+m2)
    m1 = m1 * u.M_sun
    m2 = m2 * u.M_sun
    mint = mint* u.M_sun
    aini = aini * u.pc

    Eb = const.G*(m1.to(u.kg)*m2.to(u.kg))/(2*aini.to(u.m))
    
    
    if(mint<m2):
        dE = f1*qout*Eb
        mbin = m1 + m2
        mej=mint
        q_new = m2/m1
        anew = ((aini/(1 + f1*qout))).to(u.pc)

    elif((mint<2*(m1+m2)) and (mint>m2)):
        
        if(m1_scatt_key==True):
            dE = f1*qout*Eb
            mbin = m2 + mint
            mej = m1
            q_new = mint/m2
            anew = ((aini/(1 + f1*qout))*(mint/mej)).to(u.pc)
            
        else:
            dE = f1*qout*Eb
            mbin = m1 + mint
            mej = m2
            q_new = mint/m1
            anew = ((aini/(1 + f1*qout))*(mint/mej)).to(u.pc)

    else:
        if(m1_scatt_key==True):
            dE = f2*Eb
            mbin = m2 + mint
            mej = m1
            q_new = mint/m2
            anew = (f3*aini*(mint/mej)).to(u.pc)

        else:
            dE = f2*Eb
            mbin = m1+mint
            mej = m2
            q_new = mint/m1
            anew = (f3*aini*(mint/mej)).to(u.pc)

    Kej = dE/(1+ (mej/mbin))
    Vsling = (np.sqrt(2*Kej/mej)).to(u.km * u.s**-1) #km/s
    q_new = 1/q_new if q_new > 1 else q_new

    return Vsling.value,anew.value,q_new.value,mbin.value

# def gw_kick_calc(qin,fgas):

#     #random
#     S1,S2 = spin.random_dry()
    
#     kick_rand = np.linalg.norm(spin.gw_kick(qin,S1,S2))

#     #hybrid
#     if(fgas<0.1):
#         #gas-poor: spins are random and misaligned
#         S1,S2 = spin.random_dry()
#         kick_hybrid = np.linalg.norm(spin.gw_kick(qin,S1,S2))
#     elif(fgas>=0.1):
#         #gas-rich: spins are nearly aligned
#         S1,S2 = spin.cold()
#         kick_hybrid = np.linalg.norm(spin.gw_kick(qin,S1,S2))
    
#     #aligned spins
#     S1,S2 = spin.deg5_high()
#     kick_5d = np.linalg.norm(spin.gw_kick(qin,S1,S2))

#     return kick_rand,kick_hybrid,kick_5d


def gw_kick_assign(obj,n_realizations=10,tr_flag="No"):
    
    v_random = []
    v_hybrid = []
    v_aligned = []

    if(tr_flag=="Yes"):
        q_merger = obj.qin_merger[obj.merger_mask]
    else:
        q_merger = obj.qin[obj.merger_mask]

    fgas_merger = obj.fgas[obj.merger_mask]

    for i in tqdm(range(n_realizations),desc="calculating kick velocities for N realizations"):
        
        v_rand_n = []
        v_hybrid_n = []
        v_aligned_n = []

        for i in range(np.sum(obj.merger_mask)):
            
            S1_rand,S2_rand = spin.random_dry()
            v_rand_n.append(np.linalg.norm(spin.gw_kick(q_merger[i],S1_rand,S2_rand)))    

            # Hybrid alignment based on gas fraction
            if fgas_merger[i] < 0.1:
                # Gas-poor: spins are random and misaligned
                S1_hybrid, S2_hybrid = spin.random_dry()
            else:
                # Gas-rich: spins are nearly aligned
                S1_hybrid, S2_hybrid = spin.cold()
            
            v_hybrid_n.append(np.linalg.norm(spin.gw_kick(q_merger[i], S1_hybrid, S2_hybrid)))
            
                # Nearly aligned spins (5-degree alignment)
            S1_aligned, S2_aligned = spin.deg5_high()
            v_aligned_n.append(np.linalg.norm(spin.gw_kick(q_merger[i], S1_aligned, S2_aligned)))

        v_random.append(v_rand_n)
        v_hybrid.append(v_hybrid_n)
        v_aligned.append(v_aligned_n)

    return v_random,v_hybrid,v_aligned


#will comment this function out later
def gw_kick_calc(qin, fgas):
    """
    Calculates and averages the gravitational wave kicks for three spin alignment cases: 
    random, hybrid (gas-poor/gas-rich based on fgas), and nearly aligned (5 degrees).
    
    Parameters:
    - qin: Mass ratio
    - fgas: Gas fraction
    - n_realizations: Number of realizations to average (default is 100)
    
    Returns:
    - Average kick velocities for each case: random, hybrid, and 5-degree aligned.
    """
    # Store kick values for averaging
    random_kicks = []
    hybrid_kicks = []
    aligned_5deg_kicks = []
    

    # Random spin alignment
    S1_rand, S2_rand = spin.random_dry()
    kick_rand = np.linalg.norm(spin.gw_kick(qin, S1_rand, S2_rand))
    random_kicks.append(kick_rand)
        
        # Hybrid alignment based on gas fraction
    if fgas < 0.1:
            # Gas-poor: spins are random and misaligned
        S1_hybrid, S2_hybrid = spin.random_dry()
    else:
            # Gas-rich: spins are nearly aligned
        S1_hybrid, S2_hybrid = spin.cold()
        
    kick_hybrid = np.linalg.norm(spin.gw_kick(qin, S1_hybrid, S2_hybrid))
    hybrid_kicks.append(kick_hybrid)
        
        # Nearly aligned spins (5-degree alignment)
    S1_aligned, S2_aligned = spin.deg5_high()
    kick_5d = np.linalg.norm(spin.gw_kick(qin, S1_aligned, S2_aligned))
    aligned_5deg_kicks.append(kick_5d)
    
    # Calculate and return the averages
    avg_kick_rand = random_kicks
    avg_kick_hybrid = hybrid_kicks
    avg_kick_5deg = aligned_5deg_kicks
    
    return avg_kick_rand, avg_kick_hybrid, avg_kick_5deg


