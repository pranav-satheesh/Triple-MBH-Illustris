from astropy import constants as const
from astropy import units as u
import numpy as np
import spin_models as spin

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

    elif((mint<2*(m1+m2)) and (mint>m2)):
        
        if(m1_scatt_key==True):
            dE = f1*qout*Eb
            mbin = m2 + mint
            mej = m1
            q_new = mint/m2
            
        else:
            dE = f1*qout*Eb
            mbin = m1 + mint
            mej = m2
            q_new = mint/m1

    else:
        if(m1_scatt_key==True):
            dE = f2*Eb
            mbin = m2 + mint
            mej = m1
            q_new = mint/m2

        else:
            dE = f2*Eb
            mbin = m1+mint
            mej = m2
            q_new = mint/m1


    anew = ((aini/(1 + f1*qout))*(mint/mej)).to(u.pc)
    Kej = dE/(1+ (mej/mbin))
    Vsling = (np.sqrt(2*Kej/mej)).to(u.km * u.s**-1) #km/s
    q_new = 1/q_new if q_new > 1 else q_new

    return Vsling.value,anew.value,q_new.value

def gw_kick_calc(qin,fgas):

    #random
    S1,S2 = spin.random_dry()
    
    kick_rand = np.linalg.norm(spin.gw_kick(qin,S1,S2))

    #hybrid
    if(fgas<0.1):
        #gas-poor: spins are random and misaligned
        S1,S2 = spin.random_dry()
        kick_hybrid = np.linalg.norm(spin.gw_kick(qin,S1,S2))
    elif(fgas>=0.1):
        #gas-rich: spins are nearly aligned
        S1,S2 = spin.cold()
        kick_hybrid = np.linalg.norm(spin.gw_kick(qin,S1,S2))
    
    #aligned spins
    S1,S2 = spin.deg5_high()
    kick_5d = np.linalg.norm(spin.gw_kick(qin,S1,S2))

    return kick_rand,kick_hybrid,kick_5d

