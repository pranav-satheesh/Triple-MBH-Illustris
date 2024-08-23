import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spin_models as spin

from astropy import constants as const
from astropy import units as u


#convertions and constants
G_value = const.G.value
Msun_to_kg = (1*u.M_sun).to(u.kg).value #Msun to kg
pc_to_m = (1*u.pc).to(u.m).value #parsecs to m

def trip_log_v():
    #from fitting the Hoffman&Loeb distribution
    mean = 2.87678065
    stdev = 0.31454837
    return st.norm.rvs(mean,stdev)

def trip_qout():
    a = 1.36869939
    loc = 1.67145567
    scale = 2.91035809
    return st.gamma.rvs(a,loc,scale)


def v_scaled(m3,qout):

    q_HL = 1/trip_qout()
    M_tot = 6 * 10**8
    m3_HL = (q_HL * M_tot)/(1+q_HL) 

    v_HL = trip_log_v()

    return (np.sqrt((m3_HL*(1+q_HL))/(m3*(1+qout))) * 10**(v_HL))

def find_q(M1,M2):
    if (M2>M1):
        q = M1/M2
    else:
        q = M2/M1
    
    return q

# def gw_kick_calc(qin):

#     #random-dry
#     S1,S2 = spin.random_dry()
#     kick_rand = np.linalg.norm(spin.gw_kick(qin,S1,S2))
#     #5deg
#     S1,S2 = spin.deg5_high()
#     kick_5d = np.linalg.norm(spin.gw_kick(qin,S1,S2))
#     #cold
#     S1,S2 = spin.cold()
#     kick_cold = np.linalg.norm(spin.gw_kick(qin,S1,S2))

#     return kick_rand,kick_5d,kick_cold


# gw-kick-with f-gas for the hybrid model

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
    

def a_hard(M1,qin):
    ahard = 0.80 * (4*qin)/(1+qin)**2 * ((M1.value*(1+qin))/10**8)**(1/2)
    return ahard

def find_exchange_DE(m1,m_ex,m_int,a_t):

    a_t = a_t * u.pc
    m1 = m1 * u.M_sun
    m_ex = m_ex * u.M_sun
    m_int = m_int*u.M_sun

    BE_i = const.G*(m1.to(u.kg)*m_ex.to(u.kg))/(2*a_t.to(u.m))
    BE_f = const.G*(m1.to(u.kg)*m_int.to(u.kg))/(2*a_t.to(u.m))

    DE = BE_f - BE_i
    
    return DE.value,BE_i.value


def v_trip_exchange(m1,m_ex,m_int,qin,a_ovtks):

    ahard = a_hard(m1* u.M_sun,qin)

    if(ahard<a_ovtks):
        a_t = ahard
    else:
        a_t = a_ovtks

    dE,_ = find_exchange_DE(m1,m_ex,m_int,a_t)

    Vsling = np.sqrt(2*(m1+m_int)*dE/(m_ex*(m1+m_ex+m_int)*(1*u.M_sun.to(u.kg))))*10**(-3)
    return Vsling


def v_trip_cal(m1,qin,qout,a_ovtks):
    
    f = 0.4
    m1 = m1 * u.M_sun

    ahard = a_hard(m1,qin)

    if(ahard<a_ovtks):
        a_t = ahard
    else:
        a_t = a_ovtks

    a_t = a_t * u.pc
    Vsling = np.sqrt(f*const.G * (m1.to(u.kg) * qin)/((a_t).to(u.m) * (1+qout) * (1+qin) )).to(u.km * u.s**-1) #km/s
    return Vsling.value

def v_and_a_after_slingshot(m1,m2,mint,aini,key):

    f1 = 0.4
    f2 = 0.9
    f3 = 0.53
    
    qout = mint/(m1+m2)
    m1 = m1 * u.M_sun
    m2 = m2 * u.M_sun
    mint = mint* u.M_sun
    aini = aini * u.pc

    Eb = const.G*(m1.to(u.kg)*m2.to(u.kg))/(2*aini.to(u.m))

    if(key==1):
        #mint < m2
        #scattering event

        dE = f1*qout*Eb
        mbin = m1 + m2
        mej = mint

        Kej = dE/(1+ (mej/mbin))
        Vsling = (np.sqrt(2*Kej/mej)).to(u.km * u.s**-1) #km/s
        anew = (aini/(1 + f1*qout)).to(u.pc)
    
    elif(key==2):
        #mint>m2 but mint<2(m1+m2)
        #exchange event

        dE = f1*qout*Eb
        mbin = m1 + mint
        mej = m2

        Kej = dE/(1+ (mej/mbin))
        Vsling = (np.sqrt(2*Kej/mej)).to(u.km * u.s**-1) #km/s
        anew = ((aini/(1 + f1*qout))*(mint/m2)).to(u.pc)
    
    elif(key==3):
        #mint > 2(m1+m2)
        #exchange event

        dE = f2*Eb
        mbin = m1+mint
        mej = m2

        Kej = dE/(1+ (mej/mbin))
        Vsling = (np.sqrt(2*Kej/mej)).to(u.km * u.s**-1) #km/s
        anew = (aini*f3*(mint/m2)).to(u.pc)

    elif(key==12):
        #mint>m2 but mint<2(m1+m2)
        #exchange event

        dE = f1*qout*Eb
        mbin = m2 + mint
        mej = m1

        Kej = dE/(1+ (mej/mbin))
        Vsling = (np.sqrt(2*Kej/mej)).to(u.km * u.s**-1) #km/s
        anew = ((aini/(1 + f1*qout))*(mint/m1)).to(u.pc)

    elif(key==13):
        #mint>m2 but mint<2(m1+m2)
        #exchange event

        dE = f2*Eb
        mbin = m2 + mint
        mej = m1

        Kej = dE/(1+ (mej/mbin))
        Vsling = (np.sqrt(2*Kej/mej)).to(u.km * u.s**-1) #km/s
        anew = (aini*f3*(mint/m1)).to(u.pc)

    return Vsling.value,anew.value


# def trip_kick_assign(filename):

#     df_trip = pd.read_csv(filename,index_col=False)

#     #slingshot data
#     slingshot_kicks = []
#     gw_kick_random = []
#     gw_kick_hybrid =[]
#     gw_kick_5deg = []

#     for i in range(len(df_trip)):

#         m1 = df_trip["M1"].iloc[i]
#         m2 = df_trip["M2"].iloc[i]
#         m3 = df_trip["M3"].iloc[i]
#         qin = df_trip["qin"].iloc[i]
#         qout = df_trip["qout"].iloc[i]
#         atrip = df_trip["a_triple"].iloc[i]
#         fgas = df_trip["f-gas"]

#         if(df_trip["merger_flag"].iloc[i] == "Tr-ej"):
#             #lightest is ejected and other two merges later
#             m_sort = np.sort([m1,m2,m3])
    
#             if(m_sort[0]==m3):
#                 slingshot_kicks.append(v_trip_cal(m1,qin,qout,atrip))
#                 rand,hyb,deg5 = gw_kick_calc(qin,fgas)
#                 gw_kick_random.append(rand)
#                 gw_kick_5deg.append(deg5)
#                 gw_kick_hybrid.append(hyb)


#             elif(m_sort[0]==m2):
#                 #m3 has replaced m2 from its orbit and m3-m1 is the new binary
                
#                 m1_new = max(m1,m3)
#                 m2_new = min(m1,m3)
#                 qin_new = m2_new/m1_new
   
#                 slingshot_kicks.append(v_trip_exchange(m1,m2,m3,qin,atrip))
#                 rand,hyb,deg5 = gw_kick_calc(qin,fgas)
#                 gw_kick_random.append(rand)
#                 gw_kick_5deg.append(deg5)
#                 gw_kick_hybrid.append(hyb)

            
#         elif(df_trip["merger_flag"].iloc[i] == "Tr-12"):
            
#             #m1-m2 merges
            
#             #slingshot_kicks.append(v_trip_cal(m1,qin,qout,atrip))
#             slingshot_kicks.append(0)
#             rand,deg5,cold = gw_kick_calc(qin)
#             gw_kick_random.append(rand)
#             gw_kick_5deg.append(deg5)
#             gw_kick_cold.append(cold)

#         elif(df_trip["merger_flag"].iloc[i] == "Tr-23"):
            
#             #m2-m3 merges

#             #slingshot_kicks.append(v_trip_cal(m1_new,qin_new,qout_new,atrip_new))
#             slingshot_kicks.append(0)
#             atrip_new = atrip * (m3/m1)
#             m1_new = max(m2,m3)
#             m2_new = min(m2,m3)
#             qin_new = m2_new/m1_new
            

#             rand,deg5,cold = gw_kick_calc(qin_new)
#             gw_kick_random.append(rand)
#             gw_kick_5deg.append(deg5)
#             gw_kick_cold.append(cold)

#         elif(df_trip["merger_flag"].iloc[i] == "Tr-13"):

#             #m1-m3 merges

#             #slingshot_kicks.append(v_trip_cal(m1_new,qin_new,qout_new,atrip_new))
#             slingshot_kicks.append(0)
#             atrip_new = atrip * (m3/m2)
#             m1_new = max(m1,m3)
#             m2_new = min(m1,m3)
#             qin_new = m2_new/m1_new
            
#             rand,deg5,cold = gw_kick_calc(qin_new)
#             gw_kick_random.append(rand)
#             gw_kick_5deg.append(deg5)
#             gw_kick_cold.append(cold)
            
#         elif(df_trip["merger_flag"].iloc[i]=="No"):
#             #lightest ejects
#             #others do not merge
#             m_sort = np.sort([m1,m2,m3])
#             # m1_new = m_sort[2]
#             # m2_new = m_sort[1]
#             # m3_new = m_sort[0]

#             # atrip_new = atrip * (m1_new/m1) * (m2_new/m2)
#             # qout_new = m3_new/(m1_new+m2_new)
#             # qin_new = m2_new/m1_new

#             if(m_sort[0]==m3):
#                 slingshot_kicks.append(v_trip_cal(m1,qin,qout,atrip))

#             elif(m_sort[0]==m2):
#                 slingshot_kicks.append(v_trip_exchange(m1,m2,m3,qin,atrip))

#             #slingshot_kicks.append(v_trip_cal(m1_new,qin_new,qout_new,atrip_new))
#             gw_kick_random.append(0)
#             gw_kick_5deg.append(0)
#             gw_kick_cold.append(0)
        
#     df_trip.insert(5,"Slingshot_kick",slingshot_kicks,True)
#     df_trip.insert(6,"gw_kick_random",gw_kick_random,True)
#     df_trip.insert(7,"gw_kick_cold",gw_kick_cold,True)
#     df_trip.insert(8,"gw_kick_5deg",gw_kick_5deg,True)
#     #df_t_slingshot.to_csv("Data/triples-slingshot-data.csv",index=False)
    
#     return df_trip
