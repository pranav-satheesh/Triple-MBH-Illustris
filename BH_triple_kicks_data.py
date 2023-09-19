import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spin_models as spin


#from fitting the Hoffman&Loeb distribution

mean = 2.87678065
stdev = 0.31454837

def trip_log_v():
    return st.norm.rvs(mean,stdev)


def v_scaled(m3,qout):

    q_HL = 1/10
    m3_HL = 10**7
    v_HL = trip_log_v()

    return (np.sqrt((m3_HL*(1+q_HL))/(m3*(1+qout))) * 10**(v_HL))


df_trip = pd.read_csv("Data/triples-data-with-flags.csv",index_col= False)
df_t_slingshot = df_trip[(df_trip["Flag"] == "Tr-ej") | (df_trip["Flag"] == "No")]
df_t_GW_kick = df_trip[(df_trip["Flag"] == "Tr-ej") | (df_trip["Flag"] == "Tr")]

#slingshot data
slingshot_kicks = []
for i in range(np.shape(df_t_slingshot)[0]):
    m1 = df_t_slingshot["Mass 1"].iloc[i]
    m2 = df_t_slingshot ["Mass 2"].iloc[i]
    m3 = df_t_slingshot ["Mass 3"].iloc[i]
    m_sort = np.sort([m1,m2,m3])

    m3 = m_sort[0]
    qout = m_sort[0]/(m_sort[1]+m_sort[2])
    slingshot_kicks.append(v_scaled(m3,qout))

df_t_slingshot.insert(5,"Slingshot_kick",slingshot_kicks,True)
df_t_slingshot.to_csv("Data/triples-slingshot-data.csv",index=False)


#binary GW kicks
vGW_random = []
vGW_aligned = []

def find_q(M1,M2):
    if (M2>M1):
        q = M1/M2
    else:
        q = M2/M1
    
    return q


for i in range(np.shape(df_t_GW_kick)[0]):
    m1 = df_t_GW_kick["Mass 1"].iloc[i]
    m2 = df_t_GW_kick["Mass 2"].iloc[i]
    m3 = df_t_GW_kick["Mass 3"].iloc[i]
    m_sort = np.sort([m1,m2,m3])

    m1 = m_sort[1]
    m2 = m_sort[2]
    q = find_q(m1,m2)

    #random-dry
    S1,S2 = spin.random_dry()
    vGW_random.append(np.linalg.norm(spin.gw_kick(q,S1,S2)))

    #aligned-5deg
    S1,S2 = spin.deg5_high()
    vGW_aligned.append(np.linalg.norm(spin.gw_kick(q,S1,S2)))

df_t_GW_kick.insert(6,"GW-kick-random",vGW_random,True)
df_t_GW_kick.insert(7,"GW-kick-aligned",vGW_aligned,True)

df_t_GW_kick.to_csv("Data/triples-GW-kick-data.csv",index=False)

