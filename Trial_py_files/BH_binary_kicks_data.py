import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spin_models as spin

def find_q(M1,M2):
    if (M2>M1):
        q = M1/M2
    else:
        q = M2/M1
    
    return q

def binary_kick_assign(filename):
    df_binary = pd.read_csv(filename,index_col= False)
    N_bbh = len(df_binary["M1"])
    fgas = df_binary["f-gas"]

    #Vesc = df_binary["Vesc"].to_numpy()  

    vGW_random = []
    vGW_5deg = []
    vGW_hybrid = []

    for i in range(N_bbh):
        q_i = find_q(df_binary["M1"].iloc[i],df_binary["M2"].iloc[i])

        #random
        S1,S2 = spin.random_dry()
        vGW_random.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))

        #5deg
        S1,S2 = spin.deg5_high()
        vGW_5deg.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))

        #hybrid
        if(fgas.iloc[i]<0.1):
            #gas-poor. spins are drawn from random
            S1,S2 = spin.random_dry()
            vGW_hybrid.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))
        elif(fgas.iloc[i]>=0.1):
            S1,S2 = spin.cold()
            vGW_hybrid.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))
            

    df_binary.insert(5,"gw_kick_random",vGW_random,True)
    df_binary.insert(6,"gw_kick_5deg",vGW_5deg,True)
    df_binary.insert(7,"gw_kick_hybrid",vGW_hybrid,True)
    
    #df_binary.to_csv("Data/binary-GW-kick-data.csv",index=False)

    return df_binary


def weak_triples_assign(df_weak_trip):

    N_bbh = len(df_weak_trip["M1"])
    fgas = df_weak_trip["f-gas"]

    vGW_random = []
    vGW_5deg = []
    vGW_hybrid = []

    for i in range(N_bbh):
        q_i = find_q(df_weak_trip["M1"].iloc[i],df_weak_trip["M2"].iloc[i])

        #random-dry
        S1,S2 = spin.random_dry()
        vGW_random.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))

        #5deg
        S1,S2 = spin.deg5_high()
        vGW_5deg.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))

        #hybrid
        if(fgas.iloc[i]<0.1):
            #gas-poor. spins are drawn from random
            S1,S2 = spin.random_dry()
            vGW_hybrid.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))
        elif(fgas.iloc[i]>=0.1):
            S1,S2 = spin.cold()
            vGW_hybrid.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))
    
    
    df_weak_trip.insert(5,"gw_kick_random",vGW_random,True)
    df_weak_trip.insert(6,"gw_kick_5deg",vGW_5deg,True)
    df_weak_trip.insert(7,"gw_kick_hybrid",vGW_hybrid,True)
    #df_weak_trip.insert(8,"Vescape",Vesc,True)
    #df_binary.to_csv("Data/binary-GW-kick-data.csv",index=False)

    return df_weak_trip   