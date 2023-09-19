import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import spin_models as spin

df_binary = pd.read_csv("Data/binary-merger-data.csv",index_col= False)
df_binary.head()
df_iso_bbh = df_binary[(df_binary["Type"] == "iso") & (df_binary["Merger"] == "Yes")]

def find_q(M1,M2):
    if (M2>M1):
        q = M1/M2
    else:
        q = M2/M1
    
    return q

N_bbh = len(df_iso_bbh["M1"])

vGW_random = []
vGW_aligned = []

for i in range(N_bbh):
    q_i = find_q(df_iso_bbh["M1"].iloc[i],df_iso_bbh["M2"].iloc[i])

    S1,S2 = spin.random_dry()
    vGW_random.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))

    S1,S2 = spin.deg5_high()
    vGW_aligned.append(np.linalg.norm(spin.gw_kick(q_i,S1,S2)))

df_iso_bbh.insert(5,"GW-kick-random",vGW_random,True)
df_iso_bbh.insert(6,"GW-kick-aligned",vGW_aligned,True)

df_iso_bbh.to_csv("Data/binary-GW-kick-data.csv",index=False)


