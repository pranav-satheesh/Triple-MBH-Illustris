import h5py    
import numpy as np    
import pandas as pd
from astropy.cosmology import WMAP9 as cosmo
import astropy.units as u
from astropy import constants as const

yrsec = (1*u.yr).to(u.s).value # 1 year in secs
solar_mass = const.M_sun.value #in Kgs

#hubble_time = (1/cosmo.H(0)).to(u.yr).value #Hubble time = 1/H0 
age_of_the_universe = cosmo.age(0).to(u.yr).value


#parent folder
pfolder = "/Users/pranavsatheesh/Triples/Github/"
import sys
sys.path.append(pfolder)


#Binary HDF5 file location
file_name = pfolder + "Illustris_Data/mbhb-evolution_no-ecc_lc-full-0.6.hdf5"

f1 = h5py.File(file_name,'r') 
Ms = f1['evolution']['masses']
t = f1['evolution']['times']
r = f1['evolution']['sep']

M1_list = []
M2_list = []
t_list = []

Nbinary = np.shape(f1['evolution']['masses'])[0]

for i in range(Nbinary):
    M1_list.append(Ms[i][0])
    M2_list.append(Ms[i][1])
    t_list.append(t[i][np.argwhere(r[i]==0)[0][0]]) #t merger is when the seperation goes to 0
    
M1_list = M1_list/solar_mass #mass in kgs
M2_list = M2_list/solar_mass #mass in kgs
t_list = t_list/yrsec #convert to years

merger = [] #to indicate if the binary actually merges before hubble time

for time in t_list:
    if time >= age_of_the_universe:
        #these black holes aren't merging
        merger.append("No")
    else:
        merger.append("Yes")




#to check if the binary has triple interactions:



#binary ids
mergers = np.load(pfolder+'Illustris_Data/ill-1_blackhole_mergers_fixed.npz')
indexes = f1['evolution']['val_inds'][:]
binary_ids = mergers['ids'][indexes]

Triple_df = pd.read_csv("Data/triple_data_ill.csv") #the triples data file from find_triples

#Black hole IDs of all the triples

bh1id1 = Triple_df["BH1_ID1"].to_numpy()
bh1id2 = Triple_df["BH1_ID2"].to_numpy()
bh2id1 = Triple_df["BH2_ID1"].to_numpy()
bh2id2 = Triple_df["BH2_ID2"].to_numpy()



def searchID(id):

    #to search if the binary ids belong to any triples, indicating a triple interaction

    tf1 = id in bh1id1
    tf2 = id in bh1id2
    tf3 = id in bh2id1
    tf4 = id in bh2id2

    tf = tf1 + tf2 + tf3 + tf4

    return tf 

type = [] #"iso" for binaries that undergo isolated evolution. "trip" for the ones that have some kind of triple interaction"



for i in range(Nbinary):

    search1 = searchID(binary_ids[i][0])
    search2 = searchID(binary_ids[i][1])

    search = search1 + search2
    if(search == 0):
        type.append("iso")
    else:
        type.append("trip")


df = pd.DataFrame([M1_list,M2_list,t_list,merger,type])
df = df.transpose()
df.columns = ['M1', 'M2', 't_merger','Merger','Type']

#Binary merger file
df.to_csv("Data/binary-merger-data.csv",index = False)
print("saved")



