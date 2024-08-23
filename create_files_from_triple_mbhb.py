# Add path to the triple_mbhs file 
import os
import sys
triple_mbhb_find = '/Users/pranavsatheesh/Triples/Github/triple_mbhs'
sys.path.append(triple_mbhb_find)

#Add your savepath here 
savepath = '/Users/pranavsatheesh/Triples/Github/Illustris_Data/'
import find_triple_mbhs as ftm
import read_mbhb
import numpy as np
import pandas as pd
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')
import h5py

#this is the file that extracts the csv files out of the tripel_mbhb code
import ms_script_pranav as mstest

path = '/Users/pranavsatheesh/Triples/Github/Illustris_Data/mbhb_data/' #path to the merger file 
fmrg='ill-1_blackhole_mergers_fixed.npz'  #merger file

mst = mstest.ms_tests(path = path, fmbhb='new_fid_e06' , mergers = np.load(path+fmrg), calculate_tinsp=True, parse_env=True) 
mst.strong_trip_stats()

mst.strong_trip_data(savepath) #strong triples + weak triples
mst.strong_trip_data(savepath,strong_trip_flag=True) #strong triples
mst.isolated_binaries_data(savepath)



