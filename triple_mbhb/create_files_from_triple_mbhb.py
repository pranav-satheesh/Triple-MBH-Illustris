# Add path to the triple_mbhs file 
import os
import sys
#triple_mbhb_find = '/Users/pranavsatheesh/Triples/Github/triple_mbhs'
#sys.path.append(triple_mbhb_find)


import numpy as np
np.seterr(divide='ignore', invalid='ignore')


#this is the file that extracts the csv files out of the tripel_mbhb code
import triple_n_binary_file_maker as filemaker
import stalled_triple_model as stall_filemaker

merger_file_path = '/Users/pranavsatheesh/Triples/Github/Illustris_Data/mbhb_data/' #path to the merger file 
fmrg='ill-1_blackhole_mergers_fixed.npz'  #merger file

mst = filemaker.ms_tests(path = merger_file_path, fmbhb='new_fid_e06' , mergers = np.load(merger_file_path+fmrg), calculate_tinsp=True, parse_env=True) 
mst.strong_trip_stats()

#Add your savepath here 
#savepath = '/Users/pranavsatheesh/Triples/Github/Illustris_Data/'
savepath ="/Users/pranavsatheesh/Triples/Github/Triple-Outcomes/Data/"

mst.trip_data(savepath) #strong triples + weak triples
mst.trip_data(savepath,strong_trip_flag=True) #strong triples
mst.isolated_binaries_data(savepath)

st_model = stall_filemaker.stalled_model(path = merger_file_path, fmbhb='new_fid_e06' , mergers = np.load(merger_file_path+fmrg), calculate_tinsp=True, parse_env=True) 
st_model.stalled_trip_data(savepath)



