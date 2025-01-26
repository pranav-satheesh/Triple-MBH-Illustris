
import pickle
from tqdm import tqdm
import os
import sys

Nruns = int(sys.argv[1])
triple_mbhb_find = sys.argv[2]
file_path = sys.argv[3]
obj_file_path = sys.argv[4]

sys.path.insert(1,triple_mbhb_find)

import BH_kicks as kick
import Triple_dynamics as Tr

# triple_mbhb_find = '/Users/pranavsatheesh/Triples/Github/Triple-Outcomes/triple_mbhb'

#import stalled_triple_model as stall


# file_path = '/Users/pranavsatheesh/Triples/Github/Triple-Outcomes/Data/'

Tr_objects = [Tr.Tripledynamics(file_path) for _ in tqdm(range(Nruns), desc="Triple MBH instances being created...")]
iso_bin = Tr.iso_binary(file_path)
weak_tr = Tr.weak_triples(file_path)
stalled_objs = [Tr.stalledtriples(file_path) for _ in tqdm(range(Nruns), desc="stalled Triple MBH instances being created...")]

iso_vrand,iso_vhybrid,iso_valigned = kick.gw_kick_assign(iso_bin,tr_flag="No")
iso_bin.v_kick_random = iso_vrand
iso_bin.v_kick_hybrid = iso_vhybrid
iso_bin.v_kick_aligned = iso_valigned


weak_tr_vrand,weak_tr_vhybrid,weak_tr_valigned = kick.gw_kick_assign(weak_tr,tr_flag="No")
weak_tr.v_kick_random = weak_tr_vrand
weak_tr.v_kick_hybrid = weak_tr_vhybrid
weak_tr.v_kick_aligned = weak_tr_valigned

for i in tqdm(range(Nruns),desc="assigning kicks to strong triple instances"):
    strong_tr_vrand,strong_tr_vhybrid,strong_tr_valigned = kick.gw_kick_assign(Tr_objects[i],tr_flag="Yes")
    Tr_objects[i].v_kick_random = strong_tr_vrand
    Tr_objects[i].v_kick_hybrid = strong_tr_vhybrid
    Tr_objects[i].v_kick_aligned = strong_tr_valigned


# iso_filename = os.path.abspath('../obj_data/iso_bin_wkick.pkl')

iso_filename = obj_file_path+'iso_bin_wkick.pkl'
weak_tr_filename = obj_file_path+'weak_tr_wkick.pkl'
strong_tr_filename = obj_file_path+f'tr{Nruns}_wkick.pkl'
stalled_tr_filename = obj_file_path+f'stalled{Nruns}.pkl'

# weak_tr_filename = os.path.abspath('../obj_data/weak_tr_wkick.pkl')
# strong_tr_filename =os.path.abspath(f'../obj_data/tr{Nruns}_wkick.pkl')
# stalled_tr_filename=os.path.abspath(f'../obj_data/stalled{Nruns}.pkl')

with open(strong_tr_filename, 'wb') as f:
    pickle.dump(Tr_objects, f, pickle.HIGHEST_PROTOCOL)

with open(iso_filename, 'wb') as f:
    pickle.dump(iso_bin, f, pickle.HIGHEST_PROTOCOL)

with open(weak_tr_filename, 'wb') as f:
    pickle.dump(weak_tr, f, pickle.HIGHEST_PROTOCOL)

with open(stalled_tr_filename, 'wb') as f:
    pickle.dump(stalled_objs, f, pickle.HIGHEST_PROTOCOL)


