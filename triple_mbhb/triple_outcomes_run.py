import Triple_dynamics as Tr
import pickle
from tqdm import tqdm
import os
import sys
triple_mbhb_find = '/Users/pranavsatheesh/Triples/Github/Triple-Outcomes/triple_mbhb'
sys.path.insert(1,triple_mbhb_find)
#import stalled_triple_model as stall

Nruns = 100
file_path = '/Users/pranavsatheesh/Triples/Github/Triple-Outcomes/Data/'

Tr_objects = [Tr.Tripledynamics(file_path) for _ in tqdm(range(Nruns), desc="Triple MBH instances being created...")]
iso_bin = Tr.iso_binary(file_path)
weak_tr = Tr.weak_triples(file_path)
stalled_objs = [Tr.stalledtriples(file_path) for _ in tqdm(range(Nruns), desc="stalled Triple MBH instances being created...")]

iso_filename = os.path.abspath('../obj_data/iso_bin.pkl')
weak_tr_filename = os.path.abspath('../obj_data/weak_tr.pkl')
strong_tr_filename =os.path.abspath(f'../obj_data/tr{Nruns}.pkl')
stalled_tr_filename=os.path.abspath(f'../obj_data/stalled{Nruns}.pkl')

with open(strong_tr_filename, 'wb') as f:
    pickle.dump(Tr_objects, f, pickle.HIGHEST_PROTOCOL)

with open(iso_filename, 'wb') as f:
    pickle.dump(iso_bin, f, pickle.HIGHEST_PROTOCOL)

with open(weak_tr_filename, 'wb') as f:
    pickle.dump(weak_tr, f, pickle.HIGHEST_PROTOCOL)

with open(stalled_tr_filename, 'wb') as f:
    pickle.dump(stalled_objs, f, pickle.HIGHEST_PROTOCOL)
