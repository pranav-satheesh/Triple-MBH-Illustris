import numpy as np
import matplotlib.pyplot as plt
import os
# from read_mbhb import mbhb_data, define_mbhb_inspiral_phases
from find_triple_mbhs import tripleMatches
from tqdm import tqdm
from matplotlib import rc
from scipy.interpolate import interp1d
# import scipy.integrate as integrate
#####for corner plots#####
import corner
#####for histogram qhist_2d_new#####
import scipy as sp
import scipy.stats
import matplotlib as mpl
import pandas as pd
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import astropy.units as u
from astropy import constants as const
import h5py

class stalled_model(tripleMatches):

    def __init__(self, path, fmbhb, mergers, **kwargs):
        
        #we use next prescription for now
        #since previous can't register all the mergers
        
        super().__init__(path, fmbhb, mergers, **kwargs)

        self.stalled_triple_mask = self._next.ix_1st_mbhb>0
        self.nmbhb = self._next.Nmbhb

        

        #initial formation redshift of inner and outer binary in Illustris  
          
        self.first_binary_idx = self._next.ix_1st_mbhb[self._next.ix_1st_mbhb>0]
        self.second_binary_idx = self._next.ix_2nd_mbhb[self._next.ix_1st_mbhb>0]

        self.inner_binary_initial_mass = self.masses[self.first_binary_idx]
        self.outer_binary_initial_mass = self.masses[self.second_binary_idx]

        self.inner_binary_times = self.times[self.first_binary_idx, :]
        self.outer_binary_times = self.times[self.second_binary_idx, :]
        

        #total mass of inner binary
        self.inner_binary_initial_mtotal = self.inner_binary_initial_mass.sum(axis=1)
        
        
    def find_first_binary_in_next_merger(self, mass_inn_bin=None):
        """
        Find the index of the inner binary in next merger using approximate tec-
        hnique of comparing masses after accretion with each index of next merger
        masses and choose the closest one to be our inner bianry 
        """
        if mass_inn_bin ==None:
            mass_inn_bin = self.inner_binary_initial_mtotal
            
        first_binary_mass = self.inner_binary_initial_mass

        q_outer = -1*np.ones(len(first_binary_mass))
        intruder_mass = -1*np.ones(len(first_binary_mass))
        inner_binary_mass_in_next_merger = -1*np.ones(len(first_binary_mass))
        idx_first_bin_in_next_merger = -1*np.ones(len(first_binary_mass))
        
        # mass_inn
        for i in range(len(mass_inn_bin)):

            dm = self.outer_binary_initial_mass[i]-mass_inn_bin[i]

            idx = np.where(dm==min(dm))[0]
            if min(dm)<0:
                # idx = np.where(abs(dm)==min(abs(dm)))[0] the abs value doesnt work better
                idx = np.where(dm!=min(dm))[0]
            # intruder mass is the mbh that is not similar to the accreted inner binary mass
            # so its the complementary set to idx meaning ~idx
            intruder_mass[i] = self.outer_binary_initial_mass[i][~idx]
            inner_binary_mass_in_next_merger[i] = self.outer_binary_initial_mass[i][idx]
            q_outer[i] = intruder_mass[i]/inner_binary_mass_in_next_merger[i]  #define q=m_intruder/m_inner
            idx_first_bin_in_next_merger[i] = idx 
            
        return   idx_first_bin_in_next_merger, intruder_mass,  inner_binary_mass_in_next_merger
    

    def stalled_trip_data(self,save_path):

            '''To extract information about the strong triples from Illustris'''

            masses_inner_binary = self.masses[self.stalled_triple_mask]
            masses_inner_binary = np.sort(masses_inner_binary)
            M1_trip = masses_inner_binary[:,1] #bigger one is primary
            M2_trip= masses_inner_binary[:,0]
            _,intruder_mass,inner_binary_mass_in_next_merger = self.find_first_binary_in_next_merger()
            #delta_m = inner_binary_mass_in_next_merger - self.inner_binary_initial_mtotal
            #print("faulty accretion %d"%(len(np.argwhere(delta_m<0))))
            #dm_mask = np.argwhere(delta_m>0)
            #sim_unit_h_scale=0.704

            #f-gas,Vesc and BHIDs
            # fgas,Vescape = self.host_galaxy_properties()
            # fgas = fgas[self.stalled_triple_mask]
            # Vescape = Vescape[self.stalled_triple_mask]   
            M3_trip = intruder_mass
            qbh_inner = M2_trip/M1_trip
            qbh_outer = intruder_mass/inner_binary_mass_in_next_merger   
            t_form1 = self.inner_binary_times[:,0]
            t_form2 = self.outer_binary_times[:,0]
        
            stalled_triple_properties={
            'M1': M1_trip,
            'M2': M2_trip,
            'M3': M3_trip,
            'qin': qbh_inner,
            'qout':qbh_outer,
            't_binary_form':t_form1,
            't_triple_form':t_form2,
            'Mbin':inner_binary_mass_in_next_merger
            }

            # host_galaxy_properties = {
            # "fgas": fgas[self.triple_mask][strong_in_triple_mask],
            # "Vescape": Vescape[self.triple_mask][strong_in_triple_mask],
            # "SubhaloMass": self.envs_SubhaloMassType[self.triple_mask][strong_in_triple_mask],
            # "SubhaloMass_in":self.envs_in_SubhaloMassType[self.triple_mask][strong_in_triple_mask],
            # "SubhaloMassInHalfRadType":self.envs_SubhaloMassInHalfRadType[self.triple_mask][strong_in_triple_mask],
            # "SubhaloMassInHalfRadType_in":self.envs_in_SubhaloMassInHalfRadType[self.triple_mask][strong_in_triple_mask],
            # "SubhaloSFR":self.envs_SubhaloSFR[self.triple_mask][strong_in_triple_mask],
            # "SubhaloSFR_in":self.envs_SubhaloSFR[self.triple_mask][strong_in_triple_mask],
            # "SubhaloVelDisp_in":self.envs_in_SubhaloVelDisp[self.triple_mask][strong_in_triple_mask],
            # "subhalo_id":self.subhalo_id[self.triple_mask][strong_in_triple_mask]
            # }

            data_to_save={
                **stalled_triple_properties
            }
            save_file = save_path + 'stalled_triples_data_from_ill.h5'

            with h5py.File(save_file, 'w') as hf:
                for key, value in data_to_save.items():
                    hf.create_dataset(key, data=value)


            # df = pd.DataFrame([M1_trip,M2_trip,M3_trip,qbh_inner,qbh_outer,inner_binary_mass_in_next_merger,t_form1,t_form2])
            # df = df.transpose()
            # df.columns = ['M1','M2','M3','qin','qout','Mbin','t_binary_form','t_triple_form']
            # save_file = save_path + 'stalled_triples_data_from_ill.csv'

            # df.to_csv(save_file,index=False)
            print("File saved at",save_file)

    