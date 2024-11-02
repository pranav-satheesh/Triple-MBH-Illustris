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
import json

class ms_tests(tripleMatches):
    
    YR_IN_S=365.2425*24*60*60 #year ins seconds

    def __init__(self, path, fmbhb, mergers, **kwargs):
        
        #we use next prescription for now
        #since previous can't register all the mergers
        
        super().__init__(path, fmbhb, mergers, **kwargs)
#         self.tinsp = self.time_scales()
        self.nmbhb = len(self.dadt)
        self.triple_mask = self._next.triple_mask
        self.binary_mask = ~self._next.triple_mask
        self.strong_triple_mask = self._next.strong_triple_mask
        self.weak_triple_mask = self._next.weak_triple_mask

        self.z_sim       = (1/self.scales[:,0])-1
        self.z_evol = (1/self.scales)-1
        self.masks = ['triple','binary','triple 1st','triple 2nd']
        
        
        #separating the inner binary and outer binary in triples
        #put the intial in naming as accretion is not considere and the data
        #are directly from Illustris. i.e. there is no sub-resolution evolution
        #yet
        self.inner_binary_initial_mass = self.masses[self._next.idx_inner_binary]
        self.outer_binary_initial_mass = self.masses[self._next.idx_outer_binary]
        
        #initial formation redshift of inner and outer binary in Illustris
        self.inner_binary_initial_z = self.z_sim[self._next.idx_inner_binary]
        self.outer_binary_initial_z = self.z_sim[self._next.idx_outer_binary]   
        
        #the accretion rates and times are from sub resolution evolution they 
        #contain aintial values along with interpolated values at later times
        self.inner_binary_mdot_eff = self.mdot_eff[self._next.idx_inner_binary,:]
        self.outer_binary_mdot_eff = self.mdot_eff[self._next.idx_outer_binary,:]

        self.inner_binary_times = self.times[self._next.idx_inner_binary, :]
        self.outer_binary_times = self.times[self._next.idx_outer_binary, :]
        
        #total mass of inner binary
        self.inner_binary_initial_mtotal = self.inner_binary_initial_mass.sum(axis=1)

    def strong_trip_stats(self):

        '''To return some stats about the strong triples'''

        self.N_triple = self._next.t_2nd_overtakes[(self.triple_mask)].size
        self.N_iso_binary = self._next.t_2nd_overtakes[(self.binary_mask)].size
        self.N_strong = self._next.t_2nd_overtakes[(self.strong_triple_mask)].size
        self.N_weak = self._next.t_2nd_overtakes[(self.weak_triple_mask)].size

  

        print("Number of MBHBs in the data: %d"%(self.nmbhb))
        print("Number of triples identified in the data %d"%(self.N_triple))
        print("Number of isolated binaries identified in the data %d"%(self.N_iso_binary))
        print("Number of strong triples in data %d"%(self.N_strong))
        print("Number of weak triples in data %d"%(self.N_weak))

        return None
    
    def get_accreted_mass(self, mass_inn_bin=None):
        """
        Calculates the accreted mass during evolution up to the point where
        the next merger is registered in Illustris (i.e. up to initial point
        of next merger). This is used to identify the inner binary and outer
        binary elements in the next merger. The accreted masses are compared to
        the individual masses of the next binary and the closest one is identif-
        ied as the system of interest
        """
        #find mass growth of inner binary till initial_time[:,0] for next merer
        #For mass growth either use ixsep_1st_overtaken or ixsep_2nd_overtakes
        if mass_inn_bin ==None:
            mass_inn_bin = self.inner_binary_initial_mtotal
            
        # binary_mass_after_accretion = mass_inn_bin
        delta_mass = -1*np.ones(len(mass_inn_bin))

        for i in range(len(mass_inn_bin)):

            dt = self.inner_binary_times[i][1:]-self.inner_binary_times[i][:-1]
            second_binary_initial_time = self.outer_binary_times[i,0]
            dm = self.inner_binary_mdot_eff[i][:-1]*dt
            # print (f"first binary formation time: {first_binary_times[i,0]:e}")
            # print (f"second binary formation time: {second_binary_initial_time:e}")
            idx_2nd_form = np.where(self.inner_binary_times[i][self.inner_binary_times[i]!=0]>=second_binary_initial_time)[0][0]
            dm = dm[:idx_2nd_form]

            delta_mass[i] = dm.sum()
            # Look at the largest growth ~10^10    
            
        binary_mass_after_accretion = mass_inn_bin + delta_mass
        
        return binary_mass_after_accretion, delta_mass
        
    def find_first_binary_in_next_merger(self, mass_inn_bin=None):
        """
        Find the index of the inner binary in next merger using approximate tec-
        hnique of comparing masses after accretion with each index of next merger
        masses and choose the closest one to be our inner bianry 
        """
        
        if mass_inn_bin ==None:
            mass_inn_bin = self.inner_binary_initial_mtotal
            
        first_binary_mass = self.inner_binary_initial_mass
        _binary_mass_after_accretion, _dm_accretion = self.get_accreted_mass(mass_inn_bin=None)        

        # Let us define the outer mass ratios and the intruder masses for triples
        q_outer = -1*np.ones(len(first_binary_mass))
        # q_subhalo_outer = -1*np.ones(len(first_binary_mass))
        intruder_mass = -1*np.ones(len(first_binary_mass))
        inner_binary_mass_in_next_merger = -1*np.ones(len(first_binary_mass))
        idx_first_bin_in_next_merger = -1*np.ones(len(first_binary_mass))
        
        # mass_inn
        for i in range(len(mass_inn_bin)):
            # _binary_mass_after_accretion[i] is just a scalar
            dm = self.outer_binary_initial_mass[i]-_binary_mass_after_accretion[i]
            # dm is an array with only two values pertaining to primary and secondary
            # one of the indices in the outer binary should be close to the accreted inner binary mass
            #find which index is closest to our calculated mass following accretion
            # jsut take the minimum from the two black holes and see what happens
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
    

    def Vesc_calc(self,Mtot,half_radius,sim_scale_factor):
        #add a calculation of Vesc
        scale_radius = (half_radius/1.1815) * sim_scale_factor 
        Vesc = 293.28 * (Mtot**(1/2)) * (scale_radius**(-1/2)) # km/s
        return Vesc
    


    def trip_data(self,save_path,strong_trip_flag = False):

        '''To extract information about the strong triples from Illustris'''

        _,intruder_mass,inner_binary_mass_in_next_merger = self.find_first_binary_in_next_merger()
    
        delta_m = inner_binary_mass_in_next_merger - self.inner_binary_initial_mtotal
        print("faulty accretion %d"%(len(np.argwhere(delta_m<0))))
        #dm_mask = np.argwhere(delta_m>0)
        #sim_unit_h_scale=0.704

        #f-gas,Vesc and BHIDs
        fgas,Vescape =  self.host_galaxy_properties()

        inner_binary_ids = self.merger_ids[self._next.idx_inner_binary]
        outer_binary_ids = self.merger_ids[self._next.idx_outer_binary]       

        masses_inner_binary = self.masses[self.triple_mask]
        masses_inner_binary = np.sort(masses_inner_binary)

        t_form1 = self.inner_binary_times[:,0] #first galaxy merger time
        t_form2 = self.outer_binary_times[:,0] # second galaxy merger time 

        M1_trip_in = masses_inner_binary[:,1] #bigger one is primary
        M2_trip_in = masses_inner_binary[:,0]
        qbh_inner_in = M2_trip_in/M1_trip_in

        M2_trip_temp = M2_trip_in + delta_m* qbh_inner_in # masses after accretion 
        M1_trip_temp = M1_trip_in + delta_m*(1-qbh_inner_in)
        #redifining M1 and M2
        M1_trip = np.maximum(M1_trip_temp,M2_trip_temp)
        M2_trip = np.minimum(M1_trip_temp,M2_trip_temp)
        M3_trip = intruder_mass
        qbh_inner = M2_trip/M1_trip
        qbh_outer = intruder_mass/inner_binary_mass_in_next_merger
               
        t_triple = self._next.t_2nd_overtakes[self.triple_mask]        
        a_triple = self._next.sep_2nd_overtakes[self.triple_mask]

        #age_of_the_universe = cosmo.age(0).to(u.yr).value
        t_evol_binary = self._next.evol_tmrg[self.triple_mask]
        binary_merger_flag,z_evol_binary = self.check_merger_before_z0(t_evol_binary)
        #print(binary_merger_flag)
        binary_merger_flag = np.array(binary_merger_flag)
        #print(binary_merger_flag)
        z_evol_binary = np.array(z_evol_binary) 
        
        strong_in_triple_mask = self.q_inner_outer_strong()
        if(strong_trip_flag == True):
            # t_evol_binary = t_evol_binary[strong_in_triple_mask]
            # binary_merged_before_z0_flag = []
            # for time in t_evol_binary:
            #     if time >= age_of_the_universe:
            #         binary_merged_before_z0_flag.append("No")
            #     else:
            #         binary_merged_before_z0_flag.append("Yes")
            strong_triple_properties={
            'M1': M1_trip[strong_in_triple_mask],
            'M2': M2_trip[strong_in_triple_mask],
            'M3': M3_trip[strong_in_triple_mask],
            'qin': qbh_inner[strong_in_triple_mask],
            'qout':qbh_outer[strong_in_triple_mask],
            't_triple_form':t_triple[strong_in_triple_mask],
            't_form1':t_form1[strong_in_triple_mask],
            't_form2':t_form2[strong_in_triple_mask],
            'bin_merge_flag':binary_merger_flag[strong_in_triple_mask],
            't_bin_merger':t_evol_binary[strong_in_triple_mask],
            'z_bin_merger':z_evol_binary[strong_in_triple_mask],
            'a_triple_form':a_triple[strong_in_triple_mask],
            'bhid_inner':inner_binary_ids[strong_in_triple_mask],
            'bhid_outer':outer_binary_ids[strong_in_triple_mask],
            'sep':self.sep[self.triple_mask][strong_in_triple_mask],
            'dadt':self.dadt[self.triple_mask][strong_in_triple_mask],
            'times':self.times[self.triple_mask][strong_in_triple_mask]
            }

            host_galaxy_properties = {
            "fgas": fgas[self.triple_mask][strong_in_triple_mask],
            "Vescape": Vescape[self.triple_mask][strong_in_triple_mask],
            "SubhaloMass": self.envs_SubhaloMassType[self.triple_mask][strong_in_triple_mask],
            "SubhaloMass_in":self.envs_in_SubhaloMassType[self.triple_mask][strong_in_triple_mask],
            "SubhaloMassInHalfRadType":self.envs_SubhaloMassInHalfRadType[self.triple_mask][strong_in_triple_mask],
            "SubhaloMassInHalfRadType_in":self.envs_in_SubhaloMassInHalfRadType[self.triple_mask][strong_in_triple_mask],
            "SubhaloSFR":self.envs_SubhaloSFR[self.triple_mask][strong_in_triple_mask],
            "SubhaloSFR_in":self.envs_SubhaloSFR[self.triple_mask][strong_in_triple_mask],
            "SubhaloVelDisp_in":self.envs_in_SubhaloVelDisp[self.triple_mask][strong_in_triple_mask],
            "subhalo_id":self.subhalo_id[self.triple_mask][strong_in_triple_mask]
            }

            data_to_save={
                **strong_triple_properties,
                **host_galaxy_properties
            }
            # Convert the dictionary to a Pandas DataFrame


            with h5py.File(save_path + 'strong_triples_data_from_ill.h5', 'w') as hf:
                for key, value in data_to_save.items():
                    if isinstance(value, np.ndarray) and value.dtype.type is np.str_:
                        value = value.astype('U')  # Convert to bytes
                        print(f"Key: {key}, Shape: {value.shape}, Data Type: {value.dtype}")
                    hf.create_dataset(key, data=value)
            # df = pd.DataFrame([M1_trip,M2_trip,M3_trip,qbh_inner,qbh_outer,t_triple,z_triple,t_form1,t_form2,t_evol_binary,a_triple,binary_merged_before_z0_flag,inner_binary_ids_st[:,0],inner_binary_ids_st[:,1],outer_binary_ids_st[:,0],outer_binary_ids_st[:,1],fgas_strong_trip,Vescape_strong_trip,Phi_DM_strong_trip,Phi_star_strong_trip,M_halo_strong_trip,M_star_strong_trip,rhalf_strong_trip,sigma_strong_trip,rbulge_strong_trip])
            # df = df.transpose()
            # df.columns = ['M1','M2','M3','qin','qout','t_triple_form','z_form','t_form1','t_form2','t_merge_bin','a_2nd_ovtks','bin_merger_flag','bhid1','bhid2','bhid3','bhid4','f-gas','Vescape','Phi_DM','Phi_star','M_halo','M_star','rhalf','sigma','rbulge']
            # save_file = save_path + 'strong_triples_data_from_ill.csv'
        
        else:

            all_triple_properties={
            'M1': M1_trip,
            'M2': M2_trip,
            'M3': M3_trip,
            'qin': qbh_inner,
            'qout':qbh_outer,
            't_triple_form':t_triple,
            't_form1':t_form1,
            't_form2':t_form2,
            'bin_merge_flag':binary_merger_flag,
            't_bin_merger':t_evol_binary,
            'z_bin_merger':z_evol_binary,
            'a_triple_form':a_triple,
            'bhid_inner':inner_binary_ids,
            'bhid_outer':outer_binary_ids,
            'sep':self.sep[self.triple_mask],
            'dadt':self.dadt[self.triple_mask],
            'times':self.times[self.triple_mask],
            'strong_trip_key':strong_in_triple_mask
            }

            host_galaxy_properties = {
            "fgas": fgas[self.triple_mask],
            "Vescape": Vescape[self.triple_mask],
            "SubhaloMass": self.envs_SubhaloMassType[self.triple_mask],
            "SubhaloMass_in":self.envs_in_SubhaloMassType[self.triple_mask],
            "SubhaloMassInHalfRadType":self.envs_SubhaloMassInHalfRadType[self.triple_mask],
            "SubhaloMassInHalfRadType_in":self.envs_in_SubhaloMassInHalfRadType[self.triple_mask],
            "SubhaloSFR":self.envs_SubhaloSFR[self.triple_mask],
            "SubhaloSFR_in":self.envs_SubhaloSFR[self.triple_mask],
            "SubhaloVelDisp_in":self.envs_in_SubhaloVelDisp[self.triple_mask],
            "subhalo_id":self.subhalo_id[self.triple_mask]
            }

            data_to_save={
                **all_triple_properties,
                **host_galaxy_properties
            }
            # df_all_triples = pd.DataFrame(data_to_save)
            # df_all_triples.to_csv(save_path + 'all_triples_data_from_ill_update.csv', index=False)
            
            with h5py.File(save_path + 'all_triples_data_from_ill.h5', 'w') as hf:
                for key, value in data_to_save.items():
                    if isinstance(value, np.ndarray):
                        if value.dtype.type is np.str_:
                            value = value.astype('U')  # Convert to Unicode
                        print(f"Key: {key}, Shape: {value.shape}, Data Type: {value.dtype}")
                    hf.create_dataset(key, data=value)

            # df = pd.DataFrame([M1_trip,M2_trip,M3_trip,qbh_inner,qbh_outer,t_triple,z_merger,t_form1,t_form2,a_triple,binary_merged_before_z0_flag,strong_in_triple_mask,inner_binary_ids[:,0],inner_binary_ids[:,1],outer_binary_ids[:,0],outer_binary_ids[:,1],fgas,Vescape,Phi_DM,Phi_star,M_halo,M_star,rhalf,sigma,rbulge])
            # df = df.transpose()
            # df.columns = ['M1','M2','M3','qin','qout','t_triple_form','z_merger','t_form1','t_form2','a_2nd_ovtks','bin_merger_flag','strong_key','bhid1','bhid2','bhid3','bhid4','f-gas','Vescape','Phi_DM','Phi_star','M_halo','M_star','rhalf','sigma','rbulge']
            # save_file = save_path + 'all_triples_data_from_ill.csv'

        # df.to_csv(save_file,index=False)
        # print("File saved at",save_file)

        return None 
    
    def q_inner_outer_strong(self):
        """
        This functions creates the mask for strong triples in
        total triple population
        """
        #all(mst._next.triple_mask == mst.triple_mask) # just for testing purposes
        # print ('triple mask and its length', 
        #        np.where(self.triple_mask)[0], len(np.where(self.triple_mask)[0]))
        # print ('strong triple mask and its length',
        #        np.where(self._next.strong_triple_mask)[0], len(np.where(self._next.strong_triple_mask)[0]))
        idx_triple_mask = np.where(self.triple_mask)[0]
        idx_strong_triple_mask = np.where(self.strong_triple_mask)[0]
        strong_in_triple_mask = [ i in idx_strong_triple_mask for i in idx_triple_mask ]
        return strong_in_triple_mask   
    
    def host_galaxy_properties(self):

        sim_unit_h_scale=0.704
        #returns gas-fraction, Vesc, Masses
        Mgas_half = (self.envs_SubhaloMassInHalfRadType[:,0]+self.envs_in_SubhaloMassInHalfRadType[:,0])
        Mstar_half = (self.envs_SubhaloMassInHalfRadType[:,4]+self.envs_in_SubhaloMassInHalfRadType[:,4])
        fgas = Mgas_half/(Mgas_half + Mstar_half)
        #fgas = Mgas/(Mgas + M*)


        #DM halo is modeled as NFW profile matched to Hernquist
        DM_halo1 = self.envs_SubhaloMassType[:,1]
        DM_halo2 = self.envs_in_SubhaloMassType[:,1]
        DM_total = (DM_halo1+DM_halo2)/sim_unit_h_scale * 1e10 * u.Msun

        # c_vir = 10.5 * (DM_total/(1e12/sim_unit_h_scale * u.Msun))**(-0.11) * self.scales[:,0]
        # f_c = np.log(1+c_vir) - c_vir/(1+c_vir)

        # cosmo_model = FlatLambdaCDM(Om0=0.2726,H0=100*sim_unit_h_scale)
        # z_values = 1/self.scales[:,0] - 1
        # H_z = cosmo_model.H(z_values)
        # x = cosmo_model.Om(z_values) -1 
        # Delta_vir = 18*np.pi**2 + 82*x - 39*x**2
        # r_vir = (2* const.G*DM_total/(H_z**2 * Delta_vir))**(1/3)
        # r_NFW = (r_vir/c_vir).to(u.kpc)
        # scale_radius_h = r_NFW/((1/(2*f_c)**(1/2)) - 1/c_vir)


        
        #scale_radius_h = stellar_halfradius * (1/(1.8153*1.33))
        DM_halfradius = (self.envs_SubhaloHalfmassRadType[:,1]+self.envs_in_SubhaloHalfmassRadType[:,1])/2 * self.scales[:,0]/sim_unit_h_scale * u.kpc
        scale_radius_h = DM_halfradius * (1/(1+np.sqrt(2)))
        Phi_DM = const.G * DM_total/scale_radius_h

        
        M_star = (self.envs_SubhaloMassType[:,4]+self.envs_in_SubhaloMassType[:,4])/sim_unit_h_scale * 1e10 * u.Msun
        BH_mass = (self.envs_SubhaloBHMass+self.envs_in_SubhaloBHMass)/sim_unit_h_scale * 1e10*u.Msun
        stellar_halfradius = (self.envs_SubhaloHalfmassRadType[:,4]+self.envs_in_SubhaloHalfmassRadType[:,4])/2 * self.scales[:,0]/sim_unit_h_scale * u.kpc
        
        Rbulge = stellar_halfradius
        iso_sigma_squared = (const.G * M_star)/Rbulge
        r_soft = const.G*BH_mass/iso_sigma_squared
        r_max = Rbulge

        Phi_star = 2*iso_sigma_squared*np.log(r_max/r_soft)

        Vesc_tot = np.sqrt(2*(Phi_DM+Phi_star)).to(u.km * u.s**(-1)).value

        return fgas,Vesc_tot

    def check_merger_before_z0(self, t_evol_binary):
        binary_merged_before_z0_flag = []
        z_merger = []
        age_of_the_universe = cosmo.age(0).to(u.yr).value
        for time in t_evol_binary:
            if time >= age_of_the_universe:
                binary_merged_before_z0_flag.append(False)
                z_merger.append(0)
            else:
                binary_merged_before_z0_flag.append(True)
                z_merger.append(z_at_value(cosmo.age,(time/10**9)*u.Gyr,zmin=1e-9).value)
        return binary_merged_before_z0_flag,z_merger
    
    def isolated_binaries_data(self,save_path):


        '''This function is for extracting all the info about isolated binaries and store them in
        a csv file'''
        masses_binary = self.masses[self.binary_mask]
        masses_binary = np.sort(masses_binary)
        M1_binary = masses_binary[:,1] #bigger one is primary
        M2_binary = masses_binary[:,0]
        #qbh_inner = M2_binary/M1_binary
        #mbh_t_form = self.sim_tmrg[self.binary_mask] #formation time of the binary
        t_evol_binary = self._next.evol_tmrg[self.binary_mask] #evolution time of binary in inspiral
        age_of_the_universe = cosmo.age(0).to(u.yr).value
        merged_before_z0_flag = []
        z_binary = []
        for time in t_evol_binary:
            if time >= age_of_the_universe:
                merged_before_z0_flag.append(False)
                z_binary.append(0)
            else:
                merged_before_z0_flag.append(True)
                z_binary.append(z_at_value(cosmo.age,(time/10**9)*u.Gyr,zmin=1e-13).value)

        binary_properties={
            'M1': M1_binary,
            'M2': M2_binary,
            'qin': M2_binary/M1_binary,
            't_form':self.sim_tmrg[self.binary_mask],
            't_merge': t_evol_binary,
            'z_merge':z_binary,
            'merger_flag':merged_before_z0_flag,
            'binary_ids':self.merger_ids[self.binary_mask],
            'sep':self.sep[self.binary_mask],
            'dadt':self.dadt[self.binary_mask],
            'times':self.times[self.binary_mask]
        }
        #gas-fraction, Escape speed and BHIDs
        fgas,Vescape= self.host_galaxy_properties()

        host_galaxy_properties = {
        "fgas": fgas[self.binary_mask],
        "Vescape": Vescape[self.binary_mask],
        "SubhaloMass": self.envs_SubhaloMassType[self.binary_mask],
        "SubhaloMass_in":self.envs_in_SubhaloMassType[self.binary_mask],
        "SubhaloMassInHalfRadType":self.envs_SubhaloMassInHalfRadType[self.binary_mask],
        "SubhaloMassInHalfRadType_in":self.envs_in_SubhaloMassInHalfRadType[self.binary_mask],
        "SubhaloSFR":self.envs_SubhaloSFR[self.binary_mask],
        "SubhaloSFR_in":self.envs_SubhaloSFR[self.binary_mask],
        "SubhaloVelDisp_in":self.envs_in_SubhaloVelDisp[self.binary_mask],
        "subhalo_id":self.subhalo_id[self.binary_mask]
        }

         # Prepare data to store in a dictionary
        data_to_save = {
            **binary_properties,
            **host_galaxy_properties  # Unpack host galaxy properties into the dictionary
        }

    # Save all data using a loop
        with h5py.File(save_path + 'iso_binaries_data_from_ill.h5', 'w') as hf:
            for key, value in data_to_save.items():
                if isinstance(value, np.ndarray) and value.dtype.type is np.str_:
                    value = value.astype('U')  # Convert to bytes
                    print(f"Key: {key}, Shape: {value.shape}, Data Type: {value.dtype}")
                hf.create_dataset(key, data=value)
        print("File saved at", save_path + "iso_binaries_data_from_ill.h5")

        # df = pd.DataFrame([M1_binary,M2_binary,qbh_inner,t_evol_binary,z_binary,mbh_t_form,merged_before_z0_flag,binary_ids[:,0],binary_ids[:,1],fgas,Vescape,Phi_DM,Phi_star,M_halo,M_star,rhalf,sigma,rbulge])
        # df = df.transpose()
        # df.columns = ['M1','M2','qin','t_merger','z_merger','t_form','merger_flag','bhid1','bhid2','f-gas','Vescape','Phi_DM','Phi_star','M_halo','M_star','rhalf','sigma','rbulge']
        # df.to_csv(save_file+'iso_binaries_data_from_ill.csv',index=False)
        # print("File saved at",save_file+"iso_binaries_data_from_ill.csv")

        return None


    def save_strong_triples_a_and_dadt(self,filepath):

        filename = filepath+"a_and_dadt_for_all_strong_triples.npz"
        #This gives True/False mask on the triples that classify them as strong triples or weak triples
        
        triples_seps = self.sep[self.triple_mask]
        triples_times = self.times[self.triple_mask]
        triples_dadt_df = self.dadt_df[self.triple_mask]
        triples_dadt_lc = self.dadt_lc[self.triple_mask]
        triples_dadt_vd = self.dadt_vd[self.triple_mask]
        triples_dadt_gw = self.dadt_gw[self.triple_mask]

        strong_in_triple_mask = self.q_inner_outer_strong()
        strong_triples_times = triples_times[strong_in_triple_mask]
        strong_triples_seps = triples_seps[strong_in_triple_mask]

        strong_triple_dadt_df = triples_dadt_df[strong_in_triple_mask]
        strong_triple_dadt_lc = triples_dadt_lc[strong_in_triple_mask]
        strong_triple_dadt_vd = triples_dadt_vd[strong_in_triple_mask]
        strong_triple_dadt_gw = triples_dadt_gw[strong_in_triple_mask]

        strong_triples_dadt = strong_triple_dadt_df+strong_triple_dadt_lc+strong_triple_dadt_vd+strong_triple_dadt_gw

        np.savez(filename,st_seps=strong_triples_seps,st_dadt=strong_triples_dadt)
        print("sep's of size %d and dadt's of size %d stored at %s"%(len(strong_triples_seps),len(strong_triples_dadt),filename))
              
        return None

