import numpy as np
import random
import pandas as pd
import interpolate as inter
from tables_merger_fraction import *
import BH_kicks as kicks
import merger_rate_calculate as mr

import scipy.stats as st
from scipy.interpolate import interp1d
import scipy.integrate as integrate

from tqdm import tqdm
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const

import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science']) 
plt.rcParams.update({'font.size': 20})


age_of_the_universe = cosmo.age(0).to(u.yr).value

savepath = '/Users/pranavsatheesh/Triples/Github/Illustris_Data/'
a_dadt_filename = savepath+"a_and_dadt_for_all_strong_triples.npz"
a_dadt_file =np.load(a_dadt_filename)
st_seps = a_dadt_file['st_seps']
st_dadt = a_dadt_file['st_dadt']

def tau_prompt_merger_bonetti():
    '''Returns the merger time for Prompt merger from Bonetti'''

    tau_lognormal_mean = np.log(10)*8.4
    tau_lognormal_std = np.log(10)*0.4

    return st.lognorm.rvs(scale=np.exp(tau_lognormal_mean),s=tau_lognormal_std)

def timsecale_to_merger(strong_triple_ix,a_triple_after):
        
        '''
        find tinsp using interplating sep, dadt and performing integration on the object function
        '''
        interp_func = tinsp_interpolate(st_seps[strong_triple_ix],st_dadt[strong_triple_ix])
        tau = integrate.quad(interp_func,a_triple_after,0)[0]

        return tau

def tinsp_interpolate(sep, dadt):
        '''
        Perform interpolation for finding t inspiral
        interpolates x=sep, y=1/dadt_total
        '''
        func = 1/dadt
        #remove infinities for interpolatio to avoid biased results
        func[func==np.inf] = np.nanmin(func)/10**10  #find a small value compared to minimum
        #perform interpolation
        
        
        #print(func.shape)
        #print(sep.shape)
        interp_func = interp1d(sep, func, bounds_error=False, fill_value=0) 
        return interp_func


class Tripledynamics:
    """
    Reads in the strong triple data returned from mstest file and interpolates it 
    with Bonetti simulation results to assign outcomes to all triple MBH population"""

    def __init__(self,file_path='/Users/pranavsatheesh/Triples/Github/Illustris_Data/'):
        self.strong_triples_file_path = file_path+"strong_triples_data_from_ill.csv"
        #self.all_triples_file_path = file_path+"all_triples_data_from_ill.csv"
        #self.is_binaries_file_path = file_path+"iso_binaries_data_from_ill.csv"
        #opening strong triples panda data frame
        df_triples = pd.read_csv(self.strong_triples_file_path,index_col=False)
        df_triples.columns = df_triples.columns.str.replace("-", "_", regex=True)    

        for column in df_triples:
            # Convert the column to a list and set it as an attribute
            setattr(self, column, df_triples[column].to_numpy())

        self.N_strong_triples = len(self.M1)
        self.M1_ill = self.M1
        self.qin_ill = self.qin
        self.qout_ill = self.qout
        self.M2_ill = self.M2
        self.M3_ill = self.M3
        self.f_gas = self.f_gas
        self.a_triple_ovtks_ill = self.a_2nd_ovtks
    
        self.interpolate_with_bonetti()

    def interpolate_with_bonetti(self):

    
        prompt_merger = 0
        merger_after_ejection=0
        no_merger= 0

        t_triple_merger_values = []
        a_triple_interaction = []
        z_triple_merger = []
        merger_flags = []

        slingshot_kicks=[]
        gw_kick_random=[]
        gw_kick_5deg=[]
        gw_kick_hybrid=[]

        for i in range(self.N_strong_triples): #looping over all strong triples

            ahard = kicks.a_hard(self.M1_ill[i], self.qin_ill[i]) #hardening radius

            if(ahard<self.a_triple_ovtks_ill[i]): #defining the seperation at the time of triple formation
                a_triple = ahard
            else:
                a_triple = self.a_triple_ovtks_ill[i]
            
            m_int = self.M3_ill[i]
            m1_bin = self.M1_ill[i]
            m2_bin = self.M2_ill[i]

            if(np.log10(self.qout_ill[i])<=0):
                a_P = trilinear_interp([np.log10(self.M1_ill[i]),np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], m1_bonetti, qout_bonetti, qin_bonetti, prompt_merger_frac12)*0.01
                b_P = trilinear_interp([np.log10(self.M1_ill[i]),np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], m1_bonetti, qout_bonetti, qin_bonetti, prompt_merger_frac13)*0.01
                c_P = trilinear_interp([np.log10(self.M1_ill[i]),np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], m1_bonetti, qout_bonetti, qin_bonetti, prompt_merger_frac23)*0.01
                d_P = (trilinear_interp([np.log10(self.M1_ill[i]),np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], m1_bonetti ,qout_bonetti, qin_bonetti, delayed_merger_frac12)+
                     trilinear_interp([np.log10(self.M1_ill[i]),np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], m1_bonetti, qout_bonetti, qin_bonetti, delayed_merger_frac13)+
                     trilinear_interp([np.log10(self.M1_ill[i]),np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], m1_bonetti, qout_bonetti, qin_bonetti, delayed_merger_frac23))*0.01
            else:
                #big-perturber
                a_P = bilinear_interp([np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], qout_bigp_bonetti, qin_bonetti, prompt_merger_frac_bigp12)*0.01
                b_P = bilinear_interp([np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], qout_bigp_bonetti, qin_bonetti, prompt_merger_frac_bigp13)*0.01
                c_P = bilinear_interp([np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], qout_bigp_bonetti, qin_bonetti, prompt_merger_frac_bigp23)*0.01
                d_P = (bilinear_interp([np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], qout_bigp_bonetti, qin_bonetti, delayed_merger_frac_bigp12)+
                     bilinear_interp([np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], qout_bigp_bonetti, qin_bonetti, delayed_merger_frac_bigp23)+
                     bilinear_interp([np.log10(self.qout_ill[i]),np.log10(self.qin_ill[i])], qout_bigp_bonetti, qin_bonetti, delayed_merger_frac_bigp13))*0.01

            P = random.uniform(0,1)

            if(P <= a_P+b_P+c_P):
                #prompt-merger
             
                t_triple_merge = self.t_triple_form[i] + tau_prompt_merger_bonetti()
                t_triple_merger_values.append(t_triple_merge)
                a_triple_interaction.append(a_triple)

                if(t_triple_merge<=age_of_the_universe):
                    #prompt merger with t<tH

                    prompt_merger = prompt_merger + 1
                    z_triple_merger.append(z_at_value(cosmo.age,(t_triple_merge/10**9)*u.Gyr,zmin=1e-9).value)
                    merger_flags.append("Tr")
                    
                    if(P > a_P+b_P and P <= a_P+b_P+c_P):
                        #m3 exchanges with m1. m2 and m3 merges
                        vsling,a_new,qin_new = kicks.v_and_a_after_slingshot(m1_bin,m2_bin,m_int,a_triple,True)

                    else:
                        vsling,a_new,qin_new = kicks.v_and_a_after_slingshot(m1_bin,m2_bin,m_int,a_triple,False)

                    slingshot_kicks.append(vsling)
                    rand,hyb,deg5 = kicks.gw_kick_calc(qin_new,self.f_gas[i])
                    gw_kick_random.append(rand)
                    gw_kick_5deg.append(deg5)
                    gw_kick_hybrid.append(hyb)
            
                else:

                    no_merger = no_merger + 1
                    merger_flags.append("No")
                    slingshot_kicks.append(0)
                    gw_kick_random.append(0)
                    gw_kick_5deg.append(0)
                    gw_kick_hybrid.append(0)
                    z_triple_merger.append(0)
                    
            
            else:
                

                vsling,a_triple_after,qin_new = kicks.v_and_a_after_slingshot(m1_bin,m2_bin,m_int,a_triple,False)
                
                slingshot_kicks.append(vsling)

                #merger time
                strong_triple_index = i
                tau_merger = timsecale_to_merger(strong_triple_index,a_triple_after)
                t_triple_merge = self.t_triple_form[i]+tau_merger
                t_triple_merger_values.append(t_triple_merge)
                a_triple_interaction.append(a_triple_after)

                if(t_triple_merge<=age_of_the_universe):
                    merger_flags.append("Tr-ej")
                    merger_after_ejection = merger_after_ejection + 1
                    rand,hyb,deg5 = kicks.gw_kick_calc(qin_new,self.f_gas[i])
                    gw_kick_random.append(rand)
                    gw_kick_5deg.append(deg5)
                    gw_kick_hybrid.append(hyb)
                    z_triple_merger.append(z_at_value(cosmo.age,(t_triple_merge/10**9)*u.Gyr,zmin=1e-9).value)

                else:
                    no_merger = no_merger + 1
                    merger_flags.append("No")  
                    gw_kick_random.append(0)
                    gw_kick_5deg.append(0)
                    gw_kick_hybrid.append(0)
                    z_triple_merger.append(0)

        z_triple_merger = np.array(z_triple_merger)
        a_triple_interaction = np.array(a_triple_interaction)
        t_triple_merger_values = np.array(t_triple_merger_values)
        
        self.triple_mergers_times = t_triple_merger_values
        self.z_triple_merger = z_triple_merger
        self.merger_flags = merger_flags
        self.a_triple_after = a_triple_interaction

        self.slingshot_kicks = slingshot_kicks
        self.gw_kick_random =gw_kick_random
        self.gw_kick_5deg =gw_kick_5deg
        self.gw_kick_hybrid =gw_kick_hybrid


        self.prompt_merger = prompt_merger 
        self.no_merger= no_merger
        self.merger_after_ejection = merger_after_ejection

        self.prompt_merger_mask = [s == 'Tr' for s in self.merger_flags]
        self.merger_after_ejection_mask = [s == 'Tr-ej' for s in self.merger_flags]
        self.no_merger_mask = [s == 'No' for s in self.merger_flags]
        self.merger_mask = [s != 'No' for s in self.merger_flags]

    # def strong_trp_stats(self):

    def total_merger_rate(self,merger_arg,zbinsize,zmax):
    
        if(merger_arg=="Tr"):
            mrate= mr.merger_rate_find(self.z_triple_merger[self.prompt_merger_mask],zbinsize,zmax)
            
        elif(merger_arg=="Tr-ej"):
            mrate = mr.merger_rate_find(self.z_triple_merger[self.merger_after_ejection_mask],zbinsize,zmax)

        else:
            mrate = mr.merger_rate_find(self.z_triple_merger[self.merger_mask],zbinsize,zmax)
            
        return mrate
        
    def diff_merger_Rate_for_plot(self,merger_arg,lgzbinsize,lgzmin,lgzmax):
    
        if(merger_arg=="Tr"):
            lgzbins,dNdlogzdt = mr.diff_merger_rate(self.z_triple_merger[self.prompt_merger_mask],lgzbinsize,lgzmin,lgzmax)
            
        elif(merger_arg=="Tr-ej"):
            lgzbins,dNdlogzdt = mr.diff_merger_rate(self.z_triple_merger[self.merger_after_ejection_mask],lgzbinsize,lgzmin,lgzmax)

        else:
            lgzbins,dNdlogzdt = mr.diff_merger_rate(self.z_triple_merger[self.merger_mask],lgzbinsize,lgzmin,lgzmax)
            
        return lgzbins,dNdlogzdt
    
    def escape_rate_for_plot(self,lgzbinsize,lgzmin,lgzmax):

        lgzbins,sling_escape = mr.diff_merger_rate(self.z_triple_merger[(self.slingshot_kicks>self.Vescape)&(self.z_triple_merger>0)],lgzbinsize,lgzmin,lgzmax)
        lgzbins,rand_escape = mr.diff_merger_rate(self.z_triple_merger[self.gw_kick_random>self.Vescape],lgzbinsize,lgzmin,lgzmax)
        lgzbins,hybrid_escape = mr.diff_merger_rate(self.z_triple_merger[self.gw_kick_hybrid>self.Vescape],lgzbinsize,lgzmin,lgzmax)
        lgzbins,deg5_escape = mr.diff_merger_rate(self.z_triple_merger[self.gw_kick_5deg>self.Vescape],lgzbinsize,lgzmin,lgzmax)

        return lgzbins,[sling_escape,rand_escape,hybrid_escape,deg5_escape]

class iso_binary():
    
    
    def __init__(self,file_path='/Users/pranavsatheesh/Triples/Github/Illustris_Data/'):
    #self.strong_triples_file_path = file_path+"strong_triples_data_from_ill.csv"
    #self.all_triples_file_path = file_path+"all_triples_data_from_ill.csv"
        self.iso_binaries_file_path = file_path+"iso_binaries_data_from_ill.csv"
        iso_binaries = pd.read_csv(self.iso_binaries_file_path)
        iso_binaries.columns = iso_binaries.columns.str.replace("-", "_", regex=True)    

        for column in iso_binaries:
        # Convert the column to a list and set it as an attribute
            setattr(self, column, iso_binaries[column].to_numpy())
    
        self.N_iso_binaries = len(self.M1)
        self.merger_mask = [s == 'Yes' for s in self.merger_flag]
        self.kick_assign()
    
    def kick_assign(self):

        gw_kick_random = []
        gw_kick_hybrid = []
        gw_kick_5deg = []

        for i in range(self.N_iso_binaries):
            vgw_rand,vgw_hybrid,vgw_5deg = kicks.gw_kick_calc(self.qin[i],self.f_gas[i])
            gw_kick_random.append(vgw_rand)
            gw_kick_hybrid.append(vgw_hybrid)
            gw_kick_5deg.append(vgw_5deg)

        self.gw_kick_random = gw_kick_random
        self.gw_kick_hybrid = gw_kick_hybrid
        self.gw_kick_5deg = gw_kick_5deg


    def total_merger_rate(self,zbinsize,zmax):
    
        mrate= mr.merger_rate_find(self.z_merger[self.merger_mask],zbinsize,zmax)
        return mrate

    def diff_merger_Rate_for_plot(self,lgzbinsize,lgzmin,lgzmax):

        lgzbins,dNdlogzdt = mr.diff_merger_rate(self.z_merger[self.merger_mask],lgzbinsize,lgzmin,lgzmax)  
        return lgzbins,dNdlogzdt

    def escape_rate_for_plot(self,lgzbinsize,lgzmin,lgzmax):

        lgzbins,rand_escape = mr.diff_merger_rate(self.z_merger[(self.gw_kick_random>self.Vescape)&(self.z_merger>0)],lgzbinsize,lgzmin,lgzmax)
        lgzbins,hybrid_escape = mr.diff_merger_rate(self.z_merger[(self.gw_kick_hybrid>self.Vescape)&(self.z_merger>0)],lgzbinsize,lgzmin,lgzmax)
        lgzbins,deg5_escape = mr.diff_merger_rate(self.z_merger[(self.gw_kick_5deg>self.Vescape)&(self.z_merger>0)],lgzbinsize,lgzmin,lgzmax)

        return lgzbins,[rand_escape,hybrid_escape,deg5_escape]
    
class weak_triples():

    def __init__(self,file_path='/Users/pranavsatheesh/Triples/Github/Illustris_Data/'):
    #self.strong_triples_file_path = file_path+"strong_triples_data_from_ill.csv"
    #self.all_triples_file_path = file_path+"all_triples_data_from_ill.csv"
        self.all_triples_file_path = file_path+"all_triples_data_from_ill.csv"
        df_all_triples = pd.read_csv(self.all_triples_file_path)
        df_weak_triples = df_all_triples[df_all_triples["strong_key"]!=True]
        df_weak_triples.columns = df_weak_triples.columns.str.replace("-", "_", regex=True)    

        for column in df_weak_triples:
        # Convert the column to a list and set it as an attribute
            setattr(self, column, df_weak_triples[column].to_numpy())
    
        self.N_weak_triples = len(self.M1)
        self.merger_mask = [s == 'Yes' for s in self.bin_merger_flag]
        self.kick_assign()
    
    def kick_assign(self):

        gw_kick_random = []
        gw_kick_hybrid = []
        gw_kick_5deg = []

        for i in range(self.N_weak_triples):         
            vgw_rand,vgw_hybrid,vgw_5deg = kicks.gw_kick_calc(self.qin[i],self.f_gas[i])
            gw_kick_random.append(vgw_rand)
            gw_kick_hybrid.append(vgw_hybrid)
            gw_kick_5deg.append(vgw_5deg)

        self.gw_kick_random = gw_kick_random
        self.gw_kick_hybrid = gw_kick_hybrid
        self.gw_kick_5deg = gw_kick_5deg

    def total_merger_rate(self,zbinsize,zmax):
    
        mrate= mr.merger_rate_find(self.z_merger[self.merger_mask],zbinsize,zmax)
        return mrate

    def diff_merger_Rate_for_plot(self,lgzbinsize,lgzmin,lgzmax):

        lgzbins,dNdlogzdt = mr.diff_merger_rate(self.z_merger[self.merger_mask],lgzbinsize,lgzmin,lgzmax)  
        return lgzbins,dNdlogzdt


    def escape_rate_for_plot(self,lgzbinsize,lgzmin,lgzmax):

        lgzbins,rand_escape = mr.diff_merger_rate(self.z_merger[(self.gw_kick_random>self.Vescape)&(self.z_merger>0)],lgzbinsize,lgzmin,lgzmax)
        lgzbins,hybrid_escape = mr.diff_merger_rate(self.z_merger[(self.gw_kick_hybrid>self.Vescape)&(self.z_merger>0)],lgzbinsize,lgzmin,lgzmax)
        lgzbins,deg5_escape = mr.diff_merger_rate(self.z_merger[(self.gw_kick_5deg>self.Vescape)&(self.z_merger>0)],lgzbinsize,lgzmin,lgzmax)

        return lgzbins,[rand_escape,hybrid_escape,deg5_escape]   

def find_invalid_mergers(strong_tr,weak_tr,iso_bin,gw_kick_key):
    '''
    gw_kick_key = {'random','5deg','hybrid'}

    '''
    gw_kick_attr = "gw_kick_"+gw_kick_key

    iso_invalid_merger_mask = np.zeros_like(iso_bin.bhid1, dtype=bool)
    weak_triple_invalid_merger_mask = np.zeros_like(weak_tr.bhid1, dtype=bool)
    strong_triple_invalid_merger_mask = np.zeros_like(strong_tr.bhid1, dtype=bool)

    iso_kick_eject_mask =  getattr(iso_bin,gw_kick_attr)>iso_bin.Vescape
    weak_triple_kick_eject_mask =  getattr(weak_tr,gw_kick_attr)>weak_tr.Vescape
    strong_triple_kick_eject_mask =  getattr(strong_tr,gw_kick_attr)>strong_tr.Vescape
    slingshot_eject_mask = strong_tr.slingshot_kicks>strong_tr.Vescape


    bhid_cols_in_iso_bins = {"bhid1","bhid2"}
    bhid_cols_in_trips = {"bhid1","bhid2","bhid3","bhid4"}


#iso affected by "gw-key"
    for i, bhid_x in enumerate(bhid_cols_in_iso_bins):
        for j, bhid_y in enumerate(bhid_cols_in_iso_bins):

            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(iso_bin,bhid_y)[iso_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidy_indices = np.array([np.where(getattr(iso_bin,bhid_y) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidy_indices
                bhidx_t_merger = iso_bin.t_merger[bhidx_indices]
                bhidy_t_merger = iso_bin.t_merger[bhidy_indices]
                iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidy_t_merger[different_indices_mask] < bhidx_t_merger[different_indices_mask]

            else:
                continue

        for k,bhid_z in enumerate(bhid_cols_in_trips):
            common_occurrences_of_z_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(weak_tr,bhid_z)[weak_triple_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_z_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(weak_tr,bhid_z) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_t_merger = iso_bin.t_merger[bhidx_indices]
                bhid_wt_t_form = weak_tr.t_triple_form[bhidz_indices]
                iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_wt_t_form[different_indices_mask] < bhid1_t_merger[different_indices_mask]
                #weak_triple_invalid_merger_mask[bhidz_indices[different_indices_mask]] |= bhid1_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            else:
                continue

            common_occurrences_of_strong_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(strong_tr,bhid_z)[strong_triple_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_strong_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_t_merger = iso_bin.t_merger[bhidx_indices]
                bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_t_merger[different_indices_mask]
                #strong_triple_invalid_merger_mask[bhidz_indices[different_indices_mask]] |= bhid1_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            else:
                continue

            common_occurrences_of_sling_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(strong_tr,bhid_z)[slingshot_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_sling_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_t_merger = iso_bin.t_merger[bhidx_indices]
                bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_t_merger[different_indices_mask]


#weak triples affected by "gw-key"

    for i,bhid_x in enumerate(bhid_cols_in_trips):
        for j, bhid_y in enumerate(bhid_cols_in_iso_bins):

            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(iso_bin,bhid_y)[iso_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]

            if len(bhidx_indices) > 0:

                bhidy_indices = np.array([np.where(getattr(iso_bin,bhid_y) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidy_indices
                bhidy_t_merger = iso_bin.t_merger[bhidy_indices]
                bhid_wt_t_form = weak_tr.t_triple_form[bhidx_indices]
                weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidy_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]

            else:
                continue

        for k, bhid_z in enumerate(bhid_cols_in_iso_bins):

            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(weak_tr,bhid_z)[weak_triple_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]


            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(weak_tr,bhid_z) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhidz_t_form = weak_tr.t_triple_form[bhidz_indices]
                bhid_wt_t_form = weak_tr.t_triple_form[bhidx_indices]
                weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidz_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]  
            
            else:
                continue
            
            common_occurrences_of_strong_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(strong_tr,bhid_z)[strong_triple_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_strong_ejected_in_x)[0]
            
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_t_form = weak_tr.t_triple_form[bhidx_indices]
                bhid_wt_t_form =strong_tr.t_triple_form[bhidz_indices]
                weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhid1_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            else:
                continue

            common_occurrences_of_sling_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(strong_tr,bhid_z)[slingshot_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_sling_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_wt_t_form = weak_tr.t_triple_form[bhidx_indices]
                bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_wt_t_form[different_indices_mask]
        
        
#strong triples affected by "gw-key"
            
    for i,bhid_x in enumerate(bhid_cols_in_trips):
        for j, bhid_y in enumerate(bhid_cols_in_iso_bins):

            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(iso_bin,bhid_y)[iso_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]

            if len(bhidx_indices) > 0:

                bhidy_indices = np.array([np.where(getattr(iso_bin,bhid_y) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidy_indices
                bhidy_t_merger = iso_bin.t_merger[bhidy_indices]
                bhid_wt_t_form = strong_tr.t_triple_form[bhidx_indices]
                strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidy_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]

            else:
                continue

        for k, bhid_z in enumerate(bhid_cols_in_iso_bins):

            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(weak_tr,bhid_z)[weak_triple_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]


            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(weak_tr,bhid_z) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhidz_t_form = weak_tr.t_triple_form[bhidz_indices]
                bhid_wt_t_form = strong_tr.t_triple_form[bhidx_indices]
                strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidz_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]  
            
            else:
                continue
            
            common_occurrences_of_strong_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(strong_tr,bhid_z)[strong_triple_kick_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_strong_ejected_in_x)[0]
            
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid_wt_t_form = strong_tr.t_triple_form[bhidz_indices]
                bhid1_t_form = strong_tr.t_triple_form[bhidx_indices]
                strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhid1_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            else:
                continue

            common_occurrences_of_sling_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(strong_tr,bhid_z)[slingshot_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_sling_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_wt_t_form = strong_tr.t_triple_form[bhidx_indices]
                bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_wt_t_form[different_indices_mask]
        
    return iso_invalid_merger_mask,weak_triple_invalid_merger_mask,strong_triple_invalid_merger_mask

def plot_merger_rates(Tr_objects,weak_tr,iso_bin,Nruns):

    color_palette = {"strong_trip":"#377eb8","weak_trip":"#a2c8ec","iso":"#ff800e","all":"#898989"}

    dNdlogzdt_strong_Tr = []
    dNdlogzdt_strong_Tr_ej = []
    dNdlogzdt_strong_tot = []
    for i in range(Nruns):
        lgzbins_strong_Tr,dNdlogzdt_strong_Tr_tmp = Tr_objects[i].diff_merger_Rate_for_plot(merger_arg="Tr",lgzbinsize=0.2,lgzmin=-3,lgzmax=1.0)
        dNdlogzdt_strong_Tr.append(dNdlogzdt_strong_Tr_tmp)
        lgzbins_strong_Tr_ej,dNdlogzdt_strong_Tr_ej_tmp = Tr_objects[i].diff_merger_Rate_for_plot(merger_arg="Tr-ej",lgzbinsize=0.2,lgzmin=-3,lgzmax=1.0)
        dNdlogzdt_strong_Tr_ej.append(dNdlogzdt_strong_Tr_ej_tmp)
        lgzbins_strong_tot,dNdlogzdt_strong_tot_tmp = Tr_objects[i].diff_merger_Rate_for_plot(merger_arg="all",lgzbinsize=0.2,lgzmin=-3,lgzmax=1.0)
        dNdlogzdt_strong_tot.append(dNdlogzdt_strong_tot_tmp)

    dNdlogzdt_strong_total = []
    for i in range(Nruns):
        lgzbins_strong_total,dNdlogzdt_strong_tot_tmp = Tr_objects[i].diff_merger_Rate_for_plot(merger_arg="all",lgzbinsize=0.15,lgzmin=-3,lgzmax=1.0)
        dNdlogzdt_strong_total.append(dNdlogzdt_strong_tot_tmp)

    lgzbins_iso,dNdlogzdt_iso = iso_bin.diff_merger_Rate_for_plot(lgzbinsize=0.15,lgzmin=-3,lgzmax=1.0)
    lgzbins_weak,dNdlogzdt_weak = weak_tr.diff_merger_Rate_for_plot(lgzbinsize=0.15,lgzmin=-3,lgzmax=1.0)
    dNdlogzdt_all = dNdlogzdt_iso + dNdlogzdt_weak + np.mean(dNdlogzdt_strong_total,axis=0)

    fig,ax = plt.subplots(1,2,figsize=(12,5),sharey=True)

    ax[0].plot(lgzbins_iso,dNdlogzdt_all,color=color_palette["all"],linewidth=2,label="Total")
    ax[0].plot(lgzbins_iso,dNdlogzdt_iso,color=color_palette["iso"],label="Isolated binaries",linewidth=2,alpha=0.5)
    ax[0].plot(lgzbins_weak,dNdlogzdt_weak,color=color_palette["weak_trip"],linewidth=2,label="Weak triples")
    ax[0].plot(lgzbins_strong_tot,np.mean(dNdlogzdt_strong_tot,axis=0),color=color_palette["strong_trip"],linewidth=2,label="Strong triples")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1e-8,1)
    ax[0].set_yticks([1e-1,1e-3,1e-5,1e-7])
    ax[0].set_xticks(np.arange(-3,1.5,1))
    ax[0].set_xlabel("$\log z$",fontsize=20)
    ax[0].set_ylabel(r"$\log (d^2 N / (d \log z dt)  \times 1\text{yr})$",fontsize=20)
    ax[0].legend(fontsize=14,loc="upper left")

    ax[1].plot(lgzbins_strong_tot,np.mean(dNdlogzdt_strong_tot,axis=0),color=color_palette["strong_trip"],linewidth=2,label="Strong triples")
    ax[1].plot(lgzbins_strong_Tr,np.mean(dNdlogzdt_strong_Tr,axis=0),color=color_palette["strong_trip"],linestyle="--",linewidth=1.5,label="Prompt merger")
    ax[1].plot(lgzbins_strong_Tr_ej,np.mean(dNdlogzdt_strong_Tr_ej,axis=0),color=color_palette["strong_trip"],linestyle="-.",linewidth=1.5,label="Merger after kick")
    ax[1].set_ylim(1e-8,1)
    ax[1].yaxis.set_tick_params(which='minor',bottom=False)
    ax[1].set_yscale("log")
    ax[1].set_yticks([1e-1,1e-3,1e-5,1e-7])
    ax[1].set_xticks(np.arange(-3,1.5,1))
    ax[1].set_xlabel("$\log z$",fontsize=20)
    ax[1].set_ylabel(r"$\log (d^2 N / (d \log z dt)  \times 1\text{yr})$",fontsize=20)
    ax[1].legend(fontsize=14,loc="upper left")

    np.savetxt(savepath+"triple_system_merger_rates.txt",(lgzbins_strong_tot,np.mean(dNdlogzdt_strong_tot,axis=0)))
    np.savetxt(savepath+"all_system_merger_rates.txt",(lgzbins_iso,dNdlogzdt_all))
    print(f"The merger rate files are saved at {savepath}")   

    plt.tight_layout()

    return fig,ax

def plot_ejection_rates(Tr_objects,weak_tr,iso_bin,Nruns):
    sling_Tr = []
    rand_Tr = []
    hybrid_Tr = []
    deg5_Tr = []
    kick_colors = {'slingshot':"#4daf4a",'aligned':"#377eb8","hybrid":"#a2c8ec","random":"#e41a1c"}
    fig,ax = plt.subplots(1,2,figsize=(15,7),sharey=True)
    #triple only 
    for i in range(Nruns):
        lgzbins_Tr,dNdlogzdt_strong = Tr_objects[i].escape_rate_for_plot(lgzbinsize=0.25,lgzmin=-3,lgzmax=1.0)

        sling_Tr.append(dNdlogzdt_strong[0])
        rand_Tr.append(dNdlogzdt_strong[1])
        hybrid_Tr.append(dNdlogzdt_strong[2])
        deg5_Tr.append(dNdlogzdt_strong[3])

    ax[1].plot(lgzbins_Tr,np.mean(sling_Tr,axis=0),color=kick_colors['slingshot'],label="slingshot",linewidth=2)
    ax[1].plot(lgzbins_Tr,np.mean(rand_Tr,axis=0),color=kick_colors['random'],label="GW-random",linewidth=2)
    ax[1].plot(lgzbins_Tr,np.mean(hybrid_Tr,axis=0),color=kick_colors['hybrid'],label="GW-hybrid",linewidth=2)
    ax[1].plot(lgzbins_Tr,np.mean(deg5_Tr,axis=0),color=kick_colors['aligned'],label="GW-aligned",linewidth=2)

    lgztrip_bins,alltrip_mr = np.loadtxt(savepath+"triple_system_merger_rates.txt")

    ax[1].plot(lgztrip_bins,alltrip_mr,color="black",linestyle="--",label="Merger rate",linewidth=2)
    ax[1].set_ylim(2*10**-8,)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("$\log z$",fontsize=25)
    ax[1].legend(fontsize=20)
    ax[1].set_title("Strong triples",fontsize=25)

    #sling_all = []
    rand_all = []
    hybrid_all = []
    deg5_all = []

    #for all systems
    for i in range(Nruns):
        lgzbins_all,dNdlogzdt_strong = Tr_objects[i].escape_rate_for_plot(lgzbinsize=0.2,lgzmin=-3,lgzmax=1.0)
        lgzbins_all,dNdlogzdt_weak = weak_tr.escape_rate_for_plot(lgzbinsize=0.2,lgzmin=-3,lgzmax=1.0)
        lgzbins_all,dNdlogzdt_iso = iso_bin.escape_rate_for_plot(lgzbinsize=0.2,lgzmin=-3,lgzmax=1.0)

        #sling_all.append(dNdlogzdt_strong[0])
        rand_all.append(dNdlogzdt_strong[1]+dNdlogzdt_weak[0]+dNdlogzdt_iso[0])
        hybrid_all.append(dNdlogzdt_strong[2]+dNdlogzdt_weak[1]+dNdlogzdt_iso[1])
        deg5_all.append(dNdlogzdt_strong[3]+dNdlogzdt_weak[2]+dNdlogzdt_iso[2])

    ax[0].plot(lgzbins_Tr,np.mean(sling_Tr,axis=0),color=kick_colors['slingshot'],label="slingshot",linewidth=2)
    ax[0].plot(lgzbins_all,np.mean(rand_all,axis=0),color=kick_colors['random'],label="GW-random",linewidth=2)
    ax[0].plot(lgzbins_all,np.mean(hybrid_all,axis=0),color=kick_colors["hybrid"],label="GW-hybrid",linewidth=2)
    ax[0].plot(lgzbins_all,np.mean(deg5_all,axis=0),color=kick_colors["aligned"],label="GW-aligned",linewidth=2)
    
    lgz_bins,all_mr = np.loadtxt(savepath+"all_system_merger_rates.txt")    
    ax[0].plot(lgz_bins,all_mr,color="black",linestyle="--",label="Merger rate",linewidth=2)
    ax[0].set_ylim(2*10**-8,3*1e-1)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("$\log z$",fontsize=25)
    ax[0].set_ylabel(r"$\log (d^2 N / (d \log z dt)  \times 1\text{yr})$",fontsize=25)
    ax[0].legend(fontsize=20)
    ax[0].set_title("Triples+Binaries",fontsize=25)
    ax[0].set_yticks([1e-1,1e-3,1e-5,1e-7])

    fig.tight_layout()
    return fig,ax












            



            




                        



            


