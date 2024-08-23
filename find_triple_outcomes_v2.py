import numpy as np
import matplotlib.pyplot as plt
import scienceplots
#from scipy.interpolate import RegularGridInterpolator
import random
import pandas as pd
import interpolate as inter
from tables_merger_fraction import *
import BH_triple_kicks_data as triple_kick
import scipy.stats as st
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from tqdm import tqdm
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
from astropy import units as u
from astropy import constants as const

save_folder = '/Users/pranavsatheesh/Triples/Github/Triple-Outcomes/Data/'
savepath = '/Users/pranavsatheesh/Triples/Github/Illustris_Data/'
a_dadt_filename = savepath+"a_and_dadt_for_all_strong_triples.npz"
a_dadt_file =np.load(a_dadt_filename)
st_seps = a_dadt_file['st_seps']
st_dadt = a_dadt_file['st_dadt']

age_of_the_universe = cosmo.age(0).to(u.yr).value


def tinsp_interpolate(sep, dadt):
        '''
        Perform interpolation for finding t inspiral
        interpolates x=sep, y=1/dadt_total
        '''
        func = 1/dadt
        #remove infinities for interpolatio to avoid biased results
        func[func==np.inf] = np.nanmin(func)/10**10  #find a small value compared to minimum
        #perform interpolation
        print(func.shape)
        print(sep.shape)
        interp_func = interp1d(sep, func, bounds_error=False, fill_value=0) 
        return interp_func

def timsecale_to_merger(strong_triple_ix,a_triple_after):
        
        '''
        find tinsp using interplating sep, dadt and performing integration on the object function
        '''
        interp_func = tinsp_interpolate(st_seps[strong_triple_ix],st_dadt[strong_triple_ix])
        tau = integrate.quad(interp_func,a_triple_after,0)[0]

        return tau

def tau_prompt_merger_bonetti():
    '''Returns the merger time for Prompt merger from Bonetti'''

    tau_lognormal_mean = np.log(10)*8.4
    tau_lognormal_std = np.log(10)*0.4

    return st.lognorm.rvs(scale=np.exp(tau_lognormal_mean),s=tau_lognormal_std)

def tinsp_interpolate(sep, dadt):
        '''
        Perform interpolation for finding t inspiral
        interpolates x=sep, y=1/dadt_total
        '''
        func = 1/dadt
        #remove infinities for interpolatio to avoid biased results
        func[func==np.inf] = np.nanmin(func)/10**10  #find a small value compared to minimum
        #perform interpolation
        interp_func = interp1d(sep, func, bounds_error=False, fill_value=0) 
        return interp_func




def add_mbh_dynamics(triples_file,iterations):

    df_triples = pd.read_csv(triples_file+"strong_triples_data_from_ill.csv",index_col=False)
    N_triples = len(df_triples["M1"])
    df_run = []
    Merger_stat_counts = []

    for j in tqdm(range(iterations)):

        prompt_merger = 0
        merger_after_ejection = 0
        no_merger = 0

        total_possible_mergers = 0
        total_ejections = 0

        merger_flags = []
        
        M1_ill = df_triples["M1"].to_numpy()
        qin_ill = df_triples["qin"].to_numpy()
        qout_ill = df_triples["qout"].to_numpy()
        M2_ill = df_triples["M2"].to_numpy()
        M3_ill = df_triples["M3"].to_numpy()
        f_gas = df_triples["f-gas"].to_numpy()

        #need to add BH IDs

        # M2_ill = M1_ill*qin_ill
        # M3_ill = (M1_ill+M2_ill)*qout_ill

        #z_triple_ill = df_triples["z_form"].to_numpy()

        a_triple_ovtks_ill = df_triples["a_2nd_ovtks"].to_numpy()
        t_triple_form = df_triples["t_triple_form"].to_numpy()
        a_triple_interaction = []
        t_triple_merger_values = []
        z_triple_merger = []
        binary_m_flag = df_triples["bin_merger_flag"].to_numpy()

        #calculating the escape speed associated with each system.
        #I'm doing this for the intruder BH subhalo.
        #Mtot = df_triples["SubhaloMass"].to_numpy() #10**10 Msun units
        #sim_scale_factor = df_triples["sim_scale"].to_numpy() #scale factor
        #scale_radius = (df_triples["Subhalohalfradius"].to_numpy()/(1.1815)) * sim_scale_factor #in kpc units

        #Vesc = 293.28 * (Mtot**(1/2)) * (scale_radius**(-1/2)) # km/s
        Vesc_file = df_triples["Vescape"].to_numpy()

        slingshot_kicks = []
        gw_kick_random = []
        gw_kick_hybrid = []
        gw_kick_5deg = []


        for i in range(N_triples):


            strong_triple_index = i
            ahard = triple_kick.a_hard(M1_ill[i]* u.M_sun,qin_ill[i]) #hardening radius

            if(ahard<a_triple_ovtks_ill[i]):
                a_triple = ahard
            else:
                a_triple = a_triple_ovtks_ill[i]
            
            #a_triple = a_triple_ovtks_ill[i]
            
            mint = M3_ill[i]
            m1_bin = M1_ill[i]
            m2_bin = M2_ill[i]

            if(np.log10(qout_ill[i])<=0):
                a_P = trilinear_interp([np.log10(M1_ill[i]),np.log10(qout_ill[i]),np.log10(qin_ill[i])], m1, qout, qin, prompt_merger_frac12)*0.01
                b_P = trilinear_interp([np.log10(M1_ill[i]),np.log10(qout_ill[i]),np.log10(qin_ill[i])], m1, qout, qin, prompt_merger_frac13)*0.01
                c_P = trilinear_interp([np.log10(M1_ill[i]),np.log10(qout_ill[i]),np.log10(qin_ill[i])], m1, qout, qin, prompt_merger_frac23)*0.01

                d_P = (trilinear_interp([np.log10(M1_ill[i]),np.log10(qout_ill[i]),np.log10(qin_ill[i])], m1, qout, qin, delayed_merger_frac12)+
                     trilinear_interp([np.log10(M1_ill[i]),np.log10(qout_ill[i]),np.log10(qin_ill[i])], m1, qout, qin, delayed_merger_frac13)+
                     trilinear_interp([np.log10(M1_ill[i]),np.log10(qout_ill[i]),np.log10(qin_ill[i])], m1, qout, qin, delayed_merger_frac23))*0.01
            else:
                a_P = bilinear_interp([np.log10(qout_ill[i]),np.log10(qin_ill[i])], qout_bigp, qin, prompt_merger_frac_bigp12)*0.01
                b_P = bilinear_interp([np.log10(qout_ill[i]),np.log10(qin_ill[i])], qout_bigp, qin, prompt_merger_frac_bigp13)*0.01
                c_P = bilinear_interp([np.log10(qout_ill[i]),np.log10(qin_ill[i])], qout_bigp, qin, prompt_merger_frac_bigp23)*0.01

                d_P = (bilinear_interp([np.log10(qout_ill[i]),np.log10(qin_ill[i])], qout_bigp, qin, delayed_merger_frac_bigp12)+
                     bilinear_interp([np.log10(qout_ill[i]),np.log10(qin_ill[i])], qout_bigp, qin, delayed_merger_frac_bigp23)+
                     bilinear_interp([np.log10(qout_ill[i]),np.log10(qin_ill[i])], qout_bigp, qin, delayed_merger_frac_bigp13))*0.01

            P = random.uniform(0,1)

            if(P <= a_P+b_P+c_P):
                total_possible_mergers+=1    
                
                t_triple_merge = t_triple_form[i] + tau_prompt_merger_bonetti()
                t_triple_merger_values.append(t_triple_merge)
                a_triple_interaction.append(a_triple)

                if(t_triple_merge<=age_of_the_universe):
                    #prompt merger
                    prompt_merger = prompt_merger + 1
                    z_triple_merger.append(z_at_value(cosmo.age,(t_triple_merge/10**9)*u.Gyr,zmin=1e-9).value)

                    if(P <= a_P):

                        merger_flags.append("Tr-12")
                        vsling,a_new = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,1)
                        slingshot_kicks.append(vsling)

                        rand,hyb,deg5 = triple_kick.gw_kick_calc(qin_ill[i],f_gas[i])
                        gw_kick_random.append(rand)
                        gw_kick_5deg.append(deg5)
                        gw_kick_hybrid.append(hyb)
            
                    elif(P > a_P and P <= a_P+b_P):

                        merger_flags.append("Tr-13")

                        if(mint>2*(m1_bin+m2_bin)):
                            vsling,a_new = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,3)
                        else:
                            vsling,a_new = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,2)

                        slingshot_kicks.append(vsling)

                        m1_new = max(M1_ill[i],M3_ill[i])
                        m2_new = min(M1_ill[i],M3_ill[i])
                        qin_new = m2_new/m1_new
                        rand,hyb,deg5 = triple_kick.gw_kick_calc(qin_ill[i],f_gas[i])
                        gw_kick_random.append(rand)
                        gw_kick_5deg.append(deg5)
                        gw_kick_hybrid.append(hyb)
                    
                    elif(P > a_P+b_P and P <= a_P+b_P+c_P):
                        merger_flags.append("Tr-23")
                        
                        if(mint>2*(m1_bin+m2_bin)):
                            vsling,a_new = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,13)
                        else:
                            vsling,a_new = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,12)
                        
                        slingshot_kicks.append(vsling)

                        m1_new = max(M2_ill[i],M3_ill[i])
                        m2_new = min(M2_ill[i],M3_ill[i])
                        qin_new = m2_new/m1_new
                        rand,hyb,deg5 = triple_kick.gw_kick_calc(qin_ill[i],f_gas[i])
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
                total_ejections +=1
                #it is an ejection. The lightest is ejected
                # m_sort = np.sort([M1_ill[i],M2_ill[i],M3_ill[i]])

                if(mint<m2_bin):
                    #m3 is scattered off
                    vsling,a_triple_after = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,1)
                    slingshot_kicks.append(vsling)

                    #merger time
                    tau_merger = timsecale_to_merger(strong_triple_index,a_triple_after)
                    t_triple_merge = t_triple_form[i]+tau_merger
                    t_triple_merger_values.append(t_triple_merge)
                    a_triple_interaction.append(a_triple_after)
                    

                    if(t_triple_merge<=age_of_the_universe):

                        #merger after an ejection
                        z_triple_merger.append(z_at_value(cosmo.age,(t_triple_merge/10**9)*u.Gyr,zmin=1e-9).value)
                        merger_flags.append("Tr-ej")
                        merger_after_ejection = merger_after_ejection + 1
                        #m1 and m2 are merging
                        rand,hyb,deg5 = triple_kick.gw_kick_calc(qin_ill[i],f_gas[i])
                        gw_kick_random.append(rand)
                        gw_kick_5deg.append(deg5)
                        gw_kick_hybrid.append(hyb) 
                    else:
                        #no merger
                        z_triple_merger.append(0)
                        merger_flags.append("No")       
                        no_merger = no_merger+1
                        gw_kick_random.append(0)
                        gw_kick_5deg.append(0)
                        gw_kick_hybrid.append(0)
                
                elif((mint>m2_bin) and (mint<2*(m1_bin+m2_bin))):
                    #m2 is ejected after an exchange here.
                    vsling,a_triple_after = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,2)
                    
                    slingshot_kicks.append(vsling)
                    a_triple_interaction.append(a_triple_after)
                    #merger time
                    tau_merger = timsecale_to_merger(strong_triple_index,a_triple_after)
                    t_triple_merge = t_triple_form[i]+tau_merger
                    t_triple_merger_values.append(t_triple_merge)

                    if(t_triple_merge<=age_of_the_universe):
                        
                        #merger after an ejection 
                        z_triple_merger.append(z_at_value(cosmo.age,(t_triple_merge/10**9)*u.Gyr,zmin=1e-9).value)
                        merger_flags.append("Tr-ej")
                        merger_after_ejection = merger_after_ejection + 1
                        #m1 and m3 are merging
                        m1_new = max(M1_ill[i],M3_ill[i])
                        m2_new = min(M1_ill[i],M3_ill[i])
                        qin_new = m2_new/m1_new
                        rand,hyb,deg5 = triple_kick.gw_kick_calc(qin_ill[i],f_gas[i])
                        gw_kick_random.append(rand)
                        gw_kick_5deg.append(deg5)
                        gw_kick_hybrid.append(hyb)
                    
                    else:

                        #no-merger
                        z_triple_merger.append(0)
                        merger_flags.append("No")
                        no_merger = no_merger + 1
                        gw_kick_random.append(0)
                        gw_kick_5deg.append(0)
                        gw_kick_hybrid.append(0)

                elif(mint>=2*(m1_bin+m2_bin)):
                        #m2 is ejected after an exchange
                        vsling,a_triple_after = triple_kick.v_and_a_after_slingshot(m1_bin,m2_bin,mint,a_triple,3)
                        slingshot_kicks.append(vsling)
                        a_triple_interaction.append(a_triple_after)

                        #merger time
                        tau_merger = timsecale_to_merger(strong_triple_index,a_triple_after)
                        t_triple_merge = t_triple_form[i]+tau_merger
                        t_triple_merger_values.append(t_triple_merge)

                        if(t_triple_merge<=age_of_the_universe):
                        
                            #merger after an ejection 
                            z_triple_merger.append(z_at_value(cosmo.age,(t_triple_merge/10**9)*u.Gyr,zmin=1e-9).value)
                            merger_flags.append("Tr-ej")
                            merger_after_ejection = merger_after_ejection + 1
                            #m1 and m3 are merging
                            m1_new = max(M1_ill[i],M3_ill[i])
                            m2_new = min(M1_ill[i],M3_ill[i])
                            qin_new = m2_new/m1_new
                            rand,hyb,deg5 = triple_kick.gw_kick_calc(qin_ill[i],f_gas[i])
                            gw_kick_random.append(rand)
                            gw_kick_5deg.append(deg5)
                            gw_kick_hybrid.append(hyb)
                    
                        else:

                            #no-merger
                            z_triple_merger.append(0)
                            merger_flags.append("No")
                            no_merger = no_merger + 1
                            gw_kick_random.append(0)
                            gw_kick_5deg.append(0)
                            gw_kick_hybrid.append(0)

        z_triple_merger = np.array(z_triple_merger)
        a_triple_interaction = np.array(a_triple_interaction)
        t_triple_merger_values = np.array(t_triple_merger_values)

        #'Slingshot_kick':slingshot_kicks,'gw_kick_random':gw_kick_random,'gw_kick_cold':gw_kick_cold,'gw_kick_5deg':gw_kick_5deg}

        df_i = pd.read_csv(triples_file+"strong_triples_data_from_ill.csv",index_col=False)
        df_i.insert(5,"t_merger",t_triple_merger_values)
        df_i.insert(6,"z_merger",z_triple_merger)
        df_i.insert(7,"a_triple_int",a_triple_interaction)
        df_i.insert(8,"merger_flag",merger_flags)
        df_i.insert(9,"Slingshot_kick",slingshot_kicks)
        df_i.insert(10,"gw_kick_random",gw_kick_random)
        df_i.insert(11,"gw_kick_hybrid",gw_kick_hybrid)
        df_i.insert(12,"gw_kick_5deg",gw_kick_5deg)


        #column_name = ['M1','M2','M3','qin','qout','t_merger','z_merger','t_form','a_triple_int','merger_flag','Slingshot_kick','gw_kick_random','gw_kick_hybrid','gw_kick_5deg','binary_merger_flag','Vescape']   
        #df_i = pd.DataFrame({'M1': M1_ill,'M2': M2_ill,'M3': M3_ill,'qin': qin_ill, 'qout': qout_ill,'t_merger':t_triple_merger_values,'z_merger' : z_triple_merger,'t_form':t_triple_form,'a_triple_int':a_triple_interaction,'merger_flag':merger_flags,'Slingshot_kick':slingshot_kicks,'gw_kick_random':gw_kick_random,'gw_kick_hybrid':gw_kick_hybrid,'gw_kick_5deg':gw_kick_5deg,'binary_merger_flag':binary_m_flag,'Vescape':Vesc_file},columns=column_name)
        df_run.append(df_i)
        Merger_stat_counts.append([total_possible_mergers,total_ejections,prompt_merger,merger_after_ejection,no_merger])

    df_pd = pd.concat(df_run)
    Outcome_data_file_name = save_folder+ "Triple_outcomes_N_"+str(iterations)+".csv"
    df_pd.to_csv(Outcome_data_file_name,index = False)  
    print("File saved at"+save_folder+"Triple_outcomes_N_"+str(iterations)+".csv")

    return Merger_stat_counts






