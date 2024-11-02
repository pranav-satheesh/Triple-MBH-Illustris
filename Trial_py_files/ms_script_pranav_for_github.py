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
    

    def strong_trip_data(self,save_path,strong_trip_flag = False):

        '''To extract information about the strong triples from Illustris'''

        _,intruder_mass,inner_binary_mass_in_next_merger = self.find_first_binary_in_next_merger()
    
        delta_m = inner_binary_mass_in_next_merger - self.inner_binary_initial_mtotal
        print("faulty accretion %d"%(len(np.argwhere(delta_m<0))))
        #dm_mask = np.argwhere(delta_m>0)
        #sim_unit_h_scale=0.704

        #f-gas,Vesc and BHIDs
        fgas,Vescape = self.host_galaxy_properties()
        fgas = fgas[self.triple_mask]
        Vescape = Vescape[self.triple_mask]
        inner_binary_ids = self.merger_ids[self._next.idx_inner_binary]
        outer_binary_ids = self.merger_ids[self._next.idx_outer_binary]       

        masses_inner_binary = self.masses[self.triple_mask]
        masses_inner_binary = np.sort(masses_inner_binary)


        # Mhalo1 = self.envs_SubhaloMass[self.triple_mask]
        # Mhalo2 = self.envs_in_SubhaloMass[self.triple_mask]
        # Mhalo_masses = Mhalo1 + Mhalo2
        # #subhalo_masses = (Mhalo1 + Mhalo2)/sim_unit_h_scale
        # Mgalaxy = Mhalo_masses/sim_unit_h_scale*1e10*u.Msun

        # sim_scale_factor0 = self.scales[self.triple_mask][:,0]
        # #subhalo_halfmassradius = self.envs_SubhaloHalfmassRad[self.triple_mask]/sim_unit_h_scale
        # subhalo_halfmassradius = (self.envs_SubhaloHalfmassRadType[:,4]/sim_unit_h_scale)[self.triple_mask]*sim_scale_factor0*u.kpc
        # scale_radius_h = subhalo_halfmassradius * (1/1.815)
        #Vescape_values = 293.28884 * ((Mgalaxy/(1e10 * u.Msun))**(1/2) * (scale_radius_h/(1*u.kpc))**(-1/2))
        #Vescape_values = np.sqrt(2*const.G*Mgalaxy/scale_radius_h).to(u.km * u.s**(-1))
        #Vescape_values = self.Vesc_calc(subhalo_masses,subhalo_halfmassradius,sim_scale_factor0)

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

        age_of_the_universe = cosmo.age(0).to(u.yr).value
        t_evol_binary = self._next.evol_tmrg[self.triple_mask]

                
        if(strong_trip_flag == True):
            strong_in_triple_mask = self.q_inner_outer_strong()

            inner_binary_ids_st = inner_binary_ids[strong_in_triple_mask]
            outer_binary_ids_st = outer_binary_ids[strong_in_triple_mask]

            fgas_strong_trip = fgas[strong_in_triple_mask]
            Vescape_strong_trip = Vescape[strong_in_triple_mask]

            M1_trip = M1_trip[strong_in_triple_mask]
            M2_trip = M2_trip[strong_in_triple_mask]
            M3_trip = M3_trip[strong_in_triple_mask]
            qbh_inner = qbh_inner[strong_in_triple_mask]
            qbh_outer = qbh_outer[strong_in_triple_mask]
            t_triple = t_triple[strong_in_triple_mask]
            a_triple = a_triple[strong_in_triple_mask]
            z_triple = z_at_value(cosmo.age,(t_triple/10**9)*u.Gyr,zmin=1e-10).value

            self.save_strong_triples_a_and_dadt(save_path)

            t_evol_binary = t_evol_binary[strong_in_triple_mask]
            binary_merged_before_z0_flag = []
            for time in t_evol_binary:
            
                if time >= age_of_the_universe:
                    binary_merged_before_z0_flag.append("No")
                else:
                    binary_merged_before_z0_flag.append("Yes")
            

            df = pd.DataFrame([M1_trip,M2_trip,M3_trip,qbh_inner,qbh_outer,t_triple,z_triple,a_triple,binary_merged_before_z0_flag,inner_binary_ids_st[:,0],inner_binary_ids_st[:,1],outer_binary_ids_st[:,0],outer_binary_ids_st[:,1],fgas_strong_trip,Vescape_strong_trip])
            df = df.transpose()
            df.columns = ['M1','M2','M3','qin','qout','t_triple_form','z_form','a_2nd_ovtks','bin_merger_flag','bhid1','bhid2','bhid3','bhid4','f-gas','Vescape']
            save_file = save_path + 'strong_triples_data_from_ill.csv'
        
        else:
            z_merger = []
            age_of_the_universe = cosmo.age(0).to(u.yr).value
            strong_in_triple_mask = self.q_inner_outer_strong()
            t_evol_binary = self._next.evol_tmrg[self.triple_mask]
            binary_merged_before_z0_flag = []

            for time in t_evol_binary:
                if time >= age_of_the_universe:
                    z_merger.append(0)
                    binary_merged_before_z0_flag.append("No")
                else:
                    z_merger.append(z_at_value(cosmo.age,(time/10**9)*u.Gyr,zmin=1e-9).value)
                    binary_merged_before_z0_flag.append("Yes")

            df = pd.DataFrame([M1_trip,M2_trip,M3_trip,qbh_inner,qbh_outer,t_triple,z_merger,a_triple,binary_merged_before_z0_flag,strong_in_triple_mask,inner_binary_ids[:,0],inner_binary_ids[:,1],outer_binary_ids[:,0],outer_binary_ids[:,1],fgas,Vescape])
            df = df.transpose()
            df.columns = ['M1','M2','M3','qin','qout','t_triple_form','z_merger','a_2nd_ovtks','bin_merger_flag','strong_key','bhid1','bhid2','bhid3','bhid4','f-gas','Vescape']
            save_file = save_path + 'all_triples_data_from_ill.csv'

        df.to_csv(save_file,index=False)
        print("File saved at",save_file)

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

    def isolated_binaries_data(self,save_file):


        '''This function is for extracting all the info about isolated binaries and store them in
        a csv file'''
        masses_binary = self.masses[self.binary_mask]
        masses_binary = np.sort(masses_binary)
        M1_binary = masses_binary[:,1] #bigger one is primary
        M2_binary = masses_binary[:,0]
        qbh_inner = M2_binary/M1_binary


        t_evol_binary = self._next.evol_tmrg[self.binary_mask]
        #evolution time of binary in inspiral


        #gas-fraction, Escape speed and BHIDs
        fgas,Vescape = self.host_galaxy_properties()
        fgas = fgas[self.binary_mask]
        Vescape = Vescape[self.binary_mask]
        binary_ids = self.merger_ids[self.binary_mask]

        age_of_the_universe = cosmo.age(0).to(u.yr).value
        merged_before_z0_flag = []
        z_binary = []
    
        
        # Mhalo1 = self.envs_SubhaloMass[self.binary_mask]
        # Mhalo2 = self.envs_in_SubhaloMass[self.binary_mask]
        # Mhalo_masses = Mhalo1 + Mhalo2
        # Mgalaxy = Mhalo_masses/sim_unit_h_scale*1e10*u.Msun
        # sim_scale_factor0 = self.scales[self.binary_mask][:,0]
    
        # subhalo_halfmassradius = (self.envs_SubhaloHalfmassRadType[:,4]/sim_unit_h_scale)[self.binary_mask]*sim_scale_factor0*u.kpc
        # scale_radius_h = subhalo_halfmassradius * (1/1.815)
        
        # Vescape_values = 293.28884 * ((Mgalaxy/(1e10 * u.Msun))**(1/2) * (scale_radius_h/(1*u.kpc))**(-1/2))

        for time in t_evol_binary:
            
            if time >= age_of_the_universe:
                merged_before_z0_flag.append("No")
                z_binary.append(0)
            else:
                merged_before_z0_flag.append("Yes")
                z_binary.append(z_at_value(cosmo.age,(time/10**9)*u.Gyr,zmin=1e-13).value)

        #z_binary = z_at_value(cosmo.age,(t_evol_binary/10**9)*u.Gyr,zmin=1e-10)

        df = pd.DataFrame([M1_binary,M2_binary,qbh_inner,t_evol_binary,z_binary,merged_before_z0_flag,binary_ids[:,0],binary_ids[:,1],fgas,Vescape])
        df = df.transpose()
        df.columns = ['M1','M2','qin','t_merger','z_merger','merger_flag','bhid1','bhid2','f-gas','Vescape']
        df.to_csv(save_file+'iso_binaries_data_from_ill.csv',index=False)
        print("File saved at",save_file+"iso_binaries_data_from_ill.csv")

        return None



    #plot 2d histogram with the mass vs redshift
    def hist2d(self, x,y, host=False, save_to_icloud=False, fname=None, ax_kwargs={}, ax_tickkwargs={}, scttr_kwargs={}):
        import scipy as sp
        import matplotlib as mpl

        if not host:
            xedges =  np.logspace(*np.log10([10**-5,10**2.5]), 21) #logspace(aa, 21)
            yedges =  np.logspace(*np.log10([10**-5,10**2.5]), 21)#logspace(bb, 21)
            histcolor = 'Reds'
        elif host: 
            xedges =  np.logspace(*np.log10([10**-8,10**8]), 21) #logspace(aa, 21)
            yedges =  np.logspace(*np.log10([10**-8,10**8]), 21)#logspace(bb, 21)
            histcolor = 'Blues'

        aa, bb = x, y

        x_diag = np.logspace(np.log10(aa.min()),0,10)
        y_diag = x_diag

        self.plotting_latex_params()
        hist, *_ = sp.stats.binned_statistic_2d(aa, bb, None, bins=(xedges, yedges), statistic='count')
        norm = mpl.colors.LogNorm(vmin=hist[hist>0].min(), vmax=hist.max())

        ax = plt.gca()
        xx, yy = np.meshgrid(xedges, yedges)
        print ('x vals max:{}, min:{}'.format(np.max(aa),np.min(aa)))
        print ('y vals max:{}, min:{}'.format(np.max(bb),np.min(bb)))
        pcm = ax.pcolormesh(xedges, yedges, hist.T, cmap=histcolor, norm=norm)
        ax.scatter(aa, bb, s=2, alpha=0.2, color='cyan', **scttr_kwargs)
        ax.plot(x_diag, y_diag, '--', color='white')
        plt.colorbar(pcm, ax=ax, label='number')
        ax.set(**ax_kwargs)
        ax.tick_params(**ax_tickkwargs)
        label_font_size = 20
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_font_size)
        self.ax_setup(ax)
        plt.tight_layout()
        if save_to_icloud:
            if fname is None:
                fname = "qhist_2d"
            fname_path = os.path.join(icloud_path, project_path, plots_path, fname + ".pdf")
            print (f'saved in {fname_path}')
            plt.savefig(fname_path, format='pdf')
            print (f'file saved to: {fname_path}')
        plt.show()
        return aa, bb, hist


    def bh_qhist_2d(self, strong_triple=False):
        _, intruder_mass,  inner_binary_mass_in_next_merger = self.find_first_binary_in_next_merger()
        qbh_outer = intruder_mass/inner_binary_mass_in_next_merger
        print(len(qbh_outer>1))
        masses_inner_binary = self.masses[self.triple_mask]
        qbh_inner = masses_inner_binary[:,0]/masses_inner_binary[:,1]
        qbh_inner[qbh_inner>1] = 1/qbh_inner[qbh_inner>1]
        print (f'the maximum of q_outer is: {max(qbh_outer)}')
        if strong_triple:
            strong_in_triple_mask = self.q_inner_outer_strong()
            qbh_outer = qbh_outer[strong_in_triple_mask]
            qbh_inner = qbh_inner[strong_in_triple_mask]
            
        fname = "bh_qhist_2d"
        label_size = 20
        tick_params_dict = {
            'axis': 'both',
            'which': 'major',
            'labelsize': label_size
        }        
        self.hist2d(qbh_inner, qbh_outer, host=False, save_to_icloud=True, fname=fname,
                    ax_kwargs={'xlabel':'${\\rm q_{inner}}$', 'ylabel':'${\\rm q_{outer}}$'}, ax_tickkwargs=tick_params_dict)
        
         
    def find_tinsp(self, sep, dadt_total, interpolated=False):
        '''
        find tinsp using two methods
        1- summation of terms (dadt)^-1dt
        2- interplating sep, dadt and performing integration on the object function
        '''
        import scipy.integrate as integrate
        tau = -np.ones(len(sep))        
        if not interpolated:

            da = np.diff(sep)
            dadt_total = dadt_total[:-1]
            dts = da/dadt_total
            for i in range(len(sep)):
                #the following deals with the 
                #array size problem
                tau[i]=np.nansum(dts[i:])            
        else:
            interp_func = self.tinsp_interpolate(sep, dadt_total)
            for i in range(len(sep)):
                tau[i] = integrate.quad(interp_func, sep[i],0 )[0]
        return tau
        
        
    def tinsp_interpolate(self, sep, dadt):
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

    
    def find_tau_merger(self,sep_start,filepath):

        filename = filepath+"a_and_dadt_for_all_strong_triples.npz"
        
        
        if sep is None and dadt is None:
            sep = self.sep
            dadt = self.dadt_df+self.dadt_lc+self.dadt_vd+self.dadt_gw
            tinsp = np.array([None]*sep.shape[0])
            assert len(tinsp.shape)<2
            for i in tqdm(range(tinsp.shape[0]), desc = 'Calculating inspiral time for all binaries all separations'):
                tinsp[i] = self.find_tinsp(sep[i], dadt[i], interpolated) 
            tinsps = np.stack(tinsp)
        return tinsps           


    
    def time_scales (self, sep=None, dadt=None, interpolated=False):
        '''
        Caculates ((da/dt)^-1)*a, where a is the separation
        by default it calculates for all binaries and and all
        separations using the total hardening rate
        ------- Finish this later---------
        '''
        filename = "tinsp_for_all_bin.npz"
        if os.path.exists(filename):
            print (f"{filename} exists! Loading the file")
            with np.load(filename) as data:
                tinsps = data['tinsps']
        else:
            print (f"{filename} does not exists! Calculating inspiral time")
            if sep is None and dadt is None:
                sep = self.sep
                dadt = self.dadt_df+self.dadt_lc+self.dadt_vd+self.dadt_gw
            tinsp = np.array([None]*sep.shape[0])        
            assert len(tinsp.shape)<2
            for i in tqdm(range(tinsp.shape[0]), desc = 'Calculating inspiral time for all binaries all separations'):
                tinsp[i] = self.find_tinsp(sep[i], dadt[i], interpolated)
            tinsps = np.stack(tinsp)
            np.savez(filename, tinsps=tinsps)
        return tinsps
        
        
    def find_time_scales (self, sep=None, dadt=None, interpolated=False):

        if sep is None and dadt is None:
            sep = self.sep
            dadt = self.dadt_df+self.dadt_lc+self.dadt_vd+self.dadt_gw
        tinsp = np.array([None]*sep.shape[0])        
        assert len(tinsp.shape)<2
        for i in tqdm(range(tinsp.shape[0]), desc = 'Calculating inspiral time for all binaries all separations'):
            tinsp[i] = self.find_tinsp(sep[i], dadt[i], interpolated)
        tinsps = np.stack(tinsp)
        #np.savez(filename, tinsps=tinsps)
        return tinsps
   
