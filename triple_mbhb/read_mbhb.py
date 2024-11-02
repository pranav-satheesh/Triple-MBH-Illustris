import numpy as np
import matplotlib.pyplot as plt
import h5py, sys, os
import astro_constants as ac
from astropy.cosmology import FlatLambdaCDM

import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/Users/sayebms1/mybitbucket/lb_illustris_python')
sys.path.insert(1, '/Users/pranavsatheesh/Triples/bitbucket/lb_illustris_python')
import init
#fname = 'mbhb-evolution_fiducial_ecc-evo-0.6_lc-shm06.hdf5'
#fname = 'mbhb-evolution_mdot_ecc-evo-0.00_lc-shm06.hdf5'
#fname = 'mbhb-evolution_mdot_ecc-evo-0.9_lc-shm06_0.01mdot.hdf5'
#fname = 'mbhb-evolution__no-ecc_lc-full-0.0.hdf5'
#fname = 'mbhb-evolution__no-ecc_lc-full-0.2.hdf5'
#fname = 'mbhb-evolution__no-ecc_lc-full-0.6.hdf5'
#fname = 'mbhb-evolution__no-ecc_lc-full-0.99.hdf5'
  
class mbhb_data:
    """
    Read in and parse the mbhb data from the evolution files
    (passed to the tripleMatches class in the find_triple_mbhs code --
    if you just want to use the find_triple_mbhs script, this class is the
    only part of this code that gets used)
    """

    def __init__(self, path, fmbhb, parse_env=False, **kwargs):

        self.mbhb_file_list = dict(fid_e06='mbhb-evolution_fiducial_ecc-evo-0.6_lc-shm06.hdf5',
                                   new_fid_e06='new_fiducial_mbhb-evolution_with_env_in_ecc-evo-0.6_lc-shm06.hdf5', # this is what we used for our fiducial model with env parameters
                                   fid_e00='mbhb-evolution_mdot_ecc-evo-0.00_lc-shm06.hdf5',
                                   fid_e09_mdot001='mbhb-evolution_mdot_ecc-evo-0.9_lc-shm06_0.01mdot.hdf5',
                                   frefill00='mbhb-evolution_no-ecc_lc-full-0.00.hdf5',
                                   frefill02='mbhb-evolution_no-ecc_lc-full-0.2.hdf5',
                                   frefill06='mbhb-evolution_no-ecc_lc-full-0.6.hdf5',
                                   frefill099='mbhb-evolution_no-ecc_lc-full-0.99.hdf5',
                                   eccen01='new_mbhb-evolution_ecc-evo-0.1_lc-shm06.hdf5',
                                   eccen05='new_mbhb-evolution_ecc-evo-0.5_lc-shm06.hdf5',
                                   eccen09='new_mbhb-evolution_ecc-evo-0.9_lc-shm06.hdf5',
                                   eccen099='new_mbhb-evolution_ecc-evo-0.99_lc-shm06.hdf5'
                                   )                                  
        
        if fmbhb in self.mbhb_file_list.keys():
            self.fpath_mbhb = path + self.mbhb_file_list[fmbhb]
        elif fmbhb in self.mbhb_file_list.values():
            self.fpath_mbhb = path + fmbhb
        else:
            print("Error: invalid mbhb file specified: {}", fmbhb)
            print("Availble files are:")
            print(self.mbhb_file_list)
            sys.stdout.flush()
            sys.exit()
        assert os.path.exists(self.fpath_mbhb), \
            "Error: file {} not found.".format(self.fpath_mbhb)


        self.cosmo_run = kwargs.get('cosmo_run', 'L75n1820FP')
        verbose = kwargs.get('verbose', False)

        #self.pars = init.params(self.cosmo_run)
        self.omega_m = 0.3089
        self.h = 0.6774
        self.cosmo = FlatLambdaCDM(Om0=self.omega_m, H0=100*self.h)
        self.tH = self.cosmo.age(0).value * 1e9 
        print("h = {}, tH = {} yr.\n".format(self.h, self.tH))

        print("Reading data from "+self.fpath_mbhb)
            
        with h5py.File(self.fpath_mbhb, 'r') as f:
            if verbose:
                print("Overview of file structure:")
                for item in f.attrs.keys():
                    print(item, f.attrs[item])
                for grp in f.keys():
                    print("grp: ",grp)
                    print(f[grp].name)
                    for a in f[grp].attrs.keys():
                        print(a)
                    for k in f[grp].keys():
                        print("key: ", k)
                        for a in f[grp][k].attrs:
                            print(a,": ",f[grp][k].attrs[a])

            self.meta = f['meta']
            self.subhalo_mass_type = np.array(self.meta['SubhaloMassInHalfRadType']) / ac.MSUN
            ## already in msun / yr:
            self.subhalo_sfr = np.array(self.meta['SubhaloSFRinHalfRad']) 
            print("subhalo SFR:")
            print(self.subhalo_sfr.min(), self.subhalo_sfr.max(), 
                  np.median(np.log10(self.subhalo_sfr[self.subhalo_sfr>0])))
            self.snapshot = np.array(self.meta['snapshot'])
            self.subhalo_id = np.array(self.meta['subhalo_id'])
            print("snapshot: ", self.snapshot.shape, self.snapshot.min(), 
                  self.snapshot.max(), self.snapshot.size)
                
            self.evol = f['evolution']
            self.dadt = np.array(self.evol['dadt']) / ac.PC * ac.YR
            print("dadt.size: ", self.dadt.size, self.dadt.shape)

            self.val_inds = np.array(self.evol['val_inds'])
            self.dadt_df = np.array(self.evol['dadt_df']) / ac.PC * ac.YR
            self.dadt_lc = np.array(self.evol['dadt_lc']) / ac.PC * ac.YR
            self.dadt_vd = np.array(self.evol['dadt_vd']) / ac.PC * ac.YR
            self.dadt_gw = np.array(self.evol['dadt_gw']) / ac.PC * ac.YR

            self.masses = np.array(self.evol['masses']) / ac.MSUN
            print("masses: ", self.masses.shape)
            self.times = np.array(self.evol['times']) / ac.YR
            print("times: ", self.times.shape)
            self.sep = np.array(self.evol['sep']) / ac.PC
            self.mdot_eff = np.array(self.evol['mdot_eff']) / ac.MSUN * ac.YR
            self.scales = np.array(self.evol['scales'])
            print("scales: ", self.scales.shape)
            
            self.tinsp = np.abs(self.sep/self.dadt)
            self.tinsp_df = np.abs(self.sep/self.dadt_df)
            self.tinsp_lc = np.abs(self.sep/self.dadt_lc)
            self.tinsp_vd = np.abs(self.sep/self.dadt_vd)
            self.tinsp_gw = np.abs(self.sep/self.dadt_gw)
    
            self.Nmbhb = self.snapshot.size
            self.Nrad = self.sep.shape[1]
            print("\nFile contains {} binaries.".format(self.Nmbhb))

            self.sim_tmrg = self.times[:,0]
            self.sim_mrg_sep = self.sep[:,0]
            self.ixsep_evol_mrg = -np.ones(self.Nmbhb).astype('int')
            self.evol_tmrg = -np.ones(self.Nmbhb)
            for i in range(self.Nmbhb):
                self.ixsep_evol_mrg[i] = np.array(np.where(self.sep[i,:]==0)[0]).min() 
                assert self.ixsep_evol_mrg[i].size==1, \
                    "more than one value returned for ixsep_evol_mrg[{}]".format(i)
                self.evol_tmrg[i] = self.times[i, self.ixsep_evol_mrg[i]]
            self.evol_mrg_mask = (self.evol_tmrg<=self.tH)
            self.evol_z = (1/self.scales)-1
            ##M.S added define total inspiral time also##
            self.total_tinsp = self.evol_tmrg - self.sim_tmrg
            ####
            self.q = self.masses[:,0] / self.masses[:,1]
            self.q[self.q>1.0] = 1.0 / self.q[self.q>1.0]
            
        if parse_env==True and 'new_fiducial' in self.fpath_mbhb:
            self.parse_envs(self.fpath_mbhb)        
        
            
    def parse_envs(self, file_path):
        with h5py.File(self.fpath_mbhb, 'r') as f:
            self.envs = f['envs']
            self.envs_in = f['envs_in']
            for keys, vals in f['envs'].items():
                try:
                    setattr(self, 'envs_'+keys, vals[()][self.val_inds])
                except:
                    setattr(self, 'envs_'+keys, vals[()])                    
            for keys, vals in f['envs_in'].items():
                try:
                    setattr(self, 'envs_in_'+keys, vals[()][self.val_inds])
                except:
                    setattr(self, 'envs_in_'+keys, vals[()])                    
        
#['disp', 'SubhaloBHMdot', 'version', 'SubhaloWindMass', 'SubhaloVelDisp', 'SubhaloSFR', 'SubhaloLen', 'names', 'run', 'SubhaloStellarPhotometrics', 'SubhaloGasMetallicity', 'SubhaloBHMass', 'SubhaloIDMostbound', 'SubhaloVmax', 'SubhaloHalfmassRad', 'SubhaloParent', 'SubhaloSpin', 'pots', 'SubhaloPos', 'cat_keys', 'rads', 'snap', 'SubhaloVel', 'SubhaloLenType', 'status', 'SubhaloMassInRad', 'SubhaloGrNr', 'SubhaloMassInHalfRad', 'SubhaloSFRinRad', 'SubhaloMassInMaxRad', 'SubhaloHalfmassRadType', 'subhalo', 'SubhaloMassInMaxRadType', 'SubhaloCM', 'SubhaloMassType', 'types', 'SubhaloMassInHalfRadType', 'center', 'nums', 'created', 'dens', 'SubhaloSFRinHalfRad', 'SubhaloMass', 'boundid', 'SubhaloStarMetallicity', 'mass', 'SubhaloMassInRadType', 'SubhaloVmaxRad', 'SubhaloStellarPhotometricsRad', 'SubhaloSFRinMaxRad'] 


def define_mbhb_inspiral_phases(d, verbose=False):

    mrg_time_idx = -np.ones(d.Nmbhb).astype('int')
    mrg_time = np.zeros(d.Nmbhb) * np.nan
    finalsep = np.zeros(d.Nmbhb) * np.nan
    tinsp_phase_min = np.zeros((d.Nmbhb, d.Nrad))
    phase = np.zeros((d.Nmbhb, d.Nrad)) * np.nan
    r_phase = np.zeros((d.Nmbhb, 4))* np.nan
    time_r_phase = np.zeros((d.Nmbhb, 4))* np.nan

    print("Nmbhb=",d.Nmbhb)
    for i in range(d.Nmbhb):
        mrg_time_idx[i] = np.array(np.where(d.sep[i,:]==0)[0]).min()
        mrg_time[i] = d.times[i, mrg_time_idx[i]]
        finalsep[i] = d.sep[i,mrg_time_idx[i]]
 
        #tinsp_df = d.tinsp_df[i,:]
        #tinsp_lc = d.tinsp_lc[i,:]
        #tinsp_vd = d.tinsp_vd[i,:]
        #tinsp_gw = d.tinsp_gw[i,:]
        tinsp_phase_min[i,:] = np.nanmin(np.array([d.tinsp_df[i,:], 
                                                   d.tinsp_lc[i,:],
                                                   d.tinsp_vd[i,:], 
                                                   d.tinsp_gw[i,:]]), axis=0)
        phase[i, (d.tinsp_df[i,:]==tinsp_phase_min[i,:])] = 0
        phase[i, (d.tinsp_lc[i,:]==tinsp_phase_min[i,:])] = 1
        phase[i, (d.tinsp_vd[i,:]==tinsp_phase_min[i,:])] = 2
        phase[i, (d.tinsp_gw[i,:]==tinsp_phase_min[i,:])] = 3

        for k in range(4):
            if d.sep[i,phase[i,:]==k].size > 0:
                r_phase[i,k] = d.sep[i,phase[i,:]==k][0]
                time_r_phase[i,k] = d.times[i,phase[i,:]==k][0]


    if verbose:
        print(tinsp_phase_min[tinsp_phase_min==d.tinsp_df].size)
        print(tinsp_phase_min[tinsp_phase_min==d.tinsp_lc].size)
        print(tinsp_phase_min[tinsp_phase_min==d.tinsp_vd].size)
        print(tinsp_phase_min[tinsp_phase_min==d.tinsp_gw].size)

        print("test:")
        print(tinsp_phase_min[(tinsp_phase_min==d.tinsp_df)
                              &(tinsp_phase_min==d.tinsp_lc)].size)
        print(tinsp_phase_min[(tinsp_phase_min==d.tinsp_df)
                              &(tinsp_phase_min==d.tinsp_vd)].size)
        print(tinsp_phase_min[(tinsp_phase_min==d.tinsp_df)
                              &(tinsp_phase_min==d.tinsp_gw)].size)
        print(tinsp_phase_min[(tinsp_phase_min==d.tinsp_lc)
                              &(tinsp_phase_min==d.tinsp_vd)].size)
        print(tinsp_phase_min[(tinsp_phase_min==d.tinsp_lc)
                              &(tinsp_phase_min==d.tinsp_gw)].size)
        print(tinsp_phase_min[(tinsp_phase_min==d.tinsp_vd)
                              &(tinsp_phase_min==d.tinsp_gw)].size)

        print("phase:")
        print(phase[phase==0].size)
        print(phase[phase==1].size)
        print(phase[phase==2].size)
        print(phase[phase==3].size)
        print(phase[phase!=phase].size)

        print("totals:")
        print(phase[phase==0].size + phase[phase==1].size
              + phase[phase==2].size + phase[phase==3].size)
        print(phase[phase==0].size + phase[phase==1].size
              + phase[phase==2].size + phase[phase==3].size 
              + phase[phase!=phase].size)
        print(d.Nmbhb*d.Nrad)

        print(tinsp_phase_min[phase!=phase])

    return mrg_time, finalsep, phase, r_phase, time_r_phase


def main():
    #for k in file_list.keys():
    for k in ['frefill00']:
    
        d = mbhb_data(path, k, verbose=verbose)
        
        mrg_time, finalsep, phase, r_phase, time_r_phase = define_mbhb_inspiral_phases(d)
        
        total_tinsp = mrg_time - d.sim_tmrg

        print("mrg_time_idx")

        mrg_mask = (mrg_time<=d.tH)
        mrg_mask_rads = np.tile(mrg_mask,(d.Nrad,1)).transpose()
        print("mrg_mask shape: ",mrg_mask.shape)
        print("mrg_mask_rads shape: ",mrg_mask_rads.shape)
        print("phase shape: ",phase.shape)
        
        print("\n{} of {} binaries ({:.6g}%) merge by z=0.\n"
              .format(mrg_time[mrg_mask].size, d.Nmbhb, 
                      mrg_time[mrg_mask].size/d.Nmbhb*100))

        print(np.nanmin(d.tinsp_lc,axis=1).shape)
        print(np.nanmin(d.tinsp_lc,axis=1).min())
        print(np.nanmin(d.tinsp_lc,axis=1).max())
        print(np.median(np.nanmin(d.tinsp_lc,axis=1)))

        
        #r_lc = np.array([sep[i,phase[i,:]==1][0] for i in range(d.Nmbhb)])
        #r_df = np.zeros(d.Nmbhb) * np.nan
        #r_lc = np.zeros(d.Nmbhb) * np.nan
        #r_vd = np.zeros(d.Nmbhb) * np.nan
        #r_gw = np.zeros(d.Nmbhb) * np.nan
        #time_r_df = np.zeros(d.Nmbhb) * np.nan
        #time_r_lc = np.zeros(d.Nmbhb) * np.nan
        #time_r_vd = np.zeros(d.Nmbhb) * np.nan
        #time_r_gw = np.zeros(d.Nmbhb) * np.nan
        #for i in range(d.Nmbhb):
        #    if d.sep[i,phase[i,:]==0].size > 0:
        #        r_df[i] = d.sep[i,phase[i,:]==0][0]
        #        time_r_df[i] = d.times[i,phase[i,:]==0][0]
        #    if d.sep[i,phase[i,:]==1].size > 0:
        #        r_lc[i] = d.sep[i,phase[i,:]==1][0]
        #        time_r_lc[i] = d.times[i,phase[i,:]==1][0]
        #    if d.sep[i,phase[i,:]==2].size > 0:
        #        r_vd[i] = d.sep[i,phase[i,:]==2][0]
        #        time_r_vd[i] = d.times[i,phase[i,:]==2][0]
        #    if d.sep[i,phase[i,:]==3].size > 0:
        #        r_gw[i] = d.sep[i,phase[i,:]==3][0]
        #        time_r_gw[i] = d.times[i,phase[i,:]==3][0]

        #ttot_df = time_r_lc - time_r_df
        #ttot_bin = mrg_time - time_r_lc

        #if verbose:
        #    print("r_df[r_df==r_df].size:",r_df[r_df==r_df].size)
        #    
        #    print("\n\n",r_df[(r_lc==r_lc)&(r_df>r_lc)].size)
        #    print(r_df[(r_vd==r_vd)&(r_df>r_vd)].size)
        #    print(r_df[(r_lc==r_lc)&(r_df<=r_lc)].size)
        #    print(r_df[(r_vd==r_vd)&(r_df<=r_vd)].size)
        #    
        #    print("\n\n",r_lc[(r_lc==r_lc)&(r_vd==r_vd)].size)
        #    print(r_lc[(r_lc==r_lc)&(r_vd==r_vd)&(r_lc>r_vd)].size)
        #    print(r_lc[(r_lc==r_lc)&(r_vd==r_vd)&(r_lc<=r_vd)].size)
        #    print(r_lc[(r_lc==r_lc)&(r_gw==r_gw)&(r_lc<=r_gw)].size)
            
        #    print("\n\n",r_vd[(r_vd==r_vd)&(r_gw==r_gw)].size)
        #    print(r_vd[(r_vd==r_vd)&(r_gw==r_gw)&(r_vd>r_gw)].size)
        #    print(r_vd[(r_vd==r_vd)&(r_gw==r_gw)&(r_vd<=r_gw)].size)


        print("\n\nr_lc:")
        print(np.nanmin(r_lc), np.nanmax(r_lc), r_lc[r_lc==r_lc].size, r_lc.size)

        tmpix=np.where(r_lc!=r_lc)[0]
        print("r_vd:")
        print(np.nanmin(r_vd), np.nanmax(r_vd), r_vd[r_vd==r_vd].size, r_vd.size)
        
        print("r_gw:")
        print(np.nanmin(r_gw), np.nanmax(r_gw), r_gw[r_gw==r_gw].size, r_gw.size)
        
        
        
        ### calculate instead the min tinsp (max dadt) in the df and lc phases 
        ### *in the stage where each is dominant*
        min_tinsp_df_in_df_phase = np.zeros(d.Nmbhb) * np.nan
        min_tinsp_lc_in_lc_phase = np.zeros(d.Nmbhb) * np.nan
        for i in range(d.Nmbhb):
            if i not in tmpix:
                min_tinsp_df_in_df_phase[i] = np.nanmin(d.tinsp_df[i,phase[i,:]==0])
                min_tinsp_lc_in_lc_phase[i] = np.nanmin(d.tinsp_lc[i,phase[i,:]==1])
                
        tarr=10**np.arange(0,12)
        tlim=(1e5,1e14)
        
        if makeplots==True:

            plt.clf()
            plt.cla()
            plt.close
            fig = plt.figure(figsize=(6,8))
            
            ### NEXT STEP: 
            ### calculate the length of each phase, 
            ### and/or the total tinsp from the relevant separation
            ### also calc the time from start to t_lc, then from t_lc to merger
            ### also check on what's up with the very short tinsp_df cases
            ### THEN get the merger tree data and select the systems that are actual potential triples
            ax1 = fig.add_subplot(321)
            plt.xscale('log'), plt.yscale('log')
            plt.xlim(10,1e14)
            plt.ylim(10,1e14)
            plt.xlabel('min $t_{insp,df}$ [yr]')
            plt.ylabel('min $t_{insp,lc}$ [yr]')
            ax1.plot(min_tinsp_df_in_df_phase[~mrg_mask], 
                     min_tinsp_lc_in_lc_phase[~mrg_mask], 'k.', ms=2, alpha=0.5)
            ax1.plot(min_tinsp_df_in_df_phase[mrg_mask], 
                     min_tinsp_lc_in_lc_phase[mrg_mask], 'r.', ms=2, alpha=0.5)
            ax1.plot(tarr,tarr,'k')
            ax1.plot((10,1e14), [d.tH, d.tH], 'gray')
            ax1.plot([d.tH, d.tH], (10,1e14), 'gray')
            

            ax2 = fig.add_subplot(322)
            plt.xscale('log'), plt.yscale('log')
            plt.xlim(tlim)
            #plt.xlim(tlim[0],tlim[1])
            plt.ylim(tlim[0],tlim[1])
            plt.xlabel('$t_{tot,df}$ [yr]')
            plt.ylabel('$t_{tot,bin}$ [yr]')
            ax2.plot(ttot_df[~mrg_mask], ttot_bin[~mrg_mask], 'k.', ms=2, alpha=0.5)
            ax2.plot(ttot_df[mrg_mask], ttot_bin[mrg_mask], 'r.', ms=2, alpha=0.5)
            ax2.plot(tarr,tarr,'k')
            ax2.plot(tlim, [d.tH, d.tH], 'gray')
            ax2.plot([d.tH, d.tH], tlim, 'gray')
            
            ax3 = fig.add_subplot(323)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('$t_{tot} [yr]$')
            plt.ylabel('$t_{tot,bin}/t_{tot,df}$ [yr]')
            plt.xlim(tlim)
            plt.ylim(1.0e-5, 1e8)
            ax3.plot(total_tinsp[~mrg_mask], 
                     ttot_bin[~mrg_mask]/ttot_df[~mrg_mask], 
                     'k.', ms=2, alpha=0.3)
            ax3.plot(total_tinsp[mrg_mask], 
                     ttot_bin[mrg_mask]/ttot_df[mrg_mask], 
                     'r.', ms=2, alpha=0.3)
            ax3.plot(tlim, [1.0,1.0],'k')
            ax3.plot([d.tH, d.tH], [1.0e-5,1.0e8],'gray')
            
            ax4 = fig.add_subplot(324)
            plt.yscale('log')
            plt.xlabel('initial binary separation [log pc]')
            ax4.hist(np.log10(d.sep[:,0]),histtype='step')
            ax4.hist(np.log10(d.sep[mrg_mask,0]),histtype='step')
            
            ax5 = fig.add_subplot(325)
            plt.yscale('log')
            plt.xlabel('a / dadt [yr]')
            ax5.hist(np.log10(d.tinsp[d.tinsp==d.tinsp]),histtype='step')
                        
            maybe_triple_mask = (np.nanmin(d.tinsp_df,axis=1)<np.nanmin(d.tinsp_lc,axis=1))
            
            Nmrg = total_tinsp[mrg_mask].size
            Nnonmrg = total_tinsp[~mrg_mask].size
            N_df_gt_bin_mrg = total_tinsp[(ttot_df>ttot_bin)&(mrg_mask)].size
            N_df_gt_bin_nonmrg = total_tinsp[(ttot_df>ttot_bin)&(~mrg_mask)].size
            
            print("\n{} of {} binaries have min(tinsp_df) < min(tinsp_lc).".
                  format(total_tinsp[maybe_triple_mask].size, d.Nmbhb))
            print("{} of {} merging binaries have min(tinsp_df) < min(tinsp_lc).\n".
                  format(total_tinsp[(maybe_triple_mask)&(mrg_mask)].size, 
                         total_tinsp[mrg_mask].size))
            
            print("{} of {} merging binaries ({:.6g}%) have min(tinsp_df) > min(tinsp_lc).".
                  format(total_tinsp[(~maybe_triple_mask)&(mrg_mask)].size, 
                         total_tinsp[mrg_mask].size,
                         total_tinsp[(~maybe_triple_mask)
                                     &(mrg_mask)].size/total_tinsp[mrg_mask].size*100))
            print("{} of {} nonmerging binaries ({:.6g}%) have min(tinsp_df) > min(tinsp_lc).\n".
                  format(total_tinsp[(~maybe_triple_mask)&(~mrg_mask)].size, 
                         total_tinsp[~mrg_mask].size,
                         total_tinsp[(~maybe_triple_mask)
                                     &(~mrg_mask)].size/total_tinsp[~mrg_mask].size*100))
            
            print("{} of {} merging binaries ({:.6g}%) have ttot_df > ttot_bin.".
                  format(N_df_gt_bin_mrg, Nmrg, N_df_gt_bin_mrg/Nmrg*100))
            print("{} of {} nonmerging binaries ({:.6g}%) have ttot_df > ttot_bin.".
                  format(N_df_gt_bin_nonmrg, Nnonmrg, N_df_gt_bin_nonmrg/Nnonmrg*100))

            ax6 = fig.add_subplot(326)
            plt.yscale('log')
            plt.xlabel('total inspiral time [yr]')
            ax6.hist(np.log10(total_tinsp),histtype='step')
            ax6.hist(np.log10(total_tinsp[mrg_mask]),histtype='step')
            ax6.hist(np.log10(total_tinsp[maybe_triple_mask]), histtype='step')
            ax6.hist(np.log10(total_tinsp[(maybe_triple_mask)&(mrg_mask)]), 
                     histtype='step')
            
            fig.suptitle(k)            
            fig.subplots_adjust(hspace=0.3,wspace=0.4,top=0.95,right=0.95)
            fig.savefig(path+'test_{}.png'.format(k))
