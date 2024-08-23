import numpy as np
import matplotlib.pyplot as plt
from read_mbhb import mbhb_data, define_mbhb_inspiral_phases
from find_triple_mbhs import tripleMatches
from tqdm import tqdm
from matplotlib import rc
from scipy.interpolate import interp1d
import scipy.integrate as integrate
#####for histogram qhist_2d_new#####
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
####################################

'''
PartType0 - GAS
PartType1 - DM
PartType2 - (unused)
PartType3 - TRACERS
PartType4 - STARS & WIND PARTICLES
PartType5 - BLACK HOLES
'''

PARTICLE_TYPES = {0:'GAS', 1:'DM',2:'UNUSED',3:'TRACERS',4:'STARS',5:'BLACK HOLES'}
icloud_path = "/Users/sayebms1/Library/Mobile Documents/com~apple~CloudDocs/paper_2/plots/"

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
        self.z_sim       = (1/self.scales[:,0])-1
        self.z_evol = (1/self.scales)-1
        self.masks = ['triple','binary','triple 1st','triple 2nd']
        
        

    def merger_rates(self, z_filtered):
        import cosmparam.cosm as csm
        from astropy.cosmology import WMAP9
        dz              = 0.01
        bins            = np.arange(0,20+dz/2,dz)
        N,binl,dummy    = plt.hist(z_filtered, bins=bins)
        dN              = -(N[1:]-N[:-1])
#         plt.show()

        HUBBLE_CONSTANT_SEC = (csm.HUBBLE_CONSTANT/(3.086*10**19))       #1/s
        HUBBLE_TIME_YR      = (1/HUBBLE_CONSTANT_SEC)/(365.25*24*60*60)  #yr

        #bin centers are going to be the z values for comoving volume
        bin_centers = np.array((binl[:-1]+binl[1:])/2  )
        V_C         = WMAP9.comoving_volume(bin_centers)
        dV_C        = WMAP9.differential_comoving_volume(bin_centers)
        N_Vc        = np.array(N)/(dz*(106.5**3))
        dzdt        = WMAP9.efunc(bin_centers)*(1+bin_centers)/HUBBLE_TIME_YR 


        factor      = np.array(1/(1+bin_centers))       #1/(1+z)

        integrand = N_Vc*dzdt*dV_C*factor*dz*4*np.pi
        print ('\nThe dN is:')
        print (min(integrand),max(integrand))
        #integrand = N_Vc[:-1]*dzdt[:-1]*dV_C*factor[:-1]#*dz
        print (integrand.shape)
        print ('Merger rate is:',np.sum(integrand))
        
        
        
    def part_mass_plots(self, population = 'both'):

        
        fig, axs = plt.subplots(2,2)
        axs = axs.ravel()
        nbins = 100
        i=0
        for i in range(len(axs)):
            if PARTICLE_TYPES[i]=='UNUSED' or PARTICLE_TYPES[i]=='TRACERS':
                continue
            bins = np.linspace(np.log10(min(self.subhalo_mass_type[:,i])+0.1),np.log10(max(self.subhalo_mass_type[:,i])+0.1), nbins)
            axs[i].set_title(PARTICLE_TYPES[i])
            axs[i].hist(np.log10(self.subhalo_mass_type[:,i][self.triple_mask]), bins = bins, density = True, histtype='step', label ='triples' )
            axs[i].hist(np.log10(self.subhalo_mass_type[:,i][self.binary_mask]), bins = bins, density = True, histtype='step', label ='binary' )
            if i==0:
                axs[i].legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        
    def ax_setup(self, ax, **kwargs):
        import matplotlib.ticker as tck     
        import numpy
        ax.xaxis.set_tick_params(direction='in', which ='both')
        ax.yaxis.set_tick_params(direction='in', which='both')
        ax.set(xscale='log', yscale='log')

        if 'grid' in kwargs.keys():
            if kwargs['grid']==True:
                ax.grid()
        return

    def find_mrgr_idx(self, z=None, merger_z=False):
        '''
        Find the index of binaries that merge by z=0 and those that don't 
        merge by z=0
        Arguments:
        ----------
        z (MxN)    :Array of redshifts for M binaries with N steps for each binary
        Returns:
        ----------
        idx_not_merged_by_z0 (Mx1) :Array of indices for binaries that merge at z>0, -1 is assigned for fill values
        idx_merged_by_z0     (Mx1) :Array of indices for binaries that merge at z<0, -1 is assigned for fill values
        '''
        if z==None:
            z=self.z_evol 
            
        idx_merged_by_z0 = -np.ones(self.Nmbhb)
        idx_not_merged_by_z0 = -np.ones(self.Nmbhb)
        z_at_merger = -np.ones(self.Nmbhb)
        
        for i in range(len(z)):
            if 0 in z[i]:
                idx_not_merged_by_z0[i] = i
            else:
                idx_merged_by_z0[i] = i
                #look for the last non_inf index
                #this is the index where the binary merges
                #the follwing gives the value of red shift at the index
                z_at_merger[i] = z[i][z[i]!=np.inf][-1]
                
        if merger_z:
            return idx_merged_by_z0, idx_not_merged_by_z0, z_at_merger
                
        return idx_merged_by_z0, idx_not_merged_by_z0
    
    def merger_z(self):
        '''
        Here we convert the merger time of the 
        1st (overtaken) and the 2nd (overtaking) binary 
        to redshift.'''
        from astropy.cosmology import WMAP9, z_at_value
        import astropy.units as u
        z_merger_1st = -1*np.ones(self.Nmbhb)
        z_merger_2nd = -1*np.ones(self.Nmbhb)

        times_1st = -1*np.ones(self.Nmbhb)
        times_2nd = -1*np.ones(self.Nmbhb)

        times_1st[self._next.evol_tmrg_1st!=-1] = self._next.evol_tmrg_1st[self._next.evol_tmrg_1st!=-1]/10**9  #division converts years to Gyrs
        times_2nd[self._next.evol_tmrg_2nd!=-1] = self._next.evol_tmrg_2nd[self._next.evol_tmrg_2nd!=-1]/10**9  #division converts years to Gyrs

        for i in tqdm(range(self.Nmbhb), desc = 'Number of binaries processed'):

            if times_1st[i]!=-1 and times_1st[i]<=WMAP9.age(0).to_value():
                z_merger_1st[i] = z_at_value(WMAP9.age, times_1st[i]*u.Gyr, zmax=1000)
            if times_2nd[i]!=-1 and times_2nd[i]<=WMAP9.age(0).to_value():
                z_merger_2nd[i] = z_at_value(WMAP9.age, times_2nd[i]*u.Gyr, zmax=1000)

        return z_merger_1st, z_merger_2nd

    def binned_z_dist(self, z=None, spacing=0.4, mask_type = None):
        
        
        z_overtake = self._next.evol_z_2nd_overtakes
        z_merger_1st, z_merger_2nd = self.merger_z()

        if z==None:
            z=self.z_evol
        
        if mask_type==None or 'binar' in mask_type:
            mask = self.z_sim>-1
        elif 'triple' in mask_type:
            mask = self._next.triple_mask
        elif 'strong' in mask_type:
            mask = self._next.strong_triple_mask
        elif 'weak' in mask_type:
            mask = self._next.weak_triple_mask

        bins = np.arange(-spacing/10,8, spacing)
        bin_centers = (bins[1:]+bins[:-1])/2
        counts_arr = np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            # print ('new bin')
            for binary in range(self.Nmbhb):
                # if binary%4000==0:
                #     print ('this is binary {}'.format(binary))
                if 'binar' in mask_type:
                    # if binary%6000==0:
                    #     print ('binary mask is used' )
                    if any((self.z_evol[binary][self.z_evol[binary]>max(z_merger_1st[binary],z_merger_2nd[binary])]>=bins[i]) & 
                           (self.z_evol[binary][self.z_evol[binary]>max(z_merger_1st[binary],z_merger_2nd[binary])]<bins[i+1])):
                        counts_arr[i]+=1
                    
                elif mask[binary]:
                    # if binary%4000==0:
                    #     print ('Non-binary mask is used' )
                    if any((self.z_evol[binary][self.z_evol[binary]>max(z_merger_1st[binary],z_merger_2nd[binary])]>=bins[i]) & 
                           (self.z_evol[binary][self.z_evol[binary]>max(z_merger_1st[binary],z_merger_2nd[binary])]<bins[i+1])):
                            # & 
                        #    (self.z_evol[binary][self.z_evol[binary]>max(z_merger_1st[binary],z_merger_2nd[binary])]<=z_overtake[binary])):
                        counts_arr[i]+=1
                    
        return counts_arr, bin_centers
    
    def z_of_merged(self,):
        
        '''
        This functions find the redshift at which the binary system merges
        '''
        return
    

    def qhist_2d(self, save_to_icloud=False, blackhole=True, capped_q=False, mask=None):
        """
        Plot 2D histogram of the mass ratios
        """
        mtot = np.sum(self.masses, axis=1)
        #1st binary in the triple mask 
        ix_triple = np.where(self._next.triple_mask)[0]
        ix_1st = self._next.ix_1st_mbhb[self._next.ix_1st_mbhb>=0]; ix_2nd = self._next.ix_2nd_mbhb[self._next.ix_2nd_mbhb>=0]
        ix_inner_triple = np.intersect1d(ix_triple, ix_1st) 
        ix_outer_triple = self._next.ix_2nd_mbhb[ix_inner_triple]
        size = 1e4
        # subhalo_1st_mass = self.envs_SubhaloMassInHalfRad[self.ix_1st_mbhb[self.ix_1st_mbhb>=0]]
        # subhalo_2nd_mass = self.envs_SubhaloMassInHalfRad[self.ix_2nd_mbhb[self.ix_2nd_mbhb>=0]]
        # q_subhalo = self.envs_in_SubhaloMassInHalfRad/self.envs_SubhaloMassInHalfRad
        q_subhalo = self.envs_in_SubhaloMass/self.envs_SubhaloMass
        if capped_q:
            q_subhalo[q_subhalo>1]=1/q_subhalo[q_subhalo>1]
        
        q_1st_subhalo = -np.ones(self.Nmbhb)    
        q_1st_subhalo[self._next.ix_1st_mbhb>=0] = q_subhalo[self._next.ix_1st_mbhb[self._next.ix_1st_mbhb>=0]]

        q_2nd_subhalo = -np.ones(self.Nmbhb)    
        q_2nd_subhalo[self._next.ix_2nd_mbhb>=0] = q_subhalo[self._next.ix_2nd_mbhb[self._next.ix_2nd_mbhb>=0]]
        

        if mask==None:
            mask = self.triple_mask
        elif 'strong' in mask:
            mask = self._next.strong_triple_mask
        elif 'weak' in mask:
            mask = self._next.weak_triple_mask
        
        if blackhole==True:
            if capped_q:
                aa = self._next.q_1st[mask]  #10**np.random.normal(-2.0, size=int(size))
                #aa = np.clip(aa, 1e-5, 1.0)
                bb = self._next.q_2nd[mask] #10**np.random.uniform(-4, 0, size=aa.size)
                #bb = np.clip(bb, 1e-5, 1.0)
                ax_lim=[10**-4,1]
            else:
                q_inner = self._next.q_1st[mask]
                m_inner_binary = mtot[ix_inner_triple]
                m_outer_binary = mtot[ix_outer_triple]
                q_outer = (m_outer_binary-m_inner_binary)/m_inner_binary
                aa = q_inner
                bb = q_outer
                ax_lim=[10**-5,10**10]
                
                
            hist_color='Blues'

        else:
            aa = q_1st_subhalo[mask]  #10**np.random.normal(-2.0, size=int(size))
            #aa = np.clip(aa, 1e-5, 1.0)
            bb = q_2nd_subhalo[mask] #10**np.random.uniform(-4, 0, size=aa.size)
            #bb = np.clip(bb, 1e-5, 1.0)
            hist_color='Reds'
            if capped_q:
                ax_lim=[10**-7,1]
            else:
                ax_lim=[10**-5,10**10]

            
        def logspace(vals, size):
            extr = [vals.min(), vals.max()]
            return np.logspace(*np.log10(extr), size)

        xedges = logspace(aa, 21)
        yedges = logspace(bb, 21)
        
        x_diag = np.logspace(np.log10(aa.min()),0,10)
        y_diag = x_diag
        self.plotting_latex_params()

        hist, *_ = sp.stats.binned_statistic_2d(aa, bb, None, bins=(xedges, yedges), statistic='count')
        norm = mpl.colors.LogNorm(vmin=hist[hist>0].min(), vmax=hist.max())

        ax = plt.gca()
        ax.set_xlabel('$q_{1st}$')
        ax.set_ylabel('$q_{2nd}$')
        ax.set_aspect('equal')
        ax.set_ylim(ax_lim)
        ax.set_xlim(ax_lim)

        xx, yy = np.meshgrid(xedges, yedges)

        pcm = ax.pcolormesh(xedges, yedges, hist.T, cmap=hist_color, norm=norm)
        ax.scatter(aa, bb, s=2, alpha=0.1, color='cyan')
        ax.plot(x_diag, y_diag, '--', color='white')
        self.ax_setup(ax)
        plt.colorbar(pcm, ax=ax, label='number')
        if save_to_icloud:
            plt.savefig(icloud_path+"qhist_2d.pdf")      
                        
    def nbh_dz(self):
        
        fig, axs = plt.subplots()
        bins = np.linspace(0,10,50)
        axs.set_title('Simulation redshift')
        axs.hist(self.z_sim[self.triple_mask], bins=bins, histtype='step', label='triple', density=True)
        axs.hist(self.z_sim[self.binary_mask], bins=bins, histtype='step', label='binary', density=True)
        axs.set_ylabel('number')
        axs.set_xlabel('z')
        axs.legend()
        plt.show()

    def env_histograms(self, pop='triple', save='False'):
        
        '''
        A grid of sub-plots for the environments of triples.
        all the subpopulations are plotted here
        for all values use half mass radius 
        '''
        import seaborn as sns
        from matplotlib.lines import Line2D
        sim_unit_h_scale=0.704
        z = self.z_sim
        # subhalo_mass = self.envs_SubhaloMassInHalfRad/sim_unit_h_scale
        subhalo_mass = self.envs_SubhaloMass/sim_unit_h_scale
        # star_mass = self.envs_SubhaloMassInHalfRadType[:,4]/sim_unit_h_scale 
        star_mass = self.envs_SubhaloMassType[:,4]/sim_unit_h_scale 
        sfr = self.envs_SubhaloSFRinHalfRad
        ssfr = sfr/star_mass
        subhalo_vdisp = self.envs_SubhaloVelDisp

        env_param = {'z':z,'SSFR':ssfr, 'subhalo mass':subhalo_mass, 'star mass':star_mass, 'velocity disp': subhalo_vdisp} 
        x_labels = {'z':'z','SSFR':'sSFR \;\\rm [10^{-10} yr^{-1}]', 'subhalo mass':'M_{\\rm subhalo} \\rm [10^{10} M_{\\odot}]', 'star mass':'M_{*}\\rm [10^{10} M_{\\odot}]', 'velocity disp': '\\sigma \\rm[km/s]'}
        self.plotting_latex_params()
        fig, axs = plt.subplots(3,2, figsize=(5,6), sharey=True,)
        ax_ylim=[10**0-0.5,10**3.5]
        bin_number=20

        for ax, key in zip(axs.flatten()[:-1], env_param.keys()):
            my_bins = logspace_bin(env_param[key], bin_number)
            x = env_param[key]
            if pop=='triple':
                ax.hist(x[(self._next.failed_triple_mask)|(self._next.triple_mask)], bins = my_bins, histtype='step', linestyle=('dashed'), label='All overlap (AO)', zorder=3)                
                ax.hist(x[self._next.strong_triple_mask], bins = my_bins, histtype='step', label='Strong triple (ST)')
                ax.hist(x[self._next.weak_triple_mask], bins = my_bins, histtype='step', label='Weak triple (WT)')
                ax.hist(x[self._next.failed_triple_mask], bins = my_bins, histtype='step', label='Failed triple (FT)')
            elif pop=='binary':
                ax.hist(x, bins = my_bins, histtype='step', color='Black', linestyle=('dashed'), label='All binaries (AB)', zorder=3)            
                ax.hist(x[~((self._next.failed_triple_mask)|(self._next.triple_mask))], bins = my_bins, histtype='step', color='Gray', label='Isolated binary (IB)')
                ax.hist(x[self._next.triple_mask], bins = my_bins, histtype='step', color='Magenta', label='1st merger')              
                ax.hist(x[self._next.ix_2nd_mbhb[self._next.triple_mask]], bins = my_bins, histtype='step', color='Teal', label='2nd merger')              

            ax.set_xlabel('$'+x_labels[key]+'$')  
            ax.set_ylim(ax_ylim)
            self.ax_setup(ax)
        ax.set_ylabel('Number')
        axs.flatten()[-1].axis('off')
        handles, labels = ax.get_legend_handles_labels()
        linstyle=['--','-','-','-']
        new_handles = [Line2D([],[], c=handles[i].get_edgecolor(), ls=linstyle[i]) for i in range(len(handles))]
        fig.legend(new_handles, labels, bbox_to_anchor=(0.42, -2, 0.5, 0.5))
        plt.tight_layout() 
        plt.subplots_adjust(wspace=0)
        if save:
            print ('saving...')
            plt.savefig('./2nd_paper_plots/env_all_'+pop+'.png')
            # plt.savefig(icloud_path+"env_all_"+pop+'.png')
        plt.show()

        
    def plot_histograms(self, nbins = 100, direction = None , parameter= None, **kwargs):
        
        '''
        Plotting function for inspiral time and mass ratios
        
        Argument:
        nibns (int)    : Number of bins for the histogram
        direction (str): Direction of tracing merger
        parameter (str): mass ratio 'q' or inspiral time 'insp'
        
        Keyword arguments:
        systems        : either triple or binary systems
        refill fraction: in the name of the parameter if '000', '060', '099'
        
        Returns:
        A histogram of the desired parameters
        
        smt.plot_histograms(nbins = 100, direction = direction , parameter= 'insp', tinsp_triple_000 =tinsp_masked_000,\
                tinsp_triple_060 = tinsp_masked_060, tinsp_triple_099 = tinsp_masked_099, systems='binary')
        '''

        import matplotlib.pyplot as plt
        if 'insp' in parameter:   #for inspiral time
            final_tinsp = []
            for key in kwargs.keys():
                final_tinsp.append(kwargs[key])
            final_tinsp = np.array(final_tinsp)

    #         bins = np.linspace(np.log10(final_tinsp.min()),np.log10(10**14), nbins)
            bins = np.linspace(np.log10(5000),np.log10(10**14), nbins)
            xlabel = '$\\log(t_{\\rm insp}\\rm [yr])$'
            sys_title = 'triple'
            if 'systems' in kwargs.keys():
                sys_title = kwargs['systems']
            if direction == 'prev':
                plt.title('Inspiral time for '+str(sys_title)+' using previous prescription')
            elif direction == 'next':
                    plt.title('Inspiral time for '+str(sys_title)+' using next prescription')


            for key in kwargs.keys():
                if '000' in key:  # for the zero lc refill one with ecc=0
                    label = '$f_{\\rm refill}=0.0$'
                elif '060' in key:
                    label = '$f_{\\rm refill}=0.6$'
                elif '099' in key:
                    label = '$f_{\\rm refill}=0.99$'
                print (key)
                if 'system' in key:
                    continue
                plt.hist(np.log10(kwargs[key]), bins = bins, linestyle='--', histtype='step', label=label)


        elif 'q' in parameter:  #for mass ratio
            bins = np.linspace(0,1,nbins)
            xlabel = '$q$'

            if direction == 'prev':
                plt.title('mass ratios for previous mergers')
            elif direction == 'next':
                plt.title('mass ratios for next mergers')
            for key in kwargs.keys():
                if key !='systems':
                    if '1' in key:
                        label = '$q\; 1st$'
                    elif '2' in key:
                        label = '$q\; 2nd$'
                    else:
                        label = '$All \; q$'
                    plt.hist(kwargs[key], bins = bins, linestyle='--',histtype = 'step', label=label)
            plt.yscale('log')

        plt.ylabel('$number$')
        plt.xlabel(xlabel)
        plt.legend()
        plt.minorticks_on()
        plt.show()
        return
    
    def get_statistics(self, env_param):
        first_merger = env_param[self._next.triple_mask]
        secnd_merger = env_param[self._next.ix_2nd_mbhb[self._next.triple_mask]]
        print ('1st merger (triple mask) Median {} IQR {} Mean {} and STD {}'.format(np.median(first_merger)
                                                                                        ,np.percentile(first_merger,75)
                                                                                        -np.percentile(first_merger,25)
                                                                                        , np.mean(first_merger)
                                                                                        , np.std(first_merger)))
        print ('2nd merger (triple mask 2nd_mbh_indx) Median {} IQR {} Mean {} and STD {}'.format(np.median(secnd_merger)
                                                                                                          , np.percentile(secnd_merger,75)
                                                                                                          -np.percentile(secnd_merger,25)
                                                                                                          , np.mean(secnd_merger)
                                                                                                          , np.std(secnd_merger)))    
    
    def print_percentages(self):
        '''
        print percentages of strong weak triple failedtriple and all overlap 
        population from the total binary population
        '''
        
        print ('Strong triple:',100*len(self.q[self._next.strong_triple_mask])/len(self.q))
        print ('Weak triple:',100*len(self.q[self._next.weak_triple_mask])/len(self.q))
        print ('Triple:',100*len(self.q[self._next.triple_mask])/len(self.q))
        print ('failed triple:',100*len(self.q[self._next.failed_triple_mask])/len(self.q))
        print ('All overlap:',100*(len(self.q[self._next.failed_triple_mask])+len(self.q[self._next.triple_mask]))/len(self.q))        
        
    def new_sep(self, n=100):
        '''
        These are the new separations over 
        which we are going to find the values
        after the ingerpolation is performed
        
        **define parameters later**
        
        '''
        new_seps = np.logspace(-8,6,n)
        return new_seps#[::-1]  #to invert separation from larger to smaller (decresing separation)
        
    
    
    def hardening_interpolate(self):    
        '''
        This function performs the interpolation for all the hardening 
        rates. The interpolation functions are then used to plot hardening 
        time scale vs separation plot. The zero values in the arrays will 
        cause issues with the interpolation. Therefore they are set to very 
        small values instead of zero
        
        Arguments:
        ----------
        
        Returns:
        ----------
        Interpolated values for the new separation
        
        '''
        
        from scipy.interpolate import interp1d

        #Zeros are set to small value to make the interpolation work properly
        dadt_df = self.dadt_df; dadt_df[dadt_df==0] = -(10**-53)
        dadt_vd = self.dadt_vd; dadt_vd[dadt_vd==0] = -(10**-53)
        dadt_lc = self.dadt_lc; dadt_lc[dadt_lc==0] = -(10**-53)        
        dadt_gw = self.dadt_gw; dadt_gw[dadt_gw==0] = -(10**-53)
        sep = self.sep; sep[self.sep==0] = 10**-53
        
        new_sep = self.new_sep()
        #invert new separation from larger to smaller for interpolation
        f_dadt_df_l=[]
        f_dadt_lc_l=[]
        f_dadt_vd_l=[]
        f_dadt_gw_l=[]

        new_dadt_df=[]
        new_dadt_lc=[]
        new_dadt_vd=[]
        new_dadt_gw=[]


        for i in tqdm(range(len(self.dadt)), desc='Performing interpolation of hardening rates'):
            
            f_dadt_df=interp1d(sep[i],dadt_df[i],bounds_error=False, fill_value=0)
            f_dadt_lc=interp1d(sep[i],dadt_lc[i],bounds_error=False, fill_value=0)
            f_dadt_vd=interp1d(sep[i],dadt_vd[i],bounds_error=False, fill_value=0)
            f_dadt_gw=interp1d(sep[i],dadt_gw[i],bounds_error=False, fill_value=0)
            f_dadt_df_l.append(f_dadt_df)
            f_dadt_lc_l.append(f_dadt_lc)
            f_dadt_vd_l.append(f_dadt_vd)
            f_dadt_gw_l.append(f_dadt_gw)
            new_dadt_df.append(f_dadt_df(new_sep))
            new_dadt_lc.append(f_dadt_lc(new_sep))
            new_dadt_vd.append(f_dadt_vd(new_sep))
            new_dadt_gw.append(f_dadt_gw(new_sep))

        f_dadt_df_l=np.array(f_dadt_df_l)
        f_dadt_lc_l=np.array(f_dadt_lc_l)
        f_dadt_vd_l=np.array(f_dadt_vd_l)
        f_dadt_gw_l=np.array(f_dadt_gw_l)

        self.new_dadt_df=np.array(new_dadt_df)
        self.new_dadt_lc=np.array(new_dadt_lc)
        self.new_dadt_vd=np.array(new_dadt_vd)
        self.new_dadt_gw=np.array(new_dadt_gw)
        
    
    def find_percentiles(self, mask):
        '''
        mask<str>   : 'triple', 'binary', 'all'
        '''
        self.hardening_interpolate()
        #find median, 25 percentile, and 75 percentile
        #medians
        self.medn_df_l=[]
        self.medn_lc_l=[]
        self.medn_vd_l=[]
        self.medn_gw_l=[]
        #25 percentile
        self.pr25_df_l=[]
        self.pr25_lc_l=[]
        self.pr25_vd_l=[]
        self.pr25_gw_l=[]
        #50 percentile
        self.pr50_df_l=[]
        self.pr50_lc_l=[]
        self.pr50_vd_l=[]
        self.pr50_gw_l=[]
        #75 percentile
        self.pr75_df_l=[]
        self.pr75_lc_l=[]
        self.pr75_vd_l=[]
        self.pr75_gw_l=[]
        #0 percentile
        self.pr00_df_l=[]
        self.pr00_lc_l=[]
        self.pr00_vd_l=[]
        self.pr00_gw_l=[]
        #100 percentile
        self.pr99_df_l=[]
        self.pr99_lc_l=[]
        self.pr99_vd_l=[]
        self.pr99_gw_l=[]

        new_sep = self.new_sep()
        bin_range = self.bin_range_from_mask(mask)

        for i in range(len(new_sep)):
            medn_df=np.median(self.new_dadt_df[bin_range,i])
            medn_lc=np.median(self.new_dadt_lc[bin_range,i])
            medn_vd=np.median(self.new_dadt_vd[bin_range,i])
            medn_gw=np.median(self.new_dadt_gw[bin_range,i])


            self.medn_df_l.append(medn_df)
            self.medn_lc_l.append(medn_lc)
            self.medn_vd_l.append(medn_vd)
            self.medn_gw_l.append(medn_gw)

            self.pr25_df_l.append(np.percentile(self.new_dadt_df[bin_range,i],25))
            self.pr25_lc_l.append(np.percentile(self.new_dadt_lc[bin_range,i],25))
            self.pr25_vd_l.append(np.percentile(self.new_dadt_vd[bin_range,i],25))
            self.pr25_gw_l.append(np.percentile(self.new_dadt_gw[bin_range,i],25))

            self.pr50_df_l.append(np.percentile(self.new_dadt_df[bin_range,i],50))
            self.pr50_lc_l.append(np.percentile(self.new_dadt_lc[bin_range,i],50))
            self.pr50_vd_l.append(np.percentile(self.new_dadt_vd[bin_range,i],50))
            self.pr50_gw_l.append(np.percentile(self.new_dadt_gw[bin_range,i],50))

            self.pr75_df_l.append(np.percentile(self.new_dadt_df[bin_range,i],75))
            self.pr75_lc_l.append(np.percentile(self.new_dadt_lc[bin_range,i],75))
            self.pr75_vd_l.append(np.percentile(self.new_dadt_vd[bin_range,i],75))
            self.pr75_gw_l.append(np.percentile(self.new_dadt_gw[bin_range,i],75))

            self.pr00_df_l.append(np.percentile(self.new_dadt_df[bin_range,i],0))
            self.pr00_lc_l.append(np.percentile(self.new_dadt_lc[bin_range,i],0))
            self.pr00_vd_l.append(np.percentile(self.new_dadt_vd[bin_range,i],0))
            self.pr00_gw_l.append(np.percentile(self.new_dadt_gw[bin_range,i],0))


            self.pr99_df_l.append(np.percentile(self.new_dadt_df[bin_range,i],99))
            self.pr99_lc_l.append(np.percentile(self.new_dadt_lc[bin_range,i],99))
            self.pr99_vd_l.append(np.percentile(self.new_dadt_vd[bin_range,i],99))
            self.pr99_gw_l.append(np.percentile(self.new_dadt_gw[bin_range,i],99))

        #for total population percentiles
        self.pr25_tot_l=(np.array(self.pr25_df_l)+np.array(self.pr25_lc_l)+np.array(self.pr25_vd_l)+np.array(self.pr25_gw_l))
        self.pr75_tot_l=(np.array(self.pr75_df_l)+np.array(self.pr75_lc_l)+np.array(self.pr75_vd_l)+np.array(self.pr75_gw_l))


    def hardening_rates(self, mask, scatter_mask='triple', mass_q_index='first', zoom=True, save_to_icloud=False):
        
        '''
        Plots the inspiral time scale vs separation for masked binaries
        usage: hardening_rates(self, mask, scatter_mask='triple', zoom=True, save_to_icloud=False)
        
        Arguments:
        ----------
        mask     <str>: 
            'triple'- for systems that overlap and merge before z0
            'binary'- complementary set to the 'triple' set 
            'all'- all of the binaries 
            'triple 1st'- the binary that is overtaken
            'triple 2nd'- the binary that overtakes the 1st
            
        Returns:
            Inspiral time vs separation plot with the shaded region for the inner quartile
        '''
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        self.find_percentiles(mask)
        new_sep = self.new_sep(n=100)
        
        #inspiral time scales
        #25 percentile
        pr25_dfts_l=new_sep/np.array(self.pr25_df_l)
        pr25_lcts_l=new_sep/self.pr25_lc_l
        pr25_vdts_l=new_sep/self.pr25_vd_l
        pr25_gwts_l=new_sep/self.pr25_gw_l
        pr25_totts_l=new_sep/(np.array(self.pr25_df_l)+np.array(self.pr25_lc_l)+np.array(self.pr25_vd_l)+np.array(self.pr25_gw_l))
        #50 percentile
        pr50_dfts_l=new_sep/self.pr50_df_l
        pr50_lcts_l=new_sep/self.pr50_lc_l
        pr50_vdts_l=new_sep/self.pr50_vd_l
        pr50_gwts_l=new_sep/self.pr50_gw_l
        pr50_totts_l=new_sep/(np.array(self.pr50_df_l)+np.array(self.pr50_lc_l)+np.array(self.pr50_vd_l)+np.array(self.pr50_gw_l))

        #75 percentile
        pr75_dfts_l=new_sep/self.pr75_df_l
        pr75_lcts_l=new_sep/self.pr75_lc_l
        pr75_vdts_l=new_sep/self.pr75_vd_l
        pr75_gwts_l=new_sep/self.pr75_gw_l
        pr75_totts_l=new_sep/(np.array(self.pr75_df_l)+np.array(self.pr75_lc_l)+np.array(self.pr75_vd_l)+np.array(self.pr75_gw_l))
        
        total_tinsp = self._next.total_tinsp_integrated
        init_tinsp_l = -np.ones(len(total_tinsp))
        init_sep_l = -np.ones(len(total_tinsp))
        
        for i in range(len(total_tinsp)):
            if self._next.ixsep_1st_overtaken[i]<0: continue;
            init_tinsp_l[i] = total_tinsp[i, 0]
            init_sep_l[i] = self.sep[i, 0]
        mask_1st_overtaken =  self._next.ixsep_1st_overtaken>0
        mask_2nd_overtakes =  self._next.ixsep_2nd_overtakes>0          
               
        self.plotting_latex_params()
        
        fig= plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111)
        ax.fill_between(new_sep, np.abs(pr25_dfts_l), np.abs(pr75_dfts_l), color='red', alpha=0.1,label='Dynamical  friction')
        ax.fill_between(new_sep, np.abs(pr25_lcts_l), np.abs(pr75_lcts_l), color='orange', alpha=0.1,label='Loss cone')
        ax.fill_between(new_sep, np.abs(pr25_vdts_l), np.abs(pr75_vdts_l), color='blue', alpha=0.1,label='Circumbinary disk')
        ax.fill_between(new_sep, np.abs(pr25_gwts_l), np.abs(pr75_gwts_l), color='maroon', alpha=0.1,label='Gravitational wave')
        
        if scatter_mask =='overtake':
            
            #plot where the 1st is overtaken and where the second overtakes in different colors ideally these need to be very similar
            ax.scatter(self._next.sep_1st_overtaken[mask_1st_overtaken], self._next.integrated_tinsp_1st_overtaken[mask_1st_overtaken], color='red'\
                       , s=12, alpha=0.1, edgecolors='none', label='$a$ 1st overtaken')
            ax.scatter(self._next.sep_2nd_overtakes[mask_2nd_overtakes], self._next.integrated_tinsp_2nd_overtakes[mask_2nd_overtakes], color='green'\
                       , s=12, alpha=0.1, edgecolors='none', label='$a$ 2nd overtakes')        
        elif scatter_mask=='triple':
            mtot = self.masses[:,0]+self.masses[:,1]
            if mass_q_index=='first':
                my_mask = np.where(self._next.triple_mask)[0] #this is the mass and mass ratio of the 1st MBHB
            elif mass_q_index=='second':
                ix_triple = np.where(self._next.triple_mask)[0]
                ix_1st = self._next.ix_1st_mbhb[self._next.ix_1st_mbhb>=0]
                ix_2nd = self._next.ix_2nd_mbhb[self._next.ix_2nd_mbhb>=0]
                ix_1st_triple = np.intersect1d(ix_triple, ix_1st)
                my_mask = self._next.ix_2nd_mbhb[ix_1st_triple]
            print ('length of my_mask is:', len(my_mask))

            ###need to specify which masses and mass ratios we are looking at

            s1= ax.scatter(self._next.sep_2nd_overtakes[self._next.triple_mask], self._next.integrated_tinsp_2nd_overtakes[self._next.triple_mask]
                           , s=0.1*(np.log10(mtot[my_mask])/(np.log10(mtot[my_mask].max())-np.log10(mtot[my_mask].min())))**6
                           , c=np.log10(self.q[my_mask]), cmap='jet'
                           , alpha=0.9, edgecolors='none', label='$a$ 2nd overtakes')
            print ('length of s is:', len(0.1*(np.log10(mtot[my_mask])/(np.log10(mtot[my_mask].max())-np.log10(mtot[my_mask].min())))**6))
            # circle_patches = mpl.patches.Circle((10**5,10**5),radius=10**np.max(np.log10(mtot[my_mask])))
            # plt.legend(handles=[circle_patches], loc='lower right')
            from matplotlib.lines import Line2D
            # import matplotlib.pyplot as plt

            red_circle = Line2D([0], [0], marker='o', color='w', label='Circle',
                        markerfacecolor='black', markersize=15),
            ax.legend(handles=red_circle, loc='lower right')
           
            
        #----to add color bar--------                           
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        #---to add axis on top for distributions-----
        distax = divider.append_axes('top', size='35%', pad=0.15)
        dist_bins = np.logspace(np.log10(10**-4),np.log10(20000),40)
        filtere_sep = self._next.sep_2nd_overtakes[my_mask]
        filtered_q = self.q[my_mask]
        filtered_mtot = mtot[my_mask]
        density=False
        distax.hist(filtere_sep, bins=dist_bins, histtype='step', linestyle=('dashed'),label='all', density=density)
        distax.hist(filtere_sep[filtered_q<1e-2], bins=dist_bins, histtype='step', linestyle=('solid'),label='$q<1e-2$', density=density)
        distax.hist(filtere_sep[filtered_mtot>3*10**8], bins=dist_bins, histtype='step', linestyle=('solid'), label='$M_{\\rm tot}>3\\times10^{8}M_{\\odot}$', density=density)
        distax.hist(filtere_sep[filtered_mtot>3*10**7], bins=dist_bins, histtype='step', linestyle=('solid'),label='$M_{\\rm tot}>3\\times10^{7}M_{\\odot}$', density=density)       
        distax.hist(filtere_sep[(filtered_mtot>3*10**7) & (filtered_q>0.1)], bins=dist_bins, histtype='step', linestyle=('solid'),label='$M_{\\rm tot}>3\\times10^{7}M_{\\odot} \\& q>0.1$ ', density=density)
        distax.set_yscale('log')
        distax.set_xscale('log')
        distax.legend(loc='upper left')
        distax.set_ylabel('number', fontsize=15)
        distax.tick_params(axis="both", labelsize=14, which='both')
        distax.get_xaxis().set_visible(False)
        
        
        fig.colorbar(s1, cax=cax, orientation='vertical', label='log(q)')
        #----------------------------        
        ax.legend(loc='upper left')
        if zoom==True:
            ax.set_xlim(xmin=10**-2.5,xmax=9000)
#             ax.set_xlim(xmin=10**2,xmax=10**4)
            ax.set_ylim(ymin=10**7,ymax=10**12)
        else:
            print ('no zoom')
            ax.set_xlim(xmin=10**-4,xmax=20000)
            ax.set_ylim(ymin=10**3,ymax=10**1)
        ax.set_xlabel('$a[{\\rm pc}]$',fontsize=20)
        ax.set_ylabel('$t_{\\rm insp}[{\\rm yr}]$',fontsize=20)
        ax.tick_params(axis="both", labelsize=14, which='both')
        ax = self.ax_setup(ax)
#         plt.minorticks_on()
        if save_to_icloud:
            plt.savefig(icloud_path+"hardening_scatter.pdf")
        
        plt.show()

    def scatter_overtk_params(self, all_combinations=False):
        '''
        plot mass, sep_ovrtk, q all with triple mask
        Figure 2 in the paper 
        '''

        sep_ovrtk_1st = self._next.sep_1st_overtaken[self._next.triple_mask]
        sep_ovrtk_2nd = self._next.sep_2nd_overtakes[self._next.triple_mask]
        q = self.q[self._next.ix_2nd_mbhb[self._next.triple_mask]]
        mtot = self.masses[:,0]+self.masses[:,1]; mtot=mtot[self._next.triple_mask]
        self.plotting_latex_params()
        if all_combinations:
            fig, axs = plt.subplots(2,2, figsize=(7, 5))
            list_of_arrays = np.array([sep_ovrtk_2nd, q, mtot])
            list_of_names = np.array(['sep_ovrtk_2nd', 'q', 'mtot'])
            combinations = self.find_combinations(list_of_arrays, 2)
            name_comb = self.find_combinations(list_of_names, 2)
            axs = axs.flatten()
            for i in range(len(axs)):
                if i==3:
                    continue
                axs[i].scatter(combinations[i][0], combinations[i][1], s=1, alpha=0.5)
                axs[i].set_xlabel(name_comb[i][0])
                axs[i].set_ylabel(name_comb[i][1])

                axs[i].set_yscale('log')
                axs[i].set_xscale('log')
                self.ax_setup(axs[i])
            plt.tight_layout()       
            axs[-1].axis('off')

            
        else:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            fig, ax = plt.subplots(1,1, figsize=(5,4.5))
            s1= ax.scatter(sep_ovrtk_2nd, q, c=np.log10(mtot), cmap='jet'\
                           , s=10, alpha=0.9, edgecolors='none', label='$a$ 1st overtaken')   
#             s1= ax.scatter(sep_ovrtk_1st, q, c=np.log10(self.z_sim[self._next.triple_mask]), cmap='jet'\
#                            , alpha=0.1, edgecolors='none', label='$a$ 1st overtaken')   
            ax.set_ylabel('$q$')
            ax.set_xlabel('$a {\\rm[pc]}$')
            ax.set_yscale('log')
            ax.set_xscale('log')
            self.ax_setup(ax)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(s1, cax=cax, orientation='vertical', label='$\\log(m_{tot})$')
        plt.show()                    

    def find_combinations(self, input_array, element_size=2):
        '''
        -------should go to math script-------
        Find combinations of an array of arrays.
        Note: this could be an array of arrays
        '''
        import itertools
        combinations = []
        
        for comb in itertools.combinations(input_array, element_size):
            combinations.append(comb)
        return combinations

        
    def tinsp_at_sep(self, sep_type=None):
        
        if sep_type == None or sep_type == 'initial':
            print ('Calculating inspiral time at Illustris separation')
            sep = self.sep[:,0]
            tinsp = self.tinsp[:,0]
            return sep, tinsp

        elif sept_type == '1st overtake':
            self.sep_1st_overtaken[i] = self.sep[self.ix_1st_mbhb[i], 
                                                 self.ixsep_1st_overtaken[i]]
            
        elif sept_type == '2nd overtake':
            pass
        
        
        
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
        
        
    def plotting_latex_params(self):
        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)
      
    
    
    def interpolate_2d_arrays(self, x_array, y_array, evaluate=False):
        '''
        This function performs interpolation on x_array and y_array.
        The two input arrays must be 2d and of the same size and shape.
        
        Arguments:
        ---------
        x_array   <numpy.ndarray> (MxN): MxN array where M is the number 
                                         of binaries and N is separations
        y_array   <numpy.ndarray> (MxN): MxN array where M is the number 
                                         of binaries and N is separations

        Returns:
        ---------
        Interpoltion functino  <interp1d> (Mx1): M is the number of binaries
                                                 Returns interpolation function 
        '''
        from scipy.interpolate import interp1d
        
        interp_function = np.array([None]*x_array.shape[0])
        for i in tqdm(range(len(interp_function)), desc = 'Performing interpolation'):
            interp_function[i] = interp1d(x_array[i], y_array[i], bounds_error=False, fill_value=0)
        if not evaluate:
            return interp_function
        else:
            new_sep = self.new_sep()
            return
     
    
    def bin_range_from_mask(self, mask):
        '''
        To find the binary range for plotting using the mask
        parameter.
        '''
        
        if mask == 'triple':
            bin_range = np.where(self._next.triple_mask==True)[0]
        elif mask == 'binary':
            bin_range = np.where(self._next.triple_mask==False)[0]
        elif mask == 'triple 1st':
            bin_range = self._next.ix_1st_mbhb[self._next.triple_mask]
        elif mask == 'triple 2nd':
            bin_range = self._next.ix_2nd_mbhb[self._next.triple_mask]            
        else:
            bin_range = range(self.nmbhb)    
        return bin_range

    def accretion_rates(self, mask='triple'):
        '''
        Interpolated accretion rates plot with mask
        '''
        sep = self.sep; sep[sep==0] = 1e-15
        mdot_eff = self.mdot_eff; mdot_eff[mdot_eff==0] = 1e-53

        f_mdot = self.interpolate_2d_arrays(sep, mdot_eff)
        new_sep = self.new_sep()
            
        bin_range = self.bin_range_from_mask(mask)

        new_mdot = -np.ones((len(f_mdot),len(new_sep)))
        for i in range(len(f_mdot)):
            new_mdot[i] = f_mdot[i](new_sep)

        pr25_mdot=[]
        pr50_mdot=[]                
        pr75_mdot=[]

        for i in range(len(new_sep)):
            pr25_mdot.append(np.percentile(new_mdot[bin_range,i],25))
            pr50_mdot.append(np.percentile(new_mdot[bin_range,i],50))
            pr75_mdot.append(np.percentile(new_mdot[bin_range,i],75))
        
        self.plotting_latex_params()
        fig= plt.figure()
        ax = fig.add_subplot(111)
#         for i in range(len(sep)):
#             ax.plot(new_sep,f_mdot[i](new_sep), color='red', alpha=0.01)
        ax.fill_between(new_sep, pr25_mdot, pr50_mdot, color='maroon', alpha=0.2,label='Accretion')
        ax.set_ylabel('$\\dot{m}\\rm{[M_{\\odot}yr^{-1}]}$', fontsize=20)
        ax.set_xlabel('$a\\rm{[pc]}$', fontsize=20)        
        ax.set_xlim(xmin=10**-9, xmax=10**4)
        ax.set_ylim(ymin=mdot_eff[mdot_eff>1e-12].min(), ymax = 1e1)
        self.ax_setup(ax)
        plt.show()
        
    def accretion_rates_binned_plot(self):
        '''
        ---WORK ON THIS LATER---
        Plot the accretion rates by binning the 
        separation polot
        '''
        
        lower_bound = np.log10(self.sep[self.sep>0].min()).astype(int)-1
        upper_bound = np.log10(self.sep.max()).astype(int)+1
        sep_bins = np.logspace(lower_bound, upper_bound, 100)
        print (sep_bins)
        
        return
    
    def q1d_histogram(self, density=True):
        '''
        1D histogram of mass ratios for all the masks
        '''
        self.plotting_latex_params() 
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        bins = np.logspace(-5,0, 20)
        for i in range(len(self.masks)):
            bin_range = self.bin_range_from_mask(self.masks[i])   
            ax.hist(self.q[bin_range], bins=bins, histtype='step', label=self.masks[i], density=density)
        ax.legend(loc='upper left')
        ax.set_xlabel('$q$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        self.ax_setup(ax)
        plt.show()
        

    def mtot_histogram(self, density=True):
        '''
        1D histogram of total binary mass for all the masks
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        bins = np.logspace(5,11, 20)
        for i in range(len(self.masks)):
            bin_range = self.bin_range_from_mask(self.masks[i])   
            ax.hist(np.sum(self.masses, axis=1)[bin_range], bins=bins, histtype='step', label=self.masks[i], density=density)
        ax.set_xlabel('masses $\\rm[M_{\\odot}]$')
        ax.legend(loc='upper left')
        ax.set_yscale('log')
        ax.set_xscale('log')
        self.plotting_latex_params()  
        plt.show()
        
        
    def plot_subhalo_mass(self, density=False, part_type = 'total'):
        '''
        Plot histogram of total subhalo mass
        '''
        subhalo_mass = np.sum(self.subhalo_mass_type, axis=1)        

        fig = plt.figure()
        ax = fig.add_subplot(111)
        if part_type=='total':
            bins = np.logspace(9,13.2, 40)
            for i in range(len(self.masks)):
                bin_range = self.bin_range_from_mask(self.masks[i])
                ax.hist(subhalo_mass[bin_range], bins=bins, histtype='step', label=self.masks[i], density=density)
            ax.hist(subhalo_mass, bins=bins, histtype='step', label='total', density=density)
            ax.set_ylabel('$P$')
            ax.set_xlabel('Subhalo mass in half rad $[M_{\\odot}]$')    
            
        elif part_type=='star':
            bins = np.logspace(7.5,13.5, 40)
            for i in range(len(self.masks)):
                bin_range = self.bin_range_from_mask(self.masks[i])
                ax.hist(self.subhalo_mass_type[:,4][bin_range], bins=bins, histtype='step', label=self.masks[i], density=density)
            ax.hist(self.subhalo_mass_type[:,4], bins=bins, histtype='step', label='total', density=density)
            ax.set_ylabel('$Number$')
            ax.set_xlabel('stellar mass $[M_{\\odot}]$')                

            
            
        self.ax_setup(ax)
        self.plotting_latex_params()
        ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()
        
            
    def time_scales (self, sep=None, dadt=None, interpolated=False):
        '''
        Caculates ((da/dt)^-1)*a, where a is the separation
        by default it calculates for all binaries and and all
        separations using the total hardening rate
        ------- Finish this later---------
        '''
        if sep is None and dadt is None:
            sep = self.sep
            dadt = self.dadt_df+self.dadt_lc+self.dadt_vd+self.dadt_gw
        tinsp = np.array([None]*sep.shape[0])        
        assert len(tinsp.shape)<2
        for i in tqdm(range(tinsp.shape[0]), desc = 'Calculating inspiral time for all binaries all separations'):
            tinsp[i] = self.find_tinsp(sep[i], dadt[i], interpolated)
        
        return np.stack(tinsp)
    
    def sep_vs_time(self, ii=None):
        '''
        Plot separation vs time plot for 1st and 2nd merger 
        for sucessful triples to track their trajectory in 
        redshift and separation
        
        RETURN:
        '''
        if ii is None:
            ii = 1
        ix1st_triple = self._next.ix_1st_mbhb[self._next.triple_mask]
        ix2nd_triple = self._next.ix_2nd_mbhb[self._next.triple_mask]
        
        idx_1st = ix1st_triple[ii]
        idx_2nd = ix2nd_triple[ii]
        
        print (idx_1st, idx_2nd)
        
        self.plotting_latex_params()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.times[idx_1st], self.sep[idx_1st], label='1st/Current merger')
        ax.plot(self.times[idx_2nd], self.sep[idx_2nd], label='2nd/subsequent merger')
        self.ax_setup(ax)
        
        ax.set_ylabel('$a[\\rm pc]$')
        ax.set_xlabel('$time[\\rm yr]$')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.legend()
        plt.show()        

def logspace_bin(vals, size):
    if vals.min()==0:
        extr = [vals[vals>0].min()/10, vals.max()]
    else:
        extr = [vals.min(), vals.max()]
    return np.logspace(*np.log10(extr), size)



def scatter_plot_with_side_histograms():
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, 
                           gridspec_kw={
                               'width_ratios': [2, 0.6],
                               'height_ratios': [1, 2]})

    sns.kdeplot(mst.envs_in_SubhaloVelDisp[mst._next.triple_mask], color='red', ax=ax[0][0], log_scale=True)
    sns.kdeplot(mst.envs_in_SubhaloVelDisp[~mst._next.triple_mask], color='blue', ax=ax[0][0], log_scale=True)

    # ax[0][1].plot(range(5), range(10, 5, -1))
    ax[1][0].scatter(mst.envs_in_SubhaloVelDisp[mst._next.triple_mask], mst.envs_SubhaloVelDisp[mst._next.triple_mask], color='red',s=1, alpha=0.2, label='triples')
    ax[1][0].scatter(mst.envs_in_SubhaloVelDisp[~mst._next.triple_mask], mst.envs_SubhaloVelDisp[~mst._next.triple_mask], color='blue',s=1, alpha=0.09, label='failed triple/binary')

    ax[1][0].set_ylim([10,600])
    ax[1][0].set_xlim([1,600])
    # ax[1][1].plot()

    ax[0][0].set_xlim([1,600])

    sns.kdeplot(mst.envs_SubhaloVelDisp[mst._next.triple_mask], color='red', ax = ax[1][1], vertical=True)
    sns.kdeplot(mst.envs_SubhaloVelDisp[~mst._next.triple_mask], color='blue', ax = ax[1][1], vertical=True)

    ax[1][1].set_ylim([10,600])

    ax[1][0].set_xlabel('envs_in_SubhaloVelDisp')
    ax[1][0].set_ylabel('envs_SubhaloVelDisp')
    ax[1][0].legend(loc='lower left')

    for ax in ax.flatten():
        ax.set_yscale('log')
        ax.set_xscale('log')
    plt.show()



    
    
    
        
#######################might be useful#######################
# tinsp = np.abs(sep/(dadt_df+dadt_lc+dadt_vd+dadt_gw))
# new_sep = mst.new_sep()
# interp_function = interp1d(sep, dadt_df+dadt_lc+dadt_vd+dadt_gw, bounds_error=False, fill_value=0)
# interp_tinsp = np.abs(new_sep/(dadt_df+dadt_lc+dadt_vd+dadt_gw))
# result = mst.find_tinsp(sep, dadt_df+dadt_lc+dadt_vd+dadt_gw, interpolated=True)
# tau = mst.find_tinsp(sep, dadt_df+dadt_lc+dadt_vd+dadt_gw)

# alpha=0.4
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.plot(sep, tinsp, '--',color='blue', label='$t=(\\frac{da}{dt})_i^{-1}a_i$', alpha=alpha)
# ax.plot(new_sep, interp_tinsp, '.',color='orange', markersize=3,label='interpolated $t=(\\frac{da}{dt})_i^{-1}a_i$', alpha=alpha)

# ax.plot(sep, result, '--',color='magenta',label='object quad integration $\\int(\\dot{a})^{-1}da$', alpha=alpha)
# ax.plot(sep, tau, '.', color='black', label='$\\sum(\\frac{da}{dt})^{-1}da$', markersize=3, alpha=alpha*0.8)

# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_xlabel('$a[pc]$')
# ax.set_ylabel('$t-scale\;/\; t_{\\rm insp}$')
# # plt.xlim(xmin=1e-4)
# ax.set_ylim(ymax=1e12)
# ax.legend(loc='best')
# plt.show()        

#-----------------------------clustering part--------------------------------------------

# from sklearn.cluster import KMeans
# sep_ovrtk_2nd = tmbh._next.sep_2nd_overtakes[tmbh._next.triple_mask]
# q = tmbh.q[tmbh._next.triple_mask]
# mtot = tmbh.masses[:,0]+tmbh.masses[:,1]; mtot=mtot[tmbh._next.triple_mask]
# cluster_array = np.vstack((np.log10(sep_ovrtk_2nd), np.log10(q), np.log10(mtot))).T
# n_clusters=3

# kmeans = KMeans(n_clusters=n_clusters)
# labels = kmeans.fit_predict(cluster_array)
# for i in np.unique(labels):
#     print (i)
#     plt.scatter(sep_ovrtk_2nd[labels==i], q[labels==i], s=1)
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:,0] , centroids[:,1] , s = 50, color = 'k')
# plt.yscale('log')
# plt.xscale('log')
# plt.title('Cluster in the log space with k={}'.format(n_clusters))
# plt.ylabel('$q$')
# plt.xlabel('$a_[pc]$')
# plt.show()

# print('from clustering:',np.median(sep_ovrtk_2nd[labels==0]), np.median(sep_ovrtk_2nd[labels==1]))
# print('from 100pc limit:',np.median(sep_ovrtk_2nd[sep_ovrtk_2nd>100]), np.median(sep_ovrtk_2nd[sep_ovrtk_2nd<100]))

# sse = {}
# for k in range(1, 10):
#     kmeans = KMeans(n_clusters=k, max_iter=1000).fit(cluster_array)
# #     data["clusters"] = kmeans.labels_
#     #print(data["clusters"])
#     sse[k] = kmeans.inertia_
# plt.plot(list(sse.keys()), list(sse.values()))
# plt.xlabel('number of clustera')
# plt.ylabel('Inertia')


# red_shift = mst._next.evol_z_2nd_overtakes
# print ('median redshift of the total triple population: {}'.format(np.median(red_shift[mst._next.triple_mask])))
# print ('median redshift of the strong triple population: {}'.format(np.median(red_shift[mst._next.f_mask])))
# print ('median redshift of the weak triple population: {}'.format(np.median(red_shift[mst._next.weak_triple_mask])))

# print (100*'-')
# #----------massratios----------
# mass_ratio = mst.q
# print ('median mass ratio of the total triple population: {}'.format(np.median(mass_ratio[mst._next.triple_mask])))
# print ('median mass ratio of the strong triple population: {}'.format(np.median(mass_ratio[mst._next.strong_triple_mask])))
# print ('median mass ratio of the weak triple population: {}'.format(np.median(mass_ratio[mst._next.weak_triple_mask])))
# print (100*'-')
# #----------total mass----------
# mass_tot = mst.masses[:,0]+mst.masses[:,1]
# print ('median mass ratio of the total triple population: {:e}'.format(np.median(mass_tot[mst._next.triple_mask])))
# print ('median mass ratio of the strong triple population: {:e}'.format(np.median(mass_tot[mst._next.strong_triple_mask])))
# print ('median mass ratio of the weak triple population: {:e}'.format(np.median(mass_tot[mst._next.weak_triple_mask])))

# print (mst.q[idx_not_merged_by_z0>0].size/mst.q.size)
# print ('Not merged by z=0:',mst.q[idx_not_merged_by_z0>0].size)
# print ('triples:',mst.q[mst.triple_mask].size)
# print ('Not merged by z=0 & triples:',mst.q[(idx_not_merged_by_z0>=0) & (mst._next.triple_mask)].size)