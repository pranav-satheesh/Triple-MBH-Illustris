import numpy as np
import matplotlib.pyplot as plt
import sys
import astro_constants as ac
from read_mbhb import mbhb_data, define_mbhb_inspiral_phases
from ms_settings import PATH, PATH_ICLOUD, FMBHB, MERGERS_FILE
from tqdm import tqdm
import pandas as pd



class tripleMatches(mbhb_data):
    """
    Finds the other binaries that might form a triple with each binary

    Because of the asymmetric nature of the merger tree, we get slightly 
    different results if we choose to loop thru mergers and identify previous 
    mergers involving those BHs, or if we loop thru mergers and identify 
    subsequent mergers involving those BHs. Thus, we define a nested class 
    MbhbOvertake that will perform the calculation in either 'match direction'.
    """

    def __init__(self, path, fmbhb, mergers, calculate_tinsp=False,**kwargs):
        """
        Initializes the tripleMatches class 
        
        Obtains mbhb inspiral data as a subclass and defines the match for both
        match directions via the nested class MbhbOvertake: 'prev' for the 
        previous mergers, and 'next' for the subsequent mergers

        Required arguments:
        path -- path where input and output files are located
        fmbhb -- name of file containing mbhb data
        mergers -- output loaded from npz file containing mergers data

        Returns: none
        """

        # this allows all the mbhb_data class vars to be defined here as self
        super().__init__(path, fmbhb, **kwargs) 
        #only take the valid inds which account to all mergers we 
        #consider (9234) note: val_inds come from hdf5 files for evolution
        #and mergers are from npz file "ill-1_blackhole_mergers_fixed.npz"
        self.merger_ids = mergers['ids'][self.val_inds]
        # Commenting out prev becuase it cannot register some mergers
#         self._prev = self.MbhbOvertake(self, 'prev')
        self._next = self.MbhbOvertake(self, 'next', calculate_tinsp)



        
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
        
        
    def calc_triple_categories(self, _mdir):
        """
        Calculates various quantities associated with the triple MBH population

        Required arguments:
        _mdir -- An instance of the nested class MbhbOvertake for a given match 
        direction (ie, tm._prev)
        
        Returns: none
        """

        # Number of binaries in which one or both BHs experience another merger:
        _mdir.N = _mdir.sim_tmrg_1st[_mdir.sim_tmrg_1st>0].size
        #n_with_subsequent = tm._next.sim_tmrg_2nd[tm._next.sim_tmrg_2nd>0].size # gives identical results
        # Number of binaries that merge by z=0 in which one or both BHs experience another merger:
        _mdir.N_zmrg1_gt0 = _mdir.sim_tmrg_1st[(_mdir.sim_tmrg_1st>0)&
                                               (_mdir.evol_mrg_1st_mask)].size
        _mdir.N_zmrg2_gt0 = _mdir.sim_tmrg_1st[(_mdir.sim_tmrg_1st>0)&
                                               (_mdir.evol_mrg_2nd_mask)].size
        #n_with_subsequent_zgt0 = tm._next.sim_tmrg_2nd[(tm._next.sim_tmrg_2nd>0)&(evol_mrg_mask)].size
        print("N: ", _mdir.N)
        print("N_zmrg1_gt0: ", _mdir.N_zmrg1_gt0)
        print("N_zmrg2_gt0: ", _mdir.N_zmrg2_gt0)

        ## Number of binaries that overlap with another binary (2nd forms before 1st merges)
        _mdir.N_overlapping_mbhb = _mdir.overlapping_mbhb_dt[(_mdir.overlap_mbhb_mask)].size
        ## Number of binaries that merge by z=0 and that overlap with another binary 
        ## (2nd forms before 1st merges)
        _mdir.N_overlapping_mbhb_zmrg1_gt0 = _mdir.overlapping_mbhb_dt[(_mdir.overlapping_mbhb_dt>0)&
                                                                       (_mdir.evol_mrg_1st_mask)].size
        _mdir.N_overlapping_mbhb_zmrg2_gt0 = _mdir.overlapping_mbhb_dt[(_mdir.overlapping_mbhb_dt>0)&
                                                                       (_mdir.evol_mrg_2nd_mask)].size

        ## Number of binaries in which 2nd mbhb merges after 1st mbhb
        _mdir.N_overlapping_merger = _mdir.overlapping_dtmrg[(_mdir.overlap_mrg_mask)].size
        _mdir.N_overlapping_merger_zmrg1_gt0 = _mdir.overlapping_dtmrg[(_mdir.overlap_mrg_mask)&
                                                                       (_mdir.evol_mrg_1st_mask)].size
        _mdir.N_overlapping_merger_zmrg2_gt0 = _mdir.overlapping_dtmrg[(_mdir.overlap_mrg_mask)&
                                                                       (_mdir.evol_mrg_2nd_mask)].size
        assert _mdir.N_overlapping_merger <= _mdir.N_overlapping_mbhb, \
            "Error: N_overlapping_merger ({}) > N_overlapping_mbhb ({})".format(_mdir.N_overlapping_merger,
                                                                                _mdir.N_overlapping_mbhb)
        
        ## Number of binaries that achieve smaller separation than a previous 
        ## binary at the same or earlier time (note that 2nd_overtakes can be 
        ## = 0 but 1st_overtaken cannot)
        _mdir.N_2nd_overtakes = _mdir.t_2nd_overtakes[(_mdir.overtakes_mask)].size
        Ntmp_1st_overtaken = _mdir.t_1st_overtaken[_mdir.t_1st_overtaken>0].size
        assert _mdir.N_2nd_overtakes == Ntmp_1st_overtaken, \
            "Error: N_2nd_overtakes ({}) != N_1st_overtaken ({})".format(_mdir.N_2nd_overtakes, 
                                                                         Ntmp_1st_overtaken)

        _mdir.N_2nd_overtakes_before_z0 = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)].size
        _mdir.N_1st_overtaken_before_z0 = _mdir.t_1st_overtaken[(_mdir.overtakes_mask)
                                                                &(_mdir.t_1st_overtaken<=self.tH)].size
        Ntmp = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                     &(_mdir.t_1st_overtaken>self.tH)].size
        assert Ntmp + _mdir.N_1st_overtaken_before_z0 == _mdir.N_2nd_overtakes_before_z0, \
            ("Error: mismatch in N overtakes/overtaken before z0: {}+{} != {}"
             .format(Ntmp, _mdir.N_1st_overtaken_before_z0, _mdir.N_2nd_overtakes_before_z0))

        _mdir.N_2nd_overtakes_before_z0_q1major = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                                                        &(_mdir.q_1st>0.2)].size
        _mdir.N_1st_overtaken_before_z0_q1major = _mdir.t_1st_overtaken[(_mdir.t_1st_overtaken>0)
                                                                        &(_mdir.t_1st_overtaken<=self.tH)
                                                                        &(_mdir.q_1st>0.2)].size
        Ntmp = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)&
                                     (_mdir.q_1st>0.2)&(_mdir.t_1st_overtaken>self.tH)].size
        assert (Ntmp + _mdir.N_1st_overtaken_before_z0_q1major 
                == _mdir.N_2nd_overtakes_before_z0_q1major), \
            ("Error: mismatch in N overtakes/overtaken before z0 q1major: {}+{} != {}"
             .format(Ntmp, _mdir.N_1st_overtaken_before_z0_q1major, 
                     _mdir.N_2nd_overtakes_before_z0_q1major))

        _mdir.N_2nd_overtakes_before_z0_q2major = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                                                        &(_mdir.q_2nd>0.2)].size
        _mdir.N_1st_overtaken_before_z0_q2major = _mdir.t_1st_overtaken[(_mdir.t_1st_overtaken>0)
                                                                        &(_mdir.t_1st_overtaken<=self.tH)
                                                                        &(_mdir.q_2nd>0.2)].size
        Ntmp = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                     &(_mdir.q_2nd>0.2)
                                     &(_mdir.t_1st_overtaken>self.tH)].size
        assert (Ntmp + _mdir.N_1st_overtaken_before_z0_q2major 
                == _mdir.N_2nd_overtakes_before_z0_q2major), \
            ("Error: mismatch in N overtakes/overtaken before z0 q2major: {}+{} != {}"
             .format(Ntmp, _mdir.N_1st_overtaken_before_z0_q2major, 
                     _mdir.N_2nd_overtakes_before_z0_q2major))

        _mdir.N_2nd_overtakes_before_z0_both_major = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                                                           &(_mdir.q_1st>0.2)
                                                                           &(_mdir.q_2nd>0.2)].size
        _mdir.N_1st_overtaken_before_z0_both_major = _mdir.t_1st_overtaken[(_mdir.t_1st_overtaken>0)
                                                                           &(_mdir.t_1st_overtaken<=self.tH)
                                                                           &(_mdir.q_1st>0.2)
                                                                           &(_mdir.q_2nd>0.2)].size
        Ntmp = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                     &(_mdir.t_1st_overtaken>self.tH)
                                     &(_mdir.q_1st>0.2)&(_mdir.q_2nd>0.2)].size
        assert (Ntmp + _mdir.N_1st_overtaken_before_z0_both_major 
                == _mdir.N_2nd_overtakes_before_z0_both_major), \
            ("Error: mismatch in N overtakes/overtaken before z0 both major: {}+{} != {}"
             .format(Ntmp, _mdir.N_1st_overtaken_before_z0_both_major, 
                     _mdir.N_2nd_overtakes_before_z0_both_major))

        ## Number of binaries that merge by z=0 and achieve smaller separation than a previous binary at the same or earlier time
        _mdir.N_2nd_overtakes_zmrg1_gt0 = _mdir.t_2nd_overtakes[(_mdir.overtakes_mask)
                                                                &(_mdir.evol_mrg_1st_mask)].size
        _mdir.N_2nd_overtakes_zmrg2_gt0 = _mdir.t_2nd_overtakes[(_mdir.overtakes_mask)
                                                                &(_mdir.evol_mrg_2nd_mask)].size
        _mdir.N_2nd_overtakes_before_z0_zmrg1_gt0 = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                                                          &(_mdir.evol_mrg_1st_mask)].size
        _mdir.N_2nd_overtakes_before_z0_zmrg2_gt0 = _mdir.t_2nd_overtakes[(_mdir.overtakes_before_z0_mask)
                                                                          &(_mdir.evol_mrg_2nd_mask)].size
        ## sanity checks
        Ntmp_1st_overtaken_zmrg1_gt0 = _mdir.t_1st_overtaken[(_mdir.t_1st_overtaken>0)
                                                             &(_mdir.evol_mrg_1st_mask)].size
        assert _mdir.N_2nd_overtakes_zmrg1_gt0 == Ntmp_1st_overtaken_zmrg1_gt0, \
            ("Error: mismatch in N overtakes/overtaken with zmrg1>0: {} {}"
             .format(_mdir.N_2nd_overtakes_zmrg1_gt0, Ntmp_1st_overtaken_zmrg1_gt0))
        Ntmp_1st_overtaken_zmrg2_gt0 = _mdir.t_1st_overtaken[(_mdir.t_1st_overtaken>0)
                                                             &(_mdir.evol_mrg_2nd_mask)].size
        assert _mdir.N_2nd_overtakes_zmrg2_gt0 == Ntmp_1st_overtaken_zmrg2_gt0, \
            ("Error: mismatch in N overtakes/overtaken with zmrg2>0: {} {}"
             .format(_mdir.N_2nd_overtakes_zmrg2_gt0, Ntmp_1st_overtaken_zmrg2_gt0))
        _mdir.N_1st_overtaken_before_z0_zmrg1_gt0 = _mdir.t_2nd_overtakes[(_mdir.t_1st_overtaken>0)
                                                                          &(_mdir.t_1st_overtaken<=self.tH)
                                                                          &(_mdir.evol_mrg_1st_mask)].size
        _mdir.N_1st_overtaken_before_z0_zmrg2_gt0 = _mdir.t_2nd_overtakes[(_mdir.t_1st_overtaken>0)
                                                                          &(_mdir.t_1st_overtaken<=self.tH)
                                                                          &(_mdir.evol_mrg_2nd_mask)].size
                                                                        
        ## Number of binaries that overlap with a previous binary but don't overtake it
        _mdir.N_chase_overlapping_mbhb = _mdir.overlapping_mbhb_dt[(_mdir.chase_mask)].size
        ## Number of binaries that overlap with a previous binary but don't overtake it, and still merge before z=0
        _mdir.N_chase_overlapping_mbhb_zmrg1_gt0 = _mdir.overlapping_mbhb_dt[(_mdir.chase_mask)
                                                                             &(_mdir.evol_mrg_1st_mask)].size
        _mdir.N_chase_overlapping_mbhb_zmrg2_gt0 = _mdir.overlapping_mbhb_dt[(_mdir.chase_mask)
                                                                             &(_mdir.evol_mrg_2nd_mask)].size

        _mdir.N_triple = _mdir.t_2nd_overtakes[(_mdir.triple_mask)].size
        _mdir.N_failed_triple = _mdir.t_2nd_overtakes[(_mdir.failed_triple_mask)].size
        #saeb added
        _mdir.N_strong_triple = _mdir.t_2nd_overtakes[(_mdir.strong_triple_mask)].size
        _mdir.N_weak_triple = _mdir.t_2nd_overtakes[(_mdir.weak_triple_mask)].size
        
        print ('number of triples {}'.format(_mdir.N_triple))
        print ('number of failed triples {}'.format(_mdir.N_failed_triple))        
        print ('number of strong triples {}'.format(_mdir.N_strong_triple))
        print ('number of weak triples {}'.format(_mdir.N_weak_triple))
            
   
    ########
    ## start of nested class
    ######## 
    class MbhbOvertake(object):
        """
        Class for triple systems matched by looping thru and finding the adjacent binary

        Allows for matching either by the previous adjacent or next adjacent binary involving
        a BH in the given binary
        """

        def __init__(self, tm_instance, match_direction, calculate_tinsp,**kwargs):
            """
            Initializes the MbhbOvertake class

            Required arguments:
            tm_instance -- allows the nested class to inherit variables from the parent class
            match_direction -- must be 'prev' or 'next', according to the direction in which matching
            to adjacent binaries will be performed
            
            Returns: none
            """

            self.tm_instance = tm_instance
            self.tH = self.tm_instance.tH
            self.Nmbhb = self.tm_instance.Nmbhb
            print("Nmbhb in nested class: ", self.Nmbhb)
            self.merger_ids = self.tm_instance.merger_ids
            self.sim_tmrg = self.tm_instance.sim_tmrg
            self.evol_tmrg = self.tm_instance.evol_tmrg
            self.times = self.tm_instance.times
            self.scales = self.tm_instance.scales
            self.sep = self.tm_instance.sep
            self.masses = self.tm_instance.masses #masses 
            self.q = self.tm_instance.q
            self.evol_z = self.tm_instance.evol_z
            if calculate_tinsp:
                self.total_tinsp_integrated = self.tm_instance.time_scales()


            ################################################ 
            # The most recent prior Illustris binary formation if one exists for either BH in a given Illustris binary
            self.ix_1st_mbhb = -np.ones(self.Nmbhb).astype('int') * 10**12
            # The next subsequent Illustris binary formation if one exists for either BH in a given Illustris binary
            self.ix_2nd_mbhb = -np.ones(self.Nmbhb).astype('int') * 10**12
            
            # If the next binary's separation becomes smaller than this binary's separation,
            # these are set to the time and separation where that occurs in the next binary's evolution
            self.ixsep_2nd_overtakes = -np.ones(self.Nmbhb).astype('int')*10**12
            self.t_2nd_overtakes = -np.ones(self.Nmbhb)
            self.scales_2nd_overtakes = -np.ones(self.Nmbhb)
            self.sep_2nd_overtakes = -np.ones(self.Nmbhb)
            self.evol_z_2nd_overtakes = -np.ones(self.Nmbhb)
            self.integrated_tinsp_2nd_overtakes = -np.ones(self.Nmbhb)
            
            # If the next binary's separation becomes smaller than this binary's separation,
            # these are set to the time and separation where that occurs in this binary's evolution
            self.ixsep_1st_overtaken = -np.ones(self.Nmbhb).astype('int')*10**12
            self.t_1st_overtaken = -np.ones(self.Nmbhb)
            self.scales_1st_overtaken = -np.ones(self.Nmbhb)
            self.sep_1st_overtaken = -np.ones(self.Nmbhb)
            self.evol_z_1st_overtaken = -np.ones(self.Nmbhb)
            self.integrated_tinsp_1st_overtaken = -np.ones(self.Nmbhb)
            ################################################ 
            
            for i in range(self.Nmbhb):
                
                if match_direction == 'next':
                    ix0n = np.where((self.merger_ids[i,0]==self.merger_ids[:,0])|
                                    (self.merger_ids[i,0]==self.merger_ids[:,1]))[0]
                    ix1n = np.where((self.merger_ids[i,1]==self.merger_ids[:,0])|
                                    (self.merger_ids[i,1]==self.merger_ids[:,1]))[0]
                    ix0_next = 10**12
                    ix1_next = 10**12
                    if len(ix0n)>0 and ix0n.max()>i: ix0_next = ix0n[ix0n>i].min()
                    if len(ix1n)>0 and ix1n.max()>i: ix1_next = ix1n[ix1n>i].min()
                    if ix0_next < 1e12 or ix1_next < 1e12:
                        self.ix_2nd_mbhb[i] = np.minimum(ix0_next, ix1_next)
                        self.ix_1st_mbhb[i] = i

                elif match_direction == 'prev':
                        ix0l = np.where((self.merger_ids[i,0]==self.merger_ids[:i,0])|
                                        (self.merger_ids[i,0]==self.merger_ids[:i,1]))[0]
                        ix1l = np.where((self.merger_ids[i,1]==self.merger_ids[:i,0])|
                                        (self.merger_ids[i,1]==self.merger_ids[:i,1]))[0]
                        ix0_last = ix0l.max() if len(ix0l) > 0 else -1
                        ix1_last = ix1l.max() if len(ix1l) > 0 else -1
                        if ix0_last > 0 or ix1_last > 0:
                            self.ix_1st_mbhb[i] = np.maximum(ix0_last, ix1_last)
                            self.ix_2nd_mbhb[i] = i

                else:
                    print("match direction must be 'prev' or 'next'.")
                    sys.exit()

                # At least one BH in this binary undergoes another merger at some point
                if self.ix_1st_mbhb[i] > 0 and self.ix_2nd_mbhb[i] > 0:

                    assert self.ix_2nd_mbhb[i] > self.ix_1st_mbhb[i], \
                        "Error: ix_2nd_mbhb={}, ix_1st_mbhb={}".format(self.ix_2nd_mbhb[i], 
                                                                       self.ix_1st_mbhb[i])

                    # The 2nd binary forms before the 1st binary merges
                    if self.sim_tmrg[self.ix_2nd_mbhb[i]] < self.evol_tmrg[self.ix_1st_mbhb[i]]:


                        if match_direction == 'prev':
                            # find index in the 1st (prev) mbhb array w/ closest time to each time 
                            # in the 2nd (current) mbhb array (requires looping thru 1st mbhb times)
                            for j in range(len(self.times[self.ix_1st_mbhb[i], :])):
                                if self.times[self.ix_1st_mbhb[i], j] == 0: continue
                
                                # we want to find the closest time in the previous merger array that's still
                                # slightly later than this merger (to make sure it's actually overtaken)
                                tdiff_last = (self.times[self.ix_1st_mbhb[i], j] - self.times[i,:])
                                if tdiff_last[(self.times[i,:]>0)&(tdiff_last>=0)].size == 0: continue
                                min_tdiff_last = np.min(tdiff_last[(self.times[i,:]>0)&(tdiff_last>=0)])
                                ixsep_last_closest_t_in_current = np.where(tdiff_last==min_tdiff_last)[0]

                                # Define the point in each mbhb inspiral where the 2nd overtakes the 1st
                                if (self.sep[i,ixsep_last_closest_t_in_current].min() 
                                    < self.sep[self.ix_1st_mbhb[i], j]):

                                    tmpix_last = np.where(self.sep[i,ixsep_last_closest_t_in_current]
                                                          < self.sep[self.ix_1st_mbhb[i], j])
                                    self.ixsep_2nd_overtakes[i] = ixsep_last_closest_t_in_current[tmpix_last][0]
                                    self.ixsep_1st_overtaken[i] = j
                                    break
                                else:
                                    # not overtaken at this tstep
                                    continue

                        else:
                            # find index in the 1st (current) mbhb array w/ closest time to each time 
                            # in the 2nd (next) mbhb array (requires looping thru 2nd mbhb times)
                            for k in range(len(self.times[self.ix_2nd_mbhb[i], :])):
                                if self.times[self.ix_2nd_mbhb[i], k] == 0: continue
                            
                                #ixsep_2nd_closest_t_in_1st = np.where(tdiff_next==min_tdiff_next)[0]
                                tdiff_next = ( self.times[i,:] - self.times[self.ix_2nd_mbhb[i], k])
                                if tdiff_next[(self.times[i,:]>0)&(tdiff_next>=0)].size == 0: continue
                                min_tdiff_next = np.min(tdiff_next[(self.times[i,:]>0)&(tdiff_next>=0)])
                                #np.argmin 
                                ixsep_next_closest_t_in_current = np.where(tdiff_next==min_tdiff_next)[0]
                                
                                # Define the point in each mbhb inspiral where the 2nd overtakes the 1st
                                if (self.sep[self.ix_2nd_mbhb[i], k] 
                                    < self.sep[i, ixsep_next_closest_t_in_current]).max():

                                    self.ixsep_2nd_overtakes[i] = k
                                    tmpix_next = np.where(self.sep[self.ix_2nd_mbhb[i],k] < 
                                                          self.sep[i, ixsep_next_closest_t_in_current])
                                    self.ixsep_1st_overtaken[i] = ixsep_next_closest_t_in_current[tmpix_next][0]
                                    break
                                else:
                                    # not overtaken at this tstep
                                    continue

                    if self.ixsep_2nd_overtakes[i] >= 0:
                        assert self.ixsep_1st_overtaken[i] > 0, \
                            "error: 2nd overtakes but 1st not overtaken? {} {}".format(self.ixsep_2nd_overtakes[i],
                                                                                       self.ixsep_1st_overtaken[i])
                        self.t_2nd_overtakes[i] = self.times[self.ix_2nd_mbhb[i], 
                                                             self.ixsep_2nd_overtakes[i]]
                        self.evol_z_2nd_overtakes[i] = self.evol_z[self.ix_2nd_mbhb[i], 
                                                             self.ixsep_2nd_overtakes[i]]
                        self.scales_2nd_overtakes[i] = self.scales[self.ix_2nd_mbhb[i], 
                                                             self.ixsep_2nd_overtakes[i]]
                        self.sep_2nd_overtakes[i] = self.sep[self.ix_2nd_mbhb[i], 
                                                             self.ixsep_2nd_overtakes[i]]
                        self.t_1st_overtaken[i] = self.times[self.ix_1st_mbhb[i], 
                                                             self.ixsep_1st_overtaken[i]]
                        self.evol_z_1st_overtaken[i] = self.evol_z[self.ix_1st_mbhb[i], 
                                                             self.ixsep_1st_overtaken[i]] 
                        self.scales_1st_overtaken[i] = self.scales[self.ix_1st_mbhb[i], 
                                                             self.ixsep_1st_overtaken[i]]
                        self.sep_1st_overtaken[i] = self.sep[self.ix_1st_mbhb[i], 
                                                             self.ixsep_1st_overtaken[i]]
                        
                        #ms added
                        if calculate_tinsp:
                            self.integrated_tinsp_1st_overtaken[i] = self.total_tinsp_integrated[self.ix_1st_mbhb[i], 
                                                             self.ixsep_1st_overtaken[i]]
                            self.integrated_tinsp_2nd_overtakes[i] = self.total_tinsp_integrated[self.ix_2nd_mbhb[i], 
                                                                 self.ixsep_2nd_overtakes[i]]                            

                
            ## end of loop over all mergers

            assert np.min(self.ixsep_1st_overtaken / self.ixsep_2nd_overtakes) >= 0, \
                "Found instance(s) where ixsep_1st_overtaken / ixsep_2nd_overtakes < 0"

            # The time when the subsequent Illustris binary forms
            self.sim_tmrg_1st = -np.ones(self.Nmbhb)
            self.sim_tmrg_1st[self.ix_1st_mbhb>=0] = self.sim_tmrg[self.ix_1st_mbhb[self.ix_1st_mbhb>=0]]
            self.sim_tmrg_2nd = -np.ones(self.Nmbhb)
            self.sim_tmrg_2nd[self.ix_2nd_mbhb>=0] = self.sim_tmrg[self.ix_2nd_mbhb[self.ix_2nd_mbhb>=0]]
            # The actual BH merger time of the next Illustris binary 
            self.evol_tmrg_1st = -np.ones(self.Nmbhb)
            self.evol_tmrg_1st[self.ix_1st_mbhb>=0] = self.evol_tmrg[self.ix_1st_mbhb[self.ix_1st_mbhb>=0]]
            self.evol_tmrg_2nd = -np.ones(self.Nmbhb)
            self.evol_tmrg_2nd[self.ix_2nd_mbhb>=0] = self.evol_tmrg[self.ix_2nd_mbhb[self.ix_2nd_mbhb>=0]]
        
            self.q_1st = -np.ones(self.Nmbhb)    
            self.q_1st[self.ix_1st_mbhb>=0] = self.q[self.ix_1st_mbhb[self.ix_1st_mbhb>=0]]

            self.masses1 = -np.ones((self.Nmbhb,2))
            self.masses2 = -np.ones((self.Nmbhb,2))

            self.masses1[self.ix_1st_mbhb>=0] = self.masses[self.ix_1st_mbhb[self.ix_1st_mbhb>=0]]
            self.masses2[self.ix_2nd_mbhb>=0] = self.masses[self.ix_2nd_mbhb[self.ix_2nd_mbhb>=0]] 
         

            self.q_2nd = -np.ones(self.Nmbhb)    
            self.q_2nd[self.ix_2nd_mbhb>=0] = self.q[self.ix_2nd_mbhb[self.ix_2nd_mbhb>=0]]
    
            # If the next binary *forms* before this binary *merges*, this
            # is set to the overlap time (the time when two binaries simultaneously exist involving the same BH)
            self.overlapping_mbhb_dt = np.zeros(self.Nmbhb)
            tmp_tmrg = np.array([np.minimum(self.evol_tmrg_2nd[n], self.evol_tmrg_1st[n]) 
                                 for n in range(self.Nmbhb)])
            self.overlapping_mbhb_dt[self.sim_tmrg_2nd>0] = ( tmp_tmrg[self.sim_tmrg_2nd>0] - 
                                                              self.sim_tmrg_2nd[self.sim_tmrg_2nd>0] )

            # If two binaries involving the same BH exist simultaneously, this 
            # is set to the difference b/t the merger time of the next binary and this binary
            # If the next binary *merges* before this binary *merges*, this quantity is > 0
            # If the next binary merges *after* this binary merges, this quantity is < 0
            self.overlapping_dtmrg = np.zeros(self.Nmbhb)
            self.overlapping_dtmrg[self.evol_tmrg_2nd>0] = ( self.evol_tmrg_1st[self.evol_tmrg_2nd>0] - 
                                                             self.evol_tmrg_2nd[self.evol_tmrg_2nd>0] )

            print("min/max/size sim_tmrg_1st: ", self.sim_tmrg_1st.min(), 
                  self.sim_tmrg_1st.max(), self.sim_tmrg_1st.size) 
            print("min/max/size evol_tmrg_1st: ", self.evol_tmrg_1st.min(), 
                  self.evol_tmrg_1st.max(), self.evol_tmrg_1st.size) 
            print("min/max/size sim_tmrg_2nd: ", self.sim_tmrg_2nd.min(), 
                  self.sim_tmrg_2nd.max(), self.sim_tmrg_2nd.size) 
            print("min/max/size evol_tmrg_2nd: ", self.evol_tmrg_2nd.min(), 
                  self.evol_tmrg_2nd.max(), self.evol_tmrg_2nd.size) 

            assert (self.ixsep_1st_overtaken[self.overlapping_dtmrg>0].all() >= 0 and 
                    self.ixsep_2nd_overtakes[self.overlapping_dtmrg>0].all() >= 0), \
                "Found merger that should have been overtaken by subsequent merger, but ixsep_overtaken and/or ixsep_overtakes < 0"


            ## Define masks for various triple categories
            self.evol_mrg_1st_mask = ((self.evol_tmrg_1st>0)&(self.evol_tmrg_1st<=self.tH))
            self.evol_mrg_2nd_mask = ((self.evol_tmrg_2nd>0)&(self.evol_tmrg_2nd<=self.tH))

            self.overlap_mbhb_mask = (self.overlapping_mbhb_dt>0)
            self.overlap_mrg_mask = (self.overlapping_dtmrg>0)
            
            self.overtakes_mask = (self.t_2nd_overtakes>=0)
            self.overtakes_before_z0_mask = ((self.overtakes_mask)&(self.t_2nd_overtakes<=self.tH))
            self.chase_mask = ((self.overlapping_mbhb_dt>0)&(self.t_2nd_overtakes<0))
            
            self.triple_mask = self.overtakes_before_z0_mask
            self.failed_triple_mask = ((self.t_2nd_overtakes>self.tH)|(self.chase_mask))
            
            #Note: for sub-populations such as strong triple, weak triples we use 2nd binaries parameters (seps)
            self.strong_triple_mask = (self.sep_2nd_overtakes>0) & (self.sep_2nd_overtakes<100) & (self.triple_mask)
            self.weak_triple_mask = (self.sep_2nd_overtakes>0) & (self.sep_2nd_overtakes>100) & (self.triple_mask)



def fmt_print_fractions(lblstr, N, Nzgt0, Nsubcat, Ntot):
    """
    Provides formatting strings for printing info about triple BH systems

    Required arguments:
    lblstr -- the preceding string that indicates which number (of binaries 
    in a given subcategory) is being printed
    N -- the number of binaries in the relevant subcategory
    Nzgt0 -- the number of binaries in this category that merge before z=0
    Nsubcat -- a parent subcategory (ie, binaries that have subsequent/previous mergers)
    Ntot -- the total number of binaries in this simulation

    Returns: none

    Currently used only in (calc_triple_defns)
    """

    print("\n{}:\nN={} ({:.4g}% of subseq., {:.4g}% of tot)".format(lblstr, N, N/Nsubcat*100,  
                                                                    N/Ntot*100))
    if Nzgt0:
        print("N(merge z>0)={} ({:.4g}% of subseq., {:.4g}% of tot)"
              .format(Nzgt0, Nzgt0/Nsubcat*100,  Nzgt0/Ntot*100))


#file_list = dict(fid_e06='mbhb-evolution_fiducial_ecc-evo-0.6_lc-shm06.hdf5',
#                 fid_e00='mbhb-evolution_mdot_ecc-evo-0.00_lc-shm06.hdf5',
#                 fid_e09_mdot001='mbhb-evolution_mdot_ecc-evo-0.9_lc-shm06_0.01mdot.hdf5',
#                 frefill00='mbhb-evolution_no-ecc_lc-full-0.00.hdf5',
#                 frefill02='mbhb-evolution_no-ecc_lc-full-0.2.hdf5',
#                 frefill06='mbhb-evolution_no-ecc_lc-full-0.6.hdf5',
#                 frefill099='mbhb-evolution_no-ecc_lc-full-0.99.hdf5')

def calc_triple_defns(path=PATH_ICLOUD, #'/Users/sayebms1/mybitbucket/lb_python_scripts/input/'
                      fmrg='ill-1_blackhole_mergers_fixed.npz', fmbhb='frefill06',
                      makeplot=True, verbose=False):
    """
    Wrapper function for basic use of the tripleMatch class

    Required arguments: none

    Optional keyword arguments:
    path ['/Users/lblecha/Dropbox/mbhb_evolution/'] -- the location of input and output files
    fmrg ['ill-1_blackhole_mergers_fixed.npz'] -- name of mergers file
    fmbhb ['frefill06'] -- name or dictionary key of mbhb inspiral file
    makeplot [True] -- if True, plot_triple_data is called
    verbose [False] -- prints extra output to screen

    Returns:
    tm -- the tripleMatch class instance

    Instantiates the tripleMatch class, performs calculations of triple categories, 
    and optionally makes basic plots of the triples data
    """
    
    mergers = np.load(path+fmrg)

    tm = tripleMatches(path, fmbhb, mergers, calculate_tinsp=False)
    if verbose:
#         print("tm._prev.ix_1st_mbhb.min()=",tm._prev.ix_1st_mbhb.min())
        print("tm._next.ix_1st_mbhb.min()=",tm._next.ix_1st_mbhb.min())


#     tm.calc_triple_categories(tm._prev)
#     if verbose:
#         print("after call to calc_triple_categories (_prev):")
#         print(tm._prev.N)
#         print(tm._prev.N_zmrg1_gt0)
#         print(tm._prev.N_zmrg2_gt0)
    tm.calc_triple_categories(tm._next)
    if verbose:
        print("after call to calc_triple_categories (_next):")
        print(tm._next.N)
        print(tm._next.N_zmrg1_gt0)
        print(tm._next.N_zmrg2_gt0)

    if verbose:
        #d = mbhb_data(path+file_list[f], verbose=False)

        # these are the indices of the mergers in the merger file
        # test to make sure it's working properly:
        print(tm.val_inds.min(), tm.val_inds.max(), tm.val_inds.shape)
        print(tm.val_inds[0])
        print(tm.masses.shape)
        print("sep shape:", tm.sep.shape)
        print("{:.2e} {:.2e}".format(tm.masses[0,0],tm.masses[0,1]))
        print("{:.2e} {:.2e}".format(*mergers['masses'][tm.val_inds[0]]*1e10/tm.pars.h))

        print("min/max/med BH mass ratio: {} / {} / {}".format(tm.q.min(), tm.q.max(), 
                                                               np.median(tm.q)))

        print("min/max/mean/median _next.q_1st: {} {} {} {}"
              .format(tm._next.q_1st.min(),tm._next.q_1st.max(), 
                      tm._next.q_1st.mean(), np.median(tm._next.q_1st)))
        print("min/max/mean/median _next.q_2nd: {} {} {} {}"
              .format(tm._next.q_2nd.min(),tm._next.q_2nd.max(), 
                      tm._next.q_2nd.mean(), np.median(tm._next.q_2nd)))


    n_zgt0 = tm.evol_tmrg[tm.evol_mrg_mask].size

    if verbose:    
        print("n_with_subsequent:",tm._next.N)
        print("n_with_subsequent, zmrg1>0:",tm._next.N_zmrg1_gt0)
        print("min/max/size next_overlapping_mbhb_dt:",tm._next.overlapping_mbhb_dt.min(),tm._next.overlapping_mbhb_dt.max())
        print("min/max/size next_overlapping_merger_dt:", 
              tm._next.overlapping_dtmrg.min(), tm._next.overlapping_dtmrg.max())

        print("\n Of {} total binaries, {} ({:4g}% of tot) merge before z=0."
              .format(tm.Nmbhb, n_zgt0, n_zgt0/tm.Nmbhb*100))

        print("\n*** BINARIES WITH PREVIOUS MERGER ***")
        print("\nNumber of binaries in which one or both BHs experienced a previous merger:")
        print("n_with_previous={} ({:.4g}% of total)"
              .format(tm._prev.N, tm._prev.N/tm.Nmbhb*100))
        print("Number of binaries that merge by z=0 in which one or both BHs experienced a previous merger:")
        print("n_with_previous_zgt0={} ({:.4g}% of total)"
              .format(tm._prev.N_zmrg2_gt0, tm._prev.N_zmrg2_gt0/tm.Nmbhb*100))

        fmt_print_fractions("# of binaries in which a previous binary *merges* after this binary *forms*",
                            tm._prev.N_overlapping_mbhb, tm._prev.N_overlapping_mbhb_zmrg2_gt0, 
                            tm._prev.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries in which a previous binary *merges* after this binary *merges*",
                            tm._prev.N_overlapping_merger, tm._prev.N_overlapping_merger_zmrg2_gt0, 
                            tm._prev.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries that overtake a previous binary",
                            tm._prev.N_2nd_overtakes, tm._prev.N_2nd_overtakes_zmrg2_gt0, 
                            tm._prev.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries that overtake a previous binary before z=0",
                            tm._prev.N_2nd_overtakes_before_z0, 
                            tm._prev.N_2nd_overtakes_before_z0_zmrg2_gt0,
                            tm._prev.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries that overtake a previous binary with q1>0.2 before z=0",
                            tm._prev.N_2nd_overtakes_before_z0_q1major, None, tm._prev.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries with q2>0.2 that overtake a previous binar before z=0",
                            tm._prev.N_2nd_overtakes_before_z0_q2major, None, tm._prev.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries with q>0.2 that overtake a previous binary with q>0.2 before z=0",
                            tm._prev.N_2nd_overtakes_before_z0_both_major, None, tm._prev.N, tm.Nmbhb)

        fmt_print_fractions("\n# of binaries that overlap with a previous binary but don't overtake it",
                            tm._prev.N_chase_overlapping_mbhb, 
                            tm._prev.N_chase_overlapping_mbhb_zmrg2_gt0, 
                            tm._prev.N, tm.Nmbhb)

        print("\n*** BINARIES WITH SUBSEQUENT MERGER ***")
        print("\n# of binaries in which one or both BHs experiences a subsequent merger:")
        print("n_with_subsequent={} ({:.4g}% of total)"
              .format(tm._next.N, tm._next.N/tm.Nmbhb*100))
        print("# of binaries that merge by z=0 in which one or both BHs experiences a subsequent merger:")
        print("n_with_subsequent_zgt0={} ({:.4g}% of total)"
              .format(tm._next.N_zmrg1_gt0, tm._next.N_zmrg1_gt0/tm.Nmbhb*100))
    

        fmt_print_fractions("# of binaries in which a subsequent binary *forms* before this binary *merges*",
                            tm._next.N_overlapping_mbhb, 
                            tm._next.N_overlapping_mbhb_zmrg1_gt0, 
                            tm._next.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries in which a subsequent binary *merges* before this binary *merges*",
                            tm._next.N_overlapping_merger, 
                            tm._next.N_overlapping_merger_zmrg1_gt0, 
                            tm._next.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries overtaken by a subsequent binary",
                            tm._next.N_2nd_overtakes, 
                            tm._next.N_2nd_overtakes_zmrg1_gt0, 
                            tm._next.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries overtaken by a subsequent binary before z=0",
                            tm._next.N_1st_overtaken_before_z0, 
                            tm._next.N_1st_overtaken_before_z0_zmrg1_gt0, 
                            tm._next.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries with q>0.2 overtaken by a subsequent binary before z=0",
                            tm._next.N_1st_overtaken_before_z0_q1major, None, tm._next.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries overtaken by a subsequent binary with q2>0.2 before z=0",
                            tm._next.N_1st_overtaken_before_z0_q2major, None, tm._next.N, tm.Nmbhb)
        fmt_print_fractions("# of binaries with q>0.2 overtaken by a subsequent binary with q>0.2 before z=0",
                            tm._next.N_1st_overtaken_before_z0_both_major, None, tm._next.N, tm.Nmbhb)
        fmt_print_fractions("Sanity check: # of subsequent binaries that overtake a current binary",
                            tm._next.N_2nd_overtakes, 
                            tm._next.N_2nd_overtakes_zmrg1_gt0, 
                            tm._next.N, tm.Nmbhb)

        fmt_print_fractions("\n# of binaries that overlap with a subsequent binary but are not overtaken",
                            tm._next.N_chase_overlapping_mbhb, 
                            tm._next.N_chase_overlapping_mbhb_zmrg1_gt0, 
                            tm._next.N, tm.Nmbhb)
    
    if makeplot: plot_triple_data(path=path, fmrg=fmrg, fmbhb=fmbhb, tm=tm)

    return tm


def do_all_inspiral_models(path='/Users/pranavsatheesh/Triples/bitbucket/Data',#/Users/sayebms1/mybitbucket/lb_python_scripts/input/',#'/Users/lblecha/Dropbox/mbhb_evolution/', 
                           fmrg='ill-1_blackhole_mergers_fixed.npz', 
                           calc=False, makeplot=True, param='frefill', **kwargs):
    """
    This is the main script in this file. will calculate a bunch of numbers
    about the triple population (using calc_triple_defns) and/or make some 
    basic plots (using plot_triple_data).
    
    Usage: ftm.do_all_inspiral_models(path='/Users/sayebms1/mybitbucket/lb_python_scripts/input/',#'/Users/lblecha/Dropbox/mbhb_evolution/', 
                           fmrg='ill-1_blackhole_mergers_fixed.npz', 
                           calc=True, makeplot=True, param='frefill')
    """
    

    if calc==False and makeplot==False:
        print("nothing to do here! calc &/or makeplot must be True")
        return
    if param=='frefill':

        mbhb_file_list = dict(frefill06 = 'mbhb-evolution_no-ecc_lc-full-0.6.hdf5')
        #mbhb_file_list = dict(frefill00='mbhb-evolution_no-ecc_lc-full-0.00.hdf5',
                              #frefill06='mbhb-evolution_no-ecc_lc-full-0.6.hdf5',
                              #frefill099='mbhb-evolution_no-ecc_lc-full-0.99.hdf5')

    elif param=='eccen':
        mbhb_file_list = dict(eccen01='new_mbhb-evolution_ecc-evo-0.1_lc-shm06.hdf5',
                              eccen05='new_mbhb-evolution_ecc-evo-0.5_lc-shm06.hdf5',
                              eccen09='new_mbhb-evolution_ecc-evo-0.9_lc-shm06.hdf5',
                              eccen099='new_mbhb-evolution_ecc-evo-0.99_lc-shm06.hdf5'
                              )
    
    mbhb_frefill_arr = np.array([0.6])
    #mbhb_frefill_arr = np.array([0.0, 0.6, 0.99])
    mbhb_ecc_arr = np.array([0.1, 0.5, 0.9, 0.99])        
    ### this includes the files that do not have 'val_inds' 
    #mbhb_file_list = dict(fid_e06='mbhb-evolution_fiducial_ecc-evo-0.6_lc-shm06.hdf5',
    #                 fid_e00='mbhb-evolution_mdot_ecc-evo-0.00_lc-shm06.hdf5',
    #                 fid_e09_mdot001='mbhb-evolution_mdot_ecc-evo-0.9_lc-shm06_0.01mdot.hdf5',
    #                 frefill00='mbhb-evolution_no-ecc_lc-full-0.00.hdf5',
    #                 frefill02='mbhb-evolution_no-ecc_lc-full-0.2.hdf5',
    #                 frefill06='mbhb-evolution_no-ecc_lc-full-0.6.hdf5',
    #                 frefill099='mbhb-evolution_no-ecc_lc-full-0.99.hdf5')

    if calc:
        Nmbhb_arr = np.array([])
#         prev_Noverlap_arr = np.array([])
#         prev_Ntriple_arr = np.array([])
#         prev_Nfailed_arr = np.array([])
        next_Noverlap_arr = np.array([])
        next_Ntriple_arr = np.array([])
        next_Nfailed_arr = np.array([])
        #saeb
        next_Nstrong_trip_arr = np.array([])
        next_Nweak_trip_arr = np.array([])
        

    for fname in mbhb_file_list.values():

        if not calc:
            if makeplot: plot_triple_data(path=path, fmrg=fmrg, fmbhb=fname)
        else: 
            tm = calc_triple_defns(path=path, fmrg=fmrg, fmbhb=fname, makeplot=makeplot)
            
            Nmbhb_arr = np.append(Nmbhb_arr, tm.Nmbhb)
#             prev_Noverlap_arr = np.append(prev_Noverlap_arr, tm._prev.N_overlapping_mbhb)
#             prev_Ntriple_arr = np.append(prev_Ntriple_arr, tm._prev.N_triple)
#             prev_Nfailed_arr = np.append(prev_Nfailed_arr, tm._prev.N_failed_triple)
            next_Noverlap_arr = np.append(next_Noverlap_arr, tm._next.N_overlapping_mbhb)
            next_Ntriple_arr = np.append(next_Ntriple_arr, tm._next.N_triple)
            next_Nfailed_arr = np.append(next_Nfailed_arr, tm._next.N_failed_triple)
            #saeb            
            next_Nstrong_trip_arr = np.append(next_Nstrong_trip_arr, tm._next.N_strong_triple )
            next_Nweak_trip_arr = np.append(next_Nweak_trip_arr, tm._next.N_weak_triple)


    if calc:
#         assert prev_Noverlap_arr.all() == (prev_Ntriple_arr+prev_Nfailed_arr).all(), \
#             "mismatch in prev_Noverlap"
        assert next_Noverlap_arr.all() == (next_Ntriple_arr+next_Nfailed_arr).all(), \
            "mismatch in next_Noverlap"
        assert Nmbhb_arr.min() == Nmbhb_arr.max(), \
            "found differing Nmbhb between models"

        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)
        plt.clf()  #Clear the current figure.
        plt.cla()  #Clear the current axes.
        plt.close()
        fig, ax = plt.subplots(figsize=(5,4))

#         Nmax = int(np.maximum(prev_Noverlap_arr.max(), 
#                               next_Noverlap_arr.max())*1.05)
        Nmax = int(next_Noverlap_arr.max()*1.05)
        ax.set_ylabel('Number of MBHBs', size=20)
        ax.set_ylim(0,Nmax)
        if param=='frefill':
    #         ax.plot(mbhb_frefill_arr, prev_Noverlap_arr, 'o-', color='k', alpha=0.3)
            ax.plot(mbhb_frefill_arr, next_Noverlap_arr, 'o-', color='k', 
                    label='all overlap')
    #         ax.plot(mbhb_frefill_arr, prev_Ntriple_arr, '^-', color='blue', alpha=0.3)
            ax.plot(mbhb_frefill_arr, next_Ntriple_arr, '^-', color='blue', 
                    label='triple')
    #         ax.plot(mbhb_frefill_arr, prev_Nfailed_arr, 's-', color='r', alpha=0.3)
            ax.plot(mbhb_frefill_arr, next_Nfailed_arr, 's-', color='r', 
                    label='failed')
            ax.plot(mbhb_frefill_arr, next_Nweak_trip_arr, '*--', color='cyan', 
                    label='weak triple')      
            ax.plot(mbhb_frefill_arr, next_Nstrong_trip_arr, 'x--', color='purple', 
                    label='strong triple')          
            ax.set_xlabel('$F_{refill}$')
        elif param=='eccen':
            ax.plot(mbhb_ecc_arr, next_Noverlap_arr, 'o-', color='k', 
                    label='all overlap')
            ax.plot(mbhb_ecc_arr, next_Ntriple_arr, '^-', color='blue', 
                    label='triple')
            ax.plot(mbhb_ecc_arr, next_Nfailed_arr, 's-', color='r', 
                    label='failed')            
            ax.plot(mbhb_ecc_arr, next_Nweak_trip_arr, '*--', color='cyan', 
                    label='weak triple')                
            ax.plot(mbhb_ecc_arr, next_Nstrong_trip_arr, 'x--', color='purple', 
                    label='strong triple')        

            ax.set_xlabel('e')
        ax2 = ax.twinx()
        ax2.set_ylim(0.0, Nmax/Nmbhb_arr.max())
        ax2.set_ylabel('Fraction of MBHBs')
#         ax.legend(loc='upper right')
        ax_setup(ax)
        ax_setup(ax2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(0.2, 0.53, 0.2, 0.3))
        if param=='eccen':
            fig.savefig(path+'ntriples_eccen.png', dpi=300)
        elif param=='frefill':
            fig.savefig(path+'ntriples_frefill.png', dpi=300)
        fig.subplots_adjust(wspace=0.4, right=0.98)
        plt.show()
    

def ax_setup(ax, **kwargs):
    import matplotlib.ticker as tck     
    import numpy
    ax.xaxis.set_tick_params(direction='in', which ='both')
    ax.yaxis.set_tick_params(direction='in', which='both')

    if 'grid' in kwargs.keys():
        if kwargs['grid']==True:
            ax.grid()
    return

def plot_triple_data(path= '/Users/sayebms1/mybitbucket/lb_python_scripts/input/',
                     fmrg='ill-1_blackhole_mergers_fixed.npz', 
                     fmbhb='frefill06', tm=None):
    """
    main plotting script. calls 'make_multiplot' for both the 'prev'
    and 'next' merger search directions
    """
    if not tm:
        mergers = np.load(path+fmrg)
        tm = tripleMatches(path, fmbhb, mergers)
    
    #evol_mrg_mask = (tm.evol_tmrg<=tm.tH)

    print("\ncalculating inspiral phases...")
    tmp, finalsep, phase, r_phase, time_r_phase = define_mbhb_inspiral_phases(tm, verbose=False)
    print("...done.")

    total_tinsp = tm.evol_tmrg - tm.sim_tmrg
    assert total_tinsp.min() > 0, "Found evol_tmrg < sim_tmrg!"

    print("min/max time_r_phase:")
    print(np.nanmin(time_r_phase[:,0]), np.nanmax(time_r_phase[:,0]))
    print(np.nanmin(time_r_phase[:,1]), np.nanmax(time_r_phase[:,1]))
    print(np.nanmin(time_r_phase[:,2]), np.nanmax(time_r_phase[:,2]))
    print(np.nanmin(time_r_phase[:,3]), np.nanmax(time_r_phase[:,3]))

    ttot_df = time_r_phase[:,1] - time_r_phase[:,0]
    ttot_bin = tm.evol_tmrg - time_r_phase[:,1]


    print("making plots...")
    make_multiplot(tm, ttot_df, ttot_bin, total_tinsp, 
                    plottype='prev',
                    path=path, fmbhb=fmbhb)

#    make_multiplot(tm, ttot_df, ttot_bin, total_tinsp,
#                    plottype='next',
#                   path=path, fmbhb=fmbhb)


def make_multiplot(tm, ttot_df, ttot_bin, total_tinsp,
                   plottype='next', path=None, fmbhb=None):
    """
    create a series of subplots of some basic data about the triples data
    """


    assert plottype in ['prev', 'next'], "Error: plottype must be 'prev' or 'next'."
    _mdir = tm._prev if plottype=='next' else tm._next
    
    print(_mdir.t_2nd_overtakes[_mdir.t_2nd_overtakes>0].size)

    if plottype=='prev':
        overtake_title = 'mbhbs that overtake prev mbhb'
        overtake_lbl = 'overtaken by prev'
        chase_title = 'mbhbs that chase but dont overtake prev'
        chase_lbl = 'chases prev'
        suptitle = 'Candidate triples w/ overlap b/t this mbhb and the previous mbhb\nInspiral model: {}'.format(fmbhb)
    else:
        overtake_title = 'mbhbs overtaken by next mbhb'
        overtake_lbl = 'next overtakes'
        chase_title = 'mbhbs chased but not overtaken by next'
        chase_lbl = 'next chases'
        suptitle = 'Candidate triples w/ overlap b/t this mbhb and the next mbhb\nInspiral model: {}'.format(fmbhb)


    ttot_df_1st = ttot_df[_mdir.ix_1st_mbhb[_mdir.ix_1st_mbhb>=0]]
    ttot_bin_1st = ttot_df[_mdir.ix_1st_mbhb[_mdir.ix_1st_mbhb>=0]]
    total_tinsp_1st = total_tinsp[_mdir.ix_1st_mbhb[_mdir.ix_1st_mbhb>=0]]

    mbin1 = _mdir.masses1[_mdir.strong_triple_mask] #masses of 1st binary
    mbin2 = _mdir.masses2[_mdir.strong_triple_mask] #masses of 2nd binary

    t_merge1 = _mdir.evol_tmrg_1st[_mdir.strong_triple_mask] #time when the 1st binary merges
    t_merge2 = _mdir.evol_tmrg_2nd[_mdir.strong_triple_mask] #time when the 2nd binary merges

    sep_2 = _mdir.sep_2nd_overtakes[_mdir.strong_triple_mask]
    sep_1 = _mdir.sep_1st_overtaken[_mdir.strong_triple_mask]

    sep_all = _mdir.sep_2nd_overtakes[_mdir.triple_mask]
    np.savetxt(path+'triples_all_sep.txt',sep_all)
    
    strong_sep = _mdir.sep_2nd_overtakes[_mdir.strong_triple_mask]
    np.savetxt(path+'triples_strong_sep.txt',strong_sep)
    weak_sep = _mdir.sep_2nd_overtakes[_mdir.weak_triple_mask]
    np.savetxt(path+'triples_weak_sep.txt',weak_sep)


    mids1 = _mdir.merger_ids[_mdir.ix_1st_mbhb[_mdir.strong_triple_mask]] #IDs of 1st binary
    mids2 = _mdir.merger_ids[_mdir.ix_2nd_mbhb[_mdir.strong_triple_mask]] #IDs of 2nd binary

    t_triple = _mdir.t_2nd_overtakes[_mdir.strong_triple_mask]

    # np.savetxt(path+"merger_ids.txt",np.column_stack((mids1[:,0],mids1[:,1],mids2[:,0],mids2[:,1])))
    # np.savetxt(path+"binary_masses.txt",np.column_stack((mbin1[:,0],mbin1[:,1],mbin2[:,0],mbin2[:,1])))
    
    df = pd.DataFrame([mids1[:,0],mids1[:,1],mids2[:,0],mids2[:,1],mbin1[:,0],mbin1[:,1],mbin2[:,0],mbin2[:,1],t_merge1,t_merge2,t_triple,sep_1,sep_2])
    df = df.transpose()
    df.columns = ['bhid1_bin_1','bhid2_bin_1','bhid1_bin_2','bhid2_bin_2','M1_bin_1','M2_bin_1','M1_bin_2','M2_bin_2','t1','t2','tmerger','sep_1_ovtkn','sep_2_ovtks']
    df.head()
    df.to_csv(path+"triple_data_ill.csv",index = False)
    print("File saved at",path,"triple_data_ill.csv")

    #tarr=10**np.arange(0,12)
    #tlim=(1e5,1e14)
    tlim=(1e5,1e12)
    leg_fontsz=6
    
    plt.clf()
    plt.cla()
    plt.close()
    fig = plt.figure(figsize=(8,8))

    scatter_subplot(fig, 331, ttot_df, ttot_bin, xlim=tlim, ylim=tlim, xlog=True, 
                    ylog=True, xlbl='$t_{tot,df}$ [yr]', ylbl='$t_{tot,bin}$ [yr]',
                    nomask_lbl='no overlap', mask1=_mdir.failed_triple_mask, 
                    mask1_lbl='failed triples', mask1_c='darkorange', 
                    mask2=_mdir.triple_mask, mask2_lbl='triples', mask2_c='g',
                    draw_diag=True, draw_hline=tm.tH, draw_vline=tm.tH, do_both=False,
                    title=overtake_title, leg_fontsz=leg_fontsz, leg_loc='lower right')

    scatter_subplot(fig, 332, ttot_df, ttot_bin, xlim=tlim, ylim=tlim, xlog=True,
                    ylog=True, xlbl='$t_{tot,df}$ [yr]', ylbl='$t_{tot,bin}$ [yr]',
                    mask1=_mdir.overlap_mbhb_mask, mask1_lbl='overlaps',
                    mask2=tm.evol_mrg_mask, mask2_lbl='merges', draw_diag=True,
                    draw_hline=tm.tH, draw_vline=tm.tH, title='overlapping mbhb',
                    leg_fontsz=leg_fontsz, leg_loc='lower right')
    
    scatter_subplot(fig, 333, ttot_df, ttot_bin, xlim=tlim, ylim=tlim, xlog=True, 
                    ylog=True, xlbl='$t_{tot,df}$ [yr]', ylbl='$t_{tot,bin}$ [yr]',
                    mask1=_mdir.overlap_mrg_mask, mask1_lbl='overlaps', 
                    mask2=tm.evol_mrg_mask, mask2_lbl='merges', draw_diag=True,
                    draw_hline=tm.tH, draw_vline=tm.tH, title='overlapping merger',
                    leg_fontsz=leg_fontsz, leg_loc='lower right')

    scatter_subplot(fig, 334, total_tinsp, ttot_bin/ttot_df, xlim=tlim, ylim=[1e-5,1e8], 
                    xlog=True, ylog=True, xlbl='$t_{tot}$ [yr]', 
                    ylbl='$t_{tot,bin}/t_{tot,df}$', nomask_lbl='no overlap',
                    mask1=_mdir.failed_triple_mask, mask1_lbl='failed triples', 
                    mask1_c='darkorange', mask2=_mdir.triple_mask, mask2_lbl='triples', 
                    mask2_c='g', do_both=False, draw_diag=True, draw_hline=1.0, 
                    draw_vline=tm.tH, title=overtake_title, leg_fontsz=leg_fontsz, 
                    leg_loc='upper left')

    scatter_subplot(fig, 335, _mdir.evol_tmrg_1st, _mdir.sim_tmrg_2nd, 
                    xlim=(8e8,tlim[1]), ylim=(8e8,tlim[1]), xlog=True, 
                    ylog=True, xlbl='tmrg of 1st mbhb', ylbl='tform of 2nd mbhb', 
                    nomask_lbl='no overlap', mask1 = _mdir.failed_triple_mask, 
                    mask1_lbl='failed triples', mask1_c='darkorange', 
                    mask2 = _mdir.triple_mask, mask2_lbl='triples', 
                    mask2_c='g', do_both=False, draw_diag=True, draw_hline=tm.tH, 
                    draw_vline=tm.tH, leg_fontsz=leg_fontsz, leg_loc='upper right')

    scatter_subplot(fig, 336, _mdir.evol_tmrg_1st, _mdir.evol_tmrg_2nd, 
                    xlim=(8e8,tlim[1]), ylim=(8e8,tlim[1]), xlog=True, 
                    ylog=True, xlbl='tmrg of 1st mbhb', ylbl='tmrg of 2nd mbhb', 
                    nomask_lbl='no overlap', mask1 = _mdir.failed_triple_mask, 
                    mask1_lbl='failed triples', mask1_c='darkorange', 
                    mask2 = _mdir.triple_mask, mask2_lbl='triples', mask2_c='g', 
                    do_both=False, draw_diag=True, draw_hline=tm.tH, draw_vline=tm.tH,
                    leg_fontsz=leg_fontsz, leg_loc='upper right')
    
    tbin_edges = np.arange(4.5,18,0.75)    
    hist_subplot(fig, 337, np.log10(total_tinsp), bins=tbin_edges, 
                 ylog=True, xlbl='total mbhb inspiral time [log yr]', 
                 nomask_lbl='no overlap', mask1 = _mdir.failed_triple_mask, 
                 mask1_lbl='failed triples', mask1_c='darkorange',
                 mask2 = _mdir.triple_mask, mask2_lbl='triples', mask2_c='g', 
                 do_both=False, title='', leg_fontsz=6, leg_loc='upper left')

    lg_abin_edges = np.arange(2.2,4.2,0.2)
    hist_subplot(fig, 338, np.log10(tm.sep[:,0]), bins=lg_abin_edges, ylog=True, 
                 xlbl='initial binary separation [log pc]', nomask_lbl='no overlap',
                 mask1 = _mdir.failed_triple_mask, mask1_lbl='failed triples', 
                 mask1_c='darkorange', mask2 = _mdir.triple_mask, mask2_lbl='triples', 
                 mask2_c='g', do_both=False, title='', leg_fontsz=6, 
                 leg_loc='upper left')

    ax9=fig.add_subplot(339)
    plt.xlabel('overlap time [log yr]')
    plt.yscale('log')
    ax9.hist(np.log10(_mdir.overlapping_mbhb_dt[_mdir.overlap_mbhb_mask]), 
             label='mbhb tolap', histtype='step', color='b',lw=1.5, 
             bins=tbin_edges)
    ax9.hist(np.log10(_mdir.overlapping_mbhb_dt[(_mdir.overlap_mbhb_mask)
                                                &(tm.evol_mrg_mask)]), 
             label='mbhb tolap (z>0)', histtype='step', color='m',lw=1.5, 
             bins=tbin_edges)
    ax9.hist(np.log10(_mdir.overlapping_dtmrg[_mdir.overlap_mrg_mask]), 
             label='dtmrg', histtype='step', color='c', bins=tbin_edges)
    ax9.hist(np.log10(_mdir.overlapping_dtmrg[(_mdir.overlap_mrg_mask)
                                              &(tm.evol_mrg_mask)]), 
             label='dtmrg (z>0)', histtype='step', color='r', bins=tbin_edges)
    ax9.legend(fontsize=leg_fontsz, loc='upper left')

    fig.suptitle(suptitle)
    
    fig.subplots_adjust(hspace=0.4,wspace=0.4,top=0.9,right=0.95)
    fig.savefig(path+'triples_tscales_{}_{}.png'.format(plottype,fmbhb), dpi=300)


def scatter_subplot(fig, sub_num, xvals, yvals, 
                    xlim=[5,14], ylim=[5,14], xlog=True, ylog=True, 
                    xlbl='$t_{tot,df}$ [yr]', ylbl='$t_{tot,bin}$ [yr]',
                    nomask_lbl='', mask1=None, mask1_lbl='', mask1_c='b',
                    mask2=None, mask2_lbl='', mask2_c='k', do_both=True,
                    draw_diag=True, draw_hline=None, draw_vline=None, 
                    title='', leg_fontsz=6, leg_loc='lower right'):

    assert len(xvals) == len(yvals), \
        "Error: xvals & yvals have different lengths: {} {}".format(len(xvals),
                                                                    len(yvals))
    ax = fig.add_subplot(sub_num)
    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    plt.xlim(xlim)                   
    plt.ylim(ylim)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    ax.plot(xvals, yvals, color='gray', marker='.', ms=2, lw=0, 
            alpha=0.2, label=nomask_lbl)

    if mask1 is not None: 
        assert len(mask1) == len(xvals), \
            "Error: mask1 & xvals have different lengths: {} {}".format(len(mask1),len(xvals))
        ax.plot(xvals[mask1], yvals[mask1], '.', color=mask1_c, ms=2, 
                alpha=0.2, label=mask1_lbl)
        if mask2 is not None:
            assert len(mask2) == len(xvals), \
                "Error: mask2 & xvals have different lengths: {} {}".format(len(mask2),len(xvals))
            ax.plot(xvals[mask2], yvals[mask2], '.', color=mask2_c, ms=2, 
                    alpha=0.2,label=mask2_lbl)
            if do_both:
                ax.plot(xvals[(mask1)&(mask2)], yvals[(mask1)&(mask2)], 
                        'm.', ms=2, alpha=0.6,label='both')

    elif mask1 is None and mask2 is not None:
        ax.plot(xvals[mask2], yvals[mask2], '.', color=mask2_c, ms=2, 
                alpha=0.2, label=mask2_lbl)
        
    if draw_diag: ax.plot(xlim, ylim, 'k')
    if draw_vline is not None: ax.plot([draw_vline, draw_vline], ylim, 'k--')
    if draw_hline is not None: ax.plot(xlim, [draw_hline, draw_hline], 'k--')

    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=leg_fontsz, loc=leg_loc)


def hist_subplot(fig, sub_num, vals, bins=None, ylog=True, xlbl='',
                 nomask_lbl='', mask1=None, mask1_lbl='', mask1_c='b', 
                 mask2=None, mask2_lbl='', mask2_c='k', do_both=True,
                 title='', leg_fontsz=6, leg_loc='lower right'):

    ax = fig.add_subplot(sub_num)

    if bins is None: bins='auto'
    if ylog: plt.yscale('log')

    plt.xlabel(xlbl)
    ax.hist(vals, bins=bins, histtype='step',color='gray', label=nomask_lbl)
    ax.hist(vals[mask1], bins=bins, histtype='step', color=mask1_c, lw=1.5,
            label=mask1_lbl)
    ax.hist(vals[mask2], bins=bins, histtype='step', color=mask2_c, lw=1.5, 
            label=mask2_lbl)
    if do_both: ax.hist(vals[(mask1)&(mask2)], bins=bins, histtype='step', 
                        color='m',lw=1.5, label='both')
    ax.legend(fontsize=leg_fontsz, loc=leg_loc)

