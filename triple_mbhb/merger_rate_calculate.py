import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
from astropy import constants as const
from tqdm import tqdm

boxsize = 75.0
omega_m = 0.2726
omega_l = 0.7274
h = 0.704
vol_comov_box = (boxsize/h)**3 #Mpc^3


def unit_comoving_vol(z):

    D_H = const.c.to('km/s').value/cosmo.H0.value
    return D_H * cosmo.comoving_transverse_distance(z).value**2 * cosmo.inv_efunc(z)

def merger_rate_find(z_bh,zbinsize = 0.2,zmax = 7):

    dVcratio = np.array([unit_comoving_vol(z) 
                                  for z in z_bh ]) * 4*np.pi/vol_comov_box
    
    Nmrg_zhist,zbin_edges = np.histogram(z_bh,range=(0,zmax),bins=int(zmax/zbinsize))
    Nmrg_zhist[Nmrg_zhist==0] = 1.0e-10
    zbins = zbin_edges[:-1]+0.5*zbinsize

    dNmrgdz,tmp = np.histogram(z_bh,weights=dVcratio,bins=zbin_edges)
    
    dt_zbins = []

    for i in range(zbins.size):
        zatage = cosmo.age(zbins[i]-0.5*zbinsize)-cosmo.age(zbins[i]+0.5*zbinsize)
        dt_zbins.append(float(zatage/u.Gyr))

    dt_zbins = np.array(dt_zbins)
    
    #dNmrg_dzdt = np.array([dNmrgdz[i]/dt_zbins[i]/10**9/vol_comov_box for i in range(zbins.size)]) ## yr^-1 cMpc^-3
    #print("total merger rate density (yr^-1 cMpc^-3): ",np.sum(dNmrg_dzdt * zbinsize))

    dNmrg_dzdt_allsky = np.array([dNmrgdz[i]/dt_zbins[i]/10**9/(1+zbins[i])
                           for i in range(zbins.size)]) ## yr^-1 

    merger_rate = np.sum(dNmrg_dzdt_allsky * zbinsize) 

    dNmrg_dzdt_allsky_cumulative = np.array([dNmrgdz[i]/dt_zbins[i]/10**9
                           for i in range(zbins.size)]) ## yr^-1 

    cumulative_merger_rate = np.sum(dNmrg_dzdt_allsky_cumulative*zbinsize)

    return merger_rate,cumulative_merger_rate

def diff_merger_rate(z_bh,lgzbinsize=0.2,lgzmin=-4.0,lgzmax = 1.0):

    dVcratio = np.array([unit_comoving_vol(z) 
                                  for z in z_bh ]) * 4*np.pi/vol_comov_box

    lgz_bh = np.log10(z_bh)
    Nmrg_lgzhist,lgzbin_edges = np.histogram(lgz_bh,range=(lgzmin,lgzmax),
                                             bins=int((lgzmax-lgzmin)/lgzbinsize))
    
    Nmrg_lgzhist[Nmrg_lgzhist==0] = 1.0e-10
    lgzbins = lgzbin_edges[:-1]+0.5*lgzbinsize
    Nmrg_lgzhist = Nmrg_lgzhist.astype('float64')
        
    dNmrgdlogz,tmp = np.histogram(lgz_bh,weights=dVcratio,bins=lgzbin_edges)
    dNmrgdlogz[dNmrgdlogz==0] = 1.0e-10
    dt_lgzbins = []
    
    for i in range(lgzbins.size):
        zatage = cosmo.age(10**(lgzbins[i]-0.5*lgzbinsize))-cosmo.age(10**(lgzbins[i]+0.5*lgzbinsize))
        dt_lgzbins.append(float(zatage/u.Gyr))
    
    dt_lgzbins = np.array(dt_lgzbins)

        #dNmrg_dlogzdt_allsky = np.array([dNmrgdlogz[i]*10**lgzbins[i]*np.log(10)/dt_lgzbins[i]/(1+10**lgzbins[i])/10**9 for i in range(lgzbins.size)])
    #dNmrg_dlogzdt = np.array([dNmrgdlogz[i]/dt_lgzbins[i]/10**9 for i in range(lgzbins.size)])
    dNmrg_dlogzdt_allsky = np.array([dNmrgdlogz[i]/dt_lgzbins[i]/10**9/(1+10**lgzbins[i]) for i in range(lgzbins.size)])

    return lgzbins, dNmrg_dlogzdt_allsky

def print_all_merger_rates(iso_bin,weak_tr,strong_tr,stalled_objs,Nruns,zbinsize=0.1,zmax=7):
    
    merger_rates = [
    obj.total_merger_rate("all", zbinsize, zmax) for obj in tqdm(strong_tr, desc="Calculating Merger Rates")
    ]

    strong_triple_merger_rate,strong_triple_cum_merger_rate = np.mean(merger_rates,axis=0)

    iso_bin_merger_rate,iso_bin_cum_merger_rate = iso_bin.total_merger_rate(zbinsize, zmax)
    weak_triples_merger_rate,weak_triples_cum_merger_rate = weak_tr.total_merger_rate(zbinsize, zmax)

    total_merger_rate = strong_triple_merger_rate+iso_bin_merger_rate+weak_triples_merger_rate

    print("------------------------------")
    print(f"Merger rate considering only isolated binaries is {iso_bin_merger_rate:.3f} yr^-1")
    print(f"Merger rate of weak triples is {weak_triples_merger_rate:.4f} yr^-1")
    print(f"Total strong triple merger rate is : {strong_triple_merger_rate:.4f} yr^{-1}")
    #print(f"Merger rate increases from {iso_bin_merger_rate:.2f} to {iso_bin_merger_rate+strong_triple_merger_rate:.2f} which is {strong_triple_merger_rate/iso_bin_merger_rate * 100:.1f} % increase when we add strong triples ")
    #print(f"After adding both strong and weak triples the merger rate is {total_merger_rate:.2f} which is a {(total_merger_rate-iso_bin_merger_rate)/iso_bin_merger_rate * 100:.1f} % increase")
    print(f"Total merger rate including strong and weak triples is {total_merger_rate:.3f} yr^-1")
    z_triple_inspiral = strong_tr[0].z_triple_merger[strong_tr[0].bin_merge_flag]
    print(f"The merger rate for strong triples under inspiral evolution (without considering strong interaction) is roughly {merger_rate_find(z_triple_inspiral,zbinsize=0.2, zmax=7)[0]:.4f} yr^{-1}")
    print("------------------------------")

    cum_merger_rate = strong_triple_cum_merger_rate+iso_bin_cum_merger_rate+weak_triples_cum_merger_rate

    print(f"Merger rate considering only isolated binaries is {iso_bin_cum_merger_rate:.3f} yr^-1")
    print(f"Merger rate of weak triples is {weak_triples_cum_merger_rate:.3f} yr^-1")
    print(f"Total strong triple merger rate is : {strong_triple_cum_merger_rate:.3f} yr^{-1}")
    #print(f"Merger rate increases from {iso_bin_merger_rate:.2f} to {iso_bin_merger_rate+strong_triple_merger_rate:.2f} which is {strong_triple_merger_rate/iso_bin_merger_rate * 100:.1f} % increase when we add strong triples ")
    #print(f"After adding both strong and weak triples the merger rate is {total_merger_rate:.2f} which is a {(total_merger_rate-iso_bin_merger_rate)/iso_bin_merger_rate * 100:.1f} % increase")
    print(f"Total merger rate including strong and weak triples is {cum_merger_rate:.3f} yr^-1")
    z_triple_inspiral = strong_tr[0].z_triple_merger[strong_tr[0].bin_merge_flag]
    print(f"The merger rate for strong triples under inspiral evolution (without considering strong interaction) is roughly {merger_rate_find(z_triple_inspiral,zbinsize=0.2, zmax=7)[1]:.4f} yr^{-1}")
    print("------------------------------")

    strong_tr_mergers = []
    for i in range(Nruns):
        strong_tr_mergers.append(np.sum(strong_tr[i].merger_mask))
                    
    total_mergers = np.mean(strong_tr_mergers)+np.sum(iso_bin.merger_flag)+np.sum(weak_tr.bin_merge_flag)
    total_system = len(strong_tr[0].merger_mask)+len(iso_bin.merger_mask)+len(weak_tr.merger_mask)
    print(f"The total number of mergers in the population is {total_mergers:.0f} out of {total_system}")

    iso_bin_mergers = np.sum(iso_bin.merger_flag)
    weak_tr_bin_mergers = np.sum(weak_tr.merger_mask)
    strong_trip_insp_mergers = np.sum(strong_tr[0].bin_merge_flag)
    Nmbhb = total_system


    print(f"{total_mergers/Nmbhb * 100:.1f} % mergers in fiducial (iso+triples) model")
    print(f"{(iso_bin_mergers)/Nmbhb * 100:.1f} % mergers in isolated binary evolution")
    print(f"{(iso_bin_mergers+weak_tr_bin_mergers+strong_trip_insp_mergers)/Nmbhb * 100:.1f} % mergers in inspiral evolution")


    print("-------------------------------------------------")

    print_stalled_merger_rates(iso_bin,weak_tr,strong_tr,stalled_objs,Nruns,total_merger_rate,zbinsize,zmax)
    return None


def print_stalled_merger_rates(iso_bin,weak_tr,strong_tr,stalled_objs,Nruns,total_merger_rate,zbinsize=0.2,zmax=7):
    Nmbhb = len(strong_tr[0].merger_mask)+len(iso_bin.merger_mask)+len(weak_tr.merger_mask)
    print(f"There are {stalled_objs[0].N_stalled_triples}({stalled_objs[0].N_stalled_triples/Nmbhb * 100:.2f}%)stalled binary systems that forms possible triple systems from {Nmbhb} total systems")

    prompt_merger = 0
    merger_after_ejections = 0

    for i in range(Nruns):
        prompt_merger+=stalled_objs[i].prompt_merger
        merger_after_ejections+= stalled_objs[i].merger_after_ejection

    prompt_merger = prompt_merger/Nruns
    merger_after_ejections = merger_after_ejections/Nruns
    # Total_mergers_in_stalled = prompt_merger+merger_after_ejections
    Total_mergers_in_stalled = prompt_merger

    iso_bin_mergers = np.sum(iso_bin.merger_flag)
    weak_tr_bin_mergers = np.sum(weak_tr.merger_mask)
    strong_trip_insp_mergers = np.sum(strong_tr[0].bin_merge_flag)

    #print(f"{Total_mergers_in_stalled/Nmbhb * 100:.1f} % mergers in stalled model")
    print(f"There are a total of {Total_mergers_in_stalled:.0f} ({Total_mergers_in_stalled/Nmbhb * 100:.2f} %)triple-induced mergers")


    # merger_rates = [obj.total_merger_rate("all", zbinsize, zmax) for obj in tqdm(stalled_objs, desc="Calculating Merger Rates")]
    merger_rates = [obj.total_merger_rate("Tr", zbinsize, zmax) for obj in tqdm(stalled_objs, desc="Calculating Merger Rates")]

    stalled_triple_merger_rate,stalled_cum_merger_rate = np.mean(merger_rates,axis=0)
    print(f"The total merger rate in the stalled model is {stalled_triple_merger_rate:.3f} yr^{-1}")
    print(f"The total merger rate in the fiducial (isolated binary+triple) model is {total_merger_rate:.2f} yr^{-1}")
    print(f"The merger rate is suppressed by a factor of {total_merger_rate/stalled_triple_merger_rate:.1f}.")

    return None

