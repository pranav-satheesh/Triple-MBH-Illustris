import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import count_vkicks as kickcount
import os

tex_path = '/apps/texlive/2023/bin/x86_64-linux/'
os.environ['PATH'] += os.pathsep + tex_path

import scienceplots
plt.style.use('science')
import pickle


boxsize = 75.0
omega_m = 0.2726
omega_l = 0.7274
h = 0.704
vol_comov_box = (boxsize/h)**3 #Mpc^3


vth_l = 50
vth_h = 2600
thresholds = np.arange(50,2600,125)
Nruns = 100


kick_colors = {'slingshot':"#4daf4a",'aligned':"#377eb8","hybrid":"#a2c8ec","random":"#e41a1c"}

def Nkicks(strong_tr,weak_tr,iso_bin,thresh,Nruns):

    N_random = []
    N_deg5 = []
    N_hybrid = []
    N_slingshot_kick = []

    for i in range(Nruns):
        sling_v = [kick for kick in strong_tr[i].slingshot_kicks if kick > thresh]
        sling_v = np.array(sling_v)

        rand_v_strong = [kick for kick in strong_tr[i].gw_kick_random if kick > thresh]
        rand_v_weak = [kick for kick in weak_tr.gw_kick_random if kick > thresh]
        rand_v_iso = [kick for kick in iso_bin.gw_kick_random if kick > thresh]
        rand_v = np.concatenate((rand_v_strong,rand_v_iso,rand_v_weak))

        hybrid_v_strong = [kick for kick in strong_tr[i].gw_kick_hybrid if kick > thresh]
        hybrid_v_weak = [kick for kick in weak_tr.gw_kick_hybrid if kick > thresh]
        hybrid_v_iso = [kick for kick in iso_bin.gw_kick_hybrid if kick > thresh]
        hybrid_v = np.concatenate((hybrid_v_strong,hybrid_v_iso,hybrid_v_weak))
        #hybrid_v = np.array(hybrid_v)

        deg5_v_strong = [kick for kick in strong_tr[i].gw_kick_aligned if kick > thresh]
        deg5_v_weak = [kick for kick in weak_tr.gw_kick_aligned if kick > thresh]
        deg5_v_iso = [kick for kick in iso_bin.gw_kick_aligned if kick > thresh]
        deg5_v = np.concatenate((deg5_v_strong,deg5_v_iso,deg5_v_weak))
        #deg5_v = np.array(deg5_v)

        N_random.append(len(rand_v))
        N_deg5.append(len(deg5_v))
        N_hybrid.append(len(hybrid_v))
        N_slingshot_kick.append(len(sling_v))

    return N_random,N_deg5,N_hybrid,N_slingshot_kick

def dNbydV(strong_tr, weak_tr, iso_bin, v_max_values, vbin_sizes, Nruns):
    dNbydV_sling = []
    dNbydV_rand = []
    dNbydV_hybrid = []
    dNbydV_deg5 = []
    vbins_sling = []
    vbins_rand = []
    vbins_hybrid = []
    vbins_deg5 = []

    for i in range(Nruns):
        # Slingshot kicks
        Nvslingshot, vbin_edges_sling = np.histogram(strong_tr[i].slingshot_kicks, 
                                                       range=(0, v_max_values['sling']), 
                                                       bins=int(v_max_values['sling'] / vbin_sizes['sling']))
        vbins_sling.append(vbin_edges_sling[:-1] + 0.5 * vbin_sizes['sling'])
        dNbydV_sling.append(Nvslingshot / vbin_sizes['sling'])

        # Random kicks
        dNv_rand_strong, vbin_edges_rand = np.histogram(strong_tr[i].gw_kick_random, 
                                                          range=(0, v_max_values['random']), 
                                                          bins=int(v_max_values['random'] / vbin_sizes['random']))
        dNv_rand_weak, _ = np.histogram(weak_tr.gw_kick_random, 
                                         range=(0, v_max_values['random']), 
                                         bins=int(v_max_values['random'] / vbin_sizes['random']))
        dNv_rand_binary, _ = np.histogram(iso_bin.gw_kick_random, 
                                           range=(0, v_max_values['random']), 
                                           bins=int(v_max_values['random'] / vbin_sizes['random']))
        dNbydV_rand.append((dNv_rand_binary + dNv_rand_strong + dNv_rand_weak) / vbin_sizes['random'])
        vbins_rand.append(vbin_edges_rand[:-1] + 0.5 * vbin_sizes['random'])

        # Hybrid kicks
        dNv_hybrid_strong, vbin_edges_hybrid = np.histogram(strong_tr[i].gw_kick_hybrid, 
                                                             range=(0, v_max_values['hybrid']), 
                                                             bins=int(v_max_values['hybrid'] / vbin_sizes['hybrid']))
        dNv_hybrid_weak, _ = np.histogram(weak_tr.gw_kick_hybrid, 
                                           range=(0, v_max_values['hybrid']), 
                                           bins=int(v_max_values['hybrid'] / vbin_sizes['hybrid']))
        dNv_hybrid_binary, _ = np.histogram(iso_bin.gw_kick_hybrid, 
                                             range=(0, v_max_values['hybrid']), 
                                             bins=int(v_max_values['hybrid'] / vbin_sizes['hybrid']))
        dNbydV_hybrid.append((dNv_hybrid_binary + dNv_hybrid_strong + dNv_hybrid_weak) / vbin_sizes['hybrid'])
        vbins_hybrid.append(vbin_edges_hybrid[:-1] + 0.5 * vbin_sizes['hybrid'])

        # Aligned kicks
        dNv_deg5_strong, vbin_edges_deg5 = np.histogram(strong_tr[i].gw_kick_aligned, 
                                                          range=(0, v_max_values['deg5']), 
                                                          bins=int(v_max_values['deg5'] / vbin_sizes['deg5']))
        dNv_deg5_weak, _ = np.histogram(weak_tr.gw_kick_aligned, 
                                         range=(0, v_max_values['deg5']), 
                                         bins=int(v_max_values['deg5'] / vbin_sizes['deg5']))
        dNv_deg5_binary, _ = np.histogram(iso_bin.gw_kick_aligned, 
                                           range=(0, v_max_values['deg5']), 
                                           bins=int(v_max_values['deg5'] / vbin_sizes['deg5']))
        dNbydV_deg5.append((dNv_deg5_binary + dNv_deg5_strong + dNv_deg5_weak) / vbin_sizes['deg5'])
        vbins_deg5.append(vbin_edges_deg5[:-1] + 0.5 * vbin_sizes['deg5'])

    # Normalize by comoving volume
    dNbydV_sling = np.array(dNbydV_sling) / vol_comov_box
    dNbydV_hybrid = np.array(dNbydV_hybrid) / vol_comov_box
    dNbydV_rand = np.array(dNbydV_rand) / vol_comov_box
    dNbydV_deg5 = np.array(dNbydV_deg5) / vol_comov_box

    return (
        vbins_sling, dNbydV_sling,
        vbins_rand, dNbydV_rand,
        vbins_hybrid, dNbydV_hybrid,
        vbins_deg5, dNbydV_deg5
    )

def plot_rate_kicks(strong_tr, weak_tr, iso_bin, v_max_values, vbin_sizes, Nruns):
    fig, ax = plt.subplots(figsize=(7, 6))

    vbins_sling, dNdVsling, vbins_rand, dNdVrand, vbins_hybrid, dNdVhybrid, vbins_deg5, dNdVdeg5 = dNbydV(
        strong_tr, weak_tr, iso_bin, v_max_values, vbin_sizes, Nruns)

    # Plotting slingshot kicks
    ax.plot(vbins_sling[0], np.mean(dNdVsling, axis=0), linestyle='-', marker='o', color=kick_colors["slingshot"], label="slingshot")
    ax.fill_between(vbins_sling[0], np.mean(dNdVsling, axis=0) - np.std(dNdVsling, axis=0), 
                    np.mean(dNdVsling, axis=0) + np.std(dNdVsling, axis=0), color=kick_colors["slingshot"], alpha=0.5)

    # Plotting random kicks
    ax.plot(vbins_rand[0], np.mean(dNdVrand, axis=0), linestyle='-', marker='o', color=kick_colors["random"], label="random")
    ax.fill_between(vbins_rand[0], np.mean(dNdVrand, axis=0) - np.std(dNdVrand, axis=0), 
                    np.mean(dNdVrand, axis=0) + np.std(dNdVrand, axis=0), color=kick_colors["random"], alpha=0.5)

    # Plotting hybrid kicks
    ax.plot(vbins_hybrid[0], np.mean(dNdVhybrid, axis=0), linestyle='-', marker='o', color=kick_colors["hybrid"], label="hybrid")
    ax.fill_between(vbins_hybrid[0], np.mean(dNdVhybrid, axis=0) - np.std(dNdVhybrid, axis=0), 
                    np.mean(dNdVhybrid, axis=0) + np.std(dNdVhybrid, axis=0), color=kick_colors["hybrid"], alpha=0.5)

    # Plotting aligned kicks
    ax.plot(vbins_deg5[0], np.mean(dNdVdeg5, axis=0), linestyle='-', marker='o', color=kick_colors['aligned'], label="aligned")
    ax.fill_between(vbins_deg5[0], np.mean(dNdVdeg5, axis=0) - np.std(dNdVdeg5, axis=0), 
                    np.mean(dNdVdeg5, axis=0) + np.std(dNdVdeg5, axis=0), color="#0571b0", alpha=0.5)

    ax.set_yscale("log", base=10)
    ax.set_xlabel("$v_t$")
    ax.set_ylabel("$R_{kicks} [Gpc^{-3} km^{-1} s]$")
    ax.legend()       
    ax.set_xlim(0, max(v_max_values.values()))
    fig.tight_layout()
    
    return fig, ax

def plot_N_kicks(df_strong,df_weak,df_binary,thresholds):
     
    fig,ax = plt.subplots(1,2,figsize=[10,6])
    N_r_trip,N_deg_trip,N_hybrid_trip,N_sling_trip = Nkicks(df_strong,df_weak,df_binary,thresholds,only_triples=True)

    ax[0].plot(thresholds,np.mean(N_sling_trip ,axis=0),color="green",label="Slingshot-kick")
    ax[0].fill_between(thresholds,np.mean(N_sling_trip,axis = 0)-np.std(N_sling_trip,axis = 0),np.mean(N_sling_trip,axis = 0)+np.std(N_sling_trip,axis = 0),color="green",alpha=0.5)
    ax[0].scatter(thresholds,np.mean(N_sling_trip,axis = 0),color="green")

    ax[0].plot(thresholds,np.mean(N_deg_trip,axis=0),color="#0571b0",label="GW-deg5")
    ax[0].scatter(thresholds,np.mean(N_deg_trip,axis = 0),color="#0571b0")
    ax[0].fill_between(thresholds,np.mean(N_deg_trip,axis = 0)-np.std(N_deg_trip,axis = 0),np.mean(N_deg_trip,axis = 0)+np.std(N_deg_trip,axis = 0),color="#0571b0",alpha=0.5)
    
    ax[0].plot(thresholds,np.mean(N_r_trip,axis=0),color="#ca0020",label="GW-random-dry")
    ax[0].scatter(thresholds,np.mean(N_r_trip,axis=0),color="#ca0020")
    ax[0].fill_between(thresholds,np.mean(N_r_trip,axis = 0)-np.std(N_r_trip,axis = 0),np.mean(N_r_trip,axis = 0)+np.std(N_r_trip,axis = 0),color="#ca0020",alpha=0.5)
    
    ax[0].plot(thresholds,np.mean(N_hybrid_trip,axis=0),color="#92c5de",label="GW-hybrid")
    ax[0].scatter(thresholds,np.mean(N_hybrid_trip,axis=0),color="#92c5de")
    ax[0].fill_between(thresholds,np.mean(N_hybrid_trip,axis = 0)-np.std(N_hybrid_trip,axis = 0),np.mean(N_hybrid_trip,axis = 0)+np.std(N_hybrid_trip,axis = 0),color="#92c5de",alpha=0.5) 
    
    ax[0].set_yscale("log",base=10)
    ax[0].set_ylim(1,)
    ax[0].set_xlim(0,2500)
    ax[0].legend()
    ax[0].set_ylabel(r"$N_{\text{kicks}} (v > v_{\text{th}}) $",fontsize=16)
    ax[0].set_xlabel(r"$v_{\text{th}} (\text{km} \, \text{s}^{-1})$",fontsize=16)
    ax[0].set_title("Triples (strong + weak)")

    N_r,N_deg,N_hybrid,N_sling = Nkicks(df_strong,df_weak,df_binary,thresholds,only_triples=False)

    ax[1].plot(thresholds,np.mean(N_sling ,axis=0),color="green",label="Slingshot-kick")
    ax[1].fill_between(thresholds,np.mean(N_sling,axis = 0)-np.std(N_sling,axis = 0),np.mean(N_sling,axis = 0)+np.std(N_sling,axis = 0),color="green",alpha=0.5)
    ax[1].scatter(thresholds,np.mean(N_sling,axis = 0),color="green")

    ax[1].plot(thresholds,np.mean(N_deg,axis=0),color="#0571b0",label="GW-deg5")
    ax[1].scatter(thresholds,np.mean(N_deg,axis = 0),color="#0571b0")
    ax[1].fill_between(thresholds,np.mean(N_deg,axis = 0)-np.std(N_deg,axis = 0),np.mean(N_deg,axis = 0)+np.std(N_deg,axis = 0),color="#0571b0",alpha=0.5)
    
    ax[1].plot(thresholds,np.mean(N_r,axis=0),color="#ca0020",label="GW-random-dry")
    ax[1].scatter(thresholds,np.mean(N_r,axis=0),color="#ca0020")
    ax[1].fill_between(thresholds,np.mean(N_r,axis = 0)-np.std(N_r,axis = 0),np.mean(N_r,axis = 0)+np.std(N_r,axis = 0),color="#ca0020",alpha=0.5)
    
    ax[1].plot(thresholds,np.mean(N_hybrid,axis=0),color="#92c5de",label="GW-hybrid")
    ax[1].scatter(thresholds,np.mean(N_hybrid,axis=0),color="#92c5de")
    ax[1].fill_between(thresholds,np.mean(N_hybrid,axis = 0)-np.std(N_hybrid,axis = 0),np.mean(N_hybrid,axis = 0)+np.std(N_hybrid,axis = 0),color="#92c5de",alpha=0.5) 

    ax[1].set_yscale("log",base=10)
    ax[1].set_ylim(1,)
    ax[1].set_xlim(0,2500)
    ax[1].legend()
    ax[1].set_ylabel(r"$N_{\text{kicks}} (v > v_{\text{th}}) $",fontsize=16)
    ax[1].set_xlabel(r"$v_{\text{th}} (\text{km} \, \text{s}^{-1})$",fontsize=16)
    ax[1].set_title("Triples + Binaries")

    fig.tight_layout()
    plt.show()
    return fig,ax

def plot_spin_dist():
    N_sample = 100000

    theta_binsize = 2
    theta_max = 90
    theta_min = -2
    theta_num = int((theta_max-theta_min)/theta_binsize)

    #random-dry
    theta_rd = np.arccos(stats.uniform(-1,2).rvs(N_sample))
    theta_rd = theta_rd * 180/np.pi
    for i in range(len(theta_rd)):
        if theta_rd[i] > 90:
            theta_rd[i] = 180 - theta_rd[i]
    theta_rd_counts,theta_rd_edges = np.histogram(theta_rd,range=(theta_min,theta_max),bins=theta_num,density=True)
    theta_rd_bins = theta_rd_edges[:-1] + (theta_rd_edges[1] - theta_rd_edges[0])/2

    alpha = 8
    beta = 4
    a_rd = stats.beta(alpha,beta).rvs(N_sample)
    a_rd_counts,a_rd_edges = np.histogram(a_rd,range=(0,1.1),bins=35,density=True)
    a_rd_bins = a_rd_edges[:-1] + (a_rd_edges[1] - a_rd_edges[0])/2

    #cold
    alpha = 2.018
    beta = 5.244
    cold = stats.beta(alpha,beta).rvs(N_sample)
    cold = cold * 180/np.pi
    cold_counts,cold_edges = np.histogram(cold,range=(theta_min,theta_max),bins=theta_num,density=True)
    cold_bins = cold_edges[:-1] + (cold_edges[1] - cold_edges[0])/2


    #aligned
    alpha = 5.935
    beta = 1.856
    a_cold = stats.beta(alpha,beta).rvs(N_sample)
    a_cold_counts,a_cold_edges = np.histogram(a_cold,range=(0,1.1),bins=35,density=True)
    a_cold_bins = a_cold_edges[:-1] + (a_cold_edges[1] - a_cold_edges[0])/2

    loc = np.cos(5*np.pi/180)
    scale = np.cos(0) - np.cos(5*np.pi/180)
    theta_5deg = np.arccos(stats.uniform(loc,scale).rvs(N_sample*10))
    theta_5deg = theta_5deg * 180/np.pi
    theta5deg_counts,theta_5deg_edges = np.histogram(theta_5deg,range=(theta_min,theta_max),bins=theta_num,density=True)
    theta5deg_bins = theta_5deg_edges[:-1] + (theta_5deg_edges[1] - theta_5deg_edges[0])/2

    color_palette={"random":"#377eb8","cold":"#a2c8ec","aligned":"#e41a1c"}

    fig,ax = plt.subplots(2,1,figsize=(8,10))
    ax[1].plot(a_rd_bins,a_rd_counts,color=color_palette["random"],label="random")
    ax[1].plot(a_cold_bins,a_cold_counts,color=color_palette["cold"],linestyle="--",label="cold")
    ax[1].axvline(x=0.9,color=color_palette["aligned"],label="aligned")
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,)
    #ax[1].legend()
    ax[1].set_ylabel(r"$P(a)$")
    ax[1].set_xlabel(r"$a$")
    ax[1] = plt.gca()
    xticks = ax[1].xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    ax[0].plot(theta_rd_bins,theta_rd_counts,color="#0571b0",label="random")
    ax[0].plot(cold_bins,cold_counts,color="#92c5de",linestyle="--",label="cold")
    ax[0].plot(theta5deg_bins,theta5deg_counts,color="#ca0020",label="aligned")
    ax[0].set_xlim(0.5,90)
    ax[0].set_ylim(0,)
    ax[0].legend()
    ax[0].set_ylabel(r"$P(\theta_{1,2})$")
    ax[0].set_xlabel(r"$\theta_{1,2} [^{\circ}]$")
    fig.tight_layout()

    return fig,ax

def Prob_kick(strong_tr,weak_tr,iso_bin,Nruns,N_v=50):
    '''Plots the probability of recoil kick 
    that is greater than a threshold kick (which is varied in the x-axis)'''

    sling_prob=[]
    rand_prob=[]
    hybrid_prob=[]
    aligned_prob=[]

    #log scale
    vmin=1
    vmax=4 
    vth_values = np.logspace(vmin,vmax,N_v)
    

    for i in range(Nruns):
        sling_v = [kick for kick in strong_tr[i].slingshot_kicks if kick > 0]
        sling_v = np.array(sling_v)

        rand_v_strong = [kick for kick in strong_tr[i].gw_kick_random if kick > 0]
        rand_v_weak = [kick for kick in weak_tr.gw_kick_random if kick > 0]
        rand_v_iso = [kick for kick in iso_bin.gw_kick_random if kick > 0]
        rand_v = np.concatenate((rand_v_strong,rand_v_iso,rand_v_weak))

        hybrid_v_strong = [kick for kick in strong_tr[i].gw_kick_hybrid if kick > 0]
        hybrid_v_weak = [kick for kick in weak_tr.gw_kick_hybrid if kick > 0]
        hybrid_v_iso = [kick for kick in iso_bin.gw_kick_hybrid if kick > 0]
        hybrid_v = np.concatenate((hybrid_v_strong,hybrid_v_iso,hybrid_v_weak))

        deg5_v_strong = [kick for kick in strong_tr[i].gw_kick_aligned if kick > 0]
        deg5_v_weak = [kick for kick in weak_tr.gw_kick_aligned if kick > 0]
        deg5_v_iso = [kick for kick in iso_bin.gw_kick_aligned if kick > 0]
        deg5_v = np.concatenate((deg5_v_strong,deg5_v_iso,deg5_v_weak))

        sling_p = [(sling_v > vth).sum() / len(sling_v) for vth in vth_values]
        rand_p = [(rand_v > vth).sum() / len(rand_v) for vth in vth_values]
        hybrid_p = [(hybrid_v > vth).sum() / len(hybrid_v) for vth in vth_values] 
        aligned_p = [(deg5_v > vth).sum() / len(deg5_v) for vth in vth_values] 

        sling_prob.append(sling_p)
        rand_prob.append(rand_p)
        hybrid_prob.append(hybrid_p)
        aligned_prob.append(aligned_p)

    fig,ax = plt.subplots(figsize=(7,5))
    ax.plot(np.log10(vth_values),np.mean(sling_prob,axis=0),label="slingshot",color=kick_colors['slingshot'],linewidth=1.5)
    ax.plot(np.log10(vth_values),np.mean(rand_prob,axis=0),label="random",color=kick_colors['random'],linewidth=1.5)
    ax.plot(np.log10(vth_values),np.mean(hybrid_prob,axis=0),label="hybrid",color=kick_colors['hybrid'],linewidth=1.5)
    ax.plot(np.log10(vth_values),np.mean(aligned_prob,axis=0),label="aligned",color=kick_colors['aligned'],linewidth=1.5)

    #ax.axvline(x=np.log10(600),color="grey",linestyle="--")
    ax.legend(fontsize=18)
    ax.set_xlabel(r"$v \, (\log \, \text{km} \, \text{s}^{-1}$)",fontsize=18)
    ax.set_ylabel('$P(v > v_{th})$',fontsize=18)
    ax.set_xlim(0,4)
    #ax.text(np.log10(600)+0.09,1000,'$v = 600 km/s$',fontsize=14)
    return fig,ax

def kick_velocity_dist_plot(strong_tr,weak_tr,iso_bin,Nruns):

    slings=[]
    sling_max = []
    sling_min = []
    sling_g600 =[]

    rands=[]
    rand_max = []
    rand_min = []
    rand_g600 =[]

    hybrids =[]
    hybrid_max = []
    hybrid_g600 = []

    deg5s =[]
    deg5_max = []
    deg5_g600 = []
    vbinsize = 0.17
    vmax = 6

    for i in range(Nruns):
        sling_v = [kick for kick in strong_tr[i].slingshot_kicks if kick > 0]
        sling_v = np.array(sling_v)
        sling_max.append(np.max(sling_v))
        sling_min.append(np.min(sling_v))
        sling_g600.append(len(sling_v[sling_v>600])/len(sling_v))

        rand_v_strong = [kick for kick in strong_tr[i].gw_kick_random if kick > 0]
        rand_v_weak = [kick for kick in weak_tr.gw_kick_random if kick > 0]
        rand_v_iso = [kick for kick in iso_bin.gw_kick_random if kick > 0]
        rand_v = np.concatenate((rand_v_strong,rand_v_iso,rand_v_weak))
        rand_v = np.array(rand_v)
        rand_max.append(np.max(rand_v))
        rand_min.append(np.min(rand_v))
        rand_g600.append(len(rand_v[rand_v>600])/len(rand_v))

        hybrid_v_strong = [kick for kick in strong_tr[i].gw_kick_hybrid if kick > 0]
        hybrid_v_weak = [kick for kick in weak_tr.gw_kick_hybrid if kick > 0]
        hybrid_v_iso = [kick for kick in iso_bin.gw_kick_hybrid if kick > 0]
        hybrid_v = np.concatenate((hybrid_v_strong,hybrid_v_iso,hybrid_v_weak))
        hybrid_v = np.array(hybrid_v)
        hybrid_max.append(np.max(hybrid_v))
        hybrid_g600.append(len(hybrid_v[hybrid_v>600])/len(hybrid_v))        

        deg5_v_strong = [kick for kick in strong_tr[i].gw_kick_aligned if kick > 0]
        deg5_v_weak = [kick for kick in weak_tr.gw_kick_aligned if kick > 0]
        deg5_v_iso = [kick for kick in iso_bin.gw_kick_aligned if kick > 0]
        deg5_v = np.concatenate((deg5_v_strong,deg5_v_iso,deg5_v_weak))
        deg5_v = np.array(deg5_v)
        deg5_max.append(np.max(deg5_v))
        deg5_g600.append(len(deg5_v[deg5_v>600])/len(deg5_v))       

        sling_vcount,vbin_edges=np.histogram(np.log10(sling_v),range=(0,vmax),bins = int(vmax/vbinsize),density=False)
        rand_vcount,vbin_edges=np.histogram(np.log10(rand_v),range=(0,vmax),bins = int(vmax/vbinsize),density=False)
        hybrid_vcount,vbin_edges=np.histogram(np.log10(hybrid_v),range=(0,vmax),bins = int(vmax/vbinsize),density=False)
        deg5_vcount,vbin_edges=np.histogram(np.log10(deg5_v),range=(0,vmax),bins = int(vmax/vbinsize),density=False)
        vbins = vbin_edges[:-1] + 0.5*vbinsize
        
        slings.append(sling_vcount)
        rands.append(rand_vcount)
        hybrids.append(hybrid_vcount)
        deg5s.append(deg5_vcount)       

    print("Fraction of sling velocity abover 600 km/s is %3.2f %%"%(np.mean(sling_g600)*100))
    print("Fraction of random velocity abover 600 km/s is %3.2f %%"%(np.mean(rand_g600)*100))
    print("Fraction of hybrid velocity abover 600 km/s is %3.2f %%"%(np.mean(hybrid_g600)*100))
    print("Fraction of aligned velocity abover 600 km/s is %3.2f %%"%(np.mean(deg5_g600)*100))
    print("--------")
    print("The maximum sling kick is %3.2f km/s"%(np.mean(sling_max)))
    print("The minimum sling kick is %3.2f km/s"%(np.mean(sling_min)))
    print("--------")
    print("The maximum random kick is %3.2f km/s"%(np.mean(rand_max)))
    print("The minimum random kick is %3.2f km/s"%(np.mean(rand_min)))
    print("--------")
    print("The maximum hybrid kick is %3.2f km/s"%(np.mean(hybrid_max)))
    print("The maximum aligned kick is %3.2f km/s"%(np.mean(deg5_max)))

    fig,ax = plt.subplots(figsize=(7,5))
    ax.plot(vbins,np.mean(slings,axis=0),label="slingshot",color=kick_colors['slingshot'],linewidth=1.5)
    ax.plot(vbins,np.mean(rands,axis=0),label="random",color=kick_colors['random'],linewidth=1.5)
    ax.plot(vbins,np.mean(hybrids,axis=0),label="hybrid",color=kick_colors['hybrid'],linewidth=1.5)
    ax.plot(vbins,np.mean(deg5s,axis=0),label="aligned",color=kick_colors['aligned'],linewidth=1.5)
    ax.axvline(x=np.log10(600),color="grey",linestyle="--")
    ax.legend(fontsize=16)
    ax.set_xlabel(r"$v \, (\log \, \text{km} \, \text{s}^{-1}$)",fontsize=18)
    ax.set_ylabel("Count",fontsize=18)
    ax.set_xlim(0,4)
    ax.text(np.log10(600)+0.09,1000,'$v = 600 km/s$',fontsize=14)

    return fig,ax

def set_plot_style(linewidth=3, titlesize=20,labelsize=25,ticksize=20,legendsize=20,bold=True):
        """Set matplotlib rcParams for consistent plot style."""
        font_weight = 'bold' if bold else 'normal'

        plt.rcParams.update({
            'lines.linewidth': linewidth,
            'axes.labelsize': labelsize,
            'axes.titlesize': titlesize,
            'xtick.labelsize': ticksize,
            'ytick.labelsize': ticksize,
            'legend.fontsize': legendsize,
            'axes.titleweight': font_weight,
            'axes.labelweight': font_weight,
            'font.weight': font_weight,
        })

def import_objects(Nruns,obj_dir="/orange/lblecha/pranavsatheesh/trip_mbh_objs/"):

    iso_filename = obj_dir+'iso_bin_wkick.pkl'
    weak_tr_filename = obj_dir+'weak_tr_wkick.pkl'
    strong_tr_filename = obj_dir+f'tr{Nruns}_wkick.pkl'
    stalled_tr_filename= obj_dir +f'stalled{Nruns}.pkl'

    # iso_filename = os.path.abspath('../obj_data/iso_bin_wkick.pkl')
    # weak_tr_filename = os.path.abspath('../obj_data/weak_tr_wkick.pkl')
    # strong_tr_filename =os.path.abspath(f'../obj_data/tr{Nruns}_wkick.pkl')
    # stalled_tr_filename=os.path.abspath(f'../obj_data/stalled{Nruns}.pkl')

    with open(iso_filename, 'rb') as f:
        iso_bin = pickle.load(f)

    with open(weak_tr_filename, 'rb') as f:
        weak_tr = pickle.load(f)

    with open(strong_tr_filename, 'rb') as f:
        strong_tr = pickle.load(f)

    with open(stalled_tr_filename, 'rb') as f:
        stalled_objs = pickle.load(f)

    return strong_tr, weak_tr, iso_bin, stalled_objs

def kick_velocity_distribution(Nruns):
    strong_tr,weak_tr,iso_bin,stalled_objs = import_objects(Nruns)

    velocity_bins_gwrecoil = np.logspace(1,4,50) 
    velocity_bins_sling = np.logspace(1.7,4,20) 

    slingshot_kick_counts = []
    for i in range(Nruns):
        sling_kick_N, _ = np.histogram(strong_tr[i].slingshot_kicks,bins=velocity_bins_sling)
        slingshot_kick_counts.append(sling_kick_N)
    
    mean_sling = np.mean(slingshot_kick_counts,axis=0)
    std_sling = np.std(slingshot_kick_counts,axis=0)

    rand_kick_counts= kickcount.Nvkicks(iso_bin,weak_tr,strong_tr,Nruns,velocity_bins_gwrecoil,kick_type="v_kick_random")
    hybrid_kick_counts= kickcount.Nvkicks(iso_bin,weak_tr,strong_tr,Nruns,velocity_bins_gwrecoil,kick_type="v_kick_hybrid")
    aligned_kick_counts = kickcount.Nvkicks(iso_bin,weak_tr,strong_tr,Nruns,velocity_bins_gwrecoil,kick_type="v_kick_aligned")

    kick_types_data = {
    'random': (rand_kick_counts, velocity_bins_gwrecoil),
    'hybrid': (hybrid_kick_counts, velocity_bins_gwrecoil),
    'aligned': (aligned_kick_counts, velocity_bins_gwrecoil),
    'slingshot': (slingshot_kick_counts, velocity_bins_sling)
    }

    fig,ax = kickcount.plot_kick_distribution(kick_types_data)
    return fig,ax
    
# kick statistics

def kick_statistics(Nruns):
    strong_tr, weak_tr, iso_bin, stalled_objs = import_objects(Nruns)

    # Calculate the maximum kick values
    max_kicks = max_kick_values(strong_tr, weak_tr, iso_bin, Nruns)
    print("--------")
    
    # Calculate the median kick values
    median_kicks = median_kick_values(strong_tr, weak_tr, iso_bin, Nruns)
    print("--------")
    # Calculate the kick percentiles
    kick_percentiles_results = kick_percentiles(strong_tr, weak_tr, iso_bin, Nruns)

    return max_kicks, median_kicks, kick_percentiles_results

def max_kick_values(strong_tr, weak_tr, iso_bin, Nruns):
    max_kicks = {
        'slingshot': [],
        'random': [],
        'hybrid': [],
        'aligned': []
    }

    for i in range(Nruns):
        max_kicks['slingshot'].append(np.max(strong_tr[i].slingshot_kicks))
        max_kicks['random'].append(np.max(np.concatenate((iso_bin.v_kick_random, weak_tr.v_kick_random, strong_tr[i].v_kick_random), axis=1)))
        max_kicks['aligned'].append(np.max(np.concatenate((iso_bin.v_kick_aligned, weak_tr.v_kick_aligned, strong_tr[i].v_kick_aligned), axis=1)))
        max_kicks['hybrid'].append(np.max(np.concatenate((iso_bin.v_kick_hybrid, weak_tr.v_kick_hybrid, strong_tr[i].v_kick_hybrid), axis=1)))
    
    #Also the standard errors

    std_err_random = np.std(max_kicks['random']) / np.sqrt(len(max_kicks['random']))
    std_err_aligned = np.std(max_kicks['aligned']) / np.sqrt(len(max_kicks['aligned']))
    std_err_hybrid = np.std(max_kicks['hybrid']) / np.sqrt(len(max_kicks['hybrid']))
    std_err_slingshot = np.std(max_kicks['slingshot']) / np.sqrt(len(max_kicks['slingshot']))

    print(f"The maximum kick produced by GW recoil using random spins is {np.mean(max_kicks['random']):.2f} ± {std_err_random:.2f} km/s")
    print(f"The maximum kick produced by GW recoil using aligned spins is {np.mean(max_kicks['aligned']):.2f} ± {std_err_aligned:.2f} km/s")
    print(f"The maximum kick produced by GW recoil using hybrid spins is {np.mean(max_kicks['hybrid']):.2f} ± {std_err_hybrid:.2f} km/s")
    print(f"The maximum kick produced by slingshot using random spins is {np.mean(max_kicks['slingshot']):.2f} ± {std_err_slingshot:.2f} km/s")

    return max_kicks
    
def median_kick_values(strong_tr, weak_tr, iso_bin, Nruns):
    median_kicks = {
        'slingshot': [],
        'random': [],
        'hybrid': [],
        'aligned': []
    }

    for i in range(Nruns):
        median_kicks['slingshot'].append(np.median(strong_tr[i].slingshot_kicks))
        median_kicks['random'].append(np.median(np.concatenate((iso_bin.v_kick_random, 
                                                     weak_tr.v_kick_random, 
                                                     strong_tr[i].v_kick_random), axis=1)))
        median_kicks['hybrid'].append(np.median(np.concatenate((iso_bin.v_kick_hybrid,
                                                     weak_tr.v_kick_hybrid, 
                                                     strong_tr[i].v_kick_hybrid), axis=1)))
        median_kicks['aligned'].append(np.median(np.concatenate((iso_bin.v_kick_aligned,
                                                     weak_tr.v_kick_aligned, 
                                                     strong_tr[i].v_kick_aligned), axis=1)))
        
    std_err_random = np.std(median_kicks['random']) / np.sqrt(len(median_kicks['random']))
    std_err_aligned = np.std(median_kicks['aligned']) / np.sqrt(len(median_kicks['aligned']))
    std_err_hybrid = np.std(median_kicks['hybrid']) / np.sqrt(len(median_kicks['hybrid']))
    std_err_slingshot = np.std(median_kicks['slingshot']) / np.sqrt(len(median_kicks['slingshot']))

    print(f"The median kick produced by GW recoil using random spins is {np.mean(median_kicks['random']):.2f} ± {std_err_random:.2f} km/s")
    print(f"The median kick produced by GW recoil using aligned spins is {np.mean(median_kicks['aligned']):.2f} ± {std_err_aligned:.2f} km/s")
    print(f"The median kick produced by GW recoil using hybrid spins is {np.mean(median_kicks['hybrid']):.2f} ± {std_err_hybrid:.2f} km/s")
    print(f"The median kick produced by slingshot using random spins is {np.mean(median_kicks['slingshot']):.2f} ± {std_err_slingshot:.2f} km/s")

    return median_kicks

def kick_percentiles(strong_tr, weak_tr, iso_bin, Nruns, thresholds=(500, 1000)):
    """
    Calculate the percentage of kicks above given thresholds for each kick model.

    Parameters:
        strong_tr: list of strong triple objects (length Nruns)
        weak_tr: weak triple object
        iso_bin: isolated binary object
        Nruns: number of runs
        n_slingshot_run: number of triples realization (default 520)
        thresholds: tuple of thresholds (default (500, 1000))

    Returns:
        results: dict with mean and std for each model and threshold
    """
    results = {}
    Nslingshot_kick = len(strong_tr[0].slingshot_kicks)
    for thresh in thresholds:
        random_percentile = []
        random_percentile_std = []
        hybrid_percentile = []
        hybrid_percentile_std = []
        aligned_percentile = []
        aligned_percentile_std = []
        slingshot_percentile = []

        for i in range(Nruns):
            # Total number of kicks
            N_kicks_i = (
                np.shape(iso_bin.v_kick_random)[1] +
                np.shape(weak_tr.v_kick_random)[1] +
                np.shape(strong_tr[i].v_kick_random)[1]
            )

            # Random kicks above threshold
            N_rand_above = (
                np.sum(np.array(iso_bin.v_kick_random) > thresh, axis=1) +
                np.sum(np.array(weak_tr.v_kick_random) > thresh, axis=1) +
                np.sum(np.array(strong_tr[i].v_kick_random) > thresh, axis=1)
            )
            random_percentile.append(np.mean(N_rand_above / N_kicks_i * 100))
            random_percentile_std.append(np.std(N_rand_above / N_kicks_i * 100))

            # Hybrid kicks above threshold
            N_hybrid_above = (
                np.sum(np.array(iso_bin.v_kick_hybrid) > thresh, axis=1) +
                np.sum(np.array(weak_tr.v_kick_hybrid) > thresh, axis=1) +
                np.sum(np.array(strong_tr[i].v_kick_hybrid) > thresh, axis=1)
            )
            hybrid_percentile.append(np.mean(N_hybrid_above / N_kicks_i * 100))
            hybrid_percentile_std.append(np.std(N_hybrid_above / N_kicks_i * 100))

            # Aligned kicks above threshold
            N_aligned_above = (
                np.sum(np.array(iso_bin.v_kick_aligned) > thresh, axis=1) +
                np.sum(np.array(weak_tr.v_kick_aligned) > thresh, axis=1) +
                np.sum(np.array(strong_tr[i].v_kick_aligned) > thresh, axis=1)
            )
            aligned_percentile.append(np.mean(N_aligned_above / N_kicks_i * 100))
            aligned_percentile_std.append(np.std(N_aligned_above / N_kicks_i * 100))

            # Slingshot kicks above threshold (only for strong_tr[i])
            N_slingshot_above = np.sum(np.array(strong_tr[i].slingshot_kicks) > thresh)
            slingshot_percentile.append(N_slingshot_above /Nslingshot_kick * 100)

        results[thresh] = {
            'random': (np.mean(random_percentile), np.mean(random_percentile_std)),
            'hybrid': (np.mean(hybrid_percentile), np.mean(hybrid_percentile_std)),
            'aligned': (np.mean(aligned_percentile), np.mean(aligned_percentile_std)),
            'slingshot': (np.mean(slingshot_percentile), np.std(slingshot_percentile)),
        }

    # Print results
    for thresh in thresholds:
        print(f"The % of kicks above {thresh} in random model is {results[thresh]['random'][0]:.2f} ± {results[thresh]['random'][1]:.2f} %")
        print(f"The % of kicks above {thresh} in hybrid model is {results[thresh]['hybrid'][0]:.2f} ± {results[thresh]['hybrid'][1]:.2f} %")
        print(f"The % of kicks above {thresh} in aligned model is {results[thresh]['aligned'][0]:.2f} ± {results[thresh]['aligned'][1]:.2f} %")
        print(f"The % of kicks above {thresh} in slingshot model is {results[thresh]['slingshot'][0]:.2f} ± {results[thresh]['slingshot'][1]:.2f} %")
        print("----------")

    return results

def count_kicks_in_range(v_min, v_max, Nruns, iso_bin, weak_tr, strong_tr):
    """
    Count the average number of kicks in the velocity range [v_min, v_max] for each kick model.

    Parameters:
        v_min (float): Minimum velocity (inclusive).
        v_max (float): Maximum velocity (inclusive).
        Nruns (int): Number of runs.
        iso_bin: Isolated binary object.
        weak_tr: Weak triple object.
        strong_tr: List of strong triple objects (length Nruns).

    Returns:
        dict: Average number of kicks in the range for each model.
    """
    kicks_in_range_random = []
    kicks_in_range_hybrid = []
    kicks_in_range_aligned = []
    kicks_in_range_slingshot = []

    for i in range(Nruns):
        # Random spins
        random_kicks = np.concatenate((iso_bin.v_kick_random, 
                                       weak_tr.v_kick_random, 
                                       strong_tr[i].v_kick_random), axis=1)
        kicks_in_range_random.append(np.sum((random_kicks >= v_min) & (random_kicks <= v_max)))

        # Hybrid spins
        hybrid_kicks = np.concatenate((iso_bin.v_kick_hybrid, 
                                       weak_tr.v_kick_hybrid, 
                                       strong_tr[i].v_kick_hybrid), axis=1)
        kicks_in_range_hybrid.append(np.sum((hybrid_kicks >= v_min) & (hybrid_kicks <= v_max)))

        # Aligned spins
        aligned_kicks = np.concatenate((iso_bin.v_kick_aligned, 
                                        weak_tr.v_kick_aligned, 
                                        strong_tr[i].v_kick_aligned), axis=1)
        kicks_in_range_aligned.append(np.sum((aligned_kicks >= v_min) & (aligned_kicks <= v_max)))

        # Slingshot kicks
        slingshot_kicks = np.array(strong_tr[i].slingshot_kicks)
        kicks_in_range_slingshot.append(np.sum((slingshot_kicks >= v_min) & (slingshot_kicks <= v_max)))

    # Print the results

    kicks_in_range_results = { 
        "random": np.mean(kicks_in_range_random),
        "hybrid": np.mean(kicks_in_range_hybrid),
        "aligned": np.mean(kicks_in_range_aligned),
        "slingshot": np.mean(kicks_in_range_slingshot)
    }
    
    print(f"The number of kicks in the {v_min}-{v_max} km/s range for random spins is {kicks_in_range_results['random']}")
    print(f"The number of kicks in the {v_min}-{v_max} km/s range for hybrid spins is {kicks_in_range_results['hybrid']}")
    print(f"The number of kicks in the {v_min}-{v_max} km/s range for aligned spins is {kicks_in_range_results['aligned']}")
    print(f"The number of kicks in the {v_min}-{v_max} km/s range for slingshot is {kicks_in_range_results['slingshot']}")

    
    return kicks_in_range_results

def compute_ejection_percentages(Nruns, iso_bin, weak_tr, strong_tr):
    random_ejections = []
    hybrid_ejections = []
    aligned_ejections = []
    slingshot_ejections = []

    for i in range(Nruns):
        # Total number of kicks
        N_kicks = (np.shape(iso_bin.v_kick_random)[1]) + (np.shape(weak_tr.v_kick_random)[1]) + (np.shape(strong_tr[i].v_kick_random)[1])
        N_sling_kicks = 520  # Slingshot kicks per iteration

        # Calculate ejections for random spins
        random_eject = (
            np.sum(iso_bin.v_kick_random > iso_bin.Vescape[iso_bin.merger_mask], axis=1) +
            np.sum(weak_tr.v_kick_random > weak_tr.Vescape[weak_tr.merger_mask], axis=1) +
            np.sum(strong_tr[i].v_kick_random > strong_tr[i].Vescape[strong_tr[i].merger_mask], axis=1)
        ) / N_kicks * 100
        random_ejections.append(random_eject)

        # Calculate ejections for hybrid spins
        hybrid_eject = (
            np.sum(iso_bin.v_kick_hybrid > iso_bin.Vescape[iso_bin.merger_mask], axis=1) +
            np.sum(weak_tr.v_kick_hybrid > weak_tr.Vescape[weak_tr.merger_mask], axis=1) +
            np.sum(strong_tr[i].v_kick_hybrid > strong_tr[i].Vescape[strong_tr[i].merger_mask], axis=1)
        ) / N_kicks * 100
        hybrid_ejections.append(hybrid_eject)

        # Calculate ejections for aligned spins
        aligned_eject = (
            np.sum(iso_bin.v_kick_aligned > iso_bin.Vescape[iso_bin.merger_mask], axis=1) +
            np.sum(weak_tr.v_kick_aligned > weak_tr.Vescape[weak_tr.merger_mask], axis=1) +
            np.sum(strong_tr[i].v_kick_aligned > strong_tr[i].Vescape[strong_tr[i].merger_mask], axis=1)
        ) / N_kicks * 100
        aligned_ejections.append(aligned_eject)

        # Slingshot kicks
        slingshot_eject = np.sum(strong_tr[i].slingshot_kicks > strong_tr[i].Vescape) / N_sling_kicks * 100
        slingshot_ejections.append(slingshot_eject)

    print(f"Random ejection percentage: {np.mean(random_ejections):.2f}% ± {np.std(random_ejections):.2f}%")
    print(f"Hybrid ejection percentage: {np.mean(hybrid_ejections):.2f}% ± {np.std(hybrid_ejections):.2f}%")
    print(f"Aligned ejection percentage: {np.mean(aligned_ejections):.2f}% ± {np.std(aligned_ejections):.2f}%")
    print(f"Slingshot ejection percentage: {np.mean(slingshot_ejections):.2f}% ± {np.std(slingshot_ejections):.2f}%")
    return None