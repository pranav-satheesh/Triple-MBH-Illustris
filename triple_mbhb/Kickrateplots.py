import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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

        deg5_v_strong = [kick for kick in strong_tr[i].gw_kick_5deg if kick > thresh]
        deg5_v_weak = [kick for kick in weak_tr.gw_kick_5deg if kick > thresh]
        deg5_v_iso = [kick for kick in iso_bin.gw_kick_5deg if kick > thresh]
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
        dNv_deg5_strong, vbin_edges_deg5 = np.histogram(strong_tr[i].gw_kick_5deg, 
                                                          range=(0, v_max_values['deg5']), 
                                                          bins=int(v_max_values['deg5'] / vbin_sizes['deg5']))
        dNv_deg5_weak, _ = np.histogram(weak_tr.gw_kick_5deg, 
                                         range=(0, v_max_values['deg5']), 
                                         bins=int(v_max_values['deg5'] / vbin_sizes['deg5']))
        dNv_deg5_binary, _ = np.histogram(iso_bin.gw_kick_5deg, 
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

# def dNbydV(strong_tr,weak_tr,iso_bin,v_max,vbin_size,Nruns):

#     dNbydV_sling = []
#     dNbydV_rand = []
#     dNbydV_hybrid = []
#     dNbydV_deg5 = []

#     for i in range(Nruns):
#         Nvslingshot,vbin_edges = np.histogram(strong_tr[i].slingshot_kicks,range=(0,v_max),bins=int(v_max/vbin_size))
#         vbins = vbin_edges[:-1] + 0.5*vbin_size
#         dNbydV_sling.append(Nvslingshot/vbin_size)

#         dNv_rand_strong,tmp = np.histogram(strong_tr[i].gw_kick_random,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNv_rand_weak,tmp = np.histogram(weak_tr.gw_kick_random,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNv_rand_binary,tmp = np.histogram(iso_bin.gw_kick_random,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNbydV_rand.append((dNv_rand_binary+dNv_rand_strong+dNv_rand_weak)/vbin_size)

#         dNv_hybrid_strong,tmp = np.histogram(strong_tr[i].gw_kick_hybrid,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNv_hybrid_weak,tmp = np.histogram(weak_tr.gw_kick_hybrid,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNv_hybrid_binary,tmp = np.histogram(iso_bin.gw_kick_hybrid,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNbydV_hybrid.append((dNv_hybrid_binary+dNv_hybrid_strong+dNv_hybrid_weak)/vbin_size)

#         dNv_deg5_strong,tmp = np.histogram(strong_tr[i].gw_kick_5deg,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNv_deg5_weak,tmp = np.histogram(weak_tr.gw_kick_5deg,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNv_deg5_binary,tmp = np.histogram(iso_bin.gw_kick_5deg,range=(0,v_max),bins=int(v_max/vbin_size))
#         dNbydV_deg5.append((dNv_deg5_binary+dNv_deg5_strong+dNv_deg5_weak)/vbin_size)  

#     dNbydV_sling = np.array(dNbydV_sling)/vol_comov_box
#     dNbydV_hybrid = np.array(dNbydV_hybrid)/vol_comov_box
#     dNbydV_rand = np.array(dNbydV_rand)/vol_comov_box
#     dNbydV_deg5 = np.array(dNbydV_deg5)/vol_comov_box

#     return vbins,dNbydV_sling,dNbydV_rand,dNbydV_hybrid,dNbydV_deg5        

# def plot_rate_kicks(strong_tr,weak_tr,iso_bin,v_max,vbin_size,Nruns):
#     fig,ax = plt.subplots(figsize=(7,6))


#     vbins,dNdVsling,dNdVrand,dNdVhybrid,dNdVdeg5 = dNbydV(strong_tr,weak_tr,iso_bin,v_max,vbin_size,Nruns)
#     ax.plot(vbins,np.mean(dNdVsling,axis=0),linestyle='-', marker='o',color=kick_colors["slingshot"],label="slingshot")
#     ax.fill_between(vbins,np.mean(dNdVsling,axis=0)-np.std(dNdVsling,axis=0),np.mean(dNdVsling,axis=0)+np.std(dNdVsling,axis=0),color=kick_colors["slingshot"],alpha=0.5)

#     ax.plot(vbins,np.mean(dNdVrand,axis=0),linestyle='-', marker='o',color=kick_colors["random"],label="random")
#     ax.fill_between(vbins,np.mean(dNdVrand,axis=0)-np.std(dNdVrand,axis=0),np.mean(dNdVrand,axis=0)+np.std(dNdVrand,axis=0),color=kick_colors["random"],alpha=0.5) 

#     ax.plot(vbins,np.mean(dNdVhybrid,axis=0),linestyle='-', marker='o',color=kick_colors["hybrid"],label="hybrid")
#     ax.fill_between(vbins,np.mean(dNdVhybrid,axis=0)-np.std(dNdVhybrid,axis=0),np.mean(dNdVhybrid,axis=0)+np.std(dNdVhybrid,axis=0),color=kick_colors["hybrid"],alpha=0.5)  

#     ax.plot(vbins,np.mean(dNdVdeg5,axis=0),linestyle='-', marker='o',color=kick_colors['aligned'],label="aligned")
#     ax.fill_between(vbins,np.mean(dNdVdeg5,axis=0)-np.std(dNdVdeg5,axis=0),np.mean(dNdVdeg5,axis=0)+np.std(dNdVdeg5,axis=0),color="#0571b0",alpha=0.5)       
#     ax.set_yscale("log",base=10)
#     ax.set_xlabel("$v_t$")
#     ax.set_ylabel("$R_{kicks} [Gpc^{-3} km^{-1} s]$")
#     ax.legend()       
#     ax.set_xlim(0,v_max)
#     #ax.set_ylim(1e-,)
#     fig.tight_layout()
    

#     return fig,ax

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
    ax[1].plot(a_rd_bins,a_rd_counts,color=color_palette["random"],lw=2,label="random-dry")
    ax[1].plot(a_cold_bins,a_cold_counts,color=color_palette["cold"],linestyle="--",lw=2,label="cold")
    ax[1].axvline(x=0.9,color=color_palette["aligned"],lw=2,label="aligned")
    ax[1].set_xlim(0,1)
    ax[1].set_ylim(0,)
    ax[1].legend(fontsize=20)
    ax[1].set_ylabel(r"$P(a)$",fontsize=20)
    ax[1].set_xlabel(r"$a$",fontsize=20)
    ax[1] = plt.gca()
    xticks = ax[1].xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    ax[0].plot(theta_rd_bins,theta_rd_counts,color="#0571b0",lw=2,label="random-dry")
    ax[0].plot(cold_bins,cold_counts,color="#92c5de",lw=2,linestyle="--",label="cold")
    ax[0].plot(theta5deg_bins,theta5deg_counts,color="#ca0020",lw=2,label="aligned")
    ax[0].set_xlim(0.5,90)
    ax[0].set_ylim(0,)
    ax[0].legend(fontsize=20)
    ax[0].set_ylabel(r"$P(\theta_{1,2})$",fontsize=20)
    ax[0].set_xlabel(r"$\theta_{1,2} [^{\circ}]$",fontsize=20)
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

        deg5_v_strong = [kick for kick in strong_tr[i].gw_kick_5deg if kick > 0]
        deg5_v_weak = [kick for kick in weak_tr.gw_kick_5deg if kick > 0]
        deg5_v_iso = [kick for kick in iso_bin.gw_kick_5deg if kick > 0]
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

        deg5_v_strong = [kick for kick in strong_tr[i].gw_kick_5deg if kick > 0]
        deg5_v_weak = [kick for kick in weak_tr.gw_kick_5deg if kick > 0]
        deg5_v_iso = [kick for kick in iso_bin.gw_kick_5deg if kick > 0]
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




