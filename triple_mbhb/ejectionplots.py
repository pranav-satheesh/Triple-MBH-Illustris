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
import count_ejections as ejects


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

def ejection_rates_plot(Nruns,merger_rate_path):
    strong_tr,weak_tr,iso_bin,stalled_objs = import_objects(Nruns)    
    ejects.assign_ejection_masks(strong_tr,iso_bin,weak_tr,Nruns)

    N_kick_realization = np.shape(strong_tr[0].v_kick_random)[0]

    lgzbins_random,st_ejection_rate_random = ejects.strong_tr_ejection_rates(strong_tr, Nruns, N_kick_realization, kick_type='random', lgzbinsize=0.25, lgzmin=-3, lgzmax=1.0)
    lgzbins_hybrid,st_ejection_rate_hybrid = ejects.strong_tr_ejection_rates(strong_tr, Nruns, N_kick_realization, kick_type='hybrid',lgzbinsize=0.4,lgzmin=-3,lgzmax=1.0)
    lgzbins_aligned,st_ejection_rate_aligned = ejects.strong_tr_ejection_rates(strong_tr, Nruns, N_kick_realization, kick_type='aligned',lgzbinsize=0.4,lgzmin=-3,lgzmax=1.0)
    lgzbins_slingshot,st_ejection_rate_slingshot = ejects.strong_tr_ejection_rates(strong_tr, Nruns, N_kick_realization, kick_type='slingshot', lgzbinsize=0.4, lgzmin=-3, lgzmax=1.0)

    lgzbins_random_tot, total_ejection_rate_random = ejects.tot_population_ejection_rates(strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, kick_type="random",lgzbinsize=0.18,lgzmin=-3.2,lgzmax=1.0)
    lgzbins_hybrid_tot, total_ejection_rate_hybrid = ejects.tot_population_ejection_rates(strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, kick_type="hybrid",lgzbinsize=0.4,lgzmin=-3,lgzmax=1.0)
    lgzbins_aligned_tot, total_ejection_rate_aligned = ejects.tot_population_ejection_rates(strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, kick_type="aligned",lgzbinsize=0.2,lgzmin=-3,lgzmax=1.0)

    kick_colors = {
        'slingshot': "#4daf4a",  # Green
        'aligned': "#377eb8",    # Blue
        'hybrid': "#a2c8ec",     # Light blue
        'random': "#e41a1c"      # Red
    }

    kick_linestyles = {
            'slingshot': "-",        # Solid line
            'aligned': "--",         # Dashed line
            'hybrid': "-.",          # Dash-dot line
            'random': ":"            # Dotted line
        }


    fig,ax = plt.subplots(1,2,figsize=(16,7),sharey=True)

    lgz_bins,all_mr = np.loadtxt(merger_rate_path+"all_system_merger_rates.txt")    
    ax[0].plot(lgz_bins,all_mr,color="black",linestyle="--",label="Merger rate")
    ax[0].plot(lgzbins_random_tot,np.mean(total_ejection_rate_random,axis=0),color=kick_colors['random'],label="random",linestyle=kick_linestyles['random'])
    ax[0].plot(lgzbins_hybrid_tot,np.mean(total_ejection_rate_hybrid,axis=0),color=kick_colors['hybrid'],label="hybrid",linestyle=kick_linestyles['hybrid'])
    ax[0].plot(lgzbins_aligned_tot,np.mean(total_ejection_rate_aligned,axis=0),color=kick_colors['aligned'],label="aligned",linestyle=kick_linestyles['aligned'])
    ax[0].plot(lgzbins_slingshot,np.mean(st_ejection_rate_slingshot,axis=0),color=kick_colors['slingshot'],label="slingshot",linestyle=kick_linestyles['slingshot'])
    #ax[0].legend()
    ax[0].set_title("Total population")
    ax[0].set_yticks([1e-1,1e-3,1e-5,1e-7])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("$\log z$")
    ax[0].set_ylabel(r"$\log (d^2 N / (d \log z dt)  \times 1\text{yr})$")

    lgztrip_bins,alltrip_mr = np.loadtxt(merger_rate_path+"triple_system_merger_rates.txt")

    ax[1].plot(lgztrip_bins,alltrip_mr,color="black",linestyle="--",label="Merger rate")
    ax[1].plot(lgzbins_random,np.mean(st_ejection_rate_random,axis=0),color=kick_colors['random'],label="random",linestyle=kick_linestyles['random'])
    ax[1].plot(lgzbins_hybrid,np.mean(st_ejection_rate_hybrid,axis=0),color=kick_colors['hybrid'],label="hybrid",linestyle=kick_linestyles['hybrid'])
    ax[1].plot(lgzbins_aligned,np.mean(st_ejection_rate_aligned,axis=0),color=kick_colors['aligned'],label="aligned",linestyle=kick_linestyles['aligned'])
    ax[1].plot(lgzbins_slingshot,np.mean(st_ejection_rate_slingshot,axis=0),color=kick_colors['slingshot'],label="slingshot",linestyle=kick_linestyles['slingshot'])

    ax[1].set_ylim(5*10**-7,10**(-1))
    ax[1].set_yscale("log")
    ax[1].set_xlabel("$\log z$")
    ax[1].legend()
    ax[1].set_title("Strong triples")

    return fig,ax