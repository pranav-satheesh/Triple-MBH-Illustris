import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


# tex_path = '/apps/texlive/2023/bin/x86_64-linux/'
# os.environ['PATH'] += os.pathsep + tex_path
# import scienceplots
# plt.style.use('science')


def M_q_z_dist_for_mergers(strong_tr,weak_tr,iso_bin,Nruns):

    lgqbinsize = 0.25
    lgqbinmax = 0.5
    lgqbinmin = -4.5

    Nqmerger_iso,lgqmerger_edges = np.histogram(np.log10(iso_bin.qin[iso_bin.merger_mask]),density=True,range=(lgqbinmin,lgqbinmax),bins=int((lgqbinmax-lgqbinmin)/lgqbinsize))
    #Nqmerger[Nqmerger==0] = 1.0e-10
    lgqmerger_bins = lgqmerger_edges[:-1]+0.5*lgqbinsize

    Nqmerger_weak,_s = np.histogram(np.log10(weak_tr.qin[weak_tr.merger_mask]),density=True,range=(lgqbinmin,lgqbinmax),bins=int((lgqbinmax-lgqbinmin)/lgqbinsize))
    #Nqmerger[Nqmerger==0] = 1.0e-10
    #lgqmerger_bins = lgqmerger_edges[:-1]+0.5*lgqbinsize

    lgMbinsize = 0.25
    lgMbinmax = 10
    lgMbinmin = 6

    NMmerger_iso,lgMmerger_edges = np.histogram(np.log10(iso_bin.M1[iso_bin.merger_mask]+iso_bin.M2[iso_bin.merger_mask]),density=True,range=(lgMbinmin,lgMbinmax),bins=int((lgMbinmax-lgMbinmin)/lgMbinsize))
    #Nqmerger[Nqmerger==0] = 1.0e-10
    lgMmerger_bins = lgMmerger_edges[:-1]+0.5*lgMbinsize

    NMmerger_weak,_ = np.histogram(np.log10(weak_tr.M1[weak_tr.weak_triple_mask][weak_tr.merger_mask]+weak_tr.M2[weak_tr.weak_triple_mask][weak_tr.merger_mask]),density=True,range=(lgMbinmin,lgMbinmax),bins=int((lgMbinmax-lgMbinmin)/lgMbinsize))
    #Nqmerger[Nqmerger==0] = 1.0e-10
    #lgMmerger_bins = lgqmerger_edges[:-1]+0.5*lgMbinsize

    lgzbinsize = 0.2
    lgzmax = 1
    lgzmin = -3

    Nz_iso,lgz_edges = np.histogram(np.log10(iso_bin.z_merger[iso_bin.merger_mask]),density=True,range=(lgzmin,lgzmax),bins=int((lgzmax-lgzmin)/lgzbinsize))
    #Nqmerger[Nqmerger==0] = 1.0e-10
    lgz_bins = lgz_edges[:-1]+0.5*lgzbinsize

    Nz_weak,_ = np.histogram(np.log10(weak_tr.z_merger[weak_tr.merger_mask]),density=True,range=(lgzmin,lgzmax),bins=int((lgzmax-lgzmin)/lgzbinsize))
    #Nqmerger[Nqmerger==0] = 1.0e-10
    #lgz_bins = lgz_edges[:-1]+0.5*lgzbinsize


    Nqmerger_strong_tot = []
    NMmerger_strong_tot = []
    Nzmerger_strong_tot = []

    for i in range(Nruns):
        Nqmerger_strong,lgqmerger_edges = np.histogram(np.log10(strong_tr[i].qin_merger[strong_tr[i].merger_mask]),density=True,range=(lgqbinmin,lgqbinmax),bins=int((lgqbinmax-lgqbinmin)/lgqbinsize))
        NMmerger_strong,lgMmerger_edges = np.histogram(np.log10(strong_tr[i].mbin_merger[strong_tr[i].merger_mask]),density=True,range=(lgMbinmin,lgMbinmax),bins=int((lgMbinmax-lgMbinmin)/lgMbinsize))
        Nz_strong,_ = np.histogram(np.log10(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask]),density=True,range=(lgzmin,lgzmax),bins=int((lgzmax-lgzmin)/lgzbinsize))
        
        Nqmerger_strong_tot.append(Nqmerger_strong)
        NMmerger_strong_tot.append(NMmerger_strong)
        Nzmerger_strong_tot.append(Nz_strong)

    Nqmerger_strong = np.mean(Nqmerger_strong_tot,axis=0)
    NMmerger_strong = np.mean(NMmerger_strong_tot,axis=0)
    Nz_strong = np.mean(Nzmerger_strong_tot,axis=0)

    lgqmerger_bins = lgqmerger_edges[:-1]+0.5*lgqbinsize
    lgMmerger_bins = lgMmerger_edges[:-1]+0.5*lgMbinsize
    lgz_bins = lgz_edges[:-1]+0.5*lgzbinsize

    return lgqmerger_bins,lgMmerger_bins,lgz_bins,[Nqmerger_iso,Nqmerger_weak,Nqmerger_strong],[NMmerger_iso,NMmerger_weak,NMmerger_strong],[Nz_iso,Nz_weak,Nz_strong]

def median_values_q_M_z(strong_tr,weak_tr,iso_bin,Nruns):

    print(f"q merger mean for iso:{np.median(iso_bin.qin[iso_bin.merger_mask]):.3f}")
    print(f"qin merger for weak:{np.median(weak_tr.qin[weak_tr.merger_mask]):.3f}")
    qin_merger_median_strong = []
    for i in range(Nruns):
        qin_merger_median_strong.append(np.median(strong_tr[i].qin_merger[strong_tr[i].merger_mask]))

    print(f"qin merger for strong:{np.mean(qin_merger_median_strong):.3f}")
    print("----------------------------------------")

    print(f"M mean for iso:{np.median(np.log10(iso_bin.M1[iso_bin.merger_mask]+iso_bin.M2[iso_bin.merger_mask])):.2f}")
    print(f"M mean for weak:{np.median(np.log10(weak_tr.M1[weak_tr.weak_triple_mask][weak_tr.merger_mask]+weak_tr.M2[weak_tr.weak_triple_mask][weak_tr.merger_mask])):.3f}")
    M_merger_median_strong = []
    for i in range(Nruns):
        M_merger_median_strong.append(np.median(np.log10(strong_tr[i].mbin_merger[strong_tr[i].merger_mask])))
    print(f"M mean for strong:{np.mean(M_merger_median_strong):.3f}")
    print("------------------------------------------")

    z_min_strong = []
    z_max_strong = []

    for i in range(Nruns):
        z_min_strong.append(np.min(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask]))
        z_max_strong.append(np.max(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask]))
    
    
    print(f"z min for strong:{np.mean(z_min_strong):.3f}")
    print(f"z max for strong:{np.mean(z_max_strong):.3f}")

    z_min_iso = np.min(iso_bin.z_merger[iso_bin.merger_mask])
    z_max_iso = np.max(iso_bin.z_merger[iso_bin.merger_mask])
    print(f"z min for iso binary {z_min_iso:.3f}")
    print(f"z max for iso binary {z_max_iso:.3f}")

    print("------------------------------------------")

    return None 

def plot_hist_q_M_z(lgq_bins, lgM_bins, lgz_bins, Nqmerger, NMmerger, Nzmerger):
    color_palette = {
        "strong_trip": "#377eb8",
        "weak_trip": "#a2c8ec",
        "iso": "#ff800e",
        "all": "#898989",
        "stalled": 'red'
    }

    fig, ax = plt.subplots(1, 3, figsize=[10, 4], sharey=True)

    # Plot q distribution
    ax[0].step(lgq_bins, Nqmerger[0], label="isolated binaries", color=color_palette["iso"])
    ax[0].step(lgq_bins, Nqmerger[1], label="weak triples", color=color_palette["weak_trip"])
    ax[0].step(lgq_bins, Nqmerger[2], label="strong triples", color=color_palette["strong_trip"])
    ax[0].set_xticks([-4, -3, -2, -1, 0])
    ax[0].set_xlim(-4.2, 0)
    ax[0].set_ylim(0,)
    ax[0].set_xlabel(r"$\log(q_{\text{mrg}})$")
    ax[0].set_ylabel("pdf")

    # Plot M distribution
    ax[1].step(lgM_bins, NMmerger[0], label="isolated binaries", color=color_palette["iso"])
    ax[1].step(lgM_bins, NMmerger[1], label="weak triples", color=color_palette["weak_trip"])
    ax[1].step(lgM_bins, NMmerger[2], label="strong triples", color=color_palette["strong_trip"])
    ax[1].set_xlabel(r"$\log(M_{\text{mrg}})$")
    ax[1].set_xticks([6, 7, 8, 9, 10])

    # Plot z distribution
    ax[2].step(lgz_bins, Nzmerger[0], label="isolated binaries", color=color_palette["iso"])
    ax[2].step(lgz_bins, Nzmerger[1], label="weak triples", color=color_palette["weak_trip"])
    ax[2].step(lgz_bins, Nzmerger[2], label="strong triples", color=color_palette["strong_trip"])
    ax[2].set_xlabel(r"$\log(z)$")
    ax[2].set_xticks([-3, -2, -1, 0, 1])

    # Add legend and adjust layout
    ax[0].legend()
    fig.tight_layout()

    return fig, ax

def strong_Tr_major_mergers(Nruns,strong_tr):
    st_major_merger = []
    for i in range(Nruns):
        st_major_merger.append(np.sum(strong_tr[i].qin_merger[strong_tr[i].merger_mask]>0.1)/520 * 100)
    print(f"The strong triple mergers consist of {np.mean(st_major_merger):.1f}% major mergers")

def strong_Tr_make_2d_hist_qin_qout(Nruns,strong_tr):
    hist_Tr_ej_tot = 0
    hist_Tr_tot = 0
    hist_no_tot = 0

    for i in range(Nruns):
        Tr_ej_qin = strong_tr[i].qin[strong_tr[i].merger_after_ejection_mask]
        Tr_ej_qout = strong_tr[i].qout[strong_tr[i].merger_after_ejection_mask]

        Tr_qin = strong_tr[i].qin[strong_tr[i].prompt_merger_mask]
        Tr_qout = strong_tr[i].qout[strong_tr[i].prompt_merger_mask]

        no_qin = strong_tr[i].qin[strong_tr[i].no_merger_mask]
        no_qout = strong_tr[i].qout[strong_tr[i].no_merger_mask]

        
        xedges =  np.linspace(-3,2,21) #logspace(aa, 21)
        yedges =  np.linspace(-2,0,21)#logspace(bb, 21)

        aa_Tr, bb_Tr = np.log10(Tr_qout),np.log10(Tr_qin)
        hist_Tr, *_ = sp.stats.binned_statistic_2d(aa_Tr, bb_Tr, None, bins=(xedges, yedges), statistic='count')
        hist_Tr_tot = hist_Tr_tot + hist_Tr

        aa_Tr_ej, bb_Tr_ej = np.log10(Tr_ej_qout),np.log10(Tr_ej_qin)
        hist_Tr_ej, *_ = sp.stats.binned_statistic_2d(aa_Tr_ej, bb_Tr_ej, None, bins=(xedges, yedges), statistic='count')
        hist_Tr_ej_tot = hist_Tr_ej_tot + hist_Tr_ej


        aa_no, bb_no = np.log10(no_qout),np.log10(no_qin)
        hist_no, *_ = sp.stats.binned_statistic_2d(aa_no, bb_no, None, bins=(xedges, yedges), statistic='count')
        hist_no_tot = hist_no_tot + hist_no
    
    hist_Tr_ej_tot = hist_Tr_ej_tot/Nruns
    hist_Tr_tot = hist_Tr_tot/Nruns
    hist_no_tot = hist_no_tot/Nruns

    return xedges, yedges, hist_Tr_tot, hist_Tr_ej_tot, hist_no_tot

def plot_Tr_2d_hist_qin_qout(Nruns,strong_tr,scatter_size=3,scatter_alpha=0.5): 


    fig,axes = plt.subplots(1,2,figsize=[16,7])
    xedges, yedges, hist_Tr_tot, hist_Tr_ej_tot, hist_no_tot = strong_Tr_make_2d_hist_qin_qout(Nruns,strong_tr)


    xx, yy = np.meshgrid(xedges, yedges)
    #norm = mpl.colors.Normalize(0,round(hist_Tr_tot.max()))

    max_val = max(hist_Tr_tot.max(), hist_Tr_ej_tot.max())
    norm = mpl.colors.Normalize(0, round(max_val))

    pcm1 = axes[0].pcolormesh(xedges, yedges, hist_Tr_tot.T, cmap="Blues", norm=norm)
    pcm2 = axes[1].pcolormesh(xedges, yedges, hist_Tr_ej_tot.T, cmap="Blues", norm=norm)
    #plt.colorbar(pcm, ax=axes[0], label='number')

    for i in range(Nruns):
        qout_stalled_but_Tr = strong_tr[i].qout[(~strong_tr[i].bin_merge_flag)&(strong_tr[i].prompt_merger_mask)]
        qin_stalled_but_Tr = strong_tr[i].qin[(~strong_tr[i].bin_merge_flag)&(strong_tr[i].prompt_merger_mask)]

        qout_stalled_but_Tr_ej = strong_tr[i].qout[(~strong_tr[i].bin_merge_flag)&(strong_tr[i].merger_after_ejection_mask)]
        qin_stalled_but_Tr_ej = strong_tr[i].qin[(~strong_tr[i].bin_merge_flag)&(strong_tr[i].merger_after_ejection_mask)]

        axes[0].scatter(np.log10(qout_stalled_but_Tr),np.log10(qin_stalled_but_Tr),s=scatter_size,color="red",alpha=scatter_alpha)
        axes[1].scatter(np.log10(qout_stalled_but_Tr_ej),np.log10(qin_stalled_but_Tr_ej),s=scatter_size,color="red",alpha=scatter_alpha)
        if i == 0:  # Add label for legend without plotting scatter
            axes[0].scatter([], [], s=2, color="red", alpha=1,label="Binaries stalled in isolation")
            axes[1].scatter([], [], s=2, color="red", alpha=1,label="Binaries stalled in isolation")


    divider = make_axes_locatable(axes[1])  # Adjust based on the third axis
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(pcm1, cax=cax, label="number")

    axes[0].set_xlabel("$\log_{10}(q_{out})$")
    axes[1].set_xlabel("$\log_{10}(q_{out})$")


    axes[0].set_ylabel("$\log_{10}(q_{in})$")
    axes[1].set_ylabel("$\log_{10}(q_{in})$")


    axes[0].set_title("Prompt Mergers")
    axes[1].set_title("Mergers after kick")

    # Add legends to both plots, outside axes with a border
    legend_props = dict(edgecolor="black", linewidth=1.5)  # Custom border

    #axes[0].legend(prop=legend_props, fancybox=True)
    axes[0].legend(fancybox=True, framealpha=0.5, frameon=True, edgecolor="black")

    axes[0].set_ylim(-1,0)
    axes[0].set_xlim(-2.5,1.5)
    axes[1].set_ylim(-1,0)
    axes[1].set_xlim(-2.5,1.5)

    plt.tight_layout()
    return fig,axes


def print_isolate_binary_but_Tr_merged_stats(nruns,strong_tr):
    Tr_merger_stalled_isolated_avg = np.mean([np.sum([(~strong_tr[i].bin_merge_flag) & (strong_tr[i].prompt_merger_mask)])/np.sum(strong_tr[i].prompt_merger_mask) for i in range(nruns)])
    Tr_ej_merger_stalled_isolated_avg = np.mean([np.sum([(~strong_tr[i].bin_merge_flag) & (strong_tr[i].merger_after_ejection_mask)])/np.sum(strong_tr[i].merger_after_ejection_mask) for i in range(nruns)])
    print("%.2f%% of prompt mergers would be binaries that would otherwise have stalled in isolation" % (Tr_merger_stalled_isolated_avg * 100))
    print("%.2f%% of merger after ejections would be binaries that would otherwise have stalled in isolation" % (Tr_ej_merger_stalled_isolated_avg * 100))
 