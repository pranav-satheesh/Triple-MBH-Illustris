import numpy as np

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
        M_merger_median_strong.append(np.median(np.log10(strong_tr[i].mbin_merger)))
    print(f"M mean for strong:{np.mean(M_merger_median_strong):.3f}")
    print("------------------------------------------")


    print(f"z mean for iso:{np.median(np.log10(iso_bin.M1[iso_bin.merger_mask]+iso_bin.M2[iso_bin.merger_mask])):.2f}")
    print(f"z mean for weak:{np.median(np.log10(weak_tr.M1[weak_tr.weak_triple_mask][weak_tr.merger_mask]+weak_tr.M2[weak_tr.weak_triple_mask][weak_tr.merger_mask])):.3f}")

    z_merger_median_strong = []
    z_min_strong = []
    z_max_strong = []

    for i in range(Nruns):
        z_merger_median_strong.append(np.median(np.log10(strong_tr[i].mbin_merger)))
        z_min_strong.append(np.min(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask]))
        z_max_strong.append(np.max(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask]))
    
    print(f"z mean for strong:{np.mean(z_merger_median_strong):.3f}")
    print(f"z min for strong:{np.mean(z_min_strong):.3f}")
    print(f"z max for strong:{np.mean(z_max_strong):.3f}")

    z_min_iso = np.min(iso_bin.z_merger[iso_bin.merger_mask])
    z_max_iso = np.max(iso_bin.z_merger[iso_bin.merger_mask])
    print(f"z min for iso binary {z_min_iso:.3f}")
    print(f"z max for iso binary {z_max_iso:.3f}")

    print("------------------------------------------")

    return None 


