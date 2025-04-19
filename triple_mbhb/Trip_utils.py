
import numpy as np
import pickle
import Triple_dynamics as Tr
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')


#some utility functions for other python scripts

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

    total_systems = strong_tr[0].N_strong_triples + iso_bin.N_iso_binaries + weak_tr.N_weak_triples
    print(f"Total number of MBH systems is {total_systems}")
    print(f"Number of iso binaries is {iso_bin.N_iso_binaries} and it is {iso_bin.N_iso_binaries/total_systems*100:.2f} %")
    print(f"Number of weak triples is {weak_tr.N_weak_triples} and it is {weak_tr.N_weak_triples/total_systems*100:.2f} %")
    print(f"Number of strong triples is {strong_tr[0].N_strong_triples} and it is {strong_tr[0].N_strong_triples/total_systems*100:.2f} %")

    print("------------------")
    #print the stats for strong triples
    total_st_mergers = Trip_stats(strong_tr)

    print("-------------")
    total_binary_mergers = np.sum(iso_bin.merger_mask) + np.sum(weak_tr.merger_mask) + np.sum(strong_tr[0].bin_merge_flag) #total number of mergers under binary inspiral evolution
    print(f"Total number of binary mergers is {total_binary_mergers} and it is {total_binary_mergers/total_systems*100:.2f} %")
    total_mergers_with_st = total_binary_mergers + total_st_mergers #total number of mergers with strong triples
    print(f"Total number of mergers with strong triples is {total_mergers_with_st} and it is {total_mergers_with_st/total_systems*100:.2f} %")
    print(f"With strong triples added mergers increase by {(total_mergers_with_st-total_binary_mergers)/total_systems*100:.2f} %")


    return strong_tr, weak_tr, iso_bin, stalled_objs

def Trip_stats(Trip_objs):

    Nruns = len(Trip_objs)
    tota_strong_Trip = Trip_objs[0].N_strong_triples

    Prompt_mergers_avg = 0
    mergers_after_ejection_avg = 0
    total_mergers_avg = 0
    merger_wt_triple_but_not_inspiral = 0


    for i in range(Nruns):
        Prompt_mergers_avg += Trip_objs[i].prompt_merger
        mergers_after_ejection_avg += Trip_objs[i].merger_after_ejection
        merger_wt_triple_but_not_inspiral+=np.sum((Trip_objs[i].bin_merge_flag)&(Trip_objs[i].merger_mask))


    Prompt_mergers_avg = Prompt_mergers_avg/Nruns
    mergers_after_ejection_avg = mergers_after_ejection_avg/Nruns

    total_mergers_avg = Prompt_mergers_avg+mergers_after_ejection_avg
    no_mergers_avg = tota_strong_Trip - total_mergers_avg

    merger_wt_triple_but_not_inspiral = merger_wt_triple_but_not_inspiral/Nruns
    
    print("In strong triples:")
    print(f"Average prompt merger is {Prompt_mergers_avg} which is {(Prompt_mergers_avg/tota_strong_Trip) * 100:.2f} %")
    print(f"Average merger after ejection is {mergers_after_ejection_avg} which is {(mergers_after_ejection_avg/tota_strong_Trip) * 100:.2f} %")
    print(f"There are {no_mergers_avg:.2f} no mergers on average which is {(no_mergers_avg/tota_strong_Trip) * 100:.2f}%")   
    print(f"Average total mergers is {total_mergers_avg} which is {(total_mergers_avg/tota_strong_Trip) * 100:.2f} %")
    print("----------------------")

    print(f"Without triple interactions {np.sum(Trip_objs[0].bin_merge_flag)}({(np.sum(Trip_objs[0].bin_merge_flag)/tota_strong_Trip * 100 ):2.1f})% strong triple inner binary merges with just inspiral")
    print(f"With triple interactions added {total_mergers_avg:2.1f}({total_mergers_avg/tota_strong_Trip* 100:2.1f})% strong triple system has mergers which is a {(total_mergers_avg-np.sum(Trip_objs[0].bin_merge_flag))/520 * 100:2.1f} % increase in mergers")
    print(f"{merger_wt_triple_but_not_inspiral:2.1f} ({merger_wt_triple_but_not_inspiral/tota_strong_Trip * 100:2.1f}%) strong triple induced mergers are otherwise non mergers under binary inspiral evolution.")

    return total_mergers_avg

def find_exchange_events_in_strong_triple(Nruns,strong_tr,weak_tr_obj,iso_bin_obj):
    exchanges_Tr_ej = []
    exchanges = []
    for i in range(Nruns):
        scatter_event_mask = (strong_tr[i].a_triple_after<=strong_tr[i].a_triple_ovtks_ill)
        exchanges.append(np.sum(~scatter_event_mask))
        exchanges_Tr_ej.append(np.sum((~scatter_event_mask)&strong_tr[i].merger_after_ejection_mask))

    print(f"{np.mean(exchanges_Tr_ej)/189 * 100:.2f} % are exchange events in mergers after kick")

def ejection_effects(Trip_objs,weak_tr_obj,iso_bin_obj):

    spin_model = ["random","hybrid","5deg"]
    Nruns = len(Trip_objs)

    for spin_arg in spin_model:
        invalid_iso,invalid_weak,invalid_strong = 0,0,0
        for i in range(Nruns):
            iso_inv,weak_inv,strong_inv = Tr.find_invalid_mergers(Trip_objs[i],weak_tr_obj,iso_bin_obj,spin_arg)
            invalid_iso += np.sum(iso_inv)
            invalid_weak += np.sum(weak_inv)
            invalid_strong += np.sum(strong_inv)

        invalid_iso = invalid_iso/Nruns
        invalid_weak = invalid_weak/Nruns
        invalid_strong = invalid_strong/Nruns

        print(f"{invalid_iso:.0f}({invalid_iso/len(iso_bin_obj.z_merger)*100:2.1f} %)iso binary systems do not form as a result of prior {spin_arg} ejection events")
        print(f"{invalid_weak:.0f}({invalid_weak/len(weak_tr_obj.z_merger)*100:2.1f} %)weak triple systems do not form as a result of prior {spin_arg} ejection events")
        print(f"{invalid_strong:.0f}({invalid_strong/len(Trip_objs[0].z_triple_merger)*100:2.1f} %)strong triple systems do not form as a result of prior {spin_arg} ejection events")
        print(f"{(invalid_iso+invalid_weak+invalid_strong):.0f}({(invalid_iso+invalid_weak+invalid_strong)/(len(iso_bin_obj.z_merger)+len(weak_tr_obj.z_merger)+len(Trip_objs[0].z_triple_merger))*100:.2f}%) of all systems do not form due to prior ejection events")
        print("----------------")

    return iso_inv,weak_inv,strong_inv

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


