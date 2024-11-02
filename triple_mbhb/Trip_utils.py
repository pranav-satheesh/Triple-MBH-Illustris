
import numpy as np
import Triple_dynamics as Tr

def Trip_stats(Trip_objs):

    Nruns = len(Trip_objs)
    tota_strong_Trip = 520

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

    print(f"Average prompt merger is {Prompt_mergers_avg} which is {(Prompt_mergers_avg/tota_strong_Trip) * 100:.2f} %")
    print(f"Average merger after ejection is {mergers_after_ejection_avg} which is {(mergers_after_ejection_avg/tota_strong_Trip) * 100:.2f} %")
    print(f"There are {no_mergers_avg:.2f} no mergers on average which is {(no_mergers_avg/tota_strong_Trip) * 100:.2f}%")   
    print(f"Average total mergers is {total_mergers_avg} which is {(total_mergers_avg/tota_strong_Trip) * 100:.2f} %")
    print("----------------------")

    print(f"Without triple interactions {np.sum(Trip_objs[0].bin_merge_flag)}({(np.sum(Trip_objs[0].bin_merge_flag)/tota_strong_Trip * 100 ):2.1f})% strong triple inner binary merges with just inspiral")
    print(f"With triple interactions added {total_mergers_avg:2.1f}({total_mergers_avg/tota_strong_Trip* 100:2.1f})% strong triple system has mergers which is a {(total_mergers_avg-np.sum(Trip_objs[0].bin_merge_flag))/520 * 100:2.1f} % increase in mergers")
    print(f"{merger_wt_triple_but_not_inspiral:2.1f} ({merger_wt_triple_but_not_inspiral/tota_strong_Trip * 100:2.1f}%) strong triple induced mergers are otherwise non mergers under binary inspiral evolution.")

    return total_mergers_avg


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

