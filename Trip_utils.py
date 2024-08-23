
def Trip_stats(Trip_objs):

    tota_strong_Trip = 520

    Prompt_mergers_avg = 0
    mergers_after_ejection_avg = 0
    total_mergers_avg = 0

    for i in range(len(Trip_objs)):
        Prompt_mergers_avg += Trip_objs[i].prompt_merger
        mergers_after_ejection_avg += Trip_objs[i].merger_after_ejection

    Prompt_mergers_avg = Prompt_mergers_avg/len(Trip_objs)
    mergers_after_ejection_avg = mergers_after_ejection_avg/len(Trip_objs)

    total_mergers_avg = Prompt_mergers_avg+mergers_after_ejection_avg
    no_mergers_avg = tota_strong_Trip - total_mergers_avg

    print(f"Average prompt merger is {Prompt_mergers_avg} which is {(Prompt_mergers_avg/tota_strong_Trip) * 100:.2f} %")
    print(f"Average merger after ejection is {mergers_after_ejection_avg} which is {(mergers_after_ejection_avg/tota_strong_Trip) * 100:.2f} %")
    print(f"Average total mergers is {total_mergers_avg} which is {(total_mergers_avg/tota_strong_Trip) * 100:.2f} %")
    print(f"There are {no_mergers_avg:.2f} no mergers on average which is {(no_mergers_avg/tota_strong_Trip) * 100:.2f}%")




