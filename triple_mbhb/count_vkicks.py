import BH_kicks as kick
import numpy as np
import matplotlib.pyplot as plt 

def count_kicks(kick_data,bin_edges):
    """
    Calculates the number of kicks in each bin for multiple data sets.
    
    Parameters:
        kick_data (list): List of kick values for each data set.
        bin_edges (array): Bin edges for counting kicks.

    Returns:
        kick_counts (array): 2D array with counts of kicks per bin for each data set.
    """
    
    kick_counts = np.zeros((len(kick_data), len(bin_edges) - 1))
    for i, kicks in enumerate(kick_data):
            kick_counts[i], _ = np.histogram(kicks, bins=bin_edges)
    return kick_counts

def Nvkicks(iso_bin,weak_tr,strong_tr,Nruns,velocity_bins,kick_type,realizations=10):
        """
        Calculates the total number of kicks in specified velocity bins across iso_bin, weak_tr, and strong_tr datasets.

        Parameters:
        iso_bin, weak_tr, strong_tr: Data objects containing kick data.
        Nruns (int): Number of strong triple realizations.
        velocity_bins (array): Array of velocity bin edges.
        kick_type (str): Type of kick attribute to analyze.
        realizations (int): Number of kick realizations.

        Returns:
        total_kicks (array): Array of total kick counts per velocity bin.
        """
    
        #first average over all strong triple realizations
        strong_kicks = 0
        for i in range(Nruns):
                kicks = getattr(strong_tr[i], kick_type)
                strong_kicks += count_kicks(kicks,velocity_bins)

        strong_kicks = strong_kicks/Nruns
        iso_kicks = count_kicks(getattr(iso_bin, kick_type),velocity_bins)
        weak_kicks = count_kicks(getattr(weak_tr, kick_type),velocity_bins)
        total_kicks = strong_kicks+iso_kicks+weak_kicks

        return total_kicks

def count_slingshot_kicks(strong_tr_obj,velocit_bins_sling,Nruns):
        
        slingshot_kick_counts = []
        for i in range(Nruns):
                sling_kick_N, _ = np.histogram(strong_tr_obj[i].slingshot_kicks,bins=velocity_bins_sling)
                slingshot_kick_counts.append(sling_kick_N)
        
        return slingshot_kick_counts

def plot_kick_distribution(kick_types_data):
    """
    Plots the distribution of kicks vs velocity with mean and standard deviation spread for specified kick types.
    
    Parameters:
        kick_types_data (dict): Dictionary where keys are labels for each kick type (e.g., 'random', 'slingshot')
                                and values are tuples of (kick_counts, velocity_bins) for each type.
                                kick_counts should be an array of shape (realizations, bins).
    """
    fig,ax = plt.subplots(1,1,figsize=(10, 7))
    
    # Predefined color-blind-friendly color map for specific kick types
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
    
    for label, (kick_counts, velocity_bins) in kick_types_data.items():
        # Calculate mean and standard deviation for each velocity bin
        mean_counts = np.mean(kick_counts, axis=0)
        std_counts = np.std(kick_counts, axis=0)
        
        # Use the color from the predefined palette for this label
        color = kick_colors.get(label, "#000000")  # Default to black if label not in kick_colors
        linestyle = kick_linestyles.get(label, "-") 

        # Plot mean and fill standard deviation
        ax.plot(velocity_bins[:-1], mean_counts, color=color, linestyle=linestyle, linewidth = 2,label=label)
        ax.fill_between(velocity_bins[:-1], mean_counts - std_counts, mean_counts + std_counts, color=color, alpha=0.2)
    
    ax.set_xlabel('Velocity (km/s)')
    ax.set_ylabel('Number')
    ax.set_xlim(10, velocity_bins[-1])  # Adjust x-axis limits based on bins
    ax.set_ylim(1, 10**3)               # Adjust y-axis limits as needed
    ax.set_yscale("log",base=10)
    ax.set_xscale("log",base=10)
    ax.legend()
    plt.show()
    return fig,ax

def kicks_above_threshold(iso_bin,weak_tr,strong_tr,Nruns,kick_type,vthreshold=500):

    # Calculate percentage of kicks above threshold for each realization
    iso_kick_above_vt = [np.sum(np.array(kicks) > vthreshold) for kicks in getattr(iso_bin, kick_type)]
    weak_tr_kick_above_vt = [np.sum(np.array(kicks) > vthreshold) for kicks in getattr(weak_tr, kick_type)]

    st_kicks_above_vt = []
    for i in range(Nruns):
        st_kicks_above_vt.append([np.sum(np.array(kicks) > vthreshold) for kicks in getattr(strong_tr[i],'v_kick_rand')])
    strong_tr_kick_above_vt =np.mean(st_kicks_above_vt)

    total_kicks = len(getattr(iso_bin,kick_type)[0])+ len(getattr(weak_tr,kick_type)[0])+len(getattr(strong_tr[0],kick_type))

    kick_percentage_above_vth = []
    for i in range(len(iso_kick_above_vt)):
        tot_kick_above_vt = iso_kick_above_vt[i]+weak_tr_kick_above_vt[i]+strong_tr_kick_above_vt
        kick_percentage_above_vth.append((tot_kick_above_vt/total_kicks)*100)


    return np.mean(kick_percentage_above_vth),np.std(kick_percentage_above_vth)

def combined_kick_type_plots(kick_types_data, iso_bin, weak_tr, strong_tr, Nruns, thresholds):
    """
    Combines two plots:
    1. Number of kicks vs velocity for each kick type, with mean and standard deviation spread.
    2. Percentage of kicks above varying thresholds for each kick type.
    
    Parameters:
        kick_types_data (dict): Dictionary with keys as kick type labels (e.g., 'random', 'slingshot')
                                and values as tuples of (kick_counts, velocity_bins) for each type.
        iso_bin, weak_tr, strong_tr, Nruns: Inputs for calculating percentage above threshold.
        thresholds (array): Array of velocity thresholds for the second plot.
    """
    # Predefined color-blind-friendly color map for specific kick types
    kick_colors = {
        'slingshot': "#4daf4a",  # Green
        'aligned': "#377eb8",    # Blue
        'hybrid': "#a2c8ec",     # Light blue
        'random': "#e41a1c"      # Red
    }
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Number of kicks vs velocity with mean and standard deviation
    for label, (kick_counts, velocity_bins) in kick_types_data.items():
        mean_counts = np.mean(kick_counts, axis=0)
        std_counts = np.std(kick_counts, axis=0)
        color = kick_colors.get(label, "#000000")  # Default to black if label not in kick_colors
        
        # Plot mean and fill standard deviation
        ax1.plot(velocity_bins[:-1], mean_counts, color=color, label=label)
        ax1.fill_between(velocity_bins[:-1], mean_counts - std_counts, mean_counts + std_counts, color=color, alpha=0.2)
    
    # Labeling for plot 1
    ax1.set_xlabel('Velocity (km/s)')
    ax1.set_ylabel('Number')
    ax1.set_xlim(10, velocity_bins[-1])
    ax1.set_ylim(1, 10**3)
    ax1.set_xscale("log", base=10)
    ax1.set_yscale("log", base=10)
    ax1.legend()
    ax1.set_title('Number of Kicks vs Velocity for Different Kick Types')

    # Plot 2: Percentage of kicks above threshold for each kick type
    for label in kick_types_data.keys():
        mean_percentages = []
        std_percentages = []
        slingshot_mean_percentages = []
        slingshot_std_percentages = []
        
        # Calculate mean and std for each threshold for the current kick type
        for threshold in thresholds:
            mean_percentage, std_percentage = kicks_above_threshold(iso_bin, weak_tr, strong_tr, Nruns, label, threshold)
            mean_percentages.append(mean_percentage)
            std_percentages.append(std_percentage)
            slingshot_percentage = []
            for i in range(Nruns):
                slingshot_percentage.append(np.sum(np.array(strong_tr[i].slingshot_kicks)>threshold)/len(strong_tr[i].slingshot_kicks)*100)
            
            slingshot_mean_percentages.append(np.mean(slingshot_percentage))
            slingshot_std_percentages.append(np.std(slingshot_percentage))
        
        # Plot mean and 1-sigma spread for the current kick type
        color = kick_colors.get(label, "#000000")  # Use color from the palette or default to black
        ax2.plot(thresholds, mean_percentages,color=color)
        ax2.fill_between(thresholds, 
                         np.array(mean_percentages) - np.array(std_percentages), 
                         np.array(mean_percentages) + np.array(std_percentages), 
                         color=color, alpha=0.2)
    

    ax2.plot(thresholds, slingshot_mean_percentages, label='slingshot')
    ax2.fill_between(thresholds, slingshot_mean_percentages - slingshot_std_percentages, 
                         slingshot_mean_percentages + slingshot_std_percentages, 
                         color=kick_colors['slingshot'], alpha=0.2)  
    # Labeling for plot 2
    ax2.set_xlabel('$v_t$ (km/s)')
    ax2.set_ylabel('\% of kicks above $v_t$')
    ax2.set_xscale("log", base=10)
    ax2.set_yscale("log", base=10)
    ax2.ylim(10**(-1),)

    plt.tight_layout()
    plt.show()

    return fig,ax1,ax2

# Define thresholds and call combined function

