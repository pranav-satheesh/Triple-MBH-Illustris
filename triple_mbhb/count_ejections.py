import numpy as np
import merger_rate_calculate as mr
from tqdm import tqdm
import matplotlib.pyplot as plt

def assign_ejection_masks(strong_tr, iso_bin, weak_tr, Nruns):
    """
    Assigns ejection masks for each kick type ('random', 'hybrid', 'aligned', 'slingshot')
    based on whether the kick velocity exceeds the escape velocity for each dataset.

    Parameters:
        strong_tr (list): List of strong triple objects for each run.
        iso_bin (object): Object representing isolated binary data.
        weak_tr (object): Object representing weak triple data.
        Nruns (int): Number of runs for strong triples.
    """
    # Define kick types and associated attributes
    kick_types = ['random', 'hybrid', 'aligned', 'slingshot']
    
    # Assign masks for each strong triple run
    for i in range(Nruns):
        for kick_type in kick_types:
            if kick_type == 'slingshot':
                mask = strong_tr[i].slingshot_kicks > strong_tr[i].Vescape
            else:
                kick_attr = f'v_kick_{kick_type}'
                mask = getattr(strong_tr[i], kick_attr) > strong_tr[i].Vescape[strong_tr[i].merger_mask]
            setattr(strong_tr[i], f'ejection_{kick_type}_mask', mask)

    # Assign masks for isolated binary and weak triple data
    for obj in [iso_bin, weak_tr]:
        for kick_type in kick_types[:-1]:  # Exclude 'slingshot' for iso_bin and weak_tr
            kick_attr = f'v_kick_{kick_type}'
            mask = getattr(obj, kick_attr) > obj.Vescape[obj.merger_mask]
            setattr(obj, f'ejection_{kick_type}_mask', mask)

    return None

#for strong triples
def strong_tr_ejection_rates(strong_tr, Nruns, N_kick_realization, kick_type, lgzbinsize=0.25, lgzmin=-3, lgzmax=1.0):
    """
    Calculates ejection rates for a specified kick type (random, hybrid, aligned, slingshot) over multiple realizations.
    
    Parameters:
        strong_tr (list): List of strong triple objects for each run.
        Nruns (int): Number of runs.
        N_kick_realization (int): Number of kick realizations.
        kick_type (str): Type of kick ('random', 'hybrid', 'aligned', or 'slingshot').
        lgzbinsize (float): Bin size for logarithmic z-bins.
        lgzmin (float): Minimum log z value.
        lgzmax (float): Maximum log z value.

    Returns:
        dict: Dictionary containing the ejection rates for the specified kick type across runs.
    """
    ejection_rates = []
    
    for i in tqdm(range(Nruns),desc=f"calculating ejection rates for {kick_type}"):
        ejection_realizations = []

        if kick_type in ["random", "hybrid", "aligned"]:
            for j in range(N_kick_realization):
                # Select the appropriate mask based on the specified kick type
                if kick_type == "random":
                    mask = strong_tr[i].ejection_random_mask[j]
                elif kick_type == "hybrid":
                    mask = strong_tr[i].ejection_hybrid_mask[j]
                elif kick_type == "aligned":
                    mask = strong_tr[i].ejection_aligned_mask[j]
                
                lgzbins, escape_rate = mr.diff_merger_rate(
                    strong_tr[i].z_triple_merger[strong_tr[i].merger_mask][mask],
                    lgzbinsize=lgzbinsize, lgzmin=lgzmin, lgzmax=lgzmax
                )
                ejection_realizations.append(escape_rate)
            
            # Calculate mean ejection rate across realizations for the given run
            ejection_rates.append(np.mean(ejection_realizations, axis=0))

        elif kick_type == "slingshot":
            # Slingshot ejections are not based on kick realizations
            lgzbins, escape_rate = mr.diff_merger_rate(
                strong_tr[i].z_triple_merger[
                    (strong_tr[i].slingshot_kicks > strong_tr[i].Vescape) & (strong_tr[i].merger_mask)
                ],
                lgzbinsize=lgzbinsize, lgzmin=lgzmin, lgzmax=lgzmax
            )
            ejection_rates.append(escape_rate)
        
        else:
            raise ValueError("Invalid kick type specified. Choose from 'random', 'hybrid', 'aligned', or 'slingshot'.")
    
    return lgzbins,ejection_rates


def tot_population_ejection_rates(strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, kick_type, lgzbinsize=0.25, lgzmin=-3, lgzmax=1.0):

    """
    Calculates combined ejection rates for a specified kick type across strong, weak, and isolated binaries.

    Parameters:
        strong_tr (list): List of strong triple objects for each run.
        weak_tr: Weak binary object.
        iso_bin: Isolated binary object.
        Nruns (int): Number of runs.
        N_kick_realization (int): Number of kick realizations.
        kick_type (str): Type of kick ('random', 'hybrid', or 'aligned').
        lgzbinsize (float): Bin size for logarithmic z-bins.
        lgzmin (float): Minimum log z value.
        lgzmax (float): Maximum log z value.

    Returns:
        dict: Dictionary containing the combined ejection rates for the specified kick type across runs.
    """
    combined_ejection_rates = []

    for i in tqdm(range(Nruns), desc=f"calculating combined ejection rates for {kick_type}"):
        ejection_realizations = []

        if kick_type in ["random", "hybrid", "aligned"]:
            for j in range(N_kick_realization):
                # Select the appropriate mask for each type (strong, weak, isolated)
                if kick_type == "random":
                    mask_strong = strong_tr[i].ejection_random_mask[j]
                    mask_weak = weak_tr.ejection_random_mask[j]
                    mask_iso = iso_bin.ejection_random_mask[j]
                elif kick_type == "hybrid":
                    mask_strong = strong_tr[i].ejection_hybrid_mask[j]
                    mask_weak = weak_tr.ejection_hybrid_mask[j]
                    mask_iso = iso_bin.ejection_hybrid_mask[j]
                elif kick_type == "aligned":
                    mask_strong = strong_tr[i].ejection_aligned_mask[j]
                    mask_weak = weak_tr.ejection_aligned_mask[j]
                    mask_iso = iso_bin.ejection_aligned_mask[j]

                # Calculate escape rates for strong, weak, and isolated binaries
                lgzbins, escape_rate_strong = mr.diff_merger_rate(
                    strong_tr[i].z_triple_merger[strong_tr[i].merger_mask][mask_strong],
                    lgzbinsize=lgzbinsize, lgzmin=lgzmin, lgzmax=lgzmax
                )
                _, escape_rate_weak = mr.diff_merger_rate(
                    weak_tr.z_merger[weak_tr.merger_mask][mask_weak],
                    lgzbinsize=lgzbinsize, lgzmin=lgzmin, lgzmax=lgzmax
                )
                _, escape_rate_iso = mr.diff_merger_rate(
                    iso_bin.z_merger[iso_bin.merger_mask][mask_iso],
                    lgzbinsize=lgzbinsize, lgzmin=lgzmin, lgzmax=lgzmax
                )

                # Sum escape rates across all realizations and add to combined list
                combined_escape_rate = escape_rate_strong + escape_rate_weak + escape_rate_iso
                ejection_realizations.append(combined_escape_rate)

            # Append the mean across all kick realizations for this run
            combined_ejection_rates.append(np.mean(ejection_realizations, axis=0))

        else:
            raise ValueError("Invalid kick type specified. Choose from 'random', 'hybrid', or 'aligned'.")

    return lgzbins, combined_ejection_rates


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


def calculate_and_plot_ejection_rates(strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization,kick_types=None):
    # Define the parameters for each kick type
    
    if kick_types is None:
        kick_types = {
            'random': {'lgzbinsize': 0.25, 'lgzmin': -3, 'lgzmax': 1.0},
            'hybrid': {'lgzbinsize': 0.4, 'lgzmin': -3, 'lgzmax': 1.0},
            'aligned': {'lgzbinsize': 0.4, 'lgzmin': -3, 'lgzmax': 1.0},
            'slingshot': {'lgzbinsize': 0.25, 'lgzmin': -3, 'lgzmax': 1.0}
        }

    # To store results for plotting and returning
    ejection_rates = {}
    lgzbins = {}

    # Calculate ejection rates for each kick type in strong triples
    for kick_type, params in kick_types.items():
        lgzbins[kick_type], ejection_rates[kick_type] = strong_tr_ejection_rates(
            strong_tr, Nruns, N_kick_realization, kick_type=kick_type, 
            lgzbinsize=params['lgzbinsize'], lgzmin=params['lgzmin'], lgzmax=params['lgzmax']
        )

    # Calculate total ejection rates for each kick type (combined population)
    for kick_type, params in kick_types.items():

        if kick_type == 'slingshot':
            continue  # Skip slingshot for total ejection rates
        lgzbins[f'{kick_type}_tot'], ejection_rates[f'{kick_type}_tot'] = tot_population_ejection_rates(
            strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, kick_type=kick_type,
            lgzbinsize=params['lgzbinsize'], lgzmin=params['lgzmin'], lgzmax=params['lgzmax']
        )

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Total Population Plot
    ax[0].plot(lgzbins['random_tot'], np.mean(ejection_rates['random_tot'], axis=0), color=kick_colors['random'], label="random", linestyle = kick_linestyles['random'],linewidth=2)
    ax[0].plot(lgzbins['hybrid_tot'], np.mean(ejection_rates['hybrid_tot'], axis=0), color=kick_colors['hybrid'], label="hybrid", linestyle = kick_linestyles['hybrid'],linewidth=2)
    ax[0].plot(lgzbins['aligned_tot'], np.mean(ejection_rates['aligned_tot'], axis=0), color=kick_colors['aligned'], label="aligned", linestyle = kick_linestyles['aligned'],linewidth=2)
    ax[0].plot(lgzbins['slingshot'], np.mean(ejection_rates['slingshot'], axis=0), color=kick_colors['slingshot'], label="slingshot", linestyle = kick_linestyles['slingshot'],linewidth=2)
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=20)
    ax[0].set_title("Total population")
    ax[0].set_yticks([1e-1,1e-3,1e-5,1e-7])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("$\log z$",fontsize=25)
    ax[0].set_ylabel(r"$\log (d^2 N / (d \log z dt)  \times 1\text{yr})$",fontsize=25)

    # Strong Triples Plot
    ax[1].plot(lgzbins['random'], np.mean(ejection_rates['random'], axis=0), color=kick_colors['random'], label="random", linestyle = kick_linestyles['random'],linewidth=2)
    ax[1].plot(lgzbins['hybrid'], np.mean(ejection_rates['hybrid'], axis=0), color=kick_colors['hybrid'], label="hybrid", linestyle = kick_linestyles['hybrid'],linewidth=2)
    ax[1].plot(lgzbins['aligned'], np.mean(ejection_rates['aligned'], axis=0), color=kick_colors['aligned'], label="aligned", linestyle = kick_linestyles['aligned'],linewidth=2)
    ax[1].plot(lgzbins['slingshot'], np.mean(ejection_rates['slingshot'], axis=0), color=kick_colors['slingshot'], label="slingshot", linestyle = kick_linestyles['slingshot'],linewidth=2)
    ax[1].set_yscale("log")
    ax[1].legend(fontsize=20)
    ax[1].set_title("Strong triples", fontsize=25)
    ax[1].set_xlabel("$\log z$", fontsize=25)

    plt.tight_layout()
    plt.show()

    return fig,ax,lgzbins, ejection_rates

def calculate_fraction_ejection(strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, kick_types, binsizes, lgzmin, lgzmax, Nmergers_thresh):
    """
    Calculate the fraction of ejections for different kick types (slingshot, random, hybrid, aligned) over multiple runs and realizations.
    
    Parameters:
    - strong_tr, weak_tr, iso_bin: Objects containing merger and ejection data.
    - Nruns: Number of runs (realizations).
    - N_kick_realization: Number of kick realizations.
    - kick_types: List of kick types (e.g., 'slingshot', 'random', etc.).
    - binsizes: Dictionary containing the bin sizes for each kick type.
    - lgzmin, lgzmax: Redshift range.
    - Nmergers_thresh: Minimum number of mergers required for valid data.
    
    Returns:
    - accumulated_fractions: Dictionary with kick types as keys, containing lists of accumulated fractions for each kick type.
    """
    
    accumulated_fractions = {kick: [] for kick in kick_types}
    bin_centers = {}

    for i in range(Nruns):
            lgz_sling_ejec = np.log10(
                strong_tr[i].z_triple_merger[
                    (strong_tr[i].ejection_slingshot_mask) & (strong_tr[i].merger_mask)
                ]
            )
            lgz_mrgrs = np.concatenate([
                np.log10(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask]),
                np.log10(iso_bin.z_merger[iso_bin.merger_mask]),
                np.log10(weak_tr.z_merger[weak_tr.merger_mask])
            ])
            bin_centers['slingshot'], fraction_ejected_sling = fraction_ejection_for_plot(
                lgz_sling_ejec, lgz_mrgrs, lgzmin, lgzmax, binsizes['slingshot'], Nmergers_thresh
            )
            accumulated_fractions['slingshot'].append(fraction_ejected_sling)

    # Random, Hybrid, and Aligned kicks: Loop over kick realizations
    for kick_type in kick_types:
        if kick_type == 'slingshot':  # Skip already processed
            continue

        for j in range(N_kick_realization):
            fraction_ejected_all = []

            for i in range(Nruns):
                # Combine data across strong, weak, and isolated
                lgz_ejec = np.concatenate([
                    np.log10(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask][getattr(strong_tr[i], f"ejection_{kick_type}_mask")[j]]),
                    np.log10(iso_bin.z_merger[iso_bin.merger_mask][getattr(iso_bin, f"ejection_{kick_type}_mask")[j]]),
                    np.log10(weak_tr.z_merger[weak_tr.merger_mask][getattr(weak_tr, f"ejection_{kick_type}_mask")[j]])
                ])

                lgz_mrgrs = np.concatenate([
                    np.log10(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask]),
                    np.log10(iso_bin.z_merger[iso_bin.merger_mask]),
                    np.log10(weak_tr.z_merger[weak_tr.merger_mask])
                ])

                # Calculate fractions for the current realization
                bin_centers[kick_type], fraction_ejected = fraction_ejection_for_plot(
                    lgz_ejec, lgz_mrgrs, lgzmin, lgzmax, binsizes[kick_type], Nmergers_thresh
                )
                fraction_ejected_all.append(fraction_ejected)

            # Average across all realizations for the current kick type
            accumulated_fractions[kick_type].append(np.mean(fraction_ejected_all, axis=0))

    return accumulated_fractions, bin_centers

def fraction_ejection_for_plot(lgz_ejec, lgz_mrgrs, lgzmin, lgzmax, binsize, Nmergers_thresh):
    """
    Calculate the fraction of ejections for a given set of data and bin settings.
    
    Parameters:
    - lgz_ejec: Ejection redshifts.
    - lgz_mrgrs: Merger redshifts.
    - lgzmin, lgzmax: Redshift range for bins.
    - binsize: Size of bins to group the data.
    - Nmergers_thresh: Threshold for the number of mergers.
    
    Returns:
    - bin_centers: Centers of the bins.
    - fraction_ejected: Fraction of ejections in each bin.
    """
    bins_kick = np.arange(lgzmin, lgzmax + binsize, binsize)
    N_mrgr, lgzbin_edges = np.histogram(lgz_mrgrs, bins=bins_kick)
    N_ejec, _ = np.histogram(lgz_ejec, bins=bins_kick)

    valid_mask = N_mrgr > Nmergers_thresh
    fraction_ejected = np.zeros_like(N_mrgr, dtype=float)
    fraction_ejected[valid_mask] = N_ejec[valid_mask] / N_mrgr[valid_mask]
    bin_centers = (lgzbin_edges[:-1] + lgzbin_edges[1:]) / 2

    return bin_centers, fraction_ejected

def plot_ejection_fractions(accumulated_fractions, bin_centers, kick_colors):
    """
    Plot the accumulated ejection fractions for each kick type.
    
    Parameters:
    - accumulated_fractions: Dictionary with accumulated ejections for each kick type.
    - bin_centers: Centers of the bins.
    - kick_colors: Dictionary mapping kick types to their respective colors for plotting.
    """
    fig,ax = plt.subplots(1,1,figsize=[9, 7])

    for kick_type, fraction_data in accumulated_fractions.items():
        mean_fraction = np.mean(fraction_data, axis=0)
        ax.step(bin_centers, mean_fraction, label=kick_type, linewidth=2.5, color=kick_colors[kick_type])

    ax.set_xlim(-3, 1.5)
    ax.set_yscale("log")
    ax.legend(fontsize=15, frameon=True, fancybox=True, framealpha=0.8, edgecolor="black", loc="upper right")
    ax.set_xlabel("$\log (z)$", fontsize=35)
    ax.set_ylabel("Fraction of ejections", fontsize=35)
    plt.show()

def calculate_relative_ejections(strong_tr,weak_triso_bin, Nruns, N_kick_realization, gwrecoil_kick_type, binsizes, lgzmin, lgzmax, Nmergers_thresh):

    '''calculate the relative fraction of ejection of slingshot kicks compared to the GW recoil for a spin type'''

    # Calculate the fraction of ejection for the slingshot kicks compared to the GW recoil kicks in redshift bins
    
    
    for i in range(Nruns):

        #slingshot_ejection_redshifts
        lgz_sling_ejec = np.log10(
                strong_tr[i].z_triple_merger[
                    (strong_tr[i].ejection_slingshot_mask) & (strong_tr[i].merger_mask)
                ]
            )                          
        
    for kick_type in kick_types:
        for j in range(N_kick_realization):
            Ejections_per_realization = []
            for i in range(Nruns):
                lgz_ejec = np.concatenate([
                        np.log10(strong_tr[i].z_triple_merger[strong_tr[i].merger_mask][getattr(strong_tr[i], f"ejection_{kick_type}_mask")[j]]),
                        np.log10(iso_bin.z_merger[iso_bin.merger_mask][getattr(iso_bin, f"ejection_{kick_type}_mask")[j]]),
                        np.log10(weak_tr.z_merger[weak_tr.merger_mask][getattr(weak_tr, f"ejection_{kick_type}_mask")[j]])
                    ])
            


    
    
    slingshot_fractions, slingshot_bin_centers = calculate_fraction_ejection(
        strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, 
        kick_types=['slingshot'], binsizes=binsizes, lgzmin=lgzmin, lgzmax=lgzmax, 
        Nmergers_thresh=Nmergers_thresh
    )

    gwrecoil_fractions, gwrecoil_bin_centers = calculate_fraction_ejection(
        strong_tr, weak_tr, iso_bin, Nruns, N_kick_realization, 
        kick_types=[gwrecoil_kick_type], binsizes=binsizes, lgzmin=lgzmin, lgzmax=lgzmax, 
        Nmergers_thresh=Nmergers_thresh
    )

    # Calculate the relative fraction of slingshot ejections compared to GW recoil ejections
    relative_fractions = {}
    for i in range(len(slingshot_fractions['slingshot'])):
        relative_fractions[i] = slingshot_fractions['slingshot'][i] / gwrecoil_fractions[gwrecoil_kick_type][i]

    return relative_fractions, slingshot_bin_centers['slingshot']


def calculate_total_number_of_ejections(N_kicks,Nruns,strong_tr, weak_tr, iso_bin):

    N_total_systems = iso_bin.N_iso_binaries+weak_tr.N_weak_triples+strong_tr[0].N_strong_triples
    N_strong_triples = strong_tr[0].N_strong_triples

    total_random_ejections = []
    total_hybrid_ejections = []
    total_aligned_ejections = []

    total_random_ejections_Tr = []
    total_hybrid_ejections_Tr = []
    total_aligned_ejections_Tr = []
    
    for i in range(N_kicks):
        total_random_ejections_i = []
        total_hybrid_ejections_i = []
        total_aligned_ejections_i = []

        total_random_ejections_Tr_i = []
        total_hybrid_ejections_Tr_i = []
        total_aligned_ejections_Tr_i = []
    
        for j in range(Nruns):
            total_random_ejections_i.append(
            np.sum(iso_bin.ejection_random_mask[i]) +
            np.sum(weak_tr.ejection_random_mask[i]) +
            np.sum(strong_tr[j].ejection_random_mask[i])
        )
            total_random_ejections_Tr_i.append(np.sum(strong_tr[j].ejection_random_mask[i]))
            total_hybrid_ejections_Tr_i.append(np.sum(strong_tr[j].ejection_hybrid_mask[i]))
            total_aligned_ejections_Tr_i.append(np.sum(strong_tr[j].ejection_aligned_mask[i]))

            total_hybrid_ejections_i.append(
            np.sum(iso_bin.ejection_hybrid_mask[i]) +
            np.sum(weak_tr.ejection_hybrid_mask[i]) +
            np.sum(strong_tr[j].ejection_hybrid_mask[i])
        )
            total_aligned_ejections_i.append(
            np.sum(iso_bin.ejection_aligned_mask[i]) +
            np.sum(weak_tr.ejection_aligned_mask[i]) +
            np.sum(strong_tr[j].ejection_aligned_mask[i])
        )
    
        total_random_ejections.append(np.mean(total_random_ejections_i))
        total_hybrid_ejections.append(np.mean(total_hybrid_ejections_i))
        total_aligned_ejections.append(np.mean(total_aligned_ejections_i))

        total_aligned_ejections_Tr.append(np.mean(total_aligned_ejections_Tr_i))
        total_hybrid_ejections_Tr.append(np.mean(total_hybrid_ejections_Tr_i))
        total_random_ejections_Tr.append(np.mean(total_random_ejections_Tr_i))
    
    average_ejection_slingshot = np.mean([np.sum(strong_tr[i].ejection_slingshot_mask) for i in range(Nruns)])
    print("The average number of slingshot ejections is: %3.1f which is %2.1f %% of all binaries"%(average_ejection_slingshot,average_ejection_slingshot/N_total_systems*100))
    print("The average number of random ejections is: %3.1f which is %2.1f %% of all binaries" % (np.mean(total_random_ejections),np.mean(total_random_ejections)/N_total_systems*100))
    print("The average number of hybrid ejections is: %3.1f which is %2.1f %% of all binaries "%(np.mean(total_hybrid_ejections),np.mean(total_hybrid_ejections)/N_total_systems*100))
    print("The average number of aligned ejections is: %3.1f which is %2.1f %% of all binaries"%(np.mean(total_aligned_ejections),np.mean(total_aligned_ejections)/N_total_systems*100))
    print("--------------------")
    print("The average number of random ejections in strong triples is: %3.1f which is %2.1f %% of all strong triples" % (np.mean(total_random_ejections_Tr),np.mean(total_random_ejections_Tr)/N_strong_triples*100))
    print("The average number of hybrid ejections in strong triples is: %3.1f which is %2.1f%% of all strong triples"%(np.mean(total_hybrid_ejections_Tr),np.mean(total_hybrid_ejections_Tr)/N_strong_triples*100))
    print("The average number of aligned ejections in strong triples is: %3.1f which is %2.1f%% of all strong triples"%(np.mean(total_aligned_ejections_Tr),np.mean(total_aligned_ejections_Tr)/N_strong_triples*100))
    print("Slingshot ejections are %2.1f %% of strong triples"%(average_ejection_slingshot/N_strong_triples*100))

def assign_bhids(iso_bin, weak_tr, strong_tr, Nruns):
    iso_bin.bhid1 = iso_bin.binary_ids[:,0]
    iso_bin.bhid2 = iso_bin.binary_ids[:,1]

    weak_tr.bhid1 = weak_tr.bhid_inner[:,0]
    weak_tr.bhid2 = weak_tr.bhid_inner[:,1]
    weak_tr.bhid3 = weak_tr.bhid_outer[:,0]
    weak_tr.bhid4 = weak_tr.bhid_outer[:,1]

    for i in range(Nruns):
        strong_tr[i].bhid1 = strong_tr[i].bhid_inner[:,0]
        strong_tr[i].bhid2 = strong_tr[i].bhid_inner[:,1]
        strong_tr[i].bhid3 = strong_tr[i].bhid_outer[:,0]
        strong_tr[i].bhid4 = strong_tr[i].bhid_outer[:,1]

    iso_invalid_merger_mask = np.zeros_like(iso_bin.bhid1, dtype=bool)
    weak_triple_invalid_merger_mask = np.zeros_like(weak_tr.bhid1, dtype=bool)
    strong_triple_invalid_merger_mask = np.zeros_like(strong_tr[0].bhid1, dtype=bool)

    return None

def find_invalid_mergers(Nruns,strong_tr,weak_tr,iso_bin,ejection_mask_key,spin_i=0,include_slingshot=False):

    #assign_bhids(iso_bin, weak_tr, strong_tr,Nruns)

    bhid_cols_in_iso_bins = {"bhid1","bhid2"}
    bhid_cols_in_trips = {"bhid1","bhid2","bhid3","bhid4"}

    # Create masks for invalid mergers
    # These masks will be updated based on the ejection conditions
    iso_invalid_merger_mask = np.zeros_like(iso_bin.bhid1, dtype=bool)
    weak_triple_invalid_merger_mask = np.zeros_like(weak_tr.bhid1, dtype=bool)
    strong_triple_invalid_merger_mask = np.zeros_like(strong_tr.bhid1, dtype=bool)
    
    # Get the ejection masks for the current spin index
    iso_bin_eject_mask = getattr(iso_bin,ejection_mask_key)[spin_i]
    weak_tr_eject_mask = getattr(weak_tr,ejection_mask_key)[spin_i]
    strong_tr_eject_mask = getattr(strong_tr,ejection_mask_key)[spin_i]

    #iso binaries affected by "ejection_mask_key"

    for i, bhid_x in enumerate(bhid_cols_in_iso_bins):
        for j, bhid_y in enumerate(bhid_cols_in_iso_bins):
                common_occurrences_of_y_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(iso_bin,bhid_y)[iso_bin.merger_mask][iso_bin_eject_mask])
                bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]
                if len(bhidx_indices) > 0:
                        bhidy_indices = np.array([np.where(getattr(iso_bin,bhid_y) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                        different_indices_mask = bhidx_indices != bhidy_indices
                        bhidx_t_merger = iso_bin.t_merge[bhidx_indices]
                        bhidy_t_merger = iso_bin.t_merge[bhidy_indices]
                        iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidy_t_merger[different_indices_mask] < bhidx_t_merger[different_indices_mask]
                else:
                        continue
                
        for k,bhid_z in enumerate(bhid_cols_in_trips):
            common_occurrences_of_z_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(weak_tr,bhid_z)[weak_tr.merger_mask][weak_tr_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_z_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(weak_tr,bhid_z) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_t_merger = iso_bin.t_merge[bhidx_indices]
                bhid_wt_t_form = weak_tr.t_triple_form[bhidz_indices]
                iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_wt_t_form[different_indices_mask] < bhid1_t_merger[different_indices_mask]
                #weak_triple_invalid_merger_mask[bhidz_indices[different_indices_mask]] |= bhid1_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            else:
                continue

            common_occurrences_of_strong_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(strong_tr,bhid_z)[strong_tr.merger_mask][strong_tr_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_strong_ejected_in_x)[0]
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_t_merger = iso_bin.t_merge[bhidx_indices]
                bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_t_merger[different_indices_mask]
                #strong_triple_invalid_merger_mask[bhidz_indices[different_indices_mask]] |= bhid1_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            else:
                continue

            if include_slingshot:
                common_occurrences_of_sling_ejected_in_x = np.in1d(getattr(iso_bin,bhid_x),getattr(strong_tr,bhid_z)[strong_tr.ejection_slingshot_mask])
                bhidx_indices = np.where(common_occurrences_of_sling_ejected_in_x)[0]
                if len(bhidx_indices) > 0:
                    bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(iso_bin,bhid_x)[i])[0][0] for i in bhidx_indices])
                    different_indices_mask = bhidx_indices != bhidz_indices
                    bhid1_t_merger = iso_bin.t_merge[bhidx_indices]
                    bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                    iso_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_t_merger[different_indices_mask]
    
    #weak triples affected by "ejection_mask_key"

    for i,bhid_x in enumerate(bhid_cols_in_trips):
        for j, bhid_y in enumerate(bhid_cols_in_iso_bins):
            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(iso_bin,bhid_y)[iso_bin.merger_mask][iso_bin_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]
            
            if len(bhidx_indices) > 0:
                bhidy_indices = np.array([np.where(getattr(iso_bin,bhid_y) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidy_indices
                bhidy_t_merger = iso_bin.t_merge[bhidy_indices]
                bhid_wt_t_form = weak_tr.t_triple_form[bhidx_indices]
                weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidy_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]

            else:
                continue

        for k, bhid_z in enumerate(bhid_cols_in_iso_bins):
            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(weak_tr,bhid_z)[weak_tr.merger_mask][weak_tr_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]
            
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(weak_tr,bhid_z) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhidz_t_form = weak_tr.t_triple_form[bhidz_indices]
                bhid_wt_t_form = weak_tr.t_triple_form[bhidx_indices]
                weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidz_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]  
                
            else:
                continue
                
            common_occurrences_of_strong_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(strong_tr,bhid_z)[strong_tr.merger_mask][strong_tr_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_strong_ejected_in_x)[0]
                
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid1_t_form = weak_tr.t_triple_form[bhidx_indices]
                bhid_wt_t_form =strong_tr.t_triple_form[bhidz_indices]
                weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhid1_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            
            else:
                continue
            
            if include_slingshot:
                common_occurrences_of_sling_ejected_in_x = np.in1d(getattr(weak_tr,bhid_x),getattr(strong_tr,bhid_z)[strong_tr.ejection_slingshot_mask])
                bhidx_indices = np.where(common_occurrences_of_sling_ejected_in_x)[0]
                
                if len(bhidx_indices) > 0:
                    bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(weak_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                    different_indices_mask = bhidx_indices != bhidz_indices
                    bhid1_wt_t_form = weak_tr.t_triple_form[bhidx_indices]
                    bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                    weak_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_wt_t_form[different_indices_mask]
                
    #strong triples affected by "gw-key"
                
    for i,bhid_x in enumerate(bhid_cols_in_trips):
        for j, bhid_y in enumerate(bhid_cols_in_iso_bins):
            
            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(iso_bin,bhid_y)[iso_bin.merger_mask][iso_bin_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]

            if len(bhidx_indices) > 0:
                bhidy_indices = np.array([np.where(getattr(iso_bin,bhid_y) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidy_indices
                bhidy_t_merger = iso_bin.t_merge[bhidy_indices]
                bhid_wt_t_form = strong_tr.t_triple_form[bhidx_indices]
                strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidy_t_merger[different_indices_mask] < bhid_wt_t_form[different_indices_mask]

            else:
                continue

        for k, bhid_z in enumerate(bhid_cols_in_iso_bins):
            common_occurrences_of_y_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(weak_tr,bhid_z)[weak_tr.merger_mask][weak_tr_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_y_ejected_in_x)[0]
            
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(weak_tr,bhid_z) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhidz_t_form = weak_tr.t_triple_form[bhidz_indices]
                bhid_wt_t_form = strong_tr.t_triple_form[bhidx_indices]
                strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhidz_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]  
                
            else:
                continue
                
            common_occurrences_of_strong_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(strong_tr,bhid_z)[strong_tr.merger_mask][strong_tr_eject_mask])
            bhidx_indices = np.where(common_occurrences_of_strong_ejected_in_x)[0]
                
            if len(bhidx_indices) > 0:
                bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                different_indices_mask = bhidx_indices != bhidz_indices
                bhid_wt_t_form = strong_tr.t_triple_form[bhidz_indices]
                bhid1_t_form = strong_tr.t_triple_form[bhidx_indices]
                strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |= bhid1_t_form[different_indices_mask] < bhid_wt_t_form[different_indices_mask]
            
            else:
                continue
            
            if include_slingshot:
                common_occurrences_of_sling_ejected_in_x = np.in1d(getattr(strong_tr,bhid_x),getattr(strong_tr,bhid_z)[strong_tr.ejection_slingshot_mask])
                bhidx_indices = np.where(common_occurrences_of_sling_ejected_in_x)[0]
                
                if len(bhidx_indices) > 0:
                    bhidz_indices = np.array([np.where(getattr(strong_tr,bhid_z) == getattr(strong_tr,bhid_x)[i])[0][0] for i in bhidx_indices])
                    different_indices_mask = bhidx_indices != bhidz_indices
                    bhid1_wt_t_form = strong_tr.t_triple_form[bhidx_indices]
                    bhid_st_t_form = strong_tr.t_triple_form[bhidz_indices]
                    strong_triple_invalid_merger_mask[bhidx_indices[different_indices_mask]] |=  bhid_st_t_form[different_indices_mask] < bhid1_wt_t_form[different_indices_mask]

                else:
                    continue

    return iso_invalid_merger_mask,weak_triple_invalid_merger_mask,strong_triple_invalid_merger_mask

def assign_invalid_merger_mask(Nruns,N_kick_realization,strong_tr,weak_tr,iso_bin,spin_key,slingshot=False):

    ejection_kick_mask = 'ejection_'+spin_key+'_mask'
    iso_inv_masks = []
    weak_inv_masks = []
    strong_inv_masks = []
    for i in tqdm(range(N_kick_realization), desc="Processing kick realizations"):
        realization_iso_masks = []   # Store iso masks for each realization
        realization_weak_masks = []  # Store weak masks for each realization
        realization_strong_masks = [] # Store strong masks for each realization
        
        for j in tqdm(range(Nruns), desc=f"Processing runs for realization {i+1}/{N_kick_realization}", leave=False):
            # Find invalid mergers for the given realization and run
            iso_inv, weak_inv, strong_inv = find_invalid_mergers(Nruns,
                strong_tr[j], weak_tr, iso_bin, ejection_kick_mask, spin_i=i,include_slingshot=slingshot)
            # Append masks for this run
            realization_iso_masks.append(iso_inv)
            realization_weak_masks.append(weak_inv)
            realization_strong_masks.append(strong_inv)
        
        # Append realization masks to the main lists
        iso_inv_masks.append(realization_iso_masks)
        weak_inv_masks.append(realization_weak_masks)
        strong_inv_masks.append(realization_strong_masks)

    # Convert the nested lists into 3D arrays if necessary
    iso_inv_masks = np.array(iso_inv_masks)  # Requires numpy
    weak_inv_masks = np.array(weak_inv_masks)
    strong_inv_masks = np.array(strong_inv_masks)

    if(slingshot):
        setattr(iso_bin,spin_key+'_invalid_mask', iso_inv_masks)
        setattr(weak_tr,spin_key+'_invalid_mask', weak_inv_masks)
        setattr(strong_tr[0],spin_key+'_invalid_mask',strong_inv_masks)
    else:
        setattr(iso_bin,spin_key+'_invalid_mask_wo_sling', iso_inv_masks)
        setattr(weak_tr,spin_key+'_invalid_mask_wo_sling', weak_inv_masks)
        setattr(strong_tr[0],spin_key+'_invalid_mask_wo_sling',strong_inv_masks)

    return None

def assigning_invalid_merger_masks_to_objs(strong_tr, weak_tr, iso_bin, N_kick_realization, Nruns):

    assign_bhids(iso_bin, weak_tr, strong_tr,Nruns)

    assign_invalid_merger_mask(Nruns,N_kick_realization,strong_tr,weak_tr,iso_bin,'random')
    assign_invalid_merger_mask(Nruns,N_kick_realization,strong_tr,weak_tr,iso_bin,'hybrid')
    assign_invalid_merger_mask(Nruns,N_kick_realization,strong_tr,weak_tr,iso_bin,'aligned')

    assign_invalid_merger_mask(Nruns,N_kick_realization,strong_tr,weak_tr,iso_bin,'random',slingshot=True)
    assign_invalid_merger_mask(Nruns,N_kick_realization,strong_tr,weak_tr,iso_bin,'hybrid',slingshot=True)
    assign_invalid_merger_mask(Nruns,N_kick_realization,strong_tr,weak_tr,iso_bin,'aligned',slingshot=True)

    return None

def calculate_invalid_merger_fractions(strong_tr, weak_tr, iso_bin, n_kick_realization, nruns, include_slingshot=True):
    """
    Calculate the fraction of mergers that are invalid due to ejections for
    different spin configurations.
    
    Parameters:
    -----------
    strong_tr : list
        List of strong triple objects
    weak_tr : object
        Weak triple object
    iso_bin : object
        Isolated binary object
    n_kick_realization : int
        Number of kick realizations
    nruns : int
        Number of simulation runs
    include_slingshot : bool, optional
        Whether to include slingshot ejections in the calculation (default: True)
        
    Returns:
    --------
    dict
        Dictionary with keys 'random', 'hybrid', 'aligned' containing the invalid merger
        fractions for each realization and run
    """
    # Set up configuration for calculation
    spin_types = ['random', 'hybrid', 'aligned']
    fractions = {spin_type: [] for spin_type in spin_types}
    
    # Get merger counts once since they're constant
    iso_mergers = np.sum(iso_bin.merger_mask)
    weak_tr_mergers = np.sum(weak_tr.merger_mask)
    
    # Select appropriate mask attribute suffix based on slingshot inclusion
    mask_suffix = "_invalid_mask" if include_slingshot else "_invalid_mask_wo_sling"
    
    # Calculate for each realization
    for i in range(n_kick_realization):
        realization_fractions = {spin_type: [] for spin_type in spin_types}
        
        for j in range(nruns):
            # Get strong triple mergers for this run
            strong_tr_mergers = np.sum(strong_tr[j].merger_mask)
            total_mergers = iso_mergers + weak_tr_mergers + strong_tr_mergers
            
            # Calculate fractions for each spin type
            for spin_type in spin_types:
                mask_attr = f"{spin_type}{mask_suffix}"
                
                # Count invalid mergers across all system types
                iso_invalid = np.sum(iso_bin.merger_mask & getattr(iso_bin, mask_attr)[i][j])
                weak_invalid = np.sum(weak_tr.merger_mask & getattr(weak_tr, mask_attr)[i][j])
                strong_invalid = np.sum(strong_tr[j].merger_mask & getattr(strong_tr[0], mask_attr)[i][j])
                
                # Calculate fraction and store
                invalid_fraction = (iso_invalid + weak_invalid + strong_invalid) / total_mergers
                realization_fractions[spin_type].append(invalid_fraction)
        
        # Store results for this realization
        for spin_type in spin_types:
            fractions[spin_type].append(realization_fractions[spin_type])
    
    return fractions

    
def summarize_invalid_merger_fractions(strong_tr, weak_tr, iso_bin, N_kick_realization, Nruns):
        """
        Summarize and calculate invalid merger fractions with and without slingshot ejections.

        Parameters:
        - strong_tr: List of strong triple objects.
        - weak_tr: Weak triple object.
        - iso_bin: Isolated binary object.
        - N_kick_realization: Number of kick realizations.
        - Nruns: Number of simulation runs.

        Returns:
        - dict: Summary statistics for invalid merger fractions with and without slingshot ejections.
        """

        assigning_invalid_merger_masks_to_objs(strong_tr, weak_tr, iso_bin, N_kick_realization, Nruns)

        # Calculate invalid merger fractions with slingshot ejections
        tot_invalid_merger_fraction = calculate_invalid_merger_fractions(
            strong_tr, weak_tr, iso_bin, N_kick_realization, Nruns, include_slingshot=True
        )

        # Calculate invalid merger fractions without slingshot ejections
        tot_invalid_merger_fraction_wo_sling = calculate_invalid_merger_fractions(
            strong_tr, weak_tr, iso_bin, N_kick_realization, Nruns, include_slingshot=False
        )

        # Summarize results
        summary = {"with_slingshot": {}, "without_slingshot": {}, "slingshot_contribution": {}}

        # With slingshot ejections
        for spin_type in ['random', 'hybrid', 'aligned']:
            mean_fraction = np.mean(tot_invalid_merger_fraction[spin_type]) * 100
            print(f"{mean_fraction:.1f}% of mergers won't happen due to GW {spin_type} + slingshot ejections")

        # Without slingshot ejections
        for spin_type in ['random', 'hybrid', 'aligned']:
            mean_fraction = np.mean(tot_invalid_merger_fraction_wo_sling[spin_type]) * 100
            print(f"{mean_fraction:.1f}% of mergers won't happen due to GW {spin_type} ejections")

        # Calculate percentage of slingshot contribution in random+slingshot
        random_with_slingshot = np.mean(tot_invalid_merger_fraction['random']) * 100
        random_without_slingshot = np.mean(tot_invalid_merger_fraction_wo_sling['random']) * 100
        slingshot_percentage = (random_with_slingshot - random_without_slingshot) / random_with_slingshot * 100
        print(f"  - Slingshot ejections account for {slingshot_percentage:.1f}% of all invalid mergers in random+slingshot")


        return None