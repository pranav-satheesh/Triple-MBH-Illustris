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
