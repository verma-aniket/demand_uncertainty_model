# import libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# 1. LINK CORE FOLDERS (Climb 2 levels out of scripts/demand/ to project root)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

# Add root to python path so it can find the 'src' directory if needed later
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

def main():
    print("Reading Input IDs...\n")

    # read input grid
    id_mat = np.load(REPO_ROOT / "Data" / "Simulator" / "Inputs" / "HH_Data" /  "hh_id_grid.npy")

    # Read DCC model coefficient weights
    DCC_weights = np.loadtxt(REPO_ROOT / "Data" / "Simulator" / "Scenarios" / 'DCC_scenarios.csv', delimiter=",", skiprows=1, usecols=20)
    weights = DCC_weights[id_mat[:, 0].astype(int)]

    # alter weights for testing - comment out for full ensemble
    weights = weights[0:9]
    id_mat = id_mat[0:9]

    # Define constants
    num_groups = id_mat.shape[1] - 1 # exclude household composition scenario group

    # build id masks and (summed) group weight vectors for each group
    print("Building ID masks and weight vectors...\n")
    group_indices = []
    group_weights = []
    for n in range(num_groups):
        id_vec = id_mat[:, n]
        unique, inverse = np.unique(id_vec, return_inverse=True)
        idx = {int(g): np.where(inverse == i)[0] for i, g in enumerate(unique)}
        group_indices.append(idx)
        gw = np.array([np.sum(weights[idx]) for _, idx in group_indices[n].items()])
        group_weights.append(gw)

    hh_ids = [0,1,2]
    hh_ids = [0] # only 1 HH for testing, comment or delete for full ensemble

    for hh_id in hh_ids:
        
        print(f"Processing household: {hh_id}...\n")
        
        # read projections data
        proj_file = REPO_ROOT / "Results" / "Projections" / "HH" / f"hh_proj_{hh_id}.npy"
        proj_mat = np.load(proj_file, mmap_mode="r")
        
        print("Computing weight mean and variance...\n")
        
        # Compute Global weighted mean and variance for each time period
        
        # Weighted mean
        Y_bar = np.average(proj_mat, axis=0, weights=weights)
        
        # Weighted variance
        Var_Y = np.average((proj_mat - Y_bar)**2, axis=0, weights=weights)
        
        # Define constants
        num_t = proj_mat.shape[1]
        var_ratio = np.zeros((num_groups+1, num_t))
        
        print("Computing variance ratios for each group...\n")
        
        # Compute variance ratios for each group for all time periods at once
        # use variance of conditional expectations (VCE) approach 
        for n in tqdm(range(num_groups), desc="Group No.", ncols=100, colour="green"):
        
            # build vector to store conditional expectations
            num_g = len(group_indices[n])
            E_Y_bar_g = np.zeros((num_g, num_t))
            
            for i, g in enumerate(sorted(group_indices[n])):
                
                # get indicies for this set of unique group IDs
                idx = group_indices[n][g]
                
                E_Y_bar_g[i] = np.average(proj_mat[idx], axis=0, weights=weights[idx])
            
            # compute VCE index - also differently based on group type
            weights_g = group_weights[n][:, None]
            VCE = np.sum(weights_g * (E_Y_bar_g - Y_bar)**2, axis=0) / np.sum(weights_g)
            var_ratio[n] = VCE / Var_Y
        
        # alternative within-group variance approach 1 - all groups
        var_ratio[-1] = 1 - np.sum(var_ratio[0:num_groups],axis=0)
        
        print("Saving Results...\n")
        
        # Store results for plotting
        var_ratio_df = pd.DataFrame(data=100*var_ratio.T)
        
        # add year and month columns
        if num_t == 312:
            var_ratio_df['Year'] = np.repeat(np.arange(2025, 2051), 12)
            var_ratio_df['Month'] = np.tile(np.arange(1, 13), num_t//12)
            current_order = var_ratio_df.columns
            new_order = [current_order[-2], current_order[-1]] + current_order[0:-2].to_list()
            var_ratio_df = var_ratio_df[new_order]
        else:
            var_ratio_df['Year'] = np.arange(2025, 2051)
            current_order = var_ratio_df.columns
            new_order = [current_order[-1]] + current_order[0:-1].to_list()
            var_ratio_df = var_ratio_df[new_order]
        
        # save to csv
        output_file = REPO_ROOT / "Results" / "Variance_Decomposition" / f"HH_monthly_{hh_id}.csv"
        var_ratio_df.to_csv(output_file, index=False, header=True)
        
        # explicitly free memory
        del proj_mat, Y_bar, Var_Y, var_ratio
        
    # End of script

if __name__ == "__main__":
    main()