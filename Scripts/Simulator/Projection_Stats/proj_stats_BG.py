import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# 1. LINK CORE FOLDERS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.utils.simulator_functions import weighted_percentiles

def main():
    # 2. ROUTE ENVIROMENT PATHS DYNAMICALLY
    proj_dir = REPO_ROOT / "Results" / "Projections" / "BG"
    id_file = REPO_ROOT / "Data" / "Simulator" / "Inputs" / "BG_Data" / "ID_Grid" / "id_mat.npy"
    scen_file = REPO_ROOT / "Data" / "Simulator" / "Scenarios" / "DCC_scenarios.csv"
    output_dir = REPO_ROOT / "Results" / "Projection_Statistics"

    # 3. LOAD ID GRID AND DCC WEIGHTS
    id_mat = np.load(id_file)

    # Read DCC model coefficient weights from column index 20
    dcc_weights_raw = np.loadtxt(scen_file, delimiter=",", skiprows=1, usecols=20)
    weights = dcc_weights_raw[id_mat[:, 0].astype(int)]

    # alter weights for testing - comment out for full ensemble
    weights = weights[0:90]

    # Target FIPS block codes
    fips_array = [60871010002, 60871012001, 60871208003]
    fips_array = [60871010002] # only 1 BG for testing, comment or delete for full ensemble
    perc_array = np.array([50, 0.5, 1, 2.5, 5, 10, 25, 75, 90, 95, 97.5, 99, 99.5])

    # 4. BATCH ITERATE TARGET BLOCK GROUPS
    for fips_code in fips_array:
        print(f"\nProcessing block group: {fips_code}")

        # Stream scenario files using efficient file memory mappings
        proj_file = proj_dir / f"proj_annual_{fips_code}.npy"
        proj_mat = np.load(proj_file, mmap_mode="r")
        n_periods = proj_mat.shape[1]
        
        # Build matrix tracker to hold results before Dataframe conversion
        results_mat = np.zeros((n_periods, 15))
        
        # Compute time series statistics across temporal windows
        for m in tqdm(range(n_periods), desc="Time Period", colour="green", ncols=100):
            # Calculate weighted mean
            mean_val = np.average(proj_mat[:, m], weights=weights)
            results_mat[m, 0] = mean_val
            
            # Calculate standard deviation using residual variances
            variance_val = np.average((proj_mat[:, m] - mean_val) ** 2, weights=weights)
            results_mat[m, 1] = np.sqrt(variance_val)
            
            # Extract weighted percentile arrays
            results_mat[m, 2:] = weighted_percentiles(values=proj_mat[:, m], weights=weights, percentiles=perc_array)
        
        # Consolidate arrays into dataframe columns
        results_df = pd.DataFrame({
            "Mean": results_mat[:, 0],
            "Std": results_mat[:, 1],
            "Median": results_mat[:, 2],
            "P0_5": results_mat[:, 3],
            "P1": results_mat[:, 4],
            "P2_5": results_mat[:, 5],
            "P5": results_mat[:, 6],
            "P10": results_mat[:, 7],
            "P25": results_mat[:, 8],
            "P75": results_mat[:, 9],
            "P90": results_mat[:, 10],
            "P95": results_mat[:, 11],
            "P97_5": results_mat[:, 12],
            "P99": results_mat[:, 13],
            "P99_5": results_mat[:, 14]
        })
        
        # Export metrics down to target project path 
        output_file = output_dir / f"BG_proj_stats_{fips_code}.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Saved: {output_file.name}")
        
        # Explicitly release memory allocations for the next block group
        del proj_mat, results_mat

if __name__ == "__main__":
    main()