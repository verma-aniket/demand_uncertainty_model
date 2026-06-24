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
    # 2. ROUTE REPO-RELATIVE INPUT AND OUTPUT PATHS
    proj_file = REPO_ROOT / "Results" / "Projections" / "City" / "proj_annual.npy"
    id_file = REPO_ROOT / "Data" / "Simulator" / "Inputs" / "ID_Grid" / "id_mat.npy"
    scen_file = REPO_ROOT / "Data" / "Simulator" / "Scenarios" / "DCC_scenarios.csv"
    output_file = REPO_ROOT / "Results" / "Projection_Statistics" / "city_proj_stats.csv"

    # 3. LOAD DATA ARRAYS USING EFFICIENT MEMORY MAPPING
    proj_mat = np.load(proj_file, mmap_mode="r")
    id_mat = np.load(id_file)

    # Read DCC model coefficient weights from column index 20
    dcc_weights_raw = np.loadtxt(scen_file, delimiter=",", skiprows=1, usecols=20)
    weights = dcc_weights_raw[id_mat[:, 0].astype(int)]

    # alter weights for testing - comment out for full ensemble
    weights = weights[0:90]

    # 4. INITIALIZE MATRIX RESULTS AND PERCENTILE BINS
    n_months = proj_mat.shape[1]
    results_mat = np.zeros((n_months, 15))
    perc_array = np.array([50, 0.5, 1, 2.5, 5, 10, 25, 75, 90, 95, 97.5, 99, 99.5])

    # 5. GENERATE STATISTICS ALONG THE TIME AXIS
    print(f"Aggregating weighted statistical parameters across {n_months} intervals...")
    for m in tqdm(range(n_months), desc="Time Period", colour="green", ncols=100):
        # Calculate mean
        mean_val = np.average(proj_mat[:, m], weights=weights)
        results_mat[m, 0] = mean_val
        
        # Calculate weighted standard deviation via residual variances
        variance_val = np.average((proj_mat[:, m] - mean_val) ** 2, weights=weights)
        results_mat[m, 1] = np.sqrt(variance_val)
        
        # Calculate structural percentiles
        results_mat[m, 2:] = weighted_percentiles(values=proj_mat[:, m], weights=weights, percentiles=perc_array)

    # 6. EXPORT CLEAN STRUCTURAL DATA FRAME CHUNKS
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

    results_df.to_csv(output_file, index=False)
    print(f"Projection statistics saved.")

if __name__ == "__main__":
    main()
