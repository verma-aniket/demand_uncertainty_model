# ============================================================
# Combined Demand Processing Script
# - Reads monthly .npy chunks (200 sets)
# - Applies conservation scenarios
# - Converts to annual
# - Saves ONLY two merged files:
#       1) merged monthly
#       2) merged annual
# ============================================================

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

# set working directories
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# route directories
data_dir = REPO_ROOT / "Data" / "Simulator" / "Scenarios"
proj_dir = REPO_ROOT / "Results" / "Projections" / "BG"

# loop over block-grops
fips_array = [60871010002, 60871012001, 60871208003]
fips_array = [60871010002] # comment out for full ensemble

for fips_code in fips_array:
    
    print(f"Processing block group: {fips_code}")
    
    output_monthly_file = proj_dir / f"proj_monthly_{fips_code}.npy"
    output_annual_file  = proj_dir / f"proj_annual_{fips_code}.npy"

    n_chunks = 1 # set to 1 for testing, change to 10 for full ensemble
    array_ids = list(range(n_chunks))

    # ------------------------------------------------------------
    # LOAD CONSERVATION SCENARIOS
    # ------------------------------------------------------------
    
    cons_mat = np.loadtxt(data_dir / "passive_conservation_scenarios_10.csv",
        delimiter=","
    ).T.astype(np.float32)
    
    num_cons = cons_mat.shape[0]
    
    # ------------------------------------------------------------
    # DETERMINE OUTPUT SHAPES (read one chunk only)
    # ------------------------------------------------------------
    
    sample = np.load(proj_dir / f"proj_chunk_0_{fips_code}.npy").astype(np.float32)
    
    num_proj, num_months = sample.shape
    num_years = num_months // 12
    
    monthly_rows = n_chunks * num_proj * num_cons
    monthly_cols = num_months
    
    annual_rows = monthly_rows
    annual_cols = num_years
    
    del sample
    
    # ------------------------------------------------------------
    # CREATE MEMORY-MAPPED OUTPUT FILES
    # ------------------------------------------------------------
    
    merged_monthly = open_memmap(
        output_monthly_file,
        mode="w+",
        dtype=np.float32,
        shape=(monthly_rows, monthly_cols)
    )
    
    merged_annual = open_memmap(
        output_annual_file,
        mode="w+",
        dtype=np.float32,
        shape=(annual_rows, annual_cols)
    )
    
    # ------------------------------------------------------------
    # PROCESS CHUNKS SEQUENTIALLY
    # ------------------------------------------------------------
    
    start_m = 0
    start_a = 0
    
    for chunk_no in tqdm(array_ids, desc="Processing Chunks", ncols=100, colour="green"):
    
        # --------------------------------------------------------
        # LOAD CHUNK
        # --------------------------------------------------------
    
        file_name = f"proj_chunk_{chunk_no}_{fips_code}.npy"
        proj_arr = np.load(proj_dir / file_name).astype(np.float32)
    
        num_proj = proj_arr.shape[0]
    
        # --------------------------------------------------------
        # APPLY CONSERVATION
        # --------------------------------------------------------
    
        proj_cons = proj_arr[:, None, :] * (1 - cons_mat[None, :, :])
        proj_cons = proj_cons.reshape(num_proj * num_cons, num_months)
    
        # --------------------------------------------------------
        # MONTHLY → ANNUAL
        # --------------------------------------------------------
    
        proj_ann = proj_cons.reshape(
            proj_cons.shape[0], num_years, 12
        ).sum(axis=2)
    
        # --------------------------------------------------------
        # WRITE DIRECTLY TO MEMMAP
        # --------------------------------------------------------
    
        end_m = start_m + proj_cons.shape[0]
        end_a = start_a + proj_ann.shape[0]
    
        merged_monthly[start_m:end_m] = proj_cons
        merged_annual[start_a:end_a]  = proj_ann
    
        start_m = end_m
        start_a = end_a
    
        # explicitly free memory
        del proj_arr, proj_cons, proj_ann
    
    # ensure data written
    merged_monthly.flush()
    merged_annual.flush()
    
    print("Done.")
