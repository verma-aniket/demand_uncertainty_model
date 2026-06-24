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
proj_dir = REPO_ROOT / "Results" / "Projections" / "City"
output_monthly_file = proj_dir / "proj_monthly.npy"
output_annual_file  = proj_dir / "proj_annual.npy"

n_chunks = 1 # set to 1 for testing, change to 200 for full ensemble
array_ids = list(range(n_chunks))

# ------------------------------------------------------------
# LOAD CONSERVATION SCENARIOS (.npy now)
# shape should be (num_cons, num_months)
# ------------------------------------------------------------

cons_mat = np.loadtxt(data_dir / 'passive_conservation_scenarios_10.csv', delimiter=",").T
num_cons = cons_mat.shape[0]

# ------------------------------------------------------------
# PROCESS FUNCTION
# ------------------------------------------------------------

def process_chunk(chunk_no):

    file_name = f"proj_chunk_{chunk_no}.npy"
    proj_arr = np.load(proj_dir / file_name)  # shape: (num_proj, num_months)

    num_proj, num_months = proj_arr.shape
    num_years = num_months // 12

    # --------------------------------------------------------
    # APPLY CONSERVATION (Vectorized )
    # --------------------------------------------------------
    # Expand dimensions to broadcast:
    # proj_arr: (num_proj, num_months)
    # cons_mat: (num_cons, num_months)
    # Result:   (num_proj, num_cons, num_months)

    proj_cons = proj_arr[:, None, :] * (1 - cons_mat[None, :, :])

    # reshape to (num_proj*num_cons, num_months)
    proj_cons = proj_cons.reshape(num_proj * num_cons, num_months)

    # --------------------------------------------------------
    # MONTHLY → ANNUAL (Vectorized)
    # --------------------------------------------------------
    proj_ann = proj_cons.reshape(
        proj_cons.shape[0], num_years, 12
    ).sum(axis=2)

    return proj_cons, proj_ann


# ------------------------------------------------------------
# STEP 1: RUN ALL CHUNKS IN PARALLEL
# ------------------------------------------------------------

results = Parallel(n_jobs=-1)(
    delayed(process_chunk)(c)
    for c in tqdm(array_ids, desc="Processing Chunks", ncols=100, colour="green")
)

# ------------------------------------------------------------
# STEP 2: DETERMINE FINAL SHAPES
# ------------------------------------------------------------

monthly_rows = sum(r[0].shape[0] for r in results)
monthly_cols = results[0][0].shape[1]

annual_rows = sum(r[1].shape[0] for r in results)
annual_cols = results[0][1].shape[1]

# ------------------------------------------------------------
# STEP 3: CREATE MEMORY-MAPPED OUTPUT FILES
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
# STEP 4: FILL MERGED FILES
# ------------------------------------------------------------

start_m = 0
start_a = 0

for monthly_chunk, annual_chunk in tqdm(results, desc="Merging", ncols=100, colour="green"):

    end_m = start_m + monthly_chunk.shape[0]
    end_a = start_a + annual_chunk.shape[0]

    merged_monthly[start_m:end_m] = monthly_chunk
    merged_annual[start_a:end_a]  = annual_chunk

    start_m = end_m
    start_a = end_a

# ensure data is written
merged_monthly.flush()
merged_annual.flush()

print("Done.")
