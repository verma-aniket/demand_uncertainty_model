# ============================================================
# Efficient Demand Processing Script
# - Processes chunks sequentially
# - Writes directly to memmap files
# - Avoids storing large intermediate arrays
# ============================================================

import sys
from pathlib import Path
import numpy as np
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
proj_dir = REPO_ROOT / "Results" / "Projections" / "HH"

# loop over households
hh_list = [0, 1, 2]
hh_list = [0] # for testing, comment out for full ensemble
n_chunks = 1
array_ids = list(range(n_chunks))

for hh in hh_list:
    
    print(f"Processing household: {hh}")
    
    input_monthly_file = proj_dir / f"hh_proj_{hh}.npy"
    output_annual_file  = proj_dir / f"hh_annual_{hh}.npy"
    
    # load data
    proj_arr = np.load(input_monthly_file).astype(np.float32)

    num_proj, num_months = proj_arr.shape
    num_years = num_months // 12
    
    # Monthly to Annual
    proj_ann = proj_arr.reshape(
        proj_arr.shape[0], num_years, 12
    ).sum(axis=2)
    
    # convert to np.float32 and save
    proj_ann = proj_ann.astype(np.float32)
    np.save(output_annual_file, proj_ann)
    
    print("Done.")