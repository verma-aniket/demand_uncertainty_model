# import libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# 1. LINK CORE FOLDERS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# define number of chunks to read
chunk_array = np.arange(0,1) # change to 200 for full city ensemble, ID grid is identical for city and block-group projections 

# set directories (City)
grid_dir = REPO_ROOT / "Data" / "Simulator" / "Inputs" / "ID_Grid"

# define number of conservation scenarios
num_cons = 10

# define function that transforms file
def process_input_id_grid(chunk_no):
    
    # read input grid
    file_name = f"input_id_chunk_{chunk_no}.npy"
    id_grid = np.load(grid_dir / file_name)

    # add column for conservation scenarios
    id_grid = np.repeat(id_grid, num_cons, axis=0)
    id_grid = np.hstack((id_grid, np.array([np.tile(np.arange(num_cons), int(id_grid.shape[0]/num_cons))]).T))

    # return chunk to growth final input grid
    return id_grid

# process input ID grid chunks in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_input_id_grid)(p) for p in tqdm(chunk_array, desc="Chunk No.", ncols=100, colour="green")
)

# concatenate results
final_input_grid = np.vstack(results)

# save as np.int8
final_input_grid = final_input_grid.astype(np.int8)

# re-order columns
print("Reordering columns...\n")
new_order = [0, 1, 2, 3, 4, 6, 5]
final_input_grid = final_input_grid[:, new_order]

print("Saving data...\n")

# save final input grid
np.save(grid_dir / "id_mat.npy", final_input_grid)
