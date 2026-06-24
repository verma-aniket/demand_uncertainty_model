import sys
from pathlib import Path
import numpy as np

# 1. LINK CORE FOLDERS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

def main():
    # 2. ROUTE OUTPUT DIRECTORY DYNAMICALLY (Simulator > Inputs > ID_Grid)
    output_dir = REPO_ROOT / "Data" / "Simulator" / "Inputs" / "ID_Grid"

    # 3. DESIGN SCENARIO BOUNDS
    num_dcc = 75                      # Number of DCC parameter sets
    num_cc = 50                       # Number of climate change scenarios
    num_pop = 50                      # Number of population growth scenarios
    num_inf = 10                      # Number of water price inflation scenarios
    cons_a = 2                        # Active conservation switch
    num_hh_comp = 3                   # Number of household composition seeds

    # Generate indices
    dcc_ids = np.arange(num_dcc)
    cc_ids = np.arange(num_cc)
    pop_ids = np.arange(num_pop)
    inf_ids = np.arange(num_inf)
    cons_a_ids = np.arange(cons_a)
    hh_comp_ids = np.arange(num_hh_comp)

    # 4. COMPUTE DIMENSIONAL MESHGRID 
    print("Constructing multi-dimensional simulation ID matrix...")
    num_proj = num_dcc * num_cc * num_pop * num_inf * cons_a * num_hh_comp
    
    id_grid = np.meshgrid(dcc_ids, cc_ids, pop_ids, inf_ids, cons_a_ids, hh_comp_ids, indexing='ij')
    id_grid = np.stack(id_grid, axis=-1).reshape(-1, 6)

    # Optimize memory footprint 
    id_grid = id_grid.astype(np.int8)

    # 5. EXECUTE STRUCTURAL NPY CHUNKING
    print(f"Dividing {num_proj:,} scenarios into flattened chunk components...")
    num_chunks = 200
    proj_per_chunk = num_proj // num_chunks

    for i in range(num_chunks):
        # Determine exact slices to guarantee no bounds truncation
        start_idx = i * proj_per_chunk
        end_idx = (i + 1) * proj_per_chunk if i < (num_chunks - 1) else num_proj
        
        chunk = id_grid[start_idx:end_idx]
        
        # Save explicitly down to Simulator/Inputs/ID_Grid/
        chunk_file = output_dir / f"input_id_chunk_{i}.npy"
        np.save(chunk_file, chunk)

    print(f"Exported {num_chunks} matrix chunks safely to: {output_dir.relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    main()