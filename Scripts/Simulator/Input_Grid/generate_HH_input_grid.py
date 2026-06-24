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
    output_dir = REPO_ROOT / "Data" / "Simulator" / "Inputs" / "HH_Data"

    # Design Input Scenario Grid
    num_dcc = 75                    # number of DCC parameter sets
    num_cc = 50                     # number of climate change scenarios
    num_inf = 10                    # number of water price inflation scenarios
    num_cons_a = 10                 # active conservation scenarios
    num_cons_p = 10                 # passive conservation scenarios

    # sample scenerio indices
    dcc_ids = np.arange(num_dcc)
    cc_ids = np.arange(num_cc)
    inf_ids = np.arange(num_inf)
    cons_a_ids = np.arange(num_cons_a)
    cons_p_ids = np.arange(num_cons_p)

    # Define Input ID matrix
    num_proj = num_dcc*num_cc*num_inf*num_cons_a*num_cons_p
    id_grid = np.meshgrid(dcc_ids, cc_ids, inf_ids, cons_a_ids, cons_p_ids, indexing='ij')
    id_grid = np.stack(id_grid, axis=-1).reshape(-1, 5)

    # convert to np.int8
    id_grid = id_grid.astype(np.int8)

    # save all input IDs into one numpy file
    np.save(output_dir / f"hh_id_grid.npy", id_grid)

    print(f"Exported HH input grid matrix chunks safely to: {output_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()