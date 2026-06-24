# import root libraries
import os
import sys
import pandas as pd
from pathlib import Path

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Add root to python path
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.utils import read_migration_data

def main():
    # define file paths
    data_dir = REPO_ROOT / "data" / "Population" / "Migration_Raw"
    data_dir_2 = REPO_ROOT / "data" / "Population" / "Population_Raw"
    output_dir = REPO_ROOT / "data" / "Population" / "Clean_Data"

    # read raw data file names
    file_list = os.listdir(data_dir)
    file_list_2 = os.listdir(data_dir_2)

    # Extract, Process, and Save Data
    print("Building Net Migration Dataset...")
    net_mig_data = read_migration_data.get_net_migration_dataset(data_dir, file_list)
    net_mig_data.to_csv(output_dir / "GEC_Model_Data_Net_Flows.csv", index=False)

    print("Building All SC Related Flows Dataset...")
    all_mig_data = read_migration_data.get_all_flows(data_dir, file_list)
    all_mig_data.to_csv(output_dir / "GEC_Model_Data_All_Flows.csv", index=False)

    print("Building Average Neighbors Dataset...")
    avg_N = read_migration_data.get_avg_neighbors(data_dir, file_list)
    avg_N.to_csv(output_dir / "Average_Neighbors.csv", index=False)

    print("Building Annual MSA Population Data...")
    msa_pop_data = read_migration_data.read_pop_data(data_dir_2, file_list_2)
    msa_pop_data.to_csv(output_dir / "MSA_Population_Data.csv", index=False)

    print("Beginning Migration Fluctuations Calculations (Diff_Data)...")
    all_mig_data = pd.read_csv(output_dir / 'GEC_Model_Data_All_Flows.csv')
    diff_df = read_migration_data.calculate_migration_fluctuations(all_mig_data)
    diff_data_path = output_dir / "Diff_Data.csv"
    diff_df.to_csv(diff_data_path, index=False)
    
    print("Data extraction complete.")

if __name__ == "__main__":
    main()