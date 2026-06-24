# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 1. LINK CORE FOLDERS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import the offloaded utility function
from src.utils.climate_functions import cc_df_to_mat

def main():
    # 2. ROUTE DIRECTORIES RELATIVE TO REPO ROOT
    climate_data_dir = REPO_ROOT / "Data" / "Climate"
    
    # Input files
    swg_data_file = climate_data_dir / "SWG_results.csv"
    sample_ids_file = climate_data_dir / "drought_IDs.csv"
    cf_data_file = climate_data_dir / "Cal_Adapt_CF_Data.csv"
    
    # Output destinations  
    precip_output = climate_data_dir / "precip_mat_final.csv"
    temp_output = climate_data_dir / "temp_mat_final.csv"

    # 3. SET RUNTIME SCENARIO METRICS
    start_year = 2025
    stop_year = 2050
    num_years = stop_year - start_year + 1
    num_months_total = num_years * 12  # 26 years * 12 months = 312 months

    print("Loading stochastic timelines and sampling indices...")
    swg_mat = np.loadtxt(swg_data_file, delimiter=",", usecols=[0, 1, 2])
    sampled_sets = np.int64(np.loadtxt(sample_ids_file, delimiter=","))

    # Clean and shape daily matrices
    extra_years = (swg_mat.shape[0] // 365) % num_years
    if extra_years > 0:
        swg_mat = swg_mat[:-extra_years * 365]
    num_sets = (swg_mat.shape[0] // 365) // num_years

    precip_mat = swg_mat[:, 0].reshape(num_sets, num_years, 365)
    tmax_mat = swg_mat[:, 1].reshape(num_sets, num_years, 365)

    # 4. TEMPORAL MONTHLY AGGREGATIONS
    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day_to_month = np.repeat(np.arange(0, 12), month_lengths)

    precip_mon = np.zeros((precip_mat.shape[0], num_years, 12))
    temp_mon = np.zeros_like(precip_mon)
    for m in range(12):
        precip_mon[:, :, m] = precip_mat[:, :, day_to_month == m].sum(axis=2)
        temp_mon[:, :, m] = tmax_mat[:, :, day_to_month == m].mean(axis=2)

    # Flatten and subset to sampled IDs
    precip_mon = precip_mon.reshape(precip_mat.shape[0], num_months_total)[sampled_sets]
    temp_mon = temp_mon.reshape(temp_mon.shape[0], num_months_total)[sampled_sets]

    # Convert to DataFrames for merging Change Factors
    precip_df = pd.DataFrame(precip_mon.reshape(-1), columns=['P'])
    temp_df = pd.DataFrame(temp_mon.reshape(-1), columns=['Tmax'])

    # Broadcast indices
    years = np.tile(np.repeat(np.arange(start_year, stop_year + 1, 1), 12), sampled_sets.shape[0])
    months = np.tile(np.arange(1, 13, 1), num_years * sampled_sets.shape[0])
    sets = np.repeat(np.arange(1, sampled_sets.shape[0] + 1, 1), num_months_total)
    
    for df in [precip_df, temp_df]:
        df['Year'] = years
        df['Month'] = months
        df['Set'] = sets

    # 5. INTEGRATE CHANGE FACTORS IN-MEMORY
    print("Integrating Cal-Adapt change factors with generated timelines...")
    cf_data = pd.read_csv(cf_data_file)
    cf_precip = cf_data.loc[cf_data['Variable'] == 'pr'].reset_index(drop=True)
    cf_temp = cf_data.loc[cf_data['Variable'] == 'tasmax'].reset_index(drop=True)

    precip_df = precip_df.merge(cf_precip, on=['Year', 'Month'], how='outer')
    temp_df = temp_df.merge(cf_temp, on=['Year', 'Month'], how='outer')

    # Apply climate modifications
    precip_df['P_adj'] = precip_df['P'] * precip_df['CF']
    temp_df['Tmax_adj'] = temp_df['Tmax'] + temp_df['CF']

    # 6. DIRECT STRUCTURAL MATRIX TRANSFORMATION (Bypassing Intermediate File Saves)
    print("Transforming vectors into final N x 312 scenario matrices...")
    GCM_list = list(precip_df['GCM'].unique())
    scenario_list = list(precip_df['Experiment'].unique())
    weather_id_list = list(precip_df['Set'].unique())

    # Map arrays via offloaded function utilities
    final_precip_mat = cc_df_to_mat(
        precip_df, GCM_list, scenario_list, weather_id_list, 'P_adj', num_months=num_months_total
    )
    final_temp_mat = cc_df_to_mat(
        temp_df, GCM_list, scenario_list, weather_id_list, 'Tmax_adj', num_months=num_months_total
    )

    # 7. EXPORT COMPRESSED DIRECT SCENARIOS
    print("Exporting finalized matrix files...")
    pd.DataFrame(data=final_precip_mat.T).to_csv(precip_output, index=False, header=False)
    pd.DataFrame(data=final_temp_mat.T).to_csv(temp_output, index=False, header=False)

    print(f"Precipitation scenario matrix saved")
    print(f"Temperature scenario matrix saved")

if __name__ == "__main__":
    main()