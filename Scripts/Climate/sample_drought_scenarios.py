# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.utils import climate_functions

def main():
    # route directors
    climate_data_dir = REPO_ROOT / "Data" / "Climate"
    simulator_data_dir = REPO_ROOT / "Data" / "Simulator"
    
    # Target input/output file structural mappings
    swg_ts_file = climate_data_dir / "SWG_results.csv"
    spi_params_file = simulator_data_dir / "Inputs" / "spi_params.csv"
    weather_data_file = simulator_data_dir / "Inputs" / "weather_data.csv"
    output_file = climate_data_dir / "drought_IDs.csv"

    # set random seed for reproducibility
    base_seed = 271124
    rng_base = np.random.default_rng(seed=base_seed)

    start_year = 2025
    stop_year = 2050
    num_years = stop_year - start_year + 1

    # load input data
    print("Loading stochastic weather timelines and index arrays...")
    precip_mat = np.loadtxt(swg_ts_file, delimiter=",", usecols=[0])                                # stationary time series from WeaGETS
    spi_params = pd.read_csv(spi_params_file, usecols=['a', 'scale', 'q']).to_numpy()               # SPI parameters
    precip_2024 = pd.read_csv(weather_data_file, usecols=['precip']).to_numpy()[-12:].reshape(-1)   # historic precip. data

    # reshape precip data into matrix
    extra_years = (precip_mat.shape[0] // 365) % num_years
    if extra_years > 0:
        precip_mat = precip_mat[:-extra_years * 365]
    num_sets = (precip_mat.shape[0] // 365) // num_years
    precip_mat = precip_mat.reshape(num_sets, num_years, 365)

    # Build calendar mapping bounds (assuming standard non-leap year length blocks)
    month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    day_to_month = np.repeat(np.arange(0, 12), month_lengths)

    # Aggregate to monthly totals
    precip_mon = np.zeros((precip_mat.shape[0], num_years, 12))
    for m in range(12):
        precip_mon[:, :, m] = precip_mat[:, :, day_to_month == m].sum(axis=2)
    precip_mon = precip_mon.reshape(precip_mat.shape[0], num_years * 12)

    # parse SPI-based drought identification variables (based on run theory)
    drought_clip = 0      # check for negative SPI
    drought_spi = -1      # drought event must achive a minimum of at least -1
    min_duration = 6      # SPI must be negative for 6 consecutive months

    print("Evaluating regional drought patterns across generated timelines...")
    num_droughts = np.zeros(precip_mon.shape[0])
    years = np.repeat(np.arange(2025, 2051, 1), 12)
    year_droughts = []

    for i in range(precip_mon.shape[0]):
        # Call offloaded functions via the custom imported climate utils package
        spi = climate_functions.calculate_spi(np.concatenate((precip_2024, precip_mon[i])), spi_params)
        drought = climate_functions.identify_drought_events(spi, drought_clip, drought_spi, min_duration)
        num_d, time_d = climate_functions.get_num_time_drought(drought)
        
        num_droughts[i] = num_d
        if num_d == 0:
            year_droughts.append([])
        else:
            year_droughts.append(list(years[time_d]))

    # Identify sets satisfying complete coverage rules
    drought_sets = climate_functions.minimal_year_cover(year_droughts, penalty_power=1)

    # EXVERT RESAMPLING STATISTICAL MODIFIERS
    drought_occ_pop = sum(num_droughts > 0) / num_droughts.shape[0]

    # Handle sampling ratios safely 
    subtotal_samples = np.int64(np.round(len(drought_sets) / drought_occ_pop, 0))
    non_drought_samples = subtotal_samples - len(drought_sets)

    # Randomly sample non-drought indices
    non_drought_sets = rng_base.choice(
        a=np.where(num_droughts == 0)[0], 
        size=non_drought_samples, 
        replace=False
    ).tolist()

    # Expand selection matrix up to final size target
    total_samples = 30
    add_samples = total_samples - subtotal_samples
    all_indices = np.arange(precip_mon.shape[0])
    
    remaining_indices = np.array([
        idx for idx in all_indices if ((idx not in drought_sets) & (idx not in non_drought_sets))
    ])
    random_sets = rng_base.choice(a=remaining_indices, size=add_samples, replace=False).tolist()

    # OUTPUT VALIDATION STATISTICS
    sample_set = drought_sets + non_drought_sets + random_sets
    sample_droughts = num_droughts[sample_set]
    
    print("\nDrought Comparison Summary:")
    print(f"{'Metric':<35}{'Full Set':>12}{'Sampled Set':>15}")
    print("-" * 62)
    print(f"{'Fraction with >=1 drought (%)':<35}{100 * (sum(num_droughts > 0) / num_droughts.shape[0]):>12.1f}{100 * (sum(sample_droughts > 0) / sample_droughts.shape[0]):>15.1f}")
    print(f"{'Average number of droughts':<35}{np.mean(num_droughts[num_droughts > 0]):>12.3f}{np.mean(sample_droughts[sample_droughts > 0]):>15.3f}")

    # SAVE FINAL REPRESENTATIVE SCENARIO SAMPLE PATH KEYS
    np.savetxt(fname=output_file, X=sample_set, delimiter=",", fmt="%d")
    print("Representative drought scenario sequence IDs saved.")

if __name__ == "__main__":
    main()