# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# 1. LINK CORE FOLDERS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# 2. IMPORT CUSTOM UTILITY ENGINE
from src.utils import climate_functions

def main():
    # 3. ROUTE DIRECTORIES RELATIVE TO REPO ROOT
    climate_data_dir = REPO_ROOT / "Data" / "Climate"
    simulator_data_dir = REPO_ROOT / "Data" / "Simulator"
    
    # Input files
    precip_input = climate_data_dir / "precip_mat_final.csv"
    temp_input = climate_data_dir / "temp_mat_final.csv"
    weather_data_file = simulator_data_dir / "Inputs" / "weather_data.csv"
    spi_params_file = simulator_data_dir / "Inputs" / "spi_params.csv"
    
    # Output destinations  
    temp_output_file = simulator_data_dir / "Scenarios" / "temperature_scenarios.csv"
    precip_output_file = simulator_data_dir / "Scenarios" / "precipitation_scenarios.csv"

    # 4. INITIALIZE RANDOM NUMBER GENERATOR AND CONFIGURATIONS
    rand_seed = 160596
    np.random.seed(rand_seed)
    rng_base = np.random.default_rng(rand_seed)
    N = 50  # Target number of samples

    print("Loading final weather scenario matrices and indexing data...")
    precip_mat = np.transpose(np.loadtxt(precip_input, delimiter=","))
    temp_mat = np.transpose(np.loadtxt(temp_input, delimiter=","))
    spi_params = pd.read_csv(spi_params_file, usecols=['a', 'scale', 'q']).to_numpy()
    precip_2024 = pd.read_csv(weather_data_file, usecols=['precip']).to_numpy()[-12:].reshape(-1)

    # Define dimensions
    num_years = precip_mat.shape[1] // 12
    num_samples = precip_mat.shape[0] // (12 * 3)  # 12 GCMs across 3 SSP scenarios

    months = np.arange(precip_mat.shape[1]) % 12 + 1
    years = np.repeat(np.arange(1, num_years + 1), 12)

    # 5. COMPUTE TEMPERATURE METRICS
    print("Extracting temperature feature vectors...")
    monthly_means = np.array([np.mean(temp_mat[:, months == m], axis=1) for m in range(1, 13)])
    annual_means = np.array([np.mean(temp_mat[:, years == y], axis=1) for y in range(1, num_years + 1)])
    temp_change = np.diff(annual_means, axis=0)

    T_mean = np.mean(temp_mat, axis=1)
    T_std = np.std(temp_mat, axis=1)
    T_seasonal_amp = np.max(monthly_means, axis=0) - np.min(monthly_means, axis=0)
    T_avg_delta = np.mean(temp_change, axis=0)

    # 6. COMPUTE PRECIPITATION METRICS
    print("Extracting precipitation feature vectors...")
    annual_P = np.array([np.sum(precip_mat[:, years == y], axis=1) for y in range(1, num_years + 1)])
    avg_q10 = np.mean(np.percentile(precip_mat, 10, axis=1))  # Dry month threshold

    P_ann_mean = np.mean(annual_P, axis=0)
    P_frac_dry_months = np.mean(precip_mat < avg_q10, axis=1)
    P_seasonal_ratio = np.max(monthly_means, axis=0) / np.mean(monthly_means, axis=0)

    # 7. COMPUTE JOINT / CLIMATE STRESS FEATURES
    print("Analyzing joint stress indices and hot-dry frequencies...")
    hot_p, dry_p = 80, 20
    avg_hot_T = np.mean(np.percentile(temp_mat, hot_p, axis=1))
    avg_dry_P = np.mean(np.percentile(precip_mat, dry_p, axis=1))
    hot_dry = (temp_mat >= avg_hot_T) & (precip_mat <= avg_dry_P)
    
    sum_mask = np.isin(months, [6, 7, 8])
    sum_hot_T = np.mean(np.percentile(temp_mat[:, sum_mask], hot_p, axis=1))
    sum_dry_P = np.mean(np.percentile(precip_mat[:, sum_mask], dry_p, axis=1))
    sum_hot_dry = (temp_mat[:, sum_mask] >= sum_hot_T) & (precip_mat[:, sum_mask] <= sum_dry_P)
    
    sum_hot_dry_freq = np.mean(sum_hot_dry, axis=1)

    # High-T Low-P stress evaluation
    P_mean = np.mean(precip_mat, axis=1)
    P_std = np.std(precip_mat, axis=1)
    
    # Broadcast adjustments for scaling
    P_norm = (precip_mat - np.repeat(P_mean, 312).reshape(num_samples * 12 * 3, 312)) / np.repeat(P_std, 312).reshape(num_samples * 12 * 3, 312)
    T_norm = (temp_mat - np.repeat(T_mean, 312).reshape(num_samples * 12 * 3, 312)) / np.repeat(T_std, 312).reshape(num_samples * 12 * 3, 312)
    stress = np.std(T_norm - P_norm, axis=1)

    # Drought timeline calculation
    drought_clip = 0      
    drought_spi = -1     
    min_duration = 6     

    num_droughts = np.zeros(precip_mat.shape[0])
    year_droughts = []

    for i in range(precip_mat.shape[0]):
        # Route logic via custom offloaded engine
        spi = climate_functions.calculate_spi(np.concatenate((precip_2024, precip_mat[i])), spi_params)
        drought = climate_functions.identify_drought_events(spi, drought_clip, drought_spi, min_duration)
        num_d, time_d = climate_functions.get_num_time_drought(drought)
        
        num_droughts[i] = num_d
        if num_d == 0:
            year_droughts.append([])
        else:
            year_droughts.append(list(years[time_d]))

    # Stack computed components into a final unified feature matrix
    feature_mat = np.vstack([
        T_mean, T_std, T_seasonal_amp, T_avg_delta, 
        P_ann_mean, P_frac_dry_months, P_seasonal_ratio, 
        sum_hot_dry_freq, stress, num_droughts
    ]).T

    # 8. EXECUTE TAIL COVERS AND SAMPLING PIPELINES
    print("Assembling greedy year covers and percentile tail metrics...")
    sample_1 = climate_functions.minimal_year_cover_random(year_droughts, penalty_power=0.95, rng=rng_base)
    
    # Placeholders for extreme thresholds
    sample_2 = np.random.choice(a=np.where(T_mean >= np.percentile(T_mean, 97.5))[0], size=0)
    sample_3 = np.random.choice(a=np.where(T_mean <= np.percentile(T_mean, 2.5))[0], size=0)
    sample_4 = np.random.choice(a=np.where(P_ann_mean >= np.percentile(P_ann_mean, 97.5))[0], size=0)
    sample_5 = np.random.choice(a=np.where(P_ann_mean <= np.percentile(P_ann_mean, 2.5))[0], size=0)

    # Combine custom tracking indices 
    custom_ids = np.concatenate((sample_1, sample_2, sample_3, sample_4, sample_5)).astype(int)
    custom_ids = np.unique(custom_ids)

    # Recalculate dynamic cluster counts
    N = N - len(custom_ids)
    print(f"Remaining active K-Medoids cluster allocations required: {N}")

    # Prune data structures for unsupervised clustering phase
    feature_mat_trunc = np.delete(feature_mat, custom_ids, axis=0)
    precip_mat_trunc = np.delete(precip_mat, custom_ids, axis=0)
    temp_mat_trunc = np.delete(temp_mat, custom_ids, axis=0)

    # 9. PERFORM K-MEDOIDS CLUSTERING
    print("Normalizing trimmed feature parameters...")
    F_std = StandardScaler().fit_transform(feature_mat_trunc)

    print("Evaluating euclidean distances across cluster nodes...")
    kmed = KMedoids(n_clusters=N, metric="euclidean")
    kmed.fit(F_std)
    selected_indices = kmed.medoid_indices_

    # 10. RECOMPILE MATRIX BLOCKS AND EXPORT SAMPLES
    print("Compiling final consolidated sample targets...")
    precip_custom = precip_mat[custom_ids]
    temp_custom = temp_mat[custom_ids]
    
    precip_kmed = precip_mat_trunc[selected_indices]
    temp_kmed = temp_mat_trunc[selected_indices]
    
    precip_sample_df = pd.DataFrame(data=np.concatenate((precip_custom, precip_kmed)).T)
    temp_sample_df = pd.DataFrame(data=np.concatenate((temp_custom, temp_kmed)).T)

    # Write files back out safely to target relative data positions
    temp_sample_df.to_csv(temp_output_file, index=False, header=False)
    precip_sample_df.to_csv(precip_output_file, index=False, header=False)

    print(f"K-Medoid clustered temperature samples saved.")
    print(f"K-Medoid clustered precipitation samples saved.")

if __name__ == "__main__":
    main()