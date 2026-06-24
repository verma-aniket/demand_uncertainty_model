# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

def main():
    # route directors
    projections_dir = REPO_ROOT / "Data" / "Housing" / "Projections"
    housing_data_dir = REPO_ROOT / "Data" / "Simulator" / "Inputs"
    output_dir = REPO_ROOT / "Data" / "Simulator" / "Scenarios"

    # define model configurations
    rand_seed = 271126  # for reproducibility
    np.random.seed(rand_seed)
    N = 50  # number of samples

    # load projections data
    input_file = projections_dir / "HH_projs.csv"
    HH_proj_df = pd.read_csv(input_file)

    # Extract projections from 2025 to 2050
    HH_proj_df = HH_proj_df[(HH_proj_df['Year'] >= 2025) & (HH_proj_df['Year'] <= 2050)]
    hh_matrix = HH_proj_df[list(HH_proj_df.columns[0:1000])].to_numpy().T

    # develop feature vector for all projections
    # Feature 1 & 2: Initial and Final Housing Units
    hh_ini = hh_matrix[:, 0]
    hh_fin = hh_matrix[:, -1]

    # Compute growth rates for all projections (g(t) = (P(t+1) - P(t)) / P(t))
    hh_gr = np.diff(hh_matrix) / hh_matrix[:, 0:-1]

    # Feature 3: Average Growth Rate 
    avg_gr = np.mean(hh_gr, axis=1)

    # Feature 4 & 5: Max and Min Growth Rate
    max_gr = np.max(hh_gr, axis=1)
    min_gr = np.min(hh_gr, axis=1)

    # Compute growth rate acceleration
    hh_acc = np.diff(hh_gr)

    # Feature 6 & 7: Peak Absolute Acceleration & Timing
    max_abs_acc = np.max(np.abs(hh_acc), axis=1)
    max_abs_acc_time = np.argmax(np.abs(hh_acc), axis=1)

    # Feature 8: Volatility (standard deviation of growth rates)
    gr_std = np.std(hh_gr, axis=1)

    # Develop feature matrix
    feature_mat = np.vstack([hh_ini, hh_fin, avg_gr, max_gr, min_gr, max_abs_acc, max_abs_acc_time, gr_std]).T

    # tail sampling and custom illustrative samples
    custom_ids = np.array([42, 37, 909, 292])

    # Sample from the upper/lower tails (Window 1)
    num_sam = 1
    window_1 = 1
    sample_high_pop = np.random.choice(a=np.where(hh_fin >= np.percentile(hh_fin, 100-window_1))[0], size=num_sam)[0]
    sample_low_pop = np.random.choice(a=np.where(hh_fin <= np.percentile(hh_fin, window_1))[0], size=num_sam)[0]
    sample_high_gr = np.random.choice(a=np.where(avg_gr >= np.percentile(avg_gr, 100-window_1))[0], size=num_sam)[0]
    sample_low_gr = np.random.choice(a=np.where(avg_gr <= np.percentile(avg_gr, window_1))[0], size=num_sam)[0]
    sample_high_acc = np.random.choice(a=np.where(max_abs_acc >= np.percentile(max_abs_acc, 100-window_1))[0], size=num_sam)[0]
    sample_low_acc = np.random.choice(a=np.where(max_abs_acc <= np.percentile(max_abs_acc, window_1))[0], size=num_sam)[0]

    # Combine custom sampled projections
    custom_ids = np.concatenate((custom_ids, np.array([sample_high_pop, sample_low_pop, 
                                                      sample_high_gr, sample_low_gr, 
                                                      sample_high_acc, sample_low_acc])))
    custom_ids = np.unique(custom_ids)

    # Sample from the upper/lower tails (Window 25)
    window_25 = 25
    sample_high_pop = np.random.choice(a=np.where(hh_fin >= np.percentile(hh_fin, 100-window_25))[0], size=num_sam)[0]
    sample_low_pop = np.random.choice(a=np.where(hh_fin <= np.percentile(hh_fin, window_25))[0], size=num_sam)[0]
    sample_high_gr = np.random.choice(a=np.where(avg_gr >= np.percentile(avg_gr, 100-window_25))[0], size=num_sam)[0]
    sample_low_gr = np.random.choice(a=np.where(avg_gr <= np.percentile(avg_gr, window_25))[0], size=num_sam)[0]
    sample_high_acc = np.random.choice(a=np.where(max_abs_acc >= np.percentile(max_abs_acc, 100-window_25))[0], size=num_sam)[0]
    sample_low_acc = np.random.choice(a=np.where(max_abs_acc <= np.percentile(max_abs_acc, window_25))[0], size=num_sam)[0]

    # Combine all custom sampled projections
    custom_ids = np.concatenate((custom_ids, np.array([sample_high_pop, sample_low_pop, 
                                                      sample_high_gr, sample_low_gr, 
                                                      sample_high_acc, sample_low_acc])))
    custom_ids = np.unique(custom_ids)

    # Update total required cluster counts dynamically
    N = N - len(custom_ids)

    # Remove custom sampled projections from feature matrix and training space
    feature_mat = np.delete(feature_mat, custom_ids, axis=0)
    hh_matrix_trunc = np.delete(hh_matrix, custom_ids, axis=0)

    # apply k-medoid clustering
    # Step 1: Standardize features so no single feature dominates
    F_std = StandardScaler().fit_transform(feature_mat)

    # Step 2: Perform k-medoids clustering to select representative samples
    kmed = KMedoids(n_clusters=N, metric="euclidean")
    kmed.fit(F_std)
    selected_indices = kmed.medoid_indices_

    # extract selected samples based on ID
    custom_samples = np.zeros((len(custom_ids), hh_matrix.shape[1]))
    for s in range(len(custom_ids)):
        custom_samples[s] = hh_matrix[custom_ids[s]]
    k_med_samples = hh_matrix_trunc[selected_indices]
    hh_samples = pd.DataFrame(data=np.concat((custom_samples, k_med_samples)).T).to_numpy().T

    # Apply bias correction factor based on housing units at the start of the simulation horizon

    # Read initial housing stock data
    hh_pool_ini = pd.read_csv(housing_data_dir / "housing_pool_ini.csv")

    # Adjust number of households to be relative to 2024 housing stock
    num_ini_hhs = len(hh_pool_ini)
    ini_mean = np.mean(hh_samples[:,0]).round(0)
    correction = num_ini_hhs - ini_mean
    hh_samples = hh_samples + correction # correction = 4016

    # Set Initial Number of Housing Units
    hh_2024 = np.full((1, hh_samples.shape[0]), num_ini_hhs)
    hh_samples = np.vstack((hh_2024, hh_samples.T)).T

    # Define Scenarios as the annual change in number of housing units
    pop_scen = np.diff(hh_samples, axis = 1)
    pop_scen = pop_scen.astype(int)

    # Save scenarios
    pop_scen_df = pd.DataFrame(data=pop_scen.T)
    pop_scen_df.to_csv(output_dir / "population_scenarios.csv", index=False, header=False)
    print("Population growth uncertainty scenarios complete.")

if __name__ == "__main__":
    main()