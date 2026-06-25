# Urban Water Single-Family Residential Demand Forecast Generator

An end-to-end multi-scenario computational pipeline designed to simulate household-level residential demand and decompose structural forecast uncertainties over multi-decade planning horizons. The architecture models intersecting demand uncertainty sources caused by climate conditions, economic trends, policy actions, population dynamics, household characteristic heterogeneity and household behavioral patterns.

---

## Repository Structure

The layout separates core module definitions (`src/`) and decoupled execution nodes (`Scripts/`) and their corresponding underlying data sources (`Data/`) and results repositories (`Results/`).

### Core Directories

| Directory Path | Layer / Classification | Primary Operational Function |
| :--- | :--- | :--- |
| `Data/Climate/`               | Storage Warehouse     | Weather related data |
| `Data/DCC_Model/`             | Storage Warehouse     | DCC model datasets |
| `Data/Economics/`             | Storage Warehouse     | Inflation data |
| `Data/Housing/`               | Storage Warehouse     | Urban scaling theory (UST) model related data |
| `Data/Population/`            | Storage Warehouse     | Growth equation of cities (GEC), population dynamics model related data |
| `Data/Simulator/Inputs/`      | Storage Warehouse     | Demand simulator input data, scenario input keys/indices, based on full factorial sampling design |
| `Data/Simulator/Scenarios/`   | Storage Warehouse     | Demand simulator input scenarios |
| `src/models/`                 | Architecture Backend  | Model training and parameter estimation for UST and GEC models |
| `src/utils/`                  | Architecture Backend  | Custom defined helper functions |
| `Scripts/Climate/`            | Processing Engine     | Climate scenarios development, SWG validation, sampling design |
| `Scripts/Conservation/`       | Processing Engine     | Conservation scenarios development |
| `Scripts/DCC_Model/`          | Processing Engine     | DCC model training using MLE |
| `Scripts/Economics/`          | Processing Engine     | Water price inflation scenarios development  |
| `Scripts/Population_Growth/`  | Processing Engine     | Population growth scenarios development using GEC and UST |
| `Scripts/Simulator/`          | Processing Engine     | Core demand simulator and pipeline post-processing and results analysis |

### Simulator Subdirectories

| Pipeline Stage Path | Operational Function |
| :--- | :--- |
| `Scripts/Simulator/Input_Grid/`             | Input scenario index grid builder |
| `Scripts/Simulator/Post_Processing/`        | Raw output matrix collection and consolidation |
| `Scripts/Simulator/Projection_Stats/`       | Demand projection statistics calcualtor |
| `Scripts/Simulator/Variance_Decomposition/` | Variance of Conditional Expectations (VCE) calculation scripts |

---

## Core System Modules (src/)

The src/ modules define the operational backend logic, estimating model parameters and formatting analytical workflows.

### 1. Estimation Models (src/models/)

#### build_GEC_model.py
* **Purpose**: Builds and trains the growth equation of cities model
* **Key Methods**:
  * `get_eta_params`: estimates steady growth random variable Guassian distribution parameters using MLE
  * `get_min_flow_model_params`: estimates parameters of minimal model for interurban migration flows using log-linear regression
  * `get_gamma`: estimates gamma parameter
  * `get_beta`: estimates beta parameter using log-log regression
  * `check_heavy_tails`: checks if migration fluctuations are heavy tailed
  * `get_zeta_params`: estiamtes migration shock random variable Levy stable distribution parameters using MLE

#### build_UST_model.py
* **Purpose**: Builds and trains the urban scaling theory model to convert population into single-family housing units
* **Key Methods**:
  * `balance_data`: Cleans and aligns panel data of population and single-family housing units cross-sectionally across all cities and time periods.
  * `fama_macbeth`: Computes Fama MacBeth estimators of urban scaling theory model parameters (model reframed using panel regression setting)
  * `compute_R2` and `get_R2_summary`: Validation metric calculation functions

### 2. Functional Utilities (src/utils/)

#### climate_functions.py
* **Purpose**: Signal processing and transformation utilities for weather time series arrays.
* **Key Methods**:
  * `compute_spi` and `calculate_spi`: Implements multi-scale Standardized Precipitation Index fits over historical rainfall data.
  * `identify_drought_events` and `get_num_time_drought`: Logical status filters to isolate drought durations, severities, and event spans.
  * `minimal_year_cover` and `minimal_year_cover_random`: Solves set-covering optimizations over variable weather data lengths to standardize historical timelines.

#### read_housing_data.py and read_migration_data.py
* **Purpose**: Automated extraction and cleaning pipelines for population, migration flow, and housing units datasets.
* **Key Methods**:
  * `get_all_flows` and `get_net_migration_dataset`: Collects and transforms raw disjointed spreadsheets from the US Census Bureau into unified matrices.
  * `get_avg_neighbors`: Measures regional demographic interaction changes by mapping network geographic adjacencies.

#### simulator_functions.py
* **Purpose**: Optimized calculation engines and helper functions built for high-throughput runtime execution loops.
* **Key Methods**:
  * `make_global_id` and `make_global_id_HH`: Combines spatial vector mappings into unique index hashes across block groups and households to keep random number generator streams indpendent.
  * `get_tier_prices`: Models increasing block-rate utility tariff structures stochastically using continuous distributions.
  * `weighted_percentiles`: Performs memory-safe, single-pass sorted quantile interpolations over massive output projection dimensions.
  
  ---

## Execution Nodes (Scripts/)

The pipeline follows a sequential execution chain where upstream scenarios feed the main simulator engine, which in turn outputs arrays for post-processing and uncertainty quantification and variance decomposition.

```text
[Climate/Economics/Conservation/DCC Model Parameter/Population Scripts] ---> [Input_Grid Builders] ---> [Simulator Engine] ---> [Post-Processing/Projection Stats/Variance Decomposition]
```

### Phase 1: Upstream Driver Generation
These independent scripts evaluate and write background boundary parameters out to the data layer.

#### Climate/
* `save_climate_change_CFs.py`: Extracts and processes baseline delta-change factor anomalies.
* `cluster_weather_projections.py`: Maps spatial weather variables via multivariate clustering.
* `sample_drought_scenarios.py`: selective sampling of stationary weather time series to include scenarios with droughts
* `save_spi_parameters.py`: Saves SPI-12 parameters
* `generate_full_weather_ensemble.py`: generates final input climate conditions scenarios assessing uncertainty in weather variability, drought occurrence, and climate change

#### Conservation/
* `generate_conservation_scenarios.py`: Models utility conservation policy intervention parameters using nonlinear logistic diffusion functions (logistic_function).

#### DCC_Model/
* `DCC_LL_Function.R` and `DCC_Model_MLE_Main.R`: Formulates maximum lg-likelihood estimations via R to train the DCC model.
* `generate_DCC_parameter_scenarios.py`: Samples DCC model parameter scenarios using importance sampling on principal components.

#### Economics/
* `generate_econ_scenarios.py`: Generates continuous water price inflation paths using mean-reverting Ornstein-Uhlenbeck processes (generate_ou_process).

#### Population_Growth/
* `GEC/get_population_projections.py`: generates city-level population projections using GEC model
  * to build GEC model: `GEC/save_migration_data.py` --> `GEC/save_model_parameters.py`
* `UST/get_housing_projections.py`: converts population projections into housing unit projections using UST model
  * to build UST model: `UST/save_migration_data.py` --> `UST/save_model_parameters.py`
* `cluster_housing_projections.py`: Applies k-medoid clustering to select representative population growth scenarios from full GEC-UST ensemble

### Phase 2: Runtime Coordinate Compilation
Before executing the simulator, target data arrays must be vectorized and mapped to strict index arrays.

#### Simulator/Input_Grid/
* **Scripts**: `generate_city_input_ID_grid.py`, `generate_BG_input_ID_grid.py`, `generate_HH_input_grid.py`, and `build_total_city_input_grid.py`.
* **Purpose**: Converts raw input scenario index matrices and shape coordinates into memory-efficient spatial index arrays (id_mat.npy). It breaks massive scenario grids down into localized, process-safe data chunks (process_input_id_grid) to enable scalable parallel loops.

### Phase 3: Demand Simulation Execution
This is the core engine layer that simulates multi-scenario and multi-scale demand projections.

#### Simulator/
* **Scripts**: `run_simulator_city.py` (city level projections); `run_simulator_BG.py` (block-group level projections); `run_simulator_HH.py` (household-level projections)
* **Purpose**: Generates demand projections across multiple scenario uncertainty sources, in a full factorial sampling design scheme
* **Computational Layout**: Operates over a nested loop framework (generate_demand_projection) to process data chunks sequentially using memory-mapped reading arrays to enforce strict memory safety. Should ideally be run parallelly (since each projection is independent) using a high-performance computing cluster - running chunks of projections at a time, to be post-processed later.

### Phase 4: Output Post-Processing and Analytics
These scripts consolidate the output projection chunks and executes statistical analysis operations.

#### Simulator/Post_Processing/
* **Scripts**: `post_process_city.py`, `post_process_BG.py`, and `post_process_HH.py`.
* **Purpose**: Gathers scattered runtime projection chunk files, applies the passive conservation scenarios, and merges them back into contiguous, chronological block arrays (process_chunk).

#### Simulator/Projection_Stats/
* **Scripts**: `proj_stats_city.py`, `proj_stats_BG.py`, and `proj_stats_HH.py`.
* **Purpose**: Calculates baseline statistical distributions. Generates single-pass weighted means, standard deviations, and smooth percentile cuts across city, block-group, and household datasets.

#### Simulator/Variance_Decomposition/
* **Scripts**: `var_decomp_city.py`, `var_decomp_BG.py`, and `var_decomp_HH.py`.
* **Purpose**: Analyzes forecast uncertainty profiles using a Variance of Conditional Expectations (VCE) approach. It isolates how much forecast variance is driven by specific input scenario parameters versus within projection noise attributed to housing characteristics heterogeneity over long horizons.

---

## Deployment and Operational Guidelines

### Standalone Local Execution
Every script is decoupled and can be run independently directly from the repository root:

# Example for City wide projections

# 1. Compile input scenario grids
python Scripts/Simulator/Input_Grid/generate_city_input_ID_grid.py

# 2. Execute localized demand simulation
python Scripts/Simulator/run_simulator_city.py

# 3. Post-process demand projections
python Scripts/Simulator/Post_Processing/post_process_city.py

# 4. Compute projections statistics and perform variance decomposition
python Scripts/Simulator/Projection_Stats/proj_stats_city.py
python Scripts/Simulator/Variance_Decomposition/var_decomp_city.py

### High-Performance Cluster (SLURM) Scheduling
Because the analytical scripts are self-contained and leverage relative paths via pathlib, you can submit heavy operations directly to separate cluster partitions without complex environment overhead.

Example for city-level projections generation script:

```bash
#!/usr/bin/bash
#SBATCH --job-name=dem_sim_city_all
#SBATCH --time=12:00:00
#SBATCH -p serc
#SBATCH --array=0-199
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=500MB
#SBATCH --hint=nomultithread
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aniketv@stanford.edu
cd $SCRATCH
ml load python/3.9.0
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "SLURM job ID: $SLURM_JOB_ID"
echo "Array ID: $SLURM_ARRAY_TASK_ID"
python3 -u Scripts/Simulator/run_simulator_city.py
```

## Pipeline Outputs

Results are saved automatically to designated directory:

### Projection Statistics
* **Path**: `Data/Simulator/Outputs/Projection_Statistics/`
* **Contents**: Tabular summary datasets tracking distribution baselines over the projection timeline. Includes calculated metrics for:
  * Arithmetic Mean
  * Standard Deviation
  * Custom Percentile cuts ranging from P0.5 up to P99.5

### Variance Decomposition
* **Path**: `Data/Simulator/Outputs/Variance_Decomposition/`
* **Contents**: Matrix outputs containing calculated Variance of Conditional Expectations (VCE) ratios. 
* **Structure**: Cleanly structured chronological tables equipped with explicit, running Year and Month index tags mapping structural uncertainty shifts from 2025 through 2050.
