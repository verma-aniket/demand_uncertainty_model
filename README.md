# Code Repository for: **"Quantifying and partitioning uncertainty in multi-scale urban water demand projections"**  
> *Authors:* Aniket Verma, Jennifer Skerker, Christian Klassert, Christa Brelsford, Sarah Fletcher
> *Code and Data Repository:* https://doi.org/10.5281/zenodo.22820914

---

## Urban Water Single-Family Residential Demand Forecast Generator

An end-to-end multi-scenario computational pipeline designed to simulate household-level residential single-family demand and decompose forecast uncertainties over multi-decade planning horizons. Specifically, the demand forecast generator models, quantifies, and partitions demand uncertainty attributed to climate conditions, economic trends, conservation policy actions, population dynamics, housing characteristic heterogeneity, and household behavioral patterns.

---

## Repository Structure

The layout separates core module definitions (`src/`) and decoupled execution nodes (`Scripts/`) and their corresponding underlying data sources (`Data/`) and results repositories (`Results/`).

### Core Directories

| Path | Classification | Notes |
| :--- | :--- | :--- |
| `Data/Climate/`                   | Data Storage            | Weather related data |
| `Data/DCC_Model/`                 | Data Storage            | Discrete Continuous Choice (DCC) model training data and results *DCC_DB.db and dccfit_results.rds not included in data repository due to Data Use Agreement (DUA)*|
| `Data/Economics/`                 | Data Storage            | Inflation data |
| `Data/Housing/`                   | Data Storage            | Urban scaling theory (UST) model related data |
| `Data/Population/`                | Data Storage            | Growth equation of cities (GEC), population dynamics model related data |
| `Data/Simulator/Inputs/`          | Data Storage            | Demand simulator input data, scenario input keys/indices, based on full factorial sampling design |
| `Data/Simulator/Scenarios/`       | Data Storage            | Demand simulator input scenarios *Sensitive data has been replaced with synthetic/publicly available data to comply with DUA* |
| `src/models/`                     | Model Training Backend  | Model training and parameter estimation for UST and GEC models |
| `src/utils/`                      | Model Training Backend  | Custom defined helper functions |
| `Scripts/Climate/`                | Processing Engine       | Climate scenarios development, stochastic weather generator (SWG) validation, sampling design |
| `Scripts/Conservation/`           | Processing Engine       | Conservation scenarios development |
| `Scripts/DCC_Model/`              | Processing Engine       | DCC model training using Maximum Likelihood Estimation (MLE) |
| `Scripts/Economics/`              | Processing Engine       | Water price inflation scenarios development  |
| `Scripts/Population_Growth/`      | Processing Engine       | Population growth scenarios development using GEC and UST |
| `Scripts/Simulator/`              | Processing Engine       | Core demand simulator and pipeline post-processing and results analysis |
| `Results/Projections/`            | Results Storage         | Monthly and annual demand projections (chunked and merged block arrays) organized by spatial scale (city, block-group, household) |
| `Results/Components/`             | Results Storage         | Individual log-demand component matrices organized by spatial scale (city, block-group, household) |
| `Results/Projection_Statistics/`  | Results Storage         | City-scale demand projection summary statistics (i.e., mean, SD, percentiles) |
| `Results/Variance_Decomposition/` | Results Storage         | Calculated Variance of Conditional Expectations (VCE) ratio matrices organized by spatial scale (city, block-group, household) |

### Important Note on Utility Billing Data Privacy & Use of Synthetic Data

To comply with data privacy regulations and protect sensitive water utility billing data, data has been replaced with publicly available and/or synthetic dummy data:
* **Pipe Sizes:** All nominal pipe diameters have been set to `0`.
* **Water Rates:** Rates are set to 2021 values derived from the Case Study Area's (City of Santa Cruz) 2020 Urban Water Management Plan (refer to references in the main manuscript).

These substitutions allow full execution of the demand uncertainty characterization model pipeline without compromising proprietary utility data.

### Simulator Subdirectories

| Path | Function |
| :--- | :--- |
| `Scripts/Simulator/Input_Grid/`             | Input scenario index grid builder |
| `Scripts/Simulator/Post_Processing/`        | Raw output matrix collection and consolidation |
| `Scripts/Simulator/Projection_Stats/`       | Demand projection statistics calculator |
| `Scripts/Simulator/Variance_Decomposition/` | Variance of Conditional Expectations (VCE) calculation scripts |

---

## Core System Modules (src/)

The src/ modules define the operational backend logic, estimating model parameters and formatting analytical workflows.

### 1. Estimation Models (src/models/)

#### build_GEC_model.py
* **Purpose**: Builds and trains the growth equation of cities model
* **Key Methods**:
  * `get_eta_params`: estimates steady growth random variable Gaussian distribution parameters using MLE
  * `get_min_flow_model_params`: estimates parameters of minimal model for interurban migration flows using log-linear regression
  * `get_gamma`: estimates gamma parameter
  * `get_beta`: estimates beta parameter using log-log regression
  * `check_heavy_tails`: checks if migration fluctuations are heavy tailed
  * `get_zeta_params`: estimates migration shock random variable Levy stable distribution parameters using MLE

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
  * `make_global_id` and `make_global_id_HH`: Combines scenario input index vector mappings into unique index hashes to keep random number generator streams independent.
  * `get_tier_prices`: Models increasing block rate (IBR) tier structures stochastically using continuous distributions.
  * `weighted_percentiles`: Performs memory-safe, single-pass sorted quantile interpolations over massive output projection dimensions.
  
  ---

## Execution Nodes (Scripts/)

The pipeline follows a sequential execution chain where upstream scenarios feed the main simulator engine, which in turn outputs arrays for post-processing, uncertainty quantification, and variance decomposition.

```text
[Climate/Economics/Conservation/DCC Model Parameter/Population Scripts] ---> [Input_Grid Builders] ---> [Simulator Engine] ---> [Post-Processing/Projection Stats/Variance Decomposition]
```

### Phase 1: Upstream Driver Generation
These independent scripts evaluate and write background boundary parameters and/or input scenarios out to the data layer.

#### Climate/
* `save_climate_change_CFs.py`: Extracts and processes baseline delta-change factor anomalies.
* `cluster_weather_projections.py`: Maps spatial weather variables via multivariate clustering.
* `sample_drought_scenarios.py`: selective sampling of stationary weather time series to include scenarios with droughts
* `save_spi_parameters.py`: Saves SPI-12 parameters
* `generate_full_weather_ensemble.py`: generates final input climate conditions scenarios assessing uncertainty in weather variability, drought occurrence, and climate change

#### Conservation/
* `generate_conservation_scenarios.py`: Models utility conservation policy intervention parameters using nonlinear logistic curve-fitting functions (logistic_function).

#### DCC_Model/
* `DCC_LL_Function.R` and `DCC_Model_MLE_Main.R`: Formulates maximum log-likelihood estimations via R to train the DCC model.
* `generate_DCC_parameter_scenarios.py`: Samples DCC model parameter scenarios using importance sampling on principal components.

#### Economics/
* `generate_econ_scenarios.py`: Generates continuous water price inflation paths using mean-reverting Ornstein-Uhlenbeck processes (generate_ou_process).

#### Population_Growth/
* `GEC/get_population_projections.py`: generates city-level population projections using GEC model
  * To build GEC model: `GEC/save_migration_data.py` --> `GEC/save_model_parameters.py`
* `UST/get_housing_projections.py`: converts population projections into housing unit projections using UST model
  * To build UST model: `UST/save_migration_data.py` --> `UST/save_model_parameters.py`
* `cluster_housing_projections.py`: Applies k-medoid clustering to select representative population growth scenarios from full GEC-UST ensemble

### Phase 2: Runtime Coordinate Compilation
Before executing the simulator, target data arrays must be vectorized and mapped to strict index arrays.

#### Simulator/Input_Grid/
* **Scripts**: `generate_city_input_ID_grid.py`, `generate_BG_input_ID_grid.py`, `generate_HH_input_grid.py`, and `build_total_city_input_grid.py`.
* **Purpose**: Converts raw input scenario index matrices and shape coordinates into memory-efficient spatial index arrays (id_mat.npy). It breaks massive scenario grids down into localized, process-safe data chunks (process_input_id_grid) to enable scalable parallel loops.

### Phase 3: Demand Simulation Execution
This is the core engine layer that simulates multi-scenario and multi-scale demand projections.

#### Simulator/
* **Scripts**: `run_simulator_city.py` (city level projections); `run_simulator_BG.py` (block group level projections); `run_simulator_HH.py` (household-level projections)
* **Purpose**: Generates demand projections across multiple scenario uncertainty sources, in a full factorial sampling design scheme
* **Computational Layout**: Operates over a nested loop framework (generate_demand_projection) to process data chunks sequentially using memory-mapped reading arrays to enforce strict memory safety. Should ideally be run parallelly (since each projection is independent) using a high-performance computing cluster - running chunks of projections at a time, to be post-processed later.

### Phase 4: Output Post-Processing and Analytics
These scripts consolidate the output projection chunks and executes statistical analysis operations.

#### Simulator/Post_Processing/
* **Scripts**: `post_process_city.py`, `post_process_BG.py`, and `post_process_HH.py`.
* **Purpose**: Gathers scattered runtime projection chunk files, applies the passive conservation scenarios, and merges them back into contiguous, chronological block arrays (process_chunk).

#### Simulator/Projection_Stats/
* **Scripts**: `proj_stats_city.py`, `proj_stats_BG.py`, and `proj_stats_HH.py`.
* **Purpose**: Calculates baseline statistical distributions. Generates single-pass weighted means, standard deviations, and smooth percentile cuts across city, block group, and household datasets.

#### Simulator/Variance_Decomposition/
* **Scripts**: `var_decomp_city.py`, `var_decomp_BG.py`, and `var_decomp_HH.py`.
* **Purpose**: Analyzes forecast uncertainty profiles using a grouped Variance of Conditional Expectations (VCE) approach. It isolates how much forecast variance is driven by specific input scenario parameters versus within projection noise attributed to housing characteristics heterogeneity over long horizons.

---

## Deployment and Operational Guidelines

### Standalone Local Execution
Every script is decoupled and can be run independently directly from the repository root:

### Example for City wide projections

#### 1. Compile input scenario grids
python Scripts/Simulator/Input_Grid/generate_city_input_ID_grid.py

#### 2. Execute localized demand simulation
python Scripts/Simulator/run_simulator_city.py

#### 3. Post-process demand projections
python Scripts/Simulator/Post_Processing/post_process_city.py

#### 4. Compute projections statistics and perform variance decomposition
python Scripts/Simulator/Projection_Stats/proj_stats_city.py
python Scripts/Simulator/Variance_Decomposition/var_decomp_city.py

### High-Performance Cluster (SLURM) Scheduling
Because the analytical scripts are self-contained and leverage relative paths via pathlib, heavy operations can be submitted directly to separate cluster partitions without complex environment overhead.

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

Results are saved automatically to designated directories:

### Projections
* **Path**: `Results/Projections/`
* **Contents**: single-family urban water monthly and annual demand projections (chunks and merged contiguous, chronological block arrays) organized by spatial scale (city, block group, and household)

### Components
* **Path**: `Results/Components/`
* **Contents**: individual log-demand components (each term of the DCC model except the intercept term and two random terms) for each demand projection (chunks and merged contiguous, chronological block model component matrices) organized by spatial scale (city, block group, and household)

### Projection Statistics
* **Path**: `Results/Projection_Statistics/`
* **Contents**: Tabular summary datasets tracking distribution baselines over the projection timeline. Includes calculated metrics for:
  * Arithmetic Mean
  * Standard Deviation
  * Custom Percentile cuts ranging from P0.5 up to P99.5

### Variance Decomposition
* **Path**: `Results/Variance_Decomposition/`
* **Contents**: Matrix outputs containing calculated Variance of Conditional Expectations (VCE) ratios. 
* **Structure**: Cleanly structured chronological tables equipped with explicit, running Year and Month index tags mapping structural uncertainty shifts from 2025 through 2050.

Due to file size restrictions, only sample demand projections and demand components outputs are provided in the Data Repository linked above. Complete demand projections and demand components available upon request.

Final city-, block group-, and household-scale demand projection statistics and variance decomposition results are provided in the Data Repository.

