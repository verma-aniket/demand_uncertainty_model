# Unified Demand Simulator and Variance Decomposition Pipeline

An end-to-end multi-scenario computational pipeline designed to simulate localized urban demand profiles and decompose structural forecast uncertainties over multi-decade planning horizons. The architecture models intersecting volatilities across downscaled climate shifts, macroeconomic pathways, demographic migrations, and granular household behavioral patterns.

---

## Repository Structure

The layout separates underlying data frameworks (`Data/`), core module definitions (`src/`), and decoupled execution nodes (`Scripts/`).

### Core Directories

| Directory Path | Layer / Classification | Primary Operational Function |
| :--- | :--- | :--- |
| `Data/Climate/` | Storage Warehouse | Downscaled meteorological arrays and drought indices |
| `Data/DCC_Model/` | Storage Warehouse | MLE parameters and Variance-Covariance datasets |
| `Data/Economics/` | Storage Warehouse | Inflation paths and historical/projected CPI series |
| `Data/Housing/` | Storage Warehouse | Historical real estate pools and multidecade records |
| `Data/Population/` | Storage Warehouse | Tracked migration matrices and census-level dynamics |
| `Data/Simulator/Inputs/` | Storage Warehouse | Runtime coordinate matrices (`id_mat.npy`, chunks) |
| `Data/Simulator/Outputs/` | Storage Warehouse | Target projection directories (ignored by git tracking) |
| `src/models/` | Architecture Backend | Structural Parameter Identification and Model Calibration |
| `src/utils/` | Architecture Backend | Core Data Ingestion and Mathematical Signal Processing Engines |
| `Scripts/Climate/` | Processing Engine | Meteorological signal extraction and ensemble sampling |
| `Scripts/Conservation/` | Processing Engine | Policy adaptation and intervention trajectory profiling |
| `Scripts/DCC_Model/` | Processing Engine | Financial econometric estimation routines (R/Python) |
| `Scripts/Economics/` | Processing Engine | Macro-economic scenario trajectory generators (OU processes) |
| `Scripts/Population_Growth/` | Processing Engine | Demographic and real estate forecasting systems (GEC/UST) |
| `Scripts/Simulator/` | Processing Engine | Core execution simulator and pipeline post-processing |

### Simulator Subdirectories

| Pipeline Stage Path | Operational Function |
| :--- | :--- |
| `Scripts/Simulator/Input_Grid/` | Spatial coordinate index grid builders and chunk packagers |
| `Scripts/Simulator/Post_Processing/`| Raw output binary matrix collectors and consolidators |
| `Scripts/Simulator/Projection_Stats/`| Distribution summary and quantile/percentile processing scripts |
| `Scripts/Simulator/Variance_Decomposition/` | Variance of Conditional Expectations (VCE) calculation scripts |

---

## Core System Modules (src/)

The src/ modules define the operational backend logic, estimating model parameters and formatting analytical workflows.

### 1. Estimation Models (src/models/)

#### build_GEC_model.py
* **Purpose**: Calibrates a Gravity-Environmental-Demographic migration model.
* **Key Methods**:
  * `get_eta_params` and `get_min_flow_model_params`: Extracts network constraints and baseline structural parameters.
  * `get_gamma` and `get_beta`: Balancing routines that handle population distribution flows.
  * `check_heavy_tails`: Conducts tail-index estimations to identify data thickness and mathematical bounds.
  * `get_zeta_params`: Derives stable final scale parameters for downstream simulation.

#### build_UST_model.py
* **Purpose**: Regresses multi-city real estate pricing and asset pools.
* **Key Methods**:
  * `balance_data`: Cleans and aligns features cross-sectionally across varying city and time horizons.
  * `fama_macbeth`: Executes two-stage asset pricing regressions to compute structural risk coefficients and save parameter vectors.
  * `compute_R2` and `get_R2`: Validation metrics evaluating out-of-sample explanatory boundaries.

### 2. Functional Utilities (src/utils/)

#### climate_functions.py
* **Purpose**: Signal processing and transformation utilities for climate arrays.
* **Key Methods**:
  * `compute_spi` and `calculate_spi`: Implements multi-scale Standardized Precipitation Index fits over historical rainfall matrices.
  * `identify_drought_events` and `get_num_time_drought`: Logical status filters to isolate drought durations, severities, and event spans.
  * `minimal_year_cover` and `minimal_year_cover_random`: Solves set-covering optimizations over variable weather data lengths to standardize historical timelines.

#### read_housing_data.py and read_migration_data.py
* **Purpose**: Automated extraction and cleaning pipelines for structural population and housing datasets.
* **Key Methods**:
  * `get_all_flows` and `get_net_migration_dataset`: Collects and transforms raw disjointed spreadsheets into unified inter-metropolitan flow matrices.
  * `get_avg_neighbors`: Measures regional demographic interaction changes by mapping network geographic adjacencies.

#### simulator_functions.py
* **Purpose**: Optimized calculation engines and helper routines built for high-throughput runtime execution loops.
* **Key Methods**:
  * `make_global_id_V2` and `make_global_id_HH`: Combines spatial vector mappings into unique index hashes across block groups and households.
  * `get_tier_prices`: Models progressive block-rate utility tariff structures stochastically using continuous distributions.
  * `weighted_percentiles`: Performs memory-safe, single-pass sorted quantile interpolations over massive output projection dimensions.
  
  ---

## Execution Nodes (Scripts/)

The pipeline follows a sequential execution chain where upstream scenarios feed the main simulator engine, which in turn outputs arrays for post-processing and variance analytics.

```text
[Climate/Econ/Pop Scripts] ---> [Input_Grid Builders] ---> [Simulator Engine] ---> [Post-Processing/Stats/VCE]
```

### Phase 1: Upstream Driver Generation
These independent scripts evaluate and write background boundary parameters out to the data layer.

#### Climate/
* `save_climate_change_CFs.py`: Extracts and processes baseline delta-change factor anomalies.
* `cluster_weather_projections.py`: Maps spatial weather variables via multivariate clustering.
* `generate_full_weather_ensemble.py`, `sample_drought_scenarios.py`, and `save_spi_parameters.py`: Fits localized historical precipitation bounds to simulate large-scale stochastic weather ensembles.

#### Conservation/
* `generate_conservation_scenarios.py`: Models utility policy intervention parameters using nonlinear logistic diffusion functions (logistic_function).

#### DCC_Model/
* `DCC_LL_Function.R` and `DCC_Model_MLE_Main.R`: Formulates maximum likelihood estimations via R engines to capture dynamic conditional correlations.
* `generate_DCC_parameter_scenarios.py`: Samples parameter scenarios from the resulting joint volatility transmission matrices.

#### Economics/
* `generate_econ_scenarios.py`: Generates continuous macro-financial paths using mean-reverting Ornstein-Uhlenbeck processes (generate_ou_process).

#### Population_Growth/
* `cluster_housing_projections.py`, `GEC/`, and `UST/`: Features parallel tracking modules estimating long-term demographic expansions and real estate resource allocations across urban networks.

### Phase 2: Runtime Coordinate Compilation
Before executing the simulator, target data arrays must be vectorized and mapped to strict index arrays.

#### Simulator/Input_Grid/
* **Scripts**: generate_city_input_ID_grid.py, generate_BG_input_ID_grid.py, generate_HH_input_grid.py, and build_total_city_input_grid.py.
* **Purpose**: Converts raw demographic matrices and shape coordinates into memory-efficient spatial index arrays (id_mat.npy). It breaks massive grids down into localized, process-safe data chunks (process_input_id_grid) to enable scalable parallel loops.

### Phase 3: Demand Simulation Execution
This is the core engine layer that simulates multi-scenario projections.

#### Simulator/
* **Scripts**: run_simulator_city.py and run_simulator_HH.py.
* **Purpose**: Ingests the pre-compiled spatial grid indices to calculate non-linear consumption equations based on baseline demand, microeconomic utility rate adjustments, structural weather sensitivities, and macro-shocks (compute_total_demand).
* **Computational Layout**: Operates over a nested loop framework (generate_demand_projection) to process data chunks sequentially using memory-mapped reading arrays to enforce strict memory safety.

### Phase 4: Output Post-Processing and Analytics
These scripts consolidate the binary output matrix data and execute statistical operations.

#### Simulator/Post_Processing/
* **Scripts**: post_process_city.py, post_process_BG.py, and post_process_HH.py.
* **Purpose**: Gathers scattered runtime binary chunk files from the scratch drive and merges them back into contiguous, chronological block arrays (process_chunk).

#### Simulator/Projection_Stats/
* **Scripts**: proj_stats_city.py, proj_stats_BG.py, and proj_stats_HH.py.
* **Purpose**: Calculates baseline statistical distributions. Generates single-pass weighted means, standard deviations, and smooth percentile cuts across city, block-group, and household datasets.

#### Simulator/Variance_Decomposition/
* **Scripts**: var_decomp_city.py, var_decomp_BG.py, and var_decomp_HH.py.
* **Purpose**: Analyzes forecast uncertainty profiles using a Variance of Conditional Expectations (VCE) approach. It isolates how much forecast variance is driven by specific scenario parameters versus baseline model noise over long horizons.

---

## Deployment and Operational Guidelines

### Standalone Local Execution
Every script is decoupled and can be run independently directly from the repository root:

# 1. Compile structural grids
python Scripts/Simulator/Input_Grid/generate_city_input_ID_grid.py

# 2. Execute localized demand simulation
python Scripts/Simulator/run_simulator_city.py

# 3. Compute city variance decomposition profile
python Scripts/Simulator/Variance_Decomposition/var_decomp_city.py

### High-Performance Cluster (SLURM) Scheduling
Because the analytical scripts are self-contained and leverage relative paths via pathlib, you can submit heavy operations directly to separate cluster partitions without complex environment overhead.

```bash
#!/bin/bash
#SBATCH --job-name=vce_decomposition
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --memory=200GB
#SBATCH --partition=serc
#SBATCH --time=00:30:00
```

# Execute processing script on an allocated compute node
python Scripts/Simulator/Variance_Decomposition/var_decomp_city.py

## Pipeline Outputs

Analytical summaries are saved automatically to your designated data directory:

### Projection Statistics
* **Path**: `Data/Simulator/Outputs/Projection_Statistics/`
* **Contents**: Tabular summary datasets tracking distribution baselines over the projection timeline. Includes calculated metrics for:
  * Arithmetic Mean
  * Standard Deviation
  * Custom Quantile/Percentile cuts ranging from P0.5 up to P99.5

### Variance Decomposition
* **Path**: `Data/Simulator/Outputs/Variance_Decomposition/`
* **Contents**: Matrix outputs containing calculated Variance of Conditional Expectations (VCE) ratios. 
* **Structure**: Cleanly structured chronological tables equipped with explicit, running Year and Month index tags mapping structural uncertainty shifts from 2025 through 2050.
