import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from joblib import Parallel, delayed
from tqdm import tqdm

# set working directories
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# route directories
input_dir = REPO_ROOT / "Data" / "Simulator" / "Inputs"
scenario_dir = REPO_ROOT / "Data" / "Simulator" / "Scenarios"
proj_output_dir = REPO_ROOT / "Results" / "Projections" / "HH"
comp_output_dir = REPO_ROOT / "Results" / "Components" / "HH"

# Import custom processing modules from repository utility space
from src.utils.simulator_functions import *
from src.utils.climate_functions import rolling_sum_nan, calculate_spi, identify_drought_events

# Read SLURM array ID - to determine which household is being modeled
array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# define base random number generator
base_seed = 271124 # for reproducibility
rng_base = np.random.default_rng(base_seed)

# Define global tier constant boundaries (Needed for Numba global scope closure lookup)
tier_bounds = np.array([5, 7, 9], dtype=np.float64)

### Define Local Functions

# Compute total monthly residential water demand
@ njit(fastmath=True)
def compute_hh_demand(log_base_term, log_hh_terms, price_effects, err_eta, err_eps):
    """
    Numba-accelerated function to compute total residential water demand for a household pool 
    given DCC model parameters, weather data, drought data, and household characteristics.

    Note: many computations are performed prior to calling this function for efficiency.

    log_base_term: sum of effects constant for the month (intercept + sine + temp + precip + precip_lag_1)
    log_hh_terms: sum of household characteristic effects (in_drought + post_drought + MHI + house_tax_val + housing charac.)
    price_effects: 1 x 4 array of price effects for n households
    err_eta: float: human behavior error sample
    err_eps: float:  random error sample

    Returns total monthly demand and log price effect (both are single floats).
    """

    # Initialize variables
    total = 0.0
    log_price = 0.0

    # Step 1: compute base log_w without price effects and random error
    log_w = log_base_term + log_hh_terms + err_eta

    # price tier checks (sequential logic per household)
    # Tier 4 test
    if np.exp(log_w + price_effects[3]) > tier_bounds[2]:
        w = np.exp(log_w + price_effects[3] + err_eps)
        price = price_effects[3]
    # Tier 3
    elif np.exp(log_w + price_effects[2]) > tier_bounds[1]:
        w = np.exp(log_w + price_effects[2] + err_eps)
        price = price_effects[3]
    # Tier 2
    elif np.exp(log_w + price_effects[1]) > tier_bounds[0]:
        w = np.exp(log_w + price_effects[1] + err_eps)
        price = price_effects[3]
    else:
        # Tier 1
        w = np.exp(log_w + price_effects[0] + err_eps)
        price = price_effects[3]
    
    # accumulate (reduction)
    total = w
    log_price = price

    # Return summed total demand across all households
    return total, log_price

# define function to build a complete residential demand projection
def generate_demand_projection(scen_idx, hh_drought_exp):
    """
    Generate a single residential demand projection given input parameters.
    """
    # Setup rngs for this projection
    proj_seed = make_seed(make_global_id_HH(id_grid[scen_idx]))
    dcc_seed = make_seed(f"DCC_{id_grid[scen_idx][0]}")
    rng_proj = np.random.default_rng(proj_seed)                                        # rng per global scenario
    rng_dcc = np.random.default_rng(dcc_seed)                                          # rng per dcc scenario
    
    # Initialize output demand arrays for this projection
    demand_array = np.zeros((num_months))                                               # total demand
    if store_components:
        demand_comp_mat = np.zeros((num_months, 16))                                    # individual log-demand components
    
    # A. DCC Parameters
    dcc_coef = dcc_samples[id_grid[scen_idx,0]]
    
    # B. Climate Inputs
    precip = precip_mat[id_grid[scen_idx,1]]                                            # extract precip forecast
    temp = temp_mat[id_grid[scen_idx,1]]                                                # extract temp forecast
    precip_lag_1 = np.roll(precip, 1)                                                   # add precip_lag_1
    precip_lag_1[0] = precip_2024[-1]                                                   # correct first value
    spi = calculate_spi(np.concatenate((precip_2024, precip)), spi_params)              # compute spi-12 values for each month
    drought = identify_drought_events(spi, drought_clip, drought_spi, min_duration)     # identify drought events

    # C. Population and Housing Stock Inputs (Not Applicable)

    # D. Demand Computation

    # D.1 Pre-compute constant terms for all months
    log_w_int = dcc_coef[1]                                                             # intercept term
    log_w_sine = dcc_coef[2] * sine_array                                               # sine wave/seasonality term
    log_w_temp = dcc_coef[3] * temp                                                     # termperature effect
    log_w_precip = dcc_coef[4] * precip                                                 # precipitation effect
    log_w_precip_lag_1 = dcc_coef[5] * precip_lag_1                                     # precipitation lag/persistence effect
    
    # city-wide "in-drought" effect
    log_w_drought_1 = dcc_coef[6] * drought * rng_proj.uniform(low=0, high=1, size=num_months)

    # D.2 Pre-compute error terms once outside of month loop
    human_error = rng_dcc.normal(loc=0, scale=dcc_coef[18], size=num_months)
    random_error = rng_dcc.normal(loc=0, scale=dcc_coef[19], size=num_months) 
    
    # iterate through each month and store total demand
    for m in range(num_months):
        
        # D.3 Household Level Terms

        # Inflation-Adjusted Water Price Term
        log_w_water_price = inf_mat[id_grid[scen_idx,2],m] * dcc_coef[0] * price_arr

        # check if drought event has passed this month - i.e., switching from "in drought" to "post drought"
        if (m > 0) and (drought[m-1] == 1) and (drought[m] == 0):
            # re-sample post drought effects
            hh_drought_exp = rng_proj.uniform(low=hh_drought_exp, high=1, size=1)[0]

        # Post-drougt term
        if drought[m] == 0: # apply decay if we are not in drought
            hh_drought_exp = post_drought_decay * hh_drought_exp
        log_w_drought_2 = dcc_coef[8] * hh_drought_exp
        
        # Income terms
        log_w_income_1 = dcc_coef[9] * hh_char_arr[0]                 # log MHI effect
        log_w_income_2 = dcc_coef[10] * hh_char_arr[1]                # log house tax value effect
        
        # Compute housing characteristics effects terms
        log_w_hh_char_1 = dcc_coef[11] * hh_char_arr[2]               # no. of full bathrooms
        log_w_hh_char_2 = dcc_coef[12] * hh_char_arr[3]               # no. of half bathrooms
        log_w_hh_char_3 = dcc_coef[13] * hh_char_arr[4]               # no. of bedrooms
        log_w_hh_char_4 = dcc_coef[14] * hh_char_arr[5]               # no. of fireplaces
        log_w_hh_char_5 = dcc_coef[15] * hh_char_arr[6]               # log main area
        log_w_hh_char_6 = dcc_coef[16] * hh_char_arr[7]               # log lawn
        log_w_hh_char_7 = dcc_coef[17] * hh_char_arr[8]               # pool presence

        # Compute total demand for this month
        d, p  = compute_hh_demand(log_base_term=log_w_int + log_w_sine[m] + log_w_temp[m] + log_w_precip[m] + log_w_precip_lag_1[m] + log_w_drought_1[m],
                                  log_hh_terms=log_w_drought_2 + log_w_income_1 + log_w_income_2 + 
                                               log_w_hh_char_1 + log_w_hh_char_2 + log_w_hh_char_3 + log_w_hh_char_4 + log_w_hh_char_5 + log_w_hh_char_6 + log_w_hh_char_7,
                                  price_effects=log_w_water_price,
                                  err_eta=human_error[m],
                                  err_eps=random_error[m])
        # Store results
        demand_array[m] = (1 - cons_a_scen[id_grid[scen_idx,3],m]) * (1 - cons_p_scen[id_grid[scen_idx,4],m]) * d

        # Store components as the sum log_w term and store the number of households (so the mean can also be calculated)
        if store_components:
            demand_comp_mat[m,0] = log_w_sine[m]                # sine wave/seasonality term
            demand_comp_mat[m,1] = log_w_temp[m]                # termperature effect
            demand_comp_mat[m,2] = log_w_precip[m]              # precipitation effect
            demand_comp_mat[m,3] = log_w_precip_lag_1[m]        # precipitation lag/persistence effect
            demand_comp_mat[m,4] = log_w_drought_1[m]           # in drought effect
            demand_comp_mat[m,5] = log_w_drought_2              # post drought effect
            demand_comp_mat[m,6] = p                            # price effect
            demand_comp_mat[m,7] = log_w_income_1               # MHI income term
            demand_comp_mat[m,8] = log_w_income_2               # house tax value income effect
            demand_comp_mat[m,9] = log_w_hh_char_1              # full bathroom effect
            demand_comp_mat[m,10] = log_w_hh_char_2             # half bathroom effect
            demand_comp_mat[m,11] = log_w_hh_char_3             # bedroom effect
            demand_comp_mat[m,12] = log_w_hh_char_4             # fireplaces effect
            demand_comp_mat[m,13] = log_w_hh_char_5             # main area effect
            demand_comp_mat[m,14] = log_w_hh_char_6             # lawn size effect
            demand_comp_mat[m,15] = log_w_hh_char_7             # pool presence effect

    if store_components:
        return demand_array, demand_comp_mat
    else:
        return demand_array, 0
        
### Main Script Start

# --- Step 1: Read and Process Input Data ---

# Read DCC Model Parameter Scenarios
dcc_samples = np.loadtxt(scenario_dir / 'DCC_scenarios.csv', delimiter=",", skiprows=1, usecols=range(20))
dcc_param_names = pd.read_csv(scenario_dir / 'DCC_scenarios.csv', nrows=0).columns.tolist()[0:-1]

# Read Climate Change Scenarios
precip_mat = np.loadtxt(scenario_dir / 'precipitation_scenarios.csv', delimiter=",").T                     # precipitation scenarios
temp_mat = np.loadtxt(scenario_dir / 'temperature_scenarios.csv', delimiter=",").T                         # temperature scenarios

# Read SPI parameters and historic precipitation data
spi_params = pd.read_csv(input_dir / 'spi_params.csv', usecols=['a', 'scale', 'q']).to_numpy()               # SPI-12 parameters
precip_2024 = pd.read_csv(input_dir / 'weather_data.csv', usecols=['precip']).to_numpy()[-12:].reshape(-1)   # 2024 precipitation

# Define dictionary to map month of the year using a sine wave (to represent seasonality)
sine_dict = {i: -1 * np.sqrt(2) * np.sin(2 * np.pi * i / 12 + np.arctan2(1, 1)) for i in range(1,13)}       # sine wave dictionary

# Single household demand modeling - so no population growth scenarios

# Read Water Price Inflation Scenarios
inf_mat = np.loadtxt(scenario_dir / 'inflation_scenarios_OU_process.csv', delimiter=",").T

# Read active conservation upper bound
cons_a_ub = np.loadtxt(scenario_dir / 'active_conservation_upper_bound.csv', delimiter=",")

# Read conservation scenarios
cons_a_scen = np.loadtxt(scenario_dir / 'active_conservation_scenarios_10.csv', delimiter=",").T
cons_p_scen = np.loadtxt(scenario_dir / 'passive_conservation_scenarios_10.csv', delimiter=",").T

# Define number of years and number of months based on the simulation horizon length
num_months = precip_mat.shape[1]
num_years = num_months // 12

# Read Housing Characteristics Data

# identify household characteristic terms used in DCC model
hh_char_terms = dcc_param_names[9:18]                                   # hh characteristics used in DCC model
hh_char_terms = [col.replace('beta_', '') for col in hh_char_terms]     # remove 'beta_' prefix from variable names

# Define Housing Stock Sample Set
hh_pool_mat = pd.read_csv(input_dir / "HH_Data" / "hh_samples.csv", usecols=hh_char_terms).to_numpy()            # smallest useful form

# Build Active Housing Stock (of a single household)
hh_char_arr = hh_pool_mat[array_id].copy()

# Retrieve inflation corrected water rate price data
price_cols = ['p_1', 'p_2', 'p_3', 'p_4']
price_arr = np.log(pd.read_csv(input_dir / "HH_Data" / "hh_samples.csv", usecols=price_cols).to_numpy()[array_id])

# Initializa drought memory effect  
hh_exp_drought = 1.0

# --- Step 2: Set Design Parameters ---

# Define Drought Indicator Variables
drought_clip = 0                        # drought clip threshold
drought_spi = -1                        # drought intensity threshold
min_duration = 6                        # minimum drought intensity threhold duration in months
post_drought_ramp = 60                  # months to ramp down post-drought indicator
post_drought_decay = 1                  # decay rate of post drought effect after drought ends (for sensitivity analysis)

# Compute Pre-defined Inputs
sine_array = np.array([sine_dict[i] for i in np.tile(np.arange(1,13), num_years)])
curtail_array = np.zeros(num_months)    # assume no curtailment
tier_bounds = np.array([5, 7, 9])       # define water tier structure boundaries in CCF

# Define boolean to control whether or not to save log demand component data
store_components = True

# Read Input ID matrix chunk
id_grid = np.load(input_dir / "HH_Data" / "hh_id_grid.npy")
num_proj = id_grid.shape[0]
num_proj = 9                            # for testing, delete/comment out to run full ensemble

# --- Step 3: Develop Demand Projections ---

# Run demand projections in parallel
results = Parallel(n_jobs=32)(
    delayed(generate_demand_projection)(scen_idx=p, hh_drought_exp = hh_exp_drought)
    for p in tqdm(range(num_proj), desc="Demand Projections", ncols=100, colour="green", mininterval=60, maxinterval=120)
)

# --- Step 4: Save Results ---

# unpack results
demand_projections, demand_components = zip(*results)
demand_matrix = np.vstack(demand_projections)
components_matrix = np.vstack(demand_components)

# Convert to float32 to half memory requirements
demand_matrix = demand_matrix.astype(np.float32)
components_matrix = components_matrix.astype(np.float32)

# Save Results
np.save(proj_output_dir /  f"hh_proj_{array_id}.npy", demand_matrix)
if store_components: 
    np.save(comp_output_dir / f"hh_comp_{array_id}.npy", components_matrix)

# --- END OF SCRIPT ---