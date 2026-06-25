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
BG_input_dir = REPO_ROOT / "Data" / "Simulator" / "Inputs" / "BG_Data"
scenario_dir = REPO_ROOT / "Data" / "Simulator" / "Scenarios"
proj_output_dir = REPO_ROOT / "Results" / "Projections" / "BG"
comp_output_dir = REPO_ROOT / "Results" / "Components" / "BG"

# Import custom processing modules from repository utility space
from src.utils.simulator_functions import *
from src.utils.climate_functions import rolling_sum_nan, calculate_spi, identify_drought_events

# Parse SLURM batch execution state variables 
array_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

# read FIPS code based on slurm job id
FIPS_samples = np.array([60871010002, 60871012001, 60871208003])
FIPS_array = np.repeat(FIPS_samples, 10)
FIPS_code = FIPS_array[array_id]

# read input ID grid index based on slurm job id
id_grid_index = np.tile(np.arange(0,10), 3)
input_index = id_grid_index[array_id]

# define base random number generator
base_seed = 160596 # for reproducibility
rng_base = np.random.default_rng(base_seed)

# Define global tier constant boundaries (Needed for Numba global scope closure lookup)
tier_bounds = np.array([5, 7, 9], dtype=np.float64)

### Define Local Functions

# Compute total monthly residential water demand
@ njit(fastmath=True)
def compute_total_demand(log_base_term, log_hh_terms, price_effects, err_eta, err_eps, cons):
    """
    Numba-accelerated function to compute total residential water demand for a household pool 
    given DCC model parameters, weather data, drought data, and household characteristics.

    Note: many computations are performed prior to calling this function for efficiency.

    log_base_term: sum of effects constant for the month (intercept + sine + temp + precip + precip_lag_1)
    log_hh_terms: n x 1 vector of household characteristic effects (in_drought + post_drought + MHI + house_tax_val + housing charac.)
    price_effects: n x 4 array of price effects for n households
    err_eta: n x 1 array of human behavior errors
    err_eps: n x 1 array of random errors
    cons: n x 1 array of active conservation effect

    Returns total monthly demand and log price effect (both are single floats).
    """
    # Initialize variables
    n = log_hh_terms.shape[0]
    total = 0.0
    log_price_total = 0.0

    # loop over households in parallel
    for i in range(n):

        # Step 1: compute base log_w without price effects and random error
        log_w = log_base_term + log_hh_terms[i] + err_eta[i]

        # price tier checks (sequential logic per household)
        # Tier 4 test
        if np.exp(log_w + price_effects[i, 3]) > tier_bounds[2]:
            w = np.exp(log_w + price_effects[i, 3] + err_eps[i])
            price = price_effects[i, 3]
        # Tier 3
        elif np.exp(log_w + price_effects[i, 2]) > tier_bounds[1]:
            w = np.exp(log_w + price_effects[i, 2] + err_eps[i])
            price = price_effects[i, 3]
        # Tier 2
        elif np.exp(log_w + price_effects[i, 1]) > tier_bounds[0]:
            w = np.exp(log_w + price_effects[i, 1] + err_eps[i])
            price = price_effects[i, 3]
        else:
            # Tier 1
            w = np.exp(log_w + price_effects[i, 0] + err_eps[i])
            price = price_effects[i, 3]
        
        # accumulate (reduction)
        total += cons[i]*w
        log_price_total += price

    # Return summed total demand across all households
    return total, log_price_total

# define function to build a complete residential demand projection
def generate_demand_projection(scen_idx, hh_pool_list, price_matrix_list, hh_drought_exp_list, hh_cons_list):
    """
    Generate a single residential demand projection given input parameters.
    """
    # Setup rngs for this projection
    proj_seed = make_seed(make_global_id(id_grid[scen_idx][0:-1]))
    dcc_seed = make_seed(f"DCC_{id_grid[scen_idx][0]}")
    pop_seed = make_seed(f"POP_{id_grid[scen_idx][2]}")
    rng_proj = np.random.default_rng(proj_seed)                                        # rng per global scenario
    rng_dcc = np.random.default_rng(dcc_seed)                                          # rng per dcc scenario
    rng_pop = np.random.default_rng(pop_seed)                                          # rng per population scenario
    ss_proj = np.random.SeedSequence(proj_seed)                                        # one SeedSequence per household composition realization
    ss_hh = ss_proj.spawn(num_hh_comp)
    rng_hh = []                                 
    for rep in range(num_hh_comp):
        ss_months = ss_hh[rep].spawn(num_months)
        rng_months = [np.random.default_rng(s) for s in ss_months]                      # rng per month per household composition
        rng_hh.append(rng_months)
    
    # Initialize output demand arrays for this projection
    demand_array = np.zeros((num_months))                                               # total demand
    if store_components:
        demand_comp_mat = np.zeros((num_months, 17))                                    # individual log-demand components
    
    # A. DCC Parameters
    dcc_coef = dcc_samples[id_grid[scen_idx,0]]
    
    # B. Climate Inputs
    precip = precip_mat[id_grid[scen_idx,1]]                                            # extract precip forecast
    temp = temp_mat[id_grid[scen_idx,1]]                                                # extract temp forecast
    precip_lag_1 = np.roll(precip, 1)                                                   # add precip_lag_1
    precip_lag_1[0] = precip_2024[-1]                                                   # correct first value
    spi = calculate_spi(np.concatenate((precip_2024, precip)), spi_params)              # compute spi-12 values for each month
    drought = identify_drought_events(spi, drought_clip, drought_spi, min_duration)     # identify drought events

    # C. Population and Housing Stock Inputs
    hh_delta = split_vector_into_12(vec=pop_mat[id_grid[scen_idx,2]], rng=rng_pop)     # decompose annual to monthly housing unit changes

    # D. Demand Computation

    # D.1 Pre-compute constant terms for all months
    log_w_int = dcc_coef[1]                                                             # intercept term
    log_w_sine = dcc_coef[2] * sine_array                                               # sine wave/seasonality term
    log_w_temp = dcc_coef[3] * temp                                                     # termperature effect
    log_w_precip = dcc_coef[4] * precip                                                 # precipitation effect
    log_w_precip_lag_1 = dcc_coef[5] * precip_lag_1                                     # precipitation lag/persistence effect
    
    # city-wide "in-drought" effect
    log_w_drought_1 = dcc_coef[6] * drought * rng_proj.uniform(low=0, high=1, size=num_months)
    
    # household-level active conservation
    cons_c = rng_proj.uniform(low=0, high=1, size=len(hh_cons_list))
    hh_cons_list = cons_c.tolist()

    # iterate through each month and store total demand
    for m in range(num_months):

        # D.2 Update Housing Stock
        # Add or remove new housing units for this month
        if hh_delta[m] > 0:
            # sample new housing units from total housing pool
            new_hh_ids = rng_hh[id_grid[scen_idx,5]][m].choice(hh_pool_ids, size=hh_delta[m], replace=False)
            new_hhs = hh_pool_mat[new_hh_ids]
            new_pipe_sizes = hh_pool_pipe_sizes[new_hh_ids]
            new_prices = get_tier_prices(new_pipe_sizes, tier_price_dict, rng=rng_hh[id_grid[scen_idx,5]][m])
            # extend both lists
            hh_pool_list.extend(new_hhs.tolist())
            price_matrix_list.extend(new_prices.tolist())

            # assume new households have not seen drought before
            # update drought experience list
            hh_drought_exp_list.extend([0.0] * hh_delta[m])
            
            # add household active conservation multiplier
            new_hh_cons = rng_proj.uniform(low=0, high=1, size=hh_delta[m])
            hh_cons_list.extend(new_hh_cons.tolist())

        elif hh_delta[m] < 0:
            # remove housing units from existing pool at random
            remove_indices = rng_hh[id_grid[scen_idx,5]][m].choice(len(hh_pool_list), size=abs(hh_delta[m]), replace=False)

            # remove in descending order to keep indices valid
            for idx in sorted(remove_indices, reverse=True):
                swap_remove(hh_pool_list, idx)
                swap_remove(price_matrix_list, idx)
                swap_remove(hh_drought_exp_list, idx)
                swap_remove(hh_cons_list, idx)
        
        # define total number of households in this month
        num_hh = len(hh_drought_exp_list)
        
        # D.3 Household Level Terms

        # Inflation-Adjusted Water Price Term
        log_w_water_price = inf_mat[id_grid[scen_idx,3],m] * dcc_coef[0] * list_to_matrix(price_matrix_list)

        # check if drought event has passed this month - i.e., switching from "in drought" to "post drought"
        if (m > 0) and (drought[m-1] == 1) and (drought[m] == 0):
            # all households have now experienced drought
            # sample post-drought demand hardening response
            temp = np.array(hh_drought_exp_list) # convert to array for easier indexing
            
            # Case 1
            mask1 = temp == 0.0 # identify households that have not experienced drought yet
            response1 = rng_proj.uniform(low=0, high=1, size=mask1.sum()) # sample response for households that have not experienced drought yet
            temp[mask1] = response1 # update drought experience list with sampled response values for those households
            
            # Case 2
            mask2 = ~mask1 # identify households that have experienced at least one drought
            response2 = rng_proj.uniform(low=temp[mask2], high=1, size=mask2.sum())
            temp[mask2] = response2
            
            hh_drought_exp_list = temp.tolist() # convert back to list for dynamic appending/removal in future months

        # Post-drougt term
        if drought[m] == 0: # apply decay if we are not in drought
            hh_drought_exp_list = (post_drought_decay * np.array(hh_drought_exp_list)).tolist() 
        log_w_drought_2 = dcc_coef[8] * np.array(hh_drought_exp_list)
        
        # convert to matrix for vector multiplication
        hh_char_mat = list_to_matrix(hh_pool_list)
        
        # Income terms
        log_w_income_1 = dcc_coef[9] * hh_char_mat[:,0]                 # log MHI effect
        log_w_income_2 = dcc_coef[10] * hh_char_mat[:,1]                # log house tax value effect
        
        # Compute housing characteristics effects terms
        log_w_hh_char_1 = dcc_coef[11] * hh_char_mat[:,2]               # no. of full bathrooms
        log_w_hh_char_2 = dcc_coef[12] * hh_char_mat[:,3]               # no. of half bathrooms
        log_w_hh_char_3 = dcc_coef[13] * hh_char_mat[:,4]               # no. of bedrooms
        log_w_hh_char_4 = dcc_coef[14] * hh_char_mat[:,5]               # no. of fireplaces
        log_w_hh_char_5 = dcc_coef[15] * hh_char_mat[:,6]               # log main area
        log_w_hh_char_6 = dcc_coef[16] * hh_char_mat[:,7]               # log lawn
        log_w_hh_char_7 = dcc_coef[17] * hh_char_mat[:,8]               # pool presence
        
        # Build human preferences and random error arrays
        human_error = rng_dcc.normal(loc=0, scale=dcc_coef[18], size=num_hh)
        random_error = rng_dcc.normal(loc=0, scale=dcc_coef[19], size=num_hh)

        # Active conservation effect
        if id_grid[scen_idx,4] == 1: # check if active conservation is on for this scenario
            hh_cons = 1.0 - (cons_a_ub[m] * np.array(hh_cons_list))
        else:
            hh_cons = np.ones(num_hh)

        # Compute total demand for this month
        d, p  = compute_total_demand(log_base_term=log_w_int + log_w_sine[m] + log_w_temp[m] + log_w_precip[m] + log_w_precip_lag_1[m] + log_w_drought_1[m],
                                     log_hh_terms=log_w_drought_2 + log_w_income_1 + log_w_income_2 + 
                                                  log_w_hh_char_1 + log_w_hh_char_2 + log_w_hh_char_3 + log_w_hh_char_4 + log_w_hh_char_5 + log_w_hh_char_6 + log_w_hh_char_7,
                                     price_effects=log_w_water_price,
                                     err_eta=human_error,
                                     err_eps=random_error,
                                     cons=hh_cons)
        # Store results
        demand_array[m] = d

        # Store components as the sum log_w term and store the number of households (so the mean can also be calculated)
        if store_components:
            demand_comp_mat[m,0] = log_w_sine[m]                # sine wave/seasonality term
            demand_comp_mat[m,1] = log_w_temp[m]                # termperature effect
            demand_comp_mat[m,2] = log_w_precip[m]              # precipitation effect
            demand_comp_mat[m,3] = log_w_precip_lag_1[m]        # precipitation lag/persistence effect
            demand_comp_mat[m,4] = log_w_drought_1[m]           # in drought effect
            demand_comp_mat[m,5] = np.mean(log_w_drought_2)     # post drought effect
            demand_comp_mat[m,6] = p/num_hh                     # price effect
            demand_comp_mat[m,7] = np.mean(log_w_income_1)      # MHI income term
            demand_comp_mat[m,8] = np.mean(log_w_income_2)      # house tax value income effect
            demand_comp_mat[m,9] = np.mean(log_w_hh_char_1)     # full bathroom effect
            demand_comp_mat[m,10] = np.mean(log_w_hh_char_2)    # half bathroom effect
            demand_comp_mat[m,11] = np.mean(log_w_hh_char_3)    # bedroom effect
            demand_comp_mat[m,12] = np.mean(log_w_hh_char_4)    # fireplaces effect
            demand_comp_mat[m,13] = np.mean(log_w_hh_char_5)    # main area effect
            demand_comp_mat[m,14] = np.mean(log_w_hh_char_6)    # lawn size effect
            demand_comp_mat[m,15] = np.mean(log_w_hh_char_7)    # pool presence effect
            demand_comp_mat[m,16] = num_hh                      # number of households

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

# Read Population Growth Scenarios
pop_mat = np.loadtxt(scenario_dir / f'population_scenarios_{FIPS_code}.csv', delimiter=",").T

# Read Water Price Inflation Scenarios
inf_mat = np.loadtxt(scenario_dir / 'inflation_scenarios_OU_process.csv', delimiter=",").T

# Read active conservation upper bound
cons_a_ub = np.loadtxt(scenario_dir / 'active_conservation_upper_bound.csv', delimiter=",")

# Define number of years and number of months based on the simulation horizon length
num_years = pop_mat.shape[1]
num_months = num_years*12

# Read Housing Characteristics Data

# build cache for tier prices by pipe size
price_df = pd.read_csv(BG_input_dir / f'tier_prices_2021_{FIPS_code}.csv')
tier_price_dict = {}
for ps, df in price_df.groupby("pipe_size"):
    counts = df["count"].to_numpy(dtype=float)
    probs = counts / counts.sum()
    tier_price_dict[ps] = {
        "tiers": df[["p_1","p_2","p_3","p_4"]].values,
        "probs": probs,
        "cdf": np.cumsum(probs)
    }

# identify household characteristic terms used in DCC model
hh_char_terms = dcc_param_names[9:18]                                   # hh characteristics used in DCC model
hh_char_terms = [col.replace('beta_', '') for col in hh_char_terms]     # remove 'beta_' prefix from variable names

# Define Housing Stock Sample Set
hh_pool_mat = pd.read_csv(BG_input_dir / f'total_housing_pool_{FIPS_code}.csv', usecols=hh_char_terms).to_numpy()            # smallest useful form
hh_pool_pipe_sizes = pd.read_csv(BG_input_dir / f'total_housing_pool_{FIPS_code}.csv', usecols=['pipe_size']).to_numpy()     # for price sampling
hh_pool_ids = np.arange(0, hh_pool_mat.shape[0])                                                            # for faster sampling

# Build Active Housing Stock
# Read data into row lists for easy appending/removing
hh_pool_ini = to_row_lists(pd.read_csv(BG_input_dir / f'2021_housing_pool_{FIPS_code}.csv', usecols=hh_char_terms).to_numpy())   # initial housing stock
price_mat_ini = to_row_lists(get_tier_prices(pipe_sizes=pd.read_csv(BG_input_dir / f'2021_housing_pool_{FIPS_code}.csv', usecols=['pipe_size']).to_numpy(),
                                             price_dict=tier_price_dict, rng=rng_base))                         # initial prices vector

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
id_grid = np.load(BG_input_dir / f"ID_Grid/input_id_chunk_{array_id}.npy")
num_proj = id_grid.shape[0]
num_hh_comp = 3                         # number of household composition seeds to sample (FIXED)
num_proj = 9                            # for testing, delete/comment out to run full ensemble

# --- Step 3: Develop Demand Projections ---

# Run demand projections in parallel
results = Parallel(n_jobs=-1)(
    delayed(generate_demand_projection)(scen_idx=p,
                                        hh_pool_list=[row.copy() for row in hh_pool_ini],
                                        price_matrix_list=[row.copy() for row in price_mat_ini],
                                        hh_drought_exp_list=[1.0] * len(hh_pool_ini),
                                        hh_cons_list=[0.0] * len(hh_pool_ini))
    for p in tqdm(range(num_proj), desc="Demand Projections", ncols=100, colour="green", mininterval=60, maxinterval=120)
)

# --- Step 4: Save Results ---

# unpack results
demand_projections, demand_components = zip(*results)
demand_matrix = np.vstack(demand_projections)
components_matrix = np.vstack(demand_components)

# Convert demand projections from CCF to MG
C = 0.000748051948051948 # conversion factor from CCF to MG
demand_matrix = C * demand_matrix

# Save Results
np.save(proj_output_dir /  f"proj_chunk_{input_index}_{FIPS_code}.npy", demand_matrix)
if store_components: 
    np.save(comp_output_dir / f"comp_chunk_{input_index}_{FIPS_code}.npy", components_matrix)

# --- END OF SCRIPT ---