import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import qmc

# 1. LINK CORE FOLDERS 
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

def logistic_function(x, a, b, c):
    return a / (1.0 + b*np.exp(-c * x))

def main():
    # 2. ROUTE OUTPUT PATHS DYNAMICALLY RELATIVE TO REPO ROOT
    output_dir = REPO_ROOT / "Data" / "Simulator" / "Scenarios"
    passive_scen_output = output_dir / "passive_conservation_scenarios_10.csv"
    active_ub_output = output_dir / "active_conservation_upper_bound.csv"
    active_scen_output = output_dir / "active_conservation_scenarios_10.csv"

    # 3. INITIALIZE PARAMETERS AND CONSERVATION PLAN VECTORS
    base_seed = 27
    num_scen = 10
    num_months = 26 * 12  # Jan 2025 to Dec 2050 (312 months)

    years = np.array([2020, 2025, 2030, 2035])
    x_data = np.arange(1, len(years) + 1)
    
    # Projections from Santa Cruz Conservation Study
    base_demand_data = np.array([3560, 3636, 3743, 3839])
    plum_demand = np.array([96, 179, 269, 329])
    plum_prog_demand = np.array([233, 411, 538, 619])
    prog_demand = plum_prog_demand - plum_demand
    con_low_data = plum_demand / base_demand_data
    con_mid_data = prog_demand / base_demand_data
    con_high_data = plum_prog_demand / base_demand_data

    # 4. EXECUTE REGRESSION LOGISTIC REGRESSION FITS
    print("Fitting historical master conservation plans to logistic curves...")
    p0 = [1, 1, 1]  # Initial parameter guesses

    p_opt_low, _ = curve_fit(logistic_function, x_data, con_low_data, p0=p0)
    p_opt_mid, _ = curve_fit(logistic_function, x_data, con_mid_data, p0=p0)
    p_opt_high, _ = curve_fit(logistic_function, x_data, con_high_data, p0=p0)

    # Compute continuous monthly timelines (December-anchored splits)
    x_plot_data = np.linspace(1, 7, 31 * 12)
    con_low_plot = logistic_function(x_plot_data, *p_opt_low)
    con_mid_plot = logistic_function(x_plot_data, *p_opt_mid)
    con_high_plot = logistic_function(x_plot_data, *p_opt_high)

    # Subset vectors to planning horizon (last 312 months)
    plumbing_scen = con_low_plot[-num_months:]
    program_scen = con_mid_plot[-num_months:]
    total_scen = con_high_plot[-num_months:]

    # Bundle array in-memory (bypassing saving intermediate con_scen_data.csv)
    con_data = np.vstack((plumbing_scen, program_scen, total_scen))

    # 5. LATIN HYPERCUBE SAMPLING (LHS) SCENARIO GENERATION
    print(f"Generating {num_scen} stochastic sampling permutations via LHS...")
    sampler = qmc.LatinHypercube(d=1, seed=base_seed)
    
    # Generate plumbing (passive) samples
    plum_samples = sampler.random(n=num_scen).reshape(-1)
    plum_scen = con_data[0] * plum_samples[:, np.newaxis]

    # Generate program (active) samples (for household-level projections only)
    prog_samples = sampler.random(n=num_scen).reshape(-1)
    prog_scen = con_data[1] * prog_samples[:, np.newaxis]

    # 6. EXPORT FINALIZED DEMAND MATRICES
    print("Saving processed conservation scenarios to the repository environment...")
    
    # Save plumbing conservation cenarios
    pd.DataFrame(data=plum_scen.T).to_csv(passive_scen_output, index=False, header=False)
    
    # Save active conservation upper bound curve
    pd.DataFrame(data=con_data[1]).to_csv(active_ub_output, index=False, header=False)

    # Save actvive conservation scenarios
    pd.DataFrame(data=prog_scen.T).to_csv(active_scen_output, index=False, header=False)

    print(f"Passive conservation scenarios saved.")
    print(f"Active conservation upper bound saved.")
    print(f"Active conservation scenarios saved.")

if __name__ == "__main__":
    main()