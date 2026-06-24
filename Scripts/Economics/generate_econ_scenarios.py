import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import qmc

# 1. LINK CORE FOLDERS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

def generate_ou_process(n_scen, n_months, k_start, k_mean, k_min, k_max, eta_m, sigma_m, rng):
    """Generates Ornstein-Uhlenbeck (OU) mean-reverting tracking state matrix paths."""
    k_paths = np.zeros((n_scen, n_months))
    k_paths[:, 0] = k_start

    for t in range(1, n_months):
        shocks = rng.normal(0, sigma_m, size=n_scen)
        reversion = eta_m * (k_mean - k_paths[:, t-1])
        k_paths[:, t] = k_paths[:, t-1] + reversion + shocks
        k_paths[:, t] = np.clip(k_paths[:, t], k_min, k_max)

    return k_paths

def main():
    # 2. ROUTE REPO-RELATIVE TARGET ENVIRONMENT DIRECTORIES
    input_inflation_file = REPO_ROOT / "Data" / "Economics" / "income_cpi_data.csv"
    output_scenario_file = REPO_ROOT / "Data" / "Simulator" / "Scenarios" / "inflation_scenarios_OU_process.csv"

    # 3. INITIALIZE PRNG AND SCENARIO SPECS
    base_seed = 271124
    rng_base = np.random.default_rng(base_seed)
    n_scen = 10
    n_months = 312

    # 4. COMPUTE BASE HISTORICAL METRICS (CAGR)
    inf_data = pd.read_csv(input_inflation_file)
    n_years = inf_data['Year'].values[-1] - inf_data['Year'].values[0]
    
    cagr_inc = (inf_data['AAP'].values[-1] / inf_data['AAP'].values[0]) ** (1 / n_years) - 1
    cagr_inf = (inf_data['CPI-U'].values[-1] / inf_data['CPI-U'].values[0]) ** (1 / n_years) - 1

    # Base monthly compounding rates
    m_inf = (1 + cagr_inf) ** (1 / 12) - 1
    m_inc = (1 + cagr_inc) ** (1 / 12) - 1

    # 5. CALIBRATE ORNSTEIN-UHLENBECK (OU) MULTIPLIERS
    k_ann_max = 3.0                        # 3x general inflation
    k_ann_min = 1.0                        # Pace with general inflation
    
    m_wat_max = ((1 + k_ann_max * cagr_inf) ** (1 / 12) - 1)
    m_wat_min = ((1 + k_ann_min * cagr_inf) ** (1 / 12) - 1)

    k_max = m_wat_max / m_inf
    k_min = m_wat_min / m_inf
    k_mean = (k_min + k_max) / 2
    k_range = k_max - k_min

    # Reversion and volatility matrices scaling
    half_life = 60                         # 5-year UWMP planning steps 
    eta_m = np.log(2) / half_life          # Monthly mean reversion strength
    eta_a = eta_m * 12                     # Annual structural scaling factor

    sd_inf = k_range / 4                   # Capture 95% range bounds inside 2 SDs
    sigma_a = sd_inf * np.sqrt(2 * eta_a)
    sigma_m = sigma_a / np.sqrt(12)        # Calibrated monthly step volatility

    # 6. LATIN HYPERCUBE SEED ENVELOPE (LHS)
    sampler = qmc.LatinHypercube(d=1, scramble=True, seed=base_seed)
    U = sampler.random(n=n_scen).reshape(-1)
    k_start = U * (k_max - k_min) + k_min

    # 7. GENERATE PROCESS TRAJECTORIES VIA PIPELINE ENG
    print("Generating mean-reverting economic paths over model horizon...")
    k_paths = generate_ou_process(
        n_scen=n_scen,
        n_months=n_months,
        k_start=k_start,
        k_mean=k_mean,
        k_min=k_min,
        k_max=k_max,
        eta_m=eta_m,
        sigma_m=sigma_m,
        rng=rng_base
    )

    # Convert paths to inflation adjustments
    growth_matrix = 1 + (k_paths * m_inf)
    wat_sample_f = np.cumprod(growth_matrix, axis=1)

    # Compute monthly compound denominator matrices
    inc_f = np.cumprod(np.repeat((1 + m_inc), n_months))

    # Normalize relative to income inflation parameters
    wat_sample_inc = wat_sample_f / inc_f

    # 8. EXPORT SCENARIOS DIRECTLY TO DESTINATION
    pd.DataFrame(data=wat_sample_inc.T).to_csv(output_scenario_file, index=False, header=False)
    print(f"Water inflation scenarios saved.")

if __name__ == "__main__":
    main()