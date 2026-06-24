# import root libraries
import sys
from pathlib import Path
import pandas as pd

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.models import build_GEC_model

def main():
    # route directories
    data_dir = REPO_ROOT / "Data" / "Population" / "Clean_Data"
    param_dir = REPO_ROOT / "Data" / "Population" / "Params"

    # read processed migration and population data
    net_flows_df = pd.read_csv(data_dir / 'GEC_Model_Data_Net_Flows.csv')
    all_flows_df = pd.read_csv(data_dir / 'GEC_Model_Data_All_Flows.csv')
    msa_pop_data = pd.read_csv(data_dir / 'MSA_Population_Data.csv')
    N_data = pd.read_csv(data_dir / 'Average_Neighbors.csv')
    diff_data = pd.read_csv(data_dir / 'Diff_Data.csv')

    years = [2014, 2015, 2016, 2017, 2018, 2019, 2020]

    # Dictionaries to gather parameter calculations across years
    eta_results = []
    min_flow_results = []
    gamma_results = []
    beta_results = []
    alpha_results = []
    zeta_results = []

    for yr in years:
        print(f"Processing Year: {yr}")

        # Compute Eta parameters
        mu, sigma = build_GEC_model.get_eta_params(msa_pop_data.copy(), yr)
        eta_results.append({'Year': yr, 'mu': mu, 'sigma': sigma})

        # Compute Minimal Flow Parameters
        ini_guess = [0.001, 0.4]
        year_flows = all_flows_df[all_flows_df['Year'] == yr].copy()
        I_0, v = build_GEC_model.get_min_flow_model_params(year_flows, yr, ini_guess)
        min_flow_results.append({'Year': yr, 'I_0': I_0, 'v': v})

        # Compute Gamma Parameters
        year_pop = all_flows_df[all_flows_df['Year'] == yr][['i', 'S_i']].copy()
        year_N = N_data[N_data['Year'] == yr][['i', 'N_i']].copy()
        gamma = build_GEC_model.get_gamma(year_pop, year_N, yr)
        gamma_results.append({'Year': yr, 'gamma': gamma})

        # Compute Beta Parameters
        year_net = net_flows_df[net_flows_df['Year'] == yr].copy()
        beta = build_GEC_model.get_beta(year_net, yr)
        beta_results.append({'Year': yr, 'beta': beta})

        # Compute Base Zeta and Alpha Tail Parameters
        year_diff = diff_data[diff_data['Year'] == yr].copy()
        
        # Check heavy tails (Alpha)
        alpha_val = build_GEC_model.check_heavy_tails(year_diff.copy(), yr, v)
        alpha_results.append({'Year': yr, 'alpha': alpha_val})

        # Compute Zeta Parameters
        z_alpha, z_beta, z_loc, z_scale = build_GEC_model.get_zeta_params(year_net.copy(), yr, beta)
        zeta_results.append({
            'Year': yr, 'alpha': z_alpha, 'beta': z_beta, 'loc': z_loc, 'scale': z_scale
        })

    # save collected values to csv files
    pd.DataFrame(eta_results).to_csv(param_dir / 'eta_param.csv', index=False)
    pd.DataFrame(min_flow_results).to_csv(param_dir / 'min_flow_model_param.csv', index=False)
    pd.DataFrame(gamma_results).to_csv(param_dir / 'gamma_param.csv', index=False)
    pd.DataFrame(beta_results).to_csv(param_dir / 'beta_param.csv', index=False)
    pd.DataFrame(alpha_results).to_csv(param_dir / 'alpha_param.csv', index=False)
    pd.DataFrame(zeta_results).to_csv(param_dir / 'zeta_param.csv', index=False)
    
    print(f"All parameters successfully calculated and saved to: {param_dir}")

if __name__ == "__main__":
    main()