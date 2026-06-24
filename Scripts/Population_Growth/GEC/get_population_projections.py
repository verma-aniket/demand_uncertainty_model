# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as ss

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# define constant share function
def get_city_pop(p0, p1, P0, P1, P2):
    # return p1 + ((p1 - p0) / (P1 - P0)) * (P2 - P1) # share-of-growth
    # return p1* (P2/P1) + 0.25*P2*(p1/P1 - p0/P0) # implicit-shift-share
    return (p1/P1)*P2 # constant share

# define number of MC samples and sample length
num_samples = 1000
sample_len = 29

# set year to extract parameter values
param_year = 2019

# set year to start developing projections
start_year = 2025

def main():
    # route directories
    data_dir = REPO_ROOT / "data" / "Population" / "Clean_Data"
    param_dir = REPO_ROOT / "data" / "Population" / "Params"
    output_dir = REPO_ROOT / "data" / "Population" / "Projections"

    # read parameters
    min_flow_model_param = pd.read_csv(param_dir / 'min_flow_model_param.csv')
    gamma_param = pd.read_csv(param_dir / 'gamma_param.csv')
    beta_param = pd.read_csv(param_dir / 'beta_param.csv')
    zeta_param = pd.read_csv(param_dir / 'zeta_param.csv')
    eta_param = pd.read_csv(param_dir / 'eta_param.csv')
    
    # extract parameters for specific year
    mu = eta_param[eta_param['Year'] == param_year]['mu'].values[0]
    sigma = eta_param[eta_param['Year'] == param_year]['sigma'].values[0]
    I_0 = min_flow_model_param[min_flow_model_param['Year'] == param_year]['I_0'].values[0]
    v = min_flow_model_param[min_flow_model_param['Year'] == param_year]['v'].values[0]
    gamma = gamma_param[gamma_param['Year'] == param_year]['gamma'].values[0]
    beta = beta_param[beta_param['Year'] == param_year]['beta'].values[0]
    z_alpha = zeta_param[zeta_param['Year'] == param_year]['alpha'].values[0]
    z_beta = zeta_param[zeta_param['Year'] == param_year]['beta'].values[0]
    z_loc = zeta_param[zeta_param['Year'] == param_year]['loc'].values[0]
    z_scale = zeta_param[zeta_param['Year'] == param_year]['scale'].values[0]

    # read historic data
    msa_data = pd.read_csv(data_dir / 'SCW_Hist_Pop.csv')
    msa_SC = msa_data[msa_data['Year'] <= start_year-1]
    msa_SC.loc[:,'Pop'] = msa_SC['Pop']*1000 # convert from 1000 people to people
    city_SC = pd.read_csv(data_dir / 'SC_City_Pop_Census_Data.csv')
    city_SC = city_SC[city_SC['Year'] <= start_year-1]
    city_SC.rename(columns={'Population': 'Pop'}, inplace=True)

    # read Santa Cruz-Watsonville MSA population in year "start_year - 1"
    start_pop = msa_SC['Pop'][len(msa_SC)-1]

    # set random seed
    np.random.seed(0)

    # define empty results array
    years = np.arange(start_year,start_year+sample_len,1)
    pop_proj = np.zeros([num_samples,sample_len])

    for i in range(num_samples):

        # sample from normal and Levy distribution
        eta_samples = ss.norm.rvs(loc=mu, scale=sigma, size = sample_len)
        zeta_samples = ss.levy_stable.rvs(alpha = z_alpha, beta=z_beta, loc=z_loc, scale=z_scale, size = sample_len)

        # apply the urban growth equations of cities model
        for c in range(sample_len):
            if c == 0:
                Si = start_pop
            else:
                Si = pop_proj[i,c-1]

            pop_proj[i,c] = Si + eta_samples[c]*Si + (Si**beta)*zeta_samples[c]

    # save MSA population projection results
    msa_proj = pd.DataFrame(np.round(pop_proj.T, decimals=0)).astype(int)
    msa_proj['Year'] = years

    # move year column to first position
    cols = msa_proj.columns.tolist()
    cols.remove('Year')
    cols.insert(0, 'Year')
    msa_proj = msa_proj[cols]

    # compute city-level population projections
    city_proj = np.zeros_like(pop_proj)
    for i in range(num_samples):
        for j in range(sample_len):
            # first two projections rely on historic data
            if j == 0:
                city_proj[i,j] = get_city_pop(p0=city_SC['Pop'][len(city_SC)-2],
                                            p1=city_SC['Pop'][len(city_SC)-1],
                                            P0=msa_SC['Pop'][len(msa_SC)-2],
                                            P1=msa_SC['Pop'][len(msa_SC)-1],
                                            P2=pop_proj[i,j])
            elif j == 1:
                city_proj[i,j] = get_city_pop(p0=city_SC['Pop'][len(city_SC)-1],
                                            p1=city_proj[i,j-1],
                                            P0=msa_SC['Pop'][len(msa_SC)-1],
                                            P1=pop_proj[i,j-1],
                                            P2=pop_proj[i,j])
            else:
                city_proj[i,j] = get_city_pop(p0=city_proj[i,j-2],
                                            p1=city_proj[i,j-1],
                                            P0=pop_proj[i,j-2],
                                            P1=pop_proj[i,j-1],
                                            P2=pop_proj[i,j])

    # save population projection results
    all_proj = pd.DataFrame(np.round(city_proj.T, decimals=0)).astype(int)
    all_proj['Year'] = years

    # move year column to first position
    cols = all_proj.columns.tolist()
    cols.remove('Year')
    cols.insert(0, 'Year')
    all_proj = all_proj[cols]

    # Save projections
    msa_proj.to_csv(output_dir / 'MSA_pop_projections.csv', index=False, sep=',')
    all_proj.to_csv(output_dir / 'pop_projections.csv', index=False, sep=',')
    print(f"Population projections complete.")

if __name__ == "__main__":
    main()