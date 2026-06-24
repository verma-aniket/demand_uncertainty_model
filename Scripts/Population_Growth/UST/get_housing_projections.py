# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# define function to apply UST
def generate_projections(param_df, pops, num_HH, use_mean=False):
    """
    Converts population projections to housing unit projections using UST parameters.
    """
    # initialize list to store projections
    HH_projs = []

    # vector to keep track of projection names
    HH_proj_names = []
    proj_num = pops.columns[1:]

    # extract fixed effect info
    logB_t_loc = param_df.loc[param_df['variable'] == "const", 'mean_coef'].values[0]
    logB_t_scale = param_df.loc[param_df['variable'] == "const", 'std_error'].values[0]

    # extract scaling exponent info
    beta_loc = param_df.loc[param_df['variable'] == "log_pop", 'mean_coef'].values[0]
    beta_scale = param_df.loc[param_df['variable'] == "log_pop", 'std_error'].values[0]

    if use_mean:
        beta_scale = 0
        logB_t_scale = 0

    # Iterate through each population projection
    for p in range(len(proj_num)):

        # sample beta once for each population projection
        beta = np.random.normal(loc=beta_loc, scale=beta_scale, size=num_HH)

        # generate housing units scenarios using UST
        for i in range(num_HH):
            # apply fixed effects
            # sample a new logB_t for each year
            logC = np.random.normal(loc=logB_t_loc, scale=logB_t_scale, size=len(pops[proj_num[p]]))

            # apply UST
            HH_projs.append(np.round(np.power(10, logC + (beta[i]) * np.log10(pops[proj_num[p]].values)), decimals=0))
            HH_proj_names.append('HH_' + str(p + 1) + '_' + str(i + 1))

    # return results as dataframe
    HH_proj_df = pd.DataFrame(HH_projs).transpose()
    HH_proj_df.columns = HH_proj_names
    HH_proj_df['Year'] = pops['Year']

    return HH_proj_df

def main():
    # route directories
    pop_proj_dir = REPO_ROOT / "data" / "Population" / "Projections"
    ust_param_dir = REPO_ROOT / "data" / "Housing" / "Params"
    output_dir = REPO_ROOT / "data" / "Housing" / "Projections"

    # Read all population projections
    pop_proj = pd.read_csv(pop_proj_dir / "pop_projections.csv")
    
    for i in range(0, 1000):
        pop_proj.rename(columns={str(i): 'Pop_' + str(i + 1)}, inplace=True)

    # Read UST parameters
    ust_param = pd.read_csv(ust_param_dir / "Single_summary.csv")

    # Set random seed
    np.random.seed(0)

    # convert population to housing units
    HH_proj_df = generate_projections(ust_param, pop_proj, num_HH=1, use_mean=True)

    # save housing unit projections
    HH_proj_df.to_csv(output_dir / "HH_projs.csv", sep=',', na_rep='', header=True, index=False)
    print(f"Housing unit projections complete.")

if __name__ == "__main__":
    main()
