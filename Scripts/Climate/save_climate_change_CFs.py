# import root libraries
import sys
from pathlib import Path
import pandas as pd

# link core folders
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# import custom functions
from src.utils.climate_functions import compute_annual_cf

def main():
    # route directories
    data_dir = REPO_ROOT / "Data" / "Climate"
    output_dir = REPO_ROOT / "Data" / "Climate"

    # Read downscaled projections for the City of Santa Cruz
    input_file = data_dir / "CalAdapt_Data_City.csv"
    cal_adapt_df = pd.read_csv(input_file)

    # Identify subset of GCMs present across all variables and experiment groups
    cal_adapt_GCM_sets = cal_adapt_df.groupby(['Variable', 'Experiment'])['GCM'].apply(set)
    cal_adapt_GCMs = set.intersection(*cal_adapt_GCM_sets)

    # Clean dataset of incomplete GCM records and unneeded columns
    cal_adapt_df = cal_adapt_df[cal_adapt_df['GCM'].isin(cal_adapt_GCMs)].reset_index(drop=True)
    cal_adapt_df.drop(columns=['Units'], inplace=True)

    # Compute daily tas (average of tasmin and tasmax)
    print("Computing derived daily mean temperatures (tas)...")
    daily_mean = (
        cal_adapt_df[cal_adapt_df['Variable'].isin(['tasmin', 'tasmax'])]
        .groupby(['Year', 'Month', 'Day', 'GCM', 'Experiment'])['Value']
        .mean()
        .reset_index()
    )
    daily_mean['Variable'] = 'tas'

    # Append long-form variables matrix
    cal_adapt_df = pd.concat([cal_adapt_df, daily_mean], ignore_index=True)

    # aggregate daily to monthly
    print("Aggregating climate variables to monthly timescales...")
    monthly_tas = (
        cal_adapt_df[cal_adapt_df['Variable'] == 'tas']
        .groupby(['Year', 'Month', 'GCM', 'Experiment'])['Value']
        .mean()
        .reset_index()
    )
    monthly_tas['Variable'] = 'tas'

    monthly_tasmax = (
        cal_adapt_df[cal_adapt_df['Variable'] == 'tasmax']
        .groupby(['Year', 'Month', 'GCM', 'Experiment'])['Value']
        .mean()
        .reset_index()
    )
    monthly_tasmax['Variable'] = 'tasmax'

    monthly_tasmin = (
        cal_adapt_df[cal_adapt_df['Variable'] == 'tasmin']
        .groupby(['Year', 'Month', 'GCM', 'Experiment'])['Value']
        .mean()
        .reset_index()
    )
    monthly_tasmin['Variable'] = 'tasmin'

    monthly_pr = (
        cal_adapt_df[cal_adapt_df['Variable'] == 'pr']
        .groupby(['Year', 'Month', 'GCM', 'Experiment'])['Value']
        .sum()
        .reset_index()
    )
    monthly_pr['Variable'] = 'pr'

    # Unify dimensions 
    cal_adapt_df_monthly = pd.concat([monthly_tas, monthly_tasmax, monthly_tasmin, monthly_pr], ignore_index=True)

    # establish historical and future anchors
    print("Constructing climate baseline timeline windows...")
    # Extract 1970 to 2000 historical reference data
    hist_data = cal_adapt_df_monthly[(cal_adapt_df_monthly['Year'] >= 1970) & (cal_adapt_df_monthly['Year'] <= 2000)].copy()
    hist_data.drop(columns=['Year'], inplace=True)

    # Compute static baseline monthly historical average
    hist_df = hist_data.groupby(by=['Month', 'GCM', 'Experiment', 'Variable']).mean().reset_index(drop=False)
    hist_df.drop(columns=['Experiment'], inplace=True)

    # Extract 2015 to 2080 prediction data bounds
    future_df = cal_adapt_df_monthly[(cal_adapt_df_monthly['Year'] >= 2015) & (cal_adapt_df_monthly['Year'] <= 2080)].copy()

    # 7. CALCULATE DYNAMIC CHANGE FACTORS (CFs) VIA SRC MODULE
    print("Calculating and interpolating annual Change Factors (CFs)...")
    result = compute_annual_cf(
        hist_df,
        future_df,
        hist_year=2025,
        target_years=2050,
        window=30
    )

    # 8. EXPORT FINALIZED CHANGE FACTORS BACK TO ENVIRONMENT
    output_file = output_dir / "Cal_Adapt_CF_Data.csv"
    result.to_csv(output_file, index=False)
    print(f"Change Factor metrics computed.")

if __name__ == "__main__":
    main()