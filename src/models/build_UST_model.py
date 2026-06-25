from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as t_dist

def balance_data(df, feature_col, pop_col='Population', time_col='Year', firm_col='City'):
    """
    Cleans and balances the panel dataset by removing zeros/negative values 
    and ensuring only cross-sections (cities) with complete temporal records are retained.
    """
    # Step 1 - Remove occurrences of values less than or equal to zeros
    df.drop(df[df[pop_col] <= 0].index, inplace=True)
    df.drop(df[df[feature_col] <= 0].index, inplace=True)

    # Step 2 - Remove Cities with incomplete data
    years = df[time_col].unique()
    n_years = len(years)

    # Count observations per city
    counts = df.groupby(firm_col)[time_col].nunique().reset_index()
    full_cities = counts[counts[time_col] == n_years][firm_col]

    # Keep only cities with all years present
    df = df[df[firm_col].isin(full_cities)].copy()

    return df

def fama_macbeth(df, feature_col, output_dir, pop_col='Population', time_col='Year', firm_col='City'):
    """
    Executes a Fama-MacBeth two-step panel regression strategy.
    Calculates year-by-year cross-sectional parameters and saves aggregated results 
    to a fully decoupled, dynamic output directory.
    """
    # Ensure output folder path object exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute Logs
    df['log_pop'] = np.log10(df[pop_col])
    df['log_unit'] = np.log10(df[feature_col])

    # Initialize variables
    time_list = sorted(df[time_col].unique())
    beta_list = []
    details = []

    # Step 1: Cross-Sectional OLS regressions for each period
    for t in time_list:
        data_t = df[df[time_col] == t]
        X = sm.add_constant(data_t['log_pop'])
        y = data_t['log_unit']

        model = sm.OLS(y, X).fit()
        params = model.params
        ses = model.bse
        t_stat = model.tvalues
        pval = model.pvalues

        # Store betas for Fama-MacBeth step
        beta_list.append(params)

        # Store detailed cross-sectional results
        row = pd.DataFrame({
            time_col: [t] * len(params),
            'variable': params.index,
            'coef': params.values,
            'std_error': ses.values,
            't_stat': t_stat.values,
            'p_val': pval.values
        })
        details.append(row)

    # Combine results
    betas_df = pd.DataFrame(beta_list, index=time_list)
    details_df = pd.concat(details, ignore_index=True)

    # Step 2: Compute Fama–MacBeth averages, temporal SEs, t-stats, and p-values
    mean_betas = betas_df.mean()
    T = len(betas_df)
    se_betas = betas_df.std(ddof=1) / np.sqrt(T)
    t_stats = mean_betas / se_betas
    
    # Compute degrees of freedom
    N_obs = len(df)       # Number of observations
    N_cov = 1             # Number of covariates
    N_t = len(time_list)  # Number of time periods
    deg_free = N_obs - N_cov - N_t - 1

    # Two-tailed t-test: intercept term
    t_stat = t_stats.values[0]
    p_value_int = 2 * t_dist.cdf(-abs(t_stat), df=deg_free)

    # Two-tailed t-test: scaling exponent term
    t_stat = t_stats.values[1]
    p_value_exp = 2 * t_dist.cdf(-abs(t_stat), df=deg_free)

    summary_df = pd.DataFrame({
        'variable': mean_betas.index,
        'mean_coef': mean_betas.values,
        'std_error': se_betas.values,
        't_stat': t_stats.values,
        'p_val': np.array([p_value_int, p_value_exp]),
        'n_periods': T
    })

    # Save results out to the relative paths specified by the caller script
    summary_df.to_csv(output_dir / f"{feature_col}_summary.csv", index=False)
    details_df.to_csv(output_dir / f"{feature_col}_details.csv", index=False)

    return summary_df, details_df

def compute_R2(y_i, y_hat):
    """Calculates coefficient of determination (R2)."""
    SS_res = np.sum((y_i - y_hat) ** 2)
    SS_tot = np.sum((y_i - np.mean(y_hat)) ** 2) 
    return 1 - SS_res / SS_tot

def get_R2_summary(df, beta, logC, feature_col, pop_col='Population'):
    """Computes cross-sectional log-space model predictive R2 fitment score."""
    # Compute Logs
    df['log_pop'] = np.log10(df[pop_col])
    df['log_unit'] = np.log10(df[feature_col])

    # prediction
    df['log_unit_pred'] = beta * df['log_pop'] + logC

    # compute R2
    r2 = compute_R2(df['log_unit'].values, df['log_unit_pred'].values)

    return r2