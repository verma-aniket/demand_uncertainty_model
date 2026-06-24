# src/models/build_model.py
import os
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
from scipy.optimize import minimize
import powerlaw

def get_eta_data(pop_data, year_range):
    eta_df = pop_data[['i']].copy()
    for y in year_range:
        nat_growth = pop_data['BIRTHS'+str(y)] - pop_data['DEATHS'+str(y)] + pop_data['INTERNATIONALMIG'+str(y)] + pop_data['RESIDUAL'+str(y)]
        eta_df[str(y)] = nat_growth/pop_data['POPESTIMATE'+str(y)]
    return np.mean(eta_df[eta_df.columns[1:]], axis=1).values

def get_eta_params(pop_data, year):
    eta_data = get_eta_data(pop_data, np.arange(year-4, year+1, 1))
    mu = np.mean(eta_data)
    sigma = np.std(eta_data)
    return mu, sigma

def r2_formula(params, S_i, S_j, I_ij):
    I_0, v = params
    y_pred = I_0 * (S_i**v * S_j**(v-1))
    y = I_ij
    ss_res = np.sum((np.log(y) - np.log(y_pred))**2)
    ss_tot = np.sum((np.log(y) - np.mean(np.log(y)))**2)
    return -(1 - (ss_res / ss_tot))

def get_min_flow_model_params(data, year, init_params):
    data['I_ij'] = data['J_ij']/data['S_j']
    res = minimize(r2_formula, init_params, args=(data['S_i'].values, data['S_j'].values, data['I_ij'].values),
                   method='L-BFGS-B', bounds=[(1e-10, None), (0, 2)])
    I_0, v = res.x
    return I_0, v

def get_gamma(pop_data, n_data, year):
    merge_df = pd.merge(pop_data, n_data, on=['i'], how='left')
    merge_df['log_N'] = np.log(merge_df['N_i'])
    merge_df['log_S'] = np.log(merge_df['S_i'])

    X = merge_df[['log_S']]
    X = sm.add_constant(X)
    y = merge_df['log_N']
    model = sm.OLS(y, X).fit()
    
    gamma = model.params['log_S']
    return gamma

def get_beta(data, year):
    data['J_net_abs'] = np.abs(data['J_net'])
    data['log_J_net_abs'] = np.log(data['J_net_abs'])
    data['log_S'] = np.log(data['S_i'])

    X = data[['log_S']]
    X = sm.add_constant(X)
    y = data['log_J_net_abs']
    model = sm.OLS(y, X).fit()
    
    beta = model.params['log_S']
    return beta

def check_heavy_tails(data, year, v):
    data['X_ij'] = data['Diff']/np.power(data['S_i'], v)
    X_ij = data.loc[data['X_ij'] > 0, 'X_ij'].values
    fit = powerlaw.Fit(X_ij, xmin=1, verbose=False)
    alpha = fit.power_law.alpha - 1
    return alpha

def get_zeta_params(data, year, beta):
    data['zeta_i'] = data['J_net']/np.power(data['S_i'], beta)
    alpha, beta_param, loc, scale = ss.levy_stable.fit(data=data['zeta_i'].values)
    return alpha, beta_param, loc, scale