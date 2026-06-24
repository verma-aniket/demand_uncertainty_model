import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, qmc

# 1. LINK CORE FOLDERS
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

def generate_dcc_samples(mu, sigma, coeff_names, n_samples=50, n_components=5, k_inflation=1.5, seed=27, display=True):
    """
    Generates weighted model coefficient samples using PCA-LHS-IS.
    
    Parameters:
    - mu: mean vector of DCC MVN
    - sigma: covariance matrix of DCC MVN
    - n_samples: Number of parameter sets to generate.
    - n_components: Number of Principal Components to capture (m).
    - k_inflation: Covariance inflation factor for Importance Sampling.
    - seed: Random seed for reproducibility.
    """
    # Step 1: Perform PCA (Eigen-decomposition)
    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    
    # Sort descending
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select subset of components
    V_m = eigenvectors[:, :n_components]
    L_m = eigenvalues[:n_components]
    
    # Step 2: Latin Hypercube Sampling in Latent Space
    sampler = qmc.LatinHypercube(d=n_components, seed=seed)
    u_samples = sampler.random(n=n_samples)
    
    # Step 3: Transform to Inflated Normal Distribution in Latent Space (Proposal q)
    z = np.zeros((n_samples, n_components))
    for i in range(n_components):
        std_dev = np.sqrt(k_inflation * L_m[i])
        z[:, i] = norm.ppf(u_samples[:, i], loc=0, scale=std_dev)
        
    # Step 4: Project back to 20-D Coefficient Space
    theta_samples = mu + (V_m @ z.T).T
    
    # Step 5: Calculate Importance Weights (p(theta) vs q(z))
    p_dist = multivariate_normal(mean=mu, cov=sigma, allow_singular=True)
    log_p = p_dist.logpdf(theta_samples)
    
    q_dist = multivariate_normal(mean=np.zeros(n_components), cov=np.diag(k_inflation * L_m))
    log_q = q_dist.logpdf(z)
    
    weights = np.exp(log_p - log_q)
    weights_norm = weights / np.mean(weights)

    # Display diagnostics details
    if display:
        explained_var = np.sum(L_m) / np.sum(eigenvalues)
        print(f"\nCaptured {explained_var:.2%} of total variance using {n_components} components.")
        print('Top 3 contributing terms associated with each component:')
        for x in range(n_components):
            pcx_recipe = pd.Series(eigenvectors[:, x], index=coeff_names)
            print(f"\nComponent {x+1}:")
            print(pcx_recipe.abs().sort_values(ascending=False).head(3))
            
        ess = np.sum(weights_norm) ** 2 / np.sum(weights_norm ** 2)
        print(f"\nEffective Sample Size (ESS) = {ess:.3f}")
        print(f"ESS/N Ratio = {100 * (ess / n_samples):.2f}%")
    
    return theta_samples, weights_norm


def main():
    # 2. ROUTE ENVIROMENT PATHS DYNAMICALLY 
    param_dir = REPO_ROOT / "Data" / "DCC_Model"
    output_dir = REPO_ROOT / "Data" / "Simulator" / "Scenarios"
    
    coeff_file = param_dir / "DCC_fit.csv"
    vcov_file = param_dir / "DCC_vcov.csv"
    output_file = output_dir / "DCC_scenarios.csv"

    # 3. CONFIGURE STRUCTURAL SEEDS AND TUNING METRICS
    base_seed = 42
    N_s = 75                 # Number of samples
    N_c = 5                  # Number of principal components
    k = 2.2                  # Extremes tail over-sampling multiplier index

    # 4. PARSE EMPIRICAL ARRAYS AND COVARIANCE ARRAYS
    dcc_coeff_df = pd.read_csv(coeff_file)
    n_terms = dcc_coeff_df.shape[0]

    # Parse vcov parsing bounds cleanly without hardcoded index ranges
    dcc_coeff_vcov = pd.read_csv(vcov_file, header=None, skiprows=1, usecols=range(1, n_terms + 1))
    
    dcc_mu = dcc_coeff_df['estimate'].to_numpy()
    dcc_sigma = dcc_coeff_vcov.to_numpy()
    dcc_names = dcc_coeff_df['term'].values

    # 5. EXECUTE PCA STRATIFIED IMPORTANCE SAMPLING
    print("Initiating Latin Hypercube Importance Sampling sequence...")
    samples, weights = generate_dcc_samples(
        dcc_mu, dcc_sigma, dcc_names, 
        n_samples=N_s, n_components=N_c, k_inflation=k, 
        seed=base_seed, display=True
    )

    # 6. EXPORT FINALIZED DRIFT MATRIX COEFFICIENTS
    sample_df = pd.DataFrame(data=samples, columns=dcc_names)
    sample_df['weights'] = weights

    sample_df.to_csv(output_file, index=False)
    print("DCC model parameter scenarios saved.")


if __name__ == "__main__":
    main()