# Log-likelihood function for DCC MLE estimation

# Global constants
CONST <- 1 / sqrt(2 * pi)

# Main log-likelihood function
LL_main <- function(beta_price = NULL, beta_intercept = NULL,
                    beta_sine = NULL, beta_temp = NULL, beta_precip = NULL, beta_precip_lag1 = NULL,
                    beta_in_drought = NULL, beta_post_drought = NULL,
                    beta_curtail = NULL, beta_curtail_high = NULL,
                    beta_surcharge = NULL, beta_surcharge_dummy = NULL,
                    beta_log_tax_val = NULL, beta_log_MHI = NULL,
                    beta_bathrooms_F = NULL, beta_bathrooms_H = NULL, beta_bedrooms = NULL, beta_fireplaces = NULL,
                    beta_log_main_area = NULL, beta_log_lawn = NULL, beta_pool = NULL,
                    sigma_eta, sigma_eps,
                    pred_list, price_list, target_list, bounds_list) {

  # Unpack beta values (same order as function inputs)
  beta_vec <- c(beta_price, beta_intercept,
                beta_sine, beta_temp, beta_precip, beta_precip_lag1,
                beta_in_drought, beta_post_drought,
                beta_curtail, beta_curtail_high,
                beta_surcharge, beta_surcharge_dummy,
                beta_log_tax_val, beta_log_MHI,
                beta_bathrooms_F, beta_bathrooms_H, beta_bedrooms, beta_fireplaces,
                beta_log_main_area, beta_log_lawn, beta_pool)

  # drop NULL values from beta vector
  beta_vec <- beta_vec[!sapply(beta_vec, is.null)]

  # convert to numeric
  beta_vec <- as.numeric(beta_vec)

  # split observation data based on rate structure change
  LL_5_tier <- LL_func(betas = beta_vec, sigma_n = sigma_eta, sigma_e = sigma_eps,
                       pred_mat = pred_list[[1]], price_mat = price_list[[1]],
                       target = target_list[[1]], bounds = bounds_list[[1]])
  if (nrow(pred_list[[2]]) == 0) {
    LL_4_tier <- 0
  } else {
    LL_4_tier <- LL_func(betas = beta_vec, sigma_n = sigma_eta, sigma_e = sigma_eps,
                         pred_mat = pred_list[[2]], price_mat = price_list[[2]],
                         target = target_list[[2]], bounds = bounds_list[[2]])
  }
  # LL_4_tier <- LL_func(betas = beta_vec, sigma_n = sigma_eta, sigma_e = sigma_eps,
  #                      pred_mat = pred_list[[2]], price_mat = price_list[[2]],
  #                      target = target_list[[2]], bounds = bounds_list[[2]])

  # Compute total log-likelihood
  LL_total <- LL_5_tier + LL_4_tier
  LL_total
}

LL_func <- function(betas, sigma_n, sigma_e, pred_mat, price_mat, target, bounds) {
  # Placeholder for the actual log-likelihood computation
  # This function should compute the log-likelihood based on the model

  # Extract constants
  sigma_v <- sqrt(sigma_n^2 + sigma_e^2)
  rho <- sigma_n / sigma_v
  sqrt_one_minus_rho2 <- sqrt(1 - rho^2)
  n_obs <- length(target)
  K <- ncol(price_mat)

  # Extract beta price
  beta_price <- betas[1]
  betas <- betas[-1]

  # Initialize lists
  s_k <- vector("list", K)
  u_k <- vector("list", K)
  t_k <- vector("list", K)
  r_k <- vector("list", K)
  m_k <- vector("list", K)
  n_k <- vector("list", K)

  # compute shared term once
  shared_term <- betas[1] + as.numeric(pred_mat %*% betas[-1])

  # loop over each tier
  for (k in 1:K) {

    # Compute ln_w_star for this block
    ln_w_star_k <- shared_term + beta_price * price_mat[, k]

    # s_k for K blocks
    s_k[[k]] <- (target - ln_w_star_k) / sigma_v

    # u_k, t_k, r_k for K-1 boundaries
    if (k <= K - 1) {
      u_k[[k]] <- (target - bounds[k]) / sigma_e
      t_k[[k]] <- (bounds[k] - ln_w_star_k) / sigma_n
      r_k[[k]] <- (t_k[[k]] - rho * s_k[[k]]) / sqrt_one_minus_rho2
    }

    if (k > 1) {
      m_k[[k - 1]] <- (bounds[k - 1] - ln_w_star_k) / sigma_n
      n_k[[k - 1]] <- (m_k[[k - 1]] - rho * s_k[[k]]) / sqrt_one_minus_rho2
    }
  }

 # Compute ST
  ST <- numeric(n_obs)
  for (k in 1:(K - 1)) {

    # Term 1: consumption in block k
    if (k == 1) {
      ST <- ST + CONST / sigma_v * exp(-0.5 * s_k[[k]]^2) * pnorm(r_k[[k]])
    } else {
      ST <- ST + CONST / sigma_v * exp(-0.5 * s_k[[k]]^2) * (pnorm(r_k[[k]]) - pnorm(n_k[[k - 1]]))
    }

    # Term 1 for the last block K
    if (k == (K - 1)) {
      ST <- ST + CONST / sigma_v * exp(-0.5 * s_k[[k + 1]]^2) * (1 - pnorm(n_k[[k]]))
    }

    # Term 2: consumption at kink between block k and k+1
    ST <- ST + CONST / sigma_e * exp(-0.5 * u_k[[k]]^2) * (pnorm(m_k[[k]]) - pnorm(t_k[[k]]))
  }

  # Avoid log of zero or negative values
  ST[ST <= 0] <- 1e-20
  LL_val <- -sum(log(ST))
  LL_val
}