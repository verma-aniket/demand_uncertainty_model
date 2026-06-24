# load packages
library(DBI)
library(RSQLite)
library(data.table)
library(stats4)
library(broom)
library(bbmle)

seeds <- c(27, 19, 16, 7, 61, 96, 42, 142, 928, 155)

# set working directory
script_dir <- dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(dirname(dirname(script_dir))) # jump up two levels

### BUILD SQL QUERY
# columns to read
cols_to_read <- paste("totwuse", "rmon", "ryr", "bill_length", "rate_str_change",
                      "location", "account", "FIPS",
                      "curtail",
                      "p_1_all", "p_2_all", "p_3_all", "p_4_all", "p_5_all",
                      "tax_val_2", "MHI",
                      "bathrooms_F", "bathrooms_H", "bedrooms", "fireplaces", "main_area",
                      "parcel_size", "pool", sep = ",")
cols_str <- paste(cols_to_read, collapse = ", ")

# NOTE: all tax_val_2 values are 2021 USD (tax_year_2 are all equal to 2021)

# define residence classes to read
res_classes <- c("020-SINGLE RESIDENCE",
                 "027-TOWNHOUSE",
                 "024-SFR W/ SECONDARY USE",
                 "028-SFR + SECOND UNIT",
                 "029-SFR + GRANNY UNIT",
                 "030-SINGLE DUPLEX",
                 "031-TWO SFRS/1 APN",
                 "060-HOMESITE/< 1 ACRE",
                 "061-HOMESITE/1-4.9 ACRES",
                 "062-HOMESITE/5-19.9 ACRE",
                 "063-HOMESITE/20-49.9 ACRES",
                 "064-HOMESITE/50-99.9 ACRES",
                 "025-AFFORDABLE HOUSING",
                 "-0-ND 020-SINGLE RESIDENCE",
                 "5 020-SINGLE RESIDENCE")
res_class_str <- paste0("'", res_classes, "'", collapse = ",")

# define query statement
query <- paste0("SELECT ", cols_str, "
                 FROM sf_data
                 WHERE drought_surcharge >= 0 AND
                 main_area > 0 AND
                 parcel_size > main_area AND
                 tax_val_2 > 0 AND
                 MHI != '' AND
                 quality = 1 AND
                 Res_Class IN (", res_class_str, ")")

# read data from SQLite3 database
# create database connection
con <- dbConnect(RSQLite::SQLite(), ":memory:",
                 dbname = "Data/Billing_Data/DCC_DB.db")

# send query and save results into a dataframe
dcc_dt <- as.data.table(dbGetQuery(con, query))

# Disconnect from the database
dbDisconnect(con)

### PREPROCESSING DATA

# rename columns
old_names <- c("p_1_all", "p_2_all", "p_3_all", "p_4_all", "p_5_all",
               "tax_val_2", "totwuse")
new_names <- c("p_1", "p_2", "p_3", "p_4", "p_5",
               "tax_val", "y")
setnames(dcc_dt, old = old_names, new = new_names)

# rename month and year safely:
setnames(dcc_dt, old = c("rmon", "ryr"), new = c("month", "year"), skip_absent = TRUE)
dcc_dt[, `:=`(
  year = as.integer(.SD$year),
  month = as.integer(.SD$month)
)]

# Remove bills with less than X months of data at a single location for a unique account
dcc_dt[, count := .N, by = .(account, location)]
if (max(dcc_dt[, count]) > 12) {
  dcc_dt <- dcc_dt[count >= 12]
}

# standardize water use data to monthly
dcc_dt[, w := y / bill_length * 30.4]

# add sine curve to represent seasonality (code from Jenny)
amp <- -1 * sqrt(1^2 + 1^2)  # Amplitude = -sqrt(2)
phi <- atan2(1, 1)         # Phase offset = pi/4
dcc_dt[, sine := amp * sin(2 * pi * month / 12 + phi)]

# compute lawn size
dcc_dt[, lawn := parcel_size - main_area]

# add drought dummy variables
dcc_dt[, `:=`(
  in_drought = fifelse(
    # Drought 1: Jan 2014 – Apr 2017
    ((year == 2014 & month >= 1) |
     (year > 2014 & year < 2017) |
     (year == 2017 & month < 5)) |
    # Drought 2: July 2021 – onward
    ((year == 2021 & month >= 7) | (year > 2021)),
    1, 0
  ),

  post_drought = fifelse(
    # Post-drought period: May 2017 – June 2021
    (
      (year == 2017 & month >= 5) |
      (year > 2017 & year < 2021) |
      (year == 2021 & month < 7)
    ),
    1, 0
  )
)]

# add curtailment dummy variable
dcc_dt[, `:=`(
  curtail_high = fifelse(curtail == 0.25, 1, 0)
)]

# add rate structure tier
dcc_dt[, num_tier := fifelse(rate_str_change == 1, 4, 5)]

# add corrected weather data
weather_data <- fread("Data/Simulator/Inputs/weather_data.csv")

# merge weather data
dcc_dt[weather_data, on = .(year, month), `:=`(
  temp = i.max_temp,
  precip = i.precip,
  precip_lag1 = i.precip_lag1
)]

# Part A: Inflation Adjustment

# Step A1: water price inflation adjustment

# read statewide monthly CPI-U data
cpi_monthly <- fread("Data/Economics/cpi_monthly.csv")

# compute factor to convert all water prices to December 2021 USD
dec2021 <- cpi_monthly[year == 2021 & month == 12]$CPI
cpi_monthly$factor <- dec2021 / cpi_monthly$CPI

# merge factor on year and month
dcc_dt[cpi_monthly, on = .(year, month), `:=`(
  price_factor = i.factor
)]

# Apply water price inflation adjustment
dcc_dt[, `:=`(
  p_1 = p_1 * price_factor,
  p_2 = p_2 * price_factor,
  p_3 = p_3 * price_factor,
  p_4 = p_4 * price_factor,
  p_5 = p_5 * price_factor
)]

# Step A2: income and house tax value inflation adjustment

# read statewide annual CPI-U data
cpi_annual <- fread("Data/Economics/cpi_annual.csv")

# compute factor to convert all water prices to December 2021 USD
cpi2021 <- cpi_annual[year == 2021]$CPI
cpi_annual$factor <- cpi2021 / cpi_annual$CPI

# merge factor on year
dcc_dt[cpi_annual, on = .(year), `:=`(
  inc_factor = i.factor
)]

# Apply income inflation adjustment (house tax value already all in 2021 USD)
dcc_dt[, `:=`(
  MHI = MHI * inc_factor
)]

# compute logs of certain predictors
dcc_dt[, `:=`(
  p_1           = log(p_1),
  p_2           = log(p_2),
  p_3           = log(p_3),
  p_4           = log(p_4),
  p_5           = log(p_5),
  log_MHI       = log(MHI),
  log_tax_val   = log(tax_val),
  log_main_area = log(main_area),
  log_lawn      = log(lawn),
  log_w         = log(w)
)]

# compute house tax value to block-group based standardized anomalies
dcc_dt[, `:=`(
  log_tax_val_mean = mean(log_tax_val),
  log_tax_val_sd   = sd(log_tax_val)
), by = FIPS]
dcc_dt[, tax_val_Z := (log_tax_val - log_tax_val_mean) / log_tax_val_sd]


# remove columns that are no longer needed
cols_to_remove <- c("y", "w", "month", "year", "bill_length", "rate_str_change",
                    "location", "account", "FIPS",
                    "curtail",
                    "tax_val", "MHI",
                    "main_area", "parcel_size",
                    "lawn", "count",
                    "log_tax_val", "log_tax_val_mean", "log_tax_val_sd",
                    "price_factor", "inc_factor")
dcc_dt[, (cols_to_remove) := NULL]

# remove rows containting any NaN values
dcc_dt <- na.omit(dcc_dt)

# define water rate structure
# 5-Tier Structure
tiers_5 <- c(-Inf, 4, 9, 14, 18, Inf)
categories_5 <- c(1, 2, 3, 4, 5)
# 4-Tier Structure
tiers_4 <- c(-Inf, 5, 7, 9, Inf)
categories_4 <- c(1, 2, 3, 4)

### DCC Model

# read DCC MLE functions
source("Scripts/DCC_Model/DCC_LL_Function.R")

# define random seed for reproducibility
set.seed(27)

# PART A: Set up inputs for MLE function

# Step 1: Create predictor dataframe with initial guesses
# create dataframe with columns: predictor name and random initial guess value
# predictor name must match column names in dcc_dt
ini_list <- as.data.frame(do.call(rbind, list(
  list("price",           runif(1, -0.2,  0)),
  list("intercept",       runif(1, -3,   -1)),
  list("sine",            runif(1,  0,    0.1)),
  list("temp",            runif(1,  0,    0.01)),
  list("precip",          runif(1, -1e-3, 0)),
  list("precip_lag1",     runif(1, -1e-4, 0)),
  list("spi",             runif(1, -1e-2, 0)),
  list("in_drought",      runif(1, -1,    0)),
  list("post_drought",    runif(1, -1,    0)),
  list("curtail_high",    runif(1, -1,    0)),
  list("log_tax_val",     runif(1,  0,    0.1)),
  list("log_MHI",         runif(1,  0,    0.01)),
  list("bathrooms_F",     runif(1,  0,    0.01)),
  list("bathrooms_H",     runif(1,  0,    0.01)),
  list("bedrooms",        runif(1,  0,    0.1)),
  list("fireplaces",      runif(1, -0.05, 0)),
  list("log_main_area",   runif(1,  0,    1)),
  list("log_lawn",        runif(1,  0,    0.1)),
  list("pool",            runif(1,  0,    1))
)))

# Assign column names
names(ini_list) <- c("name", "init")

# Convert init column to numeric
ini_list$init <- as.numeric(ini_list$init)

# store sigmas separately
# Inverse-Gamma distribution parameters
alpha <- 0.01
scale <- 0.01
rate <- 1 / scale  # convert scale to rate

# Sample two draws from the Inverse-Gamma distribution
# update this later!!
sigma_draws <- c(0.1, 0.5)

# define input names
pred_names <- as.character(ini_list$name)
input_names <- c(paste0("beta_", pred_names), "sigma_eta", "sigma_eps")

# define full initial guess vector
init_vec <- c(ini_list$init, sigma_draws)

# Step 2: Define water rate structure parameters
# Define water rate price column names
price_cols_5 <- c("p_1", "p_2", "p_3", "p_4", "p_5")
price_cols_4 <- c("p_1", "p_2", "p_3", "p_4")

# Define 5- and 4-tier boundaries as numeric vectors
tier_5_bounds <- log(c(4.0, 9.0, 14.0, 18.0))
tier_4_bounds <- log(c(5.0, 7.0, 9.0))

# PART B: Extract data needed for MLE function

# Step 1. Extract predictor names, excluding 'price' and 'intercept' which will be handled separately
pred_names <- pred_names[-c(1, 2)]

# Step 2. Define masks for tier-specific indexing
mask_5 <- dcc_dt$num_tier == 5
mask_4 <- dcc_dt$num_tier == 4

# Step 3. Create tier-specific matrices for predictors, prices, and target variable

# 5-tier customers: all price columns
pred_mat_5  <- as.matrix(dcc_dt[mask_5, ..pred_names])
price_mat_5 <- as.matrix(dcc_dt[mask_5, ..price_cols_5])
target_5 <- dcc_dt[mask_5, log_w]

# 4-tier customers: only first 4 price columns
pred_mat_4  <- as.matrix(dcc_dt[mask_4, ..pred_names])
price_mat_4 <- as.matrix(dcc_dt[mask_4, ..price_cols_4])
target_4 <- dcc_dt[mask_4, log_w]

# wrap data into lists
pred_list <- list(pred_mat_5, pred_mat_4)
price_list <- list(price_mat_5, price_mat_4)
target_list <- list(target_5, target_4)
bounds_list <- list(tier_5_bounds, tier_4_bounds)

# PART C: Run MLE function

# Maximize log-likelihood function
start_time <- Sys.time()
# set_up data inputs
data_inputs <- list(pred_list = pred_list,
                    price_list = price_list,
                    target_list = target_list,
                    bounds_list = bounds_list)
DCC_fit <- mle2(LL_main,  start = as.list(setNames(init_vec, input_names)),
                data = data_inputs)
end_time <- Sys.time()
# print(end_time - start_time)
# summary(DCC_fit)

# Save DCC results
saveRDS(DCC_fit, "Data/DCC_Model/dccfit_results.rds")