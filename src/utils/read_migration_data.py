# reads Metro-to-Metro flow data (in excel format) from US Census
# source of data: https://www.census.gov/topics/population/migration/guidance/metro-to-metro-migration-flows.html
#                 https://www.census.gov/programs-surveys/metro-micro/data/datasets.html
from pathlib import Path
import pandas as pd
import numpy as np

def read_file(file_path):

    # read and format Excel file containing migration data give file location and file name
    cols_to_read = [0,1,3,16,26]
    col_names = ['j', 'i', 'S_j', 'S_i_ya', 'J_ij']

    df = pd.read_excel(file_path, usecols = cols_to_read, skiprows = 4, header=None)

    # set columns names
    df.columns = col_names

    # format data

    # Define a regex pattern for a five-digit number
    pattern = r'^\d{5}$'

    # Remove any rows with a non-MSA code
    df = df[df['j'].astype(str).str.match(pattern) & df['i'].astype(str).str.match(pattern)]

    # convert MSA codes into integers
    df[['j', 'i']] = df[['j', 'i']].astype(int)

    # Remove flows from Outside Metro Area within U.S. or Puerto Rico (i.e., with code 99999)
    df = df[~((df['j'] == 99999) | (df['i'] == 99999))]

    # add populations S_i and S_j_ya
    Si_lookup = dict(zip(df['j'], df['S_j']))
    df['S_i'] = df['i'].map(Si_lookup)

    Sj_lookup = dict(zip(df['i'], df['S_i_ya']))
    df['S_j_ya'] = df['j'].map(Sj_lookup)

    # check if mapping is complete (all MSAs accounted for)
    if np.sum(df.isna().values, axis = None) !=0:
        print('Incomplete population data for %d survey!' %get_year(file_path))

    # reorder columns
    df = df[['i','j','S_i','S_j','S_i_ya','S_j_ya','J_ij']]

    return df

def get_year(file_path):
    filename_str = file_path.name
    return int(filename_str[-9:-5])

def get_net_migration(df, msa_code):

    # sum of all inflows
    inflows = sum(df.loc[df['j'] == msa_code, 'J_ij'])

    # sum of all outflows
    outflows = sum(df.loc[df['i'] == msa_code, 'J_ij'])

    return inflows - outflows

def get_all_flows(data_path, files):

    # loop through migration data and extract population and migration flows

    c = 0 # define counter
    for fn in files:
        # read migration data
        data = read_file(Path(data_path) / fn)

        # add year to dataframe
        data['Year'] = get_year(Path(data_path) / fn)

        # save data instance and grow data frame
        if c == 0:
            data_save = data.copy()
        else:
            data_save = pd.concat([data_save, data.copy()], ignore_index=True)

        print(get_year(Path(data_path) / fn))
        c = c + 1

    return data_save

def get_net_migration_dataset(data_path, files):
    # loop through migration data and extract population and net migration of all MSAs

    # Initialize list to store data
    years = []
    codes = []
    population = []
    population_ya = []
    net_migration = []

    for fn in files:

        # read migration data
        data = read_file(Path(data_path) / fn)

        # initialize values
        system = np.unique(np.concatenate((data['i'].values, data['j'].values)))
        year = np.ones(len(system)) * get_year(Path(data_path) / fn)
        net_mig = np.zeros(len(system))
        pop = np.zeros(len(system))
        pop_ya = np.zeros(len(system))

        # iterate through MSA system and retrieve population and net migration
        for i in range(0,len(system)):
            pop[i] = data.loc[data['j'] == system[i], 'S_j'].values[0]
            pop_ya[i] = data.loc[data['j'] == system[i], 'S_j_ya'].values[0]
            net_mig[i] = get_net_migration(data, system[i])

        # append data to list
        years.extend(year)
        codes.extend(system)
        population.extend(pop)
        population_ya.extend(pop_ya)
        net_migration.extend(net_mig)

        # display status
        print(year[0])

    # save results into dataframe
    model_data = pd.DataFrame({'i': codes, 'Year': years, 'S_i': population,
                               'S_i_ya' : population_ya, 'J_net': net_migration})

    return model_data

def get_num_neighbors(df):

    # get number of in neighbors of each city i
    in_neigh = df.groupby('j')['i'].nunique().reset_index()
    in_neigh.rename(columns={'j':'i','i': 'N_in'}, inplace=True)

    # get number of out-neighbors of each city i
    out_neigh = df.groupby('i')['j'].nunique().reset_index()
    out_neigh.rename(columns={'j': 'N_out'}, inplace=True)

    # merge on i
    avg_neigh = pd.merge(in_neigh, out_neigh, on='i', how='outer')

    # compute average number of neighbors
    avg_neigh['N_i'] = (avg_neigh['N_in'] + avg_neigh['N_out'])/2

    return avg_neigh

def get_avg_neighbors(data_path, files):

    counter = 0

    for fn in files:

        # read migration data
        data = read_file(Path(data_path) / fn)

        if counter == 0:
            df = get_num_neighbors(data)
            df['Year'] = get_year(Path(data_path) / fn)
        else:
            temp = get_num_neighbors(data)
            temp['Year'] = get_year(Path(data_path) / fn)
            df = pd.concat([df, temp], ignore_index=True)

        # display status
        print(get_year(Path(data_path) / fn))
        counter = counter + 1

    return df

def read_pop_data(data_path, files):

    counter = 0

    for fn in files:

        # get year
        year = int(fn[8:12])

        if year == 2019:
            years_to_read = np.arange(2010,2020,1)

        elif year == 2024:
            years_to_read = np.arange(2020,2025,1)

        # read all data
        raw_data = pd.read_csv(Path(data_path) / fn, encoding='cp1252')

        # format data
        temp_df = format_pop_data(raw_data.copy(), years_to_read)

        # append data
        if counter == 0:
            df = temp_df.copy()
        else:
            df = pd.merge(df, temp_df.copy(), on='i', how='inner')

        counter = counter + 1

    return df

def format_pop_data(raw_data, year_range):

    # retain only MSAs
    raw_data = raw_data[raw_data['LSAD'] == 'Metropolitan Statistical Area']

    # define columns to retain
    pop_col = ['POPESTIMATE'+str(y) for y in year_range]
    birth_col = ['BIRTHS'+str(y) for y in year_range]
    death_col = ['DEATHS'+str(y) for y in year_range]
    intmig_col = ['INTERNATIONALMIG'+str(y) for y in year_range]
    resid_col = ['RESIDUAL'+str(y) for y in year_range]

    # retain desired data columns
    filter_df = raw_data[['CBSA'] + pop_col + birth_col + death_col + intmig_col + resid_col].copy()

    # rename column
    filter_df.rename(columns={'CBSA':'i'}, inplace=True)

    filter_df.reset_index(drop=True, inplace=True)

    return filter_df

def calculate_migration_fluctuations(all_flows_df):
    """
    Calculates the migration fluctuations (Diff) on a year-by-year basis
    and returns a combined DataFrame.
    """
    years = np.sort(np.unique(all_flows_df['Year']))
    counter = 0
    X_df = pd.DataFrame()

    for yr in years:
        print(f"Calculating migration fluctuations for Year: {yr}")
        # filter dataframe based on year
        data = all_flows_df[all_flows_df['Year'] == yr]

        # get all unique locations
        MSA_list = np.unique(np.concatenate([data['i'].values, data['j'].values]))

        # Create all pair combinations
        i, j = np.meshgrid(MSA_list, MSA_list)

        # Flatten and filter out self-pairings
        mask = i != j
        X_temp = pd.DataFrame({'i': i[mask], 'j': j[mask]})

        # Initialize fluctuation column
        X_temp['Diff'] = 0
        
        # Optimize row lookups by setting an index on the year's data
        indexed_data = data.set_index(['i', 'j'])

        for c in range(len(X_temp)):
            loc_i = X_temp.loc[c, 'i']
            loc_j = X_temp.loc[c, 'j']

            # Safer and faster positional lookup using a multi-index tuple
            try:
                J_ij = indexed_data.loc[(loc_i, loc_j), 'J_ij']
                J_ij = J_ij.values[0] if isinstance(J_ij, pd.Series) else J_ij
            except KeyError:
                J_ij = 0

            try:
                J_ji = indexed_data.loc[(loc_j, loc_i), 'J_ij']
                J_ji = J_ji.values[0] if isinstance(J_ji, pd.Series) else J_ji
            except KeyError:
                J_ji = 0

            X_temp.loc[c, 'Diff'] = J_ji - J_ij

            if c % 10000 == 0:
                print(f"  {c} out of {len(X_temp)} pairs complete")

        # add S_i and S_j lookups
        pop_i = dict(zip(data['i'], data['S_i']))
        pop_j = dict(zip(data['j'], data['S_j']))
        pop_lookup = {**pop_i, **pop_j}
        X_temp['S_i'] = X_temp['i'].map(pop_lookup)
        X_temp['S_j'] = X_temp['j'].map(pop_lookup)

        # add year
        X_temp['Year'] = yr

        if counter == 0:
            X_df = X_temp
        else:
            X_df = pd.concat([X_df, X_temp], ignore_index=True)

        counter += 1

    # remove all zero migration fluctuations
    X_df = X_df[X_df['Diff'] != 0]
    return X_df