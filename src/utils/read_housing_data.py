from pathlib import Path
import pandas as pd
import numpy as np

def get_year(file_path):
    filename_str = Path(file_path).name
    return int(filename_str[-9:-5])

def format_data(df):
    # convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

    # add month column
    df['Month'] = df['Date'].dt.month

    # add Year column
    df['Year'] = df['Date'].dt.year

    # filter unwanted data
    df = df[df['County'] != 'California'] # remove state totals
    df = df[df['City'] != 'County Total'] # remove county totals
    df = df[df['City'] != 'Incorporated'] # remove county incorporated/non-res
    df = df[df['City'] != 'Balance of County'] # remove county balances
    df = df[df['Month'] != 4] # remove starting/base 2000 population

    # remove unwanted columns
    df.drop(columns=['Date', 'Month'], inplace=True)

    return df

def read_file(file_path):
    # Ensure file_path is a formal Path object
    file_path = Path(file_path)
    data_year = get_year(file_path)

    # read and format Excel file containing population and housing estimates data given file path
    if data_year <= 2001:
        cols_to_read = [0, 1, 2, 3, 7, 8, 11]
        df = pd.read_excel(file_path, sheet_name='RawData', usecols=cols_to_read, skiprows=2, header=None)
        df = df[~(df == 'x').any(axis=1)]
        col_names = ['County', 'City', 'Date', 'Population', 'Single', 'Multi', 'Vacancy_Rate']
        df.columns = col_names

    elif data_year <= 2010:
        cols_to_read = [0, 1, 2, 3, 7, 8, 12]
        df = pd.read_excel(file_path, sheet_name='RawData', usecols=cols_to_read, skiprows=2, header=None)
        df = df[~(df == 'x').any(axis=1)]
        col_names = ['County', 'City', 'Date', 'Population', 'Single', 'Multi', 'Vacancy_Rate']
        df.columns = col_names

    else:
        cols_to_read = [0, 1, 2, 3, 7, 8, 9, 10, 13]
        df = pd.read_excel(file_path, sheet_name='RawData', usecols=cols_to_read, skiprows=2, header=None)
        df = df[~(df == 'x').any(axis=1)]

        # add single-family housing columns and remove original values
        df['Single'] = df[7] + df[8]
        df.drop(columns=[7, 8], inplace=True)

        # add multi-family housing columns and remove original values
        df['Multi'] = df[9] + df[10]
        df.drop(columns=[9, 10], inplace=True)

        # set columns names
        col_names = ['County', 'City', 'Date', 'Population', 'Vacancy_Rate', 'Single', 'Multi']
        df.columns = col_names

        # reorder columns
        df = df[['County', 'City', 'Date', 'Population', 'Single', 'Multi', 'Vacancy_Rate']]

    # format data
    df = format_data(df)
    return df

def get_housing_data(data_path, files):
    c = 0 
    data_save = pd.DataFrame()

    for fn in files:
        full_file_path = Path(data_path) / fn
        data = read_file(full_file_path)

        if c == 0:
            data_save = data.copy()
        else:
            data_save = pd.concat([data_save, data.copy()], ignore_index=True)

        c = c + 1

    # reorder columns
    data_save = data_save[['County', 'City', 'Year', 'Population', 'Single', 'Multi', 'Vacancy_Rate']]
    # sort data
    data_save.sort_values(by=['County', 'City', 'Year'], inplace=True)

    # drop vacancy rate
    data_save.drop(columns=['Vacancy_Rate'], inplace=True)

    return data_save