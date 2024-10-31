'''
Authors: Antra, Harman, Daniel
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer


########################## HELPER FUNCTIONS ###################################
'''
Fill missing values of col columns of df with the mean of those columns.
Return the transformed columns.
'''
def _impute_by_mean(df: pd.DataFrame, cols):
    mean_imputer = SimpleImputer(strategy='mean')
    return mean_imputer.fit_transform(df[cols])
'''
Fill missing values of col columns of df with the mode of those columns.
Return the transformed columns.
'''
def _impute_by_mode(df: pd.DataFrame, cols):
    mode_imputer = SimpleImputer(strategy='most_frequent')
    return mode_imputer.fit_transform(df[cols])

'''
Fill missing values of col columns of df with the median of those columns.
Return the transformed columns.
'''
def _impute_by_median(df: pd.DataFrame, cols):
    median_imputer = SimpleImputer(strategy='median')
    return median_imputer.fit_transform(df[cols])

'''
Fill missing values of col columns of df with a constant time value. This was
the only way we could think of to impute time columns.
Return the transformed columns.
'''
def _impute_by_constant_time(df: pd.DataFrame, cols):
    constant_time = pd.Timestamp('00:00:00')
    return df[cols].fillna(constant_time)

def _impute_by_constant(df: pd.DataFrame, cols, constant):
    return df[cols].fillna(constant)

'''
Transform the int/float values of Z_CARD_VALID into datetime. 
Returns the transformed column in datetime format.
'''
def _transform_z_card_valid(df: pd.DataFrame):
    df["Z_CARD_VALID"] = df["Z_CARD_VALID"].astype(str)
    df["Z_CARD_VALID"] = df["Z_CARD_VALID"].str.replace('.', '-', regex=False)
    return pd.to_datetime(df["Z_CARD_VALID"], format='%m-%Y', errors='coerce')

def _num_cat_tim(df: pd.DataFrame):

    num = df.select_dtypes(include=['float64', 'int64']).columns
    cat = df.select_dtypes(include=['object']).columns
    tim = df.select_dtypes(include=['datetime64[ns]']).columns

    return (num, cat, tim)

############################### MAIN FUNCTIONS $$###############################

'''
Clean dataframe df for further use, specifically:

    1. Drop null/mostly null columns.
    2. Transform time columns into datetime format.
    3. Handle missing values based on the column type (num., cat., time).

    Return the cleaned dataframe.
'''
def clean_df(df: pd.DataFrame):

    # Drop mostly null columns
    df.drop(columns=["ANUMMER_03", "ANUMMER_02", "ANUMMER_04", "ANUMMER_05",
                     "ANUMMER_06", "ANUMMER_07", "ANUMMER_08", "ANUMMER_09", 
                     "ANUMMER_10"], inplace=True)

    ############################## DATA TRANSFORMS #############################

    # Replace '?' with NAN values.
    df.replace('?', np.nan, inplace=True) 


    # Datetime transformations
    df['Z_CARD_VALID'] = _transform_z_card_valid(df)
    df["B_BIRTHDATE"] = pd.to_datetime(df["B_BIRTHDATE"], errors='coerce')
    df["DATE_LORDER"] = pd.to_datetime(df["DATE_LORDER"], errors='coerce')
    df["TIME_ORDER"] = pd.to_datetime(df["TIME_ORDER"], format='%H:%M', 
                                      errors='coerce')

    ########################### MISSING VALUE HANDLING #########################

    # Find the numerical and categorical columns, we have a different imputation
    # strategy them.
    numerical_cols, categorical_cols, time_cols = _num_cat_tim(df)

    # Impute columns based on a strategy, subject to experimentation to improve
    # performance.
    #df[categorical_cols] = _impute_by_mode(df, categorical_cols)
    df[time_cols] = _impute_by_constant(df, time_cols, 
                                        pd.Timestamp('00:00:00'))
    #df[numerical_cols] = _impute_by_mean(df, numerical_cols)
    
    ############################################################################
    return df




def standardize_df(df: pd.DataFrame):

    # Get a list of the types of columns
    nume_cols, cat_cols, tim_cols = _num_cat_tim(df)


    # Standardize the numerical columns by subtracting mean, dividing by std.
    for n in nume_cols:
        mean = df[n].mean()
        std = df[n].std()
        df[n] = (df[n]-mean) / std

    # Ensure that the target atribute is mapped to 
    # low risk -> 0
    # high risk -> 1
    df['CLASS'] = df['CLASS'].map({'no': 0, 'yes':1})

    # Get all categorical columns except CLASS (target col)
    onehot_cols = [c for c in cat_cols if c != 'CLASS']

    # Encode those categorical columns using one-hot
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    # Multiple columns by 1, exclude time columns
    for c in df.columns:
        if not c in tim_cols:
            df[c] = df[c] * 1

    return df
