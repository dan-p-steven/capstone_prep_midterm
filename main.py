# Launch script
import pandas as pd
from src.data import clean_df, standardize_df 

if __name__ == "__main__":

    # Set the file path for the dataset
    dataset_file_path = 'data/raw/risk-train.txt'


    # Read from csv
    df = pd.read_csv(dataset_file_path, sep='\t')

    # Clean df
    df_cleaned = clean_df(df)

    # Standardize df
    standardize_df(df_cleaned)

    print(df_cleaned.head())

