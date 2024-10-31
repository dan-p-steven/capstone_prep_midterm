# Launch script
import pandas as pd
from src.data import clean_df, standardize_df
from imblearn.over_sampling import SMOTE

if __name__ == "__main__":

    # Set the file path for the dataset
    dataset_file_path = 'data/raw/risk-train.txt'


    # Read from csv
    df = pd.read_csv(dataset_file_path, sep='\t')

    # Data Processing
    df = clean_df(df)
    df = standardize_df(df)

    print(df['CLASS'].isna().sum())
    print(df.shape)

    #print(risk_corr)



    #print(df.head())
    

