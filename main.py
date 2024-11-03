# Launch script
import numpy as np
import pandas as pd

# Our libraries
#from src.data import clean_df, standardize_df
from src.models import model_exploration



if __name__ == "__main__":

    # Set the file path for the dataset
    dataset_file_path = 'data/raw/risk-train.txt'
    cleaned_file_path = './data/processed/cleaned_risk.csv'

    

    # Read cleaned dataframe from csv
    df = pd.read_csv(cleaned_file_path)

    model_exploration(df)

    #X = df.drop(columns='CLASS')
    #y = df['CLASS']

    ## Define the custom cost penalties
    #fneg = 50
    #fpos = 5

    #cost_penalties = { 0:fpos, 1:fneg }


    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    #brain_bits_model = LogisticRegression(class_weight=cost_penalties, solver='liblinear')

    #brain_bits_model.fit(X_train, y_train)

    #y_pred = brain_bits_model.predict(X_test)

    #cm = confusion_matrix(y_test, y_pred)
    #print( f'Cost: {cost_metric(cm, COST_MATRIX): .2f}' )

    #print(f'Confusion matrix:\n {cm}')
    #print(f'Classification report')
    #print(classification_report(y_test, y_pred))


