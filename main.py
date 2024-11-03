# Launch script
import numpy as np
import pandas as pd
from src.data import clean_df, standardize_df
from src.evaluate import cost_metric

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

COST_MATRIX = np.array([[0, 5],
                        [50, 0]])

if __name__ == "__main__":

    # Set the file path for the dataset
    dataset_file_path = 'data/raw/risk-train.txt'


    # Read from csv
    df = pd.read_csv(dataset_file_path, sep='\t')

    # Data Processing
    df = clean_df(df)
    df = standardize_df(df)

    X = df.drop(columns='CLASS')
    y = df['CLASS']

    # Define the custom cost penalties
    fneg = 50
    fpos = 5

    cost_penalties = { 0:fpos, 1:fneg }


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    brain_bits_model = LogisticRegression(class_weight=cost_penalties, solver='liblinear')

    brain_bits_model.fit(X_train, y_train)

    y_pred = brain_bits_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print( f'Cost: {cost_metric(cm, COST_MATRIX)}' )

    print(f'Confusion matrix:\n {cm}')
    print(f'Classification report')
    print(classification_report(y_test, y_pred))


