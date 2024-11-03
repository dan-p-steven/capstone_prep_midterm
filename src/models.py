'''
Author: Daniel Steven
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Metrics
from sklearn.metrics import make_scorer, confusion_matrix, classification_report

# Tools
from sklearn.model_selection import train_test_split, GridSearchCV

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# Block an annoying deprecation warning
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def cost_score(y, y_pred):
    '''
    Return normalized cost given confusion matrix and cost matrix. Used to 
    evaluate the perfomance of models. 

    Inputs:
        confusion_matrix (np.ndarray): confusion matrix from a model
        cost_matrix (np.ndarray): cost matrix for a problem
    Outputs:
        cost (int): performance metric (cost per sample)
    '''

    COST_MAT = np.array([[0, 5],
                            [50, 0]])
    conf_mat = confusion_matrix(y, y_pred)

    # Get the average cost (penalty) per sample.
    cost_score = np.sum(conf_mat * COST_MAT) / np.sum(conf_mat)
    return cost_score

def model_exploration(df: pd.DataFrame):

    # Set the base estimator for AdaBoost
    adaboost_estimator = DecisionTreeClassifier(max_depth=1, 
                                                class_weight='balanced')

    y = df['CLASS']
    X = df.drop('CLASS', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.3,
                                                        random_state=420)

    models = {

    "Logistic Regression": {
        "instance": LogisticRegression(penalty='l1',
                                       solver='liblinear',
                                       class_weight='balanced'
                                       ),
        "params": {
            "C": [0.01, 0.1, 1, 10],
            'max_iter': [100, 250, 500, 1000],
            }
        },

    "Support Vector Classifier": {
        "instance": SVC(kernel='rbf',
                        class_weight='balanced'),
        "params": {
            "C": [0.01, 0.1, 1, 10, 100],
            }
        },

    "Random Forest": {
        "instance": RandomForestClassifier(n_jobs=-1,
                                        class_weight='balanced'),
        "params": {
            "n_estimators": [50, 175, 300],
            "max_depth": [None, 10, 20, 30]
            }
        },
    
    "K-Nearest Neighbors": {
        "instance": KNeighborsClassifier(n_jobs=-1),
        "params": {
            "n_neighbors": list(range(5, 50, 2)),
            "weights": ["uniform", "distance"]
            }
        },

    "AdaBoost": {
        "instance": AdaBoostClassifier(estimator=adaboost_estimator),
        "params": {
            "n_estimators": [50, 100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.5, 1.0]
            }
        }

    }

    for model_name, model_info in models.items():

        # Set up grid search
        grid_search = GridSearchCV(estimator=model_info['instance'],
                                   param_grid=model_info['params'],
                                   # implement my custom scoring function here
                                   scoring=make_scorer(cost_score,
                                                       greater_is_better=False),
                                                       n_jobs=-1,
                                                       verbose=0
                                                       )

        # Fit search to training data
        grid_search.fit(X_train, y_train)

        # Retrieve the best scores and params
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Use the best model to predict test set
        y_pred = best_model.predict(X_test)

        # Print info to terminal
        print (f'\n{model_name}\n')
        print (f'best params: {best_params}')
        print (f'best training score: {best_score:.2f}')

        print('\nConfusion Matrix')
        print(confusion_matrix(y_test, y_pred))
        print (f'testing set score: {-1*cost_score(y_test, y_pred):.2f}')
        print(classification_report(y_test, y_pred))





