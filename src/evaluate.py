import numpy as np
from sklearn.metrics import make_scorer, confusion_matrix

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
