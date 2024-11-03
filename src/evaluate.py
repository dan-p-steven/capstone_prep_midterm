import numpy as np

def cost_metric(confusion_matrix: np.ndarray, cost_matrix: np.ndarray):
    '''
    Return normalized cost given confusion matrix and cost matrix. Used to 
    evaluate the perfomance of models.

    Inputs:
        confusion_matrix (np.ndarray): confusion matrix from a model
        cost_matrix (np.ndarray): cost matrix for a problem
    Outputs:
        cost (int): performance metric
    '''

    # We have to divide the cost by total num. of samples to get a meaningful
    # score that can be compared across different testing sets.
    return np.sum(confusion_matrix * cost_matrix) / np.sum(confusion_matrix)
