import numpy as np

class COMETRecommender:
    def __init__(self, weights):
        self.weights = np.array(weights) / np.sum(weights)

    def fit_predict(self, matrix):
        min_vals = np.min(matrix, axis=0)
        max_vals = np.max(matrix, axis=0)
        norm_matrix = (matrix - min_vals) / (max_vals - min_vals + 1e-9)
        
        utility = np.dot(norm_matrix, self.weights)
        return utility
