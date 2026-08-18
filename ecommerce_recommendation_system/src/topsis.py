import numpy as np

class TOPSISRecommender:
    def __init__(self, weights, criteria_types):
        self.weights = np.array(weights) / np.sum(weights)
        self.criteria_types = np.array(criteria_types)

    def fit_predict(self, matrix):
        norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
        weighted_matrix = norm_matrix * self.weights
        
        pis = np.where(self.criteria_types, np.max(weighted_matrix, axis=0), np.min(weighted_matrix, axis=0))
        nis = np.where(self.criteria_types, np.min(weighted_matrix, axis=0), np.max(weighted_matrix, axis=0))
        
        d_pos = np.sqrt(np.sum((weighted_matrix - pis)**2, axis=1))
        d_neg = np.sqrt(np.sum((weighted_matrix - nis)**2, axis=1))
        
        closeness = d_neg / (d_pos + d_neg + 1e-9)
        return closeness
