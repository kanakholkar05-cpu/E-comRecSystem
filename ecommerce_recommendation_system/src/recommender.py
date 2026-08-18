import pandas as pd
from .data_loader import DataProcessor
from .topsis import TOPSISRecommender
from .comet import COMETRecommender

class RecommendationEngine:
    def __init__(self, data_path):
        self.processor = DataProcessor(data_path)
        self.df = self.processor.load_and_preprocess()
        self.matrix = self.processor.get_feature_matrix(self.df)
        
    def recommend(self, weights, criteria_types, method="TOPSIS", top_n=5):
        if method == "TOPSIS":
            model = TOPSISRecommender(weights, criteria_types)
            scores = model.fit_predict(self.matrix)
            self.df['TOPSIS_Score'] = scores
            ranked_df = self.df.sort_values(by='TOPSIS_Score', ascending=False)
        elif method == "COMET":
            model = COMETRecommender(weights)
            scores = model.fit_predict(self.matrix)
            self.df['COMET_Utility'] = scores
            ranked_df = self.df.sort_values(by='COMET_Utility', ascending=False)
        else:
            raise ValueError("Method must be either 'TOPSIS' or 'COMET'")
            
        return ranked_df.head(top_n)
