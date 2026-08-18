import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.encoders = {}
        self.scaler = MinMaxScaler()
        
    def load_and_preprocess(self):
        df = pd.read_csv(self.filepath)
        required_cols = ['Rating', 'Sentiment_Score', 'Price', 'Rec_Probability', 'Season', 'Geo_Location', 'Holiday']
        df = df.dropna(subset=required_cols).reset_index(drop=True)
        
        cat_cols = ['Season', 'Geo_Location', 'Holiday']
        for col in cat_cols:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
            self.encoders[col] = le
            
        return df

    def get_feature_matrix(self, df):
        feature_cols = ['Rating', 'Sentiment_Score', 'Price', 'Rec_Probability', 
                        'Season_Encoded', 'Geo_Location_Encoded', 'Holiday_Encoded']
        return df[feature_cols].values
