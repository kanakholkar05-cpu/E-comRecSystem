import streamlit as st
import pandas as pd
from src.recommender import RecommendationEngine

st.set_page_config(page_title="Multi-Criteria Recommendation System", layout="wide")

st.title("🛒 E-Commerce Multi-Criteria Recommendation System")
st.markdown("Combines **TOPSIS** and **COMET** MCDM methods with personalized criteria (Season, Location, Holiday).")

st.sidebar.header("1. Custom Feature Weights")
w_rating = st.sidebar.slider("Rating Weight", 0.0, 1.0, 0.25)
w_sentiment = st.sidebar.slider("Sentiment Score Weight", 0.0, 1.0, 0.25)
w_price = st.sidebar.slider("Price Weight", 0.0, 1.0, 0.15)
w_rec = st.sidebar.slider("Recommendation Prob. Weight", 0.0, 1.0, 0.15)
w_season = st.sidebar.slider("Seasonality Weight", 0.0, 1.0, 0.07)
w_geo = st.sidebar.slider("Geographical Weight", 0.0, 1.0, 0.08)
w_holiday = st.sidebar.slider("Holiday Relevance Weight", 0.0, 1.0, 0.05)

weights = [w_rating, w_sentiment, w_price, w_rec, w_season, w_geo, w_holiday]
criteria_types = [True, True, False, True, True, True, True]

st.sidebar.header("2. Model Selection")
method = st.sidebar.radio("Method", ["TOPSIS", "COMET"])
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

engine = RecommendationEngine("data/dataset.csv")

if st.button("Generate Recommendations"):
    results = engine.recommend(weights, criteria_types, method=method, top_n=top_n)
    
    st.subheader(f"Top {top_n} Recommendations using {method}")
    
    display_cols = ['Product_ID', 'Product_Name', 'Rating', 'Sentiment_Score', 'Price', 'Season', 'Geo_Location', 'Holiday']
    if method == "TOPSIS":
        display_cols.append('TOPSIS_Score')
    else:
        display_cols.append('COMET_Utility')
        
    st.dataframe(results[display_cols].reset_index(drop=True), use_container_width=True)
