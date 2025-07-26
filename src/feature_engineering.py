# src/feature_engineering.py

import pandas as pd
import numpy as np
import joblib
import string
from src.config_loader import load_config

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import os


def extract_behavioral_features(df):
    """
    Generate behavioral features from review data.
    Ex: length, capital ratio, rating mismatch, punctuation usage.
    """
    df['review_length'] = df['clean_text'].apply(lambda x: len(x.split()))
    df['char_length'] = df['clean_text'].apply(len)
    df['punctuation_count'] = df['review_text'].apply(lambda x: sum([1 for c in str(x) if c in string.punctuation]))
    
    # Text sentiment deviation from rating (map rating to sentiment)
    # 1-2: negative, 3: neutral, 4-5: positive
    def get_rating_sentiment(rating):
        if rating <= 2:
            return 'negative'
        elif rating == 3:
            return 'neutral'
        else:
            return 'positive'

    if 'predicted_sentiment' in df.columns:
        df['rating_sentiment'] = df['rating'].apply(get_rating_sentiment)
        df['rating_mismatch'] = (df['predicted_sentiment'] != df['rating_sentiment']).astype(int)
    else:
        df['rating_mismatch'] = 0  # Default when sentiment model hasn't run

    return df


def generate_tfidf_features(texts, fit=True):
    """
    Convert clean text into TF-IDF features.
    Save vectorizer if `fit=True`. Else, load and transform.
    """
    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    vectorizer_path = os.path.join(base_dir, config['models']['vectorizer'])

    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

    if fit:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, vectorizer_path)
        print(f"✅ TF-IDF vectorizer saved at → {vectorizer_path}")
    else:
        print(f"📁 Loading vectorizer from: {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        tfidf_matrix = vectorizer.transform(texts)

    return tfidf_matrix, vectorizer


def combine_features(tfidf, behavioral_df,save_scaler=True):
    """Combine TF-IDF sparse matrix and numeric features (standardized)."""

    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    scaler_path = os.path.join(base_dir, config['models']['scaler'])

    behavioral_features = behavioral_df[['review_length', 'char_length', 'punctuation_count', 'rating_mismatch']].copy()
    scaler = StandardScaler()
    scaled_feats = scaler.fit_transform(behavioral_features)

    if save_scaler:
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler saved at → {scaler_path}")
    
    # joblib.dump(scaler, '../models/scaler.pkl')

    from scipy.sparse import hstack
    combined = hstack([tfidf, scaled_feats])
    return combined


if __name__ == "__main__":
    # Load preprocessed data
    df = pd.read_csv("data/processed/processed_reviews.csv")
    df = extract_behavioral_features(df)

    # Generate TF-IDF
    tfidf_matrix, _ = generate_tfidf_features(df['clean_text'], fit=True)

    # Combine features for model input
    X = combine_features(tfidf_matrix, df,save_scaler=False)
    print(f"✅ Final feature matrix shape: {X.shape}")
