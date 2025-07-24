# # src/predict.py

# import pandas as pd
# import joblib
# import os
# from feature_engineering import extract_behavioral_features, generate_tfidf_features, combine_features


# def load_models():
#     fake_model = joblib.load("../models/fake_review_model.pkl")
#     sentiment_model = joblib.load("../models/sentiment_model.pkl")
#     vectorizer = joblib.load("../models/vectorizer.pkl")
#     scaler = joblib.load("../models/scaler.pkl")
#     label_encoder = joblib.load("../models/sentiment_label_encoder.pkl")
#     return fake_model, sentiment_model, vectorizer, scaler, label_encoder


# def predict_all(df):
#     """
#     Apply fake review classification → then sentiment analysis → then corrected rating.
#     """
#     print("🚀 Running inference pipeline...")
#     # 1. Feature extraction
#     df = extract_behavioral_features(df)
#     tfidf, _ = generate_tfidf_features(df['clean_text'], fit=False)
#     X = combine_features(tfidf, df)

#     # 2. Load models
#     fake_model, sentiment_model, vectorizer, scaler, label_encoder = load_models()

#     # 3. Predict fake reviews
#     df['predicted_fake'] = fake_model.predict(X)

#     # 4. Sentiment prediction for genuine reviews
#     df['predicted_sentiment'] = 'n/a'  # Default
#     genuine_indices = df[df['predicted_fake'] == 0].index
#     if len(genuine_indices) > 0:
#         genuine_df = df.loc[genuine_indices]
        
#         # Generate features again for just genuine
#         tfidf_genuine = vectorizer.transform(genuine_df['clean_text'])
#         behavioral_genuine = extract_behavioral_features(genuine_df)
#         from scipy.sparse import hstack
#         combined = hstack([
#             tfidf_genuine,
#             scaler.transform(behavioral_genuine[['review_length', 'char_length', 'punctuation_count', 'rating_mismatch']])
#         ])

#         preds = sentiment_model.predict(combined)
#         df.loc[genuine_indices, 'predicted_sentiment'] = label_encoder.inverse_transform(preds)

#     # 5. Corrected ratings
#     def correct_rating(row):
#         if row['predicted_fake'] == 1:
#             if row['predicted_sentiment'] == 'positive': return 5
#             elif row['predicted_sentiment'] == 'neutral': return 3
#             elif row['predicted_sentiment'] == 'negative': return 1
#             else: return 3  # fallback
#         else:
#             return row['rating']  # keep original if genuine

#     df['corrected_rating'] = df.apply(correct_rating, axis=1)

#     return df


# # For testing
# if __name__ == '__main__':
#     df = pd.read_csv("../data/processed/processed_reviews.csv")
#     output_df = predict_all(df)
#     os.makedirs("outputs/predictions/", exist_ok=True)
#     output_df.to_csv("outputs/predictions/predicted_reviews.csv", index=False)
#     print("✅ Predictions saved to → outputs/predictions/predicted_reviews.csv")


# src/predict.py

import os
import joblib
import pandas as pd

from src.feature_engineering import extract_behavioral_features, generate_tfidf_features, combine_features
from scipy.sparse import hstack


def load_models():
    
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    models_dir = os.path.join(root_dir, 'models')

    print("✅ [DEBUG] Using UPDATED load_models() from src/predict.py")

    print(f"🔍 Current file: {current_file}")
    print(f"📁 Models directory expected: {models_dir}")

    print(f"📂 Listing contents of models_dir:")
    if os.path.exists(models_dir):
        print(os.listdir(models_dir))
    else:
        raise Exception("❌ models/ directory not found at expected location!")

    vectorizer_path = os.path.join(models_dir, 'vectorizer.pkl')

    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"❌ File not found: {vectorizer_path}")

    return {
        'fake_model': joblib.load(os.path.join(models_dir, 'fake_review_model.pkl')),
        'sentiment_model': joblib.load(os.path.join(models_dir, 'sentiment_model.pkl')),
        'vectorizer': joblib.load(vectorizer_path),
        'scaler': joblib.load(os.path.join(models_dir, 'scaler.pkl')),
        'label_encoder': joblib.load(os.path.join(models_dir, 'sentiment_label_encoder.pkl')),
    }


def correct_rating(row):
    """
    Correct the rating based on predicted sentiment
    """
    if row['predicted_fake'] == 1:
        sentiment = row['predicted_sentiment']
        if sentiment == 'positive': return 5
        elif sentiment == 'neutral': return 3
        elif sentiment == 'negative': return 1
        else: return 3  # fallback
    return row['rating']


def predict_all(df):
    """
    Predict fake reviews, sentiment (for genuine), and compute corrected rating
    """
    print("⚙️  Starting prediction pipeline...")

    # Load models
    models = load_models()

    # Extract common features
    df = df.copy()
    df = extract_behavioral_features(df)
    tfidf_matrix, _ = generate_tfidf_features(df["clean_text"], fit=False)
    x = combine_features(tfidf_matrix, df, save_scaler=False)
    # tfidf_matrix, _ = generate_tfidf_features(df['clean_text'], fit=False)
    # X = combine_features(tfidf_matrix, df)

    # Predict fake vs genuine
    df['predicted_fake'] = models['fake_model'].predict(x)

    # Predict sentiment for genuine reviews only
    df['predicted_sentiment'] = 'n/a'
    genuine_df = df[df['predicted_fake'] == 0].copy()

    if not genuine_df.empty:
        tfidf_g = models['vectorizer'].transform(genuine_df['clean_text'])
        behavioral = extract_behavioral_features(genuine_df)
        scaled = models['scaler'].transform(
            behavioral[['review_length', 'char_length', 'punctuation_count', 'rating_mismatch']]
        )
        X_sent = hstack([tfidf_g, scaled])
        sentiment_preds = models['sentiment_model'].predict(X_sent)
        sentiments = models['label_encoder'].inverse_transform(sentiment_preds)
        df.loc[genuine_df.index, 'predicted_sentiment'] = sentiments

    # Apply corrected ratings
    df['corrected_rating'] = df.apply(correct_rating, axis=1)

    print("✅ Prediction pipeline completed.")
    return df


# For manual testing
if __name__ == '__main__':
    print("[🧪 Testing Mode]")

    # Use path relative to this script
    processed_path = os.path.join(os.path.dirname(__file__), "../data/processed/processed_reviews.csv")
    output_path = os.path.join(os.path.dirname(__file__), "../outputs/predictions/predicted_reviews.csv")

    df = pd.read_csv(processed_path)
    results = predict_all(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to → {output_path}")