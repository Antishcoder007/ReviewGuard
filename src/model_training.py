# # src/model_training.py

# import pandas as pd
# import joblib
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# import xgboost as xgb
# from feature_engineering import extract_behavioral_features, generate_tfidf_features, combine_features


# def train_fake_review_classifier(df):
#     """
#     Train binary classifier to detect fake/genuine reviews
#     """
#     # Generate features
#     df = extract_behavioral_features(df)
#     tfidf, _ = generate_tfidf_features(df['clean_text'], fit=True)
#     X = combine_features(tfidf, df)
#     y = df['label_fake']

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Model options
#     model = xgb.XGBClassifier(eval_metric='logloss',verbosity=0, scale_pos_weight=5)
#     model.fit(X_train, y_train)

#     # Evaluate
#     y_pred = model.predict(X_test)
#     print("🔍 Fake Review Classifier Performance:")
#     print(classification_report(y_test, y_pred))
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#     # Save model
#     os.makedirs("models", exist_ok=True)
#     joblib.dump(model, "models/fake_review_model.pkl")
#     print("✅ Saved fake review model to → models/fake_review_model.pkl")


# def train_sentiment_classifier(df):
#     """
#     Train multi-class classifier to predict sentiment: positive, negative, neutral
#     """
#     # Keep only genuine reviews (label_fake == 0), train sentiment classifier on them
#     df = df[df['label_fake'] == 0]

#     if 'sentiment' not in df.columns:
#         raise ValueError("❌ 'sentiment' column missing for training sentiment model.")

#     df = extract_behavioral_features(df)  # Optional for shallow model
#     tfidf, _ = generate_tfidf_features(df['clean_text'], fit=True)
#     X = combine_features(tfidf, df)
#     y = df['sentiment']  # Assume values: 'positive', 'neutral', 'negative'

#     # Encode classes
#     from sklearn.preprocessing import LabelEncoder
#     le = LabelEncoder()
#     y_encoded = le.fit_transform(y)

#     # Save label encoder
#     joblib.dump(le, "../models/sentiment_label_encoder.pkl")

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#     # Model
#     model = LogisticRegression(max_iter=500,class_weight='balanced')
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)
#     print("🔍 Sentiment Classifier Performance:")
#     print(classification_report(y_test, y_pred))

#     joblib.dump(model, "../models/sentiment_model.pkl")
#     print("✅ Saved sentiment model to → models/sentiment_model.pkl")


# if __name__ == "__main__":
#     # Load preprocessed dataset
#     df = pd.read_csv("../data/processed/processed_reviews.csv")

#     # Train both models
#     train_fake_review_classifier(df)
#     train_sentiment_classifier(df)

# src/model_training.py

import pandas as pd
import joblib
import os


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb

from src.feature_engineering import extract_behavioral_features, generate_tfidf_features, combine_features
from src.config_loader import load_config

def get_model_path(key):
    config = load_config()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_dir, config['models'][key])

def train_fake_review_classifier(df):
    print("\n📌 Training Fake Review Classifier...")

    df = extract_behavioral_features(df)

    # Handle class imbalance smartly
    tfidf_matrix, _ = generate_tfidf_features(df['clean_text'], fit=True)
    X = combine_features(tfidf_matrix, df)
    y = df['label_fake']

    print(f"  ▶ Class distribution:\n{y.value_counts()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Use XGBoost with class imbalance correction
    pos_weight = (y == 0).sum() / (y == 1).sum()  # ratio of negatives to positives
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        verbosity=0
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("🔍 Fake Review Classifier Performance:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    # joblib.dump(model, "../models/fake_review_model.pkl")
    
    joblib.dump(model, get_model_path("fake_classifier"))
    print(f"✅ Saved fake review model to → {get_model_path('fake_classifier')}")


def train_sentiment_classifier(df):
    print("\n📌 Training Sentiment Classifier...")

    # Use only genuine reviews
    df = df[df['label_fake'] == 0].copy()

    if 'sentiment' not in df.columns:
        raise ValueError("Missing 'sentiment' column for training.")

    df = extract_behavioral_features(df)
    tfidf_matrix, _ = generate_tfidf_features(df['clean_text'], fit=True)
    X = combine_features(tfidf_matrix, df)

    # Encode sentiment labels
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])

    # Split and train logistic regression
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("🔍 Sentiment Classifier Performance:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, get_model_path("sentiment_classifier"))
    joblib.dump(le, get_model_path("label_encoder"))
    print(f"✅ Saved sentiment model to → {get_model_path('sentiment_classifier')}")
    print(f"✅ Saved label encoder to → {get_model_path('label_encoder')}")

    # joblib.dump(model, "../models/sentiment_model.pkl")
    # joblib.dump(le, "../models/sentiment_label_encoder.pkl")
    # print("✅ Saved sentiment model to → models/sentiment_model.pkl")
    # print("✅ Saved label encoder to → models/sentiment_label_encoder.pkl")


if __name__ == "__main__":
    df = pd.read_csv("data/processed/processed_reviews.csv")

    if 'label_fake' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Missing required columns: 'label_fake' and 'sentiment'.")

    train_fake_review_classifier(df)
    train_sentiment_classifier(df)