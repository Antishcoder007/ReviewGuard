# src/data_preparation.py

import pandas as pd
import re
import nltk
import emoji
import spacy

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# nltk.download('punkt')
# nltk.download('stopwords')
# STOPWORDS = set(stopwords.words('english'))
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))


try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    import subprocess
    import sys
    print("Downloading spaCy model...")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text):
    if pd.isna(text): return ""
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)

def tokenize(text):
    tokens = word_tokenize(text)
    return " ".join([w for w in tokens if w not in STOPWORDS])

def lemmatize(text):
    return " ".join([token.lemma_ for token in nlp(text)])

def preprocess(df, text_col="review_text"):
    df[text_col] = df[text_col].astype(str)
    df["clean_text"] = df[text_col].apply(clean_text)
    df["clean_text"] = df["clean_text"].apply(tokenize).apply(lemmatize)
    return df

def load_and_process(filepath):
    df = pd.read_csv(filepath)
    return preprocess(df)

if __name__ == "__main__":
    df = load_and_process("data/raw/sample_reviews.csv")
    df.to_csv("data/processed/processed_reviews.csv", index=False)
    print("[✅] Processed and saved.")
