# from datasets import load_dataset
# import pandas as pd
# from pathlib import Path

# # Configure paths
# dataset_dir = Path("data/raw")
# dataset_dir.mkdir(parents=True, exist_ok=True)
# csv_path = dataset_dir / "yelp_reviews.csv"

# # Load Yelp dataset (small subset for testing)
# print("Loading Yelp dataset...")
# dataset = load_dataset("yelp_review_full")

# # Convert to pandas DataFrame and take a sample (10,000 reviews)
# print("Processing data...")
# df = pd.DataFrame(dataset['train']).sample(10000, random_state=42)

# # Rename columns to match your expected format
# df = df.rename(columns={
#     "text": "review_text",
#     "label": "sentiment"
# })

# # # Map numeric labels to text (0-4 → 1-5 stars → sentiment)
# # df['sentiment'] = df['sentiment'].map({
# #     0: "negative",
# #     1: "negative",
# #     2: "neutral", 
# #     3: "positive",
# #     4: "positive"
# # })

# # Save as CSV
# print(f"Saving to {csv_path}...")
# df.to_csv(csv_path, index=False)

# print("Done! Dataset saved with columns:", df.columns.tolist())


import pandas as pd
import random

# Sample phrases for each sentiment
positive_reviews = [
    "Excellent product! Highly recommended.",
    "Totally worth the money 👌",
    "Amazing. I’ll buy it again.",
    "Perfect fit and great quality!",
    "Loved it! Five stars from me ⭐⭐⭐⭐⭐",
]

neutral_reviews = [
    "It’s okay, not great not terrible.",
    "Average quality, meets expectations.",
    "Decent for the price.",
    "Neither good nor bad.",
    "It works as expected."
]

negative_reviews = [
    "Terrible quality. Broke in days.",
    "Very disappointed. Wouldn’t buy again.",
    "Waste of money!",
    "Worst experience ever!",
    "Poor performance, not recommended."
]

# Generate 1000 records
data = []

for _ in range(1000):
    sentiment = random.choices(
        ['positive', 'neutral', 'negative'], weights=[0.5, 0.2, 0.3])[0]

    if sentiment == "positive":
        review = random.choice(positive_reviews)
        rating = random.choice([4, 5])
    elif sentiment == "neutral":
        review = random.choice(neutral_reviews)
        rating = 3
    else:  # negative
        review = random.choice(negative_reviews)
        rating = random.choice([1, 2])

    # Introduce fake reviews:
    label_fake = random.choices([0, 1], weights=[0.85, 0.15])[0]
    data.append({
        "review_text": review,
        "rating": rating,
        "label_fake": label_fake,
        "sentiment": sentiment
    })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("data/raw/sample_reviews.csv", index=False)
print("✅ Generated sample_reviews.csv with 1000 rows.")