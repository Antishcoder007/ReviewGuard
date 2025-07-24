
import random
import pandas as pd
from faker import Faker
from textblob import TextBlob

# Initialize Faker for realistic text generation
fake = Faker()

def generate_realistic_review(rating):
    """Generate a realistic review based on the rating"""
    products = [
        "smartphone", "laptop", "headphones", "smartwatch", "TV",
        "blender", "book", "video game", "shoes", "jacket"
    ]
    product = random.choice(products)
    
    # Templates for different rating levels
    if rating == 5:
        templates = [
            f"I absolutely love this {product}! It's perfect in every way.",
            f"This {product} exceeded all my expectations. Worth every penny!",
            f"Best {product} I've ever owned. Highly recommend to everyone.",
            f"Flawless {product}. Works perfectly and looks amazing.",
            f"Five stars for this amazing {product}. Couldn't be happier!"
        ]
    elif rating == 4:
        templates = [
            f"Great {product} overall with just minor issues.",
            f"Very good {product}, but there's room for improvement.",
            f"I'm quite satisfied with this {product}. Works well.",
            f"Excellent {product} with just a couple small drawbacks.",
            f"Really like this {product}. Would buy again."
        ]
    elif rating == 3:
        templates = [
            f"This {product} is okay, but nothing special.",
            f"Average {product}. Does the job but has some flaws.",
            f"Mediocre {product}. Not bad, but not great either.",
            f"The {product} works, but I expected better quality.",
            f"Decent {product}, though it has some issues."
        ]
    elif rating == 2:
        templates = [
            f"Disappointed with this {product}. Many problems.",
            f"This {product} is below average. Wouldn't recommend.",
            f"Poor quality {product}. Doesn't work as advertised.",
            f"Frustrating {product}. Many things need improvement.",
            f"Not happy with this {product}. Has major flaws."
        ]
    else:  # rating == 1
        templates = [
            f"Terrible {product}! Complete waste of money.",
            f"Worst {product} ever. Do not buy this!",
            f"This {product} is defective. Absolute garbage.",
            f"I regret buying this awful {product}. Broken on arrival.",
            f"1 star because I can't give zero. This {product} is horrible."
        ]
    
    review = random.choice(templates)
    
    # Add some realistic variations
    if random.random() > 0.7:
        review += " " + fake.sentence()
    if random.random() > 0.8:
        review = review.capitalize() + " " + fake.sentence().lower()
    
    return review

def determine_sentiment(review_text):
    """Determine sentiment polarity using TextBlob"""
    analysis = TextBlob(review_text)
    # Convert polarity (-1 to 1) to sentiment (0, 1, 2)
    if analysis.sentiment.polarity > 0.1:
        return "positive"
    elif analysis.sentiment.polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_fake_review(rating):
    """Generate a fake review that might be less coherent"""
    products = [
        "item", "product", "thing", "device", "gadget"
    ]
    product = random.choice(products)
    
    # Less coherent templates
    templates = [
        f"Good {product} buy now best quality!!!",
        f"{product} works good perfect no problems.",
        f"Bad {product} not working waste money.",
        f"Excellent {product} very nice love it!",
        f"{product} is okay could be better.",
        f"{random.randint(1,5)} stars {product} fine.",
        f"Not bad {product} for the price.",
        f"Terrible experience with {product}.",
        f"{product.upper()} IS AWESOME MUST BUY!!!",
        f"Would recommend {product} to friends."
    ]
    
    review = random.choice(templates)
    
    # Add some random elements to make it less coherent
    if random.random() > 0.5:
        review += " " + " ".join(fake.words(random.randint(1, 3)))
    if random.random() > 0.7:
        review = review.upper()
    
    return review

def generate_dataset(num_rows=1000):
    """Generate the complete dataset"""
    data = []
    
    for _ in range(num_rows):
        # Decide if this will be a fake review (10% chance)
        is_fake = random.random() < 0.1
        
        # Generate rating (weighted toward higher ratings)
        rating = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.1, 0.15, 0.2, 0.25, 0.3]
        )[0]
        
        # Generate appropriate review text
        if is_fake:
            review_text = generate_fake_review(rating)
        else:
            review_text = generate_realistic_review(rating)
        
        # Determine sentiment
        sentiment = determine_sentiment(review_text)
        
        

        # Add to dataset
        data.append({
            "review_text": review_text,
            "rating": rating,
            "label_fake": is_fake,
            "sentiment": sentiment
        })
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_dataset(1000)

df['label_fake'] = df['label_fake'].astype(int)
df.to_csv("data/raw/sample_reviews1.csv", index=False)


# Verify some basic statistics
print("\nDataset Overview:")
print(f"Total reviews: {len(df)}")
print(f"Fake reviews: {df['label_fake'].sum()} ({df['label_fake'].mean()*100:.1f}%)")
print("\nRating distribution:")
print(df['rating'].value_counts().sort_index())
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())