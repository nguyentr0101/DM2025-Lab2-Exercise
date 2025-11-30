"""
================================================================================
EMOTION CLASSIFICATION - TEXT PREPROCESSING PIPELINE
================================================================================
Kaggle Competition: Predicting 6 emotions (anger, disgust, fear, sadness, surprise, joy)

This script follows the patterns learned in:
- DM2025-Lab1: Data preparation, transformation, missing values, duplicates, 
               CountVectorizer, feature creation
- DM2025-Lab2: Feature engineering with BOW/TF-IDF, emotion classification

Author: Text Mining Course Homework
================================================================================
"""

# ================================================================================
# SETUP - Same as Lab 1 & Lab 2
# ================================================================================
import pandas as pd
import numpy as np
import json
import re
import matplotlib.pyplot as plt

# NLTK - Same as Lab 1
import nltk
nltk.download('punkt')  # download the NLTK datasets (as shown in Lab 1)
nltk.download('punkt_tab')
nltk.download('stopwords')

# Scikit-learn - Same as Lab 1 & Lab 2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("EMOTION CLASSIFICATION - PREPROCESSING PIPELINE")
print("Following DM2025-Lab1 & Lab2 patterns")
print("=" * 80)

# ================================================================================
# 1. DATA PREPARATION (Same pattern as Lab 2 Section 1.1)
# ================================================================================
print("\n" + "=" * 80)
print("1. DATA PREPARATION")
print("=" * 80)

# 1.1 Load data_identification.csv (train/test split)
print("\n[1.1] Loading data files...")
data_identification = pd.read_csv("data_identification.csv")
print(f"data_identification shape: {data_identification.shape}")

# 1.2 Load emotion.csv (labels for training data)
emotion_df = pd.read_csv("emotion.csv")
print(f"emotion_df shape: {emotion_df.shape}")

# 1.3 Load final_posts.json (text data)
with open("final_posts.json", 'r', encoding='utf-8') as f:
    posts_json = json.load(f)

# Extract posts into a DataFrame (similar to Lab 1 dictionary to dataframe)
posts_data = []
for item in posts_json:
    post = item['root']['_source']['post']
    posts_data.append({
        'id': post['post_id'],
        'text': post['text'],
        'hashtags': post.get('hashtags', [])
    })

posts_df = pd.DataFrame(posts_data)
print(f"posts_df shape: {posts_df.shape}")

# 1.4 Load samplesubmission.csv (for reference)
sample_submission = pd.read_csv("samplesubmission.csv")
print(f"sample_submission shape: {sample_submission.shape}")

# ================================================================================
# 2. DATA TRANSFORMATION (Same pattern as Lab 1 Section 3)
# ================================================================================
print("\n" + "=" * 80)
print("2. DATA TRANSFORMATION - Merging DataFrames")
print("=" * 80)

# Merge posts with split information (similar to Lab 1 adding columns)
print("\n[2.1] Merging posts with split information...")
X = posts_df.merge(data_identification, on='id', how='left')
print(f"Merged dataframe shape: {X.shape}")
print(X.head())

# Create train and test DataFrames (similar to Lab 2)
print("\n[2.2] Creating train and test DataFrames...")
train_df = X[X['split'] == 'train'].copy()
test_df = X[X['split'] == 'test'].copy()

# Add emotion labels to training data
train_df = train_df.merge(emotion_df, on='id', how='left')

print(f"Shape of Training df: {train_df.shape}")
print(f"Shape of Testing df: {test_df.shape}")

# ================================================================================
# 3. DATA QUALITY - Missing Values (Same pattern as Lab 1 Section 4.1)
# ================================================================================
print("\n" + "=" * 80)
print("3. DATA QUALITY - Missing Values & Duplicates")
print("=" * 80)

# Check missing values (same as Lab 1)
print("\n[3.1] Checking for missing values...")
print("Training set missing values:")
print(train_df.isnull().sum())

print("\nTest set missing values:")
print(test_df.isnull().sum())

# Check for duplicates (same as Lab 1 Section 4.2)
print("\n[3.2] Checking for duplicate texts...")
print(f"Duplicate texts in training: {sum(train_df.duplicated('text'))}")
print(f"Duplicate texts in test: {sum(test_df.duplicated('text'))}")

# Drop duplicates if any (same as Lab 1)
train_df.drop_duplicates(subset='text', keep='first', inplace=True)
test_df.drop_duplicates(subset='text', keep='first', inplace=True)
print(f"After removing duplicates - Training: {len(train_df)}, Test: {len(test_df)}")

# ================================================================================
# 4. EXPLORATORY DATA ANALYSIS (Same pattern as Lab 2 Section 1.3)
# ================================================================================
print("\n" + "=" * 80)
print("4. EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# Group to find distribution (same as Lab 2)
print("\n[4.1] Emotion distribution in training data:")
print(train_df.groupby(['emotion']).count()['text'])

# Plot emotion distribution (same pattern as Lab 2)
print("\n[4.2] Plotting emotion distribution...")
labels = train_df['emotion'].unique()
post_total = len(train_df)
df1 = train_df.groupby(['emotion']).count()['text']
df1_pct = df1.apply(lambda x: round(x*100/post_total, 3))

fig, ax = plt.subplots(figsize=(8, 4))
plt.bar(df1.index, df1.values)
plt.ylabel('Number of instances')
plt.xlabel('Emotion')
plt.title('Emotion distribution in Training Data')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('emotion_distribution.png', dpi=100)
plt.close()
print("Saved: emotion_distribution.png")

# ================================================================================
# 5. FEATURE CREATION (Same pattern as Lab 1 Section 5.2)
# ================================================================================
print("\n" + "=" * 80)
print("5. FEATURE CREATION - Tokenization")
print("=" * 80)

# Tokenize using NLTK (same as Lab 1)
from nltk.tokenize import word_tokenize

def tokenize_text(text):
    """Tokenize text using NLTK word_tokenize (same as Lab 1)"""
    if pd.isna(text) or not text:
        return []
    try:
        return word_tokenize(text.lower())
    except:
        return text.lower().split()

print("\n[5.1] Creating unigrams feature (tokenization)...")
train_df['unigrams'] = train_df['text'].apply(lambda x: tokenize_text(x))
test_df['unigrams'] = test_df['text'].apply(lambda x: tokenize_text(x))

# Show sample (same as Lab 1)
print("Sample unigrams:")
print(train_df[['text', 'unigrams']].head(3))

# Add text length feature
train_df['text_length'] = train_df['text'].str.len()
test_df['text_length'] = test_df['text'].str.len()

# Add token count feature
train_df['token_count'] = train_df['unigrams'].apply(len)
test_df['token_count'] = test_df['unigrams'].apply(len)

print(f"\nAverage text length (train): {train_df['text_length'].mean():.2f}")
print(f"Average token count (train): {train_df['token_count'].mean():.2f}")

# ================================================================================
# 6. FEATURE ENGINEERING - Bag of Words (Same pattern as Lab 2 Section 2)
# ================================================================================
print("\n" + "=" * 80)
print("6. FEATURE ENGINEERING - Bag of Words (BOW)")
print("=" * 80)

# Build BOW vectorizer (same as Lab 2)
print("\n[6.1] Building CountVectorizer (Bag of Words)...")
BOW_vectorizer = CountVectorizer(
    max_features=5000,           # Limit vocabulary size
    tokenizer=nltk.word_tokenize # Use NLTK tokenizer (same as Lab 2)
)

# Learn vocabulary and transform (same as Lab 2)
BOW_vectorizer.fit(train_df['text'])
train_data_BOW_features = BOW_vectorizer.transform(train_df['text'])
test_data_BOW_features = BOW_vectorizer.transform(test_df['text'])

print(f"BOW training matrix shape: {train_data_BOW_features.shape}")
print(f"BOW test matrix shape: {test_data_BOW_features.shape}")

# Show sample feature names (same as Lab 2)
feature_names_bow = BOW_vectorizer.get_feature_names_out()
print(f"Sample feature names [100:110]: {list(feature_names_bow[100:110])}")

# ================================================================================
# 7. FEATURE ENGINEERING - TF-IDF (Same pattern as Lab 2 Exercise 2)
# ================================================================================
print("\n" + "=" * 80)
print("7. FEATURE ENGINEERING - TF-IDF Vectorizer")
print("=" * 80)

# Build TF-IDF vectorizer (same as Lab 2 Exercise 2)
print("\n[7.1] Building TfidfVectorizer...")
tfidf_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',                    # Remove stopwords
    token_pattern=r'(?u)\b[a-zA-Z]{2,}\b',   # Only alphabetic tokens, min 2 chars
    max_features=10000,                       # Vocabulary size
    ngram_range=(1, 2),                       # Unigrams and bigrams
    min_df=2,                                 # Minimum document frequency
    max_df=0.95                               # Maximum document frequency
)

# Fit and transform (same pattern as Lab 2)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text'].fillna('').astype(str))
X_test_tfidf = tfidf_vectorizer.transform(test_df['text'].fillna('').astype(str))

print(f"TF-IDF training matrix shape: {X_train_tfidf.shape}")
print(f"TF-IDF test matrix shape: {X_test_tfidf.shape}")

# Show sample feature names (same as Lab 2)
feat_names_tfidf = tfidf_vectorizer.get_feature_names_out()
print(f"Sample feature names [100:110]: {list(feat_names_tfidf[100:110])}")

# ================================================================================
# 8. PREPARE LABELS (Same pattern as Lab 2 Section 3)
# ================================================================================
print("\n" + "=" * 80)
print("8. PREPARE LABELS FOR CLASSIFICATION")
print("=" * 80)

# Define emotion mapping (for Kaggle submission)
emotion_mapping = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5
}
reverse_mapping = {v: k for k, v in emotion_mapping.items()}

# Encode labels (same pattern as Lab 2)
train_df['emotion_encoded'] = train_df['emotion'].map(emotion_mapping)

# Prepare X and y (same as Lab 2 Section 3.1)
X_train = X_train_tfidf
y_train = train_df['emotion']  # String labels for sklearn
y_train_encoded = train_df['emotion_encoded']  # Numeric labels

X_test = X_test_tfidf

print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}")

print("\nLabel distribution:")
print(y_train.value_counts())

# ================================================================================
# 9. SAVE PREPROCESSED DATA (Same pattern as Lab 2 Section 1.2)
# ================================================================================
print("\n" + "=" * 80)
print("9. SAVE PREPROCESSED DATA")
print("=" * 80)

# Save to pickle format (same as Lab 2)
print("\n[9.1] Saving to pickle format...")
train_df.to_pickle("train_df.pkl")
test_df.to_pickle("test_df.pkl")
print("Saved: train_df.pkl, test_df.pkl")

# Save to CSV format
print("\n[9.2] Saving to CSV format...")
train_output = train_df[['id', 'text', 'text_length', 'token_count', 'emotion', 'emotion_encoded']].copy()
train_output.to_csv("train_preprocessed.csv", index=False)

test_output = test_df[['id', 'text', 'text_length', 'token_count']].copy()
test_output.to_csv("test_preprocessed.csv", index=False)
print("Saved: train_preprocessed.csv, test_preprocessed.csv")

# Save vectorized data (sparse matrices)
print("\n[9.3] Saving vectorized data...")
from scipy import sparse
sparse.save_npz('X_train_tfidf.npz', X_train_tfidf)
sparse.save_npz('X_test_tfidf.npz', X_test_tfidf)
sparse.save_npz('X_train_bow.npz', train_data_BOW_features)
sparse.save_npz('X_test_bow.npz', test_data_BOW_features)
print("Saved: X_train_tfidf.npz, X_test_tfidf.npz, X_train_bow.npz, X_test_bow.npz")

# Save vectorizers for later use
import pickle
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open('bow_vectorizer.pkl', 'wb') as f:
    pickle.dump(BOW_vectorizer, f)
print("Saved: tfidf_vectorizer.pkl, bow_vectorizer.pkl")

# ================================================================================
# 10. SAMPLE SUBMISSION FORMAT
# ================================================================================
print("\n" + "=" * 80)
print("10. SAMPLE SUBMISSION FORMAT")
print("=" * 80)

print("\nExpected submission format:")
print(sample_submission.head())
print(f"\nTotal predictions needed: {len(sample_submission)}")
print(f"Emotion categories: {list(emotion_mapping.keys())}")

# ================================================================================
# 11. SUMMARY
# ================================================================================
print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE - SUMMARY")
print("=" * 80)

print(f"""
DATA SUMMARY:
- Training samples: {len(train_df)}
- Test samples: {len(test_df)}

FEATURES CREATED:
- Bag of Words (BOW): {train_data_BOW_features.shape[1]} features
- TF-IDF: {X_train_tfidf.shape[1]} features

OUTPUT FILES:
- train_df.pkl, test_df.pkl (pickle format - as Lab 2)
- train_preprocessed.csv, test_preprocessed.csv
- X_train_tfidf.npz, X_test_tfidf.npz (sparse matrices)
- X_train_bow.npz, X_test_bow.npz (sparse matrices)
- tfidf_vectorizer.pkl, bow_vectorizer.pkl

EMOTION ENCODING:
""")
for emotion, code in emotion_mapping.items():
    count = (train_df['emotion'] == emotion).sum()
    print(f"  {code}: {emotion} ({count} samples)")

print("\n" + "=" * 80)
print("READY FOR MODEL TRAINING!")
print("Use the patterns from Lab 2 Section 3 for classification")
print("=" * 80)

# ================================================================================
# EXAMPLE: HOW TO USE IN MODEL TRAINING (Same as Lab 2 Section 3)
# ================================================================================
"""
# Load preprocessed data (same as Lab 2)
import pandas as pd
from scipy import sparse

train_df = pd.read_pickle("train_df.pkl")
test_df = pd.read_pickle("test_df.pkl")

X_train = sparse.load_npz('X_train_tfidf.npz')
X_test = sparse.load_npz('X_test_tfidf.npz')
y_train = train_df['emotion']

# Model training (same pattern as Lab 2 Section 3)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Decision Tree (same as Lab 2)
DT_model = DecisionTreeClassifier(random_state=1)
DT_model.fit(X_train, y_train)
y_pred = DT_model.predict(X_test)

# Naive Bayes (same as Lab 2 Exercise 4)
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
y_pred = NB_model.predict(X_test)

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'emotion': y_pred
})
submission.to_csv('submission.csv', index=False)
"""
