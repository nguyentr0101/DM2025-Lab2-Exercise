"""
================================================================================
EMOTION CLASSIFICATION - TEXT PREPROCESSING PIPELINE
================================================================================
Kaggle Competition: Predicting 6 emotions (anger, disgust, fear, sadness, surprise, joy)
This script performs comprehensive text preprocessing following KDD methodology.

Author: Generated for Text Mining Course Homework
================================================================================
"""

import pandas as pd
import numpy as np
import json
import re
import string
from collections import Counter

# NLTK imports
import nltk

# ================================================================================
# NLTK DATA DOWNLOAD - Run this section first if you haven't downloaded NLTK data
# ================================================================================
# Uncomment the following lines if running for the first time:
print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("NLTK data downloaded successfully.")
except Exception as e:
    print(f"Note: NLTK download issue (may already be installed): {e}")
    print("If NLTK data is not installed, run: python -m nltk.downloader all")

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

# Scikit-learn imports for vectorization
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION - MODIFY PATHS AS NEEDED FOR YOUR PYCHARM PROJECT
# ================================================================================
# Option 1: Use absolute paths (recommended for clarity)
# DATA_PATH = "C:/Users/YourName/path/to/data/"

# Option 2: Use relative paths (if files are in same folder as script)
DATA_PATH = ""  # Leave empty if files are in same folder as script

# File names
DATA_IDENTIFICATION_FILE = DATA_PATH + "data_identification.csv"
EMOTION_FILE = DATA_PATH + "emotion.csv"
FINAL_POSTS_FILE = DATA_PATH + "final_posts.json"
SAMPLE_SUBMISSION_FILE = DATA_PATH + "samplesubmission.csv"

# Output file names
OUTPUT_TRAIN_FILE = DATA_PATH + "train_preprocessed.csv"
OUTPUT_TEST_FILE = DATA_PATH + "test_preprocessed.csv"
OUTPUT_TRAIN_VECTORS_FILE = DATA_PATH + "train_vectors.npz"
OUTPUT_TEST_VECTORS_FILE = DATA_PATH + "test_vectors.npz"

print("=" * 80)
print("EMOTION CLASSIFICATION - TEXT PREPROCESSING PIPELINE")
print("=" * 80)

# ================================================================================
# STEP 1: LOAD ALL DATA FILES
# ================================================================================
print("\n" + "=" * 80)
print("STEP 1: LOADING DATA FILES")
print("=" * 80)

# 1.1 Load data_identification.csv (train/test split)
print("\n[1.1] Loading data_identification.csv...")
data_identification = pd.read_csv(DATA_IDENTIFICATION_FILE)
print(f"      Total records: {len(data_identification)}")
print(f"      Columns: {list(data_identification.columns)}")
print(f"      Split distribution:")
print(data_identification['split'].value_counts())

# 1.2 Load emotion.csv (labels for training data)
print("\n[1.2] Loading emotion.csv...")
emotion_df = pd.read_csv(EMOTION_FILE)
print(f"      Total labeled records: {len(emotion_df)}")
print(f"      Columns: {list(emotion_df.columns)}")
print(f"      Emotion distribution:")
print(emotion_df['emotion'].value_counts())

# 1.3 Load final_posts.json (text data)
print("\n[1.3] Loading final_posts.json...")
with open(FINAL_POSTS_FILE, 'r', encoding='utf-8') as f:
    posts_json = json.load(f)

# Extract posts into a DataFrame
posts_data = []
for item in posts_json:
    post = item['root']['_source']['post']
    posts_data.append({
        'id': post['post_id'],
        'text': post['text'],
        'hashtags': post.get('hashtags', [])
    })

posts_df = pd.DataFrame(posts_data)
print(f"      Total posts: {len(posts_df)}")
print(f"      Columns: {list(posts_df.columns)}")
print(f"      Sample post:")
print(f"      ID: {posts_df.iloc[0]['id']}")
print(f"      Text: {posts_df.iloc[0]['text'][:100]}...")

# 1.4 Load samplesubmission.csv (for reference)
print("\n[1.4] Loading samplesubmission.csv...")
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_FILE)
print(f"      Sample submission records: {len(sample_submission)}")
print(f"      Columns: {list(sample_submission.columns)}")
print(f"      Expected emotions: {sample_submission['emotion'].unique()}")

# ================================================================================
# STEP 2: MERGE DATA AND CREATE TRAIN/TEST SPLITS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 2: MERGING DATA AND CREATING TRAIN/TEST SPLITS")
print("=" * 80)

# 2.1 Merge posts with split information
print("\n[2.1] Merging posts with split information...")
merged_df = posts_df.merge(data_identification, on='id', how='left')
print(f"      Merged records: {len(merged_df)}")

# 2.2 Create train and test DataFrames
print("\n[2.2] Creating train and test DataFrames...")
train_df = merged_df[merged_df['split'] == 'train'].copy()
test_df = merged_df[merged_df['split'] == 'test'].copy()
print(f"      Training samples: {len(train_df)}")
print(f"      Test samples: {len(test_df)}")

# 2.3 Add emotion labels to training data
print("\n[2.3] Adding emotion labels to training data...")
train_df = train_df.merge(emotion_df, on='id', how='left')
print(f"      Training samples with labels: {len(train_df[train_df['emotion'].notna()])}")

# Check for any missing labels
missing_labels = train_df['emotion'].isna().sum()
if missing_labels > 0:
    print(f"      WARNING: {missing_labels} training samples missing labels!")
else:
    print("      All training samples have labels.")

print(f"\n      Training emotion distribution:")
print(train_df['emotion'].value_counts())

# ================================================================================
# STEP 3: DATA QUALITY ANALYSIS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA QUALITY ANALYSIS")
print("=" * 80)

# 3.1 Check for missing values
print("\n[3.1] Checking for missing values...")
print("      Training set missing values:")
print(train_df.isnull().sum())
print("\n      Test set missing values:")
print(test_df.isnull().sum())

# 3.2 Check for empty texts
print("\n[3.2] Checking for empty texts...")
train_empty = train_df[train_df['text'].str.strip() == '']
test_empty = test_df[test_df['text'].str.strip() == '']
print(f"      Empty texts in training: {len(train_empty)}")
print(f"      Empty texts in test: {len(test_empty)}")

# 3.3 Check for duplicates
print("\n[3.3] Checking for duplicate texts...")
train_duplicates = train_df['text'].duplicated().sum()
test_duplicates = test_df['text'].duplicated().sum()
print(f"      Duplicate texts in training: {train_duplicates}")
print(f"      Duplicate texts in test: {test_duplicates}")

# 3.4 Text length statistics
print("\n[3.4] Text length statistics...")
train_df['text_length'] = train_df['text'].str.len()
test_df['text_length'] = test_df['text'].str.len()

print("      Training set:")
print(f"        Mean length: {train_df['text_length'].mean():.2f}")
print(f"        Median length: {train_df['text_length'].median():.2f}")
print(f"        Min length: {train_df['text_length'].min()}")
print(f"        Max length: {train_df['text_length'].max()}")

print("      Test set:")
print(f"        Mean length: {test_df['text_length'].mean():.2f}")
print(f"        Median length: {test_df['text_length'].median():.2f}")
print(f"        Min length: {test_df['text_length'].min()}")
print(f"        Max length: {test_df['text_length'].max()}")

# ================================================================================
# STEP 4: TEXT PREPROCESSING FUNCTIONS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 4: DEFINING TEXT PREPROCESSING FUNCTIONS")
print("=" * 80)

# Initialize NLTK resources
# Fallback English stopwords in case NLTK data isn't downloaded
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 
    'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
    've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 
    'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
    "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', 
    "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

try:
    stop_words = set(stopwords.words('english'))
    print("      Using NLTK stopwords.")
except Exception:
    stop_words = FALLBACK_STOPWORDS
    print("      Using fallback stopwords (NLTK data not available).")

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# 4.1 Basic text cleaning function
def clean_text(text):
    """
    Basic text cleaning:
    - Convert to lowercase
    - Remove URLs
    - Remove mentions (@username)
    - Remove hashtags (#tag) or keep hashtag text
    - Remove special characters
    - Remove extra whitespace
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# 4.2 Tokenization function
def tokenize_text(text):
    """
    Tokenize text using NLTK word_tokenize
    Falls back to simple split if NLTK data not available
    """
    if not text:
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        # Fallback to simple tokenization
        return text.split()

# 4.3 Remove stopwords function
def remove_stopwords(tokens, custom_stopwords=None):
    """
    Remove stopwords from token list
    Can add custom stopwords if needed
    """
    if custom_stopwords:
        all_stopwords = stop_words.union(set(custom_stopwords))
    else:
        all_stopwords = stop_words
    
    return [token for token in tokens if token not in all_stopwords]

# 4.4 Lemmatization function with POS tagging
def get_wordnet_pos(tag):
    """
    Convert POS tag to wordnet POS tag for lemmatization
    """
    if tag.startswith('J'):
        return 'a'  # Adjective
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('R'):
        return 'r'  # Adverb
    else:
        return 'n'  # Default to noun

def lemmatize_tokens(tokens):
    """
    Lemmatize tokens using POS tagging for better accuracy
    Falls back to simple lemmatization if POS tagger not available
    """
    if not tokens:
        return []
    
    try:
        pos_tags = pos_tag(tokens)
        lemmatized = []
        
        for token, tag in pos_tags:
            wordnet_pos = get_wordnet_pos(tag)
            try:
                lemma = lemmatizer.lemmatize(token, pos=wordnet_pos)
            except LookupError:
                lemma = token  # Return original if wordnet not available
            lemmatized.append(lemma)
        
        return lemmatized
    except LookupError:
        # Fallback: try lemmatization without POS tagging
        lemmatized = []
        for token in tokens:
            try:
                lemma = lemmatizer.lemmatize(token)
            except LookupError:
                lemma = token
            lemmatized.append(lemma)
        return lemmatized

# 4.5 Stemming function (alternative to lemmatization)
def stem_tokens(tokens):
    """
    Apply Porter stemming to tokens
    """
    return [stemmer.stem(token) for token in tokens]

# 4.6 Complete preprocessing pipeline
def preprocess_text(text, use_lemmatization=True, remove_stops=True, min_token_length=2):
    """
    Complete preprocessing pipeline:
    1. Clean text
    2. Tokenize
    3. Remove stopwords (optional)
    4. Lemmatize or Stem
    5. Filter by minimum token length
    """
    # Step 1: Clean text
    cleaned = clean_text(text)
    
    # Step 2: Tokenize
    tokens = tokenize_text(cleaned)
    
    # Step 3: Remove stopwords
    if remove_stops:
        tokens = remove_stopwords(tokens)
    
    # Step 4: Lemmatize or Stem
    if use_lemmatization:
        tokens = lemmatize_tokens(tokens)
    else:
        tokens = stem_tokens(tokens)
    
    # Step 5: Filter by minimum length
    tokens = [t for t in tokens if len(t) >= min_token_length]
    
    return tokens

def preprocess_text_to_string(text, use_lemmatization=True, remove_stops=True, min_token_length=2):
    """
    Same as preprocess_text but returns a string instead of list
    """
    tokens = preprocess_text(text, use_lemmatization, remove_stops, min_token_length)
    return ' '.join(tokens)

print("      Text preprocessing functions defined successfully.")

# ================================================================================
# STEP 5: APPLY PREPROCESSING TO ALL DATA
# ================================================================================
print("\n" + "=" * 80)
print("STEP 5: APPLYING PREPROCESSING TO ALL DATA")
print("=" * 80)

# 5.1 Apply cleaning
print("\n[5.1] Applying text cleaning...")
train_df['text_cleaned'] = train_df['text'].apply(clean_text)
test_df['text_cleaned'] = test_df['text'].apply(clean_text)
print("      Text cleaning completed.")

# 5.2 Apply full preprocessing (tokens as string for vectorization)
print("\n[5.2] Applying full preprocessing pipeline...")
print("      This may take a few minutes for large datasets...")

train_df['text_preprocessed'] = train_df['text'].apply(
    lambda x: preprocess_text_to_string(x, use_lemmatization=True, remove_stops=True)
)
test_df['text_preprocessed'] = test_df['text'].apply(
    lambda x: preprocess_text_to_string(x, use_lemmatization=True, remove_stops=True)
)
print("      Full preprocessing completed.")

# 5.3 Create tokens list for analysis
print("\n[5.3] Creating token lists for analysis...")
train_df['tokens'] = train_df['text'].apply(
    lambda x: preprocess_text(x, use_lemmatization=True, remove_stops=True)
)
test_df['tokens'] = test_df['text'].apply(
    lambda x: preprocess_text(x, use_lemmatization=True, remove_stops=True)
)
print("      Token lists created.")

# 5.4 Calculate token counts
train_df['token_count'] = train_df['tokens'].apply(len)
test_df['token_count'] = test_df['tokens'].apply(len)

print("\n[5.4] Token count statistics after preprocessing:")
print("      Training set:")
print(f"        Mean tokens: {train_df['token_count'].mean():.2f}")
print(f"        Median tokens: {train_df['token_count'].median():.2f}")
print(f"        Min tokens: {train_df['token_count'].min()}")
print(f"        Max tokens: {train_df['token_count'].max()}")
print(f"        Zero-token samples: {(train_df['token_count'] == 0).sum()}")

print("      Test set:")
print(f"        Mean tokens: {test_df['token_count'].mean():.2f}")
print(f"        Median tokens: {test_df['token_count'].median():.2f}")
print(f"        Min tokens: {test_df['token_count'].min()}")
print(f"        Max tokens: {test_df['token_count'].max()}")
print(f"        Zero-token samples: {(test_df['token_count'] == 0).sum()}")

# ================================================================================
# STEP 6: VOCABULARY ANALYSIS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 6: VOCABULARY ANALYSIS")
print("=" * 80)

# 6.1 Build vocabulary from training data
print("\n[6.1] Building vocabulary from training data...")
all_train_tokens = [token for tokens in train_df['tokens'] for token in tokens]
vocab_counter = Counter(all_train_tokens)

print(f"      Total tokens in training: {len(all_train_tokens)}")
print(f"      Unique tokens (vocabulary size): {len(vocab_counter)}")

# 6.2 Most common words
print("\n[6.2] Top 20 most common words:")
for word, count in vocab_counter.most_common(20):
    print(f"        {word}: {count}")

# 6.3 Rare words analysis
print("\n[6.3] Rare words analysis:")
rare_words = [word for word, count in vocab_counter.items() if count == 1]
print(f"      Words appearing only once: {len(rare_words)} ({len(rare_words)/len(vocab_counter)*100:.2f}%)")

# 6.4 Vocabulary by emotion (for training data)
print("\n[6.4] Vocabulary analysis by emotion:")
for emotion in train_df['emotion'].unique():
    if pd.notna(emotion):
        emotion_tokens = [token for tokens in train_df[train_df['emotion'] == emotion]['tokens'] for token in tokens]
        emotion_counter = Counter(emotion_tokens)
        print(f"\n      {emotion.upper()}:")
        print(f"        Total tokens: {len(emotion_tokens)}")
        print(f"        Unique tokens: {len(emotion_counter)}")
        print(f"        Top 5 words: {emotion_counter.most_common(5)}")

# ================================================================================
# STEP 7: TEXT VECTORIZATION
# ================================================================================
print("\n" + "=" * 80)
print("STEP 7: TEXT VECTORIZATION")
print("=" * 80)

# 7.1 TF-IDF Vectorization
print("\n[7.1] Creating TF-IDF vectors...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,      # Maximum vocabulary size
    min_df=2,                # Minimum document frequency
    max_df=0.95,             # Maximum document frequency
    ngram_range=(1, 2),      # Unigrams and bigrams
    sublinear_tf=True        # Use log normalization
)

# Fit on training data only
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['text_preprocessed'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['text_preprocessed'])

print(f"      TF-IDF training matrix shape: {X_train_tfidf.shape}")
print(f"      TF-IDF test matrix shape: {X_test_tfidf.shape}")
print(f"      Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# 7.2 Count Vectorization (Bag of Words)
print("\n[7.2] Creating Count vectors (Bag of Words)...")
count_vectorizer = CountVectorizer(
    max_features=10000,
    min_df=2,
    max_df=0.95,
    ngram_range=(1, 2)
)

X_train_count = count_vectorizer.fit_transform(train_df['text_preprocessed'])
X_test_count = count_vectorizer.transform(test_df['text_preprocessed'])

print(f"      Count training matrix shape: {X_train_count.shape}")
print(f"      Count test matrix shape: {X_test_count.shape}")

# 7.3 Feature names sample
print("\n[7.3] Sample feature names (TF-IDF):")
feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"      First 20 features: {list(feature_names[:20])}")
print(f"      Last 20 features: {list(feature_names[-20:])}")

# ================================================================================
# STEP 8: PREPARE LABELS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 8: PREPARING LABELS")
print("=" * 80)

# 8.1 Encode labels for training
print("\n[8.1] Encoding emotion labels...")
emotion_mapping = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5
}
reverse_emotion_mapping = {v: k for k, v in emotion_mapping.items()}

train_df['emotion_encoded'] = train_df['emotion'].map(emotion_mapping)
print("      Emotion mapping:")
for emotion, code in emotion_mapping.items():
    count = (train_df['emotion_encoded'] == code).sum()
    print(f"        {code}: {emotion} ({count} samples)")

# 8.2 Prepare y_train
y_train = train_df['emotion_encoded'].values
print(f"\n      y_train shape: {y_train.shape}")
print(f"      y_train dtype: {y_train.dtype}")

# ================================================================================
# STEP 9: SAVE PREPROCESSED DATA
# ================================================================================
print("\n" + "=" * 80)
print("STEP 9: SAVING PREPROCESSED DATA")
print("=" * 80)

# 9.1 Save preprocessed training data
print("\n[9.1] Saving preprocessed training data...")
train_output = train_df[['id', 'text', 'text_cleaned', 'text_preprocessed', 
                         'token_count', 'emotion', 'emotion_encoded']].copy()
train_output.to_csv(OUTPUT_TRAIN_FILE, index=False)
print(f"      Saved to: {OUTPUT_TRAIN_FILE}")

# 9.2 Save preprocessed test data
print("\n[9.2] Saving preprocessed test data...")
test_output = test_df[['id', 'text', 'text_cleaned', 'text_preprocessed', 
                       'token_count']].copy()
test_output.to_csv(OUTPUT_TEST_FILE, index=False)
print(f"      Saved to: {OUTPUT_TEST_FILE}")

# 9.3 Save vectorized data (sparse matrices)
print("\n[9.3] Saving vectorized data...")
from scipy import sparse

# Save TF-IDF vectors
sparse.save_npz(OUTPUT_TRAIN_VECTORS_FILE.replace('.npz', '_tfidf.npz'), X_train_tfidf)
sparse.save_npz(OUTPUT_TEST_VECTORS_FILE.replace('.npz', '_tfidf.npz'), X_test_tfidf)
print(f"      TF-IDF vectors saved.")

# Save Count vectors
sparse.save_npz(OUTPUT_TRAIN_VECTORS_FILE.replace('.npz', '_count.npz'), X_train_count)
sparse.save_npz(OUTPUT_TEST_VECTORS_FILE.replace('.npz', '_count.npz'), X_test_count)
print(f"      Count vectors saved.")

# 9.4 Save vectorizers for later use
import pickle
with open(DATA_PATH + 'tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
with open(DATA_PATH + 'count_vectorizer.pkl', 'wb') as f:
    pickle.dump(count_vectorizer, f)
print("      Vectorizers saved.")

# ================================================================================
# STEP 10: DISPLAY SAMPLE SUBMISSION FORMAT REMINDER
# ================================================================================
print("\n" + "=" * 80)
print("STEP 10: SAMPLE SUBMISSION FORMAT REMINDER")
print("=" * 80)

print("\n      Expected submission format:")
print(sample_submission.head(10).to_string())
print(f"\n      Total predictions needed: {len(sample_submission)}")
print(f"      Emotion categories: {list(emotion_mapping.keys())}")

# ================================================================================
# STEP 11: FINAL SUMMARY
# ================================================================================
print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE - FINAL SUMMARY")
print("=" * 80)

print(f"""
      DATA SUMMARY:
      -------------
      Training samples: {len(train_df)}
      Test samples: {len(test_df)}
      
      PREPROCESSING APPLIED:
      ----------------------
      1. Text cleaning (lowercase, URL/mention removal, special char removal)
      2. Tokenization (NLTK word_tokenize)
      3. Stopword removal (NLTK English stopwords)
      4. Lemmatization (NLTK WordNetLemmatizer with POS tagging)
      5. Minimum token length filter (2 characters)
      
      VECTORIZATION:
      --------------
      TF-IDF features: {X_train_tfidf.shape[1]}
      Count features: {X_train_count.shape[1]}
      N-gram range: (1, 2) - unigrams and bigrams
      
      OUTPUT FILES:
      -------------
      - {OUTPUT_TRAIN_FILE}
      - {OUTPUT_TEST_FILE}
      - train_vectors_tfidf.npz
      - test_vectors_tfidf.npz
      - train_vectors_count.npz
      - test_vectors_count.npz
      - tfidf_vectorizer.pkl
      - count_vectorizer.pkl
      
      LABEL ENCODING:
      ---------------
""")
for emotion, code in emotion_mapping.items():
    print(f"      {code}: {emotion}")

print("\n" + "=" * 80)
print("READY FOR MODEL TRAINING!")
print("=" * 80)

# ================================================================================
# OPTIONAL: DISPLAY SOME PREPROCESSING EXAMPLES
# ================================================================================
print("\n" + "=" * 80)
print("PREPROCESSING EXAMPLES")
print("=" * 80)

print("\n      Sample preprocessing results:")
for i in range(5):
    print(f"\n      Example {i+1}:")
    print(f"        Original: {train_df.iloc[i]['text'][:80]}...")
    print(f"        Cleaned:  {train_df.iloc[i]['text_cleaned'][:80]}...")
    print(f"        Preprocessed: {train_df.iloc[i]['text_preprocessed'][:80]}...")
    print(f"        Emotion: {train_df.iloc[i]['emotion']}")

# ================================================================================
# EXPORT VARIABLES FOR NEXT STEPS (can be imported in another script)
# ================================================================================
"""
To use the preprocessed data in your model training script:

import pandas as pd
from scipy import sparse
import pickle

# Load preprocessed data
train_df = pd.read_csv('train_preprocessed.csv')
test_df = pd.read_csv('test_preprocessed.csv')

# Load TF-IDF vectors
X_train = sparse.load_npz('train_vectors_tfidf.npz')
X_test = sparse.load_npz('test_vectors_tfidf.npz')

# Load vectorizer (if needed for new data)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Get labels
y_train = train_df['emotion_encoded'].values

# Ready for model training!
"""

print("\n" + "=" * 80)
print("SCRIPT EXECUTION COMPLETED SUCCESSFULLY")
print("=" * 80)
