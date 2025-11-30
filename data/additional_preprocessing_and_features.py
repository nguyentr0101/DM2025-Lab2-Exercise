"""
================================================================================
ADDITIONAL PREPROCESSING & FEATURE ENGINEERING
================================================================================
Based on DM2025-Lab1 and DM2025-Lab2 Phase 1 techniques

ADD THESE SECTIONS TO YOUR DM2025-Lab2-Homework.ipynb
================================================================================
"""

# ================================================================================
# PART A: ADDITIONAL PREPROCESSING STEPS (Add after Section 5)
# ================================================================================

"""
Add this code AFTER your Section 5 (Feature Creation - Tokenization)
and BEFORE Section 6 (Feature Engineering - BOW)
"""

# =============================================================================
# 5.2 TEXT CLEANING (Based on Lab 1 - Attribute Transformation)
# =============================================================================
import re
import string

def clean_text(text):
    """
    Clean text data - Based on Lab 1 preprocessing techniques
    
    Steps:
    1. Lowercase conversion
    2. Remove URLs
    3. Remove mentions (@user)
    4. Remove hashtag symbols (keep the text)
    5. Remove special characters
    6. Remove extra whitespace
    """
    if pd.isna(text) or not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtag symbol but keep the text
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("\n[5.2] Cleaning text data...")
train_df['text_cleaned'] = train_df['text'].apply(clean_text)
test_df['text_cleaned'] = test_df['text'].apply(clean_text)

# Show sample
print("Sample cleaned text:")
print(train_df[['text', 'text_cleaned']].head(3))


# =============================================================================
# 5.3 STOPWORD REMOVAL (Based on Lab 1 & Lab 2)
# =============================================================================
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Get English stopwords
stop_words = set(stopwords.words('english'))
print(f"\nNumber of stopwords: {len(stop_words)}")
print(f"Sample stopwords: {list(stop_words)[:10]}")

def remove_stopwords(text):
    """Remove stopwords from text - Based on Lab 1"""
    if pd.isna(text) or not text:
        return ""
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(filtered_tokens)

print("\n[5.3] Removing stopwords...")
train_df['text_no_stopwords'] = train_df['text_cleaned'].apply(remove_stopwords)
test_df['text_no_stopwords'] = test_df['text_cleaned'].apply(remove_stopwords)

# Show sample
print("Sample after stopword removal:")
print(train_df[['text_cleaned', 'text_no_stopwords']].head(3))


# =============================================================================
# 5.4 LEMMATIZATION (Based on Lab 1 - Stemming and Lemmatization)
# =============================================================================
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    """Convert POS tag to WordNet format - Based on Lab 1"""
    if tag.startswith('J'):
        return 'a'  # adjective
    elif tag.startswith('V'):
        return 'v'  # verb
    elif tag.startswith('N'):
        return 'n'  # noun
    elif tag.startswith('R'):
        return 'r'  # adverb
    else:
        return 'n'  # default to noun

def lemmatize_text(text):
    """Lemmatize text with POS tagging - Based on Lab 1"""
    if pd.isna(text) or not text:
        return ""
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(tag)) 
                  for word, tag in pos_tags if len(word) > 1]
    return ' '.join(lemmatized)

print("\n[5.4] Lemmatizing text...")
train_df['text_lemmatized'] = train_df['text_no_stopwords'].apply(lemmatize_text)
test_df['text_lemmatized'] = test_df['text_no_stopwords'].apply(lemmatize_text)

# Show sample
print("Sample after lemmatization:")
print(train_df[['text_no_stopwords', 'text_lemmatized']].head(3))


# =============================================================================
# 5.5 FINAL PREPROCESSED TEXT
# =============================================================================
# Use the fully preprocessed text for feature engineering
train_df['text_preprocessed'] = train_df['text_lemmatized']
test_df['text_preprocessed'] = test_df['text_lemmatized']

print("\n[5.5] Preprocessing complete!")
print(f"Original: {train_df['text'].iloc[0][:80]}...")
print(f"Preprocessed: {train_df['text_preprocessed'].iloc[0][:80]}...")


# ================================================================================
# PART B: ADDITIONAL FEATURE ENGINEERING (Add after Section 7)
# ================================================================================

"""
Add this code AFTER your Section 7 (TF-IDF) 
These are advanced features from Lab 2 Phase 1
"""

# =============================================================================
# 8. WORD2VEC EMBEDDINGS (Based on Lab 2 Phase 1, Section 7)
# =============================================================================
import gensim
from gensim.models import Word2Vec
import numpy as np

print("\n" + "=" * 80)
print("8. FEATURE ENGINEERING - Word2Vec (Lab 2 Phase 1)")
print("=" * 80)

# 8.1 Prepare training corpus (same as Lab 2)
print("\n[8.1] Preparing training corpus...")
train_df['text_tokenized'] = train_df['text_preprocessed'].apply(lambda x: word_tokenize(x) if x else [])
training_corpus = train_df['text_tokenized'].values.tolist()
print(f"Training corpus size: {len(training_corpus)}")
print(f"Sample: {training_corpus[0][:10]}")

# 8.2 Train Word2Vec model (same as Lab 2)
print("\n[8.2] Training Word2Vec model...")
vector_dim = 100      # Same as Lab 2
window_size = 5       # Same as Lab 2
min_count = 2         # Words must appear at least 2 times
training_epochs = 20  # Same as Lab 2

word2vec_model = Word2Vec(
    sentences=training_corpus,
    vector_size=vector_dim,
    window=window_size,
    min_count=min_count,
    epochs=training_epochs,
    workers=4
)

print(f"Word2Vec vocabulary size: {len(word2vec_model.wv)}")

# 8.3 Create sentence embeddings using mean pooling (Lab 2 Exercise 7)
print("\n[8.3] Creating sentence embeddings (mean pooling)...")

def get_sentence_vector(tokens, model, dim=100):
    """
    Convert sentence to vector by averaging word vectors
    Based on Lab 2 Phase 1, Exercise 7
    """
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if len(vectors) == 0:
        return np.zeros(dim)
    
    return np.mean(vectors, axis=0)

# Create Word2Vec features
X_train_w2v = np.array([
    get_sentence_vector(tokens, word2vec_model, vector_dim) 
    for tokens in train_df['text_tokenized']
])

test_df['text_tokenized'] = test_df['text_preprocessed'].apply(lambda x: word_tokenize(x) if x else [])
X_test_w2v = np.array([
    get_sentence_vector(tokens, word2vec_model, vector_dim) 
    for tokens in test_df['text_tokenized']
])

print(f"Word2Vec training features shape: {X_train_w2v.shape}")
print(f"Word2Vec test features shape: {X_test_w2v.shape}")


# =============================================================================
# 9. PRE-TRAINED EMBEDDINGS - GloVe (Based on Lab 2 Phase 1, Section 7.4)
# =============================================================================
print("\n" + "=" * 80)
print("9. FEATURE ENGINEERING - Pre-trained GloVe Embeddings")
print("=" * 80)

import gensim.downloader as api
import ssl
import urllib.request

# Fix SSL certificate issue (same as Lab 2)
ssl._create_default_https_context = ssl._create_unverified_context

print("\n[9.1] Loading pre-trained GloVe model (Twitter, 25-dim)...")
print("This may take a few minutes on first run...")

try:
    glove_model = api.load("glove-twitter-25")
    glove_dim = 25
    print(f"GloVe model loaded! Vocabulary size: {len(glove_model)}")
    
    # Test the model
    print(f"Most similar to 'happy': {glove_model.most_similar('happy', topn=5)}")
    
    # 9.2 Create GloVe sentence embeddings
    print("\n[9.2] Creating GloVe sentence embeddings...")
    
    def get_glove_vector(tokens, model, dim=25):
        """Get sentence embedding using pre-trained GloVe"""
        vectors = []
        for token in tokens:
            if token in model:
                vectors.append(model[token])
        
        if len(vectors) == 0:
            return np.zeros(dim)
        
        return np.mean(vectors, axis=0)
    
    X_train_glove = np.array([
        get_glove_vector(tokens, glove_model, glove_dim) 
        for tokens in train_df['text_tokenized']
    ])
    
    X_test_glove = np.array([
        get_glove_vector(tokens, glove_model, glove_dim) 
        for tokens in test_df['text_tokenized']
    ])
    
    print(f"GloVe training features shape: {X_train_glove.shape}")
    print(f"GloVe test features shape: {X_test_glove.shape}")
    
except Exception as e:
    print(f"Could not load GloVe model: {e}")
    print("Skipping GloVe features...")
    X_train_glove = None
    X_test_glove = None


# =============================================================================
# 10. COMBINED FEATURES (Optional - Advanced)
# =============================================================================
print("\n" + "=" * 80)
print("10. COMBINED FEATURES SUMMARY")
print("=" * 80)

print("""
AVAILABLE FEATURES FOR MODEL TRAINING:

1. TF-IDF Features (from your Section 7):
   - X_train_tfidf: {tfidf_shape}
   
2. BOW Features (from your Section 6):
   - train_data_BOW_features: CountVectorizer output
   
3. Word2Vec Features (NEW - Section 8):
   - X_train_w2v: {w2v_shape}
   
4. GloVe Features (NEW - Section 9):
   - X_train_glove: {glove_shape}

You can:
- Use each feature set separately with different models
- Combine features using scipy.sparse.hstack() or np.hstack()
- Try different combinations to see which works best
""".format(
    tfidf_shape="(train_samples, tfidf_features)",
    w2v_shape=X_train_w2v.shape,
    glove_shape=X_train_glove.shape if X_train_glove is not None else "Not loaded"
))


# =============================================================================
# 11. SAVE ALL FEATURES
# =============================================================================
print("\n" + "=" * 80)
print("11. SAVING ALL FEATURES")
print("=" * 80)

# Save Word2Vec features
np.save('X_train_w2v.npy', X_train_w2v)
np.save('X_test_w2v.npy', X_test_w2v)
print("Saved: X_train_w2v.npy, X_test_w2v.npy")

# Save GloVe features if available
if X_train_glove is not None:
    np.save('X_train_glove.npy', X_train_glove)
    np.save('X_test_glove.npy', X_test_glove)
    print("Saved: X_train_glove.npy, X_test_glove.npy")

# Save Word2Vec model
word2vec_model.save('word2vec_model.model')
print("Saved: word2vec_model.model")

# Save updated dataframes with all preprocessing
train_df.to_pickle("train_df_full.pkl")
test_df.to_pickle("test_df_full.pkl")
print("Saved: train_df_full.pkl, test_df_full.pkl")

print("\n" + "=" * 80)
print("ALL PREPROCESSING AND FEATURE ENGINEERING COMPLETE!")
print("=" * 80)
