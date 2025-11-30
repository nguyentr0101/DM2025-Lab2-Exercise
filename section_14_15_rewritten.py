"""
================================================================================
REWRITTEN SECTION 14 & 15
================================================================================
Replace your old Section 14 and 15 with this code
================================================================================
"""

# =============================================================================
# 14. PREPARE LABELS & COMPARE FEATURES
# =============================================================================
print("\n" + "=" * 80)
print("14. PREPARE LABELS & COMPARE FEATURES")
print("=" * 80)

# 14.1 Define emotion mapping
emotion_mapping = {
    'anger': 0,
    'disgust': 1,
    'fear': 2,
    'joy': 3,
    'sadness': 4,
    'surprise': 5
}
reverse_mapping = {v: k for k, v in emotion_mapping.items()}

# Encode labels
train_df['emotion_encoded'] = train_df['emotion'].map(emotion_mapping)
y_train = train_df['emotion']  # String labels for sklearn
y_train_encoded = train_df['emotion_encoded']  # Numeric labels

print(f"y_train shape: {y_train.shape}")
print(f"\nLabel distribution:")
print(y_train.value_counts())

# 14.2 Compare all feature sets
print("\n" + "-" * 50)
print("14.2 COMPARING ALL FEATURE SETS")
print("-" * 50)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Define all available features
features = {
    "TF-IDF": (X_train_tfidf, X_test_tfidf),
    "BOW": (train_data_BOW_features, test_data_BOW_features),
    "Word2Vec": (X_train_w2v, X_test_w2v),
}

# Add GloVe if available
if X_train_glove is not None:
    features["GloVe"] = (X_train_glove, X_test_glove)

print("\nFeature Comparison (Logistic Regression, 5-fold CV):")
print("-" * 50)

best_score = 0
best_feature = None
feature_results = {}

for name, (X_tr, X_te) in features.items():
    scores = cross_val_score(lr, X_tr, y_train, cv=cv, scoring='accuracy')
    feature_results[name] = {'mean': scores.mean(), 'std': scores.std(), 'data': (X_tr, X_te)}
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_feature = name

print(f"\n★ Best feature: {best_feature} ({best_score:.4f})")

# 14.3 Use the best feature set
print("\n" + "-" * 50)
print(f"14.3 USING BEST FEATURE: {best_feature}")
print("-" * 50)

X_train, X_test = feature_results[best_feature]['data']

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")


# =============================================================================
# 15. BASIC MODELS - CROSS VALIDATION
# =============================================================================
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("\n" + "=" * 80)
print("15. BASIC MODELS - CROSS VALIDATION")
print("=" * 80)

# Check if features are sparse or dense (some models need sparse)
from scipy.sparse import issparse
is_sparse = issparse(X_train)
print(f"Features are sparse: {is_sparse}")

# Define models based on feature type
if is_sparse:
    # Sparse features (TF-IDF, BOW) - can use all models
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
else:
    # Dense features (Word2Vec, GloVe) - skip Naive Bayes (needs non-negative)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    print("Note: Skipping Naive Bayes (requires non-negative features)")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n[15.1] Cross-Validation Results (5-fold) using {best_feature}:")
print("-" * 50)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Find best model
best_model_name = max(cv_results, key=lambda x: cv_results[x]['mean'])
print(f"\n★ Best model: {best_model_name} ({cv_results[best_model_name]['mean']:.4f})")
print(f"★ Best feature: {best_feature}")
