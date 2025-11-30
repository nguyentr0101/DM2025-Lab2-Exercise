"""
================================================================================
MODEL IMPLEMENTATION STEPS
================================================================================
Add these cells to your DM2025-Lab2-Homework.ipynb after Section 3 header

Based on:
- Lab 2 Phase 1: Section 3 (Decision Tree), Section 4 (Evaluation), 
                 Section 6 (Deep Learning), Exercise 4 (Naive Bayes)
- Lab 1: Section 7 (Model Comparison with Cross-Validation)
================================================================================
"""

# =============================================================================
# 14. PREPARE LABELS (You already have this - keep it)
# =============================================================================
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

# Encode labels
train_df['emotion_encoded'] = train_df['emotion'].map(emotion_mapping)

# Prepare X and y (using TF-IDF features)
X_train = X_train_tfidf
y_train = train_df['emotion']  # String labels for sklearn
y_train_encoded = train_df['emotion_encoded']  # Numeric labels

X_test = X_test_tfidf

print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}")

print("\nLabel distribution:")
print(y_train.value_counts())


# =============================================================================
# 15. TRAIN-VALIDATION SPLIT (For model evaluation)
# =============================================================================
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

print("\n" + "=" * 80)
print("15. TRAIN-VALIDATION SPLIT")
print("=" * 80)

# Split training data for validation (80% train, 20% validation)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_train  # Maintain class distribution
)

print(f"Training set: {X_tr.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")


# =============================================================================
# 16. BASIC MODELS (Based on Lab 2 Phase 1)
# =============================================================================
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("\n" + "=" * 80)
print("16. BASIC MODELS TRAINING & COMPARISON")
print("=" * 80)

# Define models (same as Lab 1 Section 7 and Lab 2)
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

# Cross-validation (same as Lab 1 Section 7)
print("\n[16.1] Cross-Validation Results (5-fold):")
print("-" * 50)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Find best model
best_model_name = max(cv_results, key=lambda x: cv_results[x]['mean'])
print(f"\nBest model by CV: {best_model_name}")


# =============================================================================
# 17. TRAIN BEST MODEL & EVALUATE
# =============================================================================
print("\n" + "=" * 80)
print("17. TRAIN BEST MODEL & EVALUATE")
print("=" * 80)

# Train on full training data
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# Predict on validation set (for evaluation)
y_val_pred = best_model.predict(X_val)

# Evaluation metrics (same as Lab 2 Section 4)
print(f"\n[17.1] {best_model_name} - Validation Results:")
print("-" * 50)
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

# Confusion Matrix (same as Lab 2)
print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, y_val_pred)
print(cm)


# =============================================================================
# 18. CONFUSION MATRIX VISUALIZATION (Based on Lab 2)
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

print("\n" + "=" * 80)
print("18. CONFUSION MATRIX VISUALIZATION")
print("=" * 80)

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    """
    Plot confusion matrix - Based on Lab 2 Section 4
    """
    classes = sorted(classes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           title=title,
           xlabel='Predicted label',
           ylabel='True label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100)
    plt.show()
    print("Saved: confusion_matrix.png")

# Plot
emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']
plot_confusion_matrix(cm, classes=emotion_classes, title=f'{best_model_name} - Confusion Matrix')


# =============================================================================
# 19. PREDICT ON TEST SET & CREATE SUBMISSION
# =============================================================================
print("\n" + "=" * 80)
print("19. PREDICT ON TEST SET & CREATE SUBMISSION")
print("=" * 80)

# Retrain on FULL training data
print("\n[19.1] Retraining on full training data...")
final_model = models[best_model_name]
final_model.fit(X_train, y_train)

# Predict on test set
print("[19.2] Predicting on test set...")
y_test_pred = final_model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'emotion': y_test_pred
})

# Verify submission format
print("\n[19.3] Submission preview:")
print(submission.head(10))
print(f"\nTotal predictions: {len(submission)}")
print(f"Emotion distribution in predictions:")
print(submission['emotion'].value_counts())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n✓ Saved: submission.csv")


# =============================================================================
# =============================================================================
# OPTIONAL ADVANCED TECHNIQUES (BONUS)
# =============================================================================
# =============================================================================

print("\n" + "=" * 80)
print("OPTIONAL: ADVANCED TECHNIQUES")
print("=" * 80)


# =============================================================================
# 20. (OPTIONAL) ENSEMBLE VOTING CLASSIFIER
# =============================================================================
from sklearn.ensemble import VotingClassifier

print("\n[20] Ensemble Voting Classifier")
print("-" * 50)

# Create ensemble of best models
ensemble = VotingClassifier(
    estimators=[
        ('nb', MultinomialNB()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', LinearSVC(max_iter=2000, random_state=42)),
    ],
    voting='hard'  # Use 'soft' if models support predict_proba
)

# Cross-validation
ensemble_scores = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring='accuracy')
print(f"Ensemble CV Accuracy: {ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")

# Train and predict
ensemble.fit(X_train, y_train)
y_ensemble_pred = ensemble.predict(X_test)

# Save ensemble submission
submission_ensemble = pd.DataFrame({
    'id': test_df['id'],
    'emotion': y_ensemble_pred
})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)
print("✓ Saved: submission_ensemble.csv")


# =============================================================================
# 21. (OPTIONAL) HYPERPARAMETER TUNING WITH GRIDSEARCH
# =============================================================================
from sklearn.model_selection import GridSearchCV

print("\n[21] Hyperparameter Tuning (GridSearchCV)")
print("-" * 50)

# Example: Tune Logistic Regression
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'saga']
}

grid_search = GridSearchCV(
    LogisticRegression(max_iter=1000, random_state=42),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Predict with best model
y_tuned_pred = grid_search.predict(X_test)

# Save tuned submission
submission_tuned = pd.DataFrame({
    'id': test_df['id'],
    'emotion': y_tuned_pred
})
submission_tuned.to_csv('submission_tuned.csv', index=False)
print("✓ Saved: submission_tuned.csv")


# =============================================================================
# 22. (OPTIONAL) DEEP LEARNING MODEL (Based on Lab 2 Phase 1, Section 6)
# =============================================================================
print("\n[22] Deep Learning Model (Neural Network)")
print("-" * 50)

try:
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, ReLU, Softmax
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data for Keras
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    def label_encode(le, labels):
        enc = le.transform(labels)
        return keras.utils.to_categorical(enc)
    
    def label_decode(le, one_hot_label):
        dec = np.argmax(one_hot_label, axis=1)
        return le.inverse_transform(dec)
    
    # Convert labels to one-hot encoding
    y_train_onehot = label_encode(label_encoder, y_train)
    
    # Model architecture (same as Lab 2)
    input_shape = X_train.shape[1]
    output_shape = len(label_encoder.classes_)
    
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output_shape}")
    
    # Build model
    model_input = Input(shape=(input_shape,))
    X = model_input
    
    # Hidden layers
    X = Dense(units=256)(X)
    X = ReLU()(X)
    X = Dropout(0.3)(X)
    
    X = Dense(units=128)(X)
    X = ReLU()(X)
    X = Dropout(0.3)(X)
    
    X = Dense(units=64)(X)
    X = ReLU()(X)
    
    # Output layer
    X = Dense(units=output_shape)(X)
    model_output = Softmax()(X)
    
    # Create and compile model
    nn_model = Model(inputs=[model_input], outputs=[model_output])
    nn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    nn_model.summary()
    
    # Train
    print("\nTraining Neural Network...")
    history = nn_model.fit(
        X_train.toarray(),  # Convert sparse to dense
        y_train_onehot,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Predict
    y_nn_pred_proba = nn_model.predict(X_test.toarray())
    y_nn_pred = label_decode(label_encoder, y_nn_pred_proba)
    
    # Save NN submission
    submission_nn = pd.DataFrame({
        'id': test_df['id'],
        'emotion': y_nn_pred
    })
    submission_nn.to_csv('submission_nn.csv', index=False)
    print("✓ Saved: submission_nn.csv")
    
except ImportError:
    print("Keras not installed. Skipping Deep Learning model.")
except Exception as e:
    print(f"Error in Deep Learning: {e}")


# =============================================================================
# 23. (OPTIONAL) TRY WORD2VEC FEATURES WITH DIFFERENT MODEL
# =============================================================================
print("\n[23] Word2Vec Features with Logistic Regression")
print("-" * 50)

# Use Word2Vec features instead of TF-IDF
lr_w2v = LogisticRegression(max_iter=1000, random_state=42)
lr_w2v.fit(X_train_w2v, y_train)

# Cross-validation on Word2Vec
w2v_scores = cross_val_score(lr_w2v, X_train_w2v, y_train, cv=cv, scoring='accuracy')
print(f"Word2Vec + LogReg CV Accuracy: {w2v_scores.mean():.4f} ± {w2v_scores.std():.4f}")

# Predict
y_w2v_pred = lr_w2v.predict(X_test_w2v)

# Save Word2Vec submission
submission_w2v = pd.DataFrame({
    'id': test_df['id'],
    'emotion': y_w2v_pred
})
submission_w2v.to_csv('submission_w2v.csv', index=False)
print("✓ Saved: submission_w2v.csv")


# =============================================================================
# 24. SUMMARY OF ALL SUBMISSIONS
# =============================================================================
print("\n" + "=" * 80)
print("24. SUMMARY OF ALL SUBMISSIONS")
print("=" * 80)

print("""
SUBMISSION FILES CREATED:
-------------------------
1. submission.csv          - Best basic model ({best_model})
2. submission_ensemble.csv - Voting Ensemble (NB + LR + SVM)
3. submission_tuned.csv    - GridSearch tuned Logistic Regression
4. submission_nn.csv       - Deep Learning Neural Network
5. submission_w2v.csv      - Word2Vec + Logistic Regression

RECOMMENDED:
- Try uploading each to Kaggle to see which performs best
- The ensemble or tuned model often performs better
- Compare your local CV scores with Kaggle leaderboard scores
""".format(best_model=best_model_name))

print("\n" + "=" * 80)
print("MODEL IMPLEMENTATION COMPLETE!")
print("=" * 80)
