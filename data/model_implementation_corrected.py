"""
================================================================================
MODEL IMPLEMENTATION STEPS (CORRECTED - No manual train-val split needed)
================================================================================
Uses Cross-Validation for evaluation instead of manual split
================================================================================
"""

# =============================================================================
# 14. PREPARE LABELS (Keep your existing code)
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
# 15. BASIC MODELS - CROSS VALIDATION (No manual split needed!)
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

# Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Linear SVM": LinearSVC(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate all models
print("\n[15.1] Cross-Validation Results (5-fold):")
print("-" * 50)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_results[name] = {'mean': scores.mean(), 'std': scores.std()}
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")

# Find best model
best_model_name = max(cv_results, key=lambda x: cv_results[x]['mean'])
print(f"\n★ Best model: {best_model_name} ({cv_results[best_model_name]['mean']:.4f})")


# =============================================================================
# 16. EVALUATE BEST MODEL (Using cross_val_predict - no manual split!)
# =============================================================================
print("\n" + "=" * 80)
print("16. EVALUATE BEST MODEL")
print("=" * 80)

# Get cross-validated predictions (each sample predicted when in validation fold)
best_model = models[best_model_name]
y_cv_pred = cross_val_predict(best_model, X_train, y_train, cv=cv)

# Evaluation metrics
print(f"\n[16.1] {best_model_name} - Cross-Validated Results:")
print("-" * 50)
print(f"Accuracy: {accuracy_score(y_train, y_cv_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_train, y_cv_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_train, y_cv_pred)
print(cm)


# =============================================================================
# 17. CONFUSION MATRIX VISUALIZATION
# =============================================================================
import matplotlib.pyplot as plt
import itertools

print("\n" + "=" * 80)
print("17. CONFUSION MATRIX VISUALIZATION")
print("=" * 80)

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    """Plot confusion matrix"""
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
# 18. TRAIN FINAL MODEL & CREATE SUBMISSION
# =============================================================================
print("\n" + "=" * 80)
print("18. TRAIN FINAL MODEL & CREATE SUBMISSION")
print("=" * 80)

# Train on FULL training data
print("\n[18.1] Training final model on all training data...")
final_model = models[best_model_name]
final_model.fit(X_train, y_train)

# Predict on test set
print("[18.2] Predicting on test set...")
y_test_pred = final_model.predict(X_test)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_df['id'],
    'emotion': y_test_pred
})

# Verify submission format
print("\n[18.3] Submission preview:")
print(submission.head(10))
print(f"\nTotal predictions: {len(submission)}")
print(f"\nEmotion distribution in predictions:")
print(submission['emotion'].value_counts())

# Save submission
submission.to_csv('submission.csv', index=False)
print("\n✓ Saved: submission.csv")


# =============================================================================
# =============================================================================
# OPTIONAL ADVANCED TECHNIQUES (For better scores)
# =============================================================================
# =============================================================================

print("\n" + "=" * 80)
print("OPTIONAL: ADVANCED TECHNIQUES")
print("=" * 80)


# =============================================================================
# 19. (OPTIONAL) ENSEMBLE VOTING CLASSIFIER
# =============================================================================
from sklearn.ensemble import VotingClassifier

print("\n[19] Ensemble Voting Classifier")
print("-" * 50)

# Create ensemble of top models
ensemble = VotingClassifier(
    estimators=[
        ('nb', MultinomialNB()),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('svm', LinearSVC(max_iter=2000, random_state=42)),
    ],
    voting='hard'
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
# 20. (OPTIONAL) HYPERPARAMETER TUNING
# =============================================================================
from sklearn.model_selection import GridSearchCV

print("\n[20] Hyperparameter Tuning (GridSearchCV)")
print("-" * 50)

# Tune Logistic Regression
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
# 21. (OPTIONAL) DEEP LEARNING MODEL
# =============================================================================
print("\n[21] Deep Learning Model (Neural Network)")
print("-" * 50)

try:
    import keras
    from keras.models import Model
    from keras.layers import Input, Dense, Dropout, ReLU, Softmax
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train_onehot = keras.utils.to_categorical(label_encoder.transform(y_train))
    
    # Model architecture
    input_shape = X_train.shape[1]
    output_shape = len(label_encoder.classes_)
    
    model_input = Input(shape=(input_shape,))
    X = Dense(units=256)(model_input)
    X = ReLU()(X)
    X = Dropout(0.3)(X)
    X = Dense(units=128)(X)
    X = ReLU()(X)
    X = Dropout(0.3)(X)
    X = Dense(units=64)(X)
    X = ReLU()(X)
    X = Dense(units=output_shape)(X)
    model_output = Softmax()(X)
    
    nn_model = Model(inputs=[model_input], outputs=[model_output])
    nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    print("Training Neural Network...")
    history = nn_model.fit(
        X_train.toarray(),
        y_train_onehot,
        epochs=15,
        batch_size=64,
        validation_split=0.2,
        verbose=1
    )
    
    # Predict
    y_nn_pred_proba = nn_model.predict(X_test.toarray())
    y_nn_pred = label_encoder.inverse_transform(np.argmax(y_nn_pred_proba, axis=1))
    
    # Save
    submission_nn = pd.DataFrame({'id': test_df['id'], 'emotion': y_nn_pred})
    submission_nn.to_csv('submission_nn.csv', index=False)
    print("✓ Saved: submission_nn.csv")
    
except Exception as e:
    print(f"Skipping Deep Learning: {e}")


# =============================================================================
# 22. (OPTIONAL) WORD2VEC FEATURES
# =============================================================================
print("\n[22] Word2Vec Features with Logistic Regression")
print("-" * 50)

try:
    lr_w2v = LogisticRegression(max_iter=1000, random_state=42)
    w2v_scores = cross_val_score(lr_w2v, X_train_w2v, y_train, cv=cv, scoring='accuracy')
    print(f"Word2Vec + LogReg CV Accuracy: {w2v_scores.mean():.4f} ± {w2v_scores.std():.4f}")
    
    lr_w2v.fit(X_train_w2v, y_train)
    y_w2v_pred = lr_w2v.predict(X_test_w2v)
    
    submission_w2v = pd.DataFrame({'id': test_df['id'], 'emotion': y_w2v_pred})
    submission_w2v.to_csv('submission_w2v.csv', index=False)
    print("✓ Saved: submission_w2v.csv")
except Exception as e:
    print(f"Skipping Word2Vec: {e}")


# =============================================================================
# 23. SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("23. SUMMARY")
print("=" * 80)

print("""
SUBMISSION FILES CREATED:
-------------------------
1. submission.csv          - Best basic model (REQUIRED)
2. submission_ensemble.csv - Voting Ensemble (OPTIONAL)
3. submission_tuned.csv    - GridSearch tuned (OPTIONAL)
4. submission_nn.csv       - Deep Learning (OPTIONAL)
5. submission_w2v.csv      - Word2Vec features (OPTIONAL)

Upload to Kaggle and compare scores!
""")

print("\n✓ MODEL IMPLEMENTATION COMPLETE!")
