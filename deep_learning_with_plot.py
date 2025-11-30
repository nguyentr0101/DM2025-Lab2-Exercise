"""
================================================================================
UPDATED DEEP LEARNING CODE WITH TRAINING PLOT
================================================================================
Replace your Section 21 (Deep Learning) with this code
================================================================================
"""

# =============================================================================
# 21. DEEP LEARNING WITH PYTORCH
# =============================================================================
print("\n" + "=" * 80)
print("21. DEEP LEARNING MODEL (PyTorch)")
print("=" * 80)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # Split for validation (to track training vs validation loss)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.toarray(), y_encoded, 
        test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Convert to tensors
    X_tr_tensor = torch.FloatTensor(X_tr)
    y_tr_tensor = torch.LongTensor(y_tr)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    X_test_tensor = torch.FloatTensor(X_test.toarray())

    # DataLoader
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model
    class EmotionClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

    model = EmotionClassifier(X_train.shape[1], len(label_encoder.classes_)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training with history tracking
    print("\nTraining Neural Network...")
    num_epochs = 15
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            X_val_dev = X_val_tensor.to(device)
            y_val_dev = y_val_tensor.to(device)
            val_outputs = model(X_val_dev)
            val_loss = criterion(val_outputs, y_val_dev).item()
            _, val_predicted = torch.max(val_outputs, 1)
            val_acc = (val_predicted == y_val_dev).sum().item() / len(y_val_dev)
        
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Plot Training History
    print("\nPlotting training history...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot Loss
    axes[0].plot(range(1, num_epochs+1), train_losses, 'b-', label='Training Loss')
    axes[0].plot(range(1, num_epochs+1), val_losses, 'r-', label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Accuracy
    axes[1].plot(range(1, num_epochs+1), train_accuracies, 'b-', label='Training Accuracy')
    axes[1].plot(range(1, num_epochs+1), val_accuracies, 'r-', label='Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('deep_learning_training.png', dpi=100)
    plt.show()
    print("Saved: deep_learning_training.png")

    # Final validation accuracy
    print(f"\nFinal Validation Accuracy: {val_accuracies[-1]:.4f}")

    # Retrain on full data for final prediction
    print("\nRetraining on full training data for final prediction...")
    X_full_tensor = torch.FloatTensor(X_train.toarray())
    y_full_tensor = torch.LongTensor(y_encoded)
    full_dataset = TensorDataset(X_full_tensor, y_full_tensor)
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=True)

    model_final = EmotionClassifier(X_train.shape[1], len(label_encoder.classes_)).to(device)
    optimizer_final = torch.optim.Adam(model_final.parameters(), lr=0.001)

    model_final.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in full_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer_final.zero_grad()
            outputs = model_final(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer_final.step()

    # Predict on test set
    model_final.eval()
    with torch.no_grad():
        y_pred_proba = model_final(X_test_tensor.to(device))
        y_pred_idx = torch.argmax(y_pred_proba, dim=1).cpu().numpy()

    y_nn_pred = label_encoder.inverse_transform(y_pred_idx)

    # Save submission
    submission_nn = pd.DataFrame({'id': test_df['id'], 'emotion': y_nn_pred})
    submission_nn.to_csv('submission_nn.csv', index=False)
    print("✓ Saved: submission_nn.csv")

except ImportError:
    print("PyTorch not installed. Run: pip install torch")
except Exception as e:
    print(f"Error: {e}")
