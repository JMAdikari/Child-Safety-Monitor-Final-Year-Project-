"""
Train LSTM activity classifier on extracted pose sequences.

Usage:
    python src/classification/train_lstm.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================
# 1. DATASET
# =============================================

class PoseSequenceDataset(Dataset):
    """Dataset of pose sequences with activity labels."""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


# =============================================
# 2. MODEL
# =============================================

class ActivityLSTM(nn.Module):
    """
    Lightweight LSTM classifier for activity recognition from pose sequences.
    Input: (batch, seq_len, 99) — 33 landmarks × 3 coords
    Output: (batch, num_classes) — activity class probabilities
    """
    
    def __init__(self, input_size=99, hidden_size=128, num_layers=2, 
                 num_classes=4, dropout=0.3):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 99)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        last_hidden = h_n[-1]  # Shape: (batch, hidden_size)
        
        # Classify
        output = self.classifier(last_hidden)
        return output


# =============================================
# 3. TRAINING
# =============================================

def load_data(data_dir="data/processed/pose_sequences"):
    """Load all processed pose sequences and create labels."""
    
    # Class mapping
    # 0 = fall, 1 = climb, 2 = fight, 3 = normal
    class_names = ['fall', 'climb', 'fight', 'normal']
    
    all_sequences = []
    all_labels = []
    
    for class_idx, class_name in enumerate(class_names):
        filepath = os.path.join(data_dir, f"{class_name}_sequences.npy")
        
        if os.path.exists(filepath):
            data = np.load(filepath)
            all_sequences.append(data)
            all_labels.extend([class_idx] * len(data))
            print(f"  Loaded {len(data)} sequences for '{class_name}'")
        else:
            print(f"  WARNING: {filepath} not found — skipping '{class_name}'")
    
    if not all_sequences:
        raise FileNotFoundError("No data files found! Run pose extraction first.")
    
    X = np.concatenate(all_sequences, axis=0)
    y = np.array(all_labels)
    
    print(f"\nTotal: {len(y)} sequences across {len(class_names)} classes")
    
    return X, y, class_names


def train_model():
    """Main training loop."""
    
    print("=" * 60)
    print("TRAINING ACTIVITY CLASSIFIER (LSTM)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    X, y, class_names = load_data()
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training: {len(y_train)} samples")
    print(f"Validation: {len(y_val)} samples")
    
    # Create datasets and dataloaders
    train_dataset = PoseSequenceDataset(X_train, y_train)
    val_dataset = PoseSequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = ActivityLSTM(
        input_size=99,
        hidden_size=128,
        num_layers=2,
        num_classes=len(class_names),
        dropout=0.3
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    num_epochs = 50
    best_val_acc = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # --- Validate ---
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/saved/activity_lstm_best.pth")
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    
    # --- Final Evaluation ---
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 60)
    
    model.load_state_dict(torch.load("models/saved/activity_lstm_best.pth"))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Activity Classification - Confusion Matrix')
    plt.tight_layout()
    plt.savefig('evaluation/confusion_matrices/lstm_validation_cm.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to evaluation/confusion_matrices/lstm_validation_cm.png")
    
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig('evaluation/results/training_curves.png', dpi=150)
    plt.close()
    print("Training curves saved to evaluation/results/training_curves.png")


if __name__ == "__main__":
    train_model()