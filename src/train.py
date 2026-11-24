import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Config (Hardcoded for simplicity in this script, or imported)
from src.config import DATA_PROCESSED_DIR, MODELS_DIR, DEVICE, LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, EPOCHS

def build_model(num_classes):
    """
    Section 7: Lightweight CNN architecture
    Conv2d(4) + ReLU + BN + MaxPool -> Global Average Pooling -> Dropout -> Linear
    """
    class PokemonCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # Input shape: [1, 64, 251] (assuming channel=1)
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2),

                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                # No MaxPool here to keep feature map size manageable before GAP
            )
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(0.3)
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
            x = self.classifier(x)
            return x

    return PokemonCNN(num_classes).to(DEVICE)

def train():
    # Load data
    print("Loading processed data...")
    try:
        train_specs = torch.load(os.path.join(DATA_PROCESSED_DIR, "train_specs.pt"))
        train_labels = torch.load(os.path.join(DATA_PROCESSED_DIR, "train_labels.pt"))
        val_specs = torch.load(os.path.join(DATA_PROCESSED_DIR, "val_specs.pt"))
        val_labels = torch.load(os.path.join(DATA_PROCESSED_DIR, "val_labels.pt"))
    except FileNotFoundError as e:
        print(f"Error: Data not found at {e.filename}. Did you run prepare_data.py?")
        return

    # Convert lists to tensors and stack them (assuming they are [N] or similar)
    # Actually torch.load on a list of Tensors returns a list. We need to stack them.
    X_train = torch.stack(train_specs)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.stack(val_specs)
    y_val = torch.tensor(val_labels, dtype=torch.long)

    num_classes = int(max(y_train.max(), y_val.max()) + 1)
    print(f"Training on {num_classes} classes.")

    dataset_train = TensorDataset(X_train, y_train)
    dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    dataset_val = TensorDataset(X_val, y_val)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='min')

    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 7 # Section 7: 7 epochs

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        for inputs, labels in dataloader_train:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = total_train_loss / len(dataloader_train)

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in dataloader_val:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = total_val_loss / len(dataloader_val)
        val_acc = correct_val / total_val if total_val > 0 else 0
        avg_train_acc = correct_train / total_train if total_train > 0 else 0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Scheduler and Early Stopping
        scheduler.step(avg_val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_weight.pth"))
            print(f"  *** New best Val Acc: {best_val_acc:.4f}! Model saved.")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered!")
                break

    # Final ONNX Export (Section 7)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_weight.pth"), map_location=DEVICE))
    model.eval()
    dummy_input = torch.randn(1, 1, 64, 251).to(DEVICE)
    onnx_path = os.path.join(MODELS_DIR, "pokemon_model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, 
                      verbose=False,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})

    # Save labels to json as well if not already done in prepare_data (though it should be)
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}. Model exported to {onnx_path}")

if __name__ == "__main__":
    train()