import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import onnx
import numpy as np

from src import config

class PokemonCNN(nn.Module):
    def __init__(self, num_classes):
        super(PokemonCNN, self).__init__()
        # Input shape: (1, 64, 251)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (32, 32, 125)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (64, 16, 62)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (128, 8, 31)

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)) # -> (128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # AdaptiveMaxPool2d returns [B, 128, 1, 1] -> flatten it to [B, 128] for linear layer
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

def train():
    # 0. Setup and Load Data
    train_pkl = os.path.join(config.DATA_PROCESSED_DIR, "train_data.pkl")
    val_pkl = os.path.join(config.DATA_PROCESSED_DIR, "val_data.pkl")

    if not (os.path.exists(train_pkl) and os.path.exists(val_pkl)):
        print("Error: Processed data files missing. Run prepare_data.py first.")
        return

    with open(train_pkl, 'rb') as f: train_list = pickle.load(f)
    with open(val_pkl, 'rb') as f: val_list = pickle.load(f)

    # Convert list of dicts to Tensors for DataLoader
    def create_dataloader(data_list, batch_size):
        specs = torch.stack([d['spectrogram'] for d in data_list])
        labels = torch.tensor([d['label'] for d in data_list], dtype=torch.long)
        dataset = TensorDataset(specs, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = create_dataloader(train_list, config.BATCH_SIZE)
    val_loader = create_dataloader(val_list, 1) # Use small batch for eval

    num_classes = len(set([d['label'] for d in train_list]))
    print(f"Starting training: {len(train_list)} samples, {num_classes} classes.")

    # Model Setup
    device = torch.device("cpu") # Instructions say CPU environment assumed
    model = PokemonCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_acc = 0.0
    epochs = config.EPOCHS

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        avg_loss = running_loss / len(train_loader.dataset)

        # Validation Loop
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.MODELS_DIR, "best_weights.pt"))
            print(f"  -> New Best Accuracy! Saved model.")

    # Final Export to ONNX (Section 6-2)
    print("Exporting best model to ONNX...")
    model.load_state_dict(torch.load(os.path.join(config.MODELS_DIR, "best_weights.pt")))
    model.eval()

    # Create dummy input for ONNX export (Batch size 1)
    dummy_input = torch.randn(1, 1, config.N_MELS, int(config.NUM_SAMPLES / config.HOP_LENGTH + 1)) # Wait... let me check shape in prepare_data
    # In prepare_data: spec is (1, 64, 251). So dummy should be (1, 1, 64, 251)
    dummy_input = torch.randn(1, 1, config.N_MELS, 251) # Hardcoded for now as it's fixed

    onnx_path = os.path.join(config.MODELS_DIR, "pokemon_model.onnx")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        input_names=['spectrogram'],
        output_names=['logits'],
        dynamic_axes={'spectrogram': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}, # Batch size dynamic
        opset_version=12
    )
    print(f"ONNX model saved at {onnx_path}")

if __name__ == "__main__":
    train()
