import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import json
import onnx
import numpy as np

from src.config import (
    DEVICE, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, 
    DATA_PROCESSED_DIR, MODELS_DIR
)

class PokemonCNN(nn.Module):
    def __init__(self, num_classes):
        super(PokemonCNN, self).__init__()
        # Input shape: [Batch, 1, 64, 251]
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)      # [B, 128, H/16, W/16]
        x = self.gap(x)           # [B, 128, 1, 1]
        x = torch.flatten(x, 1)   # [B, 128]
        x = self.dropout(x)
        logits = self.classifier(x) # [B, num_classes]
        return logits

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load data from disk and ensure they are tensors!
    train_specs_loaded = torch.load(os.path.join(DATA_PROCESSED_DIR, "train_specs.pt"), map_location='cpu')
    train_labels_loaded = torch.load(os.path.join(DATA_PROCESSED_DIR, "train_labels.pt"), map_location='cpu')
    val_specs_loaded = torch.load(os.path.join(DATA_PROCESSED_DIR, "val_specs.pt"), map_location='cpu')
    val_labels_loaded = torch.load(os.path.join(DATA_PROCESSED_DIR, "val_labels.pt"), map_location='cpu')

    # Robust conversion to Tensors (handles both lists and tensors)
    def ensure_tensor(data):
        if isinstance(data, list): return torch.tensor(data)
        return data

    train_x = torch.stack(train_specs_loaded).to(DEVICE, dtype=torch.float32) if not isinstance(train_specs_loaded[0], torch.Tensor) else torch.stack(train_specs_loaded).to(DEVICE, dtype=torch.float32)
    # Wait, my prepare_data uses a list of tensors. So stack is needed.
    if len(train_specs_loaded) > 0 and not isinstance(train_specs_loaded[0], torch.Tensor):
        # It's a list of Tensors (from append in loop). If it was saved with save(), might be different?
        pass # Let me handle the case where prepare_data saves [tensor, tensor] vs single stacked Tensor

    # Standardized robust way:
    if isinstance(train_specs_loaded, list): train_x = torch.stack(train_specs_loaded).to(DEVICE)
    else: train_x = train_specs_loaded.to(DEVICE)

    if isinstance(train_labels_loaded, list): train_y = torch.tensor(train_labels_loaded).to(DEVICE)
    else: train_y = train_labels_loaded.to(DEVICE)

    if isinstance(val_specs_loaded, list): val_x = torch.stack(val_specs_loaded).to(DEVICE)
    else: val_x = val_specs_loaded.to(DEVICE)

    if isinstance(val_labels_loaded, list): val_y = torch.tensor(val_labels_loaded).to(DEVICE)
    else: val_y = val_labels_loaded.to(DEVICE)

    num_classes = int(max(train_y.max(), val_y.max()) + 1)
    print(f"Training with {num_classes} classes on device: {DEVICE}")

    model = PokemonCNN(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=BATCH_SIZE, shuffle=False)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item() * inputs.size(0); correct += (outputs.max(1)[1] == labels).sum().item(); total += labels.size(0)

        model.eval()
        val_acc, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_correct += (outputs.max(1)[1] == labels).sum().item(); val_total += labels.size(0)
        if val_total > 0: val_acc = 100.0 * val_correct / val_total

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {running_loss/total:.4f}, Acc: {correct*100/total if total>0 else 0:.2f}% | Val Acc: {val_acc:.2f}%")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pth"))

    # Export to ONNX (Requirement 7)
    print("Exporting model...")
    if best_acc > 0:
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_model.pth")))
        model.eval()
        dummy_input = torch.randn(1, 1, 64, 251).to(DEVICE)
        onnx_path = os.path.join(MODELS_DIR, "pokemon_model.onnx")
        torch.onnx.export(model, dummy_input, onnx_path, input_names=['spectrogram'], output_names=['logits'], dynamic_axes={'spectrogram': {0: 'batch_size'}, 'logits': {0: 'batch_size'}}, opset_version=12)
        try:
            import onnx
            onnx.load(onnx_path, load_external_data=True)
            onnx.save_model(onnx.load(onnx_path), onnx_path, save_as_external_data=False)
        except Exception as e: print(f"Consolidation notice: {e}")

if __name__ == "__main__": train()
