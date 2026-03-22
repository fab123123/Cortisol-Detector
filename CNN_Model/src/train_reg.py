import torch
from torch.utils.data import TensorDataset, DataLoader
from src.cnn import CNN48x48
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Load saved tensors
train_data = torch.load("../saved_tensors/train_dataset2.pt")
X, y = train_data["images"], train_data["labels"]  # X: [N, 1, 48, 48]

# ---- Convert labels for regression ----
y = y.float().unsqueeze(1)  # shape: [N, 1]

# Optional: normalize targets (recommended)
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# Split into train / validation (80/20)
num_samples = X.shape[0]
split = int(0.8 * num_samples)

X_train, X_valid = X[:split], X[split:]
y_train, y_valid = y[:split], y[split:]

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# ---- Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN48x48(num_classes=1).to(device)

# ---- Loss + Optimizer ----
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# TRAINING
num_epochs = 10

train_losses = []
val_losses = []
train_mae_list = []
val_mae_list = []

if __name__ == "__main__":
    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        total = 0
        total_error = 0.0

        print(f"Epoch {epoch+1}")

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # MAE tracking
            error = torch.abs(outputs - labels)
            total_error += error.sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_mae = total_error / total

        train_losses.append(train_loss)
        train_mae_list.append(train_mae)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_error = 0.0

        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                error = torch.abs(outputs - labels)
                val_error += error.sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_mae = val_error / val_total

        val_losses.append(val_loss)
        val_mae_list.append(val_mae)

        print(
            f"Train Loss={train_loss:.4f}, Train MAE={train_mae:.4f}, "
            f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}"
        )

    # ---- Save model ----
    torch.save(model.state_dict(),"../saved_models/cnn48x48_regression.pth")

    # ---- Plot ----
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(10,4))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # MAE
    plt.subplot(1,2,2)
    plt.plot(epochs, train_mae_list, label='Train MAE')
    plt.plot(epochs, val_mae_list, label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE over Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()