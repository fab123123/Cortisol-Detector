import torch
from torch.utils.data import TensorDataset, DataLoader
from cnn import CNN48x48
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

# Load saved tensors
train_data = torch.load("../saved_tensors/train_dataset.pt")
X, y = train_data["images"], train_data["labels"]  # X: [N, 1, 48, 48], y: [N]

# # Split into train / validation if not done already
# # 80% train, 20% valid
num_samples = X.shape[0]
split = int(0.8 * num_samples)

X_train, X_valid = X[:split], X[split:]
y_train, y_valid = y[:split], y[split:]
# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# # Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN48x48(num_classes=2).to(device)

criterion = nn.CrossEntropyLoss()  # suitable for multi-class
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# TRAINING
num_epochs = 10

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

if __name__ == "__main__":
    for epoch in range(num_epochs):
        # ----- Training -----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(epoch)
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    torch.save(model.state_dict(), "../saved_models/cnn48x48.pth")


    epochs = range(1, num_epochs+1)

    # Plot accuracy
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot loss
    plt.subplot(1,2,2)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.show()