import torch
from cnn import CNN48x48
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the same model architecture
model = CNN48x48(num_classes=1).to(device)
# Load saved state dict
model.load_state_dict(torch.load("../saved_models/cnn48x48_regression.pth", map_location=device))


train_data = torch.load("../saved_tensors/train_dataset2.pt")
test_data = torch.load("../saved_tensors/test_dataset2.pt")

X_train, y_train = train_data["images"], train_data["labels"].unsqueeze(1)
X_valid, y_valid = test_data["images"], test_data["labels"].unsqueeze(1)

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)



# New optimizer (Adam or AdamW)
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # lower LR for fine-tuning
# Loss function
criterion = nn.SmoothL1Loss()
# Number of epochs
num_epochs = 10

if __name__ == "__main__":
    for epoch in range(num_epochs):
        print(epoch+1)
        model.train()
        running_loss = 0.0
        total_error = 0.0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total_error += torch.abs(outputs - labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_mae = total_error / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_error = 0.0
        val_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_error += torch.abs(outputs - labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_mae = val_error / val_total

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train MAE={train_mae:.4f}, "
              f"Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}")

    # Save the fine-tuned model
    torch.save(model.state_dict(),"../saved_models/cnn48x48_regression_finetuned.pth")