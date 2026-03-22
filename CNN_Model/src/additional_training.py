# Additional file to train a saved model

from cnn import CNN48x48
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# Load saved tensors
train_data = torch.load("../saved_tensors/train_dataset.pt")
X, y = train_data["images"], train_data["labels"]  # X: [N, 1, 48, 48], y: [N]

# # Split into train / validation if not done already
# # 80% train, 20% valid
num_samples = X.shape[0]
split = int(0.8 * num_samples)

X_train, X_valid = X[:split], X[split:]
y_train, y_valid = y[:split], y[split:]

train_dataset = TensorDataset(X_train, y_train)
valid_dataset = TensorDataset(X_valid, y_valid)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)


# Recreate model
model = CNN48x48(num_classes=2)  # same architecture as before
model.load_state_dict(torch.load("../saved_models/cnn48x48.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

# New optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()  # suitable for multi-class

# New Learning Rate
for param_group in optimizer.param_groups:
    param_group['lr'] = 1e-4

# Scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

if __name__ == "__main__":
    num_epochs = 10  # additional epochs
    for epoch in range(num_epochs):
        print(epoch)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the model
    torch.save(model.state_dict(), "../saved_models/cnn48x48_improved.pth")

