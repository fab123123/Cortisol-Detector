# Uses the test dataset to get test accuracy

import torch
from cnn import CNN48x48
from torch.utils.data import TensorDataset, DataLoader
LOADED_MODEL_DIR = "../saved_models/cnn48x48.pth"

# load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Recreate model architecture
model = CNN48x48(num_classes=2)
model.load_state_dict(torch.load(LOADED_MODEL_DIR))  # or .pth if full model
model.to(device)
model.eval()  # important! disables dropout, batchnorm in training mode

# Assume X_test: [N,1,48,48], y_test: [N]
# Load saved tensors
test_data = torch.load("../saved_tensors/test_dataset.pt")
X, y = test_data["images"], test_data["labels"]  # X: [N, 1, 48, 48], y: [N]
test_dataset = TensorDataset(X, y)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

correct = 0
total = 0
all_preds = []
all_labels = []

if __name__ == "__main__":
    with torch.no_grad():  # no gradient computation for testing
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")