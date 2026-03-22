import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

# Training data transform
train_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
# Test data transform
test_transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
train_dataset = ImageFolder(root="../cortisol_dataset/train", transform=train_transform)
test_dataset = ImageFolder(root="../cortisol_dataset/test", transform=test_transform)

# Preprocess all images and stack
train_images = []
train_labels = []

for img, label in train_dataset:
    train_images.append(img)
    train_labels.append(label)

test_images = []
test_labels = []

for img, label in test_dataset:
    test_images.append(img)
    test_labels.append(label)

train_images_tensor = torch.stack(train_images)   # shape: [N, C, H, W]
train_labels_tensor = torch.tensor(train_labels)
test_images_tensor = torch.stack(test_images)
test_labels_tensor = torch.tensor(test_labels)

# Save tensors
torch.save({"images": train_images_tensor, "labels": train_labels_tensor}, "../saved_tensors/train_dataset.pt")
torch.save({"images": test_images_tensor, "labels": test_labels_tensor}, "../saved_tensors/test_dataset.pt")