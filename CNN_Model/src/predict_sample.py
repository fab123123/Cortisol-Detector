import os
import random
from PIL import Image
import torch
from cnn import CNN48x48
from torchvision import transforms

LOADED_MODEL_DIR = "../saved_models/cnn48x48.pth"  # Path to model

# Choose path to folder
choice = random.choice(["high", "low"])
folder_path = "..\\cortisol_dataset\\test\\" + choice
# List files
image_files = [f for f in os.listdir(folder_path) 
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))]

# Choose a random image
random_image = random.choice(image_files)
random_image_path = os.path.join(folder_path, random_image)
print("Random image selected:", random_image_path)

# Open and display image
img = Image.open(random_image_path)
img.show()
print("Cortisol: " + choice)


# Define preprocessing
test_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),  # converts to [C,H,W] tensor
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img = Image.open(random_image_path).convert("RGB")
img = test_transform(img).unsqueeze(0).to(device)  # add batch dim -> [1,1,48,48]

# load device
# Recreate model architecture
model = CNN48x48(num_classes=2)
model.load_state_dict(torch.load(LOADED_MODEL_DIR))  # or .pth if full model
model.to(device)
model.eval()  # important! disables dropout, batchnorm in training mode

with torch.no_grad():  # no gradient needed
    output = model(img)        # output shape: [1, num_classes]
    _, predicted = torch.max(output, 1)  # returns index of max logi

class_names = ["high", "low"]
print(f"Predicted label: {class_names[predicted.item()]}")
