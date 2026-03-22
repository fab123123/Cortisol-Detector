# Class for Cortisol detector

import torch
from CNN_Model.src import cnn
from torchvision import transforms
from PIL import Image
import cv2

CLASS_MODEL_PATH = "CNN_Model/saved_models/cnn48x48.pth"
REG_MODEL_PATH = "CNN_Model/saved_models/cnn48x48_regression_finetuned.pth"
class FaceCortisol:
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load device
        # Recreate model architecture
        self._class_model = cnn.CNN48x48(num_classes=2)
        self._class_model.load_state_dict(torch.load(CLASS_MODEL_PATH))
        self._class_model.to(self._device)
        self._class_model.eval()

        self._reg_model = cnn.CNN48x48(num_classes=1)
        self._reg_model.load_state_dict(torch.load(REG_MODEL_PATH))
        self._reg_model.to(self._device)
        self._reg_model.eval()

        self._test_transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),  # converts to [C,H,W] tensor
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def predict(self, img) -> float:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = Image.fromarray(img, 'RGB')
        img = self._test_transform(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._class_model(img)
            _, predicted = torch.max(output, 1)

        class_names = ["high", "low"]
        res = .05 if class_names[predicted.item()] == "low" else .45

        with torch.no_grad():
            output = self._reg_model(img)
            score = output.item()

        res += score * .6

        return res
