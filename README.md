# Cortisol-Detector

## Overview
A Hackathon Project for BeachHacks 9.0 to detect user cortisol.

This project implements an interface that prompts a camera in order to take a picture. Using a Convolutional Neural Network (CNN), the program detects the stress level of the user. Taking into account time, weather, and facial expressions, the program rates the cortisol level of the user. Then, the user is given further advice and an assisstant chatbot in order to reduce these cortisol levels.

---

## Features

* CNN architecture for 48x48 grayscale images
* Regression output (continuous values instead of classes)
* Training with PyTorch
* MAE + Loss tracking
* Model saving and fine-tuning support

* Front end implementation with Streamlit
* Chatbot inquiry with uAgents

## Installation

```bash
pip install -r requirements.txt
```

---

## Dataset

Dataset is from Kaggle: Fer2013. PIP and import kagglehub and run download_dataset.py to download the dataset.
* The dataset features ~30000 48x48 grayscaled images.

---

## Model Details
Models are saved within CNN_Model/saved_models
Model: Convulutional Neural Network
Current results:
* Classification model: 77% success rate
* Regression model: ~.18 MAE

### Training Details:
Classification CNN:
* Loss: `CrossEntropyLoss`
* Optimizer: `Adam`
* Learning Rate: `1e-3`
* Batch Size: `32`

Regression CNN:
* Loss: `SmoothL1Loss`
* Optimizer: `Adam`
* Learning Rate: `1e-4`
* Batch Size: `32`

## Fine-Tuning

To continue training from a saved model, run CNN_Model/src/add_reg_training.py or CNN_Model/src/additional_training.py
Make sure to tweak hyperparameters and model input/output.

---

## Contributing

Feel free to fork the repo and submit pull requests!

---

## License

MIT License

---

## Authors

Ryan Vo - ML Back End
Omar Juarez - Streamlit Front End
Fabien Duran - uAgent Full Stack
