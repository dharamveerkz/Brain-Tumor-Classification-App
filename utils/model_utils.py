import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
from layers.cbam import CBAM

# Load the models
def load_models():
    # Standard ResNet-18 (Scratch)
    model_scratch = models.resnet18(weights=None)
    model_scratch.fc = nn.Linear(model_scratch.fc.in_features, 4)

    # Standard ResNet-18 (Pretrained)
    model_pretrained = models.resnet18(weights="IMAGENET1K_V1")
    model_pretrained.fc = nn.Linear(model_pretrained.fc.in_features, 4)

    # ResNet-18 with CBAM
    model_cbam = models.resnet18(weights="IMAGENET1K_V1")
    features = list(model_cbam.children())[:-2]
    features.append(CBAM(512))
    backbone = nn.Sequential(*features)
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(512, 4)
    )
    model_cbam = nn.Sequential(backbone, classifier)

    # Load state dictionaries
    model_scratch.load_state_dict(torch.load('models/scratch/best_model.pth', map_location=torch.device('cpu')))
    model_pretrained.load_state_dict(torch.load('models/pretrained/best_model.pth', map_location=torch.device('cpu')))
    model_cbam.load_state_dict(torch.load('models/cbam/best_model.pth', map_location=torch.device('cpu')))

    # Set models to evaluation mode
    model_scratch.eval()
    model_pretrained.eval()
    model_cbam.eval()

    return model_scratch, model_pretrained, model_cbam

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load image function
def load_image(file_path):
    image = Image.open(file_path).convert('RGB')
    return transform(image).unsqueeze(0), image