# ============================================================
# Step 1: Project Setup - image_classification.py
# Commit: "Initial project setup"
# ============================================================

# ============================================================
# Step 3: Load and Prepare the Dataset
# Commit: "Loaded and visualized CIFAR-10 dataset subset"
# ============================================================
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

# Define dataset transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10 dataset
print("Loading CIFAR-10 dataset...")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use 3 classes: Airplane=0, Automobile=1, Bird=2
selected_classes = [0, 1, 2]
class_names = ["Airplane", "Automobile", "Bird"]

# Filter dataset to selected classes
train_mask = [i for i, t in enumerate(trainset.targets) if t in selected_classes]
test_mask  = [i for i, t in enumerate(testset.targets)  if t in selected_classes]

trainset.data    = trainset.data[train_mask]
trainset.targets = [trainset.targets[i] for i in train_mask]
testset.data     = testset.data[test_mask]
testset.targets  = [testset.targets[i] for i in test_mask]

# Display a 3x3 grid of sample images and save
fig, axes = plt.subplots(3, 3, figsize=(6, 6))
fig.suptitle("CIFAR-10 Sample Images (3 Classes)", fontsize=13, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(trainset.data[i])
    ax.set_title(class_names[trainset.targets[i]], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("sample_images.png", dpi=120)
plt.close()
print("Saved: sample_images.png")
