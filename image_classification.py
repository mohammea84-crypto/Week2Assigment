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


# ============================================================
# Step 4: Implement Classification Models
# ============================================================

# Flatten images: 32x32x3 → 3072
X_train = trainset.data.reshape(len(trainset.data), -1)
y_train = trainset.targets
X_test  = testset.data.reshape(len(testset.data), -1)
y_test  = testset.targets

# ---- 4a: SVM Classifier ----
# Commit: "Implemented SVM classifier"
print("Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train[:3000], y_train[:3000])   # subset for speed
y_pred_svm = svm.predict(X_test[:1000])
svm_accuracy = accuracy_score(y_test[:1000], y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy:.4f}")


# ---- 4b: Softmax Classifier ----
# Commit: "Implemented Softmax classifier"
print("Training Softmax (Logistic Regression)...")
softmax = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
softmax.fit(X_train[:3000], y_train[:3000])
y_pred_softmax = softmax.predict(X_test[:1000])
softmax_accuracy = accuracy_score(y_test[:1000], y_pred_softmax)
print(f"Softmax Accuracy: {softmax_accuracy:.4f}")



# ---- 4c: Two-layer Neural Network ----
# Commit: "Implemented Two-layer Neural Network classifier"
class TwoLayerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNN, self).__init__()
        self.fc1     = nn.Linear(input_size, hidden_size)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

input_size  = 32 * 32 * 3
hidden_size = 128
output_size = len(selected_classes)

model     = TwoLayerNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Normalize pixel values and convert to tensors
X_train_t = torch.tensor(X_train[:3000] / 255.0, dtype=torch.float32)
y_train_t = torch.tensor(y_train[:3000], dtype=torch.long)
X_test_t  = torch.tensor(X_test[:1000]  / 255.0, dtype=torch.float32)

print("Training Neural Network (10 epochs)...")
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

# Evaluate NN
model.eval()
with torch.no_grad():
    preds = model(X_test_t).argmax(dim=1).numpy()
nn_accuracy = accuracy_score(y_test[:1000], preds)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
