# -*- coding: utf-8 -*-
"""AlexNetscats&dogs.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1TKCgDtQ8SbIj65hcNa-x_NOyBUNkdW6Z
"""

from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from google.colab import drive
drive.mount('/content/drive')

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50

train_images_path = '/content/drive/MyDrive/cats_and_dogs/cats_and_dogs/train'

images_paths = []
labels = []

for filename in os.listdir(train_images_path):
  img_path = os.path.join(train_images_path, filename)

  images_paths.append(img_path)
  label = filename.split(".")[0]
  labels.append(label)


labels = np.array(labels)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the data into training and validation sets.
train_images, gtest_images, train_labels, gtest_labels = train_test_split(images_paths, labels, test_size=0.3, random_state=42)

test_images, val_images, test_labels, val_labels = train_test_split(gtest_images, gtest_labels, test_size=0.1, random_state=42)

train_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomResizedCrop(227, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
    transforms.CenterCrop(227),
    transforms.RandomHorizontalFlip(),
    #transforms.TrivialAugmentWide(), #Automatic Augumentation
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]) #Training augumentation


test_transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std =(0.229, 0.224, 0.225))
    ]) # No augumentation for test set

class CatAndDogDataset(torch.utils.data.Dataset):
    def __init__(self, images_paths, labels, transform=None):
        super(CatAndDogDataset, self).__init__()
        self.images_paths = images_paths
        self.labels = labels
        self.length = len(images_paths)
        self.transform = transform

    def __getitem__(self, idx):
        image = Image.open(self.images_paths[idx])
        return self.transform(image), self.labels[idx]

    def __len__(self):
        return self.length

train_dataset = CatAndDogDataset(train_images, train_labels, transform = train_transform)
val_dataset = CatAndDogDataset(val_images, val_labels, transform = train_transform)
test_dataset = CatAndDogDataset(test_images, test_labels, transform= test_transform)

train_loader =torch.utils.data.DataLoader(dataset= train_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_loader =torch.utils.data.DataLoader(dataset= val_dataset, batch_size = BATCH_SIZE, shuffle = False)
test_loader =torch.utils.data.DataLoader(dataset= test_dataset, batch_size = BATCH_SIZE, shuffle = False)

class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__() # Input_size = b x 3 x 227 x 227
    self.layer1 = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4),  #b x 3 x 55 x 55
                                nn.ReLU(),
                                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                                nn.MaxPool2d(kernel_size=3, stride=2)) #b x 96 x 27 x 27
    self.layer2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),   #b x 256 x 27 x 27
                                nn.ReLU(),
                                nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
                                nn.MaxPool2d(kernel_size=3, stride=2))  #b x 256 x 13 x 13
    self.layer3 = nn.Sequential(nn.Conv2d(256, 384, kernel_size=3, padding=1), #b x 384 x 13 x 13
                                  nn.ReLU())
    self.layer4 = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1), #b x 384 x 13 x 13
                                nn.ReLU())
    self.layer5 = nn.Sequential(nn.Conv2d(384, 256, kernel_size=3, padding=1), #b x 256 x 13 x 13
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2)) #b x 256 x 6 x 6
    self.fc1 = nn.Sequential(nn.Linear(256*6*6, 4096),
                                nn.ReLU(),
                                nn.Dropout1d(p=0.5))
    self.fc2 = nn.Sequential(nn.Linear(4096, 4096),
                                nn.ReLU(),
                                nn.Dropout1d(p=0.5))
    self.fc3 = nn.Sequential(nn.Linear(4096, 1),
                            nn.Sigmoid())


  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = out.view(-1, 256*6*6)
    out = self.fc1(out)
    out = self.fc2(out)
    out = self.fc3(out)
    return out

model = AlexNet().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

n_total_steps = int(len(train_dataset)/BATCH_SIZE) #No of training iteration per epoch.

for epoch in range(NUM_EPOCHS):
  for i, (images, labels) in enumerate (train_loader):
    images = images.reshape(-1, 3, 227, 227).to(DEVICE)
    labels = labels.to(DEVICE)

    outputs = model(images)
    loss = criterion(outputs, labels.unsqueeze(1).float())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 10 == 0: #Evaluate the validation loss after every 10 training iterations
      model.eval()
      with torch.no_grad():
        val_losses = []
        for val_images, val_labels in val_loader:
          val_images = val_images.reshape(-1, 3, 227, 227).to(DEVICE)
          val_labels = val_labels.to(DEVICE)
          preds = model(val_images)
          error = criterion(preds, val_labels.unsqueeze(1).float())
          val_losses.append(error.item())

          print(f"Epoch[{epoch+1}/{NUM_EPOCHS}], Step[{i+1}/{n_total_steps}], Training Loss:{loss.item():.4f}, Validation_Loss: {mean(val_losses):.4f}")
        model.train()

model.eval()
with torch.no_grad():
  n_correct = 0
  n_samples = 0
  for test_images, test_labels in test_loader:
    test_images = test_images.reshape(-1, 3, 227, 227).to(DEVICE)
    test_labels = test_labels.to(DEVICE)
    outputs = model(test_images)
    _, pred = torch.max(outputs, 1)
    n_samples += test_labels.shape[0]
    n_correct += (pred == test_labels).sum().item()
  model_accuracy = 100* (n_correct/n_samples)

  print(f'Test Accuracy = {model_accuracy}')