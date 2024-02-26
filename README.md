### AlexNet for Cat and Dog Classification
This repository contains an implementation of the AlexNet convolutional neural network (CNN) architecture using Pytorch. The model is trained to classify images of cats and dogs.

# About AlexNet
AlexNet is a convolutional neural network architecture proposed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. It gained significant attention after winning the ImageNet Large Scale Visual Recognition Challenge in 2012, demonstrating remarkable performance in image classification tasks. The architecture consists of five convolutional layers followed by three fully connected layers, along with max-pooling layers and ReLU activation functions.
![alex](https://github.com/MyDarapy/AlexNet-Implementation/assets/125401026/e8db13cc-413f-4f93-aba6-4deff730d662)

# Dataset
The model is trained on a dataset containing images of cats and dogs. The dataset is divided into training and validation sets for model training and evaluation.
The dataset can be found here- https://drive.google.com/drive/folders/1-7jdS8Zt8JQBB6L8Ma545-HmP2qbtnbV?usp=sharing

# Prerequisites 
Python 3.10
PyTorch

# Usage
Clone the Repository:

git clone https://github.com/MyDarapy/alexnet-implementation.git
cd alexnet-cat-dog

Install Dependencies
- Torchvision
- PIL
- Numpy
- Matplotlib
- os
- Torch

Training:
python alexnetscatsanddogs.py
Replace path_to_image.jpg with the path to the image you want to classify.

# Results
After training and evaluation, the model achieves a certain accuracy on the validation set. The performance can be further analyzed through confusion matrices, accuracy, precision, recall, and F1-score metrics.

# Acknowledgments
The implementation is inspired by the original AlexNet paper and PyTorch's documentation.
The dataset used in this project is sourced from [source_link], and we acknowledge their contribution to making it publicly available.

License
This project is licensed under the MIT License - see the LICENSE file for details.
