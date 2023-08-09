import argparse
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim
import seaborn as sns
import numpy as np
from pathlib import Path
import json
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plot
import os

parser = argparse.ArgumentParser(
    description='Argparsing for train.py'
)

parser.add_argument('data_dir', action='store', default='./flowers', nargs='?')
parser.add_argument('--save_dir', action='store', default='./checkpoint.pth')
parser.add_argument('--arch', action='store', default='densenet121', choices=('densenet121', 'vgg16'))
parser.add_argument('--learning_rate', action='store', type=float, default=0.003)
parser.add_argument('--hidden_units', action='store', type=int, default=512)
parser.add_argument('--epochs', action='store', type=int, default=5)
parser.add_argument('--dropout', action='store', type=float, default=0.4)
parser.add_argument('--gpu', action='store', default='gpu')

args = parser.parse_args()

data_dir = args.data_dir
checkpoint_path = args.save_dir
learning_rate = args.learning_rate
model_arch = args.arch
hidden_units = args.hidden_units
device = args.gpu
epochs = args.epochs
dropout = args.dropout


train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
valid_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(root=valid_dir, transform=valid_transforms)
batch_size = 64
trainload = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
validload = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
exec('model = models.' + model_arch + '(pretrained=True)')
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 512)),
                          ('relu1', nn.ReLU()),
                          ('drop', nn.Dropout(p=0.4)),
                          ('fc2', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
devices = torch.device("cuda" if in_arg.gpu == 'cuda' and torch.cuda.is_available() else "cpu")

criteria = nn.NLLLoss()

optimize = optim.Adam(models.classifier.parameters(), lr=0.003)

model.to(devices);
epochs = 5
running_loss = 0
steps = 0
print_every = 20
for epoch in range(epochs):
    for inputs, labels in iter(trainload):
        steps += 1
        inputs, labels = inputs.to(devices), labels.to(devices)
        optimize.zero_grad()
        logps = model.forward(inputs)
        loss = criteria(logps, labels)
        loss.backward()
        optimize.step()
        running_loss += loss.item()
        if steps % print_every == 0:
            model.eval()
        with torch.no_grad():
            test_loss = 0
            accuracy = 0
            for inputs, labels in validload:
                inputs, labels = inputs.to(devices), labels.to(devices)
                logps = models.forward(inputs)
                batch_loss = criteria(logps, labels)
                test_loss += batch_loss.item()
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validload):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validload):.3f}")
                running_loss = 0
                models.train()
models.class_to_idx = train_data.class_to_idx
models.to('cpu')
checkpoint = {'arch': 'densenet121',
              'input_size': 1024,
              'output_size': 102,
              'state_dict': models.state_dict(),
              'class_to_idx': models.class_to_idx,
              'epochs': 5}
torch.save(checkpoint, args.save_dir)
