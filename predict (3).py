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
checkpoint = torch.load(chck_path)
imge_path = args.input
topk = args.top_k

exec('model = models.' + checkpoint['arch'] + '(pretrained=True)')
for param in models.parameters():
    param.requires_grad = False
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 512)),
                          ('relu1', nn.ReLU()),
                          ('drop', nn.Dropout(p=0.4)),
                          ('fc2', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
models.classifier = classifier
devices = torch.device("cuda" if in_arg.gpu == 'cuda'and torch.cuda.is_available() else "cpu")
criteria = nn.NLLLoss()
optimize = optim.Adam(models.classifier.parameters(), lr=0.003)
models.to(devices);
def process_image(image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
    tensor_image = transform(image)
    return tensor_image
with open('cat_to_name.json', 'r') as f:
	cat_to_name = json.load(f,)
def predict(image_path, models, top=5):
    processed_image = process_image(image_path)
    processed_image.unsqueeze_(0)
    models.eval()
    probs = torch.exp(models.forward(processed_image))
    top_probabilities, top_class = probs.topk(top)

    idx_to_classes = {}
    for key, value in models.class_to_idx.items():
        idx_to_classes[value] = key

    np_top_labels = top_class[0].numpy()

    top_class = [int(idx_to_classes[label]) for label in np_top_labels]

    
    return top_probabilities, top_class
prob, label = predict(imge_path, models, topk)
prob1 = prob.to('cpu')
prob2 = prob1[0].detach().numpy() 
for i in range(topk):
	print('It is a {} with a probability of {:3f}'.format(cat_to_name[str(label[i])], prob2[i]))