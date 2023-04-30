import numpy as np 
import pandas as pd 
%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from collections import OrderedDict
from PIL import Image

import torch 
from torch.autograd import Variable
import torch.nn as nn 
from torch.autograd import Function 
from torch.nn.parameter import Parameter 
from torch import optim 
import torch.nn.functional 
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader 

class FashionMNIST(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        fashion_df = pd.read_csv('fashion-mnist_test.csv')
        self.labels = fashion_df.label.values
        self.images = fashion_df.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])
        
        if self.transform:
            img = self.transform(img)

        return img, label

def train_model(model):
    '''
    Function trains the model and prints out the training log.
    '''
    criterion = nn.NLLLoss()
    learning_rate = 0.003
    epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1)
            log_ps = model(images)
            loss = criterion(log_ps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss}")

class LeakyReLU(nn.Module):
    '''
    Implementation of LeakyReLU activation.

    '''
    def __init__(self, in_features, alpha = None):
        super(LeakyReLU,self).__init__()
        self.in_features = in_features

    def forward(self, x):
        return torch.max(torch.zeros(self.in_features), x) + 0.01 * torch.minimum(torch.zeros(self.in_features), x)

class ClassifierSExp(nn.Module):
    '''
    Basic fully-connected network to test Soft Exponential activation.
    '''
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.a1 = LeakyReLU(256)
        self.a2 = LeakyReLU(128)
        self.a3 = LeakyReLU(64)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.a3(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x

transform = transforms.Compose([transforms.ToTensor()])
trainset = FashionMNIST(transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)    
model = ClassifierSExp()
train_model(model)