from random import randrange
from statistics import mode
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchaudio
import matplotlib.pyplot as plt

data = pd.read_csv('Chapter 03/01. MLP Binary Classification/data.csv', header=None)

X = torch.tensor(data.drop(2, axis=1).values, dtype=torch.float)
y = torch.tensor(data[1].values, dtype=torch.float).reshape(-1,1)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        layer1 = self.fc1(x)
        act1 = self.sigmoid(layer1)
        layer2 = self.fc2(act1)
        act2 = self.sigmoid(layer2)
        layer3 = self.fc3(act2)
        out = self.sigmoid(layer3)
        return out


model = NeuralNetwork(X.shape[1], num_hidden1=5, num_hidden2=6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
loss = nn.BCELoss()       
loss_func = []
epochs = 100
for epoch in range(epochs):

    pred = model(X)
    ls = loss(pred, y)
    ls.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_func.append(ls.detach())
print(loss_func)
plt.plot(loss_func)  
plt.show()  