
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
data = pd.read_csv('Chapter 03/01. MLP Binary Classification/data.csv', header=None)

X = torch.tensor(data.drop(2, axis=1).values, dtype=torch.float)
y = torch.tensor(data[2].values, dtype=torch.float).reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2, num_hidden3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, num_hidden1)
        self.fc2 = nn.Linear(num_hidden1, num_hidden2)
        self.fc3 = nn.Linear(num_hidden2,num_hidden3)
        self.fc4 = nn.Linear(num_hidden3, 1)
        self.sigmoid = nn.Sigmoid()
        self.logsigmoid = nn.LogSigmoid()
    
    def forward(self, x):
        layer1 = self.fc1(x)
        act1 = self.sigmoid(layer1)
        layer2 = self.fc2(act1)
        act2 = self.sigmoid(layer2)
        layer3 = self.fc3(act2)
        act3 = self.logsigmoid(layer3)
        layer4 = self.fc4(act3)
        out = self.sigmoid(layer4)
        return out


model = NeuralNetwork(X.shape[1], num_hidden1=5, num_hidden2=6, num_hidden3=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
loss = nn.BCELoss()     
train_loss = []
test_loss = []
test_acc = []

epochs = 100

for epoch in range(epochs):

    pred = model(X_train)
    ls = loss(y_train, pred)
    ls.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_loss.append(ls.item())

    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        ls_test = loss(y_test, test_pred)
        test_loss.append(ls.item())
        test_pred.detach().apply_(lambda x: 1 if x >= 0.5 else 0.)

        acc = (test_pred == y_test).sum()/len(y_test)
        test_acc.append(acc)

    # w.grad.zero_()    
    # optimizer.step()
    # optimizer.zero_grad()
    # if epoch % 2 == 0:
    #     loss_func.append(ls.detach())
    model.train()
# print(test_loss)
print(test_acc)
# plt.plot(loss_func)  
# plt.show()  