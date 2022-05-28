#import the needed libraries

from sklearn import neural_network
import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from model import model_train #import your model here
import data_handler as dh
torch.manual_seed(0)


x_train, x_test, y_train, y_test = dh.load_data('Chapter 03/03. MLP Regression/data/turkish_stocks.csv')

x_train, x_test, y_train, y_test = dh.to_batches(x_train, x_test, y_train, y_test, batch_size=20)

neural_model = model_train(input_size=x_train.shape[2], hidden_layer1= 300, hidden_layer2=100, output_size=1 )

criterion = nn.L1Loss()
optimizer = optim.Adam(neural_model.parameters(), lr=0.0001)

# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train
epochs = 40
train_loss = []
test_loss = []
score_loss = []
best_score = 0.95
for e in range(epochs):
    running_loss = 0
    print(f"Epoch: {e+1}/{epochs}")

    for x_train_sample, y_train_label in zip(x_train, y_train):

        
        
        optimizer.zero_grad()
        
        train_output = neural_model(x_train_sample)   # 1) Forward pass
        train_losses = criterion(train_output, y_train_label) # 2) Compute loss
    
        
        train_losses.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
        running_loss += train_losses.item()

        
    
        train_loss.append(running_loss/x_train.shape[0])
        

    neural_model.eval()
    test_score_loss = 0
    running_loss = 0
    with torch.no_grad():
        for x_test_sample, y_test_label in zip(x_test, y_test):
            test_output = neural_model(x_test_sample)
            test_score_loss += accuracy_score(y_test_label, test_output)

            test_losses = criterion(test_output, y_test_label)
            running_loss += test_losses.item()

        score_loss.append(test_score_loss/x_test.shape[0])
        test_loss.append(running_loss/x_test.shape[0])

        if score_loss > best_score:
            torch.save(neural_model, 'best_model.pth')

    neural_model.train()        


# Plots
x_epochs = list(range(epochs))
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(x_epochs, train_loss, marker='o', label='train')
plt.plot(x_epochs, test_loss, marker='o', label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_epochs, score_loss, marker='o',
         c='black', label='accuracy_score')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.axhline(best_score, c='green', ls='--',
            label=f'benchmark_score({best_score})')
plt.legend()

plt.savefig('accuracy_score_losses.jpg')
plt.show()