import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary

# YOUR CODE HERE
class Net(nn.Module):
   def __init__(self,input_size, hidden_layer1, hidden_layer2, output_size):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(input_size, hidden_layer1)
       self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
       self.fc3 = nn.Linear(input_size, output_size)

   def forward(self, x):
       layer1 = self.fc1(x)
       layer2 = self.fc2(layer1)
       output = self.fc3(layer2)

       return output

model = Net()




        