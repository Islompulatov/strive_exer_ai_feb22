import torch
import torch.nn.functional as F
from torch import nn

from collections import OrderedDict

# YOUR CODE HERE

def model_train(input_size, hidden_layer1, hidden_layer2, output_size):

    model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_layer1)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
    ('relu2', nn.ReLU()),
    ('logits', nn.Linear(hidden_layer2, output_size))]))

    return model

       



        