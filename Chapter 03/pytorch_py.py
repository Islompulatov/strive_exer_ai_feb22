from traceback import print_tb
import torch
x = torch.rand(2,3,3, dtype=torch.double)

x = torch.rand(2,3)
y = torch.rand(2,3)
print(x)
print(y)
z = x+y
z = torch.add(x,y)
y.add_(x)
print(x)
