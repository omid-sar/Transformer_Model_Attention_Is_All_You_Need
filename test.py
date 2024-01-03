import torch 
import torch.nn as nn    
from torch.autograd import grad
import  torch.nn.functional as F


x = torch.tensor([3.])
w = torch.tensor([2.], requires_grad=True)
b = torch.tensor([2.], requires_grad=True)
a = F.relu( w*x + b)
a

