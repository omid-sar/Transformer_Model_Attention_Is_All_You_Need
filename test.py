import torch 
import torch.nn as nn    
from torch.autograd import grad
import  torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.tensor([3.])
w = torch.tensor([2.], requires_grad=True)
b = torch.tensor([2.], requires_grad=True)
a = F.relu( w*x + b)
a

grad(outputs=a, inputs=b, retain_graph=True)



# ----------------------------------------------------------------------

import pandas as pd

df = pd.read_csv('data/test_iris/iris.data', index_col=None, header=None)
df.columns = ['x1', 'x2', 'x3', 'x4', 'y']
df = df.iloc[50:150]
df['y'] = df['y'].apply(lambda x: 0 if x == 'Iris-versicolor' else 1)
df

# Assign features and target

X = torch.tensor(df[['x2', 'x3']].values, dtype=torch.float)
y = torch.tensor(df['y'].values, dtype=torch.int)

# Shuffling & train/test split

torch.manual_seed(123)
shuffle_idx = torch.randperm(y.size(0), dtype=torch.long)

X, y = X[shuffle_idx], y[shuffle_idx]

percent70 = int(shuffle_idx.size(0)*0.7)

X_train, X_test = X[shuffle_idx[:percent70]], X[shuffle_idx[percent70:]]
y_train, y_test = y[shuffle_idx[:percent70]], y[shuffle_idx[percent70:]]

# Normalize (mean zero, unit variance)

mu, sigma = X_train.mean(dim=0), X_train.std(dim=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma