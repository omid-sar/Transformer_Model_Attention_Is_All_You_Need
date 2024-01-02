import torch 
import torch.nn as nn
import numpy as np
import math


# -------------------------------------------------------------------------------------

d_model = 512
vocab_size = 10000
embedding = nn.Embedding(vocab_size, d_model)

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    



input_emb = InputEmbeddings(512, 3)
input_emb.forward(torch.tensor(2)).shape
# -------------------------------------------------------------------------------------
seq_len = 6
dropout = 0.1
math.log(2)

pe = torch.zeros(seq_len, d_model)
position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
torch.arange(0, d_model, 2).float()* (- math.log(10000.0)/ d_model)
torch.exp(torch.arange(0, d_model, 2).float()* (- math.log(10000.0)/ d_model))

class PositionalEncodding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

    # Create a matrix of shape(seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    # Create po
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)


