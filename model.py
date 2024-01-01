import torch 
import torch.nn as nn
import numpy as np

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
        return x



seq_len = 6
dropout = 0.1
torch.arange(0, seq_len, dtype=torch.float).shape
torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
torch.arange(0, seq_len, dtype=torch.float).reshape(seq_len, 1)

class PositionalEncodding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
