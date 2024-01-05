import torch 
import torch.nn as nn
import numpy as np
import math


# -------------------------------------------------------------------------------------


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    


# -------------------------------------------------------------------------------------


class PositionalEncodding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create poition vector size (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # Calculate positional encoding to EVEN and ODD columns 
        div_term = torch.exp(torch.arange(0, d_model, 2).float()* (- math.log(10000.0)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension to tensor (seq_len, d_model) -> (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Register "pe" in the buffer of this module:It ensures that the buffer's state is included 
        # when you save (torch.save) and load (torch.load) the model
        # Also we wont learn them , we calcualte once and then just we used it
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + pe[:,x.shap[1],:].requires_grad(False)
        return self.dropout(x)



# -------------------------------------------------------------------------------------
    

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6 ) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive 


    def forward(self, x):
        mean = x.mean( dim=-1,keepdim=True)
        std = x.std( dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

# -----------------------------------------------------------------------------------



class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias= True) # Define W1,B1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True) # Define W2, B2

    def forward(self, x):
        out = self.linear_1(x)
        out = torch.nn.functional.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        return out



