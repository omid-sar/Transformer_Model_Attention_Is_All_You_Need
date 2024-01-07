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
d_model = 512
vocab_size = 10000
batch_size = 64
seq_len = 100 
embeddings = InputEmbeddings(d_model, vocab_size)
# Create a tensor of random integers representing word indices
# The integers should be in the range [0, vocab_size - 1]
word_list = torch.randint(0, 10000, (batch_size, seq_len))
print('word_list',word_list.shape)
word_embeddings = embeddings.forward(word_list)
print('word_embeddings',word_embeddings.shape)
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
        x = x + self.pe[:, :x.shape[1], :]

        return self.dropout(x)
# -------------------------------------------------------------------------------------
dropout = 0.0
positional_encodding = PositionalEncodding(d_model, seq_len, dropout)
word_emb_pos = positional_encodding.forward(word_embeddings)
print('word_emb_pos',word_emb_pos.shape)


# -------------------------------------------------------------------------------------


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisble by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)


    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1))/ math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value) , attention_scores


    def forward(self, q, k, v):
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) ->(Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1 , self.h*self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
    
# -------------------------------------------------------------------------------------
h = 4
multi_head_attention = MultiHeadAttentionBlock(d_model, h, dropout)




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
    

# -------------------------------------------------------------------------------------
layer_normalization = LayerNormalization()


# -----------------------------------------------------------------------------------



class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias= True) # Define W1,B1 
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True) # Define W2, B2

    def forward(self, x):
        out = self.linear_1(x)
        out = nn.functional.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        return out
# -------------------------------------------------------------------------------------
    
# --------------------------------------------------------------------------------
d_model = torch.tensor([512])
h = torch.tensor([4])
d_k = d_model // h

query = torch.randn(64,100, 512)
key = torch.randn(64,100, 512)
value = torch.randn(64,100, 512)

query = query.view(query.shape[0], query.shape[1], h, d_k).transpose(1, 2)
key = key.view(key.shape[0], key.shape[1], h, d_k).transpose(1, 2)
value = value.view(value.shape[0], value.shape[1], h, d_k).transpose(1, 2)

value.view(value.shape[0], value.shape[1], h, d_k).transpose(1, 2).shape
value.transpose(1,2).contiguous().view(value.shape[0], -1, h*d_k).shape
value.transpose(1,2).contiguous().view(value.shape[0], value.shape[1], h*d_k).shape

print("query:" , query.shape, '\n' , 
      "key:" , key.shape, '\n',
      "value:" , value.shape)


key.transpose(-2,-1).shape


    
# ------------------------------------------------------------------------------------

class ResidualConnection(nn.Module):

    def __init__(self,dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)





