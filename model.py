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


    def forward(self, q, k, v, mask):
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
q = word_emb_pos
k = word_emb_pos
v = word_emb_pos
mask = torch.ones(1, 1, seq_len, seq_len)
multi_head_attention = MultiHeadAttentionBlock(d_model, h, dropout)
multi_head = multi_head_attention.forward(q, k, v, mask)
print('concatinated multi_head:', multi_head.shape)
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
normal_multi_head = layer_normalization.forward(multi_head)
print("normalized concatinated multi_head: ", normal_multi_head.shape)
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
d_ff = 2048
feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
fully_connected_layer = feed_forward_block.forward(normal_multi_head)
print('fully connected feedforwarded normalized concatinated multi_head:', fully_connected_layer.shape)
# ------------------------------------------------------------------------------------

class ResidualConnection(nn.Module):

    def __init__(self,dropout: float) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


# ------------------------------------------------------------------------------------

# the original code has an axtra "feature" attribute 
# self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])
class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout:float) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

# ------------------------------------------------------------------------------------

class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# ------------------------------------------------------------------------------------
    
class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_msk, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_msk))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
# ------------------------------------------------------------------------------------
    
class Decoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask ):
        for layer in self.layers:
            x = layer( x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
# ------------------------------------------------------------------------------------
    
class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return self.proj(x)
# ------------------------------------------------------------------------------------
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, projection_layer: ProjectionLayer,
                 src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncodding, tgt_pos: PositionalEncodding) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
    

    def encode(self, src, src_mask):
        # (Batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor, src_mask: torch.Tensor):
         # (Batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
         # (Batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
# ------------------------------------------------------------------------------------


def built_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512,
                      N: int=6, h: int=6, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncodding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncodding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    # Initilizing : Encoder-> EncoderBlock -> MultiHeadAttention/ FeedForwrd/ ResidualCoonection 
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block,decoder_self_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer 
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer 
    transformer = Transformer(encoder, decoder, projection_layer, src_embed, tgt_embed, src_pos, tgt_pos)

    # Initialize the prameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer
