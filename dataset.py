from typing import Any
import torch
import torch.nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
     
        # text to tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_src.encode(tgt_text).ids
        
        # Calculate padding number
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        #  Raise Error if the number get negative

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long base on the sequence length has defined ') 
        
        # Add SOS, EOS and PADDINGS to the source text
        encoder_input = torch.concat(
            [
                self.sos_token,
                torch.Tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)

                  ]
        )

        decoder_input = torch.concat(
            [
                self.sos_token,
                torch.Tensor(dec_input_tokens, dtype=torch.int64),
                torch.Tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)

                  ]
        )

        label = torch.concat(
            [
                torch.Tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.Tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)

                  ]
        )

        # Check the size of the tensors to make sure they are all seq_len 
        assert encoder_input.size(0) == len(self.seq_len)
        assert decoder_input.size(0) == len(self.seq_len)
        assert label.size(0) == len(self.seq_len)


        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }



def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

size = 4
torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)

import torch

def causal_mask(size):
    return torch.tril(torch.ones(size, size)).unsqueeze(0).int()

# Example decoder input (batch size = 2, sequence length = 5)
decoder_input = torch.tensor([[12, 5, 0, 0, 0],  # First sequence with padding
                              [7, 8, 9, 10, 0]]) # Second sequence with padding
pad_token = 0

# Create the padding mask
padding_mask = (decoder_input != pad_token).unsqueeze(1).int()

# Create the causal mask
seq_len = decoder_input.size(1)
causal_mask = causal_mask(seq_len)

# Combine the masks
decoder_mask = padding_mask & causal_mask


print("Padding Mask:\n", padding_mask)
print("\nCausal Mask:\n", causal_mask)
print("\nDecoder Mask:\n", decoder_mask)