from typing import Any
import torch
import torch.nn
from torch.utils.data import Dataset


torch.Tensor(tokenizer_src.token_)

class BilingualDataset(Dataset):

    def __init__(self, self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
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


        assert encoder_input.size(0) == len(self.seq_len)
        assert decoder_input.size(0) == len(self.seq_len)
        assert label.size(0) == len(self.seq_len)
