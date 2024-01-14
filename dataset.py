import torch
import torch.nn
from torch.utils.data import Dataset


torch.Tensor(tokenizer_src.token_)

class BilingualDataset(Dataset):

    def __init__(self, self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
