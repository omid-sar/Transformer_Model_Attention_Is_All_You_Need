# Local Modules
from dataset import BilingualDataset, causal_mask


import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

# Huggingface Tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


# ----------------------------------------------------------------------------------------------------
# my cheat sheets 
config = {'tokenizer_file': '../data/raw/tokenizer_file_{}.json', 'datasource': 'tep_en_fa_para',
          'lang_src': 'en', 'lang_tgt': 'fa', 'seq_len': 50, 'batch_size':8
           }
# ----------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------
# help me to realize the Path Module
"""config = {'tokenizer_file': '../data/raw/tokenizer_file_{}.json'}
lang = 'en'
tokenizer_path = Path(config['tokenizer_file'].format(lang))"""


# ----------------------------------------------------------------------------------------------------
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        # Huggigface code
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        # added line of code 
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

# ----------------------------------------------------------------------------------------------------
# the differnce between "opus_books" dataset and "tep_en_fa_para"
"""
load_dataset('tep_en_fa_para')

Max length of source sentence: 37
Max length of target sentence: 37


DatasetDict({
    train: Dataset({
        features: ['translation'],
        num_rows: 612087
    })
})

#SAMPLE FROM TEP DATASET
{
    'translation':
                    {'en': 'i mean , the tartans for shit .',
                    'fa': 'منظورم اينه که تارتان بدرد نميخوره .'}
    }


load_dataset('opus_books', 'en-it')

DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 32332
    })
})


#SAMPLE FROM OPUS DATASET
{
    'id': '4',
 'translation':
                 {'en': 'There was no possibility of taking a walk that day.',
                'it': 'I. In quel giorno era impossibile passeggiare.'}
  }


"""



# ----------------------------------------------------------------------------------------------------


def get_ds(config):
    # Dataset only has a train split, so we divide it
    ds_raw = load_dataset(f"{config['datasource']}", split='train', )

    # Build Tokenoizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90% training, 10% validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


    # Convert train and validate dataset to tokens by padding, mask, SOS, EOS and etc.
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
       src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
       tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
       max_len_src = max(len(src_ids), max_len_src)
       max_len_tgt = max(len(tgt_ids), max_len_tgt)

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

----------------------------------------------------------------------------------------------------   

""""
ds_raw = load_dataset(f"{config['datasource']}", split='train', )
# Build Tokenoizers
tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
# 90% training, 10% validation
train_ds_size = int(0.9 * len(ds_raw))
val_ds_size = len(ds_raw) - train_ds_size
train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])


tep_dataset_sample = train_ds_raw[9]
src_sample = tep_dataset_sample['translation']['en']
tgt_sample = tep_dataset_sample['translation']['fa']

src_token_id = tokenizer_src.encode(src_sample).ids
tgt_token_id = tokenizer_tgt.encode(tgt_sample).ids


enc_input_tokens =tokenizer_src.encode(src_sample).ids
dec_input_tokens =tokenizer_tgt.encode(tgt_sample).ids
seq_len = 50
enc_num_padding_tokens = seq_len - len(enc_input_tokens) - 2

print(f"src_sample:  {src_sample} \n src_token_id: {src_token_id} \n enc_input_tokens: {enc_input_tokens} \n seq_len: {seq_len} \n enc_num_padding_tokens: {enc_num_padding_tokens}")
print(f"tgt_sample:  {tgt_sample} \n tgt_token_id: {tgt_token_id}")


sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)


encoder_input = torch.concat([sos_token, torch.Tensor(enc_input_tokens), eos_token, torch.Tensor([pad_token] * enc_num_padding_tokens)])
encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int()

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0

decoder_mask = causal_mask(seq_len)

print(f" encoder_input: {encoder_input} \n encoder_mask: {encoder_mask} \n encoder_mask.shape: {encoder_mask.shape} \n decoder_mask: {decoder_mask}")
""""
