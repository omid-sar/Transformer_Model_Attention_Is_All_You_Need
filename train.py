# Local Modules
from dataset import BilingualDataset, causal_mask
from model import built_transformer
from config import get_config, get_weights_file_path



import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


# Huggingface Tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import os
import sys

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
# The differnce between "opus_books" dataset and "tep_en_fa_para"
"""
#SAMPLE FROM TEP DATASET
DatasetDict({
    train: Dataset({
        features: ['translation'],
        num_rows: 612087
    })
})


{
    'translation':
                    {'en': 'i mean , the tartans for shit .',
                    'fa': 'منظورم اينه که تارتان بدرد نميخوره .'}
    }

#SAMPLE FROM OPUS DATASET
DatasetDict({
    train: Dataset({
        features: ['id', 'translation'],
        num_rows: 32332
    })
})



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

# ----------------------------------------------------------------------------------------------------   
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
"""
# ----------------------------------------------------------------------------------------------------   
config = get_config()
def get_model(config, src_vocab_size, tgt_vocab_size):
    model = built_transformer(src_vocab_size, tgt_vocab_size, src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'], d_model=config['d_model'])
    return model


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f" Using device is {device}")
device = torch.device(device)
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
tokenizer_src.get_vocab_size()
tokenizer_tgt.get_vocab_size()
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
model.parameters()
writer = SummaryWriter(config['experiment_name'])



def train_model(config):

    # Choose the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f" Using device is {device}")
    device = torch.device(device)
    os.makedirs(Path(config['model_folder']), exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(mode)
    
    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)



