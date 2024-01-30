# Local Modules
from dataset import BilingualDataset, causal_mask
from model import built_transformer
from config import get_config, get_weights_file_path, latest_weights_file_path



import torch
import torch.nn as nn
from torch.utils.data import DataLoader,  random_split

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Huggingface Tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
import os
import sys
import warnings

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
    if not Path.exists(tokenizer_path):
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
    ds_raw = load_dataset(f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train[:5%]')

        # It only has the train split, so we divide it overselves
    
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
       tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
       max_len_src = max(len(src_ids), max_len_src)
       max_len_tgt = max(len(tgt_ids), max_len_tgt)

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

# ----------------------------------------------------------------------------------------------------   

# ----------------------------------------------------------------------------------------------------   

def get_model(config, src_vocab_size, tgt_vocab_size):
    model = built_transformer(src_vocab_size, tgt_vocab_size, src_seq_len=config['seq_len'], tgt_seq_len=config['seq_len'], d_model=config['d_model'])
    return model
# ----------------------------------------------------------------------------------------------------   


def train_model(config):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

    (Path(f"{config['datasource']}_{config['model_folder']}")).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])

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
        batch_iterator = tqdm(train_dataloader, desc=f" Processing epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
        
            encoder_output = model.encode(encoder_input, encoder_mask) #(B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, decoder_input, decoder_mask, encoder_mask) #(B, seq_len, d_model)
            proj_output = model.project(decoder_output) #(B, seq_len, tgt_vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            #proj_output: (B, seq_len, tgt_vocab_size) -> (B *seq_len, tgt_vocab_size)
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

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step 
        }, model_filename)


if __name__ == '__main__':
   # warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
