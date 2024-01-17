from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 50,
        "d_model": 512,
        "datasource": "tep_en_fa_para",
        "lang_src": "en",
        "lang_tgt": "fa",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_file_{0}.json",
        "experiment_name": "runs/tmodel"
    }

