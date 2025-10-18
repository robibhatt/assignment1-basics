from dataclasses import asdict, dataclass
import time
import os
import yaml
import numpy as np
from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.run_tokenizer import tokenize_file_parallel
from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.tokenizer import Tokenizer
import random
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from cs336_basics.data.data_loader import get_batch

@dataclass
class TrainConfig:
    debug: bool
    seed: int
    production: bool
    train_text_path: str
    val_text_path: str
    tokenizer_dir: str
    num_workers: int
    pretoken_chunk_size: int
    vocab_size: int
    special_tokens: list[str]
    tokenizer_chunk_size: int
    batch_size: int
    context_length: int
    total_step_count: int
    

def make_run_dir(cfg: TrainConfig)->str:
    """
    Creates the directory we put all of the runs results in.
    Name scheme based on whether we are doing a real production run or just testing
    """

    # create directory
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.production:
        run_dir = 'experiments/production/'+current_time
    else:
        run_dir = 'experiments/test/'+current_time

    os.makedirs(run_dir, exist_ok=True)
    #TODO handle the case where the file already exists due to 2 experiments running concurrently

    # save cfg as yaml inside it
    with open(run_dir+"/config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)

    return run_dir


def create_data(cfg: TrainConfig,
                run_dir: str):
    """
    trains a tokenizer based on params from cfg.
    saves the tokenizers params.
    tokenizes the entire train and valid datasets.
    """

    # add the all important end of text token
    updated_special_tokens = [token for token in cfg.special_tokens]
    updated_special_tokens.append("<|endoftext|>")

    
    # create the data dir for the train and valid
    data_dir = run_dir + '/data'
    os.mkdir(data_dir)
    train_dir = data_dir+'/train'
    valid_dir = data_dir+'/valid'
    os.mkdir(train_dir)
    os.mkdir(valid_dir)

    # grab the files where the tokenizer lives
    vocab_filepath = cfg.tokenizer_dir+'/vocab.pkl'
    merges_filepath = cfg.tokenizer_dir+'/merges.pkl'

    # tokenize tran and validation
    tokenize_file_parallel(training_text_filename=cfg.train_text_path,
                           vocab_filepath=vocab_filepath,
                           merges_filepath=merges_filepath,
                           special_tokens=cfg.special_tokens,
                           chunk_size=cfg.tokenizer_chunk_size,
                           num_workers=cfg.num_workers,
                           output_dir=train_dir,
                           vocab_size=cfg.vocab_size)
    tokenize_file_parallel(training_text_filename=cfg.val_text_path,
                           vocab_filepath=vocab_filepath,
                           merges_filepath=merges_filepath,
                           special_tokens=cfg.special_tokens,
                           chunk_size=cfg.tokenizer_chunk_size,
                           num_workers=cfg.num_workers,
                           output_dir=valid_dir,
                           vocab_size=cfg.vocab_size)
    
    # if we are debugging, verify that the tokenizer properly encoded/decoded the data
    if cfg.debug:
        valid_text = ''
        with open(cfg.val_text_path, "rb") as f:
            num_processes = 7
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                string_iterator = FileStringIterator(file = f,
                                                    start = start,
                                                    end = end,
                                                    chunk_size = 10000)
                original_text = ''.join([string for string in string_iterator])
                valid_text = valid_text + original_text

        # decode the validation text from the encoding
        decoded_text = ''
        arr = np.fromfile(valid_dir+'/tokens.uint16', dtype=np.uint16)
        tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                        merges_filepath=merges_filepath,
                                        special_tokens = cfg.special_tokens,
                                        vocab_size=cfg.vocab_size)
        decoded_text = decoded_text + tokenizer.decode(token_id_list=list(arr))

        if valid_text != decoded_text:
            assert(False)


def run_training_loop(cfg: TrainConfig,
                  run_dir: str,
                  device: torch.device):
    
    train_array = np.memmap(run_dir + '/data/train/tokens.uint16', dtype=np.uint16, mode="r")
    valid_array = np.memmap(run_dir + '/data/valid/tokens.uint16', dtype=np.uint16, mode="r")

    for step in range(cfg.total_step_count):
        input_batch, output_batch = get_batch(x=train_array,
                                              batch_size=cfg.batch_size,
                                              context_length=cfg.context_length,
                                              device=device)

    

def train(cfg: TrainConfig):
    # set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # create the directory
    run_dir = make_run_dir(cfg=cfg)

    # populate the data
    create_data(cfg=cfg,
                run_dir=run_dir)
    
    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # run the training loop
    run_training_loop(cfg=cfg,
                      run_dir=run_dir,
                      device=device)






