from dataclasses import asdict, dataclass
import time
import os
import numpy as np
import json
from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.run_tokenizer import tokenize_file_parallel
from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.nn_modules.transformer_lm import TransformerLM
from cs336_basics.training.train_config import TrainConfig, to_yaml
import random
import torch
from torch import Tensor
from jaxtyping import Bool, Float, Int
from cs336_basics.data.data_loader import get_batch
from cs336_basics.optimizers.adamw import AdamW
from cs336_basics.optimizers.optimizer_utils import lr_cosine_schedule, clip_gradients
from cs336_basics.checkpoints.checkpoint import save_checkpoint
from cs336_basics.torch_utils import cross_entropy


def count_graph_nodes_from_loss(loss) -> int:
    """Count autograd FunctionNodes reachable from this loss."""
    seen = set()
    stack = [loss.grad_fn]  # start at the sink
    n = 0
    while stack:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        n += 1
        # follow edges to parents
        for nxt, _ in getattr(fn, "next_functions", []):
            stack.append(nxt)
    return n


def log_metrics(step: int, 
                log_path: str,
                **metrics):
    record = {"step": step, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    

def make_run_dir(cfg: TrainConfig)->str:
    """
    Creates the directory we put all of the runs results in.
    Name scheme based on whether we are doing a real production run or just testing
    """

    # create directory
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.home_dir+'/'+current_time

    os.makedirs(run_dir, exist_ok=True)
    #TODO handle the case where the file already exists due to 2 experiments running concurrently

    # save cfg as yaml inside it
    to_yaml(cfg=cfg,
            yamlfile=run_dir+"/config.yaml")

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


def validation_loss(valid_array: np.ndarray,
                    val_batch_ids: list[list[int]],
                    model: torch.nn.Module,
                    cfg: TrainConfig,
                    device: torch.device):
    with torch.inference_mode():
        total = 0.0
        val_batch = 0
        for batch_ids in val_batch_ids:
            print('val', val_batch)
            val_batch += 1
            input_batch, output_batch = get_batch(x=valid_array,
                                              batch_size=cfg.batch_size,
                                              context_length=cfg.context_length,
                                              device=device,
                                              batch_indices=batch_ids)
            o = model(input_batch)
            total += cross_entropy(o = o, x = output_batch).item()
        return total / len(val_batch_ids)


def run_training_loop(cfg: TrainConfig,
                  run_dir: str,
                  device: torch.device):
    

    # grab data as memmaps
    train_array = np.memmap(run_dir + '/data/train/tokens.uint16', dtype=np.uint16, mode="r")
    valid_array = np.memmap(run_dir + '/data/valid/tokens.uint16', dtype=np.uint16, mode="r")

    # create the model
    model = TransformerLM(vocab_size=cfg.vocab_size,
                          context_length=cfg.context_length,
                          d_model=cfg.d_model,
                          num_layers=cfg.num_layers,
                          num_heads=cfg.num_heads,
                          d_ff=cfg.d_ff,
                          rope_theta=cfg.rope_theta,
                          device=device,
                          dtype=cfg.dtype,
                          weight_tying=cfg.weight_tying)
    
    # create the optimizer
    optimizer = AdamW(params=model.parameters(),
                      lr=cfg.lr,
                      betas=cfg.betas,
                      weight_decay=cfg.weight_decay,
                      eps=cfg.opt_eps)
    
    # create a minimal log and checkpoint system
    os.mkdir(run_dir + '/logs')
    os.mkdir(run_dir + '/checkpoints')
    log_path = run_dir + '/logs/metrics.jsonl'
    checkpoint_path = run_dir + '/checkpoints'

    # get val batches
    val_batch_ids = [[random.randint(cfg.context_length, len(valid_array) - 1) for i in range(cfg.batch_size)] for j in range(cfg.val_batches)]

    decay = 0.9
    train_loss_avg = None

    for step in range(cfg.total_step_count+1):
        print('train step', step)
        # zero out all the gradients
        optimizer.zero_grad()


        # grab the batch to train on
        input_batch, output_batch = get_batch(x=train_array,
                                              batch_size=cfg.batch_size,
                                              context_length=cfg.context_length,
                                              device=device)
        
        
        # compute the loss with a forward pass
        logits = model(input_batch)
        loss = cross_entropy(o=logits, x=output_batch)
        
        # update our ewma
        if train_loss_avg is None:
           train_loss_avg = loss.item()
        else:
           train_loss_avg += decay * (loss.item() - train_loss_avg)
        
        # every so often we log 
        if step % cfg.checkpoint_interval == 0:

            # prepare a dict to log
            metrics = {}

            # compute the validation loss
            metrics['valid_loss'] = validation_loss(valid_array=valid_array,
                                                    val_batch_ids=val_batch_ids,
                                                    model=model,
                                                    cfg=cfg,
                                                    device=device)
            metrics['train_loss'] = train_loss_avg

            log_metrics(step=step,
                        log_path=log_path,
                        metrics=metrics)
            
            # this is the model AFTER we have trained for step steps. Thus before step number step. 
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=step,
                            out=checkpoint_path+'/'+str(step))
            
        # this is since we wanna log after we are done training
        if step == cfg.total_step_count:
            break
            
        # compute the gradients
        loss.backward()

        # clip them
        clip_gradients(parameters=model.parameters(),
                       max_norm=cfg.max_norm,
                       eps=cfg.grad_clip_eps)
        
        # schedule the learning rate
        current_lr = lr_cosine_schedule(t = step,
                                        alpha_max=cfg.alpha_max,
                                        alpha_min=cfg.alpha_min,
                                        T_w=cfg.warmup_steps,
                                        T_c=cfg.total_step_count-cfg.warmup_steps)
        
        for group in optimizer.param_groups:
            group['lr'] = current_lr

        # finally we do an optimizer update
        optimizer.step()


def train(cfg: TrainConfig)->str:
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
    
    return run_dir






