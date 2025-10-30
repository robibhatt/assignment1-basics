from dataclasses import asdict
import time
import os
import numpy as np
import json
import wandb
from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.run_tokenizer import tokenize_file_parallel
from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.nn_modules.transformer_lm import TransformerLM
from cs336_basics.training.train_config import TrainConfig, to_yaml, flatten_dict
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


def make_run_dir(cfg: TrainConfig) -> str:
    """
    Creates the directory we put all of the runs results in.
    Name scheme based on whether we are doing a real production run or just testing
    """
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = cfg.out.home_dir + '/' + current_time

    os.makedirs(run_dir, exist_ok=True)
    # save cfg as yaml inside it
    to_yaml(cfg=cfg, yamlfile=run_dir + "/config.yaml")
    return run_dir


def create_data(cfg: TrainConfig, run_dir: str):
    """
    Tokenize the entire train and valid datasets.
    """
    # add the all important end of text token
    updated_special_tokens = [token for token in cfg.data.special_tokens]
    updated_special_tokens.append("<|endoftext|>")

    # create the data dir for the train and valid
    data_dir = run_dir + '/data'
    os.mkdir(data_dir)
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    os.mkdir(train_dir)
    os.mkdir(valid_dir)

    # grab the files where the tokenizer lives
    vocab_filepath = cfg.data.tokenizer_dir + '/vocab.pkl'
    merges_filepath = cfg.data.tokenizer_dir + '/merges.pkl'

    # tokenize train and validation
    tokenize_file_parallel(training_text_filename=cfg.data.train_text_path,
                           vocab_filepath=vocab_filepath,
                           merges_filepath=merges_filepath,
                           special_tokens=cfg.data.special_tokens,
                           chunk_size=cfg.data.tokenizer_chunk_size,
                           num_workers=cfg.data.num_workers,
                           output_dir=train_dir,
                           vocab_size=cfg.model.vocab_size)
    tokenize_file_parallel(training_text_filename=cfg.data.val_text_path,
                           vocab_filepath=vocab_filepath,
                           merges_filepath=merges_filepath,
                           special_tokens=cfg.data.special_tokens,
                           chunk_size=cfg.data.tokenizer_chunk_size,
                           num_workers=cfg.data.num_workers,
                           output_dir=valid_dir,
                           vocab_size=cfg.model.vocab_size)

    # if we are debugging, verify that the tokenizer properly encoded/decoded the data
    if cfg.out.debug:
        valid_text = ''
        with open(cfg.data.val_text_path, "rb") as f:
            num_processes = 7
            boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                string_iterator = FileStringIterator(file=f,
                                                     start=start,
                                                     end=end,
                                                     chunk_size=10000)
                original_text = ''.join([string for string in string_iterator])
                valid_text = valid_text + original_text

        # decode the validation text from the encoding
        decoded_text = ''
        arr = np.fromfile(valid_dir + '/tokens.uint16', dtype=np.uint16)
        tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                         merges_filepath=merges_filepath,
                                         special_tokens=cfg.data.special_tokens,
                                         vocab_size=cfg.model.vocab_size)
        decoded_text = decoded_text + tokenizer.decode(token_id_list=list(arr))

        if valid_text != decoded_text:
            assert False


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
                                                  batch_size=cfg.data.batch_size,
                                                  context_length=cfg.model.context_length,
                                                  device=device,
                                                  batch_indices=batch_ids)
            o = model(input_batch)
            total += cross_entropy(o=o, x=output_batch).item()
            if cfg.out.debug and val_batch == 1:
                return total
        return total / len(val_batch_ids)


def run_training_loop(cfg: TrainConfig, run_dir: str, device: torch.device):
    # grab data as memmaps
    train_array = np.memmap(run_dir + '/data/train/tokens.uint16', dtype=np.uint16, mode="r")
    valid_array = np.memmap(run_dir + '/data/valid/tokens.uint16', dtype=np.uint16, mode="r")

    # create the model
    model = TransformerLM(vocab_size=cfg.model.vocab_size,
                          context_length=cfg.model.context_length,
                          d_model=cfg.model.d_model,
                          num_layers=cfg.model.num_layers,
                          num_heads=cfg.model.num_heads,
                          d_ff=cfg.model.d_ff,
                          rope_theta=cfg.model.rope_theta,
                          device=device,
                          dtype=getattr(torch, cfg.model.dtype),
                          weight_tying=cfg.model.weight_tying)

    # create the optimizer
    optimizer = AdamW(params=model.parameters(),
                      lr=cfg.optim.lr,
                      betas=tuple(cfg.optim.betas),
                      weight_decay=cfg.optim.weight_decay,
                      eps=cfg.optim.opt_eps)

    # create a minimal log and checkpoint system
    os.makedirs(run_dir + '/logs', exist_ok=True)
    os.mkdir(run_dir + '/checkpoints')
    log_path = run_dir + '/logs/metrics.jsonl'
    checkpoint_path = run_dir + '/checkpoints'

    # log gradients for wandb
    if wandb.run is not None:
        # gradient logging is lightweight at this scale; adjust log_freq if needed
        wandb.watch(model, log="gradients", log_freq=max(1, cfg.out.checkpoint_interval))

    # get val batches
    val_batch_ids = [[random.randint(cfg.model.context_length, len(valid_array) - 1) for _ in range(cfg.data.batch_size)]
                     for _ in range(cfg.out.val_batches)]

    decay = 0.9
    train_loss_avg = None
    current_lr = 0.0

    for step in range(cfg.optim.total_step_count + 1):
        print('train step', step)
        # zero out all the gradients
        optimizer.zero_grad()

        # grab the batch to train on
        if not cfg.out.debug:
            input_batch, output_batch = get_batch(x=train_array,
                                                  batch_size=cfg.data.batch_size,
                                                  context_length=cfg.model.context_length,
                                                  device=device)
        else:
            # we are training on a single batch
            input_batch, output_batch = get_batch(x=train_array,
                                                  batch_size=cfg.data.batch_size,
                                                  context_length=cfg.model.context_length,
                                                  device=device,
                                                  batch_indices=val_batch_ids[0])

        # compute the loss with a forward pass
        logits = model(input_batch)
        loss = cross_entropy(o=logits, x=output_batch)

        # update our ewma
        if train_loss_avg is None:
            train_loss_avg = loss.item()
        else:
            train_loss_avg += decay * (loss.item() - train_loss_avg)

        # every so often we log 
        if step % cfg.out.checkpoint_interval == 0:
            metrics = {}
            metrics['valid_loss'] = validation_loss(valid_array=valid_array,
                                                    val_batch_ids=val_batch_ids,
                                                    model=model,
                                                    cfg=cfg,
                                                    device=device)
            metrics['train_loss'] = train_loss_avg

            log_metrics(step=step, log_path=log_path, metrics=metrics)

            # this is the model AFTER we have trained for step steps
            save_checkpoint(model=model,
                            optimizer=optimizer,
                            iteration=step,
                            out=checkpoint_path + '/' + str(step))

            # log with wandb
            if wandb.run is not None:
                wandb.log({
                    "step": step,
                    "valid/loss": metrics['valid_loss'],
                    "train/loss_ewma": train_loss_avg,
                    "optimizer/lr": current_lr
                }, step=step)

        else:
            if wandb.run is not None:
                wandb.log({
                    "step": step,
                    "train/loss_ewma": train_loss_avg,
                    "optimizer/lr": current_lr
                }, step=step)

        # this is since we wanna log after we are done training
        if step == cfg.optim.total_step_count:
            break

        # compute the gradients
        loss.backward()

        # clip them
        clip_gradients(parameters=model.parameters(),
                       max_norm=cfg.optim.max_norm,
                       eps=cfg.optim.grad_clip_eps)

        # schedule the learning rate
        current_lr = lr_cosine_schedule(t=step,
                                        alpha_max=cfg.optim.alpha_max,
                                        alpha_min=cfg.optim.alpha_min,
                                        T_w=cfg.optim.warmup_steps,
                                        T_c=cfg.optim.total_step_count - cfg.optim.warmup_steps)

        for group in optimizer.param_groups:
            group['lr'] = current_lr

        # finally we do an optimizer update
        optimizer.step()


def _set_by_dotted_attr(obj, dotted_key: str, value):
    """
    Helper: set nested dataclass attributes given a dotted path like 'model.num_layers'.
    """
    parts = dotted_key.split(".")
    target = obj
    for p in parts[:-1]:
        target = getattr(target, p)
    setattr(target, parts[-1], value)


def train(cfg: TrainConfig) -> str:
    # set random seeds
    random.seed(cfg.out.seed)
    np.random.seed(cfg.out.seed)
    torch.manual_seed(cfg.out.seed)

    # create the directory
    run_dir = make_run_dir(cfg=cfg)

    # check if we are in a sweep
    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    name = None if sweep_id else f"{os.path.basename(cfg.out.home_dir)}_{os.path.basename(run_dir)}"

    # initialize wandb (only if enabled)
    if cfg.out.use_wandb:
        wandb.init(
            project="stanford_class_assignment_1",
            name=name,
            config=flatten_dict(asdict(cfg)),
            dir=run_dir,
        )

        # Merge sweep-provided overrides into cfg (supports dotted keys)
        wc = dict(wandb.config)
        for k, v in wc.items():
            if "." in k:
                try:
                    _set_by_dotted_attr(cfg, k, v)
                except Exception:
                    pass

        # set metrics
        if wandb.run is not None:
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")

    # populate the data
    if cfg.out.profile:
        start = time.perf_counter()
    create_data(cfg=cfg, run_dir=run_dir)
    if cfg.out.profile:
        end = time.perf_counter()
        size_mb = (os.path.getsize(cfg.data.train_text_path) + os.path.getsize(cfg.data.val_text_path)) / (1024 * 1024)
        mbps = size_mb / (end-start)

        # make the logs if we dont ahve it yet
        log_dir = os.path.join(run_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # log how long we took
        with open(run_dir+'/logs/timing.txt', 'a', encoding='utf-8') as f:
            f.write('megabytes of text processed by tokenizer per second: '+str(mbps) + '\n')

    # set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # run the training loop
    if cfg.out.profile:
        start = time.perf_counter()
    run_training_loop(cfg=cfg, run_dir=run_dir, device=device)
    if cfg.out.profile:
        end = time.perf_counter()
        timing = end-start

        # forward flop computations (basically ignore rearranges)
        b = cfg.data.batch_size
        l = cfg.model.context_length
        d_model = cfg.model.d_model
        d_ff = cfg.model.d_ff
        h = cfg.model.num_heads

        # first layer norm
        ln1_flops = d_model * b * l * 5

        # attn stuff
        attn_qkv_flops = 2*3*d_model * d_model * b * l
        rope_flops = 8 * 2 * d_model # just queries and keys get roped up
        attn_dot_prod_flops = b * l * l * (2 * d_model + h) # + num_heads for the masking
        attn_softmax_flops = b * h * 3 * l * l # 3 * l * l cost of softmaxing each row of mat
        attn_get_values = 2 * b * l * l * d_model
        attn_out_flops = b * l * 2 * d_model * d_model

        total_attn_flops = attn_qkv_flops + rope_flops + attn_dot_prod_flops + attn_softmax_flops + attn_get_values + attn_out_flops

        # adding into resid
        adding_atn_flops = b * l * d_model

        # layernorm
        ln2_flops = d_model * b * l * 5

        # swiglu
        swiglu_flops  = b * l * (6*d_ff*d_model + 3*d_ff)

        # adding it in
        adding_swiglu = b * l * d_model

        transformer_block_flops = ln1_flops + total_attn_flops + adding_atn_flops + ln2_flops + swiglu_flops + adding_swiglu

        # total transformer
        tot_transformer_layer_flops = cfg.model.num_layers * transformer_block_flops
        final_ln = d_model * b * l * 5
        output_logits = b * l * (2 * d_model * cfg.model.vocab_size)

        # flops for a single forward pass at our batch size
        flops_per_forward = tot_transformer_layer_flops + final_ln + output_logits

        total_forwards =  cfg.optim.total_step_count + (cfg.optim.total_step_count // cfg.out.checkpoint_interval + 1) * cfg.out.val_batches
        total_flops = total_forwards * flops_per_forward
        with open(run_dir+'/logs/timing.txt', 'a', encoding='utf-8') as f:
            f.write('Gflops per second: '+str(total_flops / timing / (10**9)) + '\n')

        
    # close wandb if needed
    if wandb.run is not None:
        wandb.finish()

    return run_dir
