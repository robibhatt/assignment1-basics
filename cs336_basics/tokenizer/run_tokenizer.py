from typing import BinaryIO
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from multiprocessing import Pool
import numpy as np
import shutil
import os


def tokenize_section(
    f: BinaryIO,
    start:int,
    end:int,
    chunk_size:int,
    tokenizer:Tokenizer,
    token_filename:str
) -> None:
    """
    maps the strings in [start, end) to ints in token_filename
    """

    new_string_iterator = FileStringIterator(file=f,
                                            start=start,
                                            end=end,
                                            chunk_size=chunk_size)
    token_iterable = tokenizer.encode_iterable(iterable=new_string_iterator)
    with open(token_filename, "ab") as g:  # append binary mode
        buf = []
        for val in token_iterable:
            buf.append(val)
            if len(buf) >= chunk_size:
                np.array(buf, dtype=np.uint16).tofile(g)
                buf.clear()

        # flush any leftovers
        if buf:
            np.array(buf, dtype=np.uint16).tofile(g)

    
def _process_chunk(args: tuple[str, int, int, int, Tokenizer, str]) -> None:
    """Worker function: pretokenize one byte range of the file."""
    (
        source_filename,
        start,
        end,
        chunk_size,
        tokenizer,
        token_filename,
    ) = args

    with open(source_filename, "rb") as f:
        tokenize_section(f=f,
                         start=start,
                         end=end,
                         chunk_size=chunk_size,
                         tokenizer=tokenizer,
                         token_filename=token_filename)


def tokenize_file_parallel(
    training_text_filename: str,
    vocab_filepath: str,
    merges_filepath: str,
    special_tokens: list[str],
    chunk_size: int,
    num_workers: int,
    output_dir: str,
    vocab_size: int|None = None,
    remake: bool|None = None,
) -> None:

    # First check if the tokens already exist --if so, then we don't need to remake them
    
    # First, make the storage location
    true_filename = training_text_filename.split("/")[-1] 
    filename_wo_ext = true_filename.split(".")[0]
    token_output_prefix = output_dir + '/' + filename_wo_ext
    token_output_location = token_output_prefix + '_tokens.uint16'

    # Now, check whether those tokens already exist; if so, then don't continue unless forced
    if os.path.exists(token_output_location) and not remake:
            return

    with open(training_text_filename, "rb") as f:
        num_processes = num_workers
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    token_filenames = [token_output_prefix + "_" + str(i)+'.uint16' for i in range(len(boundaries[:-1]))]
    tokenizers = [Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                 merges_filepath=merges_filepath,
                                 special_tokens=special_tokens,
                                 vocab_size=vocab_size) for i in range(len(boundaries[:-1]))]

    # Build tasks as tuples of arguments
    tasks = [
        (
            training_text_filename,
            start,
            end,
            chunk_size,
            tokenizers[i],
            token_filenames[i]
        )
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]

    # Run tasks in parallel with map
    with Pool(processes=num_processes) as pool:
        pool.map(_process_chunk, tasks)

    # delete shards
    with open(token_output_location, "wb") as w:
        for shard in token_filenames:
            with open(shard, "rb") as r:
                shutil.copyfileobj(r, w)

    assert os.path.exists(token_output_location)

    # Delete the shards
    for shard in token_filenames:
        os.remove(shard)
