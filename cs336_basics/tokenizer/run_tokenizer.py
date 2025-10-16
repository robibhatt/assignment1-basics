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
    num_workers: int
) -> None:


    with open(training_text_filename, "rb") as f:
        num_processes = num_workers
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Build tasks as tuples of arguments
    tasks = [
        (
            training_text_filename,
            start,
            end,
            chunk_size,
            Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                 merges_filepath=merges_filepath,
                                 special_tokens=special_tokens),
            training_text_filename[:-4] + '_tokens_' + str(i) + '.uint16'
        )
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]

    # Run tasks in parallel with map
    with Pool(processes=num_processes) as pool:
        pool.map(_process_chunk, tasks)


def main():
    filename = 'data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train.txt'
    vocab_filepath = 'data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_vocab.pkl'
    merge_filepath = 'data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_merges.pkl'
    special_tokens = None
    chunk_size = 10000
    num_workers = 5
    tokenize_file_parallel(
        training_text_filename=filename,
        vocab_filepath=vocab_filepath,
        merges_filepath=merge_filepath,
        special_tokens=special_tokens,
        chunk_size=chunk_size,
        num_workers=num_workers
    )

    print('ran bpe_tokenizer on ', filename)
    shards = [filename[:-4] + '_tokens_' + str(i) + '.uint16' for i in range(num_workers)]
    with open(filename[:-4]+'-data.uint16', "wb") as w:
        for shard in shards:
            with open(shard, "rb") as r:
                shutil.copyfileobj(r, w)

    assert os.path.exists(filename[:-4]+'-data.uint16')

    # Delete the shards
    for shard in shards:
        os.remove(shard)


if __name__ == "__main__":
    main()
