from cs336_basics.pretokenizer import PAT, pretokenize_section, pretokenize_chunk, find_chunk_boundaries, pretokenize_file_parallel
from cs336_basics.utils import print_counter, compare_counters
import regex as re
from collections import Counter
from typing import BinaryIO, List



def test_pretokenize_section():
    special_tokens = ['<|endoftext|>']
    max_length = max([len(special_token.encode('utf-8')) for special_token in special_tokens])
    special_token_pattern = '|'.join([re.escape(spec) for spec in special_tokens])
    with open('data/TinyStoriesV2-GPT4-valid.txt', 'rb') as f:
        start = 1354
        end = 34536
        chunk_size = 400
        first_count = pretokenize_section_slow(f, start, end, special_token_pattern)
        second_count = pretokenize_section(f, start, end, special_token_pattern, max_length, chunk_size)

    assert(first_count == second_count)


def pretokenize_section_slow(
    f: BinaryIO,
    start:int,
    end:int,
    special_token_pattern:str
) -> Counter[str]:
    """
    SormehSamin
    """
    f.seek(start)
    text = f.read(end - start).decode("utf-8", errors="ignore")
    text_chunks = re.split(special_token_pattern, text)
    count = Counter()
    for chunk in text_chunks:
        for match in re.finditer(PAT, chunk):
            count[match.group(0)] += 1
    return count


def test_pretokenize_file():
    filename = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    first_count = pretokenize_file(filename, special_tokens, 1024)
    second_count = pretokenize_file(filename, special_tokens, 24123)
    #compare_counters(first_count, second_count)
    assert(first_count == second_count)


def pretokenize_file(
    filename: str,
    special_tokens: List[str],
    chunk_size: int,
) -> Counter[str]:
    
    count = Counter()
    max_special_length = max([len(special_token.encode('utf-8')) for special_token in special_tokens])
    special_token_pattern = '|'.join([re.escape(spec) for spec in special_tokens])
    with open(filename, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            # f.seek(start)
            # chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            new_count = pretokenize_section(f, start, end, special_token_pattern, max_special_length, chunk_size)
            count.update(new_count)
    return count


def test_pretokenize_parllel():
    filename = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    first_count = pretokenize_file(filename, special_tokens, 1024)
    second_count = pretokenize_file_parallel(filename, special_tokens, 1024, 4)
    #compare_counters(first_count, second_count)
    assert(first_count == second_count)