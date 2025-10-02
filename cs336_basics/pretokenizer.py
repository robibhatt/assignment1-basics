import regex as re
import os
from collections import Counter
from typing import BinaryIO, List, Tuple
from cs336_basics.utils import print_counter, compare_counters
import cProfile, pstats
from multiprocessing import Pool


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_chunk(
    f: BinaryIO,
    start:int,
    end:int,
    special_token_pattern:str,
    max_special_length:int,
    count: Counter[str]
) -> tuple[str, int]:
    """
    updates a counter with pretokenized string counts
    is careful to handle special_strings which straddle the boundary line
    """
    f.seek(start)
    text = f.read(end - start).decode("utf-8", errors="ignore")
    real_end = start + len(text.encode('utf-8'))
    text_chunks = re.split(special_token_pattern, text)
    for chunk in text_chunks[:-1]:
        for match in re.finditer(PAT, chunk):
            count[match.group(0)] += 1
    last_end = 0
    text_chunk = text_chunks[-1]
    chunk_length = len(text_chunk)
    for match in re.finditer(PAT, text_chunk):
        if chunk_length - match.end() >= max_special_length:
            count[match.group(0)] += 1
            last_end = match.end()
    new_start = real_end - len(text_chunk[last_end:chunk_length].encode('utf-8'))
    if start == new_start:
        assert(len(text_chunks) == 1)
        assert(last_end == 0)
        for match in re.finditer(PAT, text_chunk):
            count[match.group(0)] += 1
            last_end = match.end()
    return real_end - len(text_chunk[last_end:chunk_length].encode('utf-8'))


def pretokenize_section(
    f: BinaryIO,
    start:int,
    end:int,
    special_token_pattern:str,
    max_special_length: int,
    chunk_size:int,
) -> Counter[str]:
    """
    SormehSamin
    """
    count = Counter()
    current_start = start
    current_end = min(end, start + chunk_size)
    while True:
        next_start = pretokenize_chunk(f, current_start, current_end, special_token_pattern, max_special_length, count)
        if current_start == next_start:
            break
        else:
            current_start = next_start
            current_end = min(end, current_start + chunk_size)
    return count


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def _process_chunk(args: Tuple[str, int, int, str, int, int]) -> Counter[str]:
    """Worker function: pretokenize one byte range of the file."""
    (
        filename,
        start,
        end,
        special_token_pattern,
        max_special_length,
        chunk_size,
    ) = args

    with open(filename, "rb") as f:
        return pretokenize_section(
            f, start, end, special_token_pattern, max_special_length, chunk_size
        )
        

def pretokenize_file_parallel(
    filename: str,
    special_tokens: List[str],
    chunk_size: int,
    num_workers: 4
) -> Counter[str]:
    count = Counter()
    max_special_length = max(len(s.encode("utf-8")) for s in special_tokens)
    special_token_pattern = "|".join([re.escape(spec) for spec in special_tokens])

    with open(filename, "rb") as f:
        num_processes = num_workers
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Build tasks as tuples of arguments
    tasks = [
        (
            filename,
            start,
            end,
            special_token_pattern,
            max_special_length,
            chunk_size,
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # Run tasks in parallel with map
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_chunk, tasks)

    # Merge all Counters
    for c in results:
        count.update(c)
    return count


if __name__ == "__main__":
    filename = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    chunk_size = 64000
    result = pretokenize_file_parallel(filename, special_tokens, True, chunk_size, 4)
    print(len(result))