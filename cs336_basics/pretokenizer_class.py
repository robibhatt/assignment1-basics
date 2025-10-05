from collections import Counter
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
import os


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EMPTY_BYTES = ''.encode('utf-8')


class Pretokenizer:

    def __init__(self, special_tokens:list[str]):
        self.special_tokens = special_tokens
        self.pretoken_counter = Counter()
        self.prefix = EMPTY_BYTES
        self.max_special_length = max(len(s) for s in special_tokens)
        self.special_token_pattern = "|".join([re.escape(spec) for spec in special_tokens])

    def process_bytes(self, input_data:bytes):
        """
        Full update of the counter when given input data.
        Should be used sequentially on chunks of bytes that are consecutive.
        Will assume we pick up where we left off. 
        """
        input_string = self.add_bytes(input_data=input_data)
        self.pretokenize_str(input_string=input_string)

    def add_bytes(self, input_data:bytes)->str:
        """
        returns the decoded string from the bytes.
        sets the undecodable part to our prefix
        """

        # decode the string after including our prefix
        input_string = (self.prefix + input_data).decode('utf-8', errors='surrogateescape')

        # reset the prefix
        self.prefix = EMPTY_BYTES

        # get the last bytes that didn't decode properly
        last_index = len(input_string)-1
        last_char_bytes_int = ord(input_string[last_index])
        while last_index >= 0 and (0xDC80 <= last_char_bytes_int <= 0xDCFF):
            self.prefix = bytes([last_char_bytes_int - 0xDC00]) + self.prefix
            last_index -= 1
            last_char_bytes_int = ord(input_string[last_index])
        return input_string[:last_index+1]
    
    def pretokenize_str(self, input_string:str):
        """
        update our pretoken counter based on the string
        """
        pretoken_chunks = re.split(self.special_token_pattern, input_string)
        for chunk in pretoken_chunks[:-1]:
            for pretoken in re.findall(PAT, chunk):
                self.pretoken_counter[pretoken] += 1

        last_chunk = pretoken_chunks[-1]
        remainder = len(last_chunk)
        for match in re.finditer(PAT, last_chunk):
            pretoken = match.group(0)
            remainder -= len(pretoken)
            if remainder >= self.max_special_length:
                self.pretoken_counter[pretoken] += 1
            else:
                remainder += len(pretoken)
                break
        self.prefix = last_chunk[-remainder:].encode('utf-8') + self.prefix

    def process_prefix(self):
        prefix_string = self.prefix.decode('utf-8',errors="ignore")
        for pretoken in re.findall(PAT, prefix_string):
            self.pretoken_counter[pretoken] += 1
        self.prefix = EMPTY_BYTES
        

def pretokenize_section(
    f: BinaryIO,
    start:int,
    end:int,
    special_tokens: list[str],
    chunk_size:int,
) -> Counter[str]:
    """
    SormehSamin
    """

    pretokenizer = Pretokenizer(special_tokens=special_tokens)
    while start < end:
        next_start = min(end, start+chunk_size)
        f.seek(start)
        data = f.read(next_start-start)
        pretokenizer.process_bytes(data)
        start = next_start
    pretokenizer.process_prefix()
    return pretokenizer.pretoken_counter


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


def _process_chunk(args: tuple[str, int, int, list[str], int]) -> Counter[str]:
    """Worker function: pretokenize one byte range of the file."""
    (
        filename,
        start,
        end,
        special_tokens,
        chunk_size,
    ) = args

    with open(filename, "rb") as f:
        return pretokenize_section(
            f, start, end, special_tokens, chunk_size
        )
        

def pretokenize_file_parallel(
    filename: str,
    special_tokens: list[str],
    chunk_size: int,
    num_workers: 4
) -> Counter[str]:

    # counts pretokens over a large file
    output_counter = Counter()

    with open(filename, "rb") as f:
        num_processes = num_workers
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Build tasks as tuples of arguments
    tasks = [
        (
            filename,
            start,
            end,
            special_tokens,
            chunk_size,
        )
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # Run tasks in parallel with map
    with Pool(processes=num_processes) as pool:
        results = pool.map(_process_chunk, tasks)

    # Merge all Counters
    for c in results:
        output_counter.update(c)
    return output_counter