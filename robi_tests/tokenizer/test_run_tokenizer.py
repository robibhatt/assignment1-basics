from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.tokenizer import Tokenizer
import numpy as np


def test_file_string_iterator_encode_decode():
    filename = 'data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid.txt'
    boundaries = None
    tokenizer = Tokenizer.from_files(vocab_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_vocab.pkl',
                                     merges_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_merges.pkl')
    entire_text = ''
    with open(filename, "rb") as f:
        num_processes = 7
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            string_iterator = FileStringIterator(file = f,
                                                 start = start,
                                                 end = end,
                                                 chunk_size = 10000)
            new_string_iterator = FileStringIterator(file= f,
                                                     start=start,
                                                     end=end,
                                                     chunk_size=10000)
            original_text = ''.join([string for string in new_string_iterator])
            entire_text = entire_text + original_text

    decoded_text = ''
    arr = np.fromfile("data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid-data.uint16", dtype=np.uint16)
    decoded_text = decoded_text + tokenizer.decode(token_id_list=list(arr))

    assert(entire_text == decoded_text)