from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.run_tokenizer import tokenize_file_parallel
from cs336_basics.tokenizer.tokenizer import Tokenizer
import numpy as np


def test_tokenizer_round_trip(tmp_path):

    # use the validation set for simplicity
    filename = 'data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid.txt'

    # train a quick tokenzier on the valid set
    tokenizer_dir = str(tmp_path)
    train_bpe(input_path=filename,
              vocab_size=1000,
              special_tokens=[],
              serialize=True,
              num_workers=5,
              output_dir=tokenizer_dir,
              chunk_size=64000)

    # run the tokenizer to tokenize the data
    vocab_filepath = tokenizer_dir+'/vocab.pkl'
    merges_filepath = tokenizer_dir+'/merges.pkl'

    tokenize_file_parallel(training_text_filename=filename,
                           vocab_filepath=vocab_filepath,
                           merges_filepath=merges_filepath,
                           special_tokens=[],
                           chunk_size=64000,
                           num_workers=5,
                           output_dir=tokenizer_dir)

    
    # get the validation text as a string
    entire_text = ''
    with open(filename, "rb") as f:
        num_processes = 7
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            string_iterator = FileStringIterator(file = f,
                                                 start = start,
                                                 end = end,
                                                 chunk_size = 10000)
            original_text = ''.join([string for string in string_iterator])
            entire_text = entire_text + original_text

    # decode the validation text from the encoding
    decoded_text = ''
    arr = np.fromfile(tokenizer_dir+'/tokens.uint16', dtype=np.uint16)
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                     merges_filepath=merges_filepath,
                                     special_tokens = None)
    decoded_text = decoded_text + tokenizer.decode(token_id_list=list(arr))

    if entire_text != decoded_text:
        assert(False)
