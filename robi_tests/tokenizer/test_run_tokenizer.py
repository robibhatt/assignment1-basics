from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.run_tokenizer import tokenize_file_parallel, tokenize_section
from cs336_basics.tokenizer.tokenizer import Tokenizer
from typing import BinaryIO
import numpy as np
import os


def file_end_offset(f: BinaryIO) -> int:
    """Return total byte length of an open binary file without changing its position."""
    cur = f.tell()
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(cur, os.SEEK_SET)
    return size

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
                            special_tokens=None,
                            chunk_size=64000,
                            num_workers=5,
                            output_dir=tokenizer_dir,
                            vocab_size=500)

    
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

    # make the tokenizer anew
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                    merges_filepath=merges_filepath,
                                    special_tokens = None,
                                    vocab_size=500)
    
    # first check direct encode and decode
    silly_text = entire_text[:10000]
    encoding = tokenizer.encode(text=silly_text)
    decoding = tokenizer.decode(token_id_list=encoding)
    assert(silly_text == decoding)


    # next we check the iterable way
    encoding_iter = tokenizer.encode_iterable(iterable=silly_text)
    encoding = [i for i in encoding_iter]
    decoding = tokenizer.decode(token_id_list=encoding)
    assert(silly_text == decoding)

    # next we check with a file
    token_file = tokenizer_dir+'/brute_force_tokens.uint16'
    with open(filename, "rb") as f:
        tokenize_section(f=f,
                         start=0,
                         end=4500912,
                         chunk_size=1000,
                         tokenizer=tokenizer,
                         token_filename=token_file)
        brute_text = ''.join([x for x in FileStringIterator(file=f,
                                            start=0,
                                            end=4500912,
                                            chunk_size=1000)])
        
    brute_arr = np.fromfile(token_file, dtype=np.uint16)
    brute_decode = tokenizer.decode(token_id_list=list(brute_arr))
    
    if brute_decode != brute_text:
        assert(False)

    # finally we check from the files
    # decode the validation text from the encoding
    decoded_text = ''
    arr = np.fromfile(tokenizer_dir+'/tokens.uint16', dtype=np.uint16)
    decoded_text = decoded_text + tokenizer.decode(token_id_list=list(arr))

    if entire_text != decoded_text:
        print('\n')
        print(entire_text[:1000])
        print('\n')
        print(decoded_text[:1000])
        assert(False)
