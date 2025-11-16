from cs336_basics.tokenizer.train_bpe import train_bpe
from cs336_basics.tokenizer.run_tokenizer import tokenize_file_parallel
import os
import shutil
import tempfile
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.training.train_config import from_yaml
import numpy as np

def delete_if_exists(filename):
    if os.path.exists(filename):
        os.remove(filename) 
        # os.remove throws an error if the file doesn't exist, so we gotta check first


def main():
    # Load config info
    cfg = from_yaml("./scripts/config.yaml")
    max_vocab_size = cfg.model.vocab_size
    chunk_size = cfg.data.pretoken_chunk_size

    train_file = cfg.data.train_text_path
    valid_file = cfg.data.val_text_path
    output_dir = cfg.data.tokenizer_dir

    debug = cfg.out.debug
    num_workers = cfg.data.num_workers

    # Make sure the output directory exists
    nominal_vocab = output_dir + '/vocab.pkl'
    nominal_merges = output_dir + '/merges.pkl'
    
    # Remove the nominal vocab and merges files
    delete_if_exists(nominal_vocab)
    delete_if_exists(nominal_merges)
    
    # make the directory for the output
    os.makedirs(output_dir, exist_ok=True)

    # Run the actual encoder
    (vocab, merges) = train_bpe(input_path=train_file,
                            vocab_size=max_vocab_size,
                            special_tokens=[],
                            serialize=True,
                            num_workers=num_workers,
                            output_dir=output_dir,
                            chunk_size=chunk_size)
    merge_set = set([])
    for merge in merges:
        merge_set.add(merge[0] + merge[1])
    assert(len(merge_set) == len(merges))
    if debug:
        vocab_size=500
        vocab_filepath = nominal_vocab
        merges_filepath = nominal_merges
        with tempfile.TemporaryDirectory() as tmpdir:

            # tokenize the validation file
            tokenize_file_parallel(training_text_filename=valid_file,
                                   vocab_filepath=vocab_filepath,
                                   merges_filepath=merges_filepath,
                                   special_tokens=[],
                                   chunk_size=chunk_size,
                                   num_workers=num_workers,
                                   output_dir=tmpdir,
                                   vocab_size=vocab_size)
            
            # create the tokenizer
            tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                             merges_filepath=merges_filepath,
                                             special_tokens = None,
                                             vocab_size=vocab_size)
            
            # get the validation text as a string
            entire_text = ''
            with open(valid_file, "rb") as f:
                num_processes = 7
                boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    string_iterator = FileStringIterator(file = f,
                                                        start = start,
                                                        end = end,
                                                        chunk_size = 10000)
                    original_text = ''.join([string for string in string_iterator])
                    entire_text = entire_text + original_text

            # decode the text
            arr = np.fromfile(tmpdir+'/tokens.uint16', dtype=np.uint16)
            decoded_text = tokenizer.decode(token_id_list=list(arr))

            if decoded_text != entire_text:
                print(decoded_text[:1000])
                print(entire_text[:1000])
                assert(False)
                        

if __name__ == '__main__':
    main()