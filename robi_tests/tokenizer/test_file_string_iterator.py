from cs336_basics.tokenizer.pretokenizer import find_chunk_boundaries
from cs336_basics.tokenizer.file_string_iterator import FileStringIterator
from cs336_basics.tokenizer.tokenizer import Tokenizer
from cs336_basics.tokenizer.train_bpe import train_bpe


def test_file_string_iterator_encode_decode(tmp_path):
    """     filename = 'data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid.txt'
        boundaries = None
        tokenizer = Tokenizer.from_files(vocab_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_vocab.pkl',
                                        merges_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_merges.pkl') """
    
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

    # instantiate the tokenizer
    vocab_filepath = tokenizer_dir+'/vocab.pkl'
    merges_filepath = tokenizer_dir+'/merges.pkl'

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,
                                     merges_filepath=merges_filepath)
    total_text = ''
    with open(filename, "rb") as f:
        num_processes = 7
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            string_iterator = FileStringIterator(file = f,
                                                 start = start,
                                                 end = end,
                                                 chunk_size = 10000)
            encoding = [encoded_token for encoded_token in tokenizer.encode_iterable(iterable=string_iterator)]
            decoding = tokenizer.decode(token_id_list=encoding)
            new_string_iterator = FileStringIterator(file= f,
                                                     start=start,
                                                     end=end,
                                                     chunk_size=10000)
            original_text = ''.join([string for string in new_string_iterator])
            total_text += original_text
            assert(original_text == decoding)