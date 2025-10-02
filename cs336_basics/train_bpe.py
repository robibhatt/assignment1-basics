from cs336_basics.pretokenizer import pretokenize_file_parallel
from cs336_basics.merger import get_tokens_and_merges


def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str]
)->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # pretokenization
    string_counter = pretokenize_file_parallel(
        filename=input_path,
        special_tokens=special_tokens,
        chunk_size=64000,
        num_workers=5
    )

    # always have to include the bytes and the special tokens as a default
    number_of_merges = vocab_size - len(special_tokens) - 256

    # time to get a bytes list of our tokens along with our merges
    final_tokens, merges = get_tokens_and_merges(
        pretokenized_counts=string_counter,
        number_of_merges=number_of_merges
    )

    start_id = 0
    vocab = {}
    for token in final_tokens:
        vocab[start_id] = token
        start_id += 1
    for special_string in special_tokens:
        vocab[start_id] = special_string.encode('utf-8')
        start_id += 1

    return (vocab, merges)


if __name__ == "__main__":
    filename = "data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=filename,
        vocab_size=500,
        special_tokens=special_tokens
    )
    print('it ran i guess')
