from cs336_basics.pretokenizer import pretokenize_file_parallel
from cs336_basics.merger import get_tokens_and_merges_real
from pathlib import Path
import pickle
import argparse
import cProfile, pstats

NUM_WORKERS = 7

def train_bpe(
    input_path:str,
    vocab_size:int,
    special_tokens:list[str],
    serialize:bool=False,
)->tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # pretokenization
    string_counter = pretokenize_file_parallel(
        filename=input_path,
        special_tokens=special_tokens,
        chunk_size=64000,
        num_workers=NUM_WORKERS
    )

    # always have to include the bytes and the special tokens as a default
    number_of_merges = vocab_size - len(special_tokens) - 256

    # time to get a bytes list of our tokens along with our merges
    final_tokens, merges = get_tokens_and_merges_real(
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

    # serialize if needed. Always stores in same directory as text file
    if serialize:

        # create paths to the desired pkl files
        file_path = Path(input_path)
        parent_dir = file_path.parent
        filename = file_path.stem
        vocab_path = parent_dir / (filename + '_vocab.pkl')
        merges_path = parent_dir / (filename + '_merges.pkl')

        # dump the info
        with open(vocab_path, "wb") as vf:
            pickle.dump((vocab, special_tokens), vf)

        with open(merges_path, "wb") as mf:
            pickle.dump(merges, mf)

    return (vocab, merges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to file or directory")
    args = parser.parse_args()

    assert(args.path.exists())
    filename = str(args.path)
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(
        input_path=filename,
        vocab_size=500,
        special_tokens=special_tokens,
        serialize=True
    )

    print('trained bpe_tokenizer on ', filename)


if __name__ == "__main__":
    main()
