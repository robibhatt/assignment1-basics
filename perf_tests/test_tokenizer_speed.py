from cs336_basics.tokenizer.tokenizer import Tokenizer
import cProfile, pstats


def read_in_chunks(file, chunk_size=1024):
    """Lazy iterator that reads `chunk_size` bytes at a time."""
    while True:
        data = file.read(chunk_size)
        if not data:
            break
        yield data


def test_iterable_encode_speed():
    # just times how fast we encode everything
    tokenizer = Tokenizer.from_files(vocab_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_vocab.pkl',
                                     merges_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_merges.pkl',
                                     special_tokens=['hello', 'cat', 'the', 'th'])

    counter = 0
    with cProfile.Profile() as pr:
        with open("data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid.txt", "r", encoding="utf-8") as f:
            for encoding in tokenizer.encode_iterable(iterable = read_in_chunks(file=f, chunk_size=1024)):
                counter += 1
                if counter % 1000 == 0:
                    print('processed', counter, 'chunks.')

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats("cumulative").print_stats(20)


