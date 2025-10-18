from cs336_basics.tokenizer.tokenizer import Tokenizer


def test_encode_decode():
    tokenizer = Tokenizer.from_files(vocab_filepath='data/TinyStoriesV2-GPT4/tokenizer/vocab.pkl',
                                     merges_filepath='data/TinyStoriesV2-GPT4/tokenizer/merges.pkl')
    text = None
    with open("data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
        text = f.read(10000)
    encoding = tokenizer.encode(text)
    decoding = tokenizer.decode(encoding)
    assert(text == decoding)


def test_iterable_encode():
    tokenizer = Tokenizer.from_files(vocab_filepath='data/TinyStoriesV2-GPT4/tokenizer/vocab.pkl',
                                     merges_filepath='data/TinyStoriesV2-GPT4/tokenizer/merges.pkl',
                                     special_tokens=['hello', 'cat', 'the', 'th'])
    text_list = []
    with open("data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
        for i in range(500):
            text_list.append(f.read(17))

    encoding_iterator = tokenizer.encode_iterable(iterable=text_list)
    encoding = [ei for ei in encoding_iterator]
    decoding = tokenizer.decode(encoding)
    assert(decoding == ''.join(text_list))
    