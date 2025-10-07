import regex as re
import pickle

class Tokenizer:

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[bytes] | None = None):
        
        # maps ids to vocab words
        self.id_vocab = {}

        # maps vocab words to ids
        self.vocab_id = {}

        # ordered list of merges to make
        self.merges = [merge for merge in merges]

        # number of vocab words that are not special
        normal_vocab_length = 256 + len(merges)

        # track the max_id for the purpose of adding new words later
        max_id = 0

        # track special_tokens
        special_tokens_set = set([])

        for id in vocab:
            # update max id
            max_id = max(max_id, id)

            # update our dictionaries
            word = vocab[id]
            self.id_vocab[id] = word
            self.vocab_id[word] = id

            # handle special tokens and assumes they appear aftre all normal vocab words
            if id >= normal_vocab_length:
                special_tokens_set.add(word)

        # add the new special tokens
        if special_tokens is not None:
            for special_word in special_tokens:
                if special_word not in special_tokens_set:
                    special_tokens_set.add(special_word)
                    self.id_vocab[max_id] = special_word
                    self.vocab_id[special_word] = max_id
                    max_id += 1

        # pattern used for pretokenized splitting
        self.special_token_pattern = "|".join([re.escape(spec) for spec in special_tokens_set])

    @classmethod
    def from_files(cls, 
                   vocab_filepath:str,
                   merges_filepath:str,
                   special_tokens: list[str]|None = None):
        
     

        # open the pickle file in binary read mode
        vocab = None
        merges = None
        with open(vocab_filepath, "rb") as f:
            vocab, _ = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab,
                   merges=merges,
                   special_tokens=special_tokens)
    
    def encode(self, text:str)->list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        

if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(vocab_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_vocab.pkl',
                                     merges_filepath='data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-train_merges.pkl')
    
