import regex as re
import pickle
from collections import deque
from collections.abc import Iterator, Iterable
from cs336_basics.utils import create_special_token_string

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None):
        
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

        # track the max special token size
        self.max_special_token_length = 0

        for id in vocab:
            # update max id
            max_id = max(max_id, id)

            # update our dictionaries
            word = vocab[id]
            self.id_vocab[id] = word
            self.vocab_id[word] = id

            # handle special tokens and assumes they appear aftre all normal vocab words
            if id >= normal_vocab_length:
                special_word_str = word.decode('utf-8')
                special_tokens_set.add(special_word_str)
                self.max_special_token_length = max(self.max_special_token_length, len(special_word_str))

        # add the new special tokens
        if special_tokens is not None:
            for special_word_str in special_tokens:
                if special_word_str not in special_tokens_set:
                    
                    # update the max id 
                    max_id += 1

                    # collect the special word string
                    special_tokens_set.add(special_word_str)

                    # update the max length
                    self.max_special_token_length = max(self.max_special_token_length, len(special_word_str))

                    # update our encode/decode dictionaries 
                    special_word = special_word_str.encode('utf-8')
                    self.id_vocab[max_id] = special_word
                    self.vocab_id[special_word] = max_id


        # pattern used for pretokenized splitting (including parenthesis this time)
        self.special_token_pattern = create_special_token_string(special_tokens=special_tokens_set,
                                                                 include_specials=True)

        # cache pretoken encodings for later use
        self.pretoken_encodings = {}

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
    
    def encode_pretoken(self, pretoken:str)->list[int]:
        if pretoken not in self.pretoken_encodings:
            bytes_list = [bytes([b]) for b in pretoken.encode('utf-8')]
            for merge in self.merges:
                new_bytes_list = []
                new_word = merge[0] + merge[1]
                prev_pair_was_merge = False
                for current_word, next_word in zip(bytes_list, bytes_list[1:]):
                    if prev_pair_was_merge:
                        prev_pair_was_merge = False
                    else:
                        if current_word == merge[0] and next_word == merge[1]:
                            new_bytes_list.append(new_word)
                            prev_pair_was_merge = True
                        else:
                            new_bytes_list.append(current_word)
                if not prev_pair_was_merge:
                    new_bytes_list.append(bytes_list[-1])
                bytes_list = new_bytes_list
            self.pretoken_encodings[pretoken] = [self.vocab_id[bytes] for bytes in bytes_list]
        return self.pretoken_encodings[pretoken]
    
    def encode(self, text:str)->list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        encoding = []
        special_split = re.split(self.special_token_pattern, text)
        for i, chunk in enumerate(special_split):
            if i % 2 == 0:
                pretokens = re.findall(PAT, chunk)
                for pretoken in pretokens:
                    encoding += self.encode_pretoken(pretoken=pretoken)
            else:
                special_word = chunk.encode('utf-8')
                encoding.append(self.vocab_id[special_word])

        return encoding
    
    def decode(self, token_id_list:list[int])->str:
        """
        Decode a sequence of token IDs into text.
        """
        bytes_rep = b''.join([self.id_vocab[id] for id in token_id_list])
        return bytes_rep.decode('utf-8', errors='replace')
    
    def encode_iterable(self, iterable:Iterable[str])-> Iterator[int]:
        return EncodingIterator(iterable=iterable,
                                tokenizer=self)
    

class EncodingIterator:

    def __init__(self, 
                 iterable:Iterable[str],
                 tokenizer:Tokenizer):
        self.string_iter = iter(iterable)
        self.tokenizer = tokenizer

        # a bunch of encoded ints that we want to output
        self.backlog = deque([])
        self.prefix_string = ''

    def __iter__(self):
        return self
    
    def process_prefix(self, fully:bool):
            # divide up the self.prefix + next_string based on special tokens
            chunks = re.split(self.tokenizer.special_token_pattern, self.prefix_string)

            stop_index = len(chunks) - 1
            if fully:
                stop_index += 1

            # process every chunk except the last one which might get added to the prefix
            for i, chunk in enumerate(chunks[:stop_index]):
                if i % 2 == 0:
                    # these are non-special chunks and get encoded with the tokenizer
                    pretokens = re.findall(PAT, chunk)
                    for pretoken in pretokens:
                        # we always just build on our backlog
                        self.backlog.extend(self.tokenizer.encode_pretoken(pretoken=pretoken))
                else:
                    # these are special chunks, and get easily encoded with the tokenizer
                    special_word = chunk.encode('utf-8')
                    # update our backlog
                    self.backlog.append(self.tokenizer.vocab_id[special_word])

            # reset the prefix
            self.prefix_string = ''
            
            if not fully:
                # now we deal with the last chunk
                last_chunk = chunks[-1]

                # we make sure to process every bit of the last chunk that we possbily can
                remainder = len(last_chunk)
                for match in re.finditer(PAT, last_chunk):
                    pretoken = match.group(0)
                    remainder -= len(pretoken)
                    if remainder >= self.tokenizer.max_special_token_length:
                        self.backlog.extend(self.tokenizer.encode_pretoken(pretoken=pretoken))
                    else:
                        remainder += len(pretoken)
                        break
                
                # set the remainder to the prefix
                self.prefix_string = last_chunk[-remainder:]

    def __next__(self):

        # keep grabbing items ffrom string_iter if the backlog is empty
        while len(self.backlog) == 0:
            # otherwise process the next string

            try:
                self.prefix_string = self.prefix_string + next(self.string_iter)
            except StopIteration:
                if len(self.prefix_string) == 0:
                    raise StopIteration
                self.process_prefix(fully=True)
            else:
                self.process_prefix(fully=False)

        return self.backlog.popleft()
        




    

    


