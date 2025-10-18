import regex as re
import pickle
from collections import deque
import heapq
from collections.abc import Iterator, Iterable
from cs336_basics.utils import create_special_token_string

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:

    def __init__(self,
                 vocab: dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] | None = None,
                 vocab_size: int | None = None):
        
        # figure out the words that are merges
        merge_words_set = set([])
        for merge in merges:
            merge_words_set.add(merge[0] + merge[1])

        assert(len(merge_words_set) == len(merges))

        # figure out the base and special words
        base_words_set = set([])
        special_words_set = set([])
        for id in vocab:
            if id < 256:
                # by convention we always start with the base bytes
                base_words_set.add(vocab[id])
            else:
                if vocab[id] not in merge_words_set:
                    special_words_set.add(vocab[id])

        # add in the new special words
        if special_tokens is not None:
            for special_word_str in special_tokens:
                special_words_set.add(special_word_str.encode('utf-8'))

        
        # maps ids to vocab words
        self.id_vocab = {}

        # maps vocab words to ids
        self.vocab_id = {}

        # we first add the base words
        for id in range(256):
            self.id_vocab[id] = vocab[id]
            self.vocab_id[vocab[id]] = id


        next_id = 256
        # quick sanity checks
        assert(next_id == len(self.id_vocab))
        assert(next_id == len(self.vocab_id))

        
        # store the merges we make
        self.merges = []

        # we add precisely enough merges so that we hit at most the desired vocab_size
        for merge in merges:
            if vocab_size is not None and next_id >= vocab_size - len(special_words_set):
                # we break out early if we have already gotten enough words
                assert(next_id == vocab_size - len(special_words_set))
                break
            else:
                # carefully check to ensure that this merge involves no special words
                if merge[0] not in special_words_set:
                    if merge[1] not in special_words_set:
                        merge_word = merge[0] + merge[1]
                        if merge_word not in special_words_set:
                            self.merges.append(merge)
                            self.id_vocab[next_id] = merge_word
                            if merge_word in self.vocab_id:
                                assert(False)
                            self.vocab_id[merge_word] = next_id
                            next_id += 1
        
        # store the ids to easily look up when a merge occurred
        self.merge_ids = {}
        for id, merge in enumerate(self.merges):
            self.merge_ids[merge] = id

        # add in the special tokens
        assert(next_id == len(self.id_vocab))
        for special_word in special_words_set:
            self.id_vocab[next_id] = special_word
            self.vocab_id[special_word] = next_id
            next_id += 1

        # quick sanity checks
        assert(next_id == len(self.id_vocab))
        assert(next_id == len(self.vocab_id))

        # figure out the maximum length of a special token in characters
        special_strings = set([token.decode('utf-8') for token in special_words_set])
        self.max_special_token_length = max([len(string) for string in special_strings])

        # pattern used for pretokenized splitting (including parenthesis this time)
        self.special_token_pattern = create_special_token_string(special_tokens=special_strings,
                                                                 include_specials=True)
        
        # cache pretoken encodings for later use
        self.pretoken_encodings = {}
        
        # some sanity checking
        for voc in self.vocab_id:
            assert(self.vocab_id[voc] < len(self.vocab_id))

        for vid in self.id_vocab:
            assert(vid < len(self.id_vocab))

        assert(len(self.vocab_id) == len(self.id_vocab))

    @classmethod
    def from_files(cls, 
                   vocab_filepath:str,
                   merges_filepath:str,
                   special_tokens: list[str]|None = None,
                   vocab_size:int|None = None):
        
        # open the pickle file in binary read mode
        vocab = None
        merges = None
        with open(vocab_filepath, "rb") as f:
            vocab, _ = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            
        return cls(vocab=vocab,
                   merges=merges,
                   special_tokens=special_tokens,
                   vocab_size=vocab_size)
    
    def encode_pretoken(self, pretoken:str)->list[int]:
        if pretoken not in self.pretoken_encodings:
            self.pretoken_encodings[pretoken] = self.get_encoding(pretoken=pretoken)
        return self.pretoken_encodings[pretoken]
        

    def get_encoding(self, pretoken:str)->list[int]:

        # convert our pretoken into an initial list of bytes
        bytes_list = [bytes([b]) for b in pretoken.encode('utf-8')]
        counter = 0
        while True:
            counter += 1
            best_merge_id = len(self.merges)

            # search through all possible to merges and see if any appear
            for potential_merge in zip(bytes_list, bytes_list[1:]):
                if potential_merge in self.merge_ids:
                    new_merge_id = self.merge_ids[potential_merge]
                    if new_merge_id < best_merge_id:
                        best_merge_id = new_merge_id

            # get the best merge
            try:
                merge = self.merges[best_merge_id]
            except IndexError:
                # if there are no merges, we are done
                break
            else:
                # make the actual merge
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

        return [self.vocab_id[bytes] for bytes in bytes_list]
    
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
        count = 0
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
        




    

    


