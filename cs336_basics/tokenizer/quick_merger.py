from collections import Counter
from cs336_basics.tokenizer.counter_heap import CounterHeap
from dataclasses import dataclass


@dataclass(eq=False, slots=True)
class BytesNode:
    """
    A simple node of a doubly linked list of bytes.
    Also stores a rep_factor which is a count of how many times this node appears
    """
    bytes_value: bytes
    rep_factor: int = 0
    prev: "BytesNode | None" = None
    nxt: "BytesNode | None" = None


class QuickMerger:
    """
    class that that figures out the best merges from a set of pretokens.
    also capable of executing those merges making it useful for encoding.
    """

    def __init__(self, pretoken_counts: Counter[str]):

        # store the counts for computing optimal merges
        self.pretoken_counts = Counter()
        for pretoken in pretoken_counts:
            self.pretoken_counts[pretoken] = pretoken_counts[pretoken]

        # the byte vocabulary
        self.vocab = [bytes([b]) for b in range(256)]

        # initially empty set of merges
        self.merges = []
       
        # adding some new stuff that we gotta use

        # create the current encodings
        self.current_encodings = {}
        for pretoken in self.pretoken_counts:
            self.current_encodings[pretoken] = []

        # create sets that correspond to each vocab word indicating the relevant pretokens
        self.word_pretoken_sets = {}
        for word in self.vocab:
            self.word_pretoken_sets[word] = set([])
            
        # populate both
        for pretoken in self.pretoken_counts:
            for b in pretoken.encode('utf-8'):
                word = bytes([b])
                self.current_encodings[pretoken].append(word)
                self.word_pretoken_sets[word].add(pretoken)

        # we initialize a heap used to determine the best merges if we are not just encoding
        self.best_pair_heap = None
        pair_counter = self._start_pair_counter()
        self.best_pair_heap = CounterHeap(pair_counter)
         
    def _start_pair_counter(self)->Counter[tuple[bytes, bytes]]:
        """
        gets initial counts of each bytes,bytes occurrences.
        """
        pair_counter = Counter()
        for pretoken in self.current_encodings:
            for word, next_word in zip(self.current_encodings[pretoken], self.current_encodings[pretoken][1:]):
                pair_counter[(word, next_word)] += self.pretoken_counts[pretoken]
        return pair_counter
        
    def get_best_merge(self)->tuple[bytes, bytes] | None:
        """
        returns the two bytes that are most frequently consecutive
        that should be merged when running the bpe algorithm.
        does so based on the current state (which means merges have already
        been made). 
        """
        return self.best_pair_heap.get_merge()
    
    def merge(self, merge_pair:tuple[bytes, bytes]| None)->None:
        """
        executes the merge throwing an error if it does not 
        recognize the bytes that are given to merge
        """

        # sometimes no merge is recommended, so just do nothing
        if merge_pair is None:
            return
        
        # extract words
        (word_0, word_1) = merge_pair

        # create the new vocab word
        new_word = word_0 + word_1

        # the new word is added to the club (even if it never occurs)
        self.word_pretoken_sets[new_word] = set([])

        cool_pair_count_delta = Counter()

        # update my cool guys
        for pretoken in (self.word_pretoken_sets[word_0] & self.word_pretoken_sets[word_1]):
            new_encoding = []
            word_0_exists = False
            word_1_exists = False
            new_word_exists = False
            prev_step_was_merge = False
            for i in range(len(self.current_encodings[pretoken]) - 1):
                current_word = self.current_encodings[pretoken][i]
                next_word = self.current_encodings[pretoken][i+1]
                # delete the current count
                cool_pair_count_delta[(current_word, next_word)] -= self.pretoken_counts[pretoken]

                # figure out the merge
                if prev_step_was_merge:
                    prev_step_was_merge = False
                else:
                    if current_word == word_0:
                        if next_word == word_1:
                            # we found a merge so do it
                            new_encoding.append(new_word)
                            new_word_exists = True
                            prev_step_was_merge = True
                        else:
                            # no merge, so just append the word but note that we found it
                            new_encoding.append(current_word)
                            word_0_exists = True
                    else:
                        # no merge, just append the word
                        new_encoding.append(current_word)

                        # do check though if it was word_1 since we must note we found it
                        if current_word == word_1:
                            word_1_exists = True

            # handle the very last word
            if not prev_step_was_merge:
                last_word = self.current_encodings[pretoken][-1]
                new_encoding.append(last_word)
                if last_word == word_0:
                    word_0_exists = True
                if last_word == word_1:
                    word_1_exists = True

            # update counts with the new_encoding
            for i in range(len(new_encoding) - 1):
                current_word = new_encoding[i]
                next_word = new_encoding[i+1]
                # delete the current count
                cool_pair_count_delta[(current_word, next_word)] += self.pretoken_counts[pretoken]

            # make sure the boolean values are correct
            if word_0 == word_1:
                word_1_exists = word_0_exists

            # set the new encoding
            self.current_encodings[pretoken] = new_encoding

            # update the word_pretoken_sets accordingly
            if word_0_exists:
                self.word_pretoken_sets[word_0].add(pretoken)
            else:
                self.word_pretoken_sets[word_0].discard(pretoken)
            if word_1_exists:
                self.word_pretoken_sets[word_1].add(pretoken)
            else:
                self.word_pretoken_sets[word_1].discard(pretoken)
            if new_word_exists:
                self.word_pretoken_sets[new_word].add(pretoken)
            else:
                self.word_pretoken_sets[new_word].discard(pretoken)
        
        # update selfs attributes accordingly
        self.vocab.append(new_word)
        self.merges.append((word_0, word_1))

        # update the pair counts
        self.best_pair_heap.update_counts(counts_delta=cool_pair_count_delta)

        # sanity check
        if (self.best_pair_heap.pair_counter[merge_pair] != 0):
            print(merge_pair, self.best_pair_heap.pair_counter[merge_pair])
            assert(False)
            pass

    def get_vocab_list(self)->list[bytes]:
        """
        returns a list of vocabulary words in the order of creation
        users can construct into dict or otherwise on their own time
        """
        return self.vocab
    
    def get_merge_history(self)->list[tuple[bytes, bytes]]:
        """
        returns the history of merges made.
        """
        return self.merges

    def tokenize(self, pretoken:str)->list[bytes]:
        """
        gives the current representation of a pretoken as a list of tokens.
        all tokens will be from the current vocabulary.
        only works for pretokens explicitly stored in this object.
        """
        return self.current_encodings[pretoken]
    

def quick_get_tokens_and_merges_real(
    pretokenized_counts: Counter[str],
    number_of_merges: int
)->tuple[list[bytes], list[tuple[bytes, bytes]]]:

    bpe_merger = QuickMerger(pretoken_counts=pretokenized_counts)

    for i in range(number_of_merges):
        best_merge = bpe_merger.get_best_merge()
        bpe_merger.merge(best_merge)

    return (bpe_merger.get_vocab_list(), bpe_merger.get_merge_history())