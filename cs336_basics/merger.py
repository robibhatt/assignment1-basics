from collections import Counter
from cs336_basics.counter_heap import CounterHeap
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


class BpeMerger:
    """
    class that that figures out the best merges from a set of pretokens.
    also capable of executing those merges making it useful for encoding.
    """

    def __init__(self, pretoken_counts: Counter[str], encode_only: bool = False):

        # store the counts for computing optimal merges
        self.pretoken_counts = pretoken_counts

        # the byte vocabulary
        self.vocab = [bytes([b]) for b in range(256)]

        # initially empty set of merges
        self.merges = []

        """         
        create the linkned lists used to store (changing) byte representations for pretokens
        head_nodes (dict[str, BytesNode]) gives a designated unchanging `head` node to each pretoken
        word_nodes (dict[bytes, set[BytesNode]]) stores all nodes that store a given word(bytes)
        pretoken always means a string in self.pretoken_counts, word always means a bytes in self.vocab 
        """
        self.head_nodes, self.word_nodes = self._get_pretoken_word_nodes()
        
        # we initialize a heap used to determine the best merges if we are not just encoding
        self.best_pair_heap = None
        if not encode_only:
             pair_counter = self._start_pair_counter()
             self.best_pair_heap = CounterHeap(pair_counter)


    def _get_pretoken_word_nodes(self) -> tuple[dict[str, BytesNode], dict[bytes, set[BytesNode]]]:
        """
        we used doubly linked lists instead of lists to maintain the bytes for each pretoken
        we want to access these nodes both based on the pretoken and based on a specific bytes
        we use pretoken to denote strings in our input, and words to denote bytes inside our vocab.
        type(pretoken)==str, type(word)==bytes
        """

        # initializing a dict of head_nodes
        pretoken_nodes = {}

        # initiailize each set of nodes for each word
        word_nodes = {}
        for word in self.vocab:
            word_nodes[word] = set([])

        for pretoken in self.pretoken_counts:

            # to be stored in all nodes for convenience
            rep_factor = self.pretoken_counts[pretoken]

            # set up the head node which never changes and never contains a bytes_value
            prev_node = BytesNode(bytes_value=None, rep_factor=rep_factor)
            pretoken_nodes[pretoken] = prev_node

            # iterate through the bytes in the utf-8 encoding of pretoken
            for b in pretoken.encode('utf-8'):
                word = bytes([b])
                node = BytesNode(bytes_value=word, rep_factor=rep_factor)

                # ensure the node is connected to the one before it
                prev_node.nxt = node
                node.prev = prev_node

                # add the node into the appropriate set
                word_nodes[word].add(node)

                # iterate to next node
                prev_node = node

        return pretoken_nodes, word_nodes
    

    def _start_pair_counter(self)->Counter[tuple[bytes, bytes]]:
        """
        gets initial counts of each bytes,bytes occurrences.
        """
        pair_counter = Counter()
        for pretoken in self.head_nodes:
            node = self.head_nodes[pretoken]
            node = node.nxt
            if node is not None:
                while node.nxt is not None:
                    next_node = node.nxt
                    pair_counter[(node.bytes_value, next_node.bytes_value)] += node.rep_factor
                    node = next_node
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
        if merge_pair is None:
            return
        (first_bytes, second_bytes) = merge_pair

        # create the new vocab word
        new_token_bytes = first_bytes + second_bytes
        self.vocab.append(new_token_bytes)

        # record the merge
        self.merges.append((first_bytes, second_bytes))

        # prepare a set of nodes that need to be deleted
        del_set = set([])

        if first_bytes != second_bytes:
            # modify all of the linked lists
            for first_bytes_node in self.word_nodes[first_bytes]:
                succ = first_bytes_node.nxt
                if succ is not None and succ.bytes_value == second_bytes:
                    new_node = BytesNode(bytes_value=new_token_bytes, rep_factor=first_bytes_node.rep_factor)
                    if new_token_bytes not in self.word_nodes:
                        self.word_nodes[new_token_bytes] = set([])
                    self.word_nodes[new_token_bytes].add(new_node)
                    new_node.nxt = succ.nxt
                    if succ.nxt is not None:
                        succ.nxt.prev = new_node
                    new_node.prev = first_bytes_node.prev
                    new_node.prev.nxt = new_node

                    del_set.add(first_bytes_node)
                    del_set.add(succ)

            for node in del_set:
                self.word_nodes[node.bytes_value].remove(node)

        else:
            new_first_bytes_set = set([])
            while len(self.word_nodes[first_bytes]) > 0:
                node = next(iter(self.word_nodes[first_bytes]))
                while node.prev.bytes_value == node.bytes_value:
                    node = node.prev
                while (node is not None) and (node.nxt is not None) and (node.bytes_value == first_bytes) and (node.nxt.bytes_value == first_bytes):

                    # create the new node
                    new_node = BytesNode(bytes_value=new_token_bytes, rep_factor=node.nxt.rep_factor)
                    if new_token_bytes not in self.word_nodes:
                        self.word_nodes[new_token_bytes] = set([])
                    self.word_nodes[new_token_bytes].add(new_node)
                    assert(new_node is not None)

                    # add its prev link
                    new_node.prev = node.prev
                    node.prev.nxt = new_node

                    # add its nxt link
                    new_node.nxt = node.nxt.nxt
                    if node.nxt.nxt is not None:
                        node.nxt.nxt.prev = new_node

                    # remove the node and its successor from our set of nodes to check
                    self.word_nodes[first_bytes].remove(node)
                    self.word_nodes[first_bytes].remove(node.nxt)
                    del_set.add(node)
                    del_set.add(node.nxt)

                    # on to the next node worth considering
                    node = new_node.nxt
                if node is not None and node.bytes_value == first_bytes:
                    self.word_nodes[first_bytes].remove(node)
                    new_first_bytes_set.add(node)
            self.word_nodes[first_bytes] = new_first_bytes_set

        # delete the removed nodes
        del_set.clear()
        del del_set

        # update the heap if it exists
        if self.best_pair_heap is not None:

            # make a set to ensure we dont update counts for the same node twice
            touched_nodes = set([])

            # counter for storing counts of changed pairs
            pair_counter_delta = Counter()

            for bytes_value in [first_bytes, second_bytes, new_token_bytes]:
                for node in self.word_nodes[bytes_value]:
                    prev_node = node.prev
                    nxt_node = node.nxt
                    if prev_node.bytes_value is not None:
                        if prev_node not in touched_nodes:
                            pair_counter_delta[(prev_node.bytes_value, node.bytes_value)] += node.rep_factor
                            touched_nodes.add(prev_node)
                    if nxt_node is not None:
                        if node not in touched_nodes:
                            pair_counter_delta[(node.bytes_value,nxt_node.bytes_value)] += node.rep_factor
                            touched_nodes.add(node)

            for bytes_value in [first_bytes, second_bytes, new_token_bytes]:
                for current_token in self.vocab:
                    pair_counter_delta[(bytes_value, current_token)] += 0
                    pair_counter_delta[(current_token, bytes_value)] += 0

            self.best_pair_heap.modify_counts(new_counts=pair_counter_delta)

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
        bytes_list = []
        node = self.head_nodes[pretoken]
        while(node.nxt is not None):
            # ignore the first node since it is just a head
            node = node.nxt
            bytes_list.append(node.bytes_value)
        return bytes_list
    

def get_tokens_and_merges_real(
    pretokenized_counts: Counter[str],
    number_of_merges: int
)->tuple[list[bytes], list[tuple[bytes, bytes]]]:

    bpe_merger = BpeMerger(pretoken_counts=pretokenized_counts, encode_only=False)

    for i in range(number_of_merges):
        best_merge = bpe_merger.get_best_merge()
        bpe_merger.merge(best_merge)

    return (bpe_merger.get_vocab_list(), bpe_merger.get_merge_history())